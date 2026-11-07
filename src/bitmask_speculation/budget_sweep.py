"""Repeated fixed-K/C4 comparisons over caller-owned model adapters."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import json
from time import perf_counter

from .benchmarking import TrialRecord, Workload, aggregate_trials, record_trial
from .decoding import DensityPolicy, SpeculativeResult, greedy_decode, speculative_decode
from .grammar import ForkableGrammarState


def run_budget_sweep(
    target, draft, workload: Workload, *,
    prompt_token_ids: Sequence[int], eos_token_ids: Sequence[int],
    grammar_factory: Callable[[], ForkableGrammarState],
    decode: Callable[[tuple[int, ...]], str], metadata: Mapping[str, str],
    synchronize: Callable[[], None], fixed_budgets: Sequence[int] = (1, 2, 4, 8),
    policy: DensityPolicy = DensityPolicy(), repetitions: int = 3, warmups: int = 1,
    clock: Callable[[], float] = perf_counter,
    on_record: Callable[[TrialRecord], None] | None = None,
    on_reference: Callable[[dict], None] | None = None,
) -> tuple[TrialRecord, ...]:
    """Warm every configuration, rotate order, and retain every measured trial.

    Models, prompts and grammar compilation are supplied by the caller. Each
    constrained request gets fresh matcher state outside the timed interval.
    synchronize must wait for all devices used by both models (a no-op on CPU).
    Reference or warmup failures abort before measurements; measured failures
    become error records. No model loading, network access or cloud operations.
    """
    for value, minimum, name in ((repetitions, 2, "repetitions"), (warmups, 1, "warmups")):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    budgets = tuple(fixed_budgets)
    if not budgets or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in budgets):
        raise ValueError("fixed budgets must be nonempty positive integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("fixed budgets must be unique")
    if not isinstance(policy, DensityPolicy):
        raise TypeError("policy must be a DensityPolicy")
    if not callable(synchronize) or not callable(grammar_factory) or not callable(clock):
        raise TypeError("synchronize, grammar_factory and clock must be callable")
    if on_record is not None and not callable(on_record):
        raise TypeError("on_record must be callable")
    if on_reference is not None and not callable(on_reference):
        raise TypeError("on_reference must be callable")
    metadata = dict(metadata)
    if metadata.get("timing_scope") not in {"full_prefix_loop", "round_cached_draft_loop"}:
        raise ValueError("budget sweeps require a supported decode-loop timing_scope")
    metadata.update(sweep_repetitions=str(repetitions), sweep_warmups=str(warmups),
                    sweep_order="rotate_left_by_repetition")
    prompt, eos = tuple(prompt_token_ids), tuple(eos_token_ids)
    configurations = [("target_unconstrained", None), ("target_constrained", None)]
    configurations += [(variant, k) for variant in ("c1", "c2", "c3") for k in budgets]
    configurations += [("c4", None)]
    # Validate metadata before any model calls, using an explicitly failed record.
    TrialRecord(workload, "target_unconstrained", None, prompt, eos,
                json.dumps(metadata), error="preflight")

    def execute(variant, budget, state):
        if variant.startswith("target_"):
            return greedy_decode(target, prompt, max_new_tokens=workload.max_new_tokens,
                                 eos_token_ids=eos, grammar=state)
        return speculative_decode(
            target, draft, prompt, max_new_tokens=workload.max_new_tokens,
            eos_token_ids=eos, grammar=state, draft_budget=budget or policy.max_budget,
            constrain_draft=variant in {"c3", "c4"},
            draft_policy=policy if variant == "c4" else None,
        )

    def fresh(variant):
        if variant in {"target_unconstrained", "c1"}:
            return None
        state = grammar_factory()
        if state is None:
            raise ValueError("grammar_factory must return fresh grammar state")
        return state

    references = {
        constrained: execute("target_constrained" if constrained else "target_unconstrained",
                             None, fresh("target_constrained") if constrained else None)
        for constrained in (False, True)
    }
    if on_reference is not None:
        on_reference(references)
    for _ in range(warmups):
        for variant, budget in configurations:
            result = execute(variant, budget, fresh(variant))
            synchronize()
            generation = result.generation if isinstance(result, SpeculativeResult) else result
            if generation != references[variant not in {"target_unconstrained", "c1"}]:
                raise ValueError(f"warmup output differs from reference: {variant}, K={budget}")

    records = []
    for repetition in range(repetitions):
        offset = repetition % len(configurations)
        order = configurations[offset:] + configurations[:offset]
        for variant, budget in order:
            options = dict(variant=variant, draft_budget=budget, repetition=repetition,
                           policy=policy if variant == "c4" else None)
            try:
                state = fresh(variant)
                synchronize()
                start = clock()
                result = execute(variant, budget, state)
                synchronize()
                elapsed = clock() - start
                record = record_trial(
                    workload, result=result,
                    reference=references[variant not in {"target_unconstrained", "c1"}],
                    decode=decode, prompt_token_ids=prompt, eos_token_ids=eos,
                    metadata=metadata, decode_seconds=elapsed, **options,
                )
            except Exception as exc:
                # Drain pending work before another trial. A broken synchronizer
                # aborts the sweep rather than allowing invalid later timings.
                synchronize()
                record = TrialRecord(
                    workload, prompt_token_ids=prompt, eos_token_ids=eos,
                    metadata_json=json.dumps(metadata), error=f"{type(exc).__name__}: {exc}",
                    **options,
                )
            records.append(record)
            if on_record is not None:
                on_record(record)  # Persistence failures abort; never relabel as model errors.
    return tuple(records)


def summarize_budget_sweep(records: Sequence[TrialRecord]) -> list[dict]:
    """Keep configurations separate and require identical repetition coverage."""
    if not records:
        raise ValueError("at least one record is required")
    first = records[0]
    context = lambda r: (r.workload, r.prompt_token_ids, r.eos_token_ids, r.metadata_json)
    if any(context(r) != context(first) for r in records):
        raise ValueError("comparison requires the same workload, prompt and execution metadata")
    groups = {}
    for record in records:
        groups.setdefault((record.variant, record.draft_budget, record.policy), []).append(record)
    repetitions = {r.repetition for r in next(iter(groups.values()))}
    if any({r.repetition for r in group} != repetitions for group in groups.values()):
        raise ValueError("configurations must have identical repetition IDs")
    return [
        {"variant": variant, "draft_budget": budget,
         "policy": asdict(policy) if policy is not None else None,
         **aggregate_trials(group)}
        for (variant, budget, policy), group in groups.items()
    ]
