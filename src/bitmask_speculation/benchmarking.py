"""Baseline workloads and lossless trial records; no model loading or timing."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .decoding import DensityPolicy, GreedyResult, ProposalAccounting, RoundSchedule, SpeculativeResult

_VARIANTS = {"target_unconstrained", "target_constrained", "c1", "c2", "c3", "c4"}
_METADATA = {
    "source_revision", "target_revision", "draft_revision", "tokenizer_revision",
    "device", "precision", "dependencies", "timing_scope",
}


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validator(schema_json: str):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ImportError("install bitmask-speculation[benchmarks] for schema validation") from exc
    schema = json.loads(schema_json)
    if not isinstance(schema, (dict, bool)):
        raise ValueError("a JSON Schema must be an object or boolean")
    if isinstance(schema, dict) and schema.get("$schema", "https://json-schema.org/draft/2020-12/schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("baseline schemas use JSON Schema draft 2020-12")
    # Workloads must be self-contained. Do not fetch remote references in a run.
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and not isinstance(child, str):
                    raise ValueError("schema references must be strings")
                if key in {"$ref", "$dynamicRef"} and not child.startswith("#"):
                    raise ValueError("external schema references are unsupported")
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True, slots=True)
class Workload:
    case_id: str
    prompt: str
    schema_json: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.prompt, "prompt")
        _integer(self.max_new_tokens, "max_new_tokens", 1)
        _validator(self.schema_json)
        object.__setattr__(self, "schema_json", _canonical(json.loads(self.schema_json)))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()


def load_workloads(path: str | Path) -> tuple[Workload, ...]:
    """Load the versioned case list, resolving schema paths beside the manifest."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if set(data) != {"version", "cases"} or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("expected workload manifest version 1")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("cases must be a nonempty list")
    cases = []
    for entry in data["cases"]:
        if set(entry) != {"case_id", "prompt", "schema", "max_new_tokens"}:
            raise ValueError("unexpected workload fields")
        schema_path = (path.parent / entry["schema"]).resolve()
        if not schema_path.is_relative_to(path.parent):
            raise ValueError("schema must be inside the workload directory")
        cases.append(Workload(entry["case_id"], entry["prompt"], schema_path.read_text(), entry["max_new_tokens"]))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    workload: Workload
    variant: str
    draft_budget: int | None
    prompt_token_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    metadata_json: str
    repetition: int = 0
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_ids: tuple[int, ...] | None = None
    stop_reason: str | None = None
    eos_token_id: int | None = None
    accounting: ProposalAccounting | None = None
    exact_match: bool | None = None
    schema_valid: bool | None = None
    decode_seconds: float | None = None
    error: str | None = None
    policy: DensityPolicy | None = None
    schedule: tuple[RoundSchedule, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError("unknown baseline variant")
        speculative = self.variant in {"c1", "c2", "c3", "c4"}
        if self.variant == "c4":
            if not isinstance(self.policy, DensityPolicy) or self.draft_budget is not None:
                raise ValueError("C4 requires a policy and no fixed draft budget")
        else:
            if self.policy is not None or self.schedule:
                raise ValueError("only C4 records contain a policy or schedule")
            if speculative:
                _integer(self.draft_budget, "draft_budget", 1)
            elif self.draft_budget is not None:
                raise ValueError("target-only records have no draft budget")
        if not isinstance(self.schedule, tuple) or any(not isinstance(s, RoundSchedule) for s in self.schedule):
            raise TypeError("schedule must be a tuple of RoundSchedule decisions")
        _integer(self.repetition, "repetition")
        for name in ("prompt_token_ids", "eos_token_ids", "token_ids"):
            tokens = getattr(self, name)
            if name == "token_ids" and tokens is None:
                continue
            if not isinstance(tokens, tuple):
                raise TypeError(f"{name} must be a tuple")
            for token in tokens:
                _integer(token, name)
        if not self.prompt_token_ids:
            raise ValueError("prompt cannot be empty")
        stamp = datetime.fromisoformat(self.recorded_at)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("recorded_at requires a timezone")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict) or not _METADATA <= metadata.keys():
            raise ValueError("missing reproducibility metadata")
        for key, value in metadata.items():
            _text(value, key)
        object.__setattr__(self, "metadata_json", _canonical(metadata))
        if self.decode_seconds is not None:
            if isinstance(self.decode_seconds, bool) or not isinstance(self.decode_seconds, (int, float)):
                raise TypeError("decode_seconds must be numeric")
            if not math.isfinite(self.decode_seconds) or self.decode_seconds <= 0:
                raise ValueError("decode_seconds must be finite and positive")
            if metadata["timing_scope"] == "untimed":
                raise ValueError("untimed records cannot contain timings")
        if self.error is not None:
            _text(self.error, "error")
            if any(v is not None for v in (self.token_ids, self.stop_reason, self.eos_token_id, self.accounting, self.exact_match, self.schema_valid)):
                raise ValueError("failed trials cannot claim successful output or comparisons")
            if self.schedule:
                raise ValueError("failed trials cannot claim a completed schedule")
            return
        if self.token_ids is None or len(self.token_ids) > self.workload.max_new_tokens:
            raise ValueError("successful trial requires output within its budget")
        if self.stop_reason not in {"eos", "length", "grammar"}:
            raise ValueError("invalid stop reason")
        if self.stop_reason == "eos":
            _integer(self.eos_token_id, "eos_token_id")
            if self.eos_token_id not in self.eos_token_ids:
                raise ValueError("selected EOS is not configured")
        elif self.eos_token_id is not None:
            raise ValueError("only EOS termination records an EOS token")
        if any(t in self.eos_token_ids for t in self.token_ids):
            raise ValueError("content tokens must exclude EOS")
        if type(self.exact_match) is not bool or type(self.schema_valid) is not bool:
            raise TypeError("successful records require explicit comparison and schema results")
        if speculative != (self.accounting is not None):
            raise ValueError("only speculative records contain proposal accounting")
        if self.accounting is not None and self.accounting.emitted != len(self.token_ids):
            raise ValueError("emitted count differs from recorded tokens")
        if self.variant == "c4":
            if len(self.schedule) != self.accounting.rounds:
                raise ValueError("C4 requires one schedule decision per round")
            previous = -1
            for decision in self.schedule:
                offset = decision.output_offset
                if offset <= previous or offset > len(self.token_ids) or offset >= self.workload.max_new_tokens:
                    raise ValueError("invalid schedule output offsets")
                if previous == -1 and offset != 0:
                    raise ValueError("schedule must start at output offset zero")
                expected = self.policy.budget(decision.density)
                if decision.requested_budget != expected or decision.effective_budget != min(expected, self.workload.max_new_tokens - offset):
                    raise ValueError("schedule disagrees with policy or output budget")
                previous = offset

    @property
    def complete(self) -> bool:
        return self.error is None and self.stop_reason in {"eos", "grammar"}

    def to_json(self) -> str:
        return _canonical({"version": 1, **asdict(self)})

    @classmethod
    def from_json(cls, payload: str) -> "TrialRecord":
        data = json.loads(payload)
        version = data.pop("version")
        if type(version) is not int or version != 1:
            raise ValueError("unsupported trial record version")
        data["workload"] = Workload(**data["workload"])
        for name in ("prompt_token_ids", "eos_token_ids", "token_ids"):
            if data[name] is not None:
                data[name] = tuple(data[name])
        if data["accounting"] is not None:
            data["accounting"] = ProposalAccounting(**data["accounting"])
        if data.get("policy") is not None:
            data["policy"] = DensityPolicy(**data["policy"])
        data["schedule"] = tuple(RoundSchedule(**entry) for entry in data.get("schedule", ()))
        return cls(**data)


def record_trial(
    workload: Workload, *, variant: str, result: GreedyResult | SpeculativeResult,
    reference: GreedyResult, decode: Callable[[tuple[int, ...]], str],
    prompt_token_ids: Sequence[int], eos_token_ids: Sequence[int],
    metadata: Mapping[str, str], draft_budget: int | None = None,
    repetition: int = 0, decode_seconds: float | None = None,
    policy: DensityPolicy | None = None,
) -> TrialRecord:
    """Validate output outside the timed region; preserve raw tokens and counts.

    Supply an independently run target-only reference with the same grammar and
    numeric settings. This helper does not execute models or measure elapsed time.
    """
    generation = result.generation if isinstance(result, SpeculativeResult) else result
    validator = _validator(workload.schema_json)
    text = decode(generation.token_ids)
    try:
        # json.loads otherwise accepts nonstandard NaN/Infinity literals.
        def reject_constant(value):
            raise ValueError(f"non-JSON number: {value}")
        output = json.loads(text, parse_constant=reject_constant)
    except ValueError:
        schema_valid = False
    else:
        schema_valid = validator.is_valid(output)
    return TrialRecord(
        workload, variant, draft_budget, tuple(prompt_token_ids), tuple(eos_token_ids),
        _canonical(dict(metadata)), repetition=repetition,
        token_ids=generation.token_ids, stop_reason=generation.stop_reason,
        eos_token_id=generation.eos_token_id,
        accounting=result.accounting if isinstance(result, SpeculativeResult) else None,
        exact_match=generation == reference, schema_valid=schema_valid,
        decode_seconds=decode_seconds, policy=policy,
        schedule=result.schedule if isinstance(result, SpeculativeResult) else (),
    )


def aggregate_trials(records: Sequence[TrialRecord]) -> dict:
    """Aggregate repetitions of one configuration; never average trial ratios."""
    if not records:
        raise ValueError("at least one trial is required")
    def key(r):
        return (r.workload, r.variant, r.draft_budget, r.policy, r.prompt_token_ids, r.eos_token_ids, r.metadata_json)
    if any(key(r) != key(records[0]) for r in records):
        raise ValueError("aggregate only identical workloads and execution configurations")
    if len({r.repetition for r in records}) != len(records):
        raise ValueError("duplicate repetition IDs")
    successful = [r for r in records if r.error is None]
    errors = len(records) - len(successful)
    timed = not errors and all(r.decode_seconds is not None for r in successful)
    seconds = math.fsum(r.decode_seconds for r in successful) if timed else None
    emitted = sum(len(r.token_ids) for r in successful)
    counts = [r.accounting for r in successful if r.accounting is not None]
    totals = {name: sum(getattr(c, name) for c in counts) for name in ("proposed", "verified", "accepted", "rounds")}
    ratios_available = bool(counts) and not errors
    return {
        "trials": len(records), "errors": errors,
        "completed": sum(r.complete for r in successful),
        "length_limited": sum(r.stop_reason == "length" for r in successful),
        "exact_matches": sum(r.exact_match for r in successful),
        "schema_valid": sum(r.schema_valid for r in successful),
        "completed_schema_valid": sum(r.complete and r.schema_valid for r in successful),
        "emitted": emitted, **totals,
        "acceptance_rate": totals["accepted"] / totals["verified"] if ratios_available and totals["verified"] else None,
        "proposal_yield": totals["accepted"] / totals["proposed"] if ratios_available and totals["proposed"] else None,
        "decode_seconds": seconds,
        "observed_tokens_per_second": emitted / seconds if seconds is not None else None,
    }
