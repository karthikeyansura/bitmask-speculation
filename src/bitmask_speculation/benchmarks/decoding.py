"""Load pinned models once, compile workloads, and persist complete budget sweeps."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .artifacts import RunDirectory, environment, gpu_environment
from ..benchmarking import load_workloads
from ..budget_sweep import run_budget_sweep, summarize_budget_sweep
from ..models.loading import ModelSpec, TransformersEngine


def model_identity(spec):
    path = Path(spec.source)
    if not path.is_dir():
        return {'source': spec.source, 'revision': spec.revision}
    files = {}
    for item in sorted(path.rglob('*')):
        if not item.is_file() or '.cache' in item.relative_to(path).parts:
            continue
        digest = sha256()
        with item.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        files[str(item.relative_to(path))] = digest.hexdigest()
    if not files:
        raise ValueError('local model directory is empty')
    identity = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'source': str(path.resolve()), 'revision': f'sha256:{identity}', 'files': files}


def run_decoding(args):
    if args.repetitions < 2 or args.warmups < 1:
        raise ValueError('use at least two repetitions and one warmup')
    if not args.budgets or min(args.budgets) < 1 or len(set(args.budgets)) != len(args.budgets):
        raise ValueError('budgets must be unique positive integers')
    workloads = load_workloads(args.workloads)
    target = ModelSpec(args.target, args.target_revision)
    draft = ModelSpec(args.draft, args.draft_revision)
    with RunDirectory(args.output, {'kind': 'decoding', 'configuration': vars(args),
                                    'workloads': [asdict(w) for w in workloads]}) as run:
        identities = {'target': model_identity(target), 'draft': model_identity(draft)}
        run.write('models.json', identities)
        start = perf_counter()
        engine = TransformersEngine.from_pretrained(target, draft, device=args.device,
            dtype=args.dtype, backend=args.backend, local_files_only=args.local_files_only)
        import torch
        device = torch.device(args.device)
        synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
        synchronize()
        load_seconds = perf_counter() - start
        env = environment()
        if device.type == 'cuda':
            env.update(gpu_environment(device))
        env.update(torch_threads=torch.get_num_threads(),
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   float32_matmul_precision=torch.get_float32_matmul_precision())
        run.write('environment.json', env)
        run.write('setup.json', {'model_load_seconds': load_seconds})
        tokenizer_json = json.dumps({'backend': json.loads(engine.tokenizer.backend_tokenizer.to_str()),
            'special_tokens': engine.tokenizer.special_tokens_map,
            'chat_template': engine.tokenizer.chat_template}, sort_keys=True)
        run.write('tokenizer.json', json.loads(tokenizer_json))
        metadata = {'source_revision': 'sha256:' + env['source_sha256'],
            'target_revision': identities['target']['revision'], 'draft_revision': identities['draft']['revision'],
            'tokenizer_revision': 'sha256:' + sha256(tokenizer_json.encode()).hexdigest(),
            'device': str(device), 'precision': args.dtype, 'backend': args.backend,
            'dependencies': json.dumps(env, sort_keys=True), 'timing_scope': 'full_prefix_loop',
            'prompt_format': 'chat_template_no_thinking' if args.chat_template else 'raw'}
        summaries = []
        for case in workloads:
            start = perf_counter()
            prompt = case.prompt
            if args.chat_template:
                prompt = engine.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=not args.chat_template)
            factory = engine.prepare_grammar(case.schema_json)
            synchronize()
            run.append('case-setup.jsonl', json.dumps({'case_id': case.case_id,
                'prompt': prompt, 'prompt_token_ids': prompt_ids,
                'tokenization_and_grammar_seconds': perf_counter() - start}))
            records = run_budget_sweep(engine.decoder.target, engine.decoder.draft, case,
                prompt_token_ids=prompt_ids, eos_token_ids=engine.eos_token_ids,
                grammar_factory=factory,
                decode=lambda ids: engine.tokenizer.decode(list(ids), skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False),
                metadata=metadata, synchronize=synchronize, fixed_budgets=args.budgets,
                repetitions=args.repetitions, warmups=args.warmups,
                on_record=lambda record: run.append('trials.jsonl', record.to_json()),
                on_reference=lambda refs: run.append('references.jsonl', json.dumps({
                    'case_id': case.case_id,
                    'unconstrained': asdict(refs[False]), 'constrained': asdict(refs[True])})))
            summaries.append({'case_id': case.case_id, 'configurations': summarize_budget_sweep(records)})
        run.write('summary.json', summaries)
