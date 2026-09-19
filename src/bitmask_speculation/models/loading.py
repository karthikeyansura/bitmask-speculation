"""Explicit model loading and text generation; optional dependencies stay lazy."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re

from ..engine import CONSTRAINED_MODES, MODES, DecodingEngine, GenerationResult


@dataclass(frozen=True, slots=True)
class ModelSpec:
    source: str
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("model source must be a nonempty path or Hub ID")
        if not Path(self.source).is_dir() and (
            not isinstance(self.revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None
        ):
            raise ValueError("Hub models require a full 40-character revision SHA")


def schema_document(schema: str | dict | bool) -> str:
    """Canonical JSON Schema input, without implicit file/network lookup."""
    if isinstance(schema, str):
        schema = json.loads(schema)
    if not isinstance(schema, (dict, bool)):
        raise ValueError("schema must be a JSON object or boolean")
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(child, str) or not child.startswith("#")):
                    raise ValueError("schema references must be local fragments")
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(schema)
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TransformersEngine:
    """Load once, then create independent request grammar state on each call.

    This is a single-device, uncached, greedy reference. It does not apply chat
    templates or generation-config processors, and does not dispatch CUDA kernels.
    """

    @classmethod
    def from_pretrained(
        cls, target: ModelSpec, draft: ModelSpec | None = None, *,
        device: str = "cpu", dtype: str = "float32", local_files_only: bool = False,
    ) -> "TransformersEngine":
        if not isinstance(target, ModelSpec) or (draft is not None and not isinstance(draft, ModelSpec)):
            raise TypeError("target and draft must be ModelSpec instances")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported dtype")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .transformers import TransformersGreedyModel

        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("supported devices are cpu and cuda")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device is unavailable")
        def options(spec):
            return dict(revision=spec.revision, local_files_only=local_files_only, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(target.source, **options(target))
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required")
        if draft is not None:
            other = AutoTokenizer.from_pretrained(draft.source, **options(draft))
            if (not other.is_fast or tokenizer.get_vocab() != other.get_vocab()
                or json.loads(tokenizer.backend_tokenizer.to_str()) != json.loads(other.backend_tokenizer.to_str())
                or tokenizer.special_tokens_map != other.special_tokens_map):
                raise ValueError("draft and target tokenizer definitions differ")
        def load(spec):
            model = AutoModelForCausalLM.from_pretrained(
                spec.source, torch_dtype=getattr(torch, dtype), **options(spec)
            ).to(resolved).eval()
            return model, TransformersGreedyModel(model)
        target_model, target_adapter = load(target)
        draft_adapter = None if draft is None else load(draft)[1]
        if max(tokenizer.get_vocab().values()) >= target_adapter.vocab_size:
            raise ValueError("tokenizer IDs exceed model vocabulary")
        eos = target_model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)
        if any(type(t) is not int or not 0 <= t < target_adapter.vocab_size for t in eos):
            raise ValueError("invalid target EOS token IDs")
        self = cls()
        self.tokenizer = tokenizer
        self.decoder = DecodingEngine(target_adapter, draft_adapter)
        self.eos_token_ids = eos
        self.target_spec, self.draft_spec = target, draft
        self._compiler = None
        return self

    def generate(
        self, prompt: str, *, max_new_tokens: int = 128,
        mode: str = "target_unconstrained", schema: str | dict | bool | None = None,
        draft_budget: int = 5, draft_policy=None,
    ) -> GenerationResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        if mode not in MODES:
            raise ValueError("unknown decoding mode")
        if (mode in CONSTRAINED_MODES) != (schema is not None):
            raise ValueError("schema is required exactly for constrained modes")
        if mode.startswith("c") and self.decoder.draft is None:
            raise ValueError("speculative modes require a draft model")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        grammar_factory = None
        if schema is not None:
            import xgrammar as xgr
            from ..grammar.xgrammar import XGrammarState

            document = schema_document(schema)
            if self._compiler is None:
                info = xgr.TokenizerInfo.from_huggingface(
                    self.tokenizer, vocab_size=self.decoder.target.vocab_size,
                    stop_token_ids=list(self.eos_token_ids),
                )
                self._compiler = xgr.GrammarCompiler(info, max_threads=1)
            compiled = self._compiler.compile_json_schema(document)
            grammar_factory = lambda: XGrammarState(compiled)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        result = self.decoder.generate(
            tokens, max_new_tokens=max_new_tokens, mode=mode,
            eos_token_ids=self.eos_token_ids, grammar_factory=grammar_factory,
            draft_budget=draft_budget, draft_policy=draft_policy,
        )
        return replace(result, text=self.tokenizer.decode(
            list(result.token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False,
        ))
