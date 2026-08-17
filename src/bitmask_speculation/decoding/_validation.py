"""Input validation shared by decoding primitives."""

from collections.abc import Iterable, Sequence


def vocabulary_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("vocab_size must be an integer")
    if value <= 0:
        raise ValueError("vocab_size must be positive")
    return value


def token_id(value: int, vocab_size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer token ID")
    if not 0 <= value < vocab_size:
        raise ValueError(f"{name} must be in [0, {vocab_size})")
    return value


def request_inputs(
    vocab_size: int,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int],
    grammar_vocab_size: int | None = None,
) -> tuple[tuple[int, ...], frozenset[int]]:
    """Validate a request and snapshot caller-owned token sequences."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    vocab_size = vocabulary_size(vocab_size)
    prompt = tuple(token_id(t, vocab_size, "prompt token") for t in prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must include at least one token")
    eos = frozenset(token_id(t, vocab_size, "EOS token") for t in eos_token_ids)
    if grammar_vocab_size is not None and vocabulary_size(grammar_vocab_size) != vocab_size:
        raise ValueError("grammar and model vocabularies differ")
    return prompt, eos
