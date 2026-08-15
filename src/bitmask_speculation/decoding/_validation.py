"""Input validation shared by decoding primitives."""


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
