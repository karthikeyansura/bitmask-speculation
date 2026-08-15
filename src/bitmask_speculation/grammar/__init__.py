"""Request-owned grammar state, independent of model frameworks."""

from .state import GrammarState, GrammarTerminatedError

__all__ = ["GrammarState", "GrammarTerminatedError"]
