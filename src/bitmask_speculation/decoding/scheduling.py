"""Fixed-threshold draft scheduling from grammar support density."""

from dataclasses import dataclass
import math


def _density(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("density must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("density must be finite and within [0, 1]")


def _budget(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("draft budgets must be integers")
    if value < 1:
        raise ValueError("draft budgets must be positive")


@dataclass(frozen=True, slots=True)
class DensityPolicy:
    """Three configurable density bands; thresholds are experiment parameters."""

    low_density: float = 0.005
    high_density: float = 0.02
    min_budget: int = 1
    max_budget: int = 8

    def __post_init__(self) -> None:
        _density(self.low_density)
        _density(self.high_density)
        if self.low_density >= self.high_density:
            raise ValueError("density thresholds must be strictly increasing")
        _budget(self.min_budget)
        _budget(self.max_budget)
        if self.min_budget > self.max_budget:
            raise ValueError("min_budget cannot exceed max_budget")

    def budget(self, density: float) -> int:
        _density(density)
        if density < self.low_density:
            return self.max_budget
        if density < self.high_density:
            return (self.min_budget + self.max_budget) // 2
        return self.min_budget


@dataclass(frozen=True, slots=True)
class RoundSchedule:
    """One decision at the committed prefix, before any drafting occurs."""

    density: float
    requested_budget: int
    effective_budget: int
    output_offset: int

    def __post_init__(self) -> None:
        _density(self.density)
        _budget(self.requested_budget)
        _budget(self.effective_budget)
        if self.effective_budget > self.requested_budget:
            raise ValueError("effective budget cannot exceed requested budget")
        if isinstance(self.output_offset, bool) or not isinstance(self.output_offset, int):
            raise TypeError("output_offset must be an integer")
        if self.output_offset < 0:
            raise ValueError("output_offset cannot be negative")
