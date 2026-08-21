"""Proposal accounting for successful speculative requests."""

from dataclasses import dataclass

from .verification import VerifiedToken


@dataclass(frozen=True, slots=True)
class ProposalAccounting:
    """Counts across rounds; emitted excludes EOS and the input prompt."""

    proposed: int = 0
    verified: int = 0
    accepted: int = 0
    emitted: int = 0
    rounds: int = 0

    def __post_init__(self) -> None:
        for name in ("proposed", "verified", "accepted", "emitted", "rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.accepted <= self.verified <= self.proposed:
            raise ValueError("expected accepted <= verified <= proposed")

    @property
    def acceptance_rate(self) -> float | None:
        """Accepted / verified, or None when no proposal was compared."""
        return self.accepted / self.verified if self.verified else None

    @property
    def proposal_yield(self) -> float | None:
        """Accepted / proposed, or None when no proposal was produced."""
        return self.accepted / self.proposed if self.proposed else None


class _RoundAccounting:
    """Validate event labels against the proposal before counting a comparison."""

    def __init__(self, proposals: tuple[int, ...]) -> None:
        self.proposals = proposals
        self.position = 0
        self.finished = False

    def record(self, event: VerifiedToken) -> tuple[int, int]:
        """Return (verified increment, accepted increment) for one event."""
        if self.finished:
            raise ValueError("verification yielded after a correction, bonus or fallback")
        if event.kind in ("accepted", "correction"):
            if self.position >= len(self.proposals):
                raise ValueError("verification comparison exceeds proposal length")
            matches = event.token_id == self.proposals[self.position]
            if matches != (event.kind == "accepted"):
                raise ValueError("verification event does not match its proposal")
            self.position += 1
            self.finished = event.kind == "correction"
            return 1, int(matches)
        if event.kind == "bonus":
            if not self.proposals or self.position != len(self.proposals):
                raise ValueError("bonus requires full proposal acceptance")
        elif event.kind == "fallback":
            if self.proposals or self.position:
                raise ValueError("fallback requires an empty proposal")
        else:
            raise ValueError(f"unknown verification event kind: {event.kind!r}")
        self.finished = True
        return 0, 0
