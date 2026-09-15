"""Picking the single best setup when several assets qualify.

The bot trades one minute at a time, so when ten assets all produce a signal on
the same bar it has to choose exactly one. Ranking on confidence first, then
score, then payout, and finally the symbol name means the choice is
deterministic — the same bar always yields the same pick, which makes the bot's
behaviour reproducible and its logs comparable between runs.

Per-asset cooldown stops one busy pair from monopolising the feed: an asset that
signalled recently sits out a few minutes so the next-best market gets a turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .engine import Signal


@dataclass
class Candidate:
    """One asset's qualifying signal, with the context needed to rank it."""

    asset: str
    signal: Signal
    payout: int = 0
    bars: int = 0

    @property
    def confidence(self) -> float:
        return self.signal.confidence

    @property
    def score(self) -> int:
        return self.signal.score

    def rank_key(self) -> tuple:
        # Higher confidence/score/payout first; symbol name breaks ties so the
        # ordering never depends on dict iteration order.
        return (-self.confidence, -self.score, -self.payout, self.asset)


def eligible(candidates: Sequence[Candidate],
             last_signal_at: Mapping[str, float],
             now: float,
             cooldown_seconds: float,
             min_confidence: float = 0.0) -> list[Candidate]:
    """Drop candidates that are on cooldown or below the confidence floor."""
    out: list[Candidate] = []
    for cand in candidates:
        if cand.confidence < min_confidence:
            continue
        last = last_signal_at.get(cand.asset)
        if last is not None and now - last < cooldown_seconds:
            continue
        out.append(cand)
    return out


def rank(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.rank_key())


def select_best(candidates: Sequence[Candidate],
                last_signal_at: Mapping[str, float],
                now: float,
                cooldown_seconds: float,
                min_confidence: float = 0.0) -> Optional[Candidate]:
    """Highest-ranked eligible candidate, or None if nothing qualifies."""
    pool = eligible(candidates, last_signal_at, now, cooldown_seconds, min_confidence)
    if not pool:
        return None
    return min(pool, key=lambda c: c.rank_key())


def describe(candidates: Sequence[Candidate], limit: int = 3) -> str:
    """Debug line: what was in the running and how it placed."""
    if not candidates:
        return "no candidates"
    parts = [
        f"{c.asset} {c.signal.direction} score={c.score} conf={c.confidence:.2f} "
        f"payout={c.payout}%"
        for c in rank(candidates)[:limit]
    ]
    return " | ".join(parts)
