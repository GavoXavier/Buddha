"""Which assets the bot is allowed to signal on.

Three filters stack:

1. **Instrument kind** — ``major`` (8 majors crossed with each other),
   ``forex`` (every currency pair, majors + minors) or ``all``.
2. **Payout floor** — a 30%-payout asset is a losing bet even at 60% accuracy,
   so anything under ``min_payout`` is dropped. Live payouts on this account
   ranged from 15% to 92%.
3. **Liveness** — the feed must have delivered ticks recently (checked per
   minute by the scheduler against ``MarketState.healthy``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping

log = logging.getLogger("pocket.universe")

# ISO 4217 currency codes (plus a few non-ISO ones Pocket Option lists) used to
# tell a forex pair apart from crypto (BTC/ETH/...), metals (XAU/XAG), stocks
# (#AAPL) and indices (DJI30/SP500). A currency pair is a 6-char symbol whose
# two 3-char halves are both in this set.
CURRENCY_CODES = {
    "AED", "AUD", "BHD", "CAD", "CHF", "CNY", "DZD", "EUR", "GBP", "HUF",
    "IDR", "IRR", "JOD", "JPY", "KES", "LBP", "MAD", "MXN", "MYR", "NGN",
    "NOK", "NZD", "OMR", "PHP", "PKR", "QAR", "RUB", "SAR", "SGD", "SYP",
    "TND", "TRY", "USD", "ZAR",
}

# The 8 most-liquid currencies. Pairs built only from these drop the thin,
# noisy exotic crosses (KES, NGN, IRR, ...).
MAJOR_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}

MODE_MAJOR = "major"
MODE_FOREX = "forex"
MODE_ALL = "all"
VALID_MODES = (MODE_MAJOR, MODE_FOREX, MODE_ALL)


@dataclass(frozen=True)
class AssetMeta:
    """Live broker metadata for one symbol."""

    symbol: str
    payout: int = 0
    digits: int = 0
    is_otc: bool = False
    active: bool = True

    @property
    def display(self) -> str:
        is_otc = self.symbol.endswith("_otc")
        base = self.symbol[:-4] if is_otc else self.symbol
        if len(base) == 6 and base[:3].isalpha() and base[3:].isalpha():
            base = f"{base[:3]}/{base[3:]}"
        return f"{base} OTC" if is_otc else base


def _base(asset: str) -> str:
    return asset[:-4] if asset.endswith("_otc") else asset


def is_currency_pair(asset: str) -> bool:
    """True if ``asset`` is a forex pair, not a stock/index/crypto."""
    base = _base(asset)
    return len(base) == 6 and base[:3] in CURRENCY_CODES and base[3:] in CURRENCY_CODES


def is_major_pair(asset: str) -> bool:
    """True if ``asset`` crosses two major currencies (e.g. 'EURUSD_otc')."""
    base = _base(asset)
    return len(base) == 6 and base[:3] in MAJOR_CURRENCIES and base[3:] in MAJOR_CURRENCIES


def matches_mode(asset: str, mode: str) -> bool:
    if mode == MODE_ALL:
        return True
    if mode == MODE_FOREX:
        return is_currency_pair(asset)
    return is_major_pair(asset)


def payout_of(metas: Mapping[str, AssetMeta], asset: str) -> int:
    meta = metas.get(asset)
    return meta.payout if meta else 0


def _candidates(metas: Mapping[str, AssetMeta], mode: str, min_payout: int,
                otc_only: bool) -> list[str]:
    return [
        symbol for symbol, meta in metas.items()
        if meta.active and matches_mode(symbol, mode)
        and meta.payout >= min_payout
        and (not otc_only or meta.is_otc)
    ]


def select_assets(metas: Mapping[str, AssetMeta], mode: str,
                  min_payout: int = 0, otc_only: bool = False) -> list[str]:
    """Symbols to track, sorted by payout (best first) then name.

    ``otc_only`` keeps the 24/7 synthetic feeds, which is what a bot running
    around the clock wants — the real (non-OTC) pairs are closed at weekends
    and thin overnight. If that filter would leave nothing to trade it is
    dropped rather than leaving the bot idle, since a silent bot is worse than
    a weekday-only one.
    """
    if not metas:
        return []
    picked = _candidates(metas, mode, min_payout, otc_only)
    if not picked and otc_only:
        log.info("no OTC assets match - falling back to all hours")
        picked = _candidates(metas, mode, min_payout, False)
    picked.sort(key=lambda s: (-metas[s].payout, s))
    return picked


def describe_skipped(metas: Mapping[str, AssetMeta], mode: str,
                     min_payout: int, otc_only: bool = False) -> dict[str, int]:
    """Counts of why candidates were dropped, for the startup log."""
    reasons = {"inactive": 0, "wrong_kind": 0, "low_payout": 0, "not_otc": 0}
    for symbol, meta in metas.items():
        if not meta.active:
            reasons["inactive"] += 1
        elif not matches_mode(symbol, mode):
            reasons["wrong_kind"] += 1
        elif meta.payout < min_payout:
            reasons["low_payout"] += 1
        elif otc_only and not meta.is_otc:
            reasons["not_otc"] += 1
    return reasons


def format_universe(symbols: Iterable[str], metas: Mapping[str, AssetMeta],
                    limit: int = 6) -> str:
    """Human-readable 'EUR/USD OTC 92%, ...' line for startup logging."""
    items = [f"{metas[s].display} {metas[s].payout}%"
             for s in symbols if s in metas]
    if len(items) > limit:
        return ", ".join(items[:limit]) + f", +{len(items) - limit} more"
    return ", ".join(items)
