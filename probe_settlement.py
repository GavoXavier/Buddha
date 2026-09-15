"""One demo order, watched from placement to settlement. Nothing else.

The settlement fix in ``execution.py`` refuses any deal whose ``close_price`` is
still 0, because on this account a just-accepted deal arrives already carrying a
close timestamp and a projected profit. That refusal is only useful if a real
exit price shows up later. This places a single minimum-stake demo order and
polls the deal storage until it does — or until the wait runs out, which is the
answer that would mean the fix needs rethinking.

Run:  python probe_settlement.py
"""

from __future__ import annotations

import asyncio
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.pocket_option import PocketOptionFeed
from execution import has_settled
from config import load_config

SYMBOL = "EURUSD_otc"
AMOUNT = 1
DURATION = 60
WATCH_SECONDS = 180


def show(moment: str, deal) -> None:
    if deal is None:
        print(f"  {moment:>7}  (no deal in storage yet)")
        return
    print(f"  {moment:>7}  close_price={deal.close_price!r:>10}  "
          f"profit={deal.profit!r:>8}  "
          f"close_ts={deal.close_timestamp!r:>16}  "
          f"settled={has_settled(deal)}")


async def main() -> int:
    cfg = load_config()
    if not cfg.pocket_is_demo:
        print("POCKET_IS_DEMO is not set — refusing to place anything.")
        return 2

    from pocket_option.contrib.deals import MemoryDealsStorage

    holder: dict = {}
    feed = PocketOptionFeed(
        cfg.pocket_ssid, cfg.pocket_uid, cfg.pocket_is_demo,
        region=cfg.pocket_region or None, is_fast_history=True,
        on_client=lambda client: holder.update(storage=MemoryDealsStorage(client)))

    await feed.connect()
    try:
        await asyncio.sleep(3)
        storage = holder.get("storage")
        if storage is None:
            print("the deals storage was never attached")
            return 1

        from pocket_option.models import Asset, DealAction

        print(f"placing {AMOUNT} on {SYMBOL} CALL for {DURATION}s (demo)…")
        deal = await storage.open_deal(
            asset=Asset(SYMBOL), amount=AMOUNT,
            action=DealAction.CALL, time=DURATION,
            is_demo=1, check_limits=True)
        deal_id = getattr(deal, "id", None) or getattr(deal, "request_id", None)
        print(f"accepted: {deal_id}")
        show("t+0", deal)

        for elapsed in range(10, WATCH_SECONDS + 1, 10):
            await asyncio.sleep(10)
            stored = await storage.get_deal(deal_id=deal_id)
            show(f"t+{elapsed}", stored)
            if has_settled(stored):
                print(f"\nA real exit price arrived after ~{elapsed}s.")
                return 0

        print(f"\nNo exit price within {WATCH_SECONDS}s — the deal never settled "
              f"in storage.")
        return 1
    finally:
        await feed.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
