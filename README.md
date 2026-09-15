# Pocket Option Signal Bot

A Python bot that generates binary-options **CALL/PUT** signals from a confluence
of technical indicators and delivers them to your **Telegram** private chat — at
most one signal per bar, naming the exact second to enter.

Ships on **5-minute bars**: a signal at 16:00 for an entry at 16:05, expiring at
16:10. One bar length (`CANDLE_PERIOD`) sets the whole rhythm; see
[The cadence](#the-cadence).

```
┌──────────────────┐  Socket.IO/WS   ┌───────────────────┐  Bot API    ┌──────────────┐
│  Pocket Option   │ ───────────────► │   Python bot      │ ──────────► │  Telegram    │
│  (SSID auth)     │  real-time ticks│  ticks → candles  │ sendMessage │ private chat │
│                  │                 │  → signal engine  │ ◄────────── │  /commands   │
└──────────────────┘                 └───────────────────┘   getUpdates└──────────────┘
```

---

## The cadence

Everything runs off one bar length. **The shipped default is 5-minute bars**: a
signal at 16:00 for an entry at 16:05, expiring at 16:10, one trade live at a
time. Set `CANDLE_PERIOD=60` with `EXPIRY=1m` and `SIGNAL_LEAD_SECONDS=10` for
the original 1-minute rhythm.

One whole bar is the longest lead allowed, which leaves two timing models:

```
Full-bar lead — 5m bars (the default): signal 16:00, entry 16:05, closes 16:10

 16:00            16:05                      16:10              16:15
   │                │                          │                  │
 wake, judge the   the trade opens       the option expires    next entry
 bar that just     (entry = the close    (exit = the close of
 closed, send      of the bar ending      the bar ending here)
 the best setup    here)
   │                │                          │                  │
   └── signal sent ─┴──────────────────────────┘                  │
                                       result → Telegram         │
                                                                  │
                          └── next signal sent at 16:05 for ──────┘
                              an entry as this one expires
```

```
Short lead — 1m bars, 10s: the bar being judged is still in progress

   ...    bar closing at T          the trade's minute         ...
 ─────────┬──────────────────────────┬──────────────────────────┬─────
        T-10s                        T                       T+60s
          │                          │                          │
     wake, judge the bar        the trade opens          the option expires
     about to close, send       (entry price = the       (exit price = the close
     the best setup             close of that bar)        of the bar ending here)
```

Both prices are **bar closes the feed delivered**, not snapshots taken whenever a
message happened to be read. That is what makes the win rate *measured* rather
than estimated: entry and exit both land on real bar boundaries, so the outcome
is read off two candles.

Details that matter:

- **Which bars the engine judges depends on the lead, and it is not a detail.**
  At a 10-second lead the engine is handed the bar closing at `T` *while it is
  still forming* — a snapshot that is right, because ten seconds is nearly a
  finished bar. At a full-bar lead that bar has not started yet, so the engine
  judges only bars that have **already closed**. Feeding it a zero-second stub
  instead would shift every indicator window by one bar.
- **One market per bar.** When several markets qualify on the same bar, the
  highest-ranked one wins: confidence, then score, then payout, then symbol name.
  The ranking is deterministic, so the same bar always yields the same pick.
- **Cooldown.** A market that has just signalled sits out `COOLDOWN_SECONDS` so
  one busy pair cannot monopolise the feed. A 5-minute rhythm never reaches a
  120-second cooldown; a 1-minute one does.
- **Circuit breaker.** After `MAX_CONSECUTIVE_LOSSES` losses in a row the bot
  pauses itself for `BREAKER_PAUSE_MINUTES` and tells you why.
- **`MARKET_TZ_OFFSET_HOURS` is the clock in every message.** It is a setting
  rather than a constant because a trade placed by hand has to land on the second
  the message names, which means matching the clock in front of you.

---

## How often it actually fires

**The cadence is "at most one signal per bar". The rate is not.** Whether any bar
produces a signal depends entirely on the selectivity dials, and indicators agree
far less often than intuition suggests.

The measurement below was taken on **1-minute bars**, which is what the dials were
calibrated against. The bar length changes what they mean — `MTF_FACTOR=5` is a
5-minute confirmation on 1m bars and a 25-minute one on 5m bars, and
`TREND_MIN_DISTANCE_PCT` is a move per 50 bars either way — so re-measure with
`python backtest.py` after the store has a couple of days of 5-minute bars in it.

Measured on the shipped dials, replaying tick-built bars from a simulated run of
three major pairs (`python backtest.py --dir <store>`, 8.3 hours, 500 minutes
judged):

```
Signals: 16  in 8.3h  =  1.9/hour
Minutes judged: 500; a signal on 3.2% of them

Why the rest produced nothing (1497 judgements):
  indicators disagree (tie)                    43.1%
  agreement below MIN_SCORE                    30.2%
  rejected by a gate (trend / MTF / ATR / S-R) 23.5%
  not enough indicators warm                    3.3%
```

Three things to read out of that:

1. **Agreement is the binding constraint**, not the trend filter. Roughly
   three bars in four fail to produce a direction at all — the indicators
   either cancel out or do not reach `MIN_SCORE` between them. That is the nature
   of a short timeframe, not a bug.
2. **Adding markets does not scale the rate.** Three markets gave 1.9/hour here;
   a five-market store of the same size gave 1.6/hour. Correlated majors go quiet
   together, so a quiet market is usually *all* of them quiet at once, and only
   one market can be picked per bar anyway. Expect a couple of signals an hour on
   1m bars, not sixty — and no realistic universe width closes that gap.
3. **Only the frequency transfers to real trading.** These numbers come from a
   simulated feed whose prices follow a slow cycle, which a trend-following
   engine beats by construction — the win rate above is an artefact of the
   simulator and says nothing about Pocket Option. The rejection breakdown is
   the useful part, because it measures the engine's selectivity rather than the
   market.

Reproduce it yourself with `python backtest.py` (it replays *your* collected
candles, so it starts reporting after the bot has been running a while, and the
sample is capped at `MAX_BARS` per market — 500 bars is 8.3 hours of 1-minute
bars, or 41 hours of 5-minute ones).

### The dials, and what they cost

| Dial | Default | Loosening it | Tightening it |
|------|---------|--------------|---------------|
| `MIN_SCORE` | `2` | `1` = trade on a single indicator. Far more signals, each on much thinner evidence — the regime the bot ran in before this rewrite (no trend filter, no S/R) | `3`+ = only broad agreement |
| `MIN_COMPONENTS` | `2` | `1` = one warm indicator is enough to judge | More independent evidence required |
| `TREND_FLAT_MIN_SCORE` | `3` | `0` = ignore a flat market and trade ranges freely | `9` = sit out ranges entirely |
| `MIN_CONFIDENCE` | `0.0` | — | Rejects noisy multi-vote setups |

`TREND_FLAT_MIN_SCORE` deserves a note. A *trending* market vetoes the
counter-trend direction outright; a *flat* one is ambiguous rather than hostile,
so instead of vetoing both directions it demands a stronger reading. In the
replay above the market read flat 42% of the time — and because **FX majors are
correlated**, when the dollar goes quiet every pair reads flat at once. A hard
flat veto starves the bot; a strength requirement keeps it in the market for the
setups that are clear even without a trend.

---

## Why candles are built from ticks, never from history

This is the single most important design decision in the project, and it was
made from measurement, not preference.

Pocket Option's history endpoint and its live tick stream **are not the same
price path**. On the OTC assets the bot trades, replaying history and watching
ticks diverged by **6–18 pips over the same 5 minutes** — different highs,
different lows, different closes. On real (non-OTC) assets the two agreed to a
constant offset (EURUSD held at +0.0009), which is what a price path looks like
when it is the same instrument quoted two ways. The OTC series are the broker's
own book, so their history is not a record of what you would have traded.

So history is used for exactly one thing — the asset list — and every bar the
engine sees is aggregated here from ticks this process actually received:

- Buckets are wall-clock aligned: `floor(ts / period) * period`, labelled by the
  bar's **open** time (the broker labels by close — the two conventions must not
  be mixed).
- A bucket that received no ticks never becomes a bar. Nothing is invented, and
  a market that goes quiet is reported as stale and skipped.
- Those bars are persisted (`CANDLE_STORE_DIR`), so a restart resumes with a warm
  buffer in seconds instead of spending another hour blind. Broker history could
  never be used for this warm-up, for the reason above.
- A hole longer than `MAX_GAP_BARS` bars **drops the bars before it** and the
  market warms up again. The indicators assume evenly spaced bars, so a window
  spanning a hole reads the whole outage as a single bar's move: after a
  92-minute feed outage in which EURUSD moved 48 pips, RSI and Stochastic were
  pinned to the same extreme and voted together, producing a signal on every bar
  off the outage rather than off the market. Only the older side is dropped — the
  bars after the hole are still good — and a persisted series is trimmed the same
  way on restore, so a restart cannot re-seed a buffer that spans the hole.

The cost is honest warm-up: the fast indicators (RSI, Bollinger, Stochastic) come
online after ~20 bars, MACD after ~36, the 50-bar trend EMA and the MTF
confirmation later still — 56 bars before everything is live. The bot **degrades
gracefully** while it fills in: it signals as soon as `MIN_COMPONENTS`
directional indicators are warm, and messages you "warming up: 12/56 bars"
rather than staying silently broken. Raise `MAX_BARS` to hold more history;
restart warm-up is free once the store has data.

**Warm-up is counted in bars, so it scales with the bar length.** On 5-minute
bars the first possible signal is 18 bars ≈ **90 minutes** after a cold start, and
full readiness is 56 bars ≈ **4h40m**. The store is stamped with the period it was
built at, so changing `CANDLE_PERIOD` discards it and the clock restarts — the
first signals after a change are not evidence of anything.

---

## Project layout

```
POCKET/
├── main.py               # wiring: feed → market → scheduler, reconnect supervisor
├── config.py             # .env → Config, with fail-fast validation
├── backtest.py           # replay the candle store through the live signal path
├── reconcile.py          # check our WIN/LOSS labels against the broker's
├── journal.py            # per-trade record of every signal sent
├── execution.py          # demo-only order placement, so the broker settles it
├── stats.py              # win/loss totals, per-market, per-hour, persisted
├── engine/scheduler.py   # THE minute loop: boundaries, cadence, settlement
├── market/
│   ├── aggregator.py     # ticks → bars, staleness, buffers
│   ├── store.py          # atomic JSON persistence of bars
│   ├── universe.py       # which markets to trade
│   └── clock.py          # RealClock / VirtualClock (makes timing testable)
├── signals/
│   ├── indicators.py     # EMA, RSI, MACD, Bollinger, Stochastic, ATR
│   ├── engine.py         # votes, gates, confidence, readiness
│   └── ranking.py        # picking one market out of many
├── data/                 # base.py, pocket_option.py, simulated.py
├── telegram/             # sender.py (formatting), control.py (/commands)
└── tests/                # 287 tests, ~14 seconds, no network
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit .env
```

### 1. Telegram (required for signals)

1. Message **@BotFather** → `/newbot` → copy the **bot token**.
2. Open a chat with your new bot and send it any message (e.g. `/start`).
3. Get your chat id from
   `https://api.telegram.org/bot<TOKEN>/getUpdates` → copy `chat.id`.
4. Put both in `.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

### 2. Pocket Option SSID (live mode only)

1. Log in to Pocket Option in your browser.
2. DevTools (F12) → **Network** → filter **WS**.
3. Open the websocket and find the message starting with `42["auth"`.
4. Copy `session` → `POCKET_SSID` and `uid` → `POCKET_UID` in `.env`.

> The Pocket Option API is unofficial and can change without notice. If the feed
> errors, `data/pocket_option.py` is the only file that should need adjusting.

## Running

```bash
FEED=simulated python main.py       # offline: no secrets, no account
FEED=pocket_option python main.py   # live ticks from your account, signals only
FEED=pocket_option POCKET_IS_DEMO=1 AUTO_TRADE=1 python main.py   # demo, and trades
```

A failed session (dead socket, expired SSID, a feed that stops ticking) tears
itself down and reconnects with exponential backoff, restoring candles from disk
so a reconnect does not cost another warm-up. Outage notices are throttled: the
first failure, then one every ten attempts.

### Leaving it running unattended

Use `-u`. Redirected to a file, Python block-buffers stdout, so `bot.log` sits
empty for minutes and then arrives in 8 KB lumps — which looks exactly like a
hung bot, and defeats any `tail -f` watching it.

```bash
nohup python -u main.py > bot.log 2>&1 &
```

The supervisor also takes a **wake lock** (`SetThreadExecutionState`) for as long
as it runs, so the host does not sleep mid-session. This is not belt-and-braces:
entering Modern Standby freezes the process rather than killing it, so the socket
dies while the bot still looks alive and the reconnect loop never sees a failure
to react to. The lock is released on exit and keeps the *system* awake only — the
screen may still switch off.

Three things no wake lock can cover, all of which need the machine rather than
the code:

| | |
|---|---|
| **Lid close** | a user-initiated transition, not an idle one. Set "when I close the lid" to *Do nothing*. |
| **Battery** | the low-battery hibernate is unconditional. Keep it plugged in. |
| **Reboot or crash** | nothing restarts it. A Scheduled Task with the default 3-day execution limit disabled will. |

### Telegram commands

| Command | Effect |
|---------|--------|
| `/status` | markets live, bars warm, when the next entry is |
| `/stats` | win rate overall, best/worst markets, best hours |
| `/pause` / `/resume` | stop and restart signalling |
| `/stop` | shut the bot down cleanly (flushes candles and stats) |

## Tests

```bash
python -m pytest -q                 # or: python -m unittest discover -s tests
```

Timing is the part that cannot be verified by running the bot for an hour, so the
scheduler, aggregator and feed all take a `Clock`. Tests drive a `VirtualClock`
and replay hours of market in milliseconds: the suite covers entry/exit landing
on bar boundaries, cooldown and circuit-breaker behaviour, an end-to-end
simulated session, and an assertion that **every signal enters on a minute
boundary exactly 60 seconds before its expiry**.

## Replay tool

```bash
python backtest.py                  # every market in the store
python backtest.py EURUSD_otc --hours 6
```

It replays the bot's own collected candles through the *same* engine, ranking and
outcome functions the live loop calls, and reports the signal rate plus a
breakdown of why the other minutes produced nothing. That breakdown is how you
should choose `MIN_SCORE` rather than guessing.

Two caveats it prints for a reason: it judges **finished** bars (a live signal
fires 10 seconds early on a bar that still has ticks to come), so its win rate is
a ceiling rather than a prediction; and the sample is only as long as the bot has
been running. Anything under a few days is a rumour, not a result.

## Trading the demo account automatically

```bash
FEED=pocket_option POCKET_IS_DEMO=1 AUTO_TRADE=1 python main.py
```

Off by default. With `AUTO_TRADE=1` the bot places the order itself, on the second
the signal named, and the WIN/LOSS it reports in Telegram is **what the broker
paid** rather than what our own bars said. That is the difference that matters:
signalling alone means every result is the bot grading its own homework.

Three refusals keep it on the demo account, and they are independent:

1. `config.py` rejects `AUTO_TRADE=1` without `FEED=pocket_option` and
   `POCKET_IS_DEMO=1` — before a socket is opened or the session string is used.
2. After the session authorises, the broker's *own* answer about the account
   (`authorization_data.is_demo`) is checked. A session string copied from a
   funded login cannot get past this by being mislabelled in `.env`: the bot
   reports it and carries on signal-only.
3. Every order carries `is_demo=1` explicitly, not just the config.

There is no setting that trades a funded account. To trade real money, place the
trades yourself.

Each order runs on a task of its own. `open_deal` waits up to 30 seconds for the
server to confirm and the settlement takes a further minute; awaiting either
inside the bar loop would cost the next signal, so the loop submits and moves
on, and reads the settlement when it next settles that trade.

What Telegram shows — the signal as before, then the broker's verdict:

| Result | Meaning |
|--------|---------|
| `✅ WIN` / `❌ LOSS` | settled by the broker, and counted |
| `↩️ REFUNDED` | profit exactly zero — a third outcome, excluded from every rate |
| `⚠️ NOT PLACED` | the broker never accepted the order; **not** a loss |

The startup message carries an `⚙️ auto-trade ON (demo, 1/trade, 5m)` line, or
the reason it is off, so "is this thing trading my account?" never needs a config
file to answer.

The journal keeps **both** answers for every trade — our label from the bar
closes, and the broker's settlement beside it. That is what keeps label agreement
measurable, which is the next section.

### Telling a settlement from a projection

A deal is not a result the moment the server accepts it, and the SDK's own
`Deal.closed` says it is. `closed` is `close_timestamp is not None`, and the
server sets that **when the deal opens** — it is the second the option is *due*
to expire, not a record that it did. Two more fields read like a result without
being one: `close_price` arrives as `0`, because no exit price exists yet, and
`profit` arrives already filled in with the payout the trade *would* win.

Trusting any of them reports every order as an instant win of exactly its
potential profit. Measured live: four orders in a row "settled" in the same
millisecond they were placed, each with `profit == amount x percent_profit/100`
and `close_price == 0`. So a deal counts as settled only when it carries a real
exit price — a test that needs no clock, which matters because the broker's
timestamps are on a different clock from ours (see the offset note below). A
deal that is still open is left unrecorded and the bot's own bar-close label
stands, which is honest: it says what we saw, not what we hope was paid.

## Does our "WIN" mean the same as theirs?

Everything above measures the bot against *itself*: unless `AUTO_TRADE=1` is on,
the WIN/LOSS you get in Telegram is read off two bar closes this process
aggregated from ticks. The broker has its own opinion, and it settles with real
money. `reconcile.py` compares the two — and it **only reads**, which is what
makes it an independent check on the record the bot wrote while trading.

```bash
python reconcile.py                     # read the account, compare, report
python reconcile.py --since-hours 24    # only the last day of signals
python reconcile.py --deals deals.json  # offline, from a saved dump
```

For each signal in the journal it finds the deal on the same market at (about)
the same second, then reports three things:

- **Label agreement** — how often our WIN/LOSS matched their profit sign, with a
  Wilson interval, and a breakdown of the disagreements that names both
  directions ("we said WIN, broker said LOSS").
- **Entry slippage** in pips, signed by what it costs the trade: for a CALL a
  fill *above* our reference is adverse and printed positive, for a PUT a fill
  *below* it is. Our exit is a bar close; yours is the broker's. This is the gap.
- **Actual payouts**, read from the settled deals rather than the asset list, so
  the break-even it prints is the one your account really faces.

**The broker's clock is not ours.** Live on the demo account, five consecutive
deals carried an `open_timestamp` 7197-7198 seconds after the second the signal
named — about two hours, and not a round 7200, so the rule behind it is not
something this project could derive. It does not have to be: the offset is
*measured* from the deals themselves, by finding the value that puts the most of
them on a signal, and printed with the report:

```
  broker clock is +7198s from ours (measured — 5 of 5 timestamped deal(s) land on a signal at this offset)
```

Matching then aligns every deal by it before pairing, so a two-hour disagreement
costs nothing. Measured rather than assumed, because the failure mode of a wrong
guess here is silent: a 20-second tolerance against a two-hour offset does not
produce a wrong answer, it produces an empty one — "no overlap between the two
records", which reads exactly like "your trades were never placed". If no offset
can be measured the report says so and points at `--offset SECONDS`, which sets
it by hand (the broker's timestamp minus yours).

Then the verdict. Because binary options are not a coin flip — at 85% payout you
need **54.1%** just to break even, at 60% you need **62.5%** — the tool refuses
to call a rate merely above 50% a win. It prints the break-even, the Wilson
interval, and how many settled trades the observed edge would need before the
*lower* bound clears it. A 2-point edge over break-even needs well over a
thousand trades; two weeks of signals will not settle the question. That is the
honest answer, and printing it is the point — a tool that flattered a 4-trade
sample would be worse than no tool.

Two exclusions it makes deliberately: refunds (`profit == 0`) are a third
outcome, reported separately and left out of every rate, and a deal with no
matching signal is reported as *traded outside the bot* rather than quietly
counted against it.

### What it needs and what it does not prove

- `JOURNAL_SIGNALS=1` and a `SIGNAL_JOURNAL_PATH` — `stats.json` keeps totals
  only, so without the per-trade journal there is nothing to match against.
- `POCKET_SSID` / `POCKET_UID` for the live read (`--deals FILE` works without
  them, which is how the parsing is tested).
- **Placing the trades.** With `AUTO_TRADE=1` (demo only — see above) the bot
  places them itself and the journal already carries the settlement it received;
  this re-reads the account independently, which is what verifies that record.
  Hand-placing works the same way, since the match is on market and second — but
  a minute where *both* you and the bot traded cannot be separated: the nearest
  deal in time wins, and the report cannot say whose it was.
- **Demo is not live.** `POCKET_IS_DEMO=1` reads demo deals, and a demo fill
  says nothing about a funded account's spread or execution.
- `--dump FILE` saves the raw payload the account returned. That file is the way
  to harden the parsing if the SDK's field names ever change — it lets the
  comparison be re-run offline, so a parsing fix can be checked against real
  data instead of guessed at.

## Configuration

Every variable, its default and its meaning is documented in `.env.example`.
The ones worth knowing first:

| Variable | Default | Meaning |
|----------|---------|---------|
| `FEED` | `simulated` | `simulated` or `pocket_option` |
| `ASSETS` | *(empty)* | empty = derive from the live asset list |
| `ASSET_MODE` | `major` | `major`, `forex` or `all` |
| `OTC_ONLY` | `1` | `_otc` assets trade 24/7 |
| `MIN_PAYOUT` | `60` | skip markets paying less than this % |
| `CANDLE_PERIOD` | `300` | seconds per bar — sets the whole rhythm |
| `EXPIRY` | `5m` | trade duration; match it to `CANDLE_PERIOD` |
| `SIGNAL_LEAD_SECONDS` | `300` | how early the signal is sent; at most one bar |
| `STALE_AFTER` | `420` | drop a market quiet this long; must exceed `CANDLE_PERIOD` |
| `COOLDOWN_SECONDS` | `120` | per-market gap between signals |
| `MARKET_TZ_OFFSET_HOURS` | `3` | the clock every message prints (UTC+3) |
| `MIN_SCORE` | `2` | weighted agreement required |
| `MIN_COMPONENTS` | `2` | directional indicators that must be warm |
| `TREND_FLAT_MIN_SCORE` | `3` | evidence demanded in a flat market |
| `MAX_GAP_BARS` | `5` | feed outage longer than this drops the bars before it |
| `PERSIST_CANDLES` | `1` | keep bars on disk for a warm restart |
| `JOURNAL_SIGNALS` | `1` | write the per-trade record `reconcile.py` reads |
| `SIGNAL_JOURNAL_PATH` | `signals.jsonl` | where that record lives |
| `AUTO_TRADE` | `0` | place the orders too — refused unless `POCKET_IS_DEMO=1` |
| `TRADE_AMOUNT` | `1.0` | stake per trade when `AUTO_TRADE=1` |

## Disclaimer

For educational use only. Binary options trading carries substantial risk.
Signals are recommendations, not guarantees — always test on a demo account
first. The replay tool measures the *engine*, not your fills: spread, payout and
execution are the broker's, not this program's. Automated order placement is
demo-only by construction, and a demo fill says nothing about a funded account's
execution.
