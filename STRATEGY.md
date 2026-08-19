# STRATEGY – Baseline Trend Strategy + Agentic Extensions

## Baseline idea

We start with a **simple trend-following rule**:

- Use EMA(50) and EMA(200) on close prices
- If EMA50 > EMA200 → uptrend → look for long
- If EMA50 < EMA200 → downtrend → look for short
- Optional ATR-based volatility filter:
  - If ATR too low → skip (market is dead)
  - If ATR too high → reduce position size

Signal states:

- `LONG` 
- `SHORT` 
- `FLAT` 

These signals are produced in `strategy.py`.

## Risk model (in `risk.py`)

- Fixed fraction risk per trade (e.g. 1% of capital)
- Stop loss based on ATR multiple (e.g. 2 * ATR)
- Take profit based on RR ratio (e.g. 2:1)

Risk module outputs:

- `qty` (position size in contracts)
- `stop_price` 
- `take_profit_price` 

## Agent loop

`agent.py` implements a simple "agentic" loop:

1. `observe()` – read market + account state
2. `think()` – generate a plan:
   - baseline: just use strategy signal + risk model
   - advanced: combine with LLM output
3. `act()` – place/cancel orders according to the plan

## Where to plug in LLM logic

Inside `agent.py`, in the `think()` method, there is a hook:

```python
plan = {
    "target_side": signal.side,
    "reason": signal.reason,
}
# TODO: integrate LLM here to refine / override plan
```

You can:

1. Log recent candles, funding rate, news summary, etc.
2. Ask LLM: "Given this context, should we go long/short/flat?"
3. Map LLM response back to `target_side`.

Keep baseline rules as a safety net, e.g.:

- If LLM says something crazy, fall back to baseline signal.
- Never exceed max position size from `risk.py`.
