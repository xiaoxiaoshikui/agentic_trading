# AgenticTrading – Binance USDT-M Futures Bot (Skeleton)

This repo is a **minimal but structured framework** for an "agentic" trading system:

- Exchange: **Binance USDT-M Futures**
- Account: runs on a **sub-account** with its own API key
- Capital: small size (e.g. 20–200 USDT)
- Style: simple trend-following baseline, pluggable with LLM / RL logic

The code is designed to be:

- **Safe by default** – starts in `DRY_RUN` mode (no real orders)
- **Modular** – clear separation between:
  - `binance_client.py` – exchange access
  - `strategy.py` – signal generation
  - `risk.py` – position sizing + risk checks
  - `agent.py` – "agent loop" (observe → think → act)
  - `main.py` – CLI entry point

> ⚠️ Disclaimer: This is **educational code**, not financial advice.  
> Real trading involves risk. Always start with `DRY_RUN = True` and very small size.

## Open-source notes

This repository is prepared to be published without local secrets or heavy generated artifacts:

- Commit `.env.example`, not `.env`.
- The default `.gitignore` excludes virtual environments, logs, generated datasets, experiment caches, and large model/result artifacts.
- Paper reviews, rebuttals, submission packages, and draft PDFs are ignored by default.
- See `OPEN_SOURCE_CHECKLIST.md` before the first GitHub push.

## Quick start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in:
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`

3. Run in dry-run mode:

```bash
python -m src.main --symbol BTCUSDT --interval 15m --dry-run
```

4. Once you are sure everything works and risk is acceptable, you may disable dry-run:

```bash
python -m src.main --symbol BTCUSDT --interval 15m
```

(But only after you fully understand what the strategy does.)

---

See `SETUP.md` for details on environment & API,  
see `STRATEGY.md` for how the baseline strategy works and how to plug in smarter logic (LLM, RL, etc.).
