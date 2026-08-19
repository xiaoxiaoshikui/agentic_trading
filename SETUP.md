# SETUP – How to run AgenticTrading

## 1. Python environment

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python version: 3.9+.

## 2. Environment variables

Create a file named `.env` in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

- `BINANCE_API_KEY` – your sub-account API key
- `BINANCE_API_SECRET` – your sub-account secret
- `BINANCE_TESTNET` – `false` for real trading, `true` to use testnet (if enabled)
- `DRY_RUN` – `true` = no orders, only logs; `false` = real orders

## 3. Binance settings checklist

You already did most of this, but for completeness:

- ✅ Sub-account created (virtual email is fine)
- ✅ USDT-M Futures enabled on sub-account
- ✅ API key created for that sub-account
- ✅ API permissions:
  - ✔ Enable Reading
  - ✔ Enable Futures
  - ✘ Withdrawals OFF
- ✅ IP restriction set to your bot server / laptop IP
- ✅ Some USDT in USDⓈ-M Futures wallet

## 4. Running the bot

Example (dry run):

```bash
python -m src.main --symbol BTCUSDT --interval 15m --dry-run
```

Other examples:

```bash
# ETHUSDT, 5 minute bars
python -m src.main --symbol ETHUSDT --interval 5m --dry-run

# Real orders (ONLY if you understand the risk)
python -m src.main --symbol BTCUSDT --interval 15m
```

## 5. Where to plug in LLM / RL

- **`strategy.py`** – implement your own `generate_signal()` using LLM output.
- **`agent.py`** – the `think()` step can call an LLM (e.g. DeepSeek, GPT) and override the signal.

You can log state → ask LLM for interpretation → convert back to a trade plan.

See `STRATEGY.md` for ideas.