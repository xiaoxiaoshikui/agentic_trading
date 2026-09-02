<div align="center">

# AgenticTrading

### *Does event-conditioned, freshness-aware multimodal fusion earn a cost-aware trading edge over price-only baselines?*

[![License: MIT](https://img.shields.io/github/license/xiaoxiaoshikui/agentic_trading?color=informational)](LICENSE)
[![Python](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Last commit](https://img.shields.io/github/last-commit/xiaoxiaoshikui/agentic_trading)](https://github.com/xiaoxiaoshikui/agentic_trading/commits/main)

Research + engineering code · BTC / ETH / SOL, 15-minute bars · **CGCMA** — Conditionally-Gated Cross-Modal Attention

</div>

> [!IMPORTANT]
> **Bottom line, stated honestly:** under the current protocol (two seeds, five rolling walk-forward folds, 5 bps/trade), CGCMA has a statistically robust cost-aware Sharpe margin over a price-only Transformer baseline — but its margin over the *strongest* price-only baseline (TimesNet-lite) is positive and not yet statistically separated, and the evaluation window (Dec 2025–Mar 2026) is a short-horizon stress test, not a market-wide claim. See [Key results](#key-results) and [Limitations](#limitations).

This is research and educational software. It is **not financial advice**, and
should not be read as a live-trading system without additional validation,
risk controls, exchange-specific testing, and human oversight.

## Contents

- [Research focus](#research-focus)
- [Pipeline](#pipeline)
- [Key results](#key-results)
- [What is included](#what-is-included)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the trading prototype](#running-the-trading-prototype)
- [Running research experiments](#running-research-experiments)
- [Deployment](#deployment)
- [Method summary](#method-summary)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Research focus

The repository has two complementary tracks:

- **Trading-system prototype**: a Binance USDT-M Futures oriented bot framework with dry-run execution, risk checks, market scanning, multi-agent analysis, and optional LLM advisory modules.
- **Research framework**: reproducible experiment code for asynchronous multimodal market prediction, including the CGCMA model, rolling walk-forward evaluation, baseline comparisons, and frozen result exports.

High-frequency markets produce dense price streams, while external context
such as news and web intelligence arrives sparsely and with variable delay.
This repository studies that setting as an asynchronous multimodal learning
problem:

- 15-minute OHLCV price windows for BTCUSDT, ETHUSDT, and SOLUSDT.
- Lagged text and web-intelligence features aligned causally at decision time.
- Explicit modality lag, `tau_lag`, so freshness is modeled rather than hidden.
- Rolling walk-forward evaluation with downstream trading metrics after transaction costs.

The main proposed architecture is **CGCMA**: Conditionally-Gated Cross-Modal Attention. CGCMA separates two roles:

- **Grounding**: text attends over the price sequence to retrieve event-relevant market states.
- **Trust control**: a learned per-dimension gate decides how much grounded context to inject based on price-text agreement, web features, and modality lag.

<p align="center">
  <img src="docs/figures/cgcma_architecture.png" alt="CGCMA architecture" width="900">
</p>

<p align="center">
  <em>CGCMA grounds event text in the price sequence, then uses a freshness- and agreement-aware gate to control residual multimodal fusion.</em>
</p>

---

## Pipeline

The empirical loop behind every table below — inputs on the left, frozen,
citable output on the right:

```mermaid
flowchart LR
    price["OHLCV 15-min bars<br/>BTC / ETH / SOL"] --> model
    text["News + web-intel embeddings<br/>lagged, causally aligned"] --> model
    lag["tau_lag<br/>modality freshness"] --> model
    model["CGCMA<br/>grounding + trust-gated fusion"] --> wf["Rolling walk-forward<br/>evaluation"]
    wf --> cost["Cost-aware trading metrics<br/>5 bps / trade"]
    cost --> tables[["Frozen result tables<br/>under paper/"]]
```

---

## Key results

The headline results below are from the committed frozen tables under
`paper/`. The primary real-news protocol uses BTC+ETH+SOL pooled data, 27,914
event-conditioned samples, 15-minute bars, a 4-bar forecast horizon, rolling
walk-forward splits, and 5 bps per-trade transaction cost.

### Main cost-aware comparison

Source: `paper/icaif_real_news_costaware_main_agg.csv`

| Model | Inputs | AUC | Trade rate | Hit rate | Net Sharpe | Total return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PriceTx-S | Price | 0.575 | 0.305 | 0.540 | -0.340 +/- 0.091 | -2.24% |
| TimesNet-lite | Price | **0.599** | 0.397 | 0.455 | -0.131 +/- 1.487 | +2.38% |
| PriceWeb | Price + web | 0.561 | 0.247 | 0.432 | -0.187 +/- 0.076 | -1.42% |
| BiLSTM fusion | Price + text + web | 0.568 | 0.301 | 0.436 | -1.021 +/- 0.346 | -6.86% |
| CGCMA | Price + text + web | 0.564 | 0.184 | **0.542** | **+1.047 +/- 0.793** | **+5.21%** |

Interpretation:

- Predictive AUC and downstream trading utility are not the same objective. TimesNet-lite has the best AUC in the main table, but its cost-aware Sharpe is not the best.
- CGCMA does not dominate every classification metric; its advantage appears in downstream selectivity after costs.
- CGCMA trades less frequently than the price-only Transformer and obtains the highest mean net Sharpe in the two-seed main comparison.

### Matched fold Sharpe deltas

Source: `paper/icaif_real_news_costaware_main_matched_deltas.csv`

| Comparison | Mean Sharpe delta | Bootstrap interval | Wins/losses |
| --- | ---: | ---: | ---: |
| CGCMA - PriceTx-S | +1.387 | [0.725, 2.107] | 10/0 |
| CGCMA - DLinear-lite | +1.577 | [0.240, 2.979] | 9/1 |
| CGCMA - PatchTST | +1.827 | [0.070, 3.280] | 9/1 |
| CGCMA - TimesNet-lite | +1.177 | [-0.352, 2.581] | 8/2 |

The strongest supported claim is that CGCMA has a robust cost-aware margin
over the price-only Transformer under the current protocol. The margin over
TimesNet-lite is positive but not statistically separated at the current
sample scale.

### Ensemble check

Source: `paper/icaif_main_late_ensemble_agg.csv`

| Model | Runs | AUC | Net Sharpe | Max drawdown | Total return |
| --- | ---: | ---: | ---: | ---: | ---: |
| CGCMA + TimesNet ensemble | 10 | 0.580 | +0.351 +/- 1.940 | 4.05% | +4.62% |
| CGCMA + PriceWeb ensemble | 10 | 0.564 | +0.334 +/- 1.475 | 4.29% | +2.29% |

The ensemble results are more conservative and more variable than the
two-seed main CGCMA result. They are useful as a robustness check, not as a
replacement for the controlled main comparison.

---

## What is included

| Path | Purpose |
| --- | --- |
| `src/` | Core trading, multi-agent, risk, LLM, market-scanning, and ToM modules. |
| `experiments/` | Research experiment runners, configs, baselines, metrics, and multimodal model code. |
| `paper/` | Frozen tables, result summaries, reproducibility notes, and LaTeX source for the research write-up. |
| `data/README.md` | Placeholder only. Large raw and processed datasets are intentionally not committed. |
| `.env.example` | Safe configuration template with no real credentials. |

Large local artifacts are excluded by `.gitignore`: virtual environments, raw
data, generated multimodal histories, experiment caches, logs, model files,
and private submission workflow artifacts.

---

## Installation

Core runtime:

```bash
git clone https://github.com/xiaoxiaoshikui/agentic_trading.git
cd agentic_trading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additional packages for deep multimodal experiments:

```bash
pip install torch scikit-learn sentence-transformers matplotlib pyarrow
```

## Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Keep these defaults while testing:

```bash
DRY_RUN=true
BINANCE_TESTNET=true
ENABLE_LLM=false
```

Only set real exchange or LLM keys in `.env`. Do not commit `.env`.

---

## Running the trading prototype

Dry-run single-symbol loop:

```bash
python3 -m src.main --symbol BTCUSDT --interval 15m --dry-run
```

Dry-run multi-asset mode:

```bash
python3 -m src.main --multi-asset --symbols BTCUSDT ETHUSDT SOLUSDT --dry-run
```

Historical backtest helpers:

```bash
python3 run_backtest.py
python3 run_multi_asset_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 15m
```

## Running research experiments

The committed result tables in `paper/` are frozen exports. Re-running the
full multimodal experiments requires local price/news artifacts that are not
committed to the public repository.

Example command for the main cost-aware deep experiment:

```bash
python3 -m experiments.run_mm_deep_experiment \
  --config experiments/configs/icaif_real_news_costaware_main.json \
  --device auto
```

Example ToM baseline tests:

```bash
python3 -m unittest \
  test_tom_horizon_features.py \
  test_tom_calibration.py \
  test_tom_multi_agent_signal.py
```

For artifact details and current reproducibility gaps, see `paper/reproducibility.md`.

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the Vast.ai GPU workflow, SSH startup
repair script, and Hugging Face artifact upload path.

---

## Method summary

CGCMA operates on:

- `P`: a 64-bar OHLCV lookback window.
- `T`: a frozen sentence-transformer text embedding of aligned web/news context.
- `W`: scalar web-intelligence features.
- `tau_lag`: the elapsed time between the latest usable web context and the decision bar.

The model computes a price representation, a text representation, and
text-conditioned attention over the price sequence. A conditional gate
receives the price state, grounded context, their difference, web features,
and freshness. The gated residual update lets the model fall back toward
price-only behavior when external context is stale or contradictory.

---

## Limitations

- The main real-news result covers December 2025 to March 2026 and should be read as a short-horizon stress test, not a market-wide claim.
- The primary main table uses two random seeds and five rolling test blocks per seed; confidence intervals remain wide.
- Downstream Sharpe is an offline utility proxy. It omits order-book depth, latency, funding, liquidation constraints, slippage beyond the configured cost, and adaptive execution.
- Raw market/news datasets are not redistributed in this repository. Reproducing the full pipeline requires rebuilding or supplying local data artifacts.
- Any real-money deployment requires independent validation, exchange testnet runs, monitoring, compliance review, and manual fail-safes.

---

## Citation

No paper is public yet — if the CGCMA architecture, the trading-prototype
code, or the frozen result tables are useful, citing the repository is
enough:

```bibtex
@misc{guo2026agentictrading,
  title  = {AgenticTrading: event-conditioned, freshness-aware multimodal fusion for crypto market prediction},
  author = {Guo, Yunxiang},
  year   = {2026},
  url    = {https://github.com/xiaoxiaoshikui/agentic_trading}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
