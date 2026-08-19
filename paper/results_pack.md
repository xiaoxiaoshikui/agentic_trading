# Results pack (ACM MM target)

This file now records the current frozen multimodal snapshot as of `2026-03-08`.
Use it to keep claims conservative and synchronized with the latest pooled multi-asset run.

## 1 Predictive main table
Target export: `tables/predictive_main.csv`

Current frozen rows from `experiments/results_mm/mm_deep_multi_asset_event_rolling_20260308_122730/summary.csv`:

| model | modalities | macro_f1 | auc | ece | brier | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `price_sequence_transformer` | price sequence | 0.4276 | 0.5671 | 0.1077 | 0.2520 | current strongest AUC baseline |
| `text_embedding_mlp` | text embedding | 0.4797 | 0.4926 | 0.1639 | 0.2770 | current strongest macro-F1 baseline |
| `cross_modal_attention` | price + text + web | 0.4322 | 0.5136 | 0.1470 | 0.2700 | current multimodal fusion result |

Frozen conclusion:
- the current multimodal fusion model does not beat the price-only sequence baseline on AUC
- the text-only branch helps macro-F1 more than calibrated decision quality
- no ACM MM headline claim should rely on this table yet

## 2 Modality ablation table
Target export: `tables/modality_ablation.csv`

Columns:
- `variant`
- `price`
- `news`
- `social`
- `event`
- `chart`
- `macro_f1`
- `auc`
- `ece`

Minimum rows:
- full model
- minus news
- minus social
- minus event
- minus chart
- price only

## 3 Participant reasoning ablation table
Target export: `tables/reasoning_ablation.csv`

Columns:
- `variant`
- `participant_reasoning`
- `macro_f1`
- `ece`
- `downstream_sharpe`
- `downstream_mdd`

Minimum rows:
- no reasoning head
- latent participant head
- participant head plus downstream strategic fusion

## 4 Robustness table
Target export: `tables/robustness.csv`

Columns:
- `scenario`
- `model`
- `macro_f1`
- `delta_vs_clean`
- `downstream_sharpe`
- `notes`

Required scenarios:
- missing news
- missing social
- stale news
- stale social
- asset transfer
- longer horizon

## 5 Downstream trading table
Target export: `tables/downstream_trading.csv`

Current frozen rows from `experiments/results_mm/mm_deep_multi_asset_event_rolling_20260308_122730/summary.csv`:

| policy | signal_source | sharpe | max_drawdown | total_return | trade_rate | win_rate | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| simple threshold | `price_sequence_transformer` | 0.0016 | 0.0301 | 0.0031 | 0.4210 | 0.4510 | weakly positive but near zero |
| simple threshold | `text_embedding_mlp` | -0.3001 | 0.0374 | -0.0129 | 0.4275 | 0.4981 | best macro-F1, worst trading |
| simple threshold | `cross_modal_attention` | -0.1272 | 0.0289 | -0.0005 | 0.4323 | 0.4500 | current multimodal result is still negative |

Current interpretation:
- pooled multi-asset downstream evidence does not support a multimodal trading gain
- thresholded predictive outputs remain too weak for a strong downstream paper claim
- significance testing remains missing and should be added before any submission decision

## 6 Dataset statistics table
Target export: `tables/dataset_stats.csv`

Frozen dataset snapshot from `experiments/results_mm/mm_deep_multi_asset_event_rolling_20260308_122730/results.json` and the finalized target files under `data/multimodal_history/`:

| asset | target_timestamps | aligned_samples | price_bars | notes |
| --- | ---: | ---: | ---: | --- |
| `BTCUSDT` | 1093 | 1708 | 12000 | includes older local history in pooled run |
| `ETHUSDT` | 1093 | 1092 | 12000 | three-month target file complete |
| `SOLUSDT` | 1093 | 1092 | 12000 | three-month target file complete |
| pooled | 3279 | 3892 | 36000 | event-aligned benchmark used in current frozen run |

Additional pooled statistics:
- `mean_modality_age_minutes = 1.1877`
- `has_news = 0.9049`
- `has_fear_greed = 1.0000`
- `has_social = 1.0000`
- `has_whale = 1.0000`
- `text_has_content = 1.0000`
- `mean_text_char_len = 539.7`
- `mean_text_source_count = 16.54`

## 7 Figures to generate
Store under `paper/figures/`.

Required:
1. `architecture.png`
2. `timeline_alignment.png`
3. `main_predictive_bar.png`
4. `modality_ablation.png`
5. `robustness_missing_modalities.png`
6. `downstream_equity_curves.png`

Optional:
7. `attention_or_saliency_case.png`
8. `calibration_reliability.png`

## 8 Claim-evidence checklist
For every claim in the paper, record the exact evidence row or figure.

Template:
- Claim: `The repo now contains a reproducible pooled BTC/ETH/SOL multimodal benchmark pipeline.`
  - Evidence: `results.json`, dataset statistics table, finalized `data/multimodal_history/*_20251001_20251231.jsonl`
  - Rows or panel: pooled dataset snapshot above
  - Strength: `strong`
  - Risk note: benchmark scope is still limited to three crypto assets and one quarter

- Claim: `Cross-modal attention improves over the price-only sequence baseline.`
  - Evidence: predictive main table and downstream trading table
  - Rows or panel: `price_sequence_transformer` vs `cross_modal_attention`
  - Strength: `weak`
  - Risk note: current evidence points the other way; do not make this claim in the paper

- Claim: `Text features contain information that differs from price-only signals.`
  - Evidence: predictive main table
  - Rows or panel: `text_embedding_mlp` macro-F1 vs `price_sequence_transformer`
  - Strength: `moderate`
  - Risk note: the effect does not yet translate into better AUC or downstream Sharpe

- Claim: `Scaling from BTC-only pilots to pooled multi-asset rolling evaluation changes the headline conclusion.`
  - Evidence: current frozen pooled tables plus earlier pilot logs in `CLAUDE.md`
  - Rows or panel: pooled tables and experiment-log comparison
  - Strength: `moderate`
  - Risk note: comparison is historical, not a fully controlled ablation

## 9 What the current repo already provides
Usable as baseline evidence only:
- `experiments/results_tom/reviewer_suite_core_*.csv`
- `experiments/results_tom/reviewer_suite_core_split_*.csv`
- `experiments/results_tom/reviewer_suite_extended_*.csv`

Do not use these files as the primary ACM MM table set. They support the price-only downstream baseline and regression checks.
