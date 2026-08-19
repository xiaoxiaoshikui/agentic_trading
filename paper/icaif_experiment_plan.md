# ICAIF Experiment Plan

## Goal

Target ICAIF with a finance-first framing: event-aligned multimodal crypto forecasting and cost-aware trading evaluation. The paper should claim robust risk-adjusted trading improvement only where the evidence supports it, and should avoid claiming broad non-financial multimodal generality.

## Reviewer Issue To Experiment Mapping

1. Weak baselines
   - Add price-only sequence baselines: DLinear-lite, TimesNet-lite, PatchTST, iTransformer-lite.
   - Keep conventional multimodal baselines: text/web MLP, price+web late fusion, BiLSTM fusion, early fusion, multimodal transformer, tensor fusion.
   - Compare CGCMA against the strongest price-only baseline, not only against a weak transformer.

2. Zero-cost trading evaluation
   - Use `transaction_cost_bps` in validation threshold selection and in test metrics.
   - Export `cost_bps` and `net_pnl` in trade prediction files.
   - Report fee sensitivity at 0, 5, 10, and 20 bps.

3. Aggregation and reproducibility
   - Fix multi-model aggregation so every row in each `summary.csv` is collected.
   - Report per-run counts, matched seed-fold deltas, per-asset slices, and volatility/return regime slices.

4. Ablation concerns
   - Keep a cost-aware CGCMA ablation config covering no cross-attention, scalar/no gate, no lag/web gate features, stale baselines, and shuffle controls.
   - Run the ablation after the main two-seed baseline table stabilizes.

## Current Executed Runs

- Smoke: `icaif_real_news_costaware_smoke`, seed 42, 1 fold, 10 models.
- Main: `icaif_real_news_costaware_main`, seeds 42 and 123, 5 rolling folds, 12 models, 5 bps cost-aware threshold selection.

## Current Result Reading

- CGCMA has the best mean cost-aware fold Sharpe across the two main seeds.
- TimesNet-lite is the strongest price-only baseline on AUC and is the closest trading competitor.
- CGCMA beats price-only in all 10 matched seed-fold units; the matched delta is positive with bootstrap CI above zero.
- CGCMA vs TimesNet-lite is positive on mean matched delta, but the bootstrap CI crosses zero. The paper should present this as a favorable but not decisive advantage over the strongest time-series baseline.
- Fee sensitivity supports the cost-aware story: CGCMA remains positive at 10 bps in pooled per-run analysis and degrades at 20 bps.

## Next Runs

1. Add seeds 456 and 789 for `icaif_real_news_costaware_main`.
2. Run `icaif_real_news_costaware_ablation` with a reduced 5-fold rolling step before deciding which ablations enter the 8-page paper.
3. Refresh paper tables from:
   - `paper/icaif_real_news_costaware_main_agg.csv`
   - `paper/icaif_real_news_costaware_main_matched_deltas.csv`
   - `paper/icaif_real_news_costaware_main_fee_sensitivity.csv`
   - `paper/icaif_real_news_costaware_main_per_asset.csv`
   - `paper/icaif_real_news_costaware_main_per_vol_regime.csv`
