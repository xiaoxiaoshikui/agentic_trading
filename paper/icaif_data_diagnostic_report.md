# ICAIF Data Diagnostic Report

Date: 2026-08-05

## Summary

The current real-news dataset is a likely contributor to unstable model rankings, but it is not the only issue. The most urgent experimental issue is that model results change substantially when the same model is placed in a different config order under the same nominal seed. This indicates random-state/order sensitivity and makes architecture comparisons unreliable until per-model/per-fold deterministic seeding is added.

## Key Data Findings

- Row count: 27,914 event-aligned rows.
- Unique `symbol, window_end` trading windows: 13,464.
- Duplicate event rows beyond unique windows: 14,450.
- Mean rows per window: 2.07.
- 50.8% of unique windows have multiple event rows.
- 23.7% of unique windows contain conflicting positive and negative `news_direction_score` values.
- Maximum event rows attached to a single trading window: 14.

This matters because training uses event rows, but downstream evaluation groups predictions back to unique trading windows. News-dense windows therefore receive more training weight than quiet windows, even though they count once during trading evaluation.

## Weak Direct News Signal

On unique trading windows:

- `news_direction_score` vs future return Spearman correlation: -0.0184.
- `web_context_strength` vs future return Spearman correlation: -0.0215.
- `rsi_14` vs future return Spearman correlation: -0.0394.

Single-feature rolling strategies:

- `news_direction_score`: mean Sharpe -1.48.
- `web_context_strength`: mean Sharpe -1.25.
- `ret_4`: mean Sharpe 0.04.
- `ret_16`: mean Sharpe -0.29.
- `rsi_14`: mean Sharpe 1.11, but with high fold variance.

The raw news direction feature is not a strong standalone predictive signal in this dataset.

## Market Drift and Fold Bias

The five test folds have positive mean returns at the unique-window level:

| fold | positive rate | mean return |
| --- | ---: | ---: |
| 0 | 0.536 | 0.000112 |
| 1 | 0.579 | 0.001191 |
| 2 | 0.532 | 0.000615 |
| 3 | 0.584 | 0.000605 |
| 4 | 0.506 | 0.000647 |

Always-long diagnostics:

- Mean 5 bps always-long Sharpe: 0.35.
- Mean zero-cost always-long Sharpe: 2.17.
- Fold 1 always-long 5 bps Sharpe: 2.53.

This means part of the measured trading performance can be driven by regime drift rather than model-specific multimodal intelligence.

## Reproducibility Warning

Same nominal seed and same CGCMA hyperparameters produced very different fold Sharpe sequences depending on config/model order:

- Original main config, seed 42: `[0.000, 2.247, 0.368, 4.228, 1.193]`.
- Targeted/utility config, seed 42: `[-2.752, -1.005, 0.521, 1.133, 2.002]`.

This is not caused by dataset changes. It indicates that global random state is consumed by earlier models in the config. Each model/fold should be reseeded deterministically from `(base_seed, model_name, fold_id)`.

## Recommended Fixes

1. Add deterministic per-model/per-fold seeding before every model initialization and DataLoader creation.
2. Train on unique `symbol, window_end` samples or weight event rows by `1 / n_events_in_window`.
3. Aggregate multiple event texts per trading window before embedding, instead of treating each event as a separate supervised row.
4. Add a deduplicated-data experiment with the same baselines.
5. Report always-long and no-trade baselines under transaction costs.
6. Add regime-balanced or longer-period tests, because the current window is short and mildly upward biased.
7. Consider actionability labels such as `future_return > cost + volatility_band` rather than plain `future_return > 0`.

