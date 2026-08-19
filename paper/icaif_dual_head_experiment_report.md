# ICAIF Dual-Head Position Experiment

Run: `experiments/results_mm/icaif_dual_head_main_20260805_225326_346260_seed42`

Artifacts:

- Aggregate summary: `paper/icaif_dual_head_main_agg.csv`
- Full JSON: `paper/icaif_dual_head_main_agg.json`
- Matched fold deltas: `paper/icaif_dual_head_main_matched_deltas.csv`
- Position diagnostics: `paper/icaif_dual_head_position_diagnostics.csv`

## What Changed

This experiment replaces threshold-derived trading with a true direct-position path:

- Direction head: trained with BCE/return-weighted BCE and evaluated by AUC/F1/Brier.
- Position head: emits continuous positions through `tanh(position_logit / T)`.
- Evaluation: downstream PnL uses the emitted position directly, with small positions optionally treated as abstain.
- Additional variants: utility-only, raw continuous position, supervised long/short/abstain target, and target+utility.

## Main Result

The dual-head variants did not beat the strongest current anchor on 5 rolling folds.

| Model | AUC | Trade Rate | Hit Rate | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| `timesnet_lite_anchor` | 0.6083 | 0.3790 | 0.5650 | 0.6137 | 0.0325 |
| `price_only_anchor` | 0.5743 | 0.3464 | 0.4929 | -0.2758 | -0.0168 |
| `timesnet_cgcma_dual_target_mean` | 0.6017 | 0.8418 | 0.4919 | -0.3186 | -0.0352 |
| `cgcma_dual_sharpe` | 0.5450 | 0.3570 | 0.2619 | -0.3682 | -0.0015 |
| `timesnet_cgcma_dual_mean` | 0.5788 | 0.0911 | 0.0865 | -0.4487 | -0.0338 |
| `cgcma_anchor` | 0.5655 | 0.4084 | 0.4725 | -1.3673 | -0.0824 |

Against `timesnet_lite_anchor`, the best dual model
`timesnet_cgcma_dual_target_mean` has mean matched fold Sharpe delta `-0.9323`
and wins only `1/5` folds.

## Diagnostic

The direct-position head improves the experimental design because the trading
claim is now evaluated on emitted actions rather than probability thresholds.
However, the observed failure mode is not just thresholding:

- Raw continuous variants trade every test window and lose after cost.
- Target-supervised variants trade heavily, but hit rate stays below 50%.
- Sharpe-loss variants reduce activity, but they do not find enough positive edge.
- `timesnet_cgcma_dual_target_mean` keeps strong AUC, but its trade rate is too high
  and its hit rate is not enough to cover 5 bps cost.

## Interpretation

This run does not support upgrading the paper claim to SOTA with the dual-head
architecture. It does support a more careful claim: direct-position learning is a
better-aligned evaluation protocol, but under the current real-news dataset the
additional multimodal trading head does not dominate a strong TimesNet price-only
baseline.

The likely bottleneck remains data quality and alignment rather than architecture
capacity. This is consistent with the previous data diagnostic: duplicated
windows, conflicting news-direction labels, and weak news-return association make
it hard for a multimodal position head to learn a stable after-cost edge.
