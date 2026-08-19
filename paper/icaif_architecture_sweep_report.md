# ICAIF Architecture Sweep Report

Date: 2026-08-05

## Purpose

Test whether modified CGCMA-style architectures improve the current ICAIF claim under the same real-news, event-conditioned, 5 bps cost-aware trading protocol.

## Implemented Variants

- `timesnet_gated_cross_modal`: TimesNet-lite price encoder with CGCMA-style text-to-price attention and zero-initialized gated residual trading logit.
- `timesnet_moe_fusion`: TimesNet-lite price encoder with a mixture over price, text, web, and interaction experts.
- `val_sharpe` checkpoint selection: optional checkpoint selection by validation-set cost-aware Sharpe instead of AUC.
- Sweep configs:
  - `experiments/configs/icaif_arch_sweep_smoke.json`
  - `experiments/configs/icaif_arch_sweep_main.json`
  - `experiments/configs/icaif_arch_targeted_main.json`

## Runs

- Smoke: `icaif_arch_sweep_smoke`, seed 42, 1 fold, 2 epochs.
- Broad sweep: `icaif_arch_sweep_main`, seed 42, 5 folds, 8 epochs.
- Targeted sweep: `icaif_arch_targeted_main`, seeds 42 and 123, 5 folds, 12 epochs.

## Targeted Two-Seed Result

| model | AUC | Brier | Trade rate | Hit rate | Sharpe | Sharpe std | Total return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `timesnet_lite_auc_12` | 0.6026 | 0.2541 | 0.2781 | 0.4693 | 0.0579 | 0.7861 | 0.0061 |
| `timesnet_cgcma_val_sharpe_12` | 0.5824 | 0.2495 | 0.2326 | 0.4985 | -0.0406 | 0.2465 | -0.0198 |
| `cgcma_auc_12` | 0.5639 | 0.2572 | 0.3530 | 0.5052 | -0.1217 | 0.1442 | -0.0167 |
| `learned_gate_val_sharpe_12` | 0.5602 | 0.2539 | 0.2288 | 0.4394 | -0.3018 | 0.5956 | -0.0264 |
| `price_only_anchor_12` | 0.5755 | 0.2524 | 0.3055 | 0.5402 | -0.3404 | 0.0913 | -0.0224 |
| `cgcma_val_sharpe_12` | 0.5516 | 0.2634 | 0.2832 | 0.4345 | -0.5687 | 0.2567 | -0.0787 |
| `timesnet_moe_val_sharpe_12` | 0.5654 | 0.2541 | 0.3655 | 0.4242 | -1.0093 | 0.3973 | -0.0979 |

## Matched Fold Signal

Target `timesnet_cgcma_val_sharpe_12`:

- vs `timesnet_lite_auc_12`: mean delta -0.0985, wins/losses 5/5, Wilcoxon p=0.423.
- vs `cgcma_auc_12`: mean delta +0.0811, wins/losses 6/4, Wilcoxon p=0.500.
- vs `price_only_anchor_12`: mean delta +0.2997, wins/losses 6/4, Wilcoxon p=0.461.

## Decision

Do not replace the paper's current CGCMA main result with these modified architectures. The best targeted two-seed model in this sweep is the price-only `timesnet_lite_auc_12`, while the strongest modified multimodal candidate, `timesnet_cgcma_val_sharpe_12`, is not statistically separated from TimesNet-lite or the CGCMA baseline.

The useful architectural lesson is narrower: a stronger price encoder improves AUC, but adding multimodal residuals does not yet improve cost-aware Sharpe. Future architecture work should focus on a true utility-trained trade/no-trade head rather than simply attaching text gates to TimesNet.

