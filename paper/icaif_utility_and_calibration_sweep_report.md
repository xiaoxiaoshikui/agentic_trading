# ICAIF Utility and Calibration Sweep Report

Date: 2026-08-05

## Goal

Test whether cost-aware training objectives, late ensembling, or validation-calibrated threshold policies can improve the current CGCMA framework under the real-news, event-conditioned, 5 bps protocol.

## Implemented Training Changes

- Return-weighted BCE: upweights samples by absolute future return.
- Soft-position mean-utility loss: treats `tanh(logit / temperature)` as a differentiable position and maximizes average net PnL after cost.
- Soft-position Sharpe loss: maximizes mini-batch mean/std net PnL after cost.
- These options are controlled in `DeepModelSpec` and are off by default.

## Utility Sweep Result

Run: `icaif_utility_sweep_main`, seed 42, 5 rolling folds, 12 epochs.

| model | AUC | trade rate | hit rate | Sharpe | total return |
| --- | ---: | ---: | ---: | ---: | ---: |
| `timesnet_lite_anchor` | 0.6083 | 0.3790 | 0.5650 | 0.6137 | 0.0325 |
| `cgcma_anchor` | 0.5782 | 0.3885 | 0.5078 | -0.0197 | 0.0004 |
| `cgcma_utility_mean` | 0.5618 | 0.2690 | 0.5466 | -0.0335 | -0.0272 |
| `price_only_anchor` | 0.5743 | 0.3464 | 0.4929 | -0.2758 | -0.0168 |
| `cgcma_return_weighted` | 0.5687 | 0.1403 | 0.3584 | -0.4463 | -0.0231 |
| `timesnet_cgcma_utility_mean` | 0.5871 | 0.3255 | 0.3787 | -0.4786 | -0.0566 |
| `timesnet_cgcma_utility_sharpe` | 0.5595 | 0.2977 | 0.3645 | -0.9683 | -0.0840 |
| `cgcma_utility_sharpe` | 0.5587 | 0.3211 | 0.4704 | -1.0165 | -0.0810 |

Decision: utility-loss training did not improve CGCMA. The least harmful variant is `cgcma_utility_mean`, but it does not beat `cgcma_anchor` or `timesnet_lite_anchor`.

## Late Ensemble Result

Post-hoc validation-selected ensemble over existing main-run predictions, two seeds / ten matched folds.

| model | AUC | trade rate | hit rate | Sharpe | total return | alpha on CGCMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ens_cgcma_timesnet` | 0.5802 | 0.2263 | 0.4761 | 0.3514 | 0.0462 | 0.435 |
| `ens_cgcma_priceweb` | 0.5640 | 0.2140 | 0.4533 | 0.3344 | 0.0229 | 0.360 |
| `ens_cgcma_early` | 0.5615 | 0.2008 | 0.4462 | 0.3311 | 0.0160 | 0.470 |
| `ens_cgcma_price` | 0.5715 | 0.2062 | 0.5193 | 0.1800 | 0.0141 | 0.375 |
| `ens_cgcma_textweb` | 0.5464 | 0.1646 | 0.4572 | -0.0409 | -0.0157 | 0.635 |

Decision: late ensembling helps compared with many failed architecture variants, but it still does not beat the existing main CGCMA result.

## Threshold Policy Sweep

Post-hoc fine threshold search over existing main-run predictions, validation only.

Best calibrated policy:

| model | policy | trade rate | hit rate | Sharpe | total return |
| --- | --- | ---: | ---: | ---: | ---: |
| `cgcma` | `fine_sharpe_mdd_min02` | 0.1095 | 0.5339 | 0.4745 | 0.0191 |

Decision: finer threshold search did not beat the existing CGCMA main result. It reduces trading activity but does not recover the original Sharpe advantage.

## Overall Decision

None of these modifications should replace the current paper's CGCMA main result. The most useful practical direction is the late `CGCMA + TimesNet` ensemble, but it should be treated as an exploratory backup rather than the new central architecture.

The next architecture attempt should not be another shallow loss tweak. It should add an explicit trade/no-trade head and evaluate positions directly, because forcing one logit to serve both classification and trading utility did not improve performance.

