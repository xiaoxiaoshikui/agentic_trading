# ECML PKDD Paper Blueprint (for `agentic_trading`)

## 0. Venue constraints (must satisfy first)
- Track target: `ECML PKDD 2026 Research Track` (double-blind, LNCS format).
- As of today (`2026-03-06`), research-track deadlines are:
  - Abstract deadline: `2026-03-05` (passed)
  - Paper deadline: `2026-03-12` (AoE)
- Length: max `16 pages including references`; appendix must be separate and not required for review.
- Proceedings: Springer LNCS.

Use this style target:
- Problem-gap-method-result narrative in abstract/introduction.
- Explicit contributions list in intro.
- Formal problem formulation before method details.
- Main table + ablation + robustness + efficiency/cost analysis.

---

## 1. Recommended paper story for this project

### Preferred framing (realistic with current evidence)
`Robustness-first empirical study of a Theory-of-Mind trading framework under multi-setting walk-forward evaluation`

Why this framing:
- Current results are setting-dependent.
- Under larger-sample checks, gains are not consistently significant.
- A top-tier "state-of-the-art" claim is hard to defend now.

This framing is still publishable if:
- You provide rigorous protocol and transparency.
- You clearly report where it works / where it fails.
- You provide actionable analysis and design insights.

---

## 2. Title / Abstract templates

## 2.1 Title candidates
1. `A Robustness-Centric Evaluation of Theory-of-Mind Trading Agents under Walk-Forward Regimes`
2. `When Does Theory-of-Mind Help in Algorithmic Trading? A Multi-Setting Empirical Study`
3. `Theory-of-Mind for Trading: Gains, Failures, and Stability under Distribution Shift`

## 2.2 Abstract template (150-200 words)
Paragraph 1: problem + gap  
`Most ToM-style decision systems in trading are evaluated in narrow settings, leaving cross-regime stability underexplored.`

Paragraph 2: method  
`We implement a modular ToM framework combining technical signal experts, opponent modeling, and strategic fusion with train-period calibration.`

Paragraph 3: protocol  
`We use walk-forward evaluation across multiple intervals (1h/4h), conservative/aggressive risk profiles, and robustness settings (asset expansion, opponent ablations), with paired statistical testing.`

Paragraph 4: key outcomes (must be numeric)  
`In core settings, ToM outperforms Technical in X/Y cases with mean delta Z; however, larger-sample checks reveal instability and mostly non-significant differences.`

Paragraph 5: takeaway  
`Results indicate ToM can help in specific regimes but requires configuration-aware control for robust gains.`

---

## 3. Section-by-section writing plan (ECML accepted-paper style)

## 3.1 Introduction (1.5-2 pages)
Include:
1. Problem importance (non-stationarity, regime shift, multi-asset decisions).
2. Gap in prior work (single-setting evaluation; weak robustness evidence).
3. Your method intuition (ToM = model other participants + strategic fusion).
4. Contributions (numbered, explicit, testable):
   - C1: modular ToM framework for trading.
   - C2: reviewer-oriented evaluation protocol (stability/ablation/robustness).
   - C3: empirical findings on when ToM helps and where it fails.
5. Paper roadmap.

Use "accepted-paper style":
- End intro with a compact bullet contribution block.

## 3.2 Related Work (1 page)
Subsections:
1. ToM and multi-agent reasoning in decision systems.
2. Algorithmic trading with ML/RL.
3. Robust evaluation in non-stationary financial ML.
4. Positioning paragraph: exactly what is new here.

## 3.3 Problem Formulation (0.8-1.2 pages)
Define:
- Market state `s_t`, action `a_t in {LONG, SHORT, FLAT}`, reward / portfolio objective.
- Opponent latent behavior estimate.
- Walk-forward train/test protocol and no-lookahead constraints.

This section should look mathematical and clean, before implementation details.

## 3.4 Method (2-3 pages)
Structure:
1. `Technical Expert Layer` (trend/mean-reversion/breakout/dynamic).
2. `Opponent Modeling Layer` (retail/momentum, calibration on train).
3. `Strategic Fusion Layer` (influence/gate/override/decision threshold).
4. `Policy Calibration` (objective used in train split).
5. Complexity / implementation notes.

Add one architecture figure early in this section.

## 3.5 Experimental Protocol (1-1.5 pages)
Must be very explicit:
- Data sources, symbols, intervals, horizon, train/test split.
- Seeds, periods, and walk-forward setup.
- Baselines and fairness controls (same risk constraints and budgets).
- Metrics: Sharpe, PnL, drawdown, win rate, profit factor, trades.
- Statistics: paired tests, effect size, confidence intervals, multiple-test correction.

## 3.6 Results (2-3 pages)
Order:
1. Main performance table (core settings).
2. Stability summary (win/loss count and mean delta).
3. Ablation table (opponent components, depth, negative influence toggle).
4. Robustness table (more assets, cost/slippage sensitivity).
5. Error/failure analysis (why aggressive 1h can underperform).

Do not hide negative cases; explain them.

## 3.7 Discussion (0.8-1 page)
Include:
- Where ToM adds value (regime and config conditions).
- Why instability appears under higher-power evaluation.
- Practical recommendations (config split, conservative defaults).

## 3.8 Limitations & Ethics (0.5 page)
Mention:
- Market drift and data-snooping risk.
- Backtest vs live execution gap.
- Financial use risks and responsible deployment.

## 3.9 Conclusion (0.4-0.6 page)
One-paragraph summary + one-paragraph next steps.

---

## 4. Tables and figures checklist (minimum viable)
- Table 1: Main comparison (Technical vs ToM-Full) across core settings.
- Table 2: Stability summary (wins/losses, mean delta, CI).
- Table 3: Ablations (opponents/depth/fusion options).
- Table 4: Robustness (more assets, slippage/fee variants).
- Figure 1: Framework architecture.
- Figure 2: Per-setting delta bar chart.
- Figure 3: Sensitivity plot (e.g., influence or slippage sweep).

---

## 5. Claim-evidence matrix (write this before final drafting)
- Claim A: `ToM improves performance in multiple core settings.`
  - Evidence: core table + paired stats.
- Claim B: `Improvements are setting-dependent.`
  - Evidence: negative deltas in aggressive/conservative edge cases.
- Claim C: `Calibration/fusion design choices affect stability.`
  - Evidence: ablation and split-profile experiments.

If any claim has weak evidence, weaken wording before submission.

---

## 6. Recommended writing sequence (fast path)
1. Freeze experiment IDs and tables first.
2. Draft Methods and Experimental Protocol.
3. Draft Results from fixed tables only.
4. Write Introduction/Abstract last.
5. Final pass: anonymity + page budget + reproducibility statement.

---

## 7. Reproducibility paragraph template
`We release configuration files, run scripts, and evaluation logs. All experiments use walk-forward splits with fixed random seeds and no-lookahead execution. We report per-setting performance and paired statistical tests, including effect sizes and confidence intervals.`

---

## 8. What to avoid (common reject triggers)
- Overclaiming SOTA while most comparisons are non-significant.
- Hiding settings where your method underperforms.
- Mixing in-sample and out-of-sample evidence.
- Missing fairness controls for baselines.
- No explanation for instability under larger sample checks.

---

## 9. Current project-specific recommendation
- Use `core_split` as base (aggressive-only special behavior), not global one-size-fits-all tuning.
- Add one more robust pass with transaction-cost/slippage sweeps.
- If significance remains weak, frame contribution as `robust empirical characterization` rather than `new SOTA algorithm`.
