# UAI Paper Outline

**Title:** LLM-Guided Bayesian Optimization in Strategy Space: Sample-Efficient Policy Search via Language Model Proposals

**Target:** UAI 2026 (Deadline: ~February 2026)

---

## Abstract (150 words)

We study whether large language models can serve as effective proposal distributions for Bayesian optimization in structured policy spaces. Using algorithmic trading as a testbed, we formulate strategy optimization as search over a space of executable programs. Unlike traditional Bayesian optimization with Gaussian process surrogates, we leverage LLMs to propose candidate strategies based on natural language descriptions of performance feedback. Our experiments across 5 cryptocurrency assets over 20 walk-forward periods demonstrate that LLM-guided search achieves [X]% higher Sharpe ratio than random search with [Y]x better sample efficiency. We analyze the calibration of LLM confidence scores, finding expected calibration error of [Z], and characterize the exploration-exploitation trade-off implicit in LLM proposals. Our results suggest that LLMs' ability to reason about code structure enables more efficient navigation of combinatorial strategy spaces compared to structure-agnostic baselines.

---

## 1. Introduction (1.5 pages)

### Opening
- Bayesian optimization is effective for black-box optimization
- Challenge: Structured search spaces (code, programs, strategies)
- Traditional BO uses GP/neural surrogates - limited to continuous spaces
- LLMs can read, write, and reason about code

### Research Question
> Can LLMs serve as effective proposal distributions for optimization in structured program spaces?

### Contributions
1. **Formulation**: Frame strategy optimization as BO with LLM proposals
2. **Analysis**: Characterize LLM calibration, regret, and sample efficiency
3. **Empirical study**: Systematic comparison on trading benchmark
4. **Insights**: When/why LLM proposals outperform random search

### Key Finding (Preview)
> LLM-guided search achieves comparable optimization quality to random search in [Y]x fewer iterations, with well-calibrated confidence estimates (ECE = [Z]).

---

## 2. Related Work (1 page)

### 2.1 Bayesian Optimization
- GP-based BO for continuous spaces
- Tree-structured Parzen estimators
- Neural network surrogates
- **Gap**: Limited to continuous/categorical spaces

### 2.2 Program Synthesis and Optimization
- Genetic programming
- Neural program synthesis
- AutoML / hyperparameter optimization
- **Gap**: Structure-agnostic mutations

### 2.3 LLMs for Code
- Code generation (Codex, StarCoder)
- Self-repair and debugging
- LLM-based optimization (OPRO, EvoPrompt)
- **Gap**: Not studied as BO proposal distribution

### 2.4 Algorithmic Trading
- RL for trading
- Genetic algorithms for strategy evolution
- **Gap**: No principled uncertainty quantification

---

## 3. Problem Formulation (1.5 pages)

### 3.1 Strategy Space
Define the space of trading strategies:

```
S = {s : D → A}
where D = market data (OHLCV), A = {LONG, SHORT, FLAT}
```

Each strategy s is represented as executable Python code.

### 3.2 Objective Function
Performance metric with noise:

```
f(s) = Sharpe(s; D_train) + ε,  ε ~ N(0, σ²)
```

Noise comes from:
- Market regime changes
- Limited sample size
- Execution variance

### 3.3 Bayesian Optimization View
- **Prior**: p(s) - uniform over syntactically valid strategies
- **Likelihood**: p(y|s,D) - backtest performance distribution
- **Posterior**: p(s|D,y) ∝ p(y|s,D)p(s)

### 3.4 LLM as Proposal Distribution

Traditional BO: `s_{t+1} ~ q(s | μ_GP, σ_GP)` (acquisition function)

Our approach: `s_{t+1} ~ q_LLM(s | s_t, f(s_t), feedback)`

The LLM proposes new strategies conditioned on:
- Current best strategy code
- Performance feedback (natural language)
- History of attempts

### 3.5 Key Assumptions
1. LLM can parse and generate valid strategy code
2. Performance feedback is informative
3. Strategy space has exploitable structure

---

## 4. Method (1.5 pages)

### 4.1 Algorithm: LLM-Guided Strategy Optimization

```
Algorithm 1: LLM-BO
Input: Initial strategy s_0, data D, iterations T
Output: Best strategy s*

1. s* ← s_0, f* ← -∞
2. for t = 1 to T:
3.   feedback ← DESCRIBE(s_{t-1}, f(s_{t-1}))
4.   s_t ← LLM.propose(s_{t-1}, feedback)
5.   if VALIDATE(s_t):
6.     f_t ← BACKTEST(s_t, D_train)
7.     if f_t > f*:
8.       s* ← s_t, f* ← f_t
9. return s*
```

### 4.2 Feedback Generation
Convert numeric performance to natural language:

```python
def describe(strategy, metrics):
    return f"""
    Current strategy achieved:
    - Sharpe ratio: {metrics.sharpe:.3f}
    - Win rate: {metrics.win_rate:.1%}
    - Max drawdown: {metrics.max_dd:.1%}

    Issues identified:
    - {identify_issues(metrics)}

    Suggest improvements focusing on:
    - {prioritize_improvements(metrics)}
    """
```

### 4.3 Confidence Estimation
Extract confidence from LLM:

```python
def get_confidence(llm_response):
    # Parse LLM's stated confidence
    # Calibrate using historical accuracy
    return calibrated_confidence
```

### 4.4 Walk-Forward Validation
- Train on D_train (70%), evaluate on D_test (30%)
- Rolling windows to prevent data leakage
- Out-of-sample performance is ground truth

---

## 5. Experiments (2.5 pages)

### 5.1 Experimental Setup

**Data:**
- Assets: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
- Period: 2 years historical data
- Interval: 15-minute candles
- Walk-forward periods: 20

**Methods compared:**
| Method | Description |
|--------|-------------|
| Static | No optimization (prior) |
| Random | Random parameter mutation |
| LLM-1 | Single LLM query (no iteration) |
| LLM-BO | Our method (iterative) |

**LLM models:**
- Qwen-2.5-Coder-7B (local)
- GPT-4o-mini (API)

**Metrics:**
- Sharpe ratio (primary)
- Cumulative regret
- Sample efficiency (iterations to 95% of best)
- Calibration (ECE, Brier score)
- Generalization gap (train-test divergence)

### 5.2 Main Results (Table 1)

| Method | Sharpe | 95% CI | Regret | ECE | Gen. Gap |
|--------|--------|--------|--------|-----|----------|
| Static | X.XXX | [X.XX, X.XX] | X.X | - | X.XXX |
| Random | X.XXX | [X.XX, X.XX] | X.X | - | X.XXX |
| LLM-1 | X.XXX | [X.XX, X.XX] | X.X | X.XXX | X.XXX |
| LLM-BO | **X.XXX** | [X.XX, X.XX] | **X.X** | X.XXX | X.XXX |

**Key findings:**
1. LLM-BO achieves [X]% higher Sharpe than Random (p < 0.05)
2. LLM-BO converges in [Y] iterations vs [Z] for Random
3. LLM confidence is well-calibrated (ECE = 0.XXX)

### 5.3 Sample Efficiency Analysis (Figure 2)

Plot: Performance vs. iteration number

- LLM-BO shows steeper learning curve
- Random requires [Y]x more samples to reach same performance
- Diminishing returns after ~[N] iterations

### 5.4 Calibration Analysis (Figure 3)

Reliability diagram showing:
- LLM confidence vs actual accuracy
- ECE comparison across methods
- Well-calibrated in [low/mid] confidence range
- Slight overconfidence at high confidence

### 5.5 Regret Analysis (Figure 4)

Cumulative regret plot:
- LLM-BO has sublinear regret growth
- Random shows linear regret
- Theoretical comparison to UCB bounds

### 5.6 Ablation Studies

**Effect of iteration count:**
| Iterations | Sharpe | Regret |
|------------|--------|--------|
| 5 | X.XXX | X.X |
| 10 | X.XXX | X.X |
| 15 | X.XXX | X.X |
| 20 | X.XXX | X.X |

**Effect of model size:**
| Model | Sharpe | Efficiency |
|-------|--------|------------|
| Qwen-7B | X.XXX | X.X iter |
| Qwen-14B | X.XXX | X.X iter |
| GPT-4o-mini | X.XXX | X.X iter |

---

## 6. Analysis (1.5 pages)

### 6.1 What Makes LLM Proposals Effective?

Analyze generated strategies:
- LLM makes semantically meaningful changes
- Exploits structure (parameter tuning vs. logic changes)
- Uses domain knowledge (RSI, EMA interpretation)

### 6.2 Failure Cases

When does LLM-BO fail?
- High market volatility periods
- Overconfident proposals
- Mode collapse (repeating same changes)

### 6.3 Exploration vs. Exploitation

Characterize LLM behavior:
- Early iterations: more exploration (diverse changes)
- Later iterations: more exploitation (refinement)
- Compare to ε-greedy, UCB

### 6.4 Generalization

Train-test gap analysis:
- LLM strategies generalize better than random
- Possible explanation: LLM avoids overfitting indicators

---

## 7. Discussion (0.5 pages)

### Limitations
1. Computational cost of LLM queries
2. Limited to strategy spaces LLM can reason about
3. Requires meaningful feedback generation

### Broader Impact
- Automated strategy development
- Risk of market manipulation
- Democratization vs. concentration of alpha

### Future Work
1. Theoretical regret bounds for LLM-BO
2. Multi-objective optimization
3. Transfer across asset classes

---

## 8. Conclusion (0.25 pages)

We demonstrated that LLMs can serve as effective proposal distributions for Bayesian optimization in structured program spaces. On a trading strategy benchmark, LLM-guided search achieves [X]% better performance than random search with [Y]x sample efficiency. Our analysis reveals well-calibrated confidence estimates and favorable exploration-exploitation dynamics. These findings suggest that LLMs' ability to reason about code structure enables more efficient optimization than structure-agnostic methods.

---

## Appendix

### A. Implementation Details
- Prompt templates
- Hyperparameters
- Compute resources

### B. Additional Results
- Per-asset breakdown
- Full ablation tables
- Strategy code examples

### C. Proofs
- Regret bound derivation (if applicable)

---

## Checklist for Submission

- [ ] Run all UAI experiments (5 seeds each)
- [ ] Generate all figures (calibration, regret, efficiency)
- [ ] Compute all statistics (p-values, CIs, effect sizes)
- [ ] Write LaTeX paper
- [ ] Prepare reproducibility package
- [ ] Submit by deadline

---

## Experiment Commands

```bash
# Main experiments
python -m experiments.run_experiment --preset uai_main_llm --seeds 42 123 456 789 1000
python -m experiments.run_experiment --preset uai_baseline_random --seeds 42 123 456 789 1000
python -m experiments.run_experiment --preset uai_baseline_static --seeds 42 123 456 789 1000

# Ablation
python -m experiments.run_experiment --preset uai_ablation_samples --seeds 42 123 456

# Generate paper outputs
python -c "from experiments.uai_analysis import UAIAnalyzer; UAIAnalyzer().generate_all_outputs()"
```
