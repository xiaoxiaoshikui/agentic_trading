# ACM MM execution plan

## Goal
Turn the current repo into a credible ACM MM submission by reframing the problem as multimodal market intelligence rather than pure trading-policy engineering.

Working title:
`Multimodal market intelligence with participant-aware reasoning for crypto trading`

## What can stay
- `src/tom/`: keep as the downstream decision layer and as a strong price-only baseline.
- `experiments/run_tom_minimal.py`: keep as the trading evaluation harness.
- `experiments/results_tom/`: keep as baseline evidence and regression checks.
- `src/web_intelligence.py`: mine it for modality definitions and prompt ideas, not as the final paper system.

## What must change
- The paper task must become multimodal prediction and decision support.
- The main method must consume synchronized price, text, event, and optional chart-image inputs.
- The main evidence must include modality ablations and predictive metrics, not only Sharpe.
- The benchmark must be time-aligned and leakage-safe.

## Target problem statement
Given a market window ending at time `t`, predict short-horizon direction/risk and produce a trading action using:
- price series and technical indicators
- news headlines and snippets
- social posts or sentiment summaries
- on-chain or whale-transfer events
- optional rendered candlestick chart images

The core research question:
`Does multimodal evidence improve participant-aware market reasoning and downstream trading robustness over price-only baselines?`

## Proposed paper contribution
1. A time-aligned multimodal crypto benchmark that joins price, text, event, and optional chart modalities.
2. A participant-aware multimodal fusion model that estimates retail, momentum, and event-driven pressure before decision making.
3. A two-level evaluation protocol: predictive performance first, downstream trading impact second.
4. A rigorous ablation suite showing when extra modalities help, hurt, or become stale.

## Required code changes

### 1. Dataset layer
Add:
- `experiments/build_multimodal_dataset.py`
- `experiments/datasets/` or `data/mm/`

Responsibilities:
- align all modalities by end timestamp
- cache normalized windows
- enforce lookback and prediction horizon
- split by time, not random shuffle
- emit train/val/test manifests

Suggested sample schema:
- `window_end`
- `symbol`
- `price_window_path`
- `news_items`
- `social_items`
- `event_items`
- `chart_image_path`
- `label_direction`
- `label_volatility`
- `label_return`

### 2. Model layer
Add:
- `src/mm/encoders.py`
- `src/mm/fusion.py`
- `src/mm/model.py`
- `src/mm/dataset.py`

Minimum viable model:
- time-series encoder for OHLCV and indicators
- text encoder for news and social snippets
- event encoder for whale and macro or exchange-flow events
- fusion block with missing-modality masking
- participant reasoning head producing latent pressure scores
- task heads for direction and confidence

### 3. Evaluation layer
Add:
- `experiments/run_mm_experiment.py`
- `experiments/mm_metrics.py`
- `experiments/configs/mm_*.json`

Metrics:
- classification: accuracy, macro-F1, AUC
- regression: MAE or rank correlation for return target
- calibration: ECE, Brier
- downstream: Sharpe, max drawdown, turnover, hit rate

### 4. Trading integration
Modify after the predictive stack is stable:
- feed multimodal outputs into the existing ToM or trading policy as an input signal
- compare `price-only ToM`, `multimodal predictor + simple policy`, and `multimodal predictor + ToM policy`

## Experimental plan

### Phase 1: benchmark construction
Deliverable:
- frozen multimodal dataset spec
- leakage check
- coverage table per symbol and modality

Must answer:
- how many windows have all modalities
- what happens when one modality is missing
- whether label horizons are realistic

### Phase 2: predictive baselines
Run:
- price-only technical baseline
- price-only sequence model
- text-only model
- event-only model
- chart-only model if chart images are included
- simple early or late fusion baseline

Acceptance bar:
- at least one multimodal setup beats price-only on predictive metrics with confidence intervals

### Phase 3: participant-aware multimodal model
Run:
- fusion model without participant reasoning
- fusion model with participant-aware latent heads
- ablations removing news, social, whale, chart
- stale-text robustness with delayed news windows

Acceptance bar:
- participant-aware model shows either predictive gains or clearly better downstream robustness

### Phase 4: downstream trading study
Run:
- price-only ToM baseline
- multimodal signal only
- multimodal + ToM
- simple threshold policy versus strategic fusion

Acceptance bar:
- downstream gains are directionally consistent across seeds and do not collapse under larger reruns

## Week-by-week plan

### Week 1
- freeze ACM MM task definition
- define label horizon and modalities
- specify dataset schema and folder layout

### Week 2
- implement dataset builder
- build one-symbol pilot dataset
- verify timestamp alignment and no-lookahead

### Week 3
- add price-only and text-only baselines
- generate the first predictive benchmark table

### Week 4
- add event modality and missing-modality handling
- decide whether chart images are worth keeping

### Week 5
- implement participant-aware fusion model
- run core ablations

### Week 6
- integrate with the current trading harness
- produce downstream trading table

### Week 7
- run robustness checks, larger seeds, and stale-modality stress tests
- freeze figures and claim-evidence mapping

### Week 8
- write paper, trim claims, finalize appendix and artifacts

## Kill criteria
Do not force ACM MM if any of the following remains true after Phase 3:
- multimodal models do not beat strong price-only baselines on predictive metrics
- gains only appear in one fragile configuration
- text or event modalities are too sparse or too noisy to support a benchmark
- downstream gains remain statistically weak and inconsistent

If killed, redirect to:
- ECML PKDD or ICAIF for multimodal finance
- UAI-style analysis if the main novelty becomes uncertainty or calibration

## Immediate next coding tasks
1. Add a dataset specification markdown file under `paper/` or `docs/`.
2. Implement a pilot multimodal dataset builder over one symbol.
3. Add predictive experiment configs before touching more trading logic.
4. Keep the current ToM reviewer suite as a regression baseline, not as the paper's main evidence.
