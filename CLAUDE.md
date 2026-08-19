# Agentic Trading runbook

## Project overview
This repo currently has two active lines of work:

- LLM-BO and ToM trading research under `experiments/` and `src/tom/`
- an offline multimodal financial forecasting line for the paper under `experiments/` and `paper/latex/`

The multimodal line is the current priority. The ToM stack is still runnable, but it is no longer the main paper path.

This file is intentionally a lean runbook. It keeps current commands, current architecture, and current conclusions. Detailed historical experiment notes should live in result directories and git history, not here.

## Current priorities
- keep the multimodal path reproducible on the real-news corpus
- keep CGCMA as the main paper architecture
- keep the paper tables tied to the correct run batches
- finish final paper cleanup and submission packaging
- keep ToM stable, but treat it as secondary work

## Quick commands
```powershell
# Install dependencies
python -m pip install --user -r requirements.txt

# LLM-BO smoke run
set DEEPSEEK_API_KEY=YOUR_KEY
set DEEPSEEK_API_BASE=https://api.deepseek.com
set DEEPSEEK_MODEL=deepseek-chat
python -m experiments.run_experiment --preset main_llm_gpt4o_mini --n-periods 3 --n-iterations 3 --seeds 42

# ToM minimal run
python -m experiments.run_tom_minimal --config experiments/configs/tom_minimal.json

# Build or resume multimodal history
python -m experiments.build_multimodal_history --symbols BTCUSDT,ETHUSDT,SOLUSDT --start-date 2025-01-01T00:00:00Z --end-date 2025-09-30T00:00:00Z --step-hours 1 --parallel-symbols 3 --sleep-seconds 1 --output-dir data/multimodal_history --api-base https://api.deepseek.com

# Summarize and clean multimodal history
python -m experiments.summarize_multimodal_history
python -m experiments.audit_multimodal_history --history-glob "data/multimodal_history/*.jsonl" --output-dir experiments/results_mm_quality/all_assets_audit
python -m experiments.clean_multimodal_history --history-glob "data/multimodal_history/*.jsonl" --output-dir data/multimodal_history_cleaned

# Final multimodal paper comparison
python -m experiments.run_mm_deep_experiment --config experiments/configs/mm_deep_multi_real_news_final.json --seed 42

# Summarize canonical 8-seed real-news results
python -m experiments.analyze_mm_real_news_stats --base-dir experiments/results_mm --run-dirs mm_deep_multi_real_news_final_20260321_085909_287359_seed42 mm_deep_multi_real_news_final_20260321_090109_571385_seed123 mm_deep_multi_real_news_final_20260321_090109_238021_seed456 mm_deep_multi_real_news_final_20260321_090109_623767_seed789 mm_deep_multi_real_news_final_20260327_211230_910863_seed101 mm_deep_multi_real_news_final_20260327_211231_744873_seed202 mm_deep_multi_real_news_final_20260327_211235_204350_seed303 mm_deep_multi_real_news_final_20260327_211221_282763_seed404 --output-csv experiments/results_mm/real_news_final_stats_8seed_20260329.csv --output-md experiments/results_mm/real_news_final_stats_8seed_20260329.md

# Gate export analysis for appendix / rebuttal only
python -m experiments.run_mm_deep_experiment --config experiments/configs/mm_deep_multi_real_news_2026_cgcma_gate_analysis.json --seed 42

# Generate ToM paper artifacts from an existing run
python experiments/generate_tom_paper_artifacts.py --run-dir experiments/results_tom/tom_minimal_20260207_151606
```

## Environment
- Python: 3.9+; tested on 3.11
- Install: `python -m pip install --user -r requirements.txt`
- Optional env vars:
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_API_BASE` with default `https://api.deepseek.com`
  - `DEEPSEEK_MODEL` such as `deepseek-chat`

## Repo map
- `experiments/`: experiment runners, configs, metrics, baselines, and analysis
- `experiments/llm_bo.py`: LLM-BO strategy evolution
- `experiments/run_experiment.py`: main LLM-BO entrypoint
- `experiments/run_tom_minimal.py`: ToM evaluation entrypoint
- `experiments/run_mm_experiment.py`: linear or lightweight multimodal pipeline
- `experiments/run_mm_deep_experiment.py`: deep multimodal training and rolling evaluation
- `experiments/mm_dataset.py`: event or bar aligned multimodal dataset builder
- `experiments/mm_deep_dataset.py`: deep multimodal sequence dataset builder
- `experiments/mm_deep_models.py`: deep multimodal model implementations
- `src/`: trading core
- `src/tom/`: ToM agents and evaluation harness
- `paper/latex/`: paper source and figure generation
- `data/multimodal_history_real/`: current primary real-news corpus

## Current multimodal status

### Dataset
- Primary dataset: `data/multimodal_history_real/`
- Source: CryptoCompare real news
- Current paper experiments use pooled BTC, ETH, and SOL data
- Real-news corpus is the default source for current claims
- Synthetic history is retained only for controlled development, lag probing, and the full-year scaling study

### Main architecture
- Current proposed model: `ConditionallyGatedCrossModalFusion` (`conditionally_gated_cross_modal`)
- Core idea:
  - text attends over the full price sequence
  - the resulting context is gated before being injected into the price representation
  - web features remain a direct predictive cue at the head
- Fusion form:
  - `h_c = LN(MHA(h_t, H^p, H^p))`
  - `Delta_pc = h_p - h_c`
  - `g = sigma(W_g [h_p, h_c, Delta_pc, h_w, tau/60])`
  - `h_f = h_p + g * h_c`

### Implemented model families
- Price-only: `price_sequence_transformer`
- Text-only: `text_embedding_mlp`
- Earlier fusion variants: `cross_modal_attention`, `staleness_aware_cross_modal`, `learned_gate_cross_modal`, `gated_late_fusion`
- Current paper model: `conditionally_gated_cross_modal`
- Published baselines:
  - `early_fusion`
  - `bilstm_fusion`
  - `multimodal_transformer`
  - `tensor_fusion`

### Current results snapshot
As of 2026-03-29, the canonical main paper comparison is the pooled real-news 8-seed batch from `experiments/configs/mm_deep_multi_real_news_final.json`.

| Model | Sharpe mean | Sharpe std |
| --- | ---: | ---: |
| PriceTx-S | +0.194 | 0.229 |
| TextOnly | -0.043 | 0.168 |
| EarlyFusion | +0.198 | 0.150 |
| BiLSTM | +0.209 | 0.196 |
| MulT | +0.187 | 0.220 |
| TFN | -0.033 | 0.118 |
| CGCMA | +0.449 | 0.257 |

Working conclusion:
- CGCMA is the current best multimodal architecture in this repo under the real-news pooled protocol
- the main paper should use the 8-seed Table 3 numbers, not the older 4-seed batch
- synthetic-data conclusions are secondary and should not override the real-news table
- the strongest paired result is currently `CGCMA - PriceTx-S = +0.256`, bootstrap 95% CI `[0.003, 0.491]`
- `CGCMA - BiLSTM` is still positive but not yet paired-significant enough to be framed as definitive separation

### Current paper table mapping
- `Table 3` (`tab:main`): canonical 8-seed pooled real-news comparison
- `Table 4` (`tab:main_aux`): auxiliary metrics aligned to `Table 3`, except `PriceWeb` which is still a 4-run control
- `Table 5` (`tab:design`): separate dedicated 4-run pooled real-news sweep for task-specific lag-aware controls; use only for within-table comparison
- `Table 6` (`tab:v5_scaling`): separate 4-run synthetic full-year BTC scaling study
- Do not mix absolute values across these tables without explicitly noting the distinct run batch

### Canonical result references
- final pooled comparison:
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260321_085909_287359_seed42/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260321_090109_571385_seed123/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260321_090109_238021_seed456/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260321_090109_623767_seed789/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260327_211230_910863_seed101/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260327_211231_744873_seed202/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260327_211235_204350_seed303/`
  - `experiments/results_mm/mm_deep_multi_real_news_final_20260327_211221_282763_seed404/`
- 8-seed summaries:
  - `experiments/results_mm/real_news_final_stats_8seed_20260329.csv`
  - `experiments/results_mm/real_news_final_stats_8seed_20260329.md`
  - `experiments/results_mm/real_news_final_paired_8seed_20260329.csv`
- PriceWeb control:
  - `experiments/results_mm/mm_deep_multi_real_news_2026_pw_baseline_*_seed*/`
- task-specific control sweep:
  - `experiments/configs/mm_deep_multi_real_news_2026_task_specific.json`
- clean ablation sweep:
  - `experiments/configs/mm_deep_multi_real_news_2026_cgcma_ablation.json`
- gate analysis (not main-paper evidence):
  - `experiments/results_mm/mm_deep_multi_real_news_2026_cgcma_gate_analysis_20260327_185108_808747_seed42/`
  - `experiments/results_mm/mm_deep_multi_real_news_2026_cgcma_gate_analysis_20260328_001113_295737_seed123/`
- paper figures:
  - `paper/latex/figures/cgcma_arch_v2.drawio.pdf`
  - `paper/latex/figures/lag_sharpe.pdf`
  - `paper/latex/figures/gate_lag_analysis.pdf` (appendix / rebuttal only; weak signal)

## Paper status
- target venue: ACM MM 2026 main track
- paper source: `paper/latex/`
- current story: event-conditioned asynchronous fusion, with crypto as a timestamped stress test

### Done
- canonical 8-seed main comparison completed and written into `Table 3`
- PriceWeb control completed and written into `Table 3/4`
- task-specific lag-aware controls completed and written into `Table 5`
- clean ablation completed and summarized in `Section 6.2`
- architecture figure, lag figure, and final bibliography audit are complete
- `Section 6.3` now explicitly links Figure 2 to `Table 5` and to grounded, lag-aware trust rather than recency-only handling
- nearby result-section floats were relaxed to `[tbp]` to reduce awkward white space before `6.3`
- paper currently compiles to `9` pages total (`8` body + references)

### Still required
- if more evidence is needed, prioritize threshold / fee sensitivity or additional seeds rather than more old baselines
- keep `07_discussion.tex` out of the main paper unless intentionally reintroduced; it is currently not `\input`
- run a final sequential LaTeX compile pass before submission packaging
- do not use `gate_lag_analysis.pdf` as main-paper evidence; keep it for appendix / rebuttal only
- if layout around `6.3` still feels loose, shorten the end of `6.2` before touching float spec again

## ToM status
- ToM is no longer the primary paper line
- the stack is still useful as a research baseline and for reviewer-oriented comparisons
- safest summary:
  - ToM has setting-dependent gains
  - robustness across regimes and larger-sample checks remains weak
  - it is not the current top-tier paper candidate compared with the multimodal line

### Canonical ToM entrypoints
- base run:
  - `python -m experiments.run_tom_minimal --config experiments/configs/tom_minimal.json`
- reviewer suite:
  - `python -m experiments.run_reviewer_suite`

### Canonical ToM references
- stable artifact run:
  - `experiments/results_tom/tom_minimal_20260207_151606/`
- reviewer suite summaries:
  - `experiments/results_tom/reviewer_suite_core_20260304_211942.md`
  - `experiments/results_tom/reviewer_suite_power_core_split_20260306_082625.md`

## Metrics and evaluation
- primary trading metric: Sharpe ratio
- other common outputs: total PnL, AUC, macro-F1, win rate
- multimodal result summaries are written under `experiments/results_mm/`
- ToM result summaries are written under `experiments/results_tom/`
- current paper wording should treat Sharpe as an offline utility proxy, not a live-trading claim

## Troubleshooting

### Missing dependencies
Symptom:
`ModuleNotFoundError: No module named 'binance'`

Fix:
`python -m pip install --user -r requirements.txt`

### PowerShell profile noise
Symptom:
PowerShell prints an execution-policy or profile-loading warning before running commands.

Fix:
Run commands with `powershell -NoProfile -Command "..."` if needed.

### Large markdown drift
If this file starts growing into a research diary again:
- keep only current commands and current conclusions here
- move detailed run notes into result directories
- rely on git history for old intermediate states

### Mixed run batches in paper tables
Symptom:
Paper tables or text compare absolute values across `Table 3`, `Table 5`, and synthetic sweeps as if they came from the same run batch.

Fix:
- treat `Table 3` as the headline absolute comparison
- treat `Table 5` as within-sweep only
- explicitly call out `PriceWeb` as a 4-run control whenever it appears next to 8-seed results

Verification:
- check `paper/latex/sections/06_results.tex`
- ensure captions and notes still mention the distinct run batches

### No active multimodal experiments
Current status:
- the canonical 8-seed main comparison has finished
- both gate-analysis reruns (`seed42`, `seed123`) have finished
- there are no active multimodal long runs that the paper currently depends on

Implication:
- update the paper from the existing result files rather than waiting on more background jobs

## Documentation rule
`CLAUDE.md` is the current runbook, not a full lab notebook.
