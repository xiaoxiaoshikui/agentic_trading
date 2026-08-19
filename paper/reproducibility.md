# Reproducibility and artifacts

## Current reproducible baseline in this repo

### Environment
- Python: `3.11`
- Virtual env: `.venv`
- Main dependency file: `requirements.txt`

### Core commands
```powershell
Set-Location "."
$py = ".venv\Scripts\python.exe"

# unit tests for the current ToM baseline
& $py -m unittest test_tom_horizon_features.py test_tom_calibration.py test_tom_multi_agent_signal.py

# reviewer suite (current downstream baseline)
& $py -m experiments.run_reviewer_suite --suite core --output-prefix reviewer_suite
& $py -m experiments.run_reviewer_suite --suite extended --output-prefix reviewer_suite

# larger rerun for regression checking
& $py -m experiments.run_reviewer_suite --suite core --output-prefix reviewer_suite_power --seeds "42,123,456,789,2026,7,8,9" --n-periods 7 --train-ratio 0.72
```

### Current output locations
- Main outputs: `experiments/results_tom/`
- Suite summaries: `reviewer_suite_*.csv`, `reviewer_suite_*.md`
- Per-run logs: `experiments/results_tom/reviewer_logs/`

### Current anti-leakage notes
- Execution uses current bar `open`.
- Signals use past bars only.
- Walk-forward train/test split is enforced in the experiment runner.

## Missing pieces before an ACM MM submission is reproducible

### Dataset artifacts
Must archive:
- multimodal dataset schema
- raw-source manifest with timestamps
- processed split manifests
- modality coverage statistics

### Preprocessing artifacts
Must archive:
- text cleaning rules
- deduplication rules
- timestamp alignment policy
- missing-modality policy
- chart rendering recipe if chart images are used

### Experiment artifacts
Must archive:
- predictive experiment configs
- downstream trading configs
- exact seeds
- model checkpoints or deterministic re-run path
- exported tables and figures

### Reporting policy
- Report predictive and downstream metrics separately.
- Report both positive and negative modality settings.
- Keep downstream claims narrower than predictive claims.
- Do not claim ACM MM novelty from `web_intelligence` prototypes alone.

## Artifact checklist to satisfy before submission
- [ ] One command builds the frozen multimodal dataset from cached sources.
- [ ] One command runs the predictive benchmark.
- [ ] One command runs downstream trading evaluation from saved predictive outputs.
- [ ] Split manifests and seeds are included in the artifact package.
- [ ] Figures and paper tables are generated from saved CSV files.
- [ ] A short hardware and runtime summary is included.
