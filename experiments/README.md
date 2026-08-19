# Experiment Framework

Research-grade experiment framework for LLM Strategy Evolution paper.

## Quick Start

```bash
# Run main experiment with Qwen-7B
python -m experiments.run_experiment --preset main_llm_qwen7b

# Run all baselines for comparison
python -m experiments.run_experiment --run-baselines

# Run with multiple seeds for statistical significance
python -m experiments.run_experiment --preset main_llm_qwen7b --seeds 42 123 456 789 1000

# List available presets
python -m experiments.run_experiment --list-presets
```

## Multimodal Pilot

The repo now includes an offline multimodal pilot built from:
- cached price data when parquet support is available
- local `data/history/*.jsonl` web-intelligence logs
- a fallback history-derived price proxy when parquet engines are missing
- two alignment modes:
  - `bar`: one sample per price bar
  - `event`: one sample per web-intelligence refresh event
- two evaluation modes:
  - single split
  - rolling walk-forward splits grouped by unique `window_end`

Build the pilot dataset only:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc.json --build-only
```

Run the full pilot comparison:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc.json
```

Run the clean event-aligned pilot with downloaded Binance prices:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc_event_clean_download.json
```

Run the rolling walk-forward event evaluation:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc_event_rolling.json
```

Run the text-aware event rolling evaluation:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc_text_event_rolling.json
```

Run the ACM MM style deep multimodal experiment:

```bash
python -m experiments.run_mm_deep_experiment --config experiments/configs/mm_deep_btc_event_rolling.json
```

Backfill historical multimodal web snapshots for three assets over three months:

```bash
python -m experiments.build_multimodal_history --symbols BTCUSDT,ETHUSDT,SOLUSDT --start-date 2025-10-01T00:00:00Z --end-date 2025-12-31T00:00:00Z --step-hours 2 --parallel-symbols 3 --sleep-seconds 1 --output-dir data/multimodal_history
```

Note: per-symbol parallel backfill is supported, but practical throughput is still constrained by OpenAI web-search quota.

Summarize current multimodal history coverage:

```bash
python -m experiments.summarize_multimodal_history
```

Run the pooled multi-asset ACM MM style deep experiment:

```bash
python -m experiments.run_mm_deep_experiment --config experiments/configs/mm_deep_multi_asset_event_rolling.json
```

Run a finer-grained 5m event rolling evaluation:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc_5m_event_rolling.json
```

Run a slower 4h-horizon event rolling evaluation:

```bash
python -m experiments.run_mm_experiment --config experiments/configs/mm_pilot_btc_event_rolling_h4.json
```

Outputs:
- cached dataset: `experiments/mm_cache/`
- run summaries: `experiments/results_mm/`
- historical multimodal backfill: `data/multimodal_history/`
- model families: `price_only_linear`, `web_only_linear`, `multimodal_linear`, `gated_residual_multimodal`
- deep model families: `price_sequence_transformer`, `text_embedding_mlp`, `cross_modal_attention`

## Directory Structure

```
experiments/
├── __init__.py           # Package init
├── config.py             # Experiment configuration
├── data_loader.py        # Data loading and caching
├── metrics.py            # Metrics and statistical tests
├── runner.py             # Main experiment runner
├── run_experiment.py     # CLI entry point
├── analysis.py           # Results analysis tools
├── baselines/            # Baseline implementations
│   ├── static.py         # No evolution baseline
│   ├── random_mutation.py # Random parameter mutation
│   └── single_shot.py    # Single-shot LLM baseline
├── data_cache/           # Cached market data
└── results/              # Experiment outputs
```

## Experiment Presets

| Preset | Description |
|--------|-------------|
| `main_llm_qwen7b` | LLM iterative evolution with Qwen-7B |
| `main_llm_gpt4o_mini` | LLM iterative evolution with GPT-4o-mini |
| `baseline_no_evolution` | Static EMA strategy (no learning) |
| `baseline_single_shot` | LLM generates once, no iteration |
| `baseline_random_mutation` | Random parameter mutation |
| `ablation_iterations_5` | 5 evolution iterations |
| `ablation_iterations_20` | 20 evolution iterations |

## Paper Experiments

### Main Results (Table 1)

```bash
# Run each method with 5 seeds
for preset in baseline_no_evolution baseline_random_mutation baseline_single_shot main_llm_qwen7b main_llm_gpt4o_mini; do
    python -m experiments.run_experiment --preset $preset --seeds 42 123 456 789 1000
done
```

### Ablation Study (Table 2)

```bash
# Effect of iteration count
python -m experiments.run_experiment --preset ablation_iterations_5 --seeds 42 123 456
python -m experiments.run_experiment --preset main_llm_qwen7b --seeds 42 123 456
python -m experiments.run_experiment --preset ablation_iterations_20 --seeds 42 123 456
```

### Statistical Analysis

```python
from experiments.analysis import ResultsAnalyzer

analyzer = ResultsAnalyzer()
results = analyzer.load_all_results()

# Generate summary table
df = analyzer.generate_summary_table(results)
print(df)

# Generate LaTeX for paper
latex = analyzer.generate_latex_table(results)
print(latex)

# Get paper statistics
stats = analyzer.generate_paper_statistics(results)
print(stats)
```

## Configuration

### Custom Config

```python
from experiments.config import ExperimentConfig, ModelConfig, ModelType

config = ExperimentConfig(
    name="my_experiment",
    symbols=["BTCUSDT", "ETHUSDT"],
    n_periods=15,
    n_iterations=10,
    model=ModelConfig(
        model_type=ModelType.QWEN_7B,
        host="http://localhost:11434"
    ),
    seed=42
)

config.save("my_config.json")
```

### Run Custom Config

```bash
python -m experiments.run_experiment --config my_config.json
```

## Output Format

Each experiment creates a directory in `results/` with:

```
results/{experiment_id}/
├── config.json           # Experiment configuration
├── results.json          # Full results
├── summary.json          # Quick summary metrics
├── checkpoint.json       # Intermediate checkpoint
├── data_snapshot.json    # Data reproducibility info
└── strategies/           # Generated strategy code
    ├── period_1.py
    ├── period_2.py
    └── ...
```

## Metrics

### Trading Metrics
- Sharpe Ratio (annualized)
- Total PnL
- Win Rate
- Profit Factor
- Max Drawdown
- Sortino Ratio
- Calmar Ratio

### Statistical Tests
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
- Bootstrap confidence intervals

## Requirements

```bash
pip install pandas numpy scipy binance-python ta openai
```

For local LLM (Ollama):
```bash
ollama pull qwen2.5-coder:7b
```

For OpenAI models:
```bash
export OPENAI_API_KEY="your-key"
```
