#!/usr/bin/env python
"""
Run Experiment CLI
==================

Command-line interface for running experiments.

Usage:
    # Run predefined experiment
    python -m experiments.run_experiment --preset main_llm_qwen7b

    # Run with custom config
    python -m experiments.run_experiment --config path/to/config.json

    # Run multiple seeds
    python -m experiments.run_experiment --preset main_llm_qwen7b --seeds 42 123 456

    # Run all baselines
    python -m experiments.run_experiment --run-baselines
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import ExperimentConfig, EXPERIMENT_CONFIGS
from experiments.runner import ExperimentRunner


def setup_logging(verbose: bool = True):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_single_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment and return results"""
    runner = ExperimentRunner(config)
    result = runner.run()
    return result.to_dict()


def run_with_seeds(base_config: ExperimentConfig, seeds: list) -> list:
    """Run experiment with multiple seeds"""
    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}\n")

        config = ExperimentConfig.from_dict({**base_config.to_dict(), "seed": seed})
        config.name = f"{base_config.name}_seed{seed}"

        result = run_single_experiment(config)
        results.append(result)

    return results


def run_baselines(seeds: list = [42]) -> dict:
    """Run all baseline experiments"""
    baseline_configs = [
        "baseline_no_evolution",
        "baseline_single_shot",
        "baseline_random_mutation",
    ]

    all_results = {}

    for config_name in baseline_configs:
        print(f"\n{'#'*60}")
        print(f"Running baseline: {config_name}")
        print(f"{'#'*60}\n")

        config = EXPERIMENT_CONFIGS[config_name]
        results = run_with_seeds(config, seeds)
        all_results[config_name] = results

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Strategy Evolution Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run predefined experiment
  python -m experiments.run_experiment --preset main_llm_qwen7b

  # Run with specific seeds for reproducibility
  python -m experiments.run_experiment --preset main_llm_qwen7b --seeds 42 123 456

  # Run all baselines
  python -m experiments.run_experiment --run-baselines

  # List available presets
  python -m experiments.run_experiment --list-presets
        """
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Predefined experiment configuration"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config JSON file"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to run (default: 42)"
    )

    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run all baseline experiments"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset configurations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--n-periods",
        type=int,
        help="Override number of walk-forward periods"
    )

    parser.add_argument(
        "--n-iterations",
        type=int,
        help="Override number of evolution iterations"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Override symbols to test"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=not args.quiet)

    # List presets
    if args.list_presets:
        print("\nAvailable experiment presets:\n")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  {name}")
            print(f"    Method: {config.method.value}")
            print(f"    Description: {config.description}")
            print()
        return

    # Run baselines
    if args.run_baselines:
        results = run_baselines(args.seeds)
        print("\n" + "="*60)
        print("Baseline experiments complete!")
        print("="*60)
        return

    # Load config
    if args.preset:
        config = EXPERIMENT_CONFIGS[args.preset]
    elif args.config:
        config = ExperimentConfig.load(args.config)
    else:
        parser.error("Either --preset or --config is required")

    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.n_periods:
        config.n_periods = args.n_periods
    if args.n_iterations:
        config.n_iterations = args.n_iterations
    if args.symbols:
        config.symbols = args.symbols

    # Run experiment(s)
    if len(args.seeds) == 1:
        config.seed = args.seeds[0]
        result = run_single_experiment(config)

        print("\n" + "="*60)
        print("Experiment complete!")
        print(f"Mean test Sharpe: {result.get('mean_test_sharpe', 0):.4f}")
        print(f"Total test PnL: {result.get('total_test_pnl', 0):.2f}")
        print("="*60)

    else:
        results = run_with_seeds(config, args.seeds)

        # Summarize across seeds
        sharpes = [r.get('mean_test_sharpe', 0) for r in results]
        pnls = [r.get('total_test_pnl', 0) for r in results]

        print("\n" + "="*60)
        print(f"Experiments complete! ({len(args.seeds)} seeds)")
        print(f"Mean test Sharpe: {sum(sharpes)/len(sharpes):.4f} +/- {(max(sharpes)-min(sharpes))/2:.4f}")
        print(f"Mean total PnL: {sum(pnls)/len(pnls):.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
