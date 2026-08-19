"""
Results Analysis
================

Tools for analyzing and comparing experiment results.
Generates tables and statistics for the research paper.
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metrics import MetricsCalculator


@dataclass
class ComparisonResult:
    """Result of comparing two methods"""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    improvement: float  # (B - A) / |A| * 100
    t_statistic: float
    p_value: float
    significance: str
    effect_size: float


class ResultsAnalyzer:
    """
    Analyze and compare experiment results.
    """

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = results_dir
        self.metrics_calc = MetricsCalculator()

    def load_all_results(self) -> Dict[str, List[Dict]]:
        """Load all experiment results grouped by method"""
        results_by_method = {}

        # Find all result files
        pattern = os.path.join(self.results_dir, "**/results.json")
        result_files = glob.glob(pattern, recursive=True)

        for path in result_files:
            try:
                with open(path, "r") as f:
                    result = json.load(f)

                method = result.get("config", {}).get("method", "unknown")

                if method not in results_by_method:
                    results_by_method[method] = []

                results_by_method[method].append(result)

            except Exception as e:
                print(f"Error loading {path}: {e}")

        return results_by_method

    def load_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Load a specific experiment result"""
        pattern = os.path.join(self.results_dir, f"*{experiment_id}*", "results.json")
        matches = glob.glob(pattern)

        if matches:
            with open(matches[0], "r") as f:
                return json.load(f)
        return None

    def compare_methods(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        method_a_name: str,
        method_b_name: str,
        metrics: List[str] = ["mean_test_sharpe", "total_test_pnl", "mean_test_win_rate"]
    ) -> List[ComparisonResult]:
        """
        Compare two methods across multiple metrics.

        Args:
            results_a: List of results for method A (baseline)
            results_b: List of results for method B (treatment)
            method_a_name: Name of method A
            method_b_name: Name of method B
            metrics: List of metrics to compare

        Returns:
            List of ComparisonResult for each metric
        """
        comparisons = []

        for metric in metrics:
            values_a = [r.get(metric, 0) for r in results_a]
            values_b = [r.get(metric, 0) for r in results_b]

            if not values_a or not values_b:
                continue

            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)

            # Calculate improvement
            if mean_a != 0:
                improvement = (mean_b - mean_a) / abs(mean_a) * 100
            else:
                improvement = float('inf') if mean_b > 0 else 0

            # Statistical tests
            t_stat, p_val, sig = self.metrics_calc.paired_t_test(values_a, values_b)
            effect_size = self.metrics_calc.effect_size_cohens_d(values_a, values_b)

            comparisons.append(ComparisonResult(
                method_a=method_a_name,
                method_b=method_b_name,
                metric=metric,
                mean_a=mean_a,
                mean_b=mean_b,
                improvement=improvement,
                t_statistic=t_stat,
                p_value=p_val,
                significance=sig,
                effect_size=effect_size
            ))

        return comparisons

    def generate_summary_table(
        self,
        results_by_method: Dict[str, List[Dict]]
    ) -> pd.DataFrame:
        """
        Generate summary table comparing all methods.

        Returns DataFrame suitable for paper table.
        """
        rows = []

        for method, results in results_by_method.items():
            if not results:
                continue

            # Collect metrics across all runs
            sharpes = [r.get("mean_test_sharpe", 0) for r in results]
            pnls = [r.get("total_test_pnl", 0) for r in results]
            win_rates = [r.get("mean_test_win_rate", 0) for r in results]

            # Calculate statistics with confidence intervals
            sharpe_mean, sharpe_ci_low, sharpe_ci_high = \
                self.metrics_calc.bootstrap_confidence_interval(sharpes)
            pnl_mean, pnl_ci_low, pnl_ci_high = \
                self.metrics_calc.bootstrap_confidence_interval(pnls)
            wr_mean, wr_ci_low, wr_ci_high = \
                self.metrics_calc.bootstrap_confidence_interval(win_rates)

            rows.append({
                "Method": method,
                "N": len(results),
                "Sharpe": f"{sharpe_mean:.3f}",
                "Sharpe_CI": f"[{sharpe_ci_low:.3f}, {sharpe_ci_high:.3f}]",
                "Total_PnL": f"{pnl_mean:.1f}",
                "PnL_CI": f"[{pnl_ci_low:.1f}, {pnl_ci_high:.1f}]",
                "Win_Rate": f"{wr_mean:.1%}",
                "WR_CI": f"[{wr_ci_low:.1%}, {wr_ci_high:.1%}]",
            })

        return pd.DataFrame(rows)

    def generate_latex_table(
        self,
        results_by_method: Dict[str, List[Dict]],
        baseline_method: str = "no_evolution"
    ) -> str:
        """
        Generate LaTeX table for paper.
        """
        df = self.generate_summary_table(results_by_method)

        # Add significance markers
        if baseline_method in results_by_method:
            baseline_results = results_by_method[baseline_method]
            baseline_sharpes = [r.get("mean_test_sharpe", 0) for r in baseline_results]

            for method, results in results_by_method.items():
                if method == baseline_method:
                    continue

                sharpes = [r.get("mean_test_sharpe", 0) for r in results]
                _, p_val, sig = self.metrics_calc.paired_t_test(baseline_sharpes, sharpes)

                # Add significance marker to table
                idx = df[df["Method"] == method].index
                if len(idx) > 0:
                    df.loc[idx[0], "Sharpe"] += f" {sig}"

        # Generate LaTeX
        latex = df.to_latex(index=False, escape=False)

        # Add caption and label
        latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Comparison of Strategy Evolution Methods}}
\\label{{tab:main_results}}
{latex}
\\end{{table}}
"""
        return latex

    def analyze_evolution_trajectory(
        self,
        results: List[Dict]
    ) -> pd.DataFrame:
        """
        Analyze how performance changes across iterations.

        Useful for understanding convergence behavior.
        """
        all_periods = []

        for result in results:
            for period in result.get("period_results", []):
                all_periods.append({
                    "experiment": result.get("experiment_id", ""),
                    "period": period.get("period", 0),
                    "n_iterations": period.get("n_iterations", 0),
                    "best_iteration": period.get("best_iteration", 0),
                    "train_sharpe": period.get("train_sharpe", 0),
                    "test_sharpe": period.get("test_sharpe", 0),
                    "evolved": period.get("evolved", False),
                })

        return pd.DataFrame(all_periods)

    def generate_paper_statistics(
        self,
        results_by_method: Dict[str, List[Dict]],
        baseline_method: str = "no_evolution",
        treatment_method: str = "llm_iterative"
    ) -> Dict[str, Any]:
        """
        Generate all statistics needed for paper claims.
        """
        stats = {}

        if baseline_method not in results_by_method:
            return {"error": f"Baseline method {baseline_method} not found"}

        if treatment_method not in results_by_method:
            return {"error": f"Treatment method {treatment_method} not found"}

        baseline = results_by_method[baseline_method]
        treatment = results_by_method[treatment_method]

        # Compare methods
        comparisons = self.compare_methods(
            baseline, treatment,
            baseline_method, treatment_method
        )

        for comp in comparisons:
            stats[f"{comp.metric}_improvement"] = f"{comp.improvement:.1f}%"
            stats[f"{comp.metric}_p_value"] = f"{comp.p_value:.4f}"
            stats[f"{comp.metric}_significance"] = comp.significance
            stats[f"{comp.metric}_effect_size"] = f"{comp.effect_size:.3f}"

        # Overall statistics
        treatment_sharpes = [r.get("mean_test_sharpe", 0) for r in treatment]
        mean, ci_low, ci_high = self.metrics_calc.bootstrap_confidence_interval(treatment_sharpes)

        stats["treatment_sharpe_mean"] = f"{mean:.4f}"
        stats["treatment_sharpe_95ci"] = f"[{ci_low:.4f}, {ci_high:.4f}]"

        return stats


def print_comparison_report(analyzer: ResultsAnalyzer):
    """Print a formatted comparison report"""
    results = analyzer.load_all_results()

    print("\n" + "="*70)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*70)

    # Summary table
    df = analyzer.generate_summary_table(results)
    print("\nSummary Table:")
    print(df.to_string(index=False))

    # Statistical comparisons
    if "no_evolution" in results and "llm_iterative" in results:
        print("\n" + "-"*70)
        print("Statistical Comparison: LLM Iterative vs No Evolution (Baseline)")
        print("-"*70)

        comparisons = analyzer.compare_methods(
            results["no_evolution"],
            results["llm_iterative"],
            "No Evolution",
            "LLM Iterative"
        )

        for comp in comparisons:
            print(f"\n{comp.metric}:")
            print(f"  Baseline mean: {comp.mean_a:.4f}")
            print(f"  LLM mean: {comp.mean_b:.4f}")
            print(f"  Improvement: {comp.improvement:.1f}%")
            print(f"  p-value: {comp.p_value:.4f} {comp.significance}")
            print(f"  Effect size (Cohen's d): {comp.effect_size:.3f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    print_comparison_report(analyzer)
