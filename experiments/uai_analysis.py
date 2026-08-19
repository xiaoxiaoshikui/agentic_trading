"""
UAI Paper Analysis
==================

Generate all figures, tables, and statistics for UAI paper.

Paper framing:
"LLM-Guided Bayesian Optimization in Strategy Space"

Key claims to support:
1. LLM proposals are more sample-efficient than random search
2. LLM confidence is well-calibrated
3. LLM-evolved strategies generalize (low distribution shift)
4. Regret bounds are favorable
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .metrics import MetricsCalculator
from .uncertainty import UncertaintyAnalyzer, UncertaintyMetrics


@dataclass
class UAIPaperResults:
    """All results needed for UAI paper"""

    # Main comparison
    llm_vs_random_sharpe_improvement: float = 0.0
    llm_vs_random_p_value: float = 1.0
    llm_vs_random_effect_size: float = 0.0

    # Sample efficiency
    llm_iterations_to_converge: float = 0.0
    random_iterations_to_converge: float = 0.0
    efficiency_ratio: float = 0.0

    # Calibration
    llm_ece: float = 0.0
    llm_brier: float = 0.0

    # Regret
    llm_cumulative_regret: float = 0.0
    random_cumulative_regret: float = 0.0
    regret_improvement: float = 0.0

    # Generalization
    llm_generalization_gap: float = 0.0
    random_generalization_gap: float = 0.0

    def to_latex_claims(self) -> str:
        """Generate LaTeX macros for paper claims"""
        return f"""
% UAI Paper Claims - Auto-generated
\\newcommand{{\\sharpeimprovement}}{{{self.llm_vs_random_sharpe_improvement:.1f}\\%}}
\\newcommand{{\\pvalue}}{{{self.llm_vs_random_p_value:.4f}}}
\\newcommand{{\\effectsize}}{{{self.llm_vs_random_effect_size:.2f}}}
\\newcommand{{\\efficiencyratio}}{{{self.efficiency_ratio:.1f}x}}
\\newcommand{{\\llmece}}{{{self.llm_ece:.3f}}}
\\newcommand{{\\regretimprovement}}{{{self.regret_improvement:.1f}\\%}}
\\newcommand{{\\gengap}}{{{self.llm_generalization_gap:.3f}}}
"""


class UAIAnalyzer:
    """
    Generate all analysis for UAI paper.
    """

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = results_dir
        self.metrics_calc = MetricsCalculator()
        self.uncertainty_analyzer = UncertaintyAnalyzer()

    def load_results(self, prefix: str = "uai_") -> Dict[str, List[Dict]]:
        """Load all UAI experiment results"""
        results_by_method = {}

        pattern = os.path.join(self.results_dir, f"*{prefix}*", "results.json")
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

    def compute_regret_curves(
        self,
        results: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute average regret curves for each method.

        Returns dict mapping method -> array of cumulative regret at each iteration.
        """
        regret_curves = {}

        for method, method_results in results.items():
            all_curves = []

            for result in method_results:
                for period in result.get("period_results", []):
                    # Extract performance history if available
                    perf_history = period.get("performance_history", [])
                    if perf_history:
                        regret_data = self.uncertainty_analyzer.compute_regret(perf_history)
                        all_curves.append(regret_data.get("regret_curve", []))

            if all_curves:
                # Pad curves to same length
                max_len = max(len(c) for c in all_curves)
                padded = []
                for curve in all_curves:
                    if len(curve) < max_len:
                        curve = list(curve) + [curve[-1]] * (max_len - len(curve))
                    padded.append(curve)

                regret_curves[method] = np.mean(padded, axis=0)

        return regret_curves

    def generate_calibration_plot(
        self,
        results: Dict[str, List[Dict]],
        output_path: str = "calibration_plot.pdf"
    ):
        """
        Generate calibration plot (reliability diagram).

        Shows confidence vs accuracy for each method.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

        colors = {'llm_iterative': 'blue', 'random_mutation': 'red', 'no_evolution': 'gray'}

        for method, method_results in results.items():
            # Aggregate calibration data
            all_confidences = []
            all_outcomes = []

            for result in method_results:
                for period in result.get("period_results", []):
                    confidences = period.get("confidences", [])
                    outcomes = period.get("outcomes", [])
                    all_confidences.extend(confidences)
                    all_outcomes.extend(outcomes)

            if all_confidences and all_outcomes:
                ece, mce, cal_data = self.uncertainty_analyzer.compute_calibration_error(
                    all_confidences, all_outcomes
                )

                if cal_data.get("bins"):
                    color = colors.get(method, 'green')
                    ax.scatter(
                        cal_data["confidence"],
                        cal_data["accuracy"],
                        s=[c * 10 for c in cal_data["count"]],
                        label=f"{method} (ECE={ece:.3f})",
                        alpha=0.7,
                        color=color
                    )

        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Actual Accuracy")
        ax.set_title("Calibration Plot")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_regret_plot(
        self,
        results: Dict[str, List[Dict]],
        output_path: str = "regret_plot.pdf"
    ):
        """
        Generate cumulative regret plot.

        Shows how quickly each method approaches optimal.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = {'llm_iterative': 'blue', 'random_mutation': 'red', 'no_evolution': 'gray'}
        labels = {'llm_iterative': 'LLM (Ours)', 'random_mutation': 'Random', 'no_evolution': 'Static'}

        regret_curves = self.compute_regret_curves(results)

        for method, curve in regret_curves.items():
            color = colors.get(method, 'green')
            label = labels.get(method, method)
            ax.plot(range(len(curve)), np.cumsum(curve), label=label, color=color, linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title("Regret Comparison")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_sample_efficiency_plot(
        self,
        results: Dict[str, List[Dict]],
        output_path: str = "efficiency_plot.pdf"
    ):
        """
        Generate sample efficiency plot.

        Shows performance vs iteration for each method.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = {'llm_iterative': 'blue', 'random_mutation': 'red', 'no_evolution': 'gray'}
        labels = {'llm_iterative': 'LLM (Ours)', 'random_mutation': 'Random', 'no_evolution': 'Static'}

        for method, method_results in results.items():
            # Collect performance at each iteration
            iteration_performances = {}

            for result in method_results:
                for period in result.get("period_results", []):
                    perf_history = period.get("performance_history", [])
                    for i, perf in enumerate(perf_history):
                        if i not in iteration_performances:
                            iteration_performances[i] = []
                        iteration_performances[i].append(perf)

            if iteration_performances:
                iterations = sorted(iteration_performances.keys())
                means = [np.mean(iteration_performances[i]) for i in iterations]
                stds = [np.std(iteration_performances[i]) for i in iterations]

                color = colors.get(method, 'green')
                label = labels.get(method, method)

                ax.plot(iterations, means, label=label, color=color, linewidth=2)
                ax.fill_between(
                    iterations,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2,
                    color=color
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Sample Efficiency: Performance vs Iterations")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_main_table(
        self,
        results: Dict[str, List[Dict]]
    ) -> str:
        """
        Generate main results table for UAI paper.
        """
        rows = []

        method_names = {
            'no_evolution': 'Static Prior',
            'random_mutation': 'Random Search',
            'llm_single_shot': 'Single Query',
            'llm_iterative': 'LLM-BO (Ours)',
        }

        for method, method_results in results.items():
            if not method_results:
                continue

            # Aggregate metrics
            test_sharpes = []
            test_pnls = []
            ecEs = []
            regrets = []
            gen_gaps = []

            for result in method_results:
                test_sharpes.append(result.get("mean_test_sharpe", 0))
                test_pnls.append(result.get("total_test_pnl", 0))

                # Get uncertainty metrics if available
                uncertainty = result.get("uncertainty_metrics", {})
                if uncertainty:
                    ecEs.append(uncertainty.get("expected_calibration_error", 0))
                    regrets.append(uncertainty.get("cumulative_regret", 0))
                    gen_gaps.append(uncertainty.get("generalization_gap", 0))

            # Calculate statistics with CIs
            sharpe_mean, sharpe_ci_l, sharpe_ci_h = self.metrics_calc.bootstrap_confidence_interval(test_sharpes)
            pnl_mean = np.mean(test_pnls)

            name = method_names.get(method, method)

            row = {
                "Method": name,
                "Sharpe": f"{sharpe_mean:.3f}",
                "95% CI": f"[{sharpe_ci_l:.3f}, {sharpe_ci_h:.3f}]",
                "Total PnL": f"{pnl_mean:.0f}",
            }

            if ecEs:
                row["ECE"] = f"{np.mean(ecEs):.3f}"
            if regrets:
                row["Regret"] = f"{np.mean(regrets):.1f}"
            if gen_gaps:
                row["Gen. Gap"] = f"{np.mean(gen_gaps):.3f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Generate LaTeX
        latex = df.to_latex(index=False, escape=False)

        return f"""
\\begin{{table}}[t]
\\centering
\\caption{{Main Results: LLM-Guided Bayesian Optimization vs Baselines}}
\\label{{tab:main}}
{latex}
\\vspace{{-2mm}}
\\end{{table}}
"""

    def generate_paper_statistics(
        self,
        results: Dict[str, List[Dict]]
    ) -> UAIPaperResults:
        """
        Generate all statistics needed for paper claims.
        """
        paper = UAIPaperResults()

        llm_results = results.get("llm_iterative", [])
        random_results = results.get("random_mutation", [])

        if llm_results and random_results:
            llm_sharpes = [r.get("mean_test_sharpe", 0) for r in llm_results]
            random_sharpes = [r.get("mean_test_sharpe", 0) for r in random_results]

            # Improvement
            llm_mean = np.mean(llm_sharpes)
            random_mean = np.mean(random_sharpes)

            if random_mean != 0:
                paper.llm_vs_random_sharpe_improvement = (llm_mean - random_mean) / abs(random_mean) * 100
            else:
                paper.llm_vs_random_sharpe_improvement = 100 if llm_mean > 0 else 0

            # Statistical test
            t_stat, p_val, sig = self.metrics_calc.paired_t_test(random_sharpes, llm_sharpes)
            paper.llm_vs_random_p_value = p_val

            # Effect size
            paper.llm_vs_random_effect_size = self.metrics_calc.effect_size_cohens_d(
                random_sharpes, llm_sharpes
            )

            # Sample efficiency (from uncertainty metrics)
            llm_convergences = []
            random_convergences = []

            for r in llm_results:
                um = r.get("uncertainty_metrics", {})
                if "iterations_to_convergence" in um:
                    llm_convergences.append(um["iterations_to_convergence"])

            for r in random_results:
                um = r.get("uncertainty_metrics", {})
                if "iterations_to_convergence" in um:
                    random_convergences.append(um["iterations_to_convergence"])

            if llm_convergences and random_convergences:
                paper.llm_iterations_to_converge = np.mean(llm_convergences)
                paper.random_iterations_to_converge = np.mean(random_convergences)
                if paper.llm_iterations_to_converge > 0:
                    paper.efficiency_ratio = paper.random_iterations_to_converge / paper.llm_iterations_to_converge

        return paper

    def generate_all_outputs(
        self,
        output_dir: str = "experiments/paper_outputs"
    ) -> Dict[str, str]:
        """
        Generate all outputs needed for UAI paper.
        """
        os.makedirs(output_dir, exist_ok=True)

        results = self.load_results()

        outputs = {}

        # Main table
        table = self.generate_main_table(results)
        table_path = os.path.join(output_dir, "main_table.tex")
        with open(table_path, "w") as f:
            f.write(table)
        outputs["main_table"] = table_path

        # Plots
        outputs["calibration_plot"] = self.generate_calibration_plot(
            results, os.path.join(output_dir, "calibration.pdf")
        )
        outputs["regret_plot"] = self.generate_regret_plot(
            results, os.path.join(output_dir, "regret.pdf")
        )
        outputs["efficiency_plot"] = self.generate_sample_efficiency_plot(
            results, os.path.join(output_dir, "efficiency.pdf")
        )

        # Paper statistics
        stats = self.generate_paper_statistics(results)
        stats_path = os.path.join(output_dir, "paper_claims.tex")
        with open(stats_path, "w") as f:
            f.write(stats.to_latex_claims())
        outputs["claims"] = stats_path

        # Summary JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "sharpe_improvement": stats.llm_vs_random_sharpe_improvement,
                "p_value": stats.llm_vs_random_p_value,
                "effect_size": stats.llm_vs_random_effect_size,
                "efficiency_ratio": stats.efficiency_ratio,
            }, f, indent=2)
        outputs["summary"] = summary_path

        print(f"Generated all outputs in {output_dir}/")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

        return outputs


if __name__ == "__main__":
    analyzer = UAIAnalyzer()
    analyzer.generate_all_outputs()
