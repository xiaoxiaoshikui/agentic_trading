"""
Uncertainty Quantification Module
=================================

UAI-focused metrics and analysis for LLM Strategy Evolution.

Key concepts:
1. LLM as proposal distribution in strategy space
2. Backtest as noisy oracle with non-stationary rewards
3. Calibration of confidence scores
4. Regret analysis over iterations
5. Sample efficiency comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.special import softmax
import warnings


@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics for UAI paper"""

    # Calibration metrics
    expected_calibration_error: float = 0.0  # ECE
    maximum_calibration_error: float = 0.0   # MCE
    brier_score: float = 0.0

    # Regret metrics
    cumulative_regret: float = 0.0
    simple_regret: float = 0.0  # Gap to best found
    normalized_regret: float = 0.0

    # Sample efficiency
    iterations_to_convergence: int = 0
    area_under_learning_curve: float = 0.0

    # Distribution shift
    train_test_divergence: float = 0.0  # KL divergence
    generalization_gap: float = 0.0

    # Exploration metrics
    strategy_diversity: float = 0.0  # Entropy of generated strategies
    exploitation_ratio: float = 0.0  # How often best strategy is refined

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "maximum_calibration_error": round(self.maximum_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "cumulative_regret": round(self.cumulative_regret, 4),
            "simple_regret": round(self.simple_regret, 4),
            "normalized_regret": round(self.normalized_regret, 4),
            "iterations_to_convergence": self.iterations_to_convergence,
            "area_under_learning_curve": round(self.area_under_learning_curve, 4),
            "train_test_divergence": round(self.train_test_divergence, 4),
            "generalization_gap": round(self.generalization_gap, 4),
            "strategy_diversity": round(self.strategy_diversity, 4),
            "exploitation_ratio": round(self.exploitation_ratio, 4),
        }


class UncertaintyAnalyzer:
    """
    Uncertainty analysis for LLM Strategy Evolution.

    Provides metrics needed for UAI paper:
    - Calibration analysis
    - Regret analysis
    - Sample efficiency
    - Distribution shift detection
    """

    @staticmethod
    def compute_calibration_error(
        confidences: List[float],
        outcomes: List[int],  # 1 = profitable, 0 = loss
        n_bins: int = 10
    ) -> Tuple[float, float, Dict]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

        For trading: confidence = strategy's predicted confidence
                    outcome = 1 if trade was profitable, 0 otherwise

        Args:
            confidences: List of confidence scores [0, 1]
            outcomes: List of binary outcomes
            n_bins: Number of bins for calibration

        Returns:
            (ECE, MCE, calibration_data)
        """
        if len(confidences) != len(outcomes) or len(confidences) == 0:
            return 0.0, 0.0, {}

        confidences = np.array(confidences)
        outcomes = np.array(outcomes)

        # Bin confidences
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

        ece = 0.0
        mce = 0.0
        calibration_data = {"bins": [], "accuracy": [], "confidence": [], "count": []}

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_confidence = confidences[mask].mean()
            bin_accuracy = outcomes[mask].mean()
            bin_count = mask.sum()

            gap = abs(bin_accuracy - bin_confidence)
            ece += (bin_count / len(confidences)) * gap
            mce = max(mce, gap)

            calibration_data["bins"].append(i)
            calibration_data["accuracy"].append(float(bin_accuracy))
            calibration_data["confidence"].append(float(bin_confidence))
            calibration_data["count"].append(int(bin_count))

        return float(ece), float(mce), calibration_data

    @staticmethod
    def compute_brier_score(
        confidences: List[float],
        outcomes: List[int]
    ) -> float:
        """
        Compute Brier score (proper scoring rule).

        Lower is better. Measures calibration + discrimination.
        """
        if len(confidences) != len(outcomes) or len(confidences) == 0:
            return 1.0

        confidences = np.array(confidences)
        outcomes = np.array(outcomes)

        return float(np.mean((confidences - outcomes) ** 2))

    @staticmethod
    def compute_regret(
        performance_history: List[float],
        oracle_performance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute regret metrics over optimization trajectory.

        Args:
            performance_history: Performance at each iteration
            oracle_performance: Best achievable performance (if known)

        Returns:
            Dictionary with regret metrics
        """
        if not performance_history:
            return {"cumulative_regret": 0, "simple_regret": 0, "normalized_regret": 0}

        performances = np.array(performance_history)

        # Use best observed as proxy for oracle if not provided
        if oracle_performance is None:
            oracle_performance = performances.max()

        # Cumulative regret: sum of gaps to oracle
        instantaneous_regret = oracle_performance - performances
        cumulative_regret = float(np.sum(np.maximum(0, instantaneous_regret)))

        # Simple regret: gap between best found and oracle
        simple_regret = float(max(0, oracle_performance - performances.max()))

        # Normalized regret
        perf_range = performances.max() - performances.min()
        if perf_range > 0:
            normalized_regret = simple_regret / perf_range
        else:
            normalized_regret = 0.0

        return {
            "cumulative_regret": cumulative_regret,
            "simple_regret": simple_regret,
            "normalized_regret": normalized_regret,
            "regret_curve": instantaneous_regret.tolist(),
        }

    @staticmethod
    def compute_sample_efficiency(
        performance_history: List[float],
        threshold_ratio: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute sample efficiency metrics.

        Args:
            performance_history: Performance at each iteration
            threshold_ratio: Fraction of best performance to consider "converged"

        Returns:
            Dictionary with sample efficiency metrics
        """
        if not performance_history:
            return {"iterations_to_convergence": 0, "auc": 0}

        performances = np.array(performance_history)
        best_performance = performances.max()
        threshold = threshold_ratio * best_performance if best_performance > 0 else 0

        # Iterations to convergence
        convergence_idx = np.where(performances >= threshold)[0]
        if len(convergence_idx) > 0:
            iterations_to_convergence = int(convergence_idx[0]) + 1
        else:
            iterations_to_convergence = len(performances)

        # Area under learning curve (normalized)
        auc = float(np.trapz(performances) / len(performances))

        # Normalized AUC (relative to best possible)
        max_auc = best_performance
        normalized_auc = auc / max_auc if max_auc > 0 else 0

        return {
            "iterations_to_convergence": iterations_to_convergence,
            "area_under_curve": auc,
            "normalized_auc": normalized_auc,
        }

    @staticmethod
    def compute_distribution_shift(
        train_returns: List[float],
        test_returns: List[float]
    ) -> Dict[str, float]:
        """
        Measure distribution shift between train and test.

        Uses KL divergence approximation via histograms.
        """
        if not train_returns or not test_returns:
            return {"kl_divergence": 0, "wasserstein_distance": 0}

        train = np.array(train_returns)
        test = np.array(test_returns)

        # Histogram-based KL divergence
        n_bins = min(20, len(train) // 5, len(test) // 5)
        n_bins = max(5, n_bins)

        # Common bin edges
        all_data = np.concatenate([train, test])
        bin_edges = np.histogram_bin_edges(all_data, bins=n_bins)

        train_hist, _ = np.histogram(train, bins=bin_edges, density=True)
        test_hist, _ = np.histogram(test, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        train_hist = train_hist + eps
        test_hist = test_hist + eps

        # Normalize
        train_hist = train_hist / train_hist.sum()
        test_hist = test_hist / test_hist.sum()

        # KL divergence: KL(test || train)
        kl_div = float(np.sum(test_hist * np.log(test_hist / train_hist)))

        # Wasserstein distance (Earth Mover's Distance)
        wasserstein = float(stats.wasserstein_distance(train, test))

        # Generalization gap (simple metric)
        gen_gap = abs(np.mean(train) - np.mean(test))

        return {
            "kl_divergence": max(0, kl_div),
            "wasserstein_distance": wasserstein,
            "generalization_gap": float(gen_gap),
            "train_mean": float(np.mean(train)),
            "test_mean": float(np.mean(test)),
            "train_std": float(np.std(train)),
            "test_std": float(np.std(test)),
        }

    @staticmethod
    def compute_strategy_diversity(
        strategy_codes: List[str]
    ) -> Dict[str, float]:
        """
        Measure diversity of generated strategies.

        Uses edit distance and structural similarity.
        """
        if not strategy_codes or len(strategy_codes) < 2:
            return {"diversity": 0, "unique_ratio": 1.0}

        # Simple diversity: ratio of unique strategies
        unique_strategies = len(set(strategy_codes))
        unique_ratio = unique_strategies / len(strategy_codes)

        # Average pairwise edit distance (normalized)
        def normalized_edit_distance(s1: str, s2: str) -> float:
            """Levenshtein distance normalized by max length"""
            if not s1 or not s2:
                return 1.0

            m, n = len(s1), len(s2)
            if m > n:
                s1, s2 = s2, s1
                m, n = n, m

            # Use simplified comparison for long strings
            if n > 1000:
                # Compare hashes of chunks
                chunk_size = 100
                s1_chunks = set(s1[i:i+chunk_size] for i in range(0, len(s1), chunk_size))
                s2_chunks = set(s2[i:i+chunk_size] for i in range(0, len(s2), chunk_size))
                jaccard = len(s1_chunks & s2_chunks) / len(s1_chunks | s2_chunks)
                return 1 - jaccard

            # Standard DP for short strings
            prev = list(range(n + 1))
            for i, c1 in enumerate(s1, 1):
                curr = [i] + [0] * n
                for j, c2 in enumerate(s2, 1):
                    curr[j] = min(
                        prev[j] + 1,
                        curr[j-1] + 1,
                        prev[j-1] + (c1 != c2)
                    )
                prev = curr

            return prev[n] / max(m, n)

        # Sample pairwise distances (for efficiency)
        n_samples = min(50, len(strategy_codes) * (len(strategy_codes) - 1) // 2)
        distances = []

        import random
        indices = list(range(len(strategy_codes)))

        for _ in range(n_samples):
            i, j = random.sample(indices, 2)
            dist = normalized_edit_distance(strategy_codes[i], strategy_codes[j])
            distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0

        return {
            "diversity": float(avg_distance),
            "unique_ratio": float(unique_ratio),
            "n_unique": unique_strategies,
            "n_total": len(strategy_codes),
        }

    @staticmethod
    def compute_exploration_exploitation(
        performance_history: List[float],
        was_best_refined: List[bool]
    ) -> Dict[str, float]:
        """
        Analyze exploration vs exploitation behavior.

        Args:
            performance_history: Performance at each iteration
            was_best_refined: Whether each iteration refined the current best

        Returns:
            Metrics about exploration/exploitation balance
        """
        if not performance_history or not was_best_refined:
            return {"exploitation_ratio": 0, "exploration_ratio": 0}

        exploitation_count = sum(was_best_refined)
        exploration_count = len(was_best_refined) - exploitation_count

        return {
            "exploitation_ratio": exploitation_count / len(was_best_refined),
            "exploration_ratio": exploration_count / len(was_best_refined),
            "total_iterations": len(was_best_refined),
        }

    @classmethod
    def compute_all_metrics(
        cls,
        confidences: List[float],
        outcomes: List[int],
        performance_history: List[float],
        train_returns: List[float],
        test_returns: List[float],
        strategy_codes: List[str],
    ) -> UncertaintyMetrics:
        """
        Compute all uncertainty metrics.
        """
        metrics = UncertaintyMetrics()

        # Calibration
        ece, mce, _ = cls.compute_calibration_error(confidences, outcomes)
        metrics.expected_calibration_error = ece
        metrics.maximum_calibration_error = mce
        metrics.brier_score = cls.compute_brier_score(confidences, outcomes)

        # Regret
        regret = cls.compute_regret(performance_history)
        metrics.cumulative_regret = regret["cumulative_regret"]
        metrics.simple_regret = regret["simple_regret"]
        metrics.normalized_regret = regret["normalized_regret"]

        # Sample efficiency
        efficiency = cls.compute_sample_efficiency(performance_history)
        metrics.iterations_to_convergence = efficiency["iterations_to_convergence"]
        metrics.area_under_learning_curve = efficiency.get("area_under_curve", 0)

        # Distribution shift
        shift = cls.compute_distribution_shift(train_returns, test_returns)
        metrics.train_test_divergence = shift["kl_divergence"]
        metrics.generalization_gap = shift["generalization_gap"]

        # Diversity
        diversity = cls.compute_strategy_diversity(strategy_codes)
        metrics.strategy_diversity = diversity["diversity"]

        return metrics


def compute_bayesian_posterior(
    prior_mean: float,
    prior_var: float,
    observations: List[float],
    noise_var: float
) -> Tuple[float, float]:
    """
    Compute Bayesian posterior for strategy performance.

    Assumes Gaussian prior and likelihood.

    Args:
        prior_mean: Prior mean for performance
        prior_var: Prior variance
        observations: Observed performances
        noise_var: Observation noise variance

    Returns:
        (posterior_mean, posterior_var)
    """
    if not observations:
        return prior_mean, prior_var

    n = len(observations)
    obs_mean = np.mean(observations)

    # Posterior precision = prior precision + n * likelihood precision
    prior_precision = 1 / prior_var if prior_var > 0 else 1e-6
    likelihood_precision = n / noise_var if noise_var > 0 else 1e-6

    posterior_precision = prior_precision + likelihood_precision
    posterior_var = 1 / posterior_precision

    # Posterior mean = weighted average
    posterior_mean = posterior_var * (
        prior_precision * prior_mean + likelihood_precision * obs_mean
    )

    return float(posterior_mean), float(posterior_var)


def thompson_sampling_score(
    mean: float,
    var: float,
    n_samples: int = 1000
) -> float:
    """
    Compute Thompson Sampling score (sample from posterior).

    For comparing strategies under uncertainty.
    """
    samples = np.random.normal(mean, np.sqrt(var), n_samples)
    return float(np.mean(samples))
