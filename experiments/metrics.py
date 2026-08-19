"""
Metrics Calculator
==================

Computes all metrics needed for research paper:
- Trading performance metrics (Sharpe, Sortino, win rate, etc.)
- Evolution metrics (improvement rate, convergence)
- Statistical significance tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Per-symbol breakdown
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "largest_win": round(self.largest_win, 4),
            "largest_loss": round(self.largest_loss, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "pnl_by_symbol": {k: round(v, 2) for k, v in self.pnl_by_symbol.items()},
        }


@dataclass
class EvolutionMetrics:
    """Metrics tracking evolution progress"""
    n_iterations: int = 0
    improvement_history: List[float] = field(default_factory=list)
    best_iteration: int = 0
    best_sharpe: float = 0.0
    convergence_iteration: int = 0  # When improvement stopped
    total_strategies_generated: int = 0
    valid_strategies: int = 0
    code_changes: List[str] = field(default_factory=list)  # Track what changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_iterations": self.n_iterations,
            "improvement_history": [round(x, 4) for x in self.improvement_history],
            "best_iteration": self.best_iteration,
            "best_sharpe": round(self.best_sharpe, 4),
            "convergence_iteration": self.convergence_iteration,
            "total_strategies_generated": self.total_strategies_generated,
            "valid_strategies": self.valid_strategies,
        }


class MetricsCalculator:
    """
    Calculates all metrics needed for the research paper.
    """

    @staticmethod
    def calculate_trading_metrics(
        trades: List[Dict[str, Any]],
        equity_curve: Optional[List[float]] = None,
        initial_capital: float = 10000.0,
        annualization_factor: float = 252 * 96  # 15-min bars per year
    ) -> TradingMetrics:
        """
        Calculate comprehensive trading metrics from trade list.

        Args:
            trades: List of trade dictionaries with 'pnl', 'symbol', etc.
            equity_curve: Optional list of equity values over time
            initial_capital: Starting capital
            annualization_factor: Factor to annualize returns

        Returns:
            TradingMetrics dataclass
        """
        metrics = TradingMetrics()

        if not trades:
            return metrics

        # Basic counts
        pnls = [t.get("pnl", 0) for t in trades]
        metrics.total_trades = len(trades)
        metrics.winning_trades = sum(1 for p in pnls if p > 0)
        metrics.losing_trades = sum(1 for p in pnls if p < 0)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # PnL metrics
        metrics.total_pnl = sum(pnls)
        metrics.gross_profit = sum(p for p in pnls if p > 0)
        metrics.gross_loss = abs(sum(p for p in pnls if p < 0))
        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss
            if metrics.gross_loss > 0 else float('inf') if metrics.gross_profit > 0 else 0
        )

        # Average metrics
        metrics.avg_trade_pnl = np.mean(pnls) if pnls else 0
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        metrics.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        metrics.largest_win = max(pnls) if pnls else 0
        metrics.largest_loss = min(pnls) if pnls else 0

        # Per-symbol breakdown
        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            pnl = trade.get("pnl", 0)
            metrics.pnl_by_symbol[symbol] = metrics.pnl_by_symbol.get(symbol, 0) + pnl

        # Risk-adjusted metrics (need equity curve or returns)
        if equity_curve and len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()

            # Sharpe Ratio
            if returns.std() > 0:
                metrics.sharpe_ratio = (
                    returns.mean() / returns.std() * np.sqrt(annualization_factor)
                )

            # Sortino Ratio (only downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                metrics.sortino_ratio = (
                    returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
                )

            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            metrics.max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

            # Calmar Ratio
            annual_return = returns.mean() * annualization_factor
            if metrics.max_drawdown > 0:
                metrics.calmar_ratio = annual_return / metrics.max_drawdown

            # Max Drawdown Duration
            in_drawdown = drawdown < 0
            if in_drawdown.any():
                drawdown_periods = (~in_drawdown).cumsum()
                drawdown_durations = in_drawdown.groupby(drawdown_periods).sum()
                metrics.max_drawdown_duration = int(drawdown_durations.max())

        return metrics

    @staticmethod
    def calculate_from_backtest_report(report: Dict[str, Any]) -> TradingMetrics:
        """
        Calculate metrics from a backtest report dictionary.

        This adapts to the format returned by MultiAssetSimulator.
        """
        metrics = TradingMetrics()

        if "error" in report:
            return metrics

        metrics.total_trades = report.get("total_trades", 0)
        metrics.winning_trades = report.get("winning_trades", 0)
        metrics.losing_trades = report.get("losing_trades", 0)
        metrics.win_rate = report.get("win_rate", 0)
        metrics.total_pnl = report.get("total_pnl", 0)
        metrics.profit_factor = report.get("profit_factor", 0)
        metrics.avg_trade_pnl = report.get("avg_trade", 0)
        metrics.largest_win = report.get("best_trade", 0)
        metrics.largest_loss = report.get("worst_trade", 0)
        metrics.pnl_by_symbol = report.get("pnl_by_symbol", {})
        metrics.sharpe_ratio = report.get("sharpe_ratio", 0)
        metrics.max_drawdown = report.get("max_drawdown", 0)

        return metrics

    @staticmethod
    def paired_t_test(
        baseline_results: List[float],
        treatment_results: List[float]
    ) -> Tuple[float, float, str]:
        """
        Perform paired t-test to compare two methods.

        Args:
            baseline_results: List of metric values for baseline
            treatment_results: List of metric values for treatment

        Returns:
            (t_statistic, p_value, significance_level)
        """
        if len(baseline_results) != len(treatment_results):
            raise ValueError("Result lists must have same length")

        if len(baseline_results) < 2:
            return 0.0, 1.0, "insufficient_data"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_value = stats.ttest_rel(treatment_results, baseline_results)

        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"

        return float(t_stat), float(p_value), significance

    @staticmethod
    def wilcoxon_test(
        baseline_results: List[float],
        treatment_results: List[float]
    ) -> Tuple[float, float, str]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative).

        More robust when data is not normally distributed.
        """
        if len(baseline_results) != len(treatment_results):
            raise ValueError("Result lists must have same length")

        if len(baseline_results) < 5:
            return 0.0, 1.0, "insufficient_data"

        differences = np.array(treatment_results) - np.array(baseline_results)
        if np.all(differences == 0):
            return 0.0, 1.0, "no_difference"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p_value = stats.wilcoxon(differences)
            except ValueError:
                return 0.0, 1.0, "test_failed"

        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"

        return float(stat), float(p_value), significance

    @staticmethod
    def effect_size_cohens_d(
        baseline_results: List[float],
        treatment_results: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        """
        if len(baseline_results) < 2 or len(treatment_results) < 2:
            return 0.0

        mean_diff = np.mean(treatment_results) - np.mean(baseline_results)
        pooled_std = np.sqrt(
            (np.var(baseline_results, ddof=1) + np.var(treatment_results, ddof=1)) / 2
        )

        if pooled_std == 0:
            return 0.0

        return mean_diff / pooled_std

    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        statistic: str = "mean"
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            data: List of values
            confidence: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            statistic: "mean" or "median"

        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        if len(data) < 2:
            val = data[0] if data else 0.0
            return val, val, val

        data = np.array(data)
        stat_func = np.mean if statistic == "mean" else np.median

        # Bootstrap sampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(stat_func(sample))

        # Calculate confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        point_estimate = stat_func(data)

        return float(point_estimate), float(ci_lower), float(ci_upper)

    @staticmethod
    def summarize_experiment_results(
        results: List[Dict[str, Any]],
        metric_key: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Summarize results across multiple experiment runs.

        Args:
            results: List of result dictionaries
            metric_key: Which metric to summarize

        Returns:
            Summary statistics dictionary
        """
        values = [r.get(metric_key, 0) for r in results if metric_key in r]

        if not values:
            return {"error": f"No values found for {metric_key}"}

        mean, ci_lower, ci_upper = MetricsCalculator.bootstrap_confidence_interval(values)

        return {
            "metric": metric_key,
            "n_samples": len(values),
            "mean": round(mean, 4),
            "std": round(np.std(values), 4),
            "median": round(np.median(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
        }
