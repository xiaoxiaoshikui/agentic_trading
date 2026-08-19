"""
Metrics for multimodal prediction and downstream trading evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
from scipy.stats import rankdata


@dataclass
class ClassificationMetrics:
    n_samples: int = 0
    positive_rate: float = 0.0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    auc: float = 0.5
    brier: float = 0.0
    ece: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DownstreamMetrics:
    n_samples: int = 0
    n_trades: int = 0
    trade_rate: float = 0.0
    hit_rate: float = 0.0
    mean_return: float = 0.0
    total_return: float = 0.0
    sharpe_like: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    cost_bps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.clip(np.asarray(y_prob, dtype=float).reshape(-1), 0.0, 1.0)
    if y_true.size == 0:
        return ClassificationMetrics()

    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = _macro_f1(y_true, y_pred)
    auc = _binary_auc(y_true, y_prob)
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = _expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return ClassificationMetrics(
        n_samples=int(y_true.size),
        positive_rate=float(y_true.mean()),
        accuracy=accuracy,
        macro_f1=macro_f1,
        auc=auc,
        brier=brier,
        ece=ece,
    )


def compute_downstream_metrics(
    future_returns: np.ndarray,
    y_prob: np.ndarray,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    cost_bps: float = 0.0,
) -> DownstreamMetrics:
    future_returns = np.asarray(future_returns, dtype=float).reshape(-1)
    y_prob = np.clip(np.asarray(y_prob, dtype=float).reshape(-1), 0.0, 1.0)
    if future_returns.size == 0:
        return DownstreamMetrics(long_threshold=long_threshold, short_threshold=short_threshold, cost_bps=cost_bps)

    positions = np.zeros_like(y_prob)
    positions[y_prob >= long_threshold] = 1.0
    positions[y_prob <= short_threshold] = -1.0

    cost = max(float(cost_bps), 0.0) * 1e-4
    pnl = positions * future_returns - np.abs(positions) * cost
    nonzero = positions != 0.0
    n_trades = int(np.sum(nonzero))
    trade_rate = float(np.mean(nonzero))
    hit_rate = float(np.mean(pnl[nonzero] > 0)) if n_trades > 0 else 0.0
    total_return = float(np.sum(pnl))
    mean_return = float(np.mean(pnl))
    pnl_std = float(np.std(pnl))
    sharpe_like = float((mean_return / pnl_std) * np.sqrt(len(pnl))) if pnl_std > 1e-12 else 0.0

    equity = np.cumprod(1.0 + pnl)
    peaks = np.maximum.accumulate(equity) if equity.size else np.array([1.0])
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    turnover = float(np.mean(np.abs(np.diff(np.concatenate([[0.0], positions])))))

    return DownstreamMetrics(
        n_samples=int(len(future_returns)),
        n_trades=n_trades,
        trade_rate=trade_rate,
        hit_rate=hit_rate,
        mean_return=mean_return,
        total_return=total_return,
        sharpe_like=sharpe_like,
        max_drawdown=max_drawdown,
        turnover=turnover,
        long_threshold=float(long_threshold),
        short_threshold=float(short_threshold),
        cost_bps=float(cost_bps),
    )


def compute_position_downstream_metrics(
    future_returns: np.ndarray,
    positions: np.ndarray,
    min_abs_position: float = 0.05,
    cost_bps: float = 0.0,
) -> DownstreamMetrics:
    """Evaluate downstream trading from model-emitted continuous positions."""
    future_returns = np.asarray(future_returns, dtype=float).reshape(-1)
    positions = np.clip(np.asarray(positions, dtype=float).reshape(-1), -1.0, 1.0)
    min_abs_position = max(float(min_abs_position), 0.0)
    if future_returns.size == 0:
        return DownstreamMetrics(
            long_threshold=min_abs_position,
            short_threshold=-min_abs_position,
            cost_bps=cost_bps,
        )
    if future_returns.shape[0] != positions.shape[0]:
        raise ValueError(
            f"future_returns and positions must have same length, got "
            f"{future_returns.shape[0]} and {positions.shape[0]}"
        )

    effective_positions = np.where(np.abs(positions) >= min_abs_position, positions, 0.0)
    cost = max(float(cost_bps), 0.0) * 1e-4
    pnl = effective_positions * future_returns - np.abs(effective_positions) * cost
    nonzero = effective_positions != 0.0
    n_trades = int(np.sum(nonzero))
    trade_rate = float(np.mean(nonzero))
    hit_rate = float(np.mean(pnl[nonzero] > 0)) if n_trades > 0 else 0.0
    total_return = float(np.sum(pnl))
    mean_return = float(np.mean(pnl))
    pnl_std = float(np.std(pnl))
    sharpe_like = float((mean_return / pnl_std) * np.sqrt(len(pnl))) if pnl_std > 1e-12 else 0.0

    equity = np.cumprod(1.0 + pnl)
    peaks = np.maximum.accumulate(equity) if equity.size else np.array([1.0])
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    turnover = float(np.mean(np.abs(np.diff(np.concatenate([[0.0], effective_positions])))))

    return DownstreamMetrics(
        n_samples=int(len(future_returns)),
        n_trades=n_trades,
        trade_rate=trade_rate,
        hit_rate=hit_rate,
        mean_return=mean_return,
        total_return=total_return,
        sharpe_like=sharpe_like,
        max_drawdown=max_drawdown,
        turnover=turnover,
        long_threshold=min_abs_position,
        short_threshold=-min_abs_position,
        cost_bps=float(cost_bps),
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_scores = []
    for label in (0, 1):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def _binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(y_prob)
    rank_sum = np.sum(ranks[y_true == 1])
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    error = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        error += (np.sum(mask) / total) * abs(acc - conf)
    return float(error)
