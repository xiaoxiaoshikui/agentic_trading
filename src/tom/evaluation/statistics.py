from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TTestResult:
    t_stat: float
    p_value: float
    n: int


def paired_t_test(a: List[float], b: List[float]) -> TTestResult:
    """
    Simple paired t-test using normal approximation for p-value.
    For exact p-values, use scipy in the experiment framework.
    """
    if len(a) != len(b) or len(a) < 2:
        return TTestResult(t_stat=0.0, p_value=1.0, n=min(len(a), len(b)))

    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return TTestResult(t_stat=0.0, p_value=1.0, n=n)

    t_stat = mean / (std / math.sqrt(n))
    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return TTestResult(t_stat=t_stat, p_value=p_value, n=n)


def effect_size_cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
