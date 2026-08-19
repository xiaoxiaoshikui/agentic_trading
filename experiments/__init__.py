"""
Experiment Framework for LLM Strategy Evolution Research
=========================================================

This framework provides:
- Rigorous walk-forward validation
- Multiple baseline comparisons
- Statistical analysis tools
- Reproducible experiment configuration
- Uncertainty quantification (UAI focus)
"""

from .config import ExperimentConfig, ModelConfig, BacktestConfig
from .metrics import MetricsCalculator

try:
    from .runner import ExperimentRunner
except Exception:  # pragma: no cover - optional import for light-weight tooling
    ExperimentRunner = None

try:
    from .uncertainty import UncertaintyAnalyzer, UncertaintyMetrics
except Exception:  # pragma: no cover - optional import for light-weight tooling
    UncertaintyAnalyzer = None
    UncertaintyMetrics = None

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "BacktestConfig",
    "ExperimentRunner",
    "MetricsCalculator",
    "UncertaintyAnalyzer",
    "UncertaintyMetrics",
]
