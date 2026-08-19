"""
Experiment Configuration
========================

Defines all configuration parameters for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import hashlib


class EvolutionMethod(str, Enum):
    """Evolution method types for comparison"""
    LLM_ITERATIVE = "llm_iterative"      # Our method: LLM iterates based on feedback
    LLM_SINGLE_SHOT = "llm_single_shot"  # Baseline: LLM writes once, no iteration
    RANDOM_MUTATION = "random_mutation"   # Baseline: Random parameter changes
    GENETIC_PROGRAMMING = "genetic_prog"  # Baseline: GP-based evolution
    NO_EVOLUTION = "no_evolution"         # Baseline: Static strategy, no changes


class ModelType(str, Enum):
    """LLM model types"""
    QWEN_7B = "qwen2.5-coder:7b"
    QWEN_14B = "qwen2.5-coder:14b"
    QWEN_32B = "qwen2.5-coder:32b"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"
    DEEPSEEK_R1 = "deepseek-r1:14b"


@dataclass
class ModelConfig:
    """LLM Model configuration"""
    model_type: ModelType = ModelType.QWEN_7B
    host: str = "http://localhost:11434"  # Ollama host
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None  # For OpenAI models
    model_name: Optional[str] = None  # Optional override for model name
    base_url: Optional[str] = None  # Optional OpenAI-compatible base URL

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "model_type": self.model_type.value,
            "host": self.host,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model_name:
            data["model_name"] = self.model_name
        if self.base_url:
            data["base_url"] = self.base_url
        return data


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    max_positions: int = 3
    capital_per_position: float = 0.3
    max_long_exposure: float = 0.6
    max_short_exposure: float = 0.6
    max_daily_loss: float = 0.05
    max_single_asset_loss: float = 0.08
    min_confidence: float = 0.3
    atr_stop_multiplier: float = 2.0
    atr_profit_multiplier: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "max_positions": self.max_positions,
            "capital_per_position": self.capital_per_position,
            "max_long_exposure": self.max_long_exposure,
            "max_short_exposure": self.max_short_exposure,
            "max_daily_loss": self.max_daily_loss,
            "max_single_asset_loss": self.max_single_asset_loss,
            "min_confidence": self.min_confidence,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "atr_profit_multiplier": self.atr_profit_multiplier,
        }


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration

    This config fully specifies a reproducible experiment run.
    """
    # Experiment identification
    name: str = "llm_evolution_experiment"
    description: str = ""

    # Random seed for reproducibility
    seed: int = 42

    # Data configuration
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    interval: str = "15m"
    data_source: str = "binance"  # binance, csv, cached
    cache_dir: str = "experiments/data_cache"
    end_time: Optional[str] = None  # Optional fixed end time for reproducibility (ISO string or timestamp)

    # Walk-forward configuration
    n_periods: int = 10           # Number of walk-forward periods
    train_ratio: float = 0.7      # Train/test split ratio
    min_train_bars: int = 500     # Minimum bars for training
    min_test_bars: int = 200      # Minimum bars for testing

    # Evolution configuration
    method: EvolutionMethod = EvolutionMethod.LLM_ITERATIVE
    n_iterations: int = 10        # Evolution iterations per period
    early_stop_patience: int = 3  # Stop if no improvement for N iterations

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Backtest configuration
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Output configuration
    output_dir: str = "experiments/results"
    save_strategies: bool = True   # Save all generated strategy code
    save_checkpoints: bool = True  # Save intermediate results
    verbose: bool = True

    def get_experiment_id(self) -> str:
        """Generate unique experiment ID based on config hash"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "symbols": self.symbols,
            "interval": self.interval,
            "data_source": self.data_source,
            "end_time": self.end_time,
            "n_periods": self.n_periods,
            "train_ratio": self.train_ratio,
            "min_train_bars": self.min_train_bars,
            "min_test_bars": self.min_test_bars,
            "method": self.method.value,
            "n_iterations": self.n_iterations,
            "early_stop_patience": self.early_stop_patience,
            "model": self.model.to_dict(),
            "backtest": self.backtest.to_dict(),
            "output_dir": self.output_dir,
            "save_strategies": self.save_strategies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Load config from dictionary"""
        model_data = data.pop("model", {})
        backtest_data = data.pop("backtest", {})

        if "method" in data:
            data["method"] = EvolutionMethod(data["method"])

        if "model_type" in model_data:
            try:
                model_data["model_type"] = ModelType(model_data["model_type"])
            except ValueError:
                model_data["model_name"] = model_data["model_type"]
                model_data["model_type"] = ModelType.QWEN_7B

        return cls(
            model=ModelConfig(**model_data),
            backtest=BacktestConfig(**backtest_data),
            **data
        )

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# Pre-defined experiment configurations for paper
EXPERIMENT_CONFIGS = {
    "main_llm_qwen7b": ExperimentConfig(
        name="main_llm_qwen7b",
        description="Main experiment: LLM iterative evolution with Qwen-7B",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=20,
        n_iterations=10,
    ),

    "main_llm_gpt4o_mini": ExperimentConfig(
        name="main_llm_gpt4o_mini",
        description="Main experiment: LLM iterative evolution with GPT-4o-mini",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.GPT4O_MINI),
        n_periods=20,
        n_iterations=10,
    ),

    "baseline_no_evolution": ExperimentConfig(
        name="baseline_no_evolution",
        description="Baseline: Static EMA strategy without evolution",
        method=EvolutionMethod.NO_EVOLUTION,
        n_periods=20,
        n_iterations=1,
    ),

    "baseline_single_shot": ExperimentConfig(
        name="baseline_single_shot",
        description="Baseline: LLM generates strategy once, no iteration",
        method=EvolutionMethod.LLM_SINGLE_SHOT,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=20,
        n_iterations=1,
    ),

    "baseline_random_mutation": ExperimentConfig(
        name="baseline_random_mutation",
        description="Baseline: Random parameter mutation",
        method=EvolutionMethod.RANDOM_MUTATION,
        n_periods=20,
        n_iterations=10,
    ),

    "ablation_iterations_5": ExperimentConfig(
        name="ablation_iterations_5",
        description="Ablation: Effect of iteration count (5 iterations)",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=20,
        n_iterations=5,
    ),

    "ablation_iterations_20": ExperimentConfig(
        name="ablation_iterations_20",
        description="Ablation: Effect of iteration count (20 iterations)",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=20,
        n_iterations=20,
    ),

    # ===========================================
    # UAI Paper Experiments
    # ===========================================

    "uai_main_llm": ExperimentConfig(
        name="uai_main_llm",
        description="UAI Main: LLM as proposal distribution in strategy space",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=20,
        n_iterations=15,  # More iterations for regret analysis
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ),

    "uai_baseline_random": ExperimentConfig(
        name="uai_baseline_random",
        description="UAI Baseline: Random search in strategy space",
        method=EvolutionMethod.RANDOM_MUTATION,
        n_periods=20,
        n_iterations=15,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ),

    "uai_baseline_static": ExperimentConfig(
        name="uai_baseline_static",
        description="UAI Baseline: No optimization (static prior)",
        method=EvolutionMethod.NO_EVOLUTION,
        n_periods=20,
        n_iterations=1,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ),

    "uai_ablation_samples": ExperimentConfig(
        name="uai_ablation_samples",
        description="UAI Ablation: Sample efficiency analysis (30 iterations)",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.QWEN_7B),
        n_periods=10,
        n_iterations=30,  # Many iterations to study convergence
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    ),

    "uai_model_comparison": ExperimentConfig(
        name="uai_model_comparison",
        description="UAI: Compare proposal quality across models",
        method=EvolutionMethod.LLM_ITERATIVE,
        model=ModelConfig(model_type=ModelType.GPT4O_MINI),
        n_periods=20,
        n_iterations=15,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ),
}
