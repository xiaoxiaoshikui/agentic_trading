"""
Experiment Runner
=================

Main experiment runner that orchestrates:
1. Data loading and splitting
2. Strategy evolution (LLM or baseline)
3. Backtesting and evaluation
4. Results logging and checkpointing
"""

import os
import sys
import json
import logging
import time
import importlib.util
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import ExperimentConfig, EvolutionMethod, ModelType
from .data_loader import DataLoader, save_data_snapshot
from .metrics import MetricsCalculator, TradingMetrics, EvolutionMetrics
from .baselines import StaticBaseline, RandomMutationBaseline, SingleShotLLMBaseline
from .llm_bo import LLMBOEvolver

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """Results for a single walk-forward period"""
    period: int
    method: str

    # Evolution info
    n_iterations: int = 0
    evolved: bool = False
    best_iteration: int = 0

    # Training metrics
    train_trades: int = 0
    train_win_rate: float = 0.0
    train_pnl: float = 0.0
    train_sharpe: float = 0.0

    # Test metrics (out-of-sample)
    test_trades: int = 0
    test_win_rate: float = 0.0
    test_pnl: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0

    # Strategy info
    final_strategy_hash: str = ""
    performance_history: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Complete experiment results"""
    experiment_id: str
    config: Dict[str, Any]
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0

    # Results by period
    period_results: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate metrics
    mean_test_sharpe: float = 0.0
    std_test_sharpe: float = 0.0
    mean_test_pnl: float = 0.0
    total_test_pnl: float = 0.0
    mean_test_win_rate: float = 0.0

    # Comparison info
    improvement_over_baseline: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """
    Main experiment runner.

    Handles the complete experiment workflow:
    1. Load and split data (walk-forward)
    2. For each period:
       a. Run evolution iterations on train data
       b. Evaluate final strategy on test data
    3. Aggregate and save results
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = DataLoader(cache_dir=config.cache_dir)
        self.metrics_calc = MetricsCalculator()

        # Set random seed
        np.random.seed(config.seed)

        # Create output directory
        self.experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = os.path.join(config.output_dir, self.experiment_id)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "strategies"), exist_ok=True)

        # Initialize evolver based on method
        self.evolver = self._create_evolver()

        # Import backtesting components
        self._import_backtest_components()

        logger.info(f"Experiment initialized: {self.experiment_id}")
        logger.info(f"Method: {config.method.value}")
        logger.info(f"Output: {self.output_dir}")

    def _import_backtest_components(self):
        """Import backtesting components from main project"""
        try:
            from src.multi_asset_simulator import MultiAssetSimulator
            from src.advanced_risk import AdvancedRiskManager
            import ta
            self.MultiAssetSimulator = MultiAssetSimulator
            self.AdvancedRiskManager = AdvancedRiskManager
            self.ta = ta
        except ImportError as e:
            logger.error(f"Failed to import backtest components: {e}")
            raise

    def _create_evolver(self):
        """Create evolution method based on config"""
        method = self.config.method

        if method == EvolutionMethod.NO_EVOLUTION:
            return StaticBaseline(seed=self.config.seed)

        elif method == EvolutionMethod.RANDOM_MUTATION:
            return RandomMutationBaseline(seed=self.config.seed)

        elif method == EvolutionMethod.LLM_SINGLE_SHOT:
            model = self.config.model
            model_name = (
                os.getenv("DEEPSEEK_MODEL")
                or os.getenv("LLM_MODEL")
                or model.model_name
                or model.model_type.value
            )
            base_url = (
                model.base_url
                or os.getenv("DEEPSEEK_API_BASE")
                or os.getenv("LLM_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
            )
            api_key = (
                model.api_key
                or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            return SingleShotLLMBaseline(
                model=model_name,
                host=model.host,
                api_key=api_key,
                seed=self.config.seed,
                base_url=base_url
            )

        elif method == EvolutionMethod.LLM_ITERATIVE:
            return LLMBOEvolver(self.config.model, seed=self.config.seed)

        else:
            raise ValueError(f"Unknown evolution method: {method}")

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.

        Returns:
            ExperimentResult with all metrics and results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting experiment: {self.experiment_id}")
        logger.info("=" * 60)

        # Save config
        self.config.save(os.path.join(self.output_dir, "config.json"))

        # Load data
        logger.info("Loading data...")
        data = self.data_loader.load_data(
            symbols=self.config.symbols,
            interval=self.config.interval,
            end_time=self.config.end_time
        )

        if not data:
            raise ValueError("Failed to load data")

        # Save data snapshot for reproducibility
        save_data_snapshot(
            data,
            os.path.join(self.output_dir, "data_snapshot.json"),
            metadata={"config": self.config.to_dict()}
        )

        # Create walk-forward splits
        logger.info("Creating walk-forward splits...")
        periods = self.data_loader.create_walk_forward_splits(
            data=data,
            n_periods=self.config.n_periods,
            train_ratio=self.config.train_ratio,
            min_train_bars=self.config.min_train_bars,
            min_test_bars=self.config.min_test_bars
        )

        if not periods:
            raise ValueError("Failed to create walk-forward periods")

        logger.info(f"Created {len(periods)} walk-forward periods")

        # Run each period
        period_results = []

        for i, (train_data, test_data) in enumerate(periods):
            logger.info("-" * 40)
            logger.info(f"Period {i+1}/{len(periods)}")
            logger.info("-" * 40)

            try:
                result = self._run_period(i + 1, train_data, test_data)
                period_results.append(result.to_dict())

                # Save checkpoint
                if self.config.save_checkpoints:
                    self._save_checkpoint(period_results)

            except Exception as e:
                logger.error(f"Period {i+1} failed: {e}")
                period_results.append({
                    "period": i + 1,
                    "error": str(e)
                })

            # Reset evolver for next period
            if hasattr(self.evolver, 'reset'):
                self.evolver.reset()

        # Aggregate results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = ExperimentResult(
            experiment_id=self.experiment_id,
            config=self.config.to_dict(),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            period_results=period_results,
        )

        # Calculate aggregate metrics
        test_sharpes = [p.get("test_sharpe", 0) for p in period_results if "test_sharpe" in p]
        test_pnls = [p.get("test_pnl", 0) for p in period_results if "test_pnl" in p]
        test_win_rates = [p.get("test_win_rate", 0) for p in period_results if "test_win_rate" in p]

        if test_sharpes:
            result.mean_test_sharpe = float(np.mean(test_sharpes))
            result.std_test_sharpe = float(np.std(test_sharpes))
        if test_pnls:
            result.mean_test_pnl = float(np.mean(test_pnls))
            result.total_test_pnl = float(np.sum(test_pnls))
        if test_win_rates:
            result.mean_test_win_rate = float(np.mean(test_win_rates))

        # Save final results
        self._save_results(result)

        logger.info("=" * 60)
        logger.info(f"Experiment complete: {self.experiment_id}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Mean test Sharpe: {result.mean_test_sharpe:.4f}")
        logger.info(f"Total test PnL: {result.total_test_pnl:.2f}")
        logger.info("=" * 60)

        return result

    def _run_period(
        self,
        period_num: int,
        train_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame]
    ) -> PeriodResult:
        """
        Run a single walk-forward period.

        1. Evolve strategy on train data
        2. Evaluate on test data
        """
        result = PeriodResult(
            period=period_num,
            method=self.config.method.value
        )

        # Get initial strategy
        strategy_code = self.evolver.get_strategy_code() if hasattr(self.evolver, 'get_strategy_code') else None

        best_train_sharpe = float('-inf')
        best_strategy_code = strategy_code
        no_improvement_count = 0
        performance_history: List[float] = []

        # Evolution loop
        for iteration in range(self.config.n_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.config.n_iterations}")

            # Evolve strategy
            if hasattr(self.evolver, 'evolve'):
                # Build feedback from last iteration
                feedback = {
                    "iteration": iteration,
                    "train_sharpe": result.train_sharpe,
                    "train_pnl": result.train_pnl,
                    "train_win_rate": result.train_win_rate,
                }

                strategy_code, evolution_info = self.evolver.evolve(feedback, train_data)

                if evolution_info.get("evolved", False):
                    result.evolved = True

            # Backtest on training data
            train_metrics = self._backtest(strategy_code, train_data)

            result.train_trades = train_metrics.total_trades
            result.train_win_rate = train_metrics.win_rate
            result.train_pnl = train_metrics.total_pnl
            result.train_sharpe = train_metrics.sharpe_ratio
            performance_history.append(train_metrics.sharpe_ratio)

            logger.info(f"    Train: trades={train_metrics.total_trades}, "
                       f"win_rate={train_metrics.win_rate:.2%}, "
                       f"sharpe={train_metrics.sharpe_ratio:.4f}")

            # Track best strategy
            if train_metrics.sharpe_ratio > best_train_sharpe:
                best_train_sharpe = train_metrics.sharpe_ratio
                best_strategy_code = strategy_code
                result.best_iteration = iteration + 1
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= self.config.early_stop_patience:
                logger.info(f"    Early stopping at iteration {iteration + 1}")
                break

        result.n_iterations = iteration + 1
        result.performance_history = performance_history

        # Evaluate best strategy on test data (out-of-sample)
        logger.info("  Evaluating on test data (out-of-sample)...")
        test_metrics, test_details = self._backtest(best_strategy_code, test_data, return_details=True)

        result.test_trades = test_metrics.total_trades
        result.test_win_rate = test_metrics.win_rate
        result.test_pnl = test_metrics.total_pnl
        result.test_sharpe = test_metrics.sharpe_ratio
        result.test_max_drawdown = test_metrics.max_drawdown
        result.confidences = test_details.get("confidences", [])
        result.outcomes = test_details.get("outcomes", [])

        logger.info(f"  Test: trades={test_metrics.total_trades}, "
                   f"win_rate={test_metrics.win_rate:.2%}, "
                   f"sharpe={test_metrics.sharpe_ratio:.4f}, "
                   f"pnl={test_metrics.total_pnl:.2f}")

        # Save strategy
        if self.config.save_strategies and best_strategy_code:
            strategy_path = os.path.join(
                self.output_dir, "strategies", f"period_{period_num}.py"
            )
            with open(strategy_path, "w", encoding="utf-8") as f:
                f.write(best_strategy_code)

            # Create hash for tracking
            import hashlib
            result.final_strategy_hash = hashlib.md5(
                best_strategy_code.encode()
            ).hexdigest()[:8]

        return result

    def _backtest(
        self,
        strategy_code: str,
        data: Dict[str, pd.DataFrame],
        return_details: bool = False
    ) -> Any:
        """
        Run backtest on given data with strategy code.
        """
        empty_details = {"confidences": [], "outcomes": [], "trade_pnls": []}
        if not strategy_code:
            return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

        # Write strategy to temp file
        temp_file = os.path.join(self.output_dir, "temp_strategy.py")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(strategy_code)

            # Load strategy module
            spec = importlib.util.spec_from_file_location("temp_strategy", temp_file)
            if spec is None or spec.loader is None:
                return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "calculate_signal"):
                return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

            # Run simulation
            cfg = self.config.backtest
            simulator = self.MultiAssetSimulator(
                initial_capital=cfg.initial_capital,
                max_positions=cfg.max_positions,
                capital_per_position=cfg.capital_per_position,
                max_long_exposure=cfg.max_long_exposure,
                max_short_exposure=cfg.max_short_exposure,
                max_daily_loss=cfg.max_daily_loss,
                max_single_asset_loss=cfg.max_single_asset_loss,
            )

            risk_manager = self.AdvancedRiskManager()
            symbols = list(data.keys())

            if not symbols:
                return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

            # Align data
            min_time = max(df.index[0] for df in data.values())
            max_time = min(df.index[-1] for df in data.values())

            for symbol in data:
                data[symbol] = data[symbol][
                    (data[symbol].index >= min_time) &
                    (data[symbol].index <= max_time)
                ]

            if len(data[symbols[0]]) < 200:
                return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

            common_times = data[symbols[0]].index
            total_bars = len(common_times)

            # Simulation loop
            equity_curve = [cfg.initial_capital]
            for i in range(200, total_bars):
                current_time = common_times[i]
                current_ts = float(current_time.timestamp()) if hasattr(current_time, 'timestamp') else float(i)

                # Get current prices (use open to avoid lookahead)
                prices = {}
                for symbol, df in data.items():
                    if current_time in df.index:
                        prices[symbol] = float(df.loc[current_time, 'open'])

                # Check positions
                simulator.check_positions(prices, current_ts)

                # Generate signals for each symbol
                for symbol, df in data.items():
                    if not simulator.can_open_position(symbol):
                        continue

                    # Use only past bars for signal calculation
                    df_slice = df.iloc[:i].copy()
                    if len(df_slice) < 200:
                        continue

                    try:
                        signal = module.calculate_signal(df_slice)
                    except Exception:
                        continue

                    side = getattr(signal, 'side', 'FLAT')
                    confidence = float(getattr(signal, 'confidence', 0))

                    if side == "FLAT" or confidence < cfg.min_confidence:
                        continue

                    current_price = prices.get(symbol)
                    if not current_price:
                        continue

                    # Calculate ATR for position sizing
                    try:
                        atr_series = self.ta.volatility.average_true_range(
                            df_slice['high'], df_slice['low'], df_slice['close'], window=14
                        )
                        atr = float(atr_series.iloc[-1])
                        if not np.isfinite(atr) or atr <= 0:
                            atr = current_price * 0.01
                    except Exception:
                        atr = current_price * 0.01

                    # Calculate stop loss and take profit
                    if side == "LONG":
                        stop_loss = current_price - (atr * cfg.atr_stop_multiplier)
                        take_profit = current_price + (atr * cfg.atr_profit_multiplier)
                    else:
                        stop_loss = current_price + (atr * cfg.atr_stop_multiplier)
                        take_profit = current_price - (atr * cfg.atr_profit_multiplier)

                    # Calculate position size
                    try:
                        plan = risk_manager.calculate_position(
                            balance=simulator.get_position_capital(),
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            signal_confidence=confidence,
                            market_regime="trending",
                            atr=atr,
                            atr_percent=atr / current_price
                        )
                    except Exception:
                        continue

                    position_size = getattr(plan, 'position_size', 0)
                    if position_size <= 0:
                        continue

                    # Open position
                    simulator.open_position(
                        symbol=symbol,
                        direction=side,
                        entry_price=current_price,
                        position_size=float(position_size),
                        stop_loss=float(stop_loss),
                        take_profit=float(take_profit),
                        current_time=current_ts,
                        confidence=confidence
                    )

                equity_curve.append(simulator.capital)

            # Close remaining positions
            for sym in list(simulator.positions.keys()):
                if sym in data:
                    final_price = float(data[sym].iloc[-1]['close'])
                else:
                    final_price = float(data[symbols[0]].iloc[-1]['close'])
                simulator.close_position(sym, final_price, "End")
            equity_curve.append(simulator.capital)

            # Get report and convert to metrics
            metrics = self.metrics_calc.calculate_trading_metrics(
                simulator.trade_history,
                equity_curve=equity_curve,
                initial_capital=cfg.initial_capital
            )

            if not return_details:
                return metrics

            trade_history = simulator.trade_history or []
            details = {
                "confidences": [float(t.get("confidence", 0.0)) for t in trade_history],
                "outcomes": [1 if t.get("pnl", 0) > 0 else 0 for t in trade_history],
                "trade_pnls": [float(t.get("pnl", 0.0)) for t in trade_history],
            }
            return metrics, details

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return (TradingMetrics(), empty_details) if return_details else TradingMetrics()

        finally:
            # Cleanup temp file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _save_checkpoint(self, period_results: List[Dict]):
        """Save intermediate checkpoint"""
        checkpoint = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "completed_periods": len(period_results),
            "results": period_results
        }

        path = os.path.join(self.output_dir, "checkpoint.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def _save_results(self, result: ExperimentResult):
        """Save final experiment results"""
        # Save full results
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save summary for quick comparison
        summary = {
            "experiment_id": result.experiment_id,
            "method": self.config.method.value,
            "model": self.config.model.model_type.value if self.config.method in [
                EvolutionMethod.LLM_ITERATIVE, EvolutionMethod.LLM_SINGLE_SHOT
            ] else "N/A",
            "n_periods": len(result.period_results),
            "mean_test_sharpe": result.mean_test_sharpe,
            "std_test_sharpe": result.std_test_sharpe,
            "mean_test_pnl": result.mean_test_pnl,
            "total_test_pnl": result.total_test_pnl,
            "mean_test_win_rate": result.mean_test_win_rate,
            "duration_seconds": result.duration_seconds,
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
