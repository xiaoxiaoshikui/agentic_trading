from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import ta

from ..base import MarketState
from ..base import BaseAgent
from ..base import Decision

from src.multi_asset_simulator import MultiAssetSimulator
from src.advanced_risk import AdvancedRiskManager
from experiments.metrics import MetricsCalculator, TradingMetrics


@dataclass
class EvalConfig:
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
    warmup_bars: int = 200


@dataclass
class EvalResult:
    agent_name: str
    sharpe: float
    max_drawdown: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    total_trades: int
    details: Dict[str, Any]


class EvaluationHarness:
    """
    Minimal evaluation harness for ToM agents.
    Uses existing MultiAssetSimulator + AdvancedRiskManager.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator()

    # Noise parameters for seed-dependent variation
    EPSILON = 0.03        # 3% chance to flip LONG<->SHORT
    CONF_JITTER = 0.05    # ±5% confidence perturbation
    SLIPPAGE = 0.0005     # ±0.05% price slippage

    def evaluate_agent(
        self,
        agent: BaseAgent,
        data: Dict[str, pd.DataFrame],
        seed: int = 42
    ) -> EvalResult:
        np.random.seed(seed)
        cfg = self.config

        simulator = MultiAssetSimulator(
            initial_capital=cfg.initial_capital,
            max_positions=cfg.max_positions,
            capital_per_position=cfg.capital_per_position,
            max_long_exposure=cfg.max_long_exposure,
            max_short_exposure=cfg.max_short_exposure,
            max_daily_loss=cfg.max_daily_loss,
            max_single_asset_loss=cfg.max_single_asset_loss,
        )
        risk_manager = AdvancedRiskManager()

        symbols = list(data.keys())
        if not symbols:
            return self._empty_result(agent.name)

        min_time = max(df.index[0] for df in data.values())
        max_time = min(df.index[-1] for df in data.values())
        for symbol in data:
            data[symbol] = data[symbol][
                (data[symbol].index >= min_time) &
                (data[symbol].index <= max_time)
            ]

        common_times = data[symbols[0]].index
        total_bars = len(common_times)
        warmup = cfg.warmup_bars
        if total_bars <= warmup:
            return self._empty_result(agent.name)

        equity_curve = [cfg.initial_capital]

        for i in range(warmup, total_bars):
            current_time = common_times[i]
            current_ts = float(current_time.timestamp()) if hasattr(current_time, "timestamp") else float(i)

            prices = {}
            for symbol, df in data.items():
                if current_time in df.index:
                    # Use current bar open for execution to avoid lookahead
                    raw_price = float(df.loc[current_time, "open"])
                    # Add small random slippage so seeds produce different results
                    slippage = np.random.uniform(-self.SLIPPAGE, self.SLIPPAGE)
                    prices[symbol] = raw_price * (1 + slippage)

            simulator.check_positions(prices, current_ts)

            for symbol, df in data.items():
                if not simulator.can_open_position(symbol):
                    continue

                # Use only past bars for signal calculation
                df_slice = df.iloc[:i].copy()
                if len(df_slice) < warmup:
                    continue

                state = self._build_state(symbol, df_slice)
                decision = agent.decide(state)

                # Seed-dependent noise: epsilon-greedy flip + confidence jitter
                action = decision.action
                confidence = decision.confidence
                if action != "FLAT":
                    # Epsilon-greedy: small chance to flip direction
                    if np.random.random() < self.EPSILON:
                        action = "SHORT" if action == "LONG" else "LONG"
                    # Confidence jitter
                    confidence += np.random.uniform(-self.CONF_JITTER, self.CONF_JITTER)
                    confidence = max(0.0, min(1.0, confidence))

                if action == "FLAT" or confidence < cfg.min_confidence:
                    continue

                current_price = prices.get(symbol)
                if not current_price:
                    continue

                atr_series = ta.volatility.average_true_range(
                    df_slice["high"], df_slice["low"], df_slice["close"], window=14
                )
                atr = float(atr_series.iloc[-1])
                if not np.isfinite(atr) or atr <= 0:
                    atr = current_price * 0.01

                if action == "LONG":
                    stop_loss = current_price - (atr * cfg.atr_stop_multiplier)
                    take_profit = current_price + (atr * cfg.atr_profit_multiplier)
                else:
                    stop_loss = current_price + (atr * cfg.atr_stop_multiplier)
                    take_profit = current_price - (atr * cfg.atr_profit_multiplier)

                plan = risk_manager.calculate_position(
                    balance=simulator.get_position_capital(),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    signal_confidence=float(confidence),
                    market_regime="trending",
                    atr=atr,
                    atr_percent=atr / current_price
                )

                position_size = getattr(plan, "position_size", 0)
                if position_size <= 0:
                    continue

                simulator.open_position(
                    symbol=symbol,
                    direction=action,
                    entry_price=current_price,
                    position_size=float(position_size),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    current_time=current_ts,
                    confidence=float(confidence)
                )

            equity_curve.append(simulator.capital)

        for sym in list(simulator.positions.keys()):
            final_price = float(data[sym].iloc[-1]["close"])
            simulator.close_position(sym, final_price, "End")
        equity_curve.append(simulator.capital)

        metrics: TradingMetrics = self.metrics_calc.calculate_trading_metrics(
            simulator.trade_history,
            equity_curve=equity_curve,
            initial_capital=cfg.initial_capital
        )

        return EvalResult(
            agent_name=agent.name,
            sharpe=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            total_pnl=metrics.total_pnl,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_trades=metrics.total_trades,
            details={"trade_history": simulator.trade_history},
        )

    def _build_state(self, symbol: str, df_slice: pd.DataFrame) -> MarketState:
        indicators = {}
        indicators["ema_50"] = float(ta.trend.ema_indicator(df_slice["close"], window=50).iloc[-1])
        indicators["ema_200"] = float(ta.trend.ema_indicator(df_slice["close"], window=200).iloc[-1])
        indicators["rsi"] = float(ta.momentum.rsi(df_slice["close"], window=14).iloc[-1])
        indicators["atr"] = float(
            ta.volatility.average_true_range(
                df_slice["high"], df_slice["low"], df_slice["close"], window=14
            ).iloc[-1]
        )

        current_price = float(df_slice["close"].iloc[-1])
        close_series = df_slice["close"]
        price_change_1h = self._pct_change_over_horizon(close_series, timedelta(hours=1))
        price_change_24h = self._pct_change_over_horizon(close_series, timedelta(hours=24))
        price_change_7d = self._pct_change_over_horizon(close_series, timedelta(days=7))

        return MarketState(
            symbol=symbol,
            df=df_slice,
            current_price=current_price,
            price_change_1h=float(price_change_1h) if np.isfinite(price_change_1h) else 0.0,
            price_change_24h=float(price_change_24h) if np.isfinite(price_change_24h) else 0.0,
            price_change_7d=float(price_change_7d) if np.isfinite(price_change_7d) else 0.0,
            volume=float(df_slice["volume"].iloc[-1]),
            volume_change=float(df_slice["volume"].pct_change(20).iloc[-1]) if len(df_slice) > 20 else 0.0,
            indicators=indicators,
            funding_rate=float(df_slice.get("funding_rate", pd.Series([0.0])).iloc[-1]),
            open_interest=float(df_slice.get("open_interest", pd.Series([0.0])).iloc[-1]),
            timestamp=df_slice.index[-1],
        )

    def _pct_change_over_horizon(self, series: pd.Series, horizon: timedelta) -> float:
        """
        Compute return from now to the closest bar at-or-before (now - horizon).

        This keeps feature semantics consistent across 15m/1h/4h runs and under
        missing bars, instead of assuming fixed bar counts.
        """
        if series.empty:
            return 0.0

        index = series.index
        if not isinstance(index, pd.DatetimeIndex):
            return 0.0

        current_ts = index[-1]
        target_ts = current_ts - horizon
        target_pos = int(index.searchsorted(target_ts, side="right")) - 1

        if target_pos < 0:
            return 0.0

        start_price = float(series.iloc[target_pos])
        end_price = float(series.iloc[-1])
        if not np.isfinite(start_price) or start_price <= 0.0:
            return 0.0

        ret = (end_price / start_price) - 1.0
        return float(ret) if np.isfinite(ret) else 0.0

    def _empty_result(self, agent_name: str) -> EvalResult:
        return EvalResult(
            agent_name=agent_name,
            sharpe=0.0,
            max_drawdown=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            details={},
        )
