"""
Static Baseline
===============

Uses a fixed EMA crossover strategy without any evolution.
This is the simplest baseline - no learning, no adaptation.
"""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


STATIC_STRATEGY_CODE = '''
"""
Static EMA Crossover Strategy (Baseline)
No evolution - fixed parameters
"""

import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass


@dataclass
class Signal:
    side: str = "FLAT"
    confidence: float = 0.0
    reason: str = ""


def calculate_signal(df: pd.DataFrame) -> Signal:
    """
    Simple EMA crossover strategy with fixed parameters.

    Parameters (fixed):
    - EMA fast: 50
    - EMA slow: 200
    - RSI period: 14
    - Min confidence: 0.4
    """
    try:
        if df is None or len(df) < 200:
            return Signal(side="FLAT", confidence=0.0, reason="Insufficient data")

        close = df["close"]

        # Calculate indicators
        ema_fast = ta.trend.ema_indicator(close, window=50)
        ema_slow = ta.trend.ema_indicator(close, window=200)
        rsi = ta.momentum.rsi(close, window=14)

        ema_f = float(ema_fast.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        rsi_val = float(rsi.iloc[-1])
        current_price = float(close.iloc[-1])

        if not all(np.isfinite([ema_f, ema_s, rsi_val])):
            return Signal(side="FLAT", confidence=0.0, reason="Invalid indicators")

        # Determine trend
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s

        # Generate signal
        long_score = 0.0
        short_score = 0.0
        reasons = []

        if uptrend:
            long_score += 0.3
            reasons.append("EMA bullish")
            if rsi_val < 70:
                long_score += 0.2
                reasons.append("RSI not overbought")

        if downtrend:
            short_score += 0.3
            reasons.append("EMA bearish")
            if rsi_val > 30:
                short_score += 0.2
                reasons.append("RSI not oversold")

        # Determine side and confidence
        min_conf = 0.4
        side = "FLAT"
        conf = 0.0

        if long_score >= min_conf and long_score > short_score:
            side = "LONG"
            conf = min(1.0, long_score)
        elif short_score >= min_conf and short_score > long_score:
            side = "SHORT"
            conf = min(1.0, short_score)

        reason = "; ".join(reasons) if reasons else "No signal"
        return Signal(side=side, confidence=conf, reason=reason)

    except Exception as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Error: {str(e)}")
'''


class StaticBaseline:
    """
    Static baseline that never changes the strategy.

    Used as the simplest baseline - if LLM evolution can't beat this,
    it's not adding value.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.strategy_code = STATIC_STRATEGY_CODE
        self.iteration = 0

    def get_strategy_code(self) -> str:
        """Return the fixed strategy code"""
        return self.strategy_code

    def evolve(
        self,
        feedback: Dict[str, Any],
        train_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Static baseline doesn't evolve - always returns same code.

        Args:
            feedback: Performance feedback from last iteration
            train_data: Training data (ignored)

        Returns:
            (strategy_code, evolution_info)
        """
        self.iteration += 1

        return self.strategy_code, {
            "iteration": self.iteration,
            "evolved": False,
            "reason": "Static baseline - no evolution",
            "code_changed": False,
        }

    def reset(self):
        """Reset for new experiment run"""
        self.iteration = 0
