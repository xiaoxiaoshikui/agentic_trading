"""
Random Mutation Baseline
========================

Randomly mutates strategy parameters without any intelligent feedback.
This tests whether LLM evolution is better than random search.
"""

import random
import re
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


TEMPLATE_STRATEGY = '''
"""
EMA Crossover Strategy with Configurable Parameters
Parameters mutated randomly each iteration
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
    EMA crossover strategy with randomly mutated parameters.
    """
    try:
        if df is None or len(df) < {ema_slow} + 50:
            return Signal(side="FLAT", confidence=0.0, reason="Insufficient data")

        close = df["close"]

        # Calculate indicators with mutated parameters
        ema_fast = ta.trend.ema_indicator(close, window={ema_fast})
        ema_slow = ta.trend.ema_indicator(close, window={ema_slow})
        rsi = ta.momentum.rsi(close, window={rsi_period})

        ema_f = float(ema_fast.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        rsi_val = float(rsi.iloc[-1])

        if not all(np.isfinite([ema_f, ema_s, rsi_val])):
            return Signal(side="FLAT", confidence=0.0, reason="Invalid indicators")

        # Determine trend
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s

        # Generate signal with mutated thresholds
        long_score = 0.0
        short_score = 0.0
        reasons = []

        # EMA weight
        ema_weight = {ema_weight}
        rsi_weight = {rsi_weight}

        if uptrend:
            long_score += ema_weight
            reasons.append("EMA bullish")
            if rsi_val < {rsi_overbought}:
                long_score += rsi_weight
                reasons.append("RSI ok")

        if downtrend:
            short_score += ema_weight
            reasons.append("EMA bearish")
            if rsi_val > {rsi_oversold}:
                short_score += rsi_weight
                reasons.append("RSI ok")

        # Determine side and confidence
        min_conf = {min_confidence}
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
        return Signal(side="FLAT", confidence=0.0, reason=f"Error: {{str(e)}}")
'''


class RandomMutationBaseline:
    """
    Randomly mutates strategy parameters.

    This baseline tests whether LLM evolution is better than
    random parameter search. If random is comparable, then
    LLM isn't adding intelligence.

    Parameter ranges:
    - ema_fast: [10, 100]
    - ema_slow: [100, 300]
    - rsi_period: [7, 21]
    - rsi_overbought: [60, 80]
    - rsi_oversold: [20, 40]
    - ema_weight: [0.1, 0.5]
    - rsi_weight: [0.1, 0.4]
    - min_confidence: [0.2, 0.6]
    """

    def __init__(self, seed: int = 42, mutation_rate: float = 0.3):
        self.seed = seed
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self.iteration = 0

        # Initialize with default parameters
        self.params = {
            "ema_fast": 50,
            "ema_slow": 200,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ema_weight": 0.3,
            "rsi_weight": 0.2,
            "min_confidence": 0.4,
        }

        # Parameter ranges for mutation
        self.param_ranges = {
            "ema_fast": (10, 100),
            "ema_slow": (100, 300),
            "rsi_period": (7, 21),
            "rsi_overbought": (60, 80),
            "rsi_oversold": (20, 40),
            "ema_weight": (0.1, 0.5),
            "rsi_weight": (0.1, 0.4),
            "min_confidence": (0.2, 0.6),
        }

        self.strategy_code = self._generate_code()

    def _generate_code(self) -> str:
        """Generate strategy code from current parameters"""
        return TEMPLATE_STRATEGY.format(**self.params)

    def _mutate_param(self, name: str, value: Any) -> Any:
        """Mutate a single parameter"""
        low, high = self.param_ranges[name]

        if isinstance(value, int):
            # Integer mutation: add noise proportional to range
            noise = self.rng.randint(-int((high - low) * 0.3), int((high - low) * 0.3))
            return max(low, min(high, value + noise))
        else:
            # Float mutation: add Gaussian noise
            noise = self.rng.gauss(0, (high - low) * 0.2)
            return max(low, min(high, round(value + noise, 2)))

    def get_strategy_code(self) -> str:
        """Return current strategy code"""
        return self.strategy_code

    def evolve(
        self,
        feedback: Dict[str, Any],
        train_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Randomly mutate parameters (ignoring feedback).

        This is intentionally "dumb" - it doesn't use feedback.
        This tests whether LLM's use of feedback actually helps.

        Args:
            feedback: Performance feedback (ignored)
            train_data: Training data (ignored)

        Returns:
            (strategy_code, evolution_info)
        """
        self.iteration += 1
        old_params = self.params.copy()

        # Randomly mutate each parameter with probability mutation_rate
        changed_params = []
        for name, value in self.params.items():
            if self.rng.random() < self.mutation_rate:
                new_value = self._mutate_param(name, value)
                if new_value != value:
                    self.params[name] = new_value
                    changed_params.append(f"{name}: {value} -> {new_value}")

        # Ensure ema_fast < ema_slow
        if self.params["ema_fast"] >= self.params["ema_slow"]:
            self.params["ema_slow"] = self.params["ema_fast"] + 50

        self.strategy_code = self._generate_code()

        return self.strategy_code, {
            "iteration": self.iteration,
            "evolved": len(changed_params) > 0,
            "reason": f"Random mutation: {', '.join(changed_params)}" if changed_params else "No changes",
            "code_changed": len(changed_params) > 0,
            "params": self.params.copy(),
            "changes": changed_params,
        }

    def reset(self):
        """Reset for new experiment run"""
        self.iteration = 0
        self.rng = random.Random(self.seed)
        self.params = {
            "ema_fast": 50,
            "ema_slow": 200,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ema_weight": 0.3,
            "rsi_weight": 0.2,
            "min_confidence": 0.4,
        }
        self.strategy_code = self._generate_code()
