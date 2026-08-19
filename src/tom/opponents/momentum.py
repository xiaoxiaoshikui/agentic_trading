from __future__ import annotations

from typing import Optional

from .base import AgentAction, OpponentPrediction, OpponentModel
from ..base import MarketState


class MomentumAlgoModel(OpponentModel):
    """
    Rule-based momentum algorithm model.

    Uses EMA cross + price change + RSI filters if present.
    """

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def market_share(self) -> float:
        if self._market_share is not None:
            return float(self._market_share)
        return 0.35

    def predict(self, state: MarketState) -> OpponentPrediction:
        ema_fast = self._get_indicator(state, "ema_50")
        ema_slow = self._get_indicator(state, "ema_200")
        rsi = self._get_indicator(state, "rsi")

        action = AgentAction.HOLD
        confidence = 0.4
        reason = "No clear momentum signal"

        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow and state.price_change_24h > 0:
                action = AgentAction.BUY
                confidence = 0.6
                reason = "EMA trend up with positive momentum"
            elif ema_fast < ema_slow and state.price_change_24h < 0:
                action = AgentAction.SELL
                confidence = 0.6
                reason = "EMA trend down with negative momentum"

        if rsi is not None:
            if rsi > 70 and action in (AgentAction.BUY, AgentAction.STRONG_BUY):
                action = AgentAction.BUY
                confidence = min(0.55, confidence)
                reason = "RSI overbought, momentum toned down"
            if rsi < 30 and action in (AgentAction.SELL, AgentAction.STRONG_SELL):
                action = AgentAction.SELL
                confidence = min(0.55, confidence)
                reason = "RSI oversold, momentum toned down"

        return self._prediction(action, confidence, reason, state)

    def get_impact(self, prediction: OpponentPrediction) -> float:
        return prediction.price_impact * self.market_share

    def _get_indicator(self, state: MarketState, key: str) -> Optional[float]:
        if state.indicators and key in state.indicators:
            try:
                return float(state.indicators[key])
            except Exception:
                return None
        return None

    def _prediction(
        self,
        action: AgentAction,
        confidence: float,
        reason: str,
        state: MarketState
    ) -> OpponentPrediction:
        volume_estimate = max(1.0, float(state.volume)) * self.market_share
        price_impact = 0.0012 * action.value
        return OpponentPrediction(
            action=action,
            confidence=confidence,
            reasoning=reason,
            volume_estimate=volume_estimate,
            price_impact=price_impact
        )
    def __init__(self, market_share: Optional[float] = None):
        self._market_share = market_share
