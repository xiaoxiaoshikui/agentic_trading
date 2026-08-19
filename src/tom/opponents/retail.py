from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import AgentAction, OpponentPrediction, OpponentModel
from ..base import MarketState


@dataclass
class RetailConfig:
    panic_drop_1h: float = -0.05
    fomo_rally_24h: float = 0.15
    capitulation_7d: float = -0.20
    capitulation_1h: float = -0.02


class RetailTraderModel(OpponentModel):
    """
    Rule-based retail behavior model.

    Intuition:
    - Panic sells on sharp drops.
    - FOMO buys on strong rallies.
    - Capitulation after prolonged declines.
    - Otherwise mildly trend-following.
    """

    def __init__(self, config: Optional[RetailConfig] = None, market_share: Optional[float] = None):
        self.config = config or RetailConfig()
        self._market_share = market_share

    @property
    def name(self) -> str:
        return "retail"

    @property
    def market_share(self) -> float:
        if self._market_share is not None:
            return float(self._market_share)
        return 0.25

    def predict(self, state: MarketState) -> OpponentPrediction:
        cfg = self.config

        if state.price_change_1h <= cfg.panic_drop_1h:
            return self._prediction(
                AgentAction.STRONG_SELL,
                0.8,
                "Panic selling after sharp drop",
                state
            )

        if state.price_change_24h >= cfg.fomo_rally_24h:
            return self._prediction(
                AgentAction.STRONG_BUY,
                0.75,
                "FOMO buying after strong rally",
                state
            )

        if state.price_change_7d <= cfg.capitulation_7d and state.price_change_1h <= cfg.capitulation_1h:
            return self._prediction(
                AgentAction.STRONG_SELL,
                0.7,
                "Capitulation after prolonged decline",
                state
            )

        if state.price_change_24h >= 0.03:
            return self._prediction(AgentAction.BUY, 0.5, "Retail follows trend up", state)
        if state.price_change_24h <= -0.03:
            return self._prediction(AgentAction.SELL, 0.5, "Retail follows trend down", state)

        return self._prediction(AgentAction.HOLD, 0.4, "No strong retail signal", state)

    def get_impact(self, prediction: OpponentPrediction) -> float:
        return prediction.price_impact * self.market_share

    def _prediction(
        self,
        action: AgentAction,
        confidence: float,
        reason: str,
        state: MarketState
    ) -> OpponentPrediction:
        volume_estimate = max(1.0, float(state.volume)) * self.market_share
        price_impact = 0.001 * action.value
        return OpponentPrediction(
            action=action,
            confidence=confidence,
            reasoning=reason,
            volume_estimate=volume_estimate,
            price_impact=price_impact
        )
