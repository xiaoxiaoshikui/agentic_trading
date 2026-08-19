from __future__ import annotations

from ..base import BaseAgent, Decision, MarketState


class BuyAndHoldAgent(BaseAgent):
    """Always holds a long position."""

    @property
    def name(self) -> str:
        return "BuyHold"

    def decide(self, state: MarketState) -> Decision:
        return Decision(
            action="LONG",
            confidence=1.0,
            reasoning="Buy and hold baseline",
            metadata={},
        )
