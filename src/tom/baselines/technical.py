from __future__ import annotations

from ..base import BaseAgent, Decision, MarketState


class TechnicalAgent(BaseAgent):
    """Technical analysis baseline using existing dynamic_strategy if available."""

    @property
    def name(self) -> str:
        return "Technical"

    def decide(self, state: MarketState) -> Decision:
        try:
            from src.dynamic_strategy import calculate_signal

            signal = calculate_signal(state.df)
            return Decision(
                action=getattr(signal, "side", "FLAT"),
                confidence=float(getattr(signal, "confidence", 0.0)),
                reasoning=getattr(signal, "reason", "dynamic_strategy"),
                metadata={},
            )
        except Exception:
            return Decision(
                action="FLAT",
                confidence=0.0,
                reasoning="dynamic_strategy unavailable",
                metadata={},
            )
