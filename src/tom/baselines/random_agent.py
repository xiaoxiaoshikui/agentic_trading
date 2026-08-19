from __future__ import annotations

import random

from ..base import BaseAgent, Decision, MarketState


class RandomAgent(BaseAgent):
    """Random decision baseline."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    def decide(self, state: MarketState) -> Decision:
        action = self.rng.choices(
            ["LONG", "SHORT", "FLAT"], weights=[0.3, 0.3, 0.4]
        )[0]
        return Decision(
            action=action,
            confidence=0.5,
            reasoning="Random baseline",
            metadata={},
        )
