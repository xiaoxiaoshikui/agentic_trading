from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from ..base import MarketState


class AgentAction(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class OpponentPrediction:
    """Prediction of what an opponent will do."""

    action: AgentAction
    confidence: float
    reasoning: str
    volume_estimate: float
    price_impact: float


class OpponentModel(ABC):
    """Base class for all opponent models."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def market_share(self) -> float:
        pass

    @abstractmethod
    def predict(self, state: MarketState) -> OpponentPrediction:
        pass

    @abstractmethod
    def get_impact(self, prediction: OpponentPrediction) -> float:
        pass
