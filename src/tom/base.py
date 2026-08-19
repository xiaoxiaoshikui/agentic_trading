from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
from abc import ABC, abstractmethod

import pandas as pd


@dataclass
class MarketState:
    """Standardized input for all agents."""

    symbol: str
    df: pd.DataFrame
    current_price: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    volume: float
    volume_change: float
    indicators: Dict[str, float]
    funding_rate: float
    open_interest: float
    timestamp: datetime


@dataclass
class Decision:
    """Standardized output from all agents."""

    action: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Interface all agents must implement."""

    @abstractmethod
    def decide(self, state: MarketState) -> Decision:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
