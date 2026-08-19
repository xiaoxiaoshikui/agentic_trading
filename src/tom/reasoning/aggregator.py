from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..base import MarketState
from ..opponents import OpponentModel, OpponentPrediction


@dataclass
class AggregatedPrediction:
    net_action: float
    net_volume: float
    net_price_impact: float
    confidence: float
    breakdown: Dict[str, OpponentPrediction]
    reasoning: str


class PredictionAggregator:
    """Combines predictions from all opponent models."""

    def __init__(self, opponent_models: List[OpponentModel]):
        self.models = {m.name: m for m in opponent_models}
        self.base_weights = {m.name: m.market_share for m in opponent_models}
        self.weights = dict(self.base_weights)

    def set_weights(self, weights: Dict[str, float], normalize: bool = True) -> None:
        """Set model weights, optionally normalizing to base total market share."""
        base_total = sum(self.base_weights.values()) or 1.0
        merged = {name: max(0.0, float(weights.get(name, self.weights.get(name, 0.0)))) for name in self.models}

        if normalize:
            current_total = sum(merged.values())
            if current_total > 0:
                scale = base_total / current_total
                merged = {k: v * scale for k, v in merged.items()}
            else:
                merged = dict(self.base_weights)

        self.weights = merged

    def reset_weights(self) -> None:
        self.weights = dict(self.base_weights)

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def aggregate(self, state: MarketState) -> AggregatedPrediction:
        predictions: Dict[str, OpponentPrediction] = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(state)

        net_action = sum(
            self.weights[name] * pred.action.value
            for name, pred in predictions.items()
        )

        net_volume = sum(pred.volume_estimate for pred in predictions.values())
        net_impact = sum(
            self.models[name].get_impact(predictions[name])
            for name in predictions
        )

        weighted_confidence = sum(
            self.weights[name] * pred.confidence
            for name, pred in predictions.items()
        )
        weight_total = sum(self.weights.values()) or 0.0
        normalized_confidence = weighted_confidence / weight_total if weight_total > 0 else 0.0

        reasoning = self._build_reasoning(predictions)

        return AggregatedPrediction(
            net_action=float(net_action),
            net_volume=float(net_volume),
            net_price_impact=float(net_impact),
            confidence=float(normalized_confidence),
            breakdown=predictions,
            reasoning=reasoning
        )

    def _build_reasoning(self, predictions: Dict[str, OpponentPrediction]) -> str:
        lines = ["Opponent analysis:"]
        for name, pred in predictions.items():
            w = self.weights.get(name, 0.0)
            lines.append(
                f"- {name}: {pred.action.name} (conf={pred.confidence:.2f}, w={w:.3f}) {pred.reasoning}"
            )
        return "\n".join(lines)
