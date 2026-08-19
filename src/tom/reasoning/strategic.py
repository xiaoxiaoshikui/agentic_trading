from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..base import Decision, MarketState
from .aggregator import AggregatedPrediction


@dataclass
class StrategicDecision:
    action: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class StrategicReasoner:
    """
    Minimal ToM-Strategic Decision Rule.

    Rule:
    - If opponent net action aligns with technical signal, follow with higher confidence.
    - If opponent net action opposes technical signal, fade when opponent confidence is low,
      otherwise reduce confidence and hold.
    - Depth > 1 applies recursive adjustment to opponent flow (anticipate opponents anticipating you).
    """

    def __init__(self, depth: int = 1):
        self.depth = max(1, int(depth))
        self.opponent_quality = 0.0
        self.influence = 0.20
        self.gate_threshold = 0.10
        self.override_threshold = 0.55
        self.decision_threshold = 0.18

    def set_opponent_quality(self, quality: float) -> None:
        self.opponent_quality = max(-1.0, min(1.0, float(quality)))

    def set_policy_params(
        self,
        influence: float | None = None,
        gate_threshold: float | None = None,
        override_threshold: float | None = None,
        decision_threshold: float | None = None,
    ) -> None:
        if influence is not None:
            self.influence = float(max(-1.5, min(1.5, influence)))
        if gate_threshold is not None:
            self.gate_threshold = float(max(0.0, min(1.0, gate_threshold)))
        if override_threshold is not None:
            self.override_threshold = float(max(0.0, min(1.5, override_threshold)))
        if decision_threshold is not None:
            self.decision_threshold = float(max(0.05, min(1.0, decision_threshold)))

    def get_policy_params(self) -> Dict[str, float]:
        return {
            "influence": float(self.influence),
            "gate_threshold": float(self.gate_threshold),
            "override_threshold": float(self.override_threshold),
            "decision_threshold": float(self.decision_threshold),
        }

    def decide(
        self,
        state: MarketState,
        technical: Decision,
        aggregated: AggregatedPrediction
    ) -> StrategicDecision:
        tech_action = technical.action
        net_action = aggregated.net_action
        opp_conf = aggregated.confidence

        if self.depth > 1:
            # Recursive adjustment: even depths invert opponent flow; odd depths keep sign.
            flip = -1.0 if (self.depth % 2 == 0) else 1.0
            attenuation = max(0.4, 1.0 - 0.15 * (self.depth - 1))
            net_action = net_action * flip * attenuation
            opp_conf = max(0.0, opp_conf - 0.10 * (self.depth - 1))

        action = "FLAT"
        confidence = 0.3
        reason = "Default neutral"

        base_conf = max(0.0, min(1.0, float(getattr(technical, "confidence", 0.0))))
        quality_scale = 0.6 + 0.4 * max(0.0, self.opponent_quality)
        effective_opp_conf = max(0.0, min(1.0, opp_conf * quality_scale))

        tech_score = 0.0
        if tech_action == "LONG":
            tech_score = base_conf
        elif tech_action == "SHORT":
            tech_score = -base_conf

        opp_score = float(net_action) * effective_opp_conf
        opp_score = max(-1.0, min(1.0, opp_score))
        opp_term = opp_score if abs(opp_score) >= self.gate_threshold else 0.0

        blended_score = tech_score + self.influence * opp_term

        # Opponent strongly disagrees while technical confidence is weak -> step aside.
        if (
            self.influence > 0.0
            and
            tech_score != 0.0
            and opp_term != 0.0
            and np.sign(opp_term) != np.sign(tech_score)
            and abs(opp_term) >= self.override_threshold
            and abs(tech_score) < 0.65
        ):
            action = "FLAT"
            confidence = min(0.65, 0.30 + abs(opp_term) * 0.30)
            reason = "Strong opposing opponent score; override weak technical conviction"
        else:
            if blended_score >= self.decision_threshold:
                action = "LONG"
                confidence = min(0.95, max(base_conf, abs(blended_score)))
                reason = "Blended technical + opponent score favors LONG"
            elif blended_score <= -self.decision_threshold:
                action = "SHORT"
                confidence = min(0.95, max(base_conf, abs(blended_score)))
                reason = "Blended technical + opponent score favors SHORT"
            else:
                action = "FLAT"
                confidence = min(0.60, max(0.20, abs(blended_score)))
                reason = "Blended score near neutral; stay flat"

        return StrategicDecision(
            action=action,
            confidence=float(confidence),
            reasoning=reason,
            metadata={
                "technical_action": tech_action,
                "technical_confidence": technical.confidence,
                "opponent_net_action": aggregated.net_action,
                "opponent_confidence": aggregated.confidence,
                "adjusted_net_action": net_action,
                "adjusted_confidence": effective_opp_conf,
                "tech_score": tech_score,
                "opp_score": opp_score,
                "opp_term": opp_term,
                "blended_score": blended_score,
                "opponent_quality": self.opponent_quality,
                "policy": self.get_policy_params(),
                "depth": self.depth,
            },
        )
