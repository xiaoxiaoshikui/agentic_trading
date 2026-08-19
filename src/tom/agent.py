from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseAgent, Decision, MarketState
from .opponents import MomentumAlgoModel, RetailTraderModel
from .reasoning import PredictionAggregator
from .reasoning.strategic import StrategicDecision, StrategicReasoner


@dataclass
class TechnicalSignal:
    action: str
    confidence: float
    reasoning: str


@dataclass
class TechnicalExpertVote:
    name: str
    score: float
    confidence: float
    reasoning: str


class TheoryOfMindAgent(BaseAgent):
    """
    Minimal ToM agent:
    - Uses a simple technical baseline.
    - Predicts opponent behavior (retail + momentum).
    - Applies strategic decision rule.
    """

    def __init__(
        self,
        opponents: Optional[List[Any]] = None,
        depth: int = 1,
        use_technical: bool = True,
        technical_mode: str = "single",
        allow_negative_influence: bool = False,
        name: Optional[str] = None
    ):
        self.opponent_models = opponents or [RetailTraderModel(), MomentumAlgoModel()]
        self.aggregator = PredictionAggregator(self.opponent_models)
        self.reasoner = StrategicReasoner(depth=depth)
        self.use_technical = bool(use_technical)
        self.technical_mode = str(technical_mode).lower().strip() if technical_mode else "single"
        if self.technical_mode not in {"single", "multi"}:
            self.technical_mode = "single"
        self.allow_negative_influence = bool(allow_negative_influence)
        self._name = name or "ToM-Minimal"
        self._regime_weights: Dict[str, Dict[str, float]] = {
            "trend": {"trend": 0.55, "mean_reversion": 0.15, "breakout": 0.30},
            "chop": {"trend": 0.25, "mean_reversion": 0.55, "breakout": 0.20},
            "high_vol": {"trend": 0.25, "mean_reversion": 0.20, "breakout": 0.55},
        }
        self._expert_entry_threshold = 0.60
        self._expert_conf_cap = 0.30
        self._calibration: Dict[str, Any] = {
            "quality": 0.0,
            "weights": self.aggregator.get_weights(),
            "model_scores": {},
            "n_samples": 0,
            "policy_params": self.reasoner.get_policy_params(),
            "expected_disagreement": 0.0,
        }

    @property
    def name(self) -> str:
        return self._name

    def decide(self, state: MarketState) -> Decision:
        technical = self._technical_signal(state)
        aggregated = self.aggregator.aggregate(state)
        strategic = self.reasoner.decide(state, technical, aggregated)

        return Decision(
            action=strategic.action,
            confidence=strategic.confidence,
            reasoning=strategic.reasoning,
            metadata={
                "technical": technical.reasoning,
                "opponent_reasoning": aggregated.reasoning,
                "opponent_net_action": aggregated.net_action,
                "opponent_confidence": aggregated.confidence,
                "opponent_weights": self.aggregator.get_weights(),
                "opponent_quality": self._calibration.get("quality", 0.0),
                "policy_params": self.reasoner.get_policy_params(),
            },
        )

    def calibration_info(self) -> Dict[str, Any]:
        return dict(self._calibration)

    def calibrate_on_history(
        self,
        train_data: Dict[str, Any],
        state_builder: Callable[[str, Any], MarketState],
        warmup_bars: int = 200,
        max_samples_per_symbol: int = 320,
    ) -> Dict[str, Any]:
        """
        Calibrate opponent model weights on train split using next-bar directional alignment.

        This uses only train data in each walk-forward period, so test evaluation remains
        out-of-sample.
        """
        _ = state_builder  # kept in signature for compatibility with existing call sites.
        model_scores: Dict[str, List[float]] = {name: [] for name in self.aggregator.models}
        samples_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        total_samples = 0

        for symbol, df in train_data.items():
            if len(df) <= warmup_bars + 2:
                continue

            start = max(warmup_bars, len(df) - max_samples_per_symbol - 1)
            end = len(df) - 1
            close = df["close"]
            volume = df["volume"]

            ema_50 = close.ewm(span=50, adjust=False).mean()
            ema_200 = close.ewm(span=200, adjust=False).mean()
            rsi = self._fast_rsi(close, window=14)
            atr = self._fast_atr(df, window=14)

            interval_minutes = self._infer_interval_minutes(df.index)
            bars_1h = max(1, int(round(60 / interval_minutes)))
            bars_24h = max(1, int(round(24 * 60 / interval_minutes)))
            bars_7d = max(1, int(round(7 * 24 * 60 / interval_minutes)))
            pct_1h = close.pct_change(bars_1h)
            pct_24h = close.pct_change(bars_24h)
            pct_7d = close.pct_change(bars_7d)
            vol_chg = volume.pct_change(20)
            samples: List[Dict[str, Any]] = []

            step = max(1, (end - start) // max_samples_per_symbol)
            for i in range(start, end, step):
                current_px = float(close.iloc[i])
                next_px = float(close.iloc[i + 1])
                if not np.isfinite(current_px) or current_px <= 0.0:
                    continue
                realized = next_px / current_px - 1.0
                if not np.isfinite(realized) or realized == 0.0:
                    continue
                realized_sign = 1 if realized > 0 else -1
                sample_weight = min(2.0, max(0.25, abs(realized) / 0.003))

                ts = df.index[i]
                ema_fast = self._safe_float(ema_50.iloc[i], default=np.nan)
                ema_slow = self._safe_float(ema_200.iloc[i], default=np.nan)
                rsi_val = self._safe_float(rsi.iloc[i], default=50.0)

                state = MarketState(
                    symbol=symbol,
                    df=df.iloc[: i + 1],
                    current_price=current_px,
                    price_change_1h=self._safe_float(pct_1h.iloc[i]),
                    price_change_24h=self._safe_float(pct_24h.iloc[i]),
                    price_change_7d=self._safe_float(pct_7d.iloc[i]),
                    volume=self._safe_float(volume.iloc[i], default=0.0),
                    volume_change=self._safe_float(vol_chg.iloc[i], default=0.0),
                    indicators={
                        "ema_50": ema_fast,
                        "ema_200": ema_slow,
                        "rsi": rsi_val,
                        "atr": self._safe_float(atr.iloc[i], default=np.nan),
                    },
                    funding_rate=0.0,
                    open_interest=0.0,
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                )

                model_signals: Dict[str, float] = {}
                for name, model in self.aggregator.models.items():
                    pred = model.predict(state)
                    pred_sign = 1 if pred.action.value > 0 else (-1 if pred.action.value < 0 else 0)
                    model_signals[name] = float(pred.action.value) * max(0.0, min(1.0, float(pred.confidence)))
                    if pred_sign == 0:
                        continue
                    score = pred_sign * realized_sign * max(0.0, min(1.0, float(pred.confidence))) * sample_weight
                    model_scores[name].append(score)

                samples.append(
                    {
                        "realized_return": float(realized),
                        "tech_score": self._signal_to_score(self._technical_signal(state)),
                        "model_signals": model_signals,
                    }
                )
                total_samples += 1

            if samples:
                samples_by_symbol[symbol] = samples

        if total_samples == 0:
            self.aggregator.reset_weights()
            self.reasoner.set_opponent_quality(0.0)
            self._calibration = {
                "quality": 0.0,
                "weights": self.aggregator.get_weights(),
                "model_scores": {},
                "n_samples": 0,
                "policy_params": self.reasoner.get_policy_params(),
                "expected_disagreement": 0.0,
            }
            return self.calibration_info()

        mean_scores = {
            name: (float(np.mean(scores)) if scores else 0.0)
            for name, scores in model_scores.items()
        }

        tuned_weights: Dict[str, float] = {}
        for name, base_weight in self.aggregator.base_weights.items():
            score = mean_scores.get(name, 0.0)
            reliability = 1.0 + 0.9 * max(-0.9, min(0.9, score))
            tuned_weights[name] = base_weight * max(0.1, reliability)

        self.aggregator.set_weights(tuned_weights, normalize=True)

        quality = float(np.mean(list(mean_scores.values()))) if mean_scores else 0.0
        quality = max(-1.0, min(1.0, quality))
        self.reasoner.set_opponent_quality(quality)

        policy_params, expected_disagreement, objective = self._optimize_policy_params(
            samples_by_symbol=samples_by_symbol,
            tuned_weights=self.aggregator.get_weights(),
            quality=quality,
        )
        self.reasoner.set_policy_params(**policy_params)

        self._calibration = {
            "quality": quality,
            "weights": self.aggregator.get_weights(),
            "model_scores": mean_scores,
            "n_samples": total_samples,
            "policy_params": self.reasoner.get_policy_params(),
            "expected_disagreement": expected_disagreement,
            "objective": objective,
        }
        return self.calibration_info()

    @staticmethod
    def _fast_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _fast_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        if not {"high", "low", "close"}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    @staticmethod
    def _infer_interval_minutes(index: Any) -> float:
        if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
            return 15.0
        diffs = np.diff(index.view("int64")) / 60_000_000_000.0
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return 15.0
        return float(max(1.0, np.median(diffs)))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            f = float(value)
            return f if np.isfinite(f) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _single_technical_score(ema_fast: float, ema_slow: float, rsi: float) -> float:
        if not np.isfinite(ema_fast) or not np.isfinite(ema_slow):
            return 0.0
        if ema_fast > ema_slow and rsi < 70:
            return 0.55
        if ema_fast < ema_slow and rsi > 30:
            return -0.55
        return 0.0

    @staticmethod
    def _fallback_technical_score(
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        price_change_1h: float = 0.0,
        price_change_24h: float = 0.0,
        volume_change: float = 0.0,
    ) -> float:
        if not np.isfinite(ema_fast) or not np.isfinite(ema_slow):
            return 0.0

        trend_score = 0.0
        if ema_fast > ema_slow and rsi < 72:
            trend_score = 0.55
        elif ema_fast < ema_slow and rsi > 28:
            trend_score = -0.55

        mean_reversion_score = 0.0
        if rsi <= 30:
            mean_reversion_score = 0.60
        elif rsi >= 70:
            mean_reversion_score = -0.60

        breakout_score = 0.0
        if abs(price_change_24h) >= 0.03 and volume_change > 0.0:
            breakout_score = np.sign(price_change_24h) * min(0.70, 0.35 + abs(price_change_24h) * 3.0)

        if abs(price_change_1h) >= 0.015 or abs(volume_change) >= 0.12:
            regime = "high_vol"
        elif abs(price_change_24h) >= 0.02:
            regime = "trend"
        else:
            regime = "chop"

        regime_weights = {
            "trend": (0.75, 0.10, 0.15),
            "chop": (0.75, 0.15, 0.10),
            "high_vol": (0.65, 0.10, 0.25),
        }
        w_t, w_m, w_b = regime_weights[regime]
        score = w_t * trend_score + w_m * mean_reversion_score + w_b * breakout_score
        return float(np.clip(score, -0.95, 0.95))

    @staticmethod
    def _weighted_opponent_score(model_signals: Dict[str, float], tuned_weights: Dict[str, float]) -> float:
        weight_total = sum(tuned_weights.values())
        if weight_total <= 0:
            return 0.0
        raw = 0.0
        for name, w in tuned_weights.items():
            raw += w * float(model_signals.get(name, 0.0))
        # Normalize by total weight and clip to keep stable score scales.
        return float(np.clip(raw / weight_total, -1.0, 1.0))

    def _optimize_policy_params(
        self,
        samples_by_symbol: Dict[str, List[Dict[str, Any]]],
        tuned_weights: Dict[str, float],
        quality: float,
    ) -> Tuple[Dict[str, float], float, float]:
        if not samples_by_symbol:
            return self.reasoner.get_policy_params(), 0.0, -1e9

        quality_scale = 0.6 + 0.4 * max(0.0, quality)
        if self.technical_mode == "multi":
            if quality < -0.02:
                if self.allow_negative_influence:
                    candidates_influence = [-0.35, -0.28, -0.20, -0.12, -0.06, -0.03, 0.00, 0.03, 0.06]
                else:
                    candidates_influence = [0.00, 0.02, 0.05, 0.08]
            elif quality <= 0:
                candidates_influence = [0.00, 0.02, 0.05, 0.08]
            else:
                candidates_influence = [0.00, 0.03, 0.06, 0.10, 0.15, 0.22]
        else:
            if quality < -0.02:
                if self.allow_negative_influence:
                    candidates_influence = [-0.35, -0.25, -0.16, -0.08, -0.03, 0.00, 0.03, 0.08]
                else:
                    candidates_influence = [0.00, 0.03, 0.08, 0.15]
            elif quality <= 0:
                candidates_influence = [0.00, 0.03, 0.08, 0.15]
            else:
                candidates_influence = [0.00, 0.05, 0.12, 0.20, 0.32, 0.45]
        candidates_gate = [0.0, 0.06, 0.10, 0.14, 0.18]
        candidates_override = [0.35, 0.50, 0.65, 0.80]
        decision_threshold = 0.18

        def _evaluate(influence: float, gate: float, override: float) -> Tuple[float, float]:
            pnl_all: List[float] = []
            turnover_sum = 0.0
            n_steps = 0
            disagree = 0

            for _, samples in samples_by_symbol.items():
                prev_pos = 0
                for s in samples:
                    tech_score = float(s["tech_score"])
                    tech_pos = 1 if tech_score > 0 else (-1 if tech_score < 0 else 0)
                    opp_score = self._weighted_opponent_score(s["model_signals"], tuned_weights) * quality_scale
                    opp_term = opp_score if abs(opp_score) >= gate else 0.0
                    blended = tech_score + influence * opp_term

                    if (
                        influence > 0.0
                        and
                        tech_pos != 0
                        and opp_term != 0.0
                        and np.sign(opp_term) != np.sign(tech_score)
                        and abs(opp_term) >= override
                        and abs(tech_score) < 0.65
                    ):
                        pos = 0
                    else:
                        if blended >= decision_threshold:
                            pos = 1
                        elif blended <= -decision_threshold:
                            pos = -1
                        else:
                            pos = 0

                    ret = float(s["realized_return"])
                    pnl_all.append(pos * ret)
                    turnover_sum += abs(pos - prev_pos)
                    prev_pos = pos
                    n_steps += 1
                    if pos != tech_pos:
                        disagree += 1

            if not pnl_all:
                return -1e9, 0.0

            pnl_arr = np.asarray(pnl_all, dtype=float)
            mean_ret = float(np.mean(pnl_arr))
            std_ret = float(np.std(pnl_arr))
            sharpe_like = (mean_ret / std_ret) * np.sqrt(len(pnl_arr)) if std_ret > 1e-12 else 0.0

            eq = np.cumprod(1.0 + pnl_arr)
            peaks = np.maximum.accumulate(eq)
            drawdowns = np.where(peaks > 0, (peaks - eq) / peaks, 0.0)
            max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0
            turnover_rate = turnover_sum / max(1, n_steps)
            disagreement_rate = disagree / max(1, n_steps)

            disagreement_penalty = 0.0
            if disagreement_rate > 0.30:
                disagreement_penalty = (disagreement_rate - 0.30) * 12.0

            objective = (
                sharpe_like
                - 1.20 * max_dd
                - 0.15 * turnover_rate
                - disagreement_penalty
            )
            return float(objective), float(disagreement_rate)

        best_obj = -1e9
        best_disagreement = 0.0
        best_params = {
            "influence": float(self.reasoner.influence),
            "gate_threshold": float(self.reasoner.gate_threshold),
            "override_threshold": float(self.reasoner.override_threshold),
            "decision_threshold": float(self.reasoner.decision_threshold),
        }

        for influence in candidates_influence:
            for gate in candidates_gate:
                for override in candidates_override:
                    objective, disagreement_rate = _evaluate(influence, gate, override)
                    if disagreement_rate > 0.35:
                        continue
                    if objective > best_obj:
                        best_obj = objective
                        best_disagreement = disagreement_rate
                        best_params = {
                            "influence": float(influence),
                            "gate_threshold": float(gate),
                            "override_threshold": float(override),
                            "decision_threshold": float(decision_threshold),
                        }

        if best_obj <= -1e8:
            return best_params, best_disagreement, best_obj

        return best_params, float(best_disagreement), float(best_obj)

    def _technical_signal(self, state: MarketState) -> TechnicalSignal:
        if not self.use_technical:
            return TechnicalSignal("FLAT", 0.0, "Technical signal disabled")
        if self.technical_mode != "multi":
            return self._single_mode_technical_signal(state)

        regime = self._detect_market_regime(state)
        votes: Dict[str, TechnicalExpertVote] = {
            "trend": self._trend_expert_vote(state),
            "mean_reversion": self._mean_reversion_vote(state),
            "breakout": self._breakout_vote(state),
        }
        dynamic_vote = self._dynamic_strategy_vote(state)
        weights = dict(self._regime_weights.get(regime, self._regime_weights["chop"]))
        expert_score = 0.0
        expert_weight_total = 0.0
        for name in ("trend", "mean_reversion", "breakout"):
            vote = votes[name]
            w = float(weights.get(name, 0.0))
            expert_score += w * float(vote.score)
            expert_weight_total += w
        if expert_weight_total > 0:
            expert_score /= expert_weight_total

        note = "no_dynamic_hold"
        weighted_score = 0.0
        if dynamic_vote is not None:
            dynamic_weight = self._dynamic_blend_weight(dynamic_vote.confidence, regime)
            if regime == "trend" and abs(dynamic_vote.score) >= 0.55:
                dynamic_weight = max(dynamic_weight, 0.82)
            weighted_score = dynamic_weight * float(dynamic_vote.score) + (1.0 - dynamic_weight) * float(expert_score)
            if (
                np.sign(dynamic_vote.score) != np.sign(expert_score)
                and abs(dynamic_vote.score) >= 0.45
                and abs(expert_score) >= 0.45
            ):
                if regime == "trend":
                    weighted_score = 0.85 * float(dynamic_vote.score) + 0.15 * float(expert_score)
                    note = "trend_conflict_dynamic_bias"
                else:
                    weighted_score = 0.0
                    note = "dynamic_expert_conflict_flat"
            else:
                note = f"dynamic_weight={dynamic_weight:.2f}"
        elif regime != "high_vol" and abs(expert_score) >= self._expert_entry_threshold:
            weighted_score = float(expert_score)
            note = "no_dynamic_expert_entry"

        weighted_score = float(np.clip(weighted_score, -0.95, 0.95))

        if weighted_score >= 0.18:
            action = "LONG"
        elif weighted_score <= -0.18:
            action = "SHORT"
        else:
            action = "FLAT"

        confidence = float(np.clip(max(0.20, abs(weighted_score)), 0.0, 0.95))
        if dynamic_vote is None and note == "no_dynamic_expert_entry":
            confidence = float(min(confidence, self._expert_conf_cap))
        vote_parts = [f"{name}:{vote.score:+.2f}" for name, vote in votes.items()]
        if dynamic_vote is not None:
            vote_parts.append(f"dynamic:{dynamic_vote.score:+.2f}")
        reasoning = f"regime={regime}; {note}; expert={expert_score:+.2f}; votes[{', '.join(vote_parts)}]"
        return TechnicalSignal(action=action, confidence=confidence, reasoning=reasoning)

    def _single_mode_technical_signal(self, state: MarketState) -> TechnicalSignal:
        dynamic_vote = self._dynamic_strategy_vote(state)
        if dynamic_vote is not None:
            if dynamic_vote.score >= 0.18:
                action = "LONG"
            elif dynamic_vote.score <= -0.18:
                action = "SHORT"
            else:
                action = "FLAT"
            return TechnicalSignal(action=action, confidence=dynamic_vote.confidence, reasoning=dynamic_vote.reasoning)
        return TechnicalSignal("FLAT", 0.0, "dynamic_strategy unavailable")

    def _dynamic_strategy_vote(self, state: MarketState) -> Optional[TechnicalExpertVote]:
        try:
            from src.dynamic_strategy import calculate_signal

            signal = calculate_signal(state.df)
            action = getattr(signal, "side", "FLAT")
            conf = max(0.0, min(1.0, float(getattr(signal, "confidence", 0.0))))
            score = self._action_to_score(action, conf)
            if abs(score) < 0.05:
                return None
            return TechnicalExpertVote(
                name="dynamic",
                score=score,
                confidence=abs(score),
                reasoning=getattr(signal, "reason", "dynamic_strategy"),
            )
        except Exception:
            return None

    def _trend_expert_vote(self, state: MarketState) -> TechnicalExpertVote:
        ema_fast = self._safe_float(state.indicators.get("ema_50", np.nan), default=np.nan)
        ema_slow = self._safe_float(state.indicators.get("ema_200", np.nan), default=np.nan)
        rsi = self._safe_float(state.indicators.get("rsi", 50.0), default=50.0)

        if not np.isfinite(ema_fast) or not np.isfinite(ema_slow) or state.current_price <= 0:
            return TechnicalExpertVote("trend", 0.0, 0.0, "Missing trend indicators")

        spread = (ema_fast - ema_slow) / max(state.current_price, 1e-9)
        if spread > 0 and rsi < 74:
            score = min(0.85, 0.35 + abs(spread) * 40.0)
            return TechnicalExpertVote("trend", score, abs(score), "EMA spread bullish")
        if spread < 0 and rsi > 26:
            score = -min(0.85, 0.35 + abs(spread) * 40.0)
            return TechnicalExpertVote("trend", score, abs(score), "EMA spread bearish")
        return TechnicalExpertVote("trend", 0.0, 0.0, "No clear trend edge")

    def _mean_reversion_vote(self, state: MarketState) -> TechnicalExpertVote:
        rsi = self._safe_float(state.indicators.get("rsi", 50.0), default=50.0)
        pc1h = self._safe_float(state.price_change_1h, default=0.0)

        if rsi <= 30:
            strength = min(0.75, 0.35 + (30 - rsi) / 40.0 + abs(pc1h) * 3.0)
            return TechnicalExpertVote("mean_reversion", strength, abs(strength), "Oversold rebound setup")
        if rsi >= 70:
            strength = -min(0.75, 0.35 + (rsi - 70) / 40.0 + abs(pc1h) * 3.0)
            return TechnicalExpertVote("mean_reversion", strength, abs(strength), "Overbought reversion setup")
        return TechnicalExpertVote("mean_reversion", 0.0, 0.0, "RSI neutral")

    def _breakout_vote(self, state: MarketState) -> TechnicalExpertVote:
        pc24h = self._safe_float(state.price_change_24h, default=0.0)
        vol_chg = self._safe_float(state.volume_change, default=0.0)
        pc1h = self._safe_float(state.price_change_1h, default=0.0)

        momentum = abs(pc24h) + 0.5 * abs(pc1h)
        if momentum >= 0.03 and vol_chg > 0.0:
            direction = 1.0 if pc24h >= 0 else -1.0
            strength = min(0.85, 0.30 + momentum * 4.0 + min(0.20, vol_chg))
            score = direction * strength
            return TechnicalExpertVote("breakout", score, abs(score), "Price and volume expansion")
        return TechnicalExpertVote("breakout", 0.0, 0.0, "No breakout confirmation")

    def _detect_market_regime(self, state: MarketState) -> str:
        atr = self._safe_float(state.indicators.get("atr", np.nan), default=np.nan)
        price = max(self._safe_float(state.current_price, default=0.0), 1e-9)
        atr_pct = atr / price if np.isfinite(atr) and atr > 0 else 0.0

        pc1h = abs(self._safe_float(state.price_change_1h, default=0.0))
        pc24h = abs(self._safe_float(state.price_change_24h, default=0.0))

        if atr_pct >= 0.02 or pc1h >= 0.015:
            return "high_vol"
        if pc24h >= 0.02:
            return "trend"
        return "chop"

    @staticmethod
    def _action_to_score(action: Any, confidence: float) -> float:
        side = str(action).upper()
        conf = max(0.0, min(1.0, float(confidence)))
        if side == "LONG":
            return conf
        if side == "SHORT":
            return -conf
        return 0.0

    def _signal_to_score(self, signal: TechnicalSignal) -> float:
        return self._action_to_score(signal.action, signal.confidence)

    @staticmethod
    def _dynamic_blend_weight(dynamic_confidence: float, regime: str) -> float:
        base = 0.40 + 0.35 * max(0.0, min(1.0, float(dynamic_confidence)))
        if regime == "trend":
            base += 0.05
        elif regime == "chop":
            base -= 0.05
        return float(np.clip(base, 0.35, 0.75))
