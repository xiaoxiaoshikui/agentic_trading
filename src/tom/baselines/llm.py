from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import OpenAI

from ..base import BaseAgent, Decision, MarketState

logger = logging.getLogger(__name__)


class LLMBaselineAgent(BaseAgent):
    """
    LLM baseline that outputs LONG/SHORT/FLAT from a compact state summary.
    Compatible with OpenAI-compatible endpoints (e.g., DeepSeek) via base_url.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 200,
        name: Optional[str] = None,
        decision_interval: int = 1,
        decision_decay: float = 0.85,
    ):
        env_model = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
        env_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        env_base = os.getenv("DEEPSEEK_API_BASE") or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")

        self.model = model or env_model or "gpt-4o-mini"
        self.api_key = api_key or env_key
        self.base_url = base_url or env_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._name = name or "LLM-Baseline"
        self.decision_interval = max(1, decision_interval)
        self.decision_decay = min(0.99, max(0.5, decision_decay))
        self._call_count = 0
        self._last_decision: Optional[Decision] = None

        # Auto-detect DeepSeek model → use DeepSeek endpoint
        if not self.base_url and "deepseek" in self.model.lower():
            self.base_url = "https://api.deepseek.com"

        self.client = None
        if self.api_key or self.base_url:
            try:
                if self.base_url:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            except Exception as exc:
                logger.warning("Failed to initialize LLM client: %s", exc)
                self.client = None

    @property
    def name(self) -> str:
        return self._name

    def decide(self, state: MarketState) -> Decision:
        if not self.client:
            return Decision(
                action="FLAT",
                confidence=0.0,
                reasoning="LLM client unavailable",
                metadata={},
            )

        # Throttle LLM calls: reuse last decision between intervals
        self._call_count += 1
        if self.decision_interval > 1 and self._last_decision is not None:
            if self._call_count % self.decision_interval != 0:
                decayed_conf = max(0.0, min(1.0, self._last_decision.confidence * self.decision_decay))
                cached = Decision(
                    action=self._last_decision.action,
                    confidence=decayed_conf,
                    reasoning=f"cached({self.decision_decay:.2f})",
                    metadata={"cached": True},
                )
                self._last_decision = cached
                return cached

        prompt = self._build_prompt(state)
        response = self._call_llm(prompt)
        action, confidence, reasoning = self._parse_response(response)

        decision = Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={},
        )
        self._last_decision = decision
        return decision

    def _build_prompt(self, state: MarketState) -> str:
        indicators = state.indicators or {}
        ema_50 = indicators.get("ema_50")
        ema_200 = indicators.get("ema_200")
        trend = "unknown"
        if ema_50 is not None and ema_200 is not None:
            if state.current_price > ema_50 and ema_50 > ema_200:
                trend = "uptrend"
            elif state.current_price < ema_50 and ema_50 < ema_200:
                trend = "downtrend"
            else:
                trend = "range"
        return f"""
You are a trading decision model.
Return ONLY JSON with keys: action (LONG/SHORT/FLAT), confidence (0-1), reasoning.
In strong trends, favor trend continuation. Do NOT short solely because RSI is high or price has risen.

Market snapshot:
- symbol: {state.symbol}
- price: {state.current_price:.2f}
- price_change_1h: {state.price_change_1h:.4f}
- price_change_24h: {state.price_change_24h:.4f}
- price_change_7d: {state.price_change_7d:.4f}
- volume_change: {state.volume_change:.4f}
- trend: {trend}
- ema_50: {ema_50}
- ema_200: {ema_200}
- rsi: {indicators.get("rsi")}
- atr: {indicators.get("atr")}
"""

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ""

    def _parse_response(self, text: str) -> tuple[str, float, str]:
        if not text:
            return "FLAT", 0.0, "Empty LLM response"

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return "FLAT", 0.0, "No JSON found"

        raw = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            data = json.loads(raw)
        except Exception:
            return "FLAT", 0.0, "Invalid JSON response"

        action = str(data.get("action", "FLAT")).upper()
        if action not in {"LONG", "SHORT", "FLAT"}:
            action = "FLAT"
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(data.get("reasoning", "LLM baseline"))
        return action, confidence, reasoning
