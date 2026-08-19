"""
LLM-BO Evolver
=============

LLM-guided Bayesian Optimization over a parameterized strategy template.
Uses OpenAI models when model name starts with "gpt-"; otherwise falls back
to Ollama-compatible API.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import requests
from openai import OpenAI

from .config import ModelConfig

logger = logging.getLogger(__name__)


STRATEGY_TEMPLATE = '''
"""
EMA + RSI Strategy (LLM-BO Template)
Parameters are proposed by LLM-BO.
"""

import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass


@dataclass
class Signal:
    side: str = "FLAT"
    confidence: float = 0.0
    reason: str = ""


def calculate_signal(df: pd.DataFrame) -> Signal:
    try:
        if df is None or len(df) < {ema_slow} + 50:
            return Signal(side="FLAT", confidence=0.0, reason="Insufficient data")

        close = df["close"]
        ema_fast = ta.trend.ema_indicator(close, window={ema_fast})
        ema_slow = ta.trend.ema_indicator(close, window={ema_slow})
        rsi = ta.momentum.rsi(close, window={rsi_period})

        ema_f = float(ema_fast.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        rsi_val = float(rsi.iloc[-1])

        if not all(np.isfinite([ema_f, ema_s, rsi_val])):
            return Signal(side="FLAT", confidence=0.0, reason="Invalid indicators")

        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s

        long_score = 0.0
        short_score = 0.0
        reasons = []

        if uptrend:
            long_score += {ema_weight}
            reasons.append("EMA bullish")
            if rsi_val < {rsi_overbought}:
                long_score += {rsi_weight}
                reasons.append("RSI ok")

        if downtrend:
            short_score += {ema_weight}
            reasons.append("EMA bearish")
            if rsi_val > {rsi_oversold}:
                short_score += {rsi_weight}
                reasons.append("RSI ok")

        min_conf = {min_confidence}
        side = "FLAT"
        conf = 0.0

        if long_score >= min_conf and long_score > short_score:
            side = "LONG"
            conf = min(1.0, long_score)
        elif short_score >= min_conf and short_score > long_score:
            side = "SHORT"
            conf = min(1.0, short_score)

        reason = "; ".join(reasons) if reasons else "No signal"
        return Signal(side=side, confidence=conf, reason=reason)

    except Exception as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Error: {{str(e)}}")
'''


DEFAULT_PARAMS: Dict[str, Any] = {
    "ema_fast": 50,
    "ema_slow": 200,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "ema_weight": 0.3,
    "rsi_weight": 0.2,
    "min_confidence": 0.4,
}


PARAM_RANGES: Dict[str, Tuple[Any, Any]] = {
    "ema_fast": (10, 100),
    "ema_slow": (100, 300),
    "rsi_period": (7, 21),
    "rsi_overbought": (60, 80),
    "rsi_oversold": (20, 40),
    "ema_weight": (0.1, 0.6),
    "rsi_weight": (0.1, 0.5),
    "min_confidence": (0.2, 0.7),
}


class LLMBOEvolver:
    """
    LLM-guided Bayesian Optimization over a fixed strategy template.
    """

    def __init__(self, model: ModelConfig, seed: int = 42):
        env_model = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL")
        self.model_name = env_model or model.model_name or model.model_type.value
        self.host = model.host
        self.temperature = model.temperature
        self.max_tokens = model.max_tokens
        self.seed = seed
        self.rng = random.Random(seed)
        self.params = DEFAULT_PARAMS.copy()
        self.strategy_code = self._render_code(self.params)
        self.iteration = 0

        self.base_url = (
            model.base_url
            or os.getenv("DEEPSEEK_API_BASE")
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.use_openai = self._is_openai_model(self.model_name, self.base_url)
        self.client = None
        if self.use_openai:
            api_key = (
                model.api_key
                or os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def get_strategy_code(self) -> str:
        return self.strategy_code

    def evolve(self, feedback: Dict[str, Any], train_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.iteration += 1
        prompt = self._build_prompt(feedback, train_data)
        response = self._call_llm(prompt)
        params, llm_confidence = self._extract_params(response)

        if not params:
            params = self._mutate_params(self.params)
            reason = "Fallback random mutation"
        else:
            reason = "LLM proposal"

        params = self._normalize_params(params)
        changed = params != self.params
        self.params = params
        self.strategy_code = self._render_code(self.params)

        return self.strategy_code, {
            "iteration": self.iteration,
            "evolved": changed,
            "reason": reason,
            "code_changed": changed,
            "params": self.params.copy(),
            "llm_confidence": llm_confidence,
        }

    def reset(self):
        self.iteration = 0
        self.params = DEFAULT_PARAMS.copy()
        self.strategy_code = self._render_code(self.params)

    def _render_code(self, params: Dict[str, Any]) -> str:
        return STRATEGY_TEMPLATE.format(**params)

    def _build_prompt(self, feedback: Dict[str, Any], train_data: Dict[str, Any]) -> str:
        summary = self._summarize_data(train_data)
        return f"""
You are optimizing a trading strategy by proposing new parameter values.
Return ONLY a JSON object with the following keys:
{list(DEFAULT_PARAMS.keys())}
Optional: include "confidence" between 0 and 1 for your proposal quality.

Parameter ranges:
- ema_fast: 10-100
- ema_slow: 100-300 (must be > ema_fast)
- rsi_period: 7-21
- rsi_overbought: 60-80
- rsi_oversold: 20-40
- ema_weight: 0.1-0.6
- rsi_weight: 0.1-0.5
- min_confidence: 0.2-0.7

Current params:
{json.dumps(self.params, indent=2)}

Feedback from last iteration:
{json.dumps(feedback, indent=2)}

Market data summary:
{summary}

Return only the JSON object and nothing else.
"""

    def _summarize_data(self, train_data: Dict[str, Any]) -> str:
        lines = []
        for symbol, df in (train_data or {}).items():
            try:
                if df is None or len(df) < 50:
                    continue
                close = df["close"]
                ret = close.iloc[-1] / close.iloc[0] - 1
                vol = close.pct_change().dropna().std()
                lines.append(
                    f"- {symbol}: bars={len(df)}, return={ret:.2%}, vol={vol:.2%}, last={close.iloc[-1]:.2f}"
                )
            except Exception:
                continue
        return "\n".join(lines) if lines else "No summary available."

    def _call_llm(self, prompt: str) -> str:
        if self.use_openai:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=min(self.max_tokens, 800),
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                return ""

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature, "seed": self.seed},
                },
                timeout=180,
            )
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return ""
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    def _extract_params(self, text: str) -> Tuple[Optional[Dict[str, Any]], float]:
        if not text:
            return None, 0.0

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, 0.0

        raw = match.group(0)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        try:
            data = json.loads(raw)
        except Exception:
            return None, 0.0

        if not isinstance(data, dict):
            return None, 0.0

        llm_confidence = float(data.get("confidence", 0.0)) if "confidence" in data else 0.0
        params = {k: data.get(k) for k in DEFAULT_PARAMS.keys() if k in data}
        if not params:
            return None, llm_confidence

        return params, llm_confidence

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.params.copy()

        for key, value in params.items():
            if key not in PARAM_RANGES:
                continue
            low, high = PARAM_RANGES[key]
            if isinstance(low, int) and isinstance(high, int):
                try:
                    value = int(float(value))
                except Exception:
                    continue
                value = max(low, min(high, value))
            else:
                try:
                    value = float(value)
                except Exception:
                    continue
                value = max(low, min(high, value))
            normalized[key] = value

        if normalized["ema_fast"] >= normalized["ema_slow"]:
            normalized["ema_slow"] = min(300, normalized["ema_fast"] + 50)

        return normalized

    def _mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mutated = params.copy()
        for key, (low, high) in PARAM_RANGES.items():
            if self.rng.random() < 0.4:
                if isinstance(low, int) and isinstance(high, int):
                    mutated[key] = self.rng.randint(low, high)
                else:
                    mutated[key] = round(self.rng.uniform(low, high), 3)
        if mutated["ema_fast"] >= mutated["ema_slow"]:
            mutated["ema_slow"] = min(300, mutated["ema_fast"] + 50)
        return mutated

    @staticmethod
    def _is_openai_model(model_name: str, base_url: Optional[str]) -> bool:
        if base_url:
            return True
        return model_name.startswith("gpt-") or model_name.startswith("deepseek-")
