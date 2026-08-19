"""
Single-Shot LLM Baseline
========================

LLM generates a strategy once based on initial data, but never iterates.
This tests whether iteration/feedback actually helps.
"""

import os
import requests
import re
import logging
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


SINGLE_SHOT_PROMPT = '''You are an expert quantitative trading strategist.

Generate a complete Python trading strategy that:
1. Uses technical indicators (EMA, RSI, Bollinger Bands, ATR, etc.)
2. Returns a Signal dataclass with side ("LONG", "SHORT", or "FLAT"), confidence (0-1), and reason
3. Handles edge cases (NaN values, insufficient data)
4. Is robust and won't crash

Here is sample market data summary:
- Asset: Cryptocurrency futures
- Timeframe: 15-minute candles
- Data columns: open, high, low, close, volume
- Typical volatility: 1-3% daily

Generate ONLY the Python code, no explanations. The code must include:
1. Import statements
2. Signal dataclass
3. calculate_signal(df: pd.DataFrame) -> Signal function

```python
'''


class SingleShotLLMBaseline:
    """
    Single-shot LLM baseline - generates strategy once, never iterates.

    This tests whether LLM iteration actually helps or if a single
    generation is sufficient.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        seed: int = 42,
        base_url: Optional[str] = None
    ):
        self.model = model
        self.host = host
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.seed = seed
        self.base_url = base_url or os.getenv("DEEPSEEK_API_BASE") or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
        self.iteration = 0
        self.strategy_code = None
        self.generated = False

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "seed": self.seed}
                },
                timeout=180
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return ""

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            if self.base_url:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return ""

    def _clean_code(self, text: str) -> str:
        """Extract Python code from LLM response"""
        # Remove think tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find code without markdown
        code_match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Return as-is if looks like code
        if "def calculate_signal" in text:
            return text.strip()

        return ""

    def _generate_strategy(self) -> str:
        """Generate strategy code using LLM (once only)"""
        logger.info(f"Generating single-shot strategy with {self.model}...")

        # Determine API type
        is_openai = self.model.startswith("gpt-") or self.model.startswith("deepseek-") or bool(self.base_url)

        if is_openai:
            response = self._call_openai(SINGLE_SHOT_PROMPT)
        else:
            response = self._call_ollama(SINGLE_SHOT_PROMPT)

        code = self._clean_code(response)

        if not code or "calculate_signal" not in code:
            logger.warning("Failed to generate valid code, using fallback")
            return self._get_fallback_code()

        return code

    def _get_fallback_code(self) -> str:
        """Fallback strategy if LLM generation fails"""
        return '''
"""
Fallback EMA Strategy (LLM generation failed)
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
        if df is None or len(df) < 200:
            return Signal(side="FLAT", confidence=0.0, reason="Insufficient data")

        close = df["close"]
        ema_fast = ta.trend.ema_indicator(close, window=50)
        ema_slow = ta.trend.ema_indicator(close, window=200)

        ema_f = float(ema_fast.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])

        if not np.isfinite(ema_f) or not np.isfinite(ema_s):
            return Signal(side="FLAT", confidence=0.0, reason="Invalid indicators")

        if ema_f > ema_s:
            return Signal(side="LONG", confidence=0.5, reason="EMA bullish")
        elif ema_f < ema_s:
            return Signal(side="SHORT", confidence=0.5, reason="EMA bearish")

        return Signal(side="FLAT", confidence=0.0, reason="No trend")

    except Exception as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Error: {str(e)}")
'''

    def get_strategy_code(self) -> str:
        """Get strategy code (generates on first call)"""
        if self.strategy_code is None:
            self.strategy_code = self._generate_strategy()
            self.generated = True
        return self.strategy_code

    def evolve(
        self,
        feedback: Dict[str, Any],
        train_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Single-shot never evolves after first generation.

        Args:
            feedback: Performance feedback (ignored after first call)
            train_data: Training data (ignored after first call)

        Returns:
            (strategy_code, evolution_info)
        """
        self.iteration += 1

        if not self.generated:
            self.strategy_code = self._generate_strategy()
            self.generated = True

            return self.strategy_code, {
                "iteration": self.iteration,
                "evolved": True,
                "reason": f"Single-shot generation with {self.model}",
                "code_changed": True,
            }
        else:
            return self.strategy_code, {
                "iteration": self.iteration,
                "evolved": False,
                "reason": "Single-shot baseline - no further evolution",
                "code_changed": False,
            }

    def reset(self):
        """Reset for new experiment run"""
        self.iteration = 0
        self.strategy_code = None
        self.generated = False
