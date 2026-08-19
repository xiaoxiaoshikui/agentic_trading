import logging
import requests
import json
from typing import Dict, Any
from dataclasses import dataclass
from src.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class LLMAdvice:
    action: str  # AGREE, DISAGREE, FLAT, LONG, SHORT
    confidence: float  # 0.0 to 1.0
    reason: str


class LLMAdvisor:
    """
    LLM Advisor for enhanced trading decisions and alpha discovery.
    Provides intelligent analysis beyond basic signals.
    """

    def __init__(self):
        self.config = load_config()
        if not self.config.enable_llm:
            raise ValueError("LLM is disabled in config")

        self.model = self.config.openai_model
        self.api_key = self.config.openai_api_key
        self.base_url = self.config.openai_base_url
        if not self.api_key:
            raise ValueError("OpenAI API key not set")

        logger.info(f"LLM Advisor initialized with model: {self.model}")

    def get_advice(self, market_context: str, signal: str, indicators: Dict[str, Any]) -> LLMAdvice:
        """
        Get LLM advice for trading decision.

        Args:
            market_context: Rich market analysis from MarketAnalyst
            signal: Current trading signal (LONG/SHORT/FLAT)
            indicators: Technical indicators dict

        Returns:
            LLMAdvice: Structured advice
        """
        try:
            prompt = self._build_prompt(market_context, signal, indicators)
            response = self._call_llm(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"LLM advice failed: {e}")
            return LLMAdvice(action="AGREE", confidence=0.5, reason="LLM error, fallback to base signal")

    def _get_system_prompt(self) -> str:
        """
        Enhanced system prompt for alpha discovery and trading.
        Focuses on identifying edge cases, anomalies, and alpha signals.
        """
        return """你是一个专业的量化交易AI顾问，擅长发现市场alpha信号。

你的核心职责：
1. 分析市场异常和alpha机会
2. 评估技术信号的可靠性
3. 识别高概率交易时机
4. 提供风险调整的建议

决策原则：
- 关注资金费率、持仓量、波动率的异常
- 识别趋势延续 vs 反转机会
- 考虑市场情绪和流动性
- 平衡风险和收益

输出格式：
返回JSON格式：
{
  "action": "AGREE|DISAGREE|OVERRIDE_LONG|OVERRIDE_SHORT|FLAT",
  "confidence": 0.0-1.0,
  "reason": "详细解释，包括alpha信号分析"
}

行动说明：
- AGREE: 同意基础信号
- DISAGREE: 不同意基础信号，建议FLAT
- OVERRIDE_LONG/SHORT: 覆盖基础信号，强力建议
- FLAT: 保持观望"""

    def _build_prompt(self, market_context: str, signal: str, indicators: Dict[str, Any]) -> str:
        """
        Build rich prompt with market data for alpha discovery.
        """
        prompt = f"""{self._get_system_prompt()}

市场分析上下文：
{market_context}

当前基础信号: {signal}

关键指标:
- 价格: ${indicators.get('close', 0):.2f}
- RSI: {indicators.get('rsi', 0):.1f}
- ATR: {indicators.get('atr', 0):.4f}
- 资金费率: {indicators.get('funding_rate', 0)*100:.3f}%
- 持仓量变化: {indicators.get('oi_change', 0)*100:.1f}%

请分析并提供交易建议，重点关注alpha信号。"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call OpenAI-compatible API for advice.
        """
        endpoint = self._build_endpoint()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }

        response = requests.post(endpoint, headers=headers, json=data, timeout=30)

        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _build_endpoint(self) -> str:
        base_url = (self.base_url or "").strip()
        if not base_url:
            return "https://api.openai.com/v1/chat/completions"
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url.rstrip('/')}/v1/chat/completions"

    def _parse_response(self, response: str) -> LLMAdvice:
        """
        Parse LLM response into structured advice.
        """
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start:end]
            data = json.loads(json_str)

            action = data.get("action", "AGREE")
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "No reason provided")

            # Normalize action
            if action == "OVERRIDE_LONG":
                action = "LONG"
            elif action == "OVERRIDE_SHORT":
                action = "SHORT"
            elif action in ["DISAGREE", "FLAT"]:
                action = "FLAT"

            return LLMAdvice(action=action, confidence=min(confidence, 1.0), reason=reason)

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}, response: {response}")
            return LLMAdvice(action="AGREE", confidence=0.5, reason="Parse error, fallback to base signal")
