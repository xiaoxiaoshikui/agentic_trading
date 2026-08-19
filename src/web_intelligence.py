"""
Web intelligence utilities backed by the OpenAI Responses API.

This module supports both real-time and historical-style queries by allowing
callers to specify an ``as_of_date`` cutoff in prompts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class WebIntelligenceAgent:
    """Collect market news, sentiment, and whale-flow summaries with web search."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, api_base: Optional[str] = None):
        if api_base:
            self.client = OpenAI(api_key=api_key or "sk-placeholder", base_url=api_base)
        elif api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        self.model = model
        # responses.create (web_search tool) only works with native OpenAI; use chat.completions for other backends
        self.use_chat_only = api_base is not None
        logger.info("Initialized web intelligence agent with model=%s use_chat_only=%s", model, self.use_chat_only)

    def search_market_news(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        coin = self._symbol_to_coin(symbol)
        prompt = f"""Use Web Search to analyze market news for {coin} cryptocurrency.

{self._build_temporal_guardrail(as_of_date)}

Search for:
1. Major news in the last 24 hours relative to the cutoff date
2. Regulatory updates
3. Institutional movements such as ETF flows and treasury allocations
4. Macroeconomic factors affecting crypto

Return strict JSON in English only:
{{
  "news_summary": ["News item 1", "News item 2"],
  "overall_sentiment": "bullish" | "bearish",
  "key_factors": ["Factor 1", "Factor 2"],
  "trading_note": "Directional trading note"
}}"""
        return self._responses_query(prompt, fallback=self._fallback_search(symbol, as_of_date))

    def get_fear_greed_index(self, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        prompt = f"""Search for the crypto Fear & Greed Index.

{self._build_temporal_guardrail(as_of_date)}

Return strict JSON in English only. index must be a specific value (not 0 or 50). status must be one of: extreme_fear, fear, greed, extreme_greed (never neutral):
{{
  "index": <specific 1-100 value>,
  "status": "extreme_fear" | "fear" | "greed" | "extreme_greed",
  "comparison": {{"yesterday": <value>, "last_week": <value>}},
  "trading_advice": "Directional advice"
}}"""
        return self._responses_query(prompt, fallback={"index": 50, "status": "neutral"})

    def get_whale_movements(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        coin = self._symbol_to_coin(symbol)
        prompt = f"""Search for large whale movements for {coin}.

{self._build_temporal_guardrail(as_of_date)}

Look for:
1. Large transfers greater than roughly $10M
2. Exchange inflows and outflows
3. Notable wallet activity

Return strict JSON in English only. net_flow must be exchange_inflow or exchange_outflow (never neutral):
{{
  "whale_activity": "low" | "medium" | "high",
  "net_flow": "exchange_inflow" | "exchange_outflow",
  "notable_transactions": ["Tx 1", "Tx 2"],
  "interpretation": "Directional interpretation"
}}"""
        return self._responses_query(prompt, fallback={"whale_activity": "unknown", "net_flow": "neutral"})

    def get_social_sentiment(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        coin = self._symbol_to_coin(symbol)
        prompt = f"""Search for social sentiment for {coin} across Twitter/X, Reddit, Telegram, and major crypto communities.

{self._build_temporal_guardrail(as_of_date)}

Return strict JSON in English only. overall_sentiment must be bullish or bearish (never neutral). sentiment_score must be non-zero:
{{
  "overall_sentiment": "bullish" | "bearish",
  "sentiment_score": <non-zero value between -100 and 100>,
  "trending_topics": ["Topic 1", "Topic 2"],
  "kol_opinions": "KOL opinions summary",
  "warnings": "Risk warnings"
}}"""
        return self._responses_query(prompt, fallback={"overall_sentiment": "neutral", "sentiment_score": 0})

    def get_comprehensive_analysis(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Collecting comprehensive web intelligence for %s as_of=%s", symbol, as_of_date or "latest")
        coin = self._symbol_to_coin(symbol)
        prompt = f"""{"Use Web Search to build" if not self.use_chat_only else "Based on your knowledge, build"} a comprehensive market-intelligence snapshot for {coin}.

{self._build_temporal_guardrail(as_of_date)}

IMPORTANT RULES — you must follow these strictly:
- overall_sentiment must be "bullish" or "bearish" only — "neutral" is not allowed
- fear_greed status must be one of: "extreme_fear", "fear", "greed", "extreme_greed" — "neutral" is not allowed
- whale net_flow must be "exchange_inflow" or "exchange_outflow" — "neutral" is not allowed
- social overall_sentiment must be "bullish" or "bearish" — "neutral" is not allowed
- fear_greed index must be a specific number (not 0 or 50); estimate based on actual market sentiment for this period
- If information is sparse, give a directional lean based on the overall crypto market trend for this period
- All text fields must be in English only

You must cover:
1. News and regulation
2. Fear & Greed style market sentiment
3. Whale movements and exchange flow
4. Social sentiment from major crypto communities

Return strict JSON in English with this schema:
{{
  "symbol": "{symbol}",
  "news_summary": {{
    "news_summary": ["Specific news item 1", "Specific news item 2"],
    "overall_sentiment": "bullish" | "bearish",
    "key_factors": ["Key factor 1", "Key factor 2"],
    "trading_note": "Directional trading note"
  }},
  "fear_greed_index": {{
    "index": <specific 1-100 value, not 0 or 50>,
    "status": "extreme_fear" | "fear" | "greed" | "extreme_greed",
    "comparison": {{"yesterday": <value>, "last_week": <value>}},
    "trading_advice": "Specific directional advice"
  }},
  "whale_movements": {{
    "whale_activity": "low" | "medium" | "high",
    "net_flow": "exchange_inflow" | "exchange_outflow",
    "notable_transactions": ["Specific tx 1", "Specific tx 2"],
    "interpretation": "Directional interpretation"
  }},
  "social_sentiment": {{
    "overall_sentiment": "bullish" | "bearish",
    "sentiment_score": <non-zero value between -100 and 100>,
    "trending_topics": ["Topic 1", "Topic 2"],
    "kol_opinions": "Specific KOL opinion summary",
    "warnings": "Specific risk warnings"
  }}
}}"""
        combined = self._responses_query(prompt, fallback={"symbol": symbol})
        if all(key in combined for key in ("news_summary", "fear_greed_index", "whale_movements", "social_sentiment")):
            combined["symbol"] = symbol
            combined["as_of_date"] = as_of_date
            combined["overall_assessment"] = self._synthesize_intelligence(
                combined.get("news_summary", {}),
                combined.get("fear_greed_index", {}),
                combined.get("whale_movements", {}),
                combined.get("social_sentiment", {}),
            )
            return combined

        news = self.search_market_news(symbol, as_of_date=as_of_date)
        fear_greed = self.get_fear_greed_index(as_of_date=as_of_date)
        whale = self.get_whale_movements(symbol, as_of_date=as_of_date)
        social = self.get_social_sentiment(symbol, as_of_date=as_of_date)
        return {
            "symbol": symbol,
            "as_of_date": as_of_date,
            "news_summary": news,
            "fear_greed_index": fear_greed,
            "whale_movements": whale,
            "social_sentiment": social,
            "overall_assessment": self._synthesize_intelligence(news, fear_greed, whale, social),
        }

    def _responses_query(self, prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_chat_only:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                text = response.choices[0].message.content
                parsed = self._parse_text_as_json(text)
                if isinstance(parsed, dict) and len(parsed) > 1:
                    return parsed
                result = dict(fallback)
                result["raw_response"] = text
                return result
            except Exception as exc:
                logger.warning("Chat completion failed: %s", exc)
                result = dict(fallback)
                result["error"] = str(exc)
                return result
        try:
            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            return self._parse_response(response)
        except Exception as exc:
            logger.warning("Web intelligence query failed: %s", exc)
            result = dict(fallback)
            result.setdefault("error", str(exc))
            return result

    def _fallback_search(self, symbol: str, as_of_date: Optional[str]) -> Dict[str, Any]:
        coin = self._symbol_to_coin(symbol)
        as_of_note = f"Treat {as_of_date} as the analysis cutoff." if as_of_date else ""
        prompt = f"""As a crypto analyst, summarize {coin}.

{as_of_note}
This fallback does not use web search, so make it explicit that the answer is knowledge-based.

Return strict JSON in English only. overall_sentiment must be "bullish" or "bearish" (never neutral):
{{
  "news_summary": ["Specific item 1"],
  "overall_sentiment": "bullish" | "bearish",
  "key_factors": ["Factor 1"],
  "trading_note": "Directional knowledge-based note"
}}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text = response.choices[0].message.content
            parsed = self._parse_text_as_json(text)
            if isinstance(parsed, dict):
                parsed.setdefault("note", "knowledge_based_fallback")
            return parsed
        except Exception as exc:
            return {"error": str(exc), "note": "knowledge_based_fallback_failed"}

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        if hasattr(response, "output"):
            for item in response.output:
                content_list = getattr(item, "content", None) or []
                for content in content_list:
                    text = getattr(content, "text", None)
                    if text:
                        parsed = WebIntelligenceAgent._parse_text_as_json(text)
                        if isinstance(parsed, dict):
                            return parsed
                        return {"raw_response": text}
        return {"raw_response": str(response)}

    @staticmethod
    def _parse_text_as_json(text: str) -> Dict[str, Any]:
        candidate = text.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"raw_response": text}

    @staticmethod
    def _symbol_to_coin(symbol: str) -> str:
        return symbol.replace("USDT", "").replace("USD", "")

    @staticmethod
    def _build_temporal_guardrail(as_of_date: Optional[str]) -> str:
        if not as_of_date:
            return "Focus on the latest available information."
        try:
            dt = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
            date_text = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_text = as_of_date
        return (
            f"IMPORTANT TIME CONSTRAINT: treat {date_text} as the analysis cutoff date. "
            f"Only use information that would have been public on or before {date_text}. "
            f"If exact same-day information is sparse, prefer the closest earlier reports and say so explicitly."
        )

    @staticmethod
    def _synthesize_intelligence(
        news: Dict[str, Any],
        fear_greed: Dict[str, Any],
        whale: Dict[str, Any],
        social: Dict[str, Any],
    ) -> Dict[str, Any]:
        signals = []

        news_sentiment = news.get("sentiment", news.get("overall_sentiment", "neutral"))
        if str(news_sentiment).lower() in {"bullish", "positive", "看多"}:
            signals.append(1)
        elif str(news_sentiment).lower() in {"bearish", "negative", "看空"}:
            signals.append(-1)
        else:
            signals.append(0)

        fg_index = fear_greed.get("index", fear_greed.get("current_value", 50))
        if isinstance(fg_index, (int, float)):
            if fg_index < 45:
                signals.append(1)
            elif fg_index > 55:
                signals.append(-1)
            else:
                signals.append(0)

        net_flow = whale.get("net_flow", "neutral")
        if net_flow == "exchange_outflow":
            signals.append(1)
        elif net_flow == "exchange_inflow":
            signals.append(-1)
        else:
            signals.append(0)

        social_sentiment = social.get("overall_sentiment", "neutral")
        if social_sentiment == "bullish":
            signals.append(1)
        elif social_sentiment == "bearish":
            signals.append(-1)
        else:
            signals.append(0)

        avg_signal = sum(signals) / len(signals) if signals else 0.0
        if avg_signal > 0.3:
            direction = "bullish"
            confidence = min(avg_signal, 1.0)
        elif avg_signal < -0.3:
            direction = "bearish"
            confidence = min(abs(avg_signal), 1.0)
        else:
            direction = "neutral"
            confidence = 0.5

        recommendation = {
            "bullish": "网络情报看多",
            "bearish": "网络情报看空",
            "neutral": "网络情报中性",
        }[direction]
        return {
            "direction": direction,
            "confidence": round(float(confidence), 2),
            "signal_breakdown": {
                "news": signals[0] if len(signals) > 0 else 0,
                "fear_greed": signals[1] if len(signals) > 1 else 0,
                "whale": signals[2] if len(signals) > 2 else 0,
                "social": signals[3] if len(signals) > 3 else 0,
            },
            "recommendation": recommendation,
        }


def create_web_intelligence_agent(model: str = "gpt-4o-mini") -> WebIntelligenceAgent:
    return WebIntelligenceAgent(model=model)
