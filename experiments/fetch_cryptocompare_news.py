"""
Fetch real historical news from CryptoCompare News API and Alternative.me Fear & Greed,
and save to JSONL format compatible with the existing multimodal pipeline.

Usage:
    # Register free API key at https://www.cryptocompare.com/cryptopian/api-keys
    export CRYPTOCOMPARE_API_KEY=your_key_here

    python -m experiments.fetch_cryptocompare_news \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \
        --start-date 2025-01-01T00:00:00Z \
        --end-date 2025-12-31T23:59:59Z \
        --output-dir data/multimodal_history_real \
        --api-key YOUR_KEY

Output JSONL format is identical to build_multimodal_history.py output, so all
downstream pipeline (mm_dataset.py, mm_deep_dataset.py, run_mm_deep_experiment.py)
work unchanged. Only difference: web_intelligence.source = "cryptocompare_real".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# CryptoCompare categories per symbol
SYMBOL_CATEGORIES = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
}

# Alternative.me Fear & Greed API (free, no key needed)
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=365&date_format=iso"


def fetch_fear_greed_history() -> dict[str, dict]:
    """Fetch full Fear & Greed history. Returns {date_str: {index, status}}."""
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        result = {}
        for entry in data:
            date = entry.get("timestamp", "")[:10]  # YYYY-MM-DD
            result[date] = {
                "index": int(entry.get("value", 50)),
                "status": entry.get("value_classification", "neutral").lower(),
            }
        log.info(f"Fear & Greed: {len(result)} daily records fetched")
        return result
    except Exception as e:
        log.warning(f"Fear & Greed fetch failed: {e}")
        return {}


def fetch_news_page(category: str, before_ts: int, api_key: str) -> list[dict]:
    """Fetch one page (50 articles) of news before given unix timestamp."""
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {
        "lang": "EN",
        "categories": category,
        "lTs": before_ts,          # CryptoCompare uses lTs (less-than-timestamp)
        "api_key": api_key,
        "sortOrder": "latest",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("Type") != 100:
        raise RuntimeError(f"CryptoCompare API error: {body.get('Message')}")
    return body.get("Data", [])


def estimate_sentiment(title: str, body: str) -> str:
    """Lightweight keyword-based sentiment (no ML, just signal)."""
    text = (title + " " + body).lower()
    bull_kw = ["surge", "rally", "bull", "gain", "rise", "high", "record", "inflow",
               "approve", "adoption", "positive", "growth", "strong", "breakout"]
    bear_kw = ["crash", "drop", "fall", "decline", "bear", "loss", "sell", "fear",
               "ban", "hack", "negative", "weak", "breakdown", "warning", "concern"]
    bull_score = sum(text.count(w) for w in bull_kw)
    bear_score = sum(text.count(w) for w in bear_kw)
    if bull_score > bear_score:
        return "bullish"
    elif bear_score > bull_score:
        return "bearish"
    return "neutral"


def article_to_record(article: dict, symbol: str, fear_greed: dict) -> dict:
    """Convert a CryptoCompare article dict to pipeline-compatible JSONL record."""
    pub_ts = int(article.get("published_on", 0))
    pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
    timestamp_str = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = pub_dt.strftime("%Y-%m-%d")

    title = article.get("title", "")
    body = article.get("body", "")
    source_name = article.get("source_info", {}).get("name", article.get("source", ""))
    url = article.get("url", "")
    tags = article.get("tags", "")
    categories = article.get("categories", "")

    sentiment = estimate_sentiment(title, body)
    text_blob = f"{title}\n\n{body}".strip()

    # Fear & Greed for this date
    fg = fear_greed.get(date_str, {"index": 50, "status": "neutral"})

    web_intelligence = {
        "symbol": symbol,
        "as_of_date": timestamp_str,
        "source": "cryptocompare_real",
        "news_summary": {
            "news_summary": [
                {
                    "title": title,
                    "body": body[:500],  # truncate for storage
                    "source": source_name,
                    "url": url,
                    "published_on": pub_ts,
                    "sentiment": sentiment,
                    "tags": tags,
                    "categories": categories,
                }
            ],
            "overall_sentiment": sentiment,
            "sentiment_score": 75 if sentiment == "bullish" else (25 if sentiment == "bearish" else 50),
        },
        "fear_greed_index": {
            "index": fg["index"],
            "status": fg["status"],
            "comparison": {},
            "trading_advice": "",
        },
        "whale_movements": {
            "whale_activity": "unknown",
            "net_flow": "unknown",
            "notable_transactions": [],
        },
        "social_sentiment": {
            "overall_sentiment": sentiment,
            "sentiment_score": 75 if sentiment == "bullish" else (25 if sentiment == "bearish" else 50),
            "trending_topics": (tags.split("|")[:3] if tags else []),
        },
        "overall_assessment": {
            "direction": "up" if sentiment == "bullish" else ("down" if sentiment == "bearish" else "neutral"),
            "confidence": 0.6,
            "strength": 0.5,
        },
    }

    return {
        "timestamp": timestamp_str,
        "timestamp_ts": pub_ts,
        "symbol": symbol,
        "entry_price": None,
        "text_blob": text_blob,
        "web_intelligence": web_intelligence,
        "source": "cryptocompare_real",
        "interval": "event",
    }


def fetch_symbol(
    symbol: str,
    start_ts: int,
    end_ts: int,
    api_key: str,
    fear_greed: dict,
    output_path: Path,
    sleep_seconds: float = 0.5,
) -> int:
    """Fetch all news for one symbol in [start_ts, end_ts] and append to output_path."""
    category = SYMBOL_CATEGORIES.get(symbol, symbol.replace("USDT", ""))
    before_ts = end_ts
    total = 0
    seen_ids: set[int] = set()

    with open(output_path, "a", encoding="utf-8") as f:
        while before_ts > start_ts:
            try:
                articles = fetch_news_page(category, before_ts, api_key)
            except Exception as e:
                log.error(f"[{symbol}] fetch failed at ts={before_ts}: {e}")
                time.sleep(5)
                continue

            if not articles:
                log.info(f"[{symbol}] No more articles before ts={before_ts}")
                break

            new_articles = [a for a in articles if int(a.get("published_on", 0)) not in seen_ids]
            if not new_articles:
                break

            # Find oldest article timestamp for pagination
            oldest_ts = min(int(a.get("published_on", before_ts)) for a in new_articles)

            # Guard: if API returned articles newer than our current before_ts (free-tier fallback),
            # we've hit the lookback limit — stop.
            if oldest_ts >= before_ts:
                log.info(f"[{symbol}] Hit API lookback limit at ts={oldest_ts} "
                         f"({datetime.fromtimestamp(oldest_ts, tz=timezone.utc).date()}), stopping.")
                break

            written = 0
            for article in new_articles:
                art_ts = int(article.get("published_on", 0))
                if art_ts < start_ts or art_ts > end_ts:
                    continue
                if art_ts in seen_ids:
                    continue
                seen_ids.add(art_ts)
                record = article_to_record(article, symbol, fear_greed)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            total += written
            log.info(f"[{symbol}] ts={oldest_ts} ({datetime.fromtimestamp(oldest_ts, tz=timezone.utc).date()}) "
                     f"| batch={written} | total={total}")

            before_ts = oldest_ts - 1
            time.sleep(sleep_seconds)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real CryptoCompare news to JSONL")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--start-date", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end-date", default="2025-12-31T23:59:59Z")
    parser.add_argument("--output-dir", default="data/multimodal_history_real")
    parser.add_argument("--api-key", default=os.environ.get("CRYPTOCOMPARE_API_KEY", ""))
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "ERROR: Set CRYPTOCOMPARE_API_KEY env var or pass --api-key.\n"
            "Register free at: https://www.cryptocompare.com/cryptopian/api-keys"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = int(datetime.fromisoformat(args.start_date.replace("Z", "+00:00")).timestamp())
    end_ts = int(datetime.fromisoformat(args.end_date.replace("Z", "+00:00")).timestamp())
    symbols = [s.strip() for s in args.symbols.split(",")]

    log.info("Fetching Fear & Greed history...")
    fear_greed = fetch_fear_greed_history()

    for symbol in symbols:
        date_tag = f"{args.start_date[:10].replace('-','')}_{args.end_date[:10].replace('-','')}"
        out_file = output_dir / f"{symbol}_{date_tag}_real.jsonl"
        log.info(f"=== {symbol} → {out_file} ===")
        n = fetch_symbol(symbol, start_ts, end_ts, args.api_key, fear_greed, out_file, args.sleep_seconds)
        log.info(f"[{symbol}] Done: {n} articles written to {out_file}")


if __name__ == "__main__":
    main()
