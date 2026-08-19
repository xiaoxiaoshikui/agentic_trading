#!/usr/bin/env python
"""
Backfill historical multimodal event snapshots for multiple symbols.

Each output row is a lightweight JSONL event compatible with the multimodal
dataset builders in this repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from experiments.data_loader import DataLoader
from src.web_intelligence import WebIntelligenceAgent

logger = logging.getLogger(__name__)


@dataclass
class BackfillConfig:
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str
    step_hours: int
    output_dir: str
    model: str
    sleep_seconds: float
    cache_dir: str
    limit: int
    parallel_symbols: int
    api_base: str = ""
    api_key: str = ""


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> BackfillConfig:
    parser = argparse.ArgumentParser(description="Backfill multimodal history with web intelligence snapshots")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--start-date", required=True, help="UTC start date, e.g. 2025-10-01T00:00:00Z")
    parser.add_argument("--end-date", required=True, help="UTC end date, e.g. 2025-12-31T00:00:00Z")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--step-hours", type=int, default=4)
    parser.add_argument("--output-dir", default="data/multimodal_history")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="experiments/data_cache")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of events to process")
    parser.add_argument("--parallel-symbols", type=int, default=1, help="Number of per-symbol workers")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", ""), help="OpenAI-compatible API base URL (e.g. https://api.deepseek.com)")
    parser.add_argument("--api-key", default="", help="API key override (defaults to DEEPSEEK_API_KEY env var when --api-base is set)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    setup_logging(verbose=not args.quiet)
    api_key = args.api_key or (os.environ.get("DEEPSEEK_API_KEY", "") if args.api_base else "")
    return BackfillConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        step_hours=int(args.step_hours),
        output_dir=args.output_dir,
        model=args.model,
        sleep_seconds=float(args.sleep_seconds),
        cache_dir=args.cache_dir,
        limit=int(args.limit),
        parallel_symbols=int(args.parallel_symbols),
        api_base=args.api_base,
        api_key=api_key,
    )


def to_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def build_schedule(start_date: str, end_date: str, step_hours: int) -> List[pd.Timestamp]:
    start_ts = to_utc_timestamp(start_date)
    end_ts = to_utc_timestamp(end_date)
    schedule = pd.date_range(start=start_ts, end=end_ts, freq=f"{step_hours}h", tz="UTC")
    return list(schedule)


def estimate_n_bars(start_date: str, end_date: str, interval: str) -> int:
    start_ts = to_utc_timestamp(start_date)
    end_ts = to_utc_timestamp(end_date)
    delta_minutes = int((end_ts - start_ts).total_seconds() // 60)
    interval_minutes = interval_to_minutes(interval)
    return max(500, int(delta_minutes / max(1, interval_minutes)) + 500)


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported interval: {interval}")


def load_price_frames(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    interval: str,
    cache_dir: str,
) -> Dict[str, pd.DataFrame]:
    loader = DataLoader(cache_dir=cache_dir)
    data = loader.load_data(
        symbols=list(symbols),
        interval=interval,
        n_bars=estimate_n_bars(start_date, end_date, interval),
        force_download=True,
        end_time=end_date,
    )
    frames: Dict[str, pd.DataFrame] = {}
    start_ts = to_utc_timestamp(start_date).tz_localize(None)
    end_ts = to_utc_timestamp(end_date).tz_localize(None)
    for symbol, df in data.items():
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frames[symbol] = frame[(frame.index >= start_ts) & (frame.index <= end_ts)].sort_index()
    return frames


def get_output_path(output_dir: str, symbol: str, start_date: str, end_date: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    start_tag = to_utc_timestamp(start_date).strftime("%Y%m%d")
    end_tag = to_utc_timestamp(end_date).strftime("%Y%m%d")
    return os.path.join(output_dir, f"{symbol}_{start_tag}_{end_tag}.jsonl")


def load_existing_keys(path: str) -> Set[Tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    keys: Set[Tuple[str, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            symbol = obj.get("symbol") or (obj.get("web_intelligence") or {}).get("symbol")
            ts = obj.get("timestamp")
            if symbol and ts:
                keys.add((symbol, ts))
    return keys


def lookup_price(price_df: pd.DataFrame, event_time: pd.Timestamp) -> float:
    event_time = pd.Timestamp(event_time).tz_localize(None) if pd.Timestamp(event_time).tzinfo else pd.Timestamp(event_time)
    eligible = price_df.loc[price_df.index <= event_time]
    if eligible.empty:
        raise ValueError(f"No price available at or before {event_time}")
    return float(eligible["close"].iloc[-1])


def write_row(path: str, row: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_blocking_error(value: object, depth: int = 0) -> bool:
    if depth > 6 or value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return "insufficient_quota" in lowered or "error code: 429" in lowered
    if isinstance(value, list):
        return any(has_blocking_error(item, depth + 1) for item in value[:20])
    if isinstance(value, dict):
        if "error" in {str(key).strip().lower() for key in value.keys()}:
            return True
        return any(has_blocking_error(item, depth + 1) for item in value.values())
    return False


def run_symbol_backfill(symbol: str, cfg: BackfillConfig, verbose: bool = True) -> Dict[str, int]:
    setup_logging(verbose=verbose)
    schedule = build_schedule(cfg.start_date, cfg.end_date, cfg.step_hours)
    price_frames = load_price_frames([symbol], cfg.start_date, cfg.end_date, cfg.interval, cfg.cache_dir)
    price_df = price_frames[symbol]
    agent = WebIntelligenceAgent(
        model=cfg.model,
        api_key=cfg.api_key or None,
        api_base=cfg.api_base or None,
    )
    output_path = get_output_path(cfg.output_dir, symbol, cfg.start_date, cfg.end_date)
    existing = load_existing_keys(output_path)
    logger.info("Backfilling %s into %s (%s existing rows)", symbol, output_path, len(existing))

    processed = 0
    failures = 0
    for event_time in schedule:
        ts_iso = event_time.isoformat().replace("+00:00", "Z")
        key = (symbol, ts_iso)
        if key in existing:
            continue
        if cfg.limit and processed >= cfg.limit:
            logger.info("Reached limit=%s for %s, stopping", cfg.limit, symbol)
            break

        try:
            entry_price = lookup_price(price_df, event_time)
            web_intelligence = agent.get_comprehensive_analysis(symbol, as_of_date=ts_iso)
            if has_blocking_error(web_intelligence):
                failures += 1
                logger.warning("Skipping %s @ %s due to blocking web-intelligence error", symbol, ts_iso)
                continue
            row = {
                "timestamp": ts_iso,
                "timestamp_ts": pd.Timestamp(event_time).timestamp(),
                "symbol": symbol,
                "entry_price": entry_price,
                "web_intelligence": web_intelligence,
                "source": "historical_backfill",
                "interval": cfg.interval,
            }
            write_row(output_path, row)
            processed += 1
            logger.info("Wrote %s @ %s", symbol, ts_iso)
            if cfg.sleep_seconds > 0:
                time.sleep(cfg.sleep_seconds)
        except Exception as exc:
            failures += 1
            logger.warning("Failed %s @ %s: %s", symbol, ts_iso, exc)

    logger.info("Finished %s: wrote %s rows, failures=%s", symbol, processed, failures)
    return {"symbol": symbol, "processed": processed, "failures": failures}


def main() -> None:
    cfg = parse_args()
    schedule = build_schedule(cfg.start_date, cfg.end_date, cfg.step_hours)
    logger.info(
        "Prepared %s schedule points across %s symbols (%s target events)",
        len(schedule),
        len(cfg.symbols),
        len(schedule) * len(cfg.symbols),
    )
    max_workers = max(1, min(cfg.parallel_symbols, len(cfg.symbols)))
    if max_workers == 1:
        results = [run_symbol_backfill(symbol, cfg, verbose=True) for symbol in cfg.symbols]
    else:
        logger.info("Running symbol backfill with %s workers", max_workers)
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_symbol_backfill, symbol, cfg, False)
                for symbol in cfg.symbols
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    processed = int(sum(item["processed"] for item in results))
    failures = int(sum(item["failures"] for item in results))
    logger.info("Backfill finished: wrote %s rows, failures=%s", processed, failures)


if __name__ == "__main__":
    main()
