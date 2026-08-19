#!/usr/bin/env python
"""
Summarize local multimodal history coverage across one or more JSONL sources.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multimodal history coverage")
    parser.add_argument(
        "--history-glob",
        nargs="+",
        default=["data/history/*.jsonl", "data/multimodal_history/*.jsonl"],
        help="One or more glob patterns for multimodal history JSONL files",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table")
    return parser.parse_args()


def resolve_paths(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(paths))


def coerce_symbol(record: Dict[str, Any]) -> str:
    web = record.get("web_intelligence") or {}
    return str(web.get("symbol") or record.get("symbol") or "UNKNOWN")


def coerce_timestamp(record: Dict[str, Any]) -> pd.Timestamp | None:
    if record.get("timestamp_ts") is not None:
        try:
            return pd.to_datetime(float(record["timestamp_ts"]), unit="s", utc=True)
        except Exception:
            pass
    if record.get("timestamp"):
        try:
            return pd.to_datetime(record["timestamp"], utc=True)
        except Exception:
            return None
    return None


def summarize(paths: List[str]) -> Dict[str, Any]:
    rows_by_symbol: Dict[str, List[pd.Timestamp]] = defaultdict(list)
    rows_by_source: Dict[str, int] = defaultdict(int)
    total_rows = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                symbol = coerce_symbol(record)
                ts = coerce_timestamp(record)
                if ts is None:
                    continue
                rows_by_symbol[symbol].append(ts)
                rows_by_source[path] += 1
                total_rows += 1

    by_symbol = {}
    for symbol, timestamps in sorted(rows_by_symbol.items()):
        series = pd.Series(sorted(pd.to_datetime(timestamps, utc=True)))
        by_symbol[symbol] = {
            "rows": int(len(series)),
            "unique_timestamps": int(series.nunique()),
            "start": str(series.iloc[0]) if not series.empty else None,
            "end": str(series.iloc[-1]) if not series.empty else None,
        }

    return {
        "total_rows": int(total_rows),
        "n_files": int(len(paths)),
        "files": rows_by_source,
        "by_symbol": by_symbol,
    }


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.history_glob)
    summary = summarize(paths)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"files={summary['n_files']} total_rows={summary['total_rows']}")
    print("symbol\trows\tunique_timestamps\tstart\tend")
    for symbol, stats in summary["by_symbol"].items():
        print(
            f"{symbol}\t{stats['rows']}\t{stats['unique_timestamps']}\t"
            f"{stats['start']}\t{stats['end']}"
        )


if __name__ == "__main__":
    main()
