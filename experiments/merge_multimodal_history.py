"""
Merge and deduplicate multimodal history files.

Combines new Jan-Sep 2025 English data with old Oct-Dec 2025 data.
When two rows have the same (symbol, timestamp), keeps English rows over Chinese rows.

Usage:
    python -m experiments.merge_multimodal_history \
        --new-glob "data/multimodal_history/*_20250101_20250930.jsonl" \
        --old-glob "data/multimodal_history/*_20251001_20251231.jsonl" \
        --output-dir data/multimodal_history_merged
"""

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _row_language(row: dict) -> str:
    """Return 'english' or 'chinese' based on news text."""
    wi = row.get("web_intelligence", {})
    ns = wi.get("news_summary", {}) or {}
    blobs = ns.get("news_summary", []) or []
    blob_text = " ".join(blobs) if blobs else ""
    return "chinese" if _has_chinese(blob_text) else "english"


def _row_key(row: dict) -> tuple:
    """Return (symbol, timestamp) dedup key."""
    wi = row.get("web_intelligence", {})
    symbol = wi.get("symbol") or row.get("symbol", "UNKNOWN")
    ts = row.get("timestamp") or row.get("as_of_date", "")
    return (symbol, ts)


def _extract_symbol(filepath: str) -> str:
    """Extract symbol from filename like BTCUSDT_20250101_20250930.jsonl."""
    basename = Path(filepath).stem
    m = re.match(r"^([A-Z]+USDT)", basename)
    return m.group(1) if m else "UNKNOWN"


def merge_files(file_paths: list[str], output_path: str) -> dict:
    """
    Merge rows from multiple files, deduplicating by (symbol, timestamp).
    Priority: English > Chinese; later file > earlier file for same language.
    """
    # bucket[key] = {"english": row, "chinese": row}
    buckets: dict[tuple, dict] = defaultdict(dict)

    for fpath in file_paths:
        log.info("Loading %s", fpath)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = _row_key(row)
                lang = _row_language(row)
                # Later files (higher quality) overwrite earlier ones for same lang
                buckets[key][lang] = row

    # Resolve: prefer English row; fall back to Chinese
    merged_rows = []
    stats = {"total": 0, "english_used": 0, "chinese_used": 0}
    for key, langs in sorted(buckets.items(), key=lambda x: x[0][1]):
        row = langs.get("english") or langs.get("chinese")
        if row is None:
            continue
        merged_rows.append(row)
        stats["total"] += 1
        if "english" in langs:
            stats["english_used"] += 1
        else:
            stats["chinese_used"] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multimodal history files")
    parser.add_argument(
        "--new-glob",
        default="data/multimodal_history/*_20250101_20250930.jsonl",
        help="Glob for new Jan-Sep 2025 files",
    )
    parser.add_argument(
        "--old-glob",
        default="data/multimodal_history/*_20251001_20251231.jsonl",
        help="Glob for old Oct-Dec 2025 files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/multimodal_history_merged",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Group files by symbol
    new_files = sorted(glob.glob(args.new_glob))
    old_files = sorted(glob.glob(args.old_glob))

    all_files_by_symbol: dict[str, list[str]] = defaultdict(list)
    for f in new_files + old_files:
        sym = _extract_symbol(f)
        all_files_by_symbol[sym].append(f)

    for sym, files in sorted(all_files_by_symbol.items()):
        output_path = os.path.join(args.output_dir, f"{sym}_20250101_20251231.jsonl")
        log.info("Merging %d file(s) for %s → %s", len(files), sym, output_path)
        stats = merge_files(files, output_path)
        log.info(
            "  %s: total=%d, english_used=%d, chinese_used=%d",
            sym,
            stats["total"],
            stats["english_used"],
            stats["chinese_used"],
        )

    log.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
