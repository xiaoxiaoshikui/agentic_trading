"""
Audit multimodal history JSONL files for text quality issues.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

import pandas as pd

from experiments.mm_quality import (
    QualityThresholds,
    evaluate_quality,
    extract_quality_fields_from_record,
    summarize_quality_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit multimodal history JSONL quality.")
    parser.add_argument("--history-glob", default="data/multimodal_history/*.jsonl")
    parser.add_argument("--output-dir", default="experiments/results_mm_quality")
    parser.add_argument("--min-text-char-len", type=int, default=120)
    parser.add_argument("--min-text-source-count", type=int, default=4)
    parser.add_argument("--max-text-error-count", type=int, default=0)
    parser.add_argument("--max-template-ratio", type=float, default=0.35)
    parser.add_argument("--min-quality-score", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = QualityThresholds(
        min_text_char_len=args.min_text_char_len,
        min_text_source_count=args.min_text_source_count,
        max_text_error_count=args.max_text_error_count,
        max_template_ratio=args.max_template_ratio,
        min_quality_score=args.min_quality_score,
    )
    paths = sorted(glob.glob(args.history_glob))
    if not paths:
        raise FileNotFoundError(f"No history files matched: {args.history_glob}")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fields = extract_quality_fields_from_record(record)
                rows.append(
                    {
                        "path": path,
                        "symbol": record.get("symbol") or (record.get("web_intelligence") or {}).get("symbol"),
                        "timestamp": record.get("timestamp"),
                        **fields,
                    }
                )

    fingerprint_counts = Counter(row["text_fingerprint"] for row in rows if row.get("text_fingerprint"))
    audited_rows: List[Dict[str, Any]] = []
    for row in rows:
        audit = evaluate_quality(
            row,
            thresholds=thresholds,
            duplicate_count=fingerprint_counts.get(row.get("text_fingerprint", ""), 1),
        )
        audited_rows.append({**row, **audit})

    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in audited_rows:
        by_symbol[row.get("symbol") or "UNKNOWN"].append(row)

    summary = {
        "thresholds": vars(args),
        "global": summarize_quality_rows(audited_rows),
        "by_symbol": {symbol: summarize_quality_rows(symbol_rows) for symbol, symbol_rows in sorted(by_symbol.items())},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, "multimodal_history_quality_audit")
    with open(f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    detail_df = pd.DataFrame(audited_rows)
    detail_df.sort_values(["symbol", "timestamp"], inplace=True)
    detail_df.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")

    issues_df = detail_df[detail_df["keep"] == 0].copy()
    issues_df["drop_reasons"] = issues_df["drop_reasons"].apply(lambda x: "|".join(x))
    issues_df.sort_values(["symbol", "text_quality_score", "text_template_ratio"], ascending=[True, True, False], inplace=True)
    issues_df.to_csv(f"{stem}_issues.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detail CSV to {stem}.csv")
    print(f"Saved issue CSV to {stem}_issues.csv")


if __name__ == "__main__":
    main()
