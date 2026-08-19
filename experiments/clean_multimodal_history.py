"""
Filter multimodal history JSONL files using text quality thresholds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List

from experiments.mm_quality import (
    QualityThresholds,
    evaluate_quality,
    extract_quality_fields_from_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean multimodal history JSONL files.")
    parser.add_argument("--history-glob", default="data/multimodal_history/*.jsonl")
    parser.add_argument("--output-dir", default="data/multimodal_history_cleaned")
    parser.add_argument("--min-text-char-len", type=int, default=120)
    parser.add_argument("--min-text-source-count", type=int, default=4)
    parser.add_argument("--max-text-error-count", type=int, default=0)
    parser.add_argument("--max-template-ratio", type=float, default=0.35)
    parser.add_argument("--min-quality-score", type=float, default=0.45)
    parser.add_argument("--max-duplicate-count", type=int, default=3)
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

    staged_rows: List[Dict[str, Any]] = []
    fingerprints: Counter[str] = Counter()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                quality_fields = extract_quality_fields_from_record(record)
                fingerprint = quality_fields.get("text_fingerprint", "")
                if fingerprint:
                    fingerprints[fingerprint] += 1
                staged_rows.append(
                    {
                        "path": path,
                        "record": record,
                        "quality_fields": quality_fields,
                    }
                )

    os.makedirs(args.output_dir, exist_ok=True)
    output_handles: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, Any]] = {}
    try:
        for item in staged_rows:
            path = item["path"]
            record = item["record"]
            quality_fields = item["quality_fields"]
            duplicate_count = fingerprints.get(quality_fields.get("text_fingerprint", ""), 1)
            audit = evaluate_quality(quality_fields, thresholds, duplicate_count=duplicate_count)
            if duplicate_count > args.max_duplicate_count and "duplicate_text_group" not in audit["drop_reasons"]:
                audit["keep"] = 0
                audit["drop_reasons"].append("duplicate_text_group")
                audit["duplicate_count"] = duplicate_count

            basename = os.path.basename(path)
            dest_path = os.path.join(args.output_dir, basename)
            if dest_path not in output_handles:
                output_handles[dest_path] = open(dest_path, "w", encoding="utf-8")
                stats[basename] = {"rows": 0, "kept_rows": 0, "dropped_rows": 0, "drop_reason_counts": Counter()}

            file_stats = stats[basename]
            file_stats["rows"] += 1
            if audit["keep"]:
                enriched = dict(record)
                enriched["quality_audit"] = {
                    **quality_fields,
                    **audit,
                }
                output_handles[dest_path].write(json.dumps(enriched, ensure_ascii=False) + "\n")
                file_stats["kept_rows"] += 1
            else:
                file_stats["dropped_rows"] += 1
                for reason in audit["drop_reasons"]:
                    file_stats["drop_reason_counts"][reason] += 1
    finally:
        for handle in output_handles.values():
            handle.close()

    serializable_stats: Dict[str, Dict[str, Any]] = {}
    for basename, file_stats in stats.items():
        serializable_stats[basename] = {
            "rows": int(file_stats["rows"]),
            "kept_rows": int(file_stats["kept_rows"]),
            "dropped_rows": int(file_stats["dropped_rows"]),
            "keep_rate": float(file_stats["kept_rows"] / max(file_stats["rows"], 1)),
            "drop_reason_counts": dict(sorted(file_stats["drop_reason_counts"].items())),
        }

    metadata = {
        "thresholds": vars(args),
        "files": serializable_stats,
    }
    metadata_path = os.path.join(args.output_dir, "_cleaning_summary.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved cleaned files to {args.output_dir}")


if __name__ == "__main__":
    main()
