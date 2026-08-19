#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def collect_runs(base_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix) and (p / "summary.csv").exists()]
    )


def summarize_runs(run_dirs: List[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        summary = pd.read_csv(run_dir / "summary.csv")
        if summary.empty:
            continue
        for row_idx, record in summary.reset_index(drop=True).iterrows():
            record = record.to_dict()
            record["run_dir"] = run_dir.name
            record["run_row"] = int(row_idx)
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "macro_f1",
        "auc",
        "brier",
        "trade_rate",
        "hit_rate",
        "downstream_sharpe",
        "downstream_mdd",
        "downstream_total_return",
        "cost_bps",
        "long_threshold",
        "short_threshold",
    ]
    available = [c for c in metrics if c in df.columns]
    grouped = df.groupby("model")[available].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate supplementary multimodal experiment runs.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--prefix", required=True, help="Run directory prefix, e.g. mm_deep_multi_real_news_2026_pw_baseline")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dirs = collect_runs(base_dir, args.prefix)
    per_run = summarize_runs(run_dirs)
    agg = aggregate_by_model(per_run) if not per_run.empty else pd.DataFrame()

    agg.to_csv(args.output_csv, index=False)
    payload: Dict[str, object] = {
        "prefix": args.prefix,
        "n_runs": int(len(run_dirs)),
        "run_dirs": [p.name for p in run_dirs],
        "per_run_records": per_run.to_dict(orient="records"),
        "aggregated": agg.to_dict(orient="records"),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
