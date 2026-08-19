"""Aggregate multiple v3 ablation runs and compute mean±std across runs."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def collect_v3_runs(results_dir: str = "experiments/results_mm") -> pd.DataFrame:
    pattern = str(Path(results_dir) / "mm_deep_btc_ablation_v3_*/summary.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print("No v3 runs found.", file=sys.stderr)
        return pd.DataFrame()

    frames = []
    for p in paths:
        run_id = Path(p).parent.name
        df = pd.read_csv(p)
        df["run_id"] = run_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_auc", "mean_sharpe"]
    available = [c for c in metrics if c in df.columns]
    if not available:
        # try without prefix
        available = [c for c in ["auc", "sharpe"] if c in df.columns]

    model_col = "model" if "model" in df.columns else df.columns[0]
    rows = []
    for model, grp in df.groupby(model_col):
        row = {"model": model, "n_runs": len(grp)}
        for col in available:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = collect_v3_runs()
    if df.empty:
        return
    print(f"Collected {len(df['run_id'].unique())} runs:")
    for rid in sorted(df['run_id'].unique()):
        print(f"  {rid}")
    print()

    summary = summarize(df)
    print("=== Mean ± Std across runs ===")
    print(summary.to_string(index=False))

    # latex table row output
    print()
    print("=== LaTeX rows (Sharpe mean±std) ===")
    metric_mean = "mean_sharpe_mean" if "mean_sharpe_mean" in summary.columns else "sharpe_mean"
    metric_std  = "mean_sharpe_std"  if "mean_sharpe_std"  in summary.columns else "sharpe_std"
    auc_mean    = "mean_auc_mean"    if "mean_auc_mean"    in summary.columns else "auc_mean"
    for _, row in summary.iterrows():
        sharpe_str = f"${row[metric_mean]:+.3f}_{{\\pm{row[metric_std]:.3f}}}$"
        auc_str    = f"${row[auc_mean]:.3f}$"
        print(f"  {row['model']:<40} & {auc_str} & {sharpe_str} \\\\")


if __name__ == "__main__":
    main()
