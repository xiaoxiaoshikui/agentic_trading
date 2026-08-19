#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def collect_run_dirs(base_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and (p / "results.json").exists()
    )


def sharpe_like(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    std = float(np.std(pnl))
    return float((np.mean(pnl) / std) * np.sqrt(len(pnl))) if std > 1e-12 else 0.0


def max_drawdown(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + pnl)
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def summarize_predictions(df: pd.DataFrame, cost_bps: float) -> dict:
    cost = float(cost_bps) / 10000.0
    position = df["position"].to_numpy(dtype=float)
    returns = df["future_return"].to_numpy(dtype=float)
    pnl = position * returns - np.abs(position) * cost
    nonzero = position != 0.0
    n_trades = int(np.sum(nonzero))
    return {
        "cost_bps": float(cost_bps),
        "n_windows": int(len(df)),
        "n_trades": n_trades,
        "trade_rate": float(np.mean(nonzero)) if len(df) else 0.0,
        "hit_rate": float(np.mean(pnl[nonzero] > 0)) if n_trades else 0.0,
        "sharpe": sharpe_like(pnl),
        "mdd": max_drawdown(pnl),
        "total_return": float(np.sum(pnl)),
        "mean_return": float(np.mean(pnl)) if len(pnl) else 0.0,
    }


def load_trade_predictions(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*_trade_predictions.csv")):
        part = pd.read_csv(path)
        part["run_dir"] = run_dir.name
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def iter_costs(raw: str) -> Iterable[float]:
    for item in raw.split(","):
        item = item.strip()
        if item:
            yield float(item)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_markdown(df: pd.DataFrame, output_md: Path) -> None:
    lines = ["# Fee sensitivity\n"]
    if df.empty:
        lines.append("No trade prediction exports found.\n")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return
    table = df.copy()
    for col in ["sharpe_mean", "sharpe_std", "mdd_mean", "trade_rate_mean", "hit_rate_mean", "total_return_mean"]:
        if col in table:
            table[col] = table[col].map(lambda x: f"{x:+.3f}" if "sharpe" in col or "return" in col else f"{x:.3f}")
    lines.append(dataframe_to_markdown(table))
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fee/slippage sensitivity from trade prediction exports.")
    parser.add_argument("--base-dir", default="experiments/results_mm")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--cost-bps", default="0,5,10,20")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dirs = collect_run_dirs(base_dir, args.prefix)
    costs = list(iter_costs(args.cost_bps))

    rows = []
    for run_dir in run_dirs:
        predictions = load_trade_predictions(run_dir)
        if predictions.empty:
            continue
        predictions = predictions[predictions["split"] == "test"].copy()
        with (run_dir / "results.json").open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        seed = str(payload.get("config", {}).get("seed", ""))
        for (model, run_name), model_df in predictions.groupby(["model", "run_dir"]):
            for cost_bps in costs:
                row = summarize_predictions(model_df, cost_bps)
                row.update({"model": model, "run_dir": run_name, "seed": seed})
                rows.append(row)

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        out = pd.DataFrame()
    else:
        grouped = per_run.groupby(["model", "cost_bps"])[
            ["sharpe", "mdd", "trade_rate", "hit_rate", "total_return", "n_trades"]
        ].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
        out = grouped.reset_index()

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    write_markdown(out, output_md)


if __name__ == "__main__":
    main()
