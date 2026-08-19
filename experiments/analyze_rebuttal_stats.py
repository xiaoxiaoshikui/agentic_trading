#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def collect_run_dirs(base_dir: Path, prefix: str) -> List[Path]:
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and (p / "results.json").exists()
    )


def load_results(run_dir: Path) -> dict:
    with (run_dir / "results.json").open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload


def bootstrap_ci(values: List[float], n_boot: int = 20000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        boots.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def load_fold_sharpes(run_dirs: List[Path]) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    for run_dir in run_dirs:
        payload = load_results(run_dir)
        for item in payload.get("results", []):
            model = item["model"]
            out.setdefault(model, [])
            out[model].append([float(fr["test_downstream"]["sharpe_like"]) for fr in item.get("fold_results", [])])
    return out


def matched_fold_average(model_runs: List[List[float]]) -> List[float]:
    if not model_runs:
        return []
    n_folds = min(len(run) for run in model_runs)
    return [
        float(np.mean([run[fold_idx] for run in model_runs if len(run) > fold_idx]))
        for fold_idx in range(n_folds)
    ]


def matched_run_fold_diffs(target_runs: List[List[float]], baseline_runs: List[List[float]]) -> tuple[List[float], int]:
    diffs: List[float] = []
    n_runs = min(len(target_runs), len(baseline_runs))
    for run_idx in range(n_runs):
        target = target_runs[run_idx]
        baseline = baseline_runs[run_idx]
        n_folds = min(len(target), len(baseline))
        diffs.extend(float(target[fold_idx] - baseline[fold_idx]) for fold_idx in range(n_folds))
    return diffs, n_runs


def summarize_deltas(fold_sharpes: Dict[str, List[List[float]]], target: str) -> pd.DataFrame:
    target_runs = fold_sharpes.get(target, [])
    rows = []
    for model, model_runs in sorted(fold_sharpes.items()):
        if model == target:
            continue
        diffs, n_runs = matched_run_fold_diffs(target_runs, model_runs)
        if not diffs:
            continue
        wins = int(sum(d > 0 for d in diffs))
        losses = int(sum(d < 0 for d in diffs))
        ci_low, ci_high = bootstrap_ci(diffs)
        try:
            wilcoxon_p = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        sign_p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else float("nan")
        rows.append({
            "comparison": f"{target} - {model}",
            "mean_delta": float(np.mean(diffs)),
            "boot_ci_low": ci_low,
            "boot_ci_high": ci_high,
            "wins": wins,
            "losses": losses,
            "ties": int(sum(d == 0 for d in diffs)),
            "n_runs": int(n_runs),
            "n_matched_units": int(len(diffs)),
            "wilcoxon_p_greater": wilcoxon_p,
            "sign_test_p_greater": sign_p,
        })
    return pd.DataFrame(rows)


def sharpe_like(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    std = float(np.std(pnl))
    return float((np.mean(pnl) / std) * math.sqrt(len(pnl))) if std > 1e-12 else 0.0


def load_trade_predictions(run_dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*_trade_predictions.csv")):
            part = pd.read_csv(path)
            part["run_dir"] = run_dir.name
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_slice(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    test = df[df["split"] == "test"].copy()
    if "net_pnl" in test.columns:
        test["pnl"] = test["net_pnl"].astype(float)
    else:
        cost_bps = test["cost_bps"].astype(float) if "cost_bps" in test.columns else 0.0
        test["pnl"] = (
            test["position"].astype(float) * test["future_return"].astype(float)
            - test["position"].astype(float).abs() * cost_bps * 1e-4
        )
    for keys, part in test.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        nonzero = part["position"].to_numpy(dtype=float) != 0
        row.update({
            "n_windows": int(len(part)),
            "n_trades": int(np.sum(nonzero)),
            "trade_rate": float(np.mean(nonzero)) if len(part) else 0.0,
            "sharpe": sharpe_like(part["pnl"].to_numpy(dtype=float)),
            "hit_rate": float(np.mean(part.loc[nonzero, "pnl"] > 0)) if np.sum(nonzero) else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    fold_stats = (
        out[out["split"] == "test"]
        .groupby("fold_id")["future_return"]
        .agg(["std", "mean"])
        .rename(columns={"std": "fold_vol", "mean": "fold_mean_return"})
        .reset_index()
    )
    if fold_stats.empty:
        out["vol_regime"] = "unknown"
        out["return_regime"] = "unknown"
        return out
    vol_quantiles = fold_stats["fold_vol"].quantile([1 / 3, 2 / 3]).to_numpy()
    if len(fold_stats) < 3 or vol_quantiles[0] == vol_quantiles[1]:
        fold_stats["vol_regime"] = "all_vol"
    else:
        fold_stats["vol_regime"] = pd.cut(
            fold_stats["fold_vol"],
            bins=[-np.inf, vol_quantiles[0], vol_quantiles[1], np.inf],
            labels=["low_vol", "mid_vol", "high_vol"],
        ).astype(str)
    eps = float(fold_stats["fold_mean_return"].abs().median()) * 0.25
    fold_stats["return_regime"] = np.where(
        fold_stats["fold_mean_return"] > eps,
        "up",
        np.where(fold_stats["fold_mean_return"] < -eps, "down", "sideways"),
    )
    return out.merge(fold_stats[["fold_id", "vol_regime", "return_regime"]], on="fold_id", how="left")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_md(title: str, tables: Dict[str, pd.DataFrame], output_md: Path) -> None:
    lines = [f"# {title}", ""]
    for name, df in tables.items():
        lines.extend([f"## {name}", ""])
        if df.empty:
            lines.extend(["No rows.", ""])
        else:
            lines.extend([dataframe_to_markdown(df), ""])
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rebuttal statistics from deep multimodal runs.")
    parser.add_argument("--base-dir", default="experiments/results_mm")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--target-model", default="cgcma")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    run_dirs = collect_run_dirs(Path(args.base_dir), args.prefix)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fold_sharpes = load_fold_sharpes(run_dirs)
    deltas = summarize_deltas(fold_sharpes, args.target_model)
    deltas.to_csv(output_prefix.with_name(output_prefix.name + "_matched_deltas.csv"), index=False)

    predictions = add_regime_labels(load_trade_predictions(run_dirs))
    per_asset = summarize_slice(predictions, ["model", "symbol"])
    per_vol = summarize_slice(predictions, ["model", "vol_regime"]) if "vol_regime" in predictions else pd.DataFrame()
    per_return = summarize_slice(predictions, ["model", "return_regime"]) if "return_regime" in predictions else pd.DataFrame()
    per_asset.to_csv(output_prefix.with_name(output_prefix.name + "_per_asset.csv"), index=False)
    per_vol.to_csv(output_prefix.with_name(output_prefix.name + "_per_vol_regime.csv"), index=False)
    per_return.to_csv(output_prefix.with_name(output_prefix.name + "_per_return_regime.csv"), index=False)

    write_md(
        "Rebuttal statistics",
        {
            "Matched fold deltas": deltas,
            "Per asset": per_asset,
            "Per volatility regime": per_vol,
            "Per return regime": per_return,
        },
        output_prefix.with_name(output_prefix.name + ".md"),
    )


if __name__ == "__main__":
    main()
