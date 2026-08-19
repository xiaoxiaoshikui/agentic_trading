from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_lag_bins(series: pd.Series) -> tuple[np.ndarray, str]:
    lag_max = float(series.max())
    if lag_max <= 15.0:
        upper = int(np.ceil(max(15.0, lag_max + 1.0)))
        return np.arange(0, upper + 1, 1), "1-minute bins"
    upper = int(np.ceil(max(120.0, lag_max + 15.0)))
    return np.arange(0, upper + 15, 15), "15-minute bins"


def make_plot(df: pd.DataFrame, output_path: str) -> None:
    lag = pd.to_numeric(df["lag_minutes"], errors="coerce")
    gate = pd.to_numeric(df["gate_mean"], errors="coerce")
    plot_df = pd.DataFrame({"lag_minutes": lag, "gate_mean": gate}).dropna()
    bins, bin_label = build_lag_bins(plot_df["lag_minutes"])
    plot_df["lag_bin"] = pd.cut(
        plot_df["lag_minutes"],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    grouped = (
        plot_df.groupby("lag_bin", observed=True)
        .agg(
            lag_mid=("lag_minutes", "mean"),
            gate_mean=("gate_mean", "mean"),
            gate_std=("gate_mean", "std"),
            n=("gate_mean", "size"),
        )
        .reset_index(drop=True)
    )
    grouped = grouped[grouped["n"] > 0].copy()
    grouped["gate_std"] = grouped["gate_std"].fillna(0.0)
    sem = grouped["gate_std"] / np.sqrt(grouped["n"].clip(lower=1))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "font.size": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    ax.plot(grouped["lag_mid"], grouped["gate_mean"], color="#C44E52", linewidth=2.2)
    ax.fill_between(
        grouped["lag_mid"],
        grouped["gate_mean"] - sem,
        grouped["gate_mean"] + sem,
        color="#C44E52",
        alpha=0.18,
    )
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Mean gate value")
    ax.set_title(f"Gate Strength vs. Modality Lag ({bin_label})")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gate-vs-lag analysis for CGCMA.")
    parser.add_argument("--input", required=True, help="Path to cgcma_gate_analysis.csv")
    parser.add_argument(
        "--output",
        default="paper/latex/figures/gate_lag_analysis.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_plot(df, args.output)


if __name__ == "__main__":
    main()
