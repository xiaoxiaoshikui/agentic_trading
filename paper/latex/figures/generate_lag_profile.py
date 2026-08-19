from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "latex" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mm_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig


def load_lags(config_relpath: str) -> np.ndarray:
    config_path = ROOT / config_relpath
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = MultimodalDatasetConfig.from_dict(config["dataset"])
    dataset, _ = MultimodalDatasetBuilder(dataset_config).build()
    return dataset["modality_age_minutes"].to_numpy(dtype=float)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def format_stats(name: str, values: np.ndarray) -> str:
    share_zero = float(np.mean(values == 0.0))
    p95 = float(np.quantile(values, 0.95))
    vmax = float(np.max(values))
    return (
        f"{name}: n={len(values)}\n"
        f"exact-match share={share_zero * 100:.1f}%\n"
        f"95th pct={p95:.1f} min, max={vmax:.1f} min"
    )


def main() -> None:
    raw = load_lags("experiments/configs/mm_text_btc_full_rolling.json")
    cleaned = load_lags("experiments/configs/mm_text_btc_full_cleaned_rolling.json")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)

    raw_x, raw_y = ecdf(raw)
    cleaned_x, cleaned_y = ecdf(cleaned)

    raw_color = "#415A77"
    clean_color = "#111111"

    ax.step(raw_x, raw_y, where="post", color=raw_color, linewidth=1.9, label="Raw full backfill")
    ax.step(cleaned_x, cleaned_y, where="post", color=clean_color, linewidth=1.6, label="Quality-filtered protocol")

    raw_p95 = float(np.quantile(raw, 0.95))
    ax.axvline(raw_p95, color=raw_color, linestyle="--", linewidth=1.0, alpha=0.85)
    ax.text(raw_p95 + 0.25, 0.18, "raw 95th pct", color=raw_color, rotation=90, va="bottom", ha="left")

    ax.set_xlim(-0.1, max(15.2, raw_p95 + 1.0))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Modality lag (minutes)")
    ax.set_ylabel("Cumulative share")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="lower right", frameon=True)
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_edgecolor("#BBBBBB")

    ax.text(
        7.55,
        0.47,
        format_stats("Raw", raw),
        fontsize=7.3,
        color=raw_color,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F7F9FC", "edgecolor": "#C9D4E2", "linewidth": 0.6},
    )
    ax.text(
        0.7,
        0.98,
        format_stats("Cleaned", cleaned),
        fontsize=7.3,
        color=clean_color,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F6F6F6", "edgecolor": "#CFCFCF", "linewidth": 0.6},
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    output_path = FIG_DIR / "lag_profile.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
