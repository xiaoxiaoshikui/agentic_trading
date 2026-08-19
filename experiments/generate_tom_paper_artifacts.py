#!/usr/bin/env python
"""
Generate paper-ready artifacts from a ToM experiment run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_latex_table(df: pd.DataFrame, caption: str, label: str, out_path: Path) -> None:
    latex = df.to_latex(index=False, escape=False)
    wrapped = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )
    out_path.write_text(wrapped, encoding="utf-8")


def _plot_depth(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ordered = df.copy()
    ordered = ordered.sort_values("agent")
    x = range(len(ordered))
    y = ordered["mean_sharpe"].astype(float).values
    yerr = ordered["std_sharpe"].astype(float).values

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, y, yerr=yerr, capsize=6)
    plt.xticks(list(x), ordered["agent"].tolist(), rotation=0)
    plt.ylabel("Mean Sharpe")
    plt.title("ToM Depth Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate(run_dir: Path, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    e1_csv = _safe_load_csv(run_dir / "E1" / "summary_table.csv")
    e2_csv = _safe_load_csv(run_dir / "E2" / "summary_table.csv")
    e3_csv = _safe_load_csv(run_dir / "E3" / "summary_table.csv")

    e1_json = _load_json(run_dir / "E1" / "results.json") if (run_dir / "E1" / "results.json").exists() else {}
    e2_json = _load_json(run_dir / "E2" / "results.json") if (run_dir / "E2" / "results.json").exists() else {}
    e3_json = _load_json(run_dir / "E3" / "results.json") if (run_dir / "E3" / "results.json").exists() else {}

    outputs: Dict[str, str] = {}

    if e1_csv is not None:
        path = output_dir / "main_results.csv"
        e1_csv.to_csv(path, index=False)
        outputs["main_results_csv"] = str(path)
        tex = output_dir / "main_results.tex"
        _save_latex_table(e1_csv, "Main performance (E1)", "tab:tom_main", tex)
        outputs["main_results_tex"] = str(tex)

    if e2_csv is not None:
        path = output_dir / "ablation_results.csv"
        e2_csv.to_csv(path, index=False)
        outputs["ablation_results_csv"] = str(path)
        tex = output_dir / "ablation_results.tex"
        _save_latex_table(e2_csv, "Opponent ablation (E2)", "tab:tom_ablation", tex)
        outputs["ablation_results_tex"] = str(tex)

    if e3_csv is not None:
        path = output_dir / "depth_results.csv"
        e3_csv.to_csv(path, index=False)
        outputs["depth_results_csv"] = str(path)
        tex = output_dir / "depth_results.tex"
        _save_latex_table(e3_csv, "Depth ablation (E3)", "tab:tom_depth", tex)
        outputs["depth_results_tex"] = str(tex)

        depth_plot = output_dir / "depth_plot.png"
        _plot_depth(e3_csv, depth_plot)
        outputs["depth_plot_png"] = str(depth_plot)

    stats_summary = {
        "run_dir": str(run_dir),
        "E1_comparisons": e1_json.get("comparisons", {}),
        "E2_comparisons": e2_json.get("comparisons", {}),
        "E3_comparisons": e3_json.get("comparisons", {}),
    }
    stats_path = output_dir / "stat_tests.json"
    stats_path.write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    outputs["stat_tests_json"] = str(stats_path)

    md_lines = [
        "# ToM Paper Artifacts",
        "",
        f"- Source run: `{run_dir}`",
        "",
        "## Files",
    ]
    for key, value in outputs.items():
        md_lines.append(f"- `{key}`: `{value}`")
    md_lines.append("")
    md_path = output_dir / "README.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    outputs["readme_md"] = str(md_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ToM paper artifacts from one run directory")
    parser.add_argument("--run-dir", required=True, help="Path to run dir, e.g. experiments/results_tom/tom_minimal_xxx")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <run-dir>/paper_artifacts",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "paper_artifacts"
    outputs = generate(run_dir=run_dir, output_dir=output_dir)

    print("Generated artifacts:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
