#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


STRONG_CONFIG = "experiments/configs/mm_deep_multi_real_news_2026_strong_baselines.json"
ABLATION_CONFIG = "experiments/configs/mm_deep_multi_real_news_2026_cgcma_ablation_extended.json"


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")
        fh.flush()


def run_step(log_path: Path, cmd: list[str]) -> None:
    write_line(log_path, f"[{ts()}] RUN {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        write_line(log_path, f"[{ts()}] FAILED returncode={proc.returncode}: {' '.join(cmd)}")
        raise SystemExit(proc.returncode)
    write_line(log_path, f"[{ts()}] DONE {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential rebuttal experiment driver for MPS/GPU runs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--strong-seeds", nargs="*", type=int, default=[42, 101, 123, 202, 303, 404, 456, 789])
    parser.add_argument("--ablation-seeds", nargs="*", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--skip-strong", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_path = Path(args.log_path) if args.log_path else repo_root / "experiments/results_mm/rebuttal_experiment_driver_latest.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_line(log_path, f"[{ts()}] Starting rebuttal experiment driver device={args.device}")

    if not args.skip_strong:
        for seed in args.strong_seeds:
            write_line(log_path, f"[{ts()}] strong_baselines seed={seed} start")
            run_step(
                log_path,
                [
                    sys.executable,
                    "-m",
                    "experiments.run_mm_deep_experiment",
                    "--config",
                    STRONG_CONFIG,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
            )
            write_line(log_path, f"[{ts()}] strong_baselines seed={seed} done")

    if not args.skip_ablation:
        for seed in args.ablation_seeds:
            write_line(log_path, f"[{ts()}] ablation_extended seed={seed} start")
            run_step(
                log_path,
                [
                    sys.executable,
                    "-m",
                    "experiments.run_mm_deep_experiment",
                    "--config",
                    ABLATION_CONFIG,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
            )
            write_line(log_path, f"[{ts()}] ablation_extended seed={seed} done")

    if not args.skip_analysis:
        write_line(log_path, f"[{ts()}] analysis start")
        analysis_steps = [
            [
                sys.executable,
                "experiments/analyze_fee_sensitivity.py",
                "--base-dir",
                "experiments/results_mm",
                "--prefix",
                "mm_deep_multi_real_news_2026_strong_baselines",
                "--output-csv",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_strong_baselines.csv",
                "--output-md",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_strong_baselines.md",
            ],
            [
                sys.executable,
                "experiments/analyze_rebuttal_stats.py",
                "--base-dir",
                "experiments/results_mm",
                "--prefix",
                "mm_deep_multi_real_news_2026_strong_baselines",
                "--target-model",
                "cgcma",
                "--output-prefix",
                "paper/supplementary_submission_20260401/results/rebuttal_strong_baselines",
            ],
            [
                sys.executable,
                "experiments/analyze_fee_sensitivity.py",
                "--base-dir",
                "experiments/results_mm",
                "--prefix",
                "mm_deep_multi_real_news_2026_cgcma_ablation_extended",
                "--output-csv",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_cgcma_ablation_extended.csv",
                "--output-md",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_cgcma_ablation_extended.md",
            ],
            [
                sys.executable,
                "experiments/analyze_rebuttal_stats.py",
                "--base-dir",
                "experiments/results_mm",
                "--prefix",
                "mm_deep_multi_real_news_2026_cgcma_ablation_extended",
                "--target-model",
                "cgcma",
                "--output-prefix",
                "paper/supplementary_submission_20260401/results/rebuttal_cgcma_ablation_extended",
            ],
        ]
        for cmd in analysis_steps:
            run_step(log_path, cmd)
        write_line(log_path, f"[{ts()}] analysis done")

    write_line(log_path, f"[{ts()}] all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
