#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STRONG_CONFIG = "experiments/configs/mm_deep_multi_real_news_2026_rebuttal_strong_targeted.json"
SHUFFLE_CONFIG = "experiments/configs/mm_deep_multi_real_news_2026_rebuttal_shuffle_ablation.json"
FULL_PILOT_PREFIX = "mm_deep_multi_real_news_2026_strong_baselines"
TARGET_PREFIX = "mm_deep_multi_real_news_2026_rebuttal_strong_targeted"
SHUFFLE_PREFIX = "mm_deep_multi_real_news_2026_rebuttal_shuffle_ablation"
EXPECTED_FULL_TRADE_FILES = {
    "price_only_trade_predictions.csv",
    "patchtst_trade_predictions.csv",
    "itransformer_lite_trade_predictions.csv",
    "bilstm_trade_predictions.csv",
    "early_fusion_trade_predictions.csv",
    "mult_trade_predictions.csv",
    "tfn_trade_predictions.csv",
    "cgcma_trade_predictions.csv",
}


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


def is_complete_run(run_dir: Path) -> bool:
    return (
        (run_dir / "summary.csv").exists()
        and (run_dir / "results.json").exists()
        and EXPECTED_FULL_TRADE_FILES.issubset({p.name for p in run_dir.glob("*_trade_predictions.csv")})
    )


def latest_complete_full_pilot(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob(f"{FULL_PILOT_PREFIX}_*_seed42"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.is_dir() and is_complete_run(candidate):
            return candidate
    return None


def process_table() -> list[tuple[int, str]]:
    proc = subprocess.run(["ps", "aux"], text=True, capture_output=True, check=True)
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split(None, 10)
        if len(parts) >= 11:
            try:
                rows.append((int(parts[1]), parts[10]))
            except ValueError:
                continue
    return rows


def stop_old_full_driver(log_path: Path, current_pid: int) -> None:
    for pid, command in process_table():
        if pid == current_pid:
            continue
        if "run_rebuttal_experiment_driver.py" in command:
            write_line(log_path, f"[{ts()}] stopping old full driver pid={pid}: {command}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for pid, command in process_table():
        if pid == current_pid:
            continue
        if "run_rebuttal_experiment_driver.py" in command:
            write_line(log_path, f"[{ts()}] force-stopping old full driver pid={pid}: {command}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if "experiments.run_mm_deep_experiment" in command and FULL_PILOT_PREFIX in command and "--seed 42" not in command:
            write_line(log_path, f"[{ts()}] stopping accidental full follow-up pid={pid}: {command}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def wait_for_full_seed42(log_path: Path, results_dir: Path, poll_seconds: int) -> Path:
    while True:
        completed = latest_complete_full_pilot(results_dir)
        if completed is not None:
            write_line(log_path, f"[{ts()}] full seed42 complete: {completed}")
            return completed
        write_line(log_path, f"[{ts()}] waiting for full seed42 completion")
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reviewer-targeted rebuttal experiments after full seed42 pilot.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--strong-seeds", nargs="*", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--shuffle-seeds", nargs="*", type=int, default=[42, 123, 456, 789])
    parser.add_argument("--log-path", default="experiments/results_mm/rebuttal_targeted_driver_latest.log")
    parser.add_argument("--results-dir", default="experiments/results_mm")
    parser.add_argument("--wait-full-seed42", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-strong", action="store_true")
    parser.add_argument("--skip-shuffle", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    write_line(log_path, f"[{ts()}] starting targeted driver device={args.device}")

    if args.wait_full_seed42:
        wait_for_full_seed42(log_path, results_dir, args.poll_seconds)
        stop_old_full_driver(log_path, os.getpid())

    if not args.skip_strong:
        for seed in args.strong_seeds:
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

    if not args.skip_shuffle:
        for seed in args.shuffle_seeds:
            run_step(
                log_path,
                [
                    sys.executable,
                    "-m",
                    "experiments.run_mm_deep_experiment",
                    "--config",
                    SHUFFLE_CONFIG,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
            )

    if not args.skip_analysis:
        steps = [
            [
                sys.executable,
                "experiments/analyze_fee_sensitivity.py",
                "--base-dir",
                str(results_dir),
                "--prefix",
                TARGET_PREFIX,
                "--output-csv",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_rebuttal_strong_targeted.csv",
                "--output-md",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_rebuttal_strong_targeted.md",
            ],
            [
                sys.executable,
                "experiments/analyze_rebuttal_stats.py",
                "--base-dir",
                str(results_dir),
                "--prefix",
                TARGET_PREFIX,
                "--target-model",
                "cgcma",
                "--output-prefix",
                "paper/supplementary_submission_20260401/results/rebuttal_strong_targeted",
            ],
            [
                sys.executable,
                "experiments/analyze_fee_sensitivity.py",
                "--base-dir",
                str(results_dir),
                "--prefix",
                SHUFFLE_PREFIX,
                "--output-csv",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_rebuttal_shuffle_ablation.csv",
                "--output-md",
                "paper/supplementary_submission_20260401/results/fee_sensitivity_rebuttal_shuffle_ablation.md",
            ],
            [
                sys.executable,
                "experiments/analyze_rebuttal_stats.py",
                "--base-dir",
                str(results_dir),
                "--prefix",
                SHUFFLE_PREFIX,
                "--target-model",
                "cgcma",
                "--output-prefix",
                "paper/supplementary_submission_20260401/results/rebuttal_shuffle_ablation",
            ],
        ]
        for cmd in steps:
            run_step(log_path, cmd)

    write_line(log_path, f"[{ts()}] targeted driver done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
