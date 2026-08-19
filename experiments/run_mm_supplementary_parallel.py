#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIGS = [
    "experiments/configs/mm_deep_multi_real_news_2026_strong_baselines.json",
    "experiments/configs/mm_deep_multi_real_news_2026_pw_baseline.json",
    "experiments/configs/mm_deep_multi_real_news_2026_cgcma_ablation.json",
    "experiments/configs/mm_deep_multi_real_news_2026_task_specific.json",
    "experiments/configs/mm_deep_btc_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_eth_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_sol_real_news_asset_consistency.json",
]

DEFAULT_SEEDS = [42, 123, 456, 789]


def has_completed_run(results_dir: Path, config_name: str, seed: int) -> bool:
    pattern = f"{config_name}_*_seed{seed}"
    for run_dir in results_dir.glob(pattern):
        if run_dir.is_dir() and (run_dir / "summary.csv").exists():
            return True
    return False


def write_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch supplementary multimodal experiments in parallel.")
    parser.add_argument("--results-dir", default="experiments/results_mm")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_log = results_dir / f"supplementary_parallel_{stamp}.log"
    manifest_path = results_dir / f"supplementary_parallel_{stamp}.json"

    manifest: list[dict[str, object]] = []
    write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Starting parallel supplementary launch")
    write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Results dir: {results_dir}")

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    for config_rel in args.configs:
        config_path = (repo_root / config_rel).resolve()
        config_name = config_path.stem
        if not config_path.exists():
            write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Missing config, skipping: {config_path}")
            continue

        for seed in args.seeds:
            if has_completed_run(results_dir, config_name, seed):
                write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Skip existing {config_name} seed {seed}")
                continue

            job_log = results_dir / f"{config_name}_seed{seed}_{stamp}.log"
            cmd = [
                sys.executable,
                "-m",
                "experiments.run_mm_deep_experiment",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]

            with job_log.open("w", encoding="utf-8") as fh:
                proc = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )

            record = {
                "config": config_name,
                "seed": seed,
                "pid": proc.pid,
                "log": str(job_log),
                "command": cmd,
                "launched_at": datetime.now().isoformat(timespec="seconds"),
            }
            manifest.append(record)
            write_line(
                launch_log,
                f"[{datetime.now().isoformat(timespec='seconds')}] Launched {config_name} seed {seed} pid={proc.pid} log={job_log.name}",
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Wrote manifest: {manifest_path.name}")
    write_line(launch_log, f"[{datetime.now().isoformat(timespec='seconds')}] Parallel supplementary launch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
