#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIGS = [
    "experiments/configs/mm_deep_multi_real_news_2026_pw_baseline.json",
    "experiments/configs/mm_deep_multi_real_news_2026_cgcma_ablation.json",
    "experiments/configs/mm_deep_multi_real_news_2026_task_specific.json",
    "experiments/configs/mm_deep_btc_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_eth_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_sol_real_news_asset_consistency.json",
]

DEFAULT_SEEDS = [42, 123, 456, 789]


def log(msg: str, fh) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def has_completed_run(results_dir: Path, config_name: str, seed: int) -> bool:
    pattern = f"{config_name}_*_seed{seed}"
    for run_dir in results_dir.glob(pattern):
        if run_dir.is_dir() and (run_dir / "summary.csv").exists():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run supplementary multimodal experiments sequentially.")
    parser.add_argument("--results-dir", default="experiments/results_mm")
    parser.add_argument("--sleep-seconds", type=int, default=5)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"supplementary_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as fh:
        log("Starting supplementary suite", fh)
        log(f"Results dir: {results_dir}", fh)

        for config_rel in args.configs:
            config_path = (repo_root / config_rel).resolve()
            config_name = config_path.stem
            if not config_path.exists():
                log(f"Missing config, skipping: {config_path}", fh)
                continue

            for seed in args.seeds:
                if has_completed_run(results_dir, config_name, seed):
                    log(f"Skip existing {config_name} seed {seed}", fh)
                    continue

                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.run_mm_deep_experiment",
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                ]
                log(f"Running {config_name} seed {seed}", fh)
                log("Command: " + " ".join(cmd), fh)

                started = time.time()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                except Exception as exc:  # pragma: no cover
                    log(f"Exception in {config_name} seed {seed}: {exc}", fh)
                    continue

                duration_min = (time.time() - started) / 60.0
                if proc.returncode == 0:
                    log(f"Finished {config_name} seed {seed} in {duration_min:.1f} min", fh)
                else:
                    log(f"FAILED {config_name} seed {seed} with code {proc.returncode} after {duration_min:.1f} min", fh)

                time.sleep(max(0, args.sleep_seconds))

        log("Supplementary suite complete", fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
