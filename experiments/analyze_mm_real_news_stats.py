import argparse
import csv
import json
import math
import os
import random
import statistics
from typing import Dict, List


TCRIT_DF3_95 = 3.182446305284263


def load_model_result(run_dir: str, model: str) -> Dict:
    path = os.path.join(run_dir, "results.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return next(item for item in data["results"] if item["model"] == model)


def mean_ci_95(values: List[float]) -> Dict[str, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    moe = TCRIT_DF3_95 * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - moe,
        "ci_high": mean + moe,
    }


def bootstrap_mean_ci(values: List[float], n_boot: int = 20000, seed: int = 42) -> Dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return {
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[int(0.975 * n_boot)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cgcma", "price_only", "bilstm", "early_fusion", "mult", "tfn", "text_only"],
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    run_level: Dict[str, List[float]] = {m: [] for m in args.models}
    fold_level: Dict[str, List[List[float]]] = {m: [] for m in args.models}

    abs_run_dirs = [os.path.join(args.base_dir, d) for d in args.run_dirs]

    for run_dir in abs_run_dirs:
        for model in args.models:
            item = load_model_result(run_dir, model)
            run_level[model].append(item["test_downstream"]["sharpe_like"])
            fold_level[model].append([fr["test_downstream"]["sharpe_like"] for fr in item["fold_results"]])

    summaries = []
    for model in args.models:
        row = {"model": model}
        row.update(mean_ci_95(run_level[model]))
        summaries.append(row)

    matched_fold_avg: Dict[str, List[float]] = {}
    n_folds = len(next(iter(fold_level.values()))[0])
    n_runs = len(args.run_dirs)
    for model in args.models:
        matched_fold_avg[model] = [
            statistics.mean(fold_level[model][run_idx][fold_idx] for run_idx in range(n_runs))
            for fold_idx in range(n_folds)
        ]

    deltas = []
    for baseline in [m for m in args.models if m != "cgcma"]:
        diffs = [a - b for a, b in zip(matched_fold_avg["cgcma"], matched_fold_avg[baseline])]
        boot = bootstrap_mean_ci(diffs)
        deltas.append(
            {
                "comparison": f"cgcma_minus_{baseline}",
                "mean_delta": statistics.mean(diffs),
                "std_delta": statistics.stdev(diffs),
                "wins": sum(d > 0 for d in diffs),
                "losses": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs),
                "boot_ci_low": boot["ci_low"],
                "boot_ci_high": boot["ci_high"],
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "name", "mean", "std", "ci_low", "ci_high", "wins", "losses", "ties"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "section": "run_level",
                    "name": row["model"],
                    "mean": f"{row['mean']:.6f}",
                    "std": f"{row['std']:.6f}",
                    "ci_low": f"{row['ci_low']:.6f}",
                    "ci_high": f"{row['ci_high']:.6f}",
                    "wins": "",
                    "losses": "",
                    "ties": "",
                }
            )
        for row in deltas:
            writer.writerow(
                {
                    "section": "matched_fold_delta",
                    "name": row["comparison"],
                    "mean": f"{row['mean_delta']:.6f}",
                    "std": f"{row['std_delta']:.6f}",
                    "ci_low": f"{row['boot_ci_low']:.6f}",
                    "ci_high": f"{row['boot_ci_high']:.6f}",
                    "wins": row["wins"],
                    "losses": row["losses"],
                    "ties": row["ties"],
                }
            )

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Real-news final statistical summary\n\n")
        f.write("## Run-level 95% t-intervals over 4 seeds\n\n")
        f.write("| Model | Mean Sharpe | Std | 95% CI |\n")
        f.write("| --- | ---: | ---: | --- |\n")
        for row in summaries:
            f.write(
                f"| {row['model']} | {row['mean']:+.3f} | {row['std']:.3f} | "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |\n"
            )
        f.write("\n## Matched-fold bootstrap deltas (CGCMA minus baseline)\n\n")
        f.write("| Comparison | Mean delta | 95% bootstrap CI | Wins | Losses |\n")
        f.write("| --- | ---: | --- | ---: | ---: |\n")
        for row in deltas:
            f.write(
                f"| {row['comparison']} | {row['mean_delta']:+.3f} | "
                f"[{row['boot_ci_low']:+.3f}, {row['boot_ci_high']:+.3f}] | "
                f"{row['wins']} | {row['losses']} |\n"
            )


if __name__ == "__main__":
    main()
