#!/usr/bin/env python
"""
Run reviewer-oriented experiment suites and generate a consolidated report.

This script focuses on the three categories reviewers usually ask for:
1) Stability across settings
2) Ablation studies
3) Robustness checks
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "configs"
RESULTS_DIR = ROOT / "experiments" / "results_tom"


@dataclass(frozen=True)
class SuiteItem:
    category: str
    config: str
    force_multi: bool = True
    disable_llm: bool = False
    tom_overrides: Optional[Dict[str, Any]] = None


SUITES: Dict[str, List[SuiteItem]] = {
    "core": [
        SuiteItem("stability", "tom_e1b_4h.json"),
        SuiteItem("stability", "tom_e1c_1h_conservative.json"),
        SuiteItem("stability", "tom_e1c_1h_aggressive.json"),
        SuiteItem("stability", "tom_e1c_4h_conservative.json"),
        SuiteItem("stability", "tom_e1c_4h_aggressive.json"),
    ],
    "extended": [
        SuiteItem("stability", "tom_e1b_4h.json"),
        SuiteItem("stability", "tom_e1c_1h_conservative.json"),
        SuiteItem("stability", "tom_e1c_1h_aggressive.json"),
        SuiteItem("stability", "tom_e1c_4h_conservative.json"),
        SuiteItem("stability", "tom_e1c_4h_aggressive.json"),
        SuiteItem("robustness", "tom_e1d_more_assets.json"),
        SuiteItem("ablation", "tom_e2b_retail_sweep.json"),
        SuiteItem("ablation", "tom_minimal.json", disable_llm=True),
    ],
    "core_split": [
        SuiteItem(
            "stability",
            "tom_e1b_4h.json",
            tom_overrides={"allow_negative_influence": False},
        ),
        SuiteItem(
            "stability",
            "tom_e1c_1h_conservative.json",
            tom_overrides={"allow_negative_influence": False},
        ),
        SuiteItem(
            "stability",
            "tom_e1c_1h_aggressive.json",
            tom_overrides={"allow_negative_influence": True},
        ),
        SuiteItem(
            "stability",
            "tom_e1c_4h_conservative.json",
            tom_overrides={"allow_negative_influence": False},
        ),
        SuiteItem(
            "stability",
            "tom_e1c_4h_aggressive.json",
            tom_overrides={"allow_negative_influence": False},
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reviewer-oriented experiment suite")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="core",
        help="Suite preset to run (default: core)",
    )
    parser.add_argument(
        "--output-prefix",
        default="reviewer_suite",
        help="Prefix for generated report files under experiments/results_tom",
    )
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep temporary configs generated for each run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned runs, do not execute",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Override seeds for all configs, e.g. '42,123,456,789,2026,7,8,9'",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=0,
        help="Override n_periods for all configs (0 keeps original)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.0,
        help="Override train_ratio for all configs (0 keeps original)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)


def latest_run_dirs() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)


def parse_seeds(raw: str) -> List[int]:
    if not raw.strip():
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def build_temp_config(
    item: SuiteItem,
    stamp: str,
    seed_override: Optional[List[int]] = None,
    n_periods_override: int = 0,
    train_ratio_override: float = 0.0,
) -> Path:
    src = CONFIG_DIR / item.config
    if not src.exists():
        raise FileNotFoundError(f"Config not found: {src}")

    cfg = load_json(src)
    if seed_override:
        cfg["seeds"] = list(seed_override)
    if n_periods_override > 0:
        cfg["n_periods"] = int(n_periods_override)
    if train_ratio_override > 0.0:
        cfg["train_ratio"] = float(train_ratio_override)

    agents = cfg.get("agents", {})
    for _, spec in agents.items():
        if not isinstance(spec, dict):
            continue
        if item.force_multi and spec.get("type") == "tom":
            spec["technical_mode"] = "multi"
            if item.tom_overrides:
                spec.update(item.tom_overrides)
        if item.disable_llm and spec.get("type") == "llm":
            spec["enabled"] = False

    stem = src.stem
    out = CONFIG_DIR / f"{stem}_reviewer_{stamp}.json"
    write_json(out, cfg)
    return out


def run_one_config(tmp_config: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_tom_minimal",
        "--config",
        str(tmp_config),
        "--quiet",
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)
    return int(proc.returncode)


def read_summary_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for exp_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        summary_path = exp_dir / "summary_table.csv"
        results_path = exp_dir / "results.json"
        if not summary_path.exists():
            continue

        exp_name = exp_dir.name
        comparisons: Dict[str, Any] = {}
        if results_path.exists():
            results_obj = load_json(results_path)
            exp_name = str(results_obj.get("name", exp_name))
            comparisons = dict(results_obj.get("comparisons", {}))

        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            summary_table = list(csv.DictReader(f))

        by_agent = {row["agent"]: row for row in summary_table}
        tech = by_agent.get("Technical")
        tom = by_agent.get("ToM-Full")
        cmp = comparisons.get("ToM-Full", {})

        if tech and tom:
            tech_sharpe = float(tech["mean_sharpe"])
            tom_sharpe = float(tom["mean_sharpe"])
            delta = tom_sharpe - tech_sharpe
            rows.append(
                {
                    "experiment_dir": exp_dir.name,
                    "experiment_name": exp_name,
                    "technical_sharpe": tech_sharpe,
                    "tom_sharpe": tom_sharpe,
                    "delta_tom_minus_tech": delta,
                    "p_value": cmp.get("p_value"),
                    "effect_size_d": cmp.get("effect_size_d"),
                    "significance": cmp.get("significance"),
                }
            )
        else:
            best_agent = ""
            best_sharpe = -1e18
            for row in summary_table:
                sharpe = float(row.get("mean_sharpe", 0.0))
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_agent = str(row.get("agent", ""))
            rows.append(
                {
                    "experiment_dir": exp_dir.name,
                    "experiment_name": exp_name,
                    "technical_sharpe": "",
                    "tom_sharpe": "",
                    "delta_tom_minus_tech": "",
                    "p_value": "",
                    "effect_size_d": "",
                    "significance": "",
                    "best_agent": best_agent,
                    "best_sharpe": best_sharpe,
                }
            )
    return rows


def summarize_core(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    valid = [r for r in rows if r.get("delta_tom_minus_tech", "") != ""]
    if not valid:
        return {"n": 0, "wins": 0, "losses": 0, "mean_delta": ""}
    deltas = [float(r["delta_tom_minus_tech"]) for r in valid]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return {
        "n": len(deltas),
        "wins": wins,
        "losses": losses,
        "mean_delta": sum(deltas) / len(deltas),
    }


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = SUITES[args.suite]
    seed_override = parse_seeds(args.seeds)

    if args.dry_run:
        print(f"[dry-run] suite={args.suite}, items={len(plan)}")
        if seed_override:
            print(f"[dry-run] seeds override: {seed_override}")
        if args.n_periods > 0:
            print(f"[dry-run] n_periods override: {args.n_periods}")
        if args.train_ratio > 0:
            print(f"[dry-run] train_ratio override: {args.train_ratio}")
        for it in plan:
            print(f"- {it.category}: {it.config} (multi={it.force_multi}, disable_llm={it.disable_llm})")
        return

    report_rows: List[Dict[str, Any]] = []
    log_dir = RESULTS_DIR / "reviewer_logs"
    existing_dirs = set(p.name for p in latest_run_dirs())

    for idx, item in enumerate(plan, start=1):
        tmp_config = build_temp_config(
            item,
            stamp=f"{stamp}_{idx:02d}",
            seed_override=seed_override,
            n_periods_override=args.n_periods,
            train_ratio_override=args.train_ratio,
        )
        log_path = log_dir / f"{tmp_config.stem}.log"
        print(f"[{idx}/{len(plan)}] running {item.config} ...")
        code = run_one_config(tmp_config, log_path)

        all_dirs = latest_run_dirs()
        new_dirs = [p for p in all_dirs if p.name not in existing_dirs]
        existing_dirs = set(p.name for p in all_dirs)
        run_dir = new_dirs[-1] if new_dirs else (all_dirs[-1] if all_dirs else None)

        if run_dir is None:
            report_rows.append(
                {
                    "category": item.category,
                    "config": item.config,
                    "status": "failed",
                    "return_code": code,
                    "run_id": "",
                    "experiment_dir": "",
                    "experiment_name": "",
                    "technical_sharpe": "",
                    "tom_sharpe": "",
                    "delta_tom_minus_tech": "",
                    "p_value": "",
                    "effect_size_d": "",
                    "significance": "",
                    "best_agent": "",
                    "best_sharpe": "",
                }
            )
        else:
            per_exp = read_summary_rows(run_dir)
            if not per_exp:
                report_rows.append(
                    {
                        "category": item.category,
                        "config": item.config,
                        "status": "failed",
                        "return_code": code,
                        "run_id": run_dir.name,
                        "experiment_dir": "",
                        "experiment_name": "",
                        "technical_sharpe": "",
                        "tom_sharpe": "",
                        "delta_tom_minus_tech": "",
                        "p_value": "",
                        "effect_size_d": "",
                        "significance": "",
                        "best_agent": "",
                        "best_sharpe": "",
                    }
                )
            else:
                for row in per_exp:
                    out = {
                        "category": item.category,
                        "config": item.config,
                        "status": "ok" if code == 0 else "warning",
                        "return_code": code,
                        "run_id": run_dir.name,
                        "experiment_dir": row.get("experiment_dir", ""),
                        "experiment_name": row.get("experiment_name", ""),
                        "technical_sharpe": row.get("technical_sharpe", ""),
                        "tom_sharpe": row.get("tom_sharpe", ""),
                        "delta_tom_minus_tech": row.get("delta_tom_minus_tech", ""),
                        "p_value": row.get("p_value", ""),
                        "effect_size_d": row.get("effect_size_d", ""),
                        "significance": row.get("significance", ""),
                        "best_agent": row.get("best_agent", ""),
                        "best_sharpe": row.get("best_sharpe", ""),
                    }
                    report_rows.append(out)

        if not args.keep_temp_configs and tmp_config.exists():
            tmp_config.unlink()

    csv_path = RESULTS_DIR / f"{args.output_prefix}_{args.suite}_{stamp}.csv"
    md_path = RESULTS_DIR / f"{args.output_prefix}_{args.suite}_{stamp}.md"

    fields = [
        "category",
        "config",
        "status",
        "return_code",
        "run_id",
        "experiment_dir",
        "experiment_name",
        "technical_sharpe",
        "tom_sharpe",
        "delta_tom_minus_tech",
        "p_value",
        "effect_size_d",
        "significance",
        "best_agent",
        "best_sharpe",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)

    core_rows = [r for r in report_rows if r.get("delta_tom_minus_tech", "") != ""]
    core_summary = summarize_core(core_rows)
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Reviewer Suite Report ({args.suite})\n\n")
        f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- CSV: `{csv_path}`\n")
        f.write(f"- Rows: {len(report_rows)}\n")
        f.write(f"- Comparable rows (ToM vs Technical): {core_summary['n']}\n")
        f.write(f"- Wins: {core_summary['wins']}, Losses: {core_summary['losses']}\n")
        f.write(f"- Mean delta (ToM - Technical): {core_summary['mean_delta']}\n\n")
        f.write("## Runs\n")
        for row in report_rows:
            f.write(
                f"- [{row['category']}] {row['config']} | run={row['run_id']} | "
                f"exp={row['experiment_dir']} | delta={row['delta_tom_minus_tech']} | status={row['status']}\n"
            )

    print(f"[done] csv: {csv_path}")
    print(f"[done] md: {md_path}")


if __name__ == "__main__":
    main()
