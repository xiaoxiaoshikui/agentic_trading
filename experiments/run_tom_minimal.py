#!/usr/bin/env python
"""
Run Minimal ToM Experiments
===========================

Runs E1/E2/E3 from the minimal ToM plan with walk-forward evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_loader import DataLoader, save_data_snapshot
from experiments.metrics import MetricsCalculator
from src.tom.agent import TheoryOfMindAgent
from src.tom.baselines import BuyAndHoldAgent, RandomAgent, TechnicalAgent, LLMBaselineAgent
from src.tom.evaluation.harness import EvaluationHarness, EvalConfig
from src.tom.opponents import RetailTraderModel, MomentumAlgoModel

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> Dict[str, Any]:
    # Support both UTF-8 and UTF-8 with BOM (common from Windows editors/PowerShell).
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_agent(
    name: str,
    agent_spec: Dict[str, Any],
    seed: int,
) -> Optional[Any]:
    agent_type = agent_spec.get("type")
    enabled = agent_spec.get("enabled", True)
    if not enabled:
        logger.info("Skipping disabled agent: %s", name)
        return None

    if agent_type == "technical":
        return TechnicalAgent()
    if agent_type == "buy_hold":
        return BuyAndHoldAgent()
    if agent_type == "random":
        return RandomAgent(seed=seed)
    if agent_type == "llm":
        return LLMBaselineAgent(
            model=agent_spec.get("model"),
            api_key=agent_spec.get("api_key"),
            base_url=agent_spec.get("base_url"),
            temperature=agent_spec.get("temperature", 0.2),
            max_tokens=agent_spec.get("max_tokens", 200),
            name=name,
            decision_interval=agent_spec.get("decision_interval", 1),
        )

    if agent_type == "tom":
        opponents = []
        opponent_params = agent_spec.get("opponent_params", {})
        for opp in agent_spec.get("opponents", ["retail", "momentum"]):
            if opp == "retail":
                params = opponent_params.get("retail", {})
                opponents.append(RetailTraderModel(market_share=params.get("market_share")))
            elif opp == "momentum":
                params = opponent_params.get("momentum", {})
                opponents.append(MomentumAlgoModel(market_share=params.get("market_share")))
            else:
                logger.warning("Unknown opponent model: %s", opp)

        return TheoryOfMindAgent(
            opponents=opponents or None,
            depth=agent_spec.get("depth", 1),
            use_technical=agent_spec.get("use_technical", True),
            technical_mode=agent_spec.get("technical_mode", "single"),
            allow_negative_influence=agent_spec.get("allow_negative_influence", False),
            name=name,
        )

    logger.warning("Unknown agent type: %s", agent_type)
    return None


def evaluate_agent(
    harness: EvaluationHarness,
    agent: Any,
    periods: List[Any],
    seed: int,
) -> List[Dict[str, Any]]:
    results = []
    for i, (train_data, test_data) in enumerate(periods):
        calibration_info: Dict[str, Any] = {}
        if hasattr(agent, "calibrate_on_history"):
            try:
                train_copy = {k: v.copy() for k, v in train_data.items()}
                calibration_info = agent.calibrate_on_history(
                    train_copy,
                    state_builder=harness._build_state,  # consistent feature construction
                    warmup_bars=harness.config.warmup_bars,
                )
            except Exception as exc:
                logger.warning("Calibration failed for %s period %s: %s", agent.name, i + 1, exc)

        test_copy = {k: v.copy() for k, v in test_data.items()}
        res = harness.evaluate_agent(agent, test_copy, seed=seed)
        results.append({
            "period": i + 1,
            "seed": seed,
            "agent": agent.name,
            "sharpe": res.sharpe,
            "max_drawdown": res.max_drawdown,
            "total_pnl": res.total_pnl,
            "win_rate": res.win_rate,
            "profit_factor": res.profit_factor,
            "total_trades": res.total_trades,
            "calibration_quality": calibration_info.get("quality"),
            "calibration_samples": calibration_info.get("n_samples"),
            "calibration_weights": calibration_info.get("weights"),
            "calibration_policy_params": calibration_info.get("policy_params"),
            "calibration_expected_disagreement": calibration_info.get("expected_disagreement"),
            "calibration_objective": calibration_info.get("objective"),
        })
    return results


def aggregate_records_by_period(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate seed-level records into period-level means.

    This keeps summary tables aligned with paired tests that compare per-period values.
    """
    if not records:
        return []

    metric_keys = [
        "sharpe",
        "max_drawdown",
        "total_pnl",
        "win_rate",
        "profit_factor",
        "total_trades",
    ]
    by_period: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        period = int(record["period"])
        by_period.setdefault(period, []).append(record)

    period_aggregates: List[Dict[str, Any]] = []
    for period in sorted(by_period):
        period_records = by_period[period]
        aggregated = {"period": period}
        for key in metric_keys:
            vals = [float(r.get(key, 0.0)) for r in period_records]
            aggregated[key] = float(np.mean(vals)) if vals else 0.0
        period_aggregates.append(aggregated)
    return period_aggregates


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    period_records = aggregate_records_by_period(records)

    def _mean(key: str) -> float:
        vals = [r[key] for r in period_records if key in r]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key: str) -> float:
        vals = [r[key] for r in period_records if key in r]
        return float(np.std(vals)) if vals else 0.0

    return {
        "n_samples": len(period_records),
        "n_raw_samples": len(records),
        "mean_sharpe": _mean("sharpe"),
        "std_sharpe": _std("sharpe"),
        "mean_pnl": _mean("total_pnl"),
        "mean_win_rate": _mean("win_rate"),
        "mean_max_drawdown": _mean("max_drawdown"),
        "mean_profit_factor": _mean("profit_factor"),
        "mean_trades": _mean("total_trades"),
    }


def build_summary_table(agent_summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for agent, summary in agent_summaries.items():
        row = {"agent": agent}
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_comparisons(
    agent_records: Dict[str, List[Dict[str, Any]]],
    baseline_name: str,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    if baseline_name not in agent_records:
        return comparisons

    baseline_records = {
        int(r["period"]): float(r["sharpe"])
        for r in aggregate_records_by_period(agent_records[baseline_name])
    }
    metrics = MetricsCalculator()

    for name, records in agent_records.items():
        if name == baseline_name:
            continue
        other_records = {
            int(r["period"]): float(r["sharpe"])
            for r in aggregate_records_by_period(records)
        }
        common_periods = sorted(set(baseline_records) & set(other_records))
        if not common_periods:
            continue
        baseline_vals = [baseline_records[p] for p in common_periods]
        other_vals = [other_records[p] for p in common_periods]

        t_stat, p_value, signif = metrics.paired_t_test(baseline_vals, other_vals)
        effect = metrics.effect_size_cohens_d(baseline_vals, other_vals)
        comparisons[name] = {
            "baseline": baseline_name,
            "t_stat": t_stat,
            "p_value": p_value,
            "significance": signif,
            "effect_size_d": effect,
            "n_pairs": len(common_periods),
        }
    return comparisons


def run_experiments(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    start_time = datetime.now()

    output_dir = config.get("output_dir", "experiments/results_tom")
    experiment_id = f"tom_minimal_{start_time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    data_loader = DataLoader(cache_dir=config.get("cache_dir", "experiments/data_cache"))
    data = data_loader.load_data(
        symbols=config.get("symbols", []),
        interval=config.get("interval", "15m"),
        n_bars=config.get("n_bars", 6000),
        end_time=config.get("end_time"),
    )

    if not data:
        raise RuntimeError("No data loaded. Check symbols/interval.")

    data_info = data_loader.get_data_info(data)
    save_data_snapshot(
        data,
        os.path.join(run_dir, "data_snapshot.json"),
        metadata={"config_path": config_path, "end_time": config.get("end_time")},
    )

    periods = data_loader.create_walk_forward_splits(
        data=data,
        n_periods=config.get("n_periods", 3),
        train_ratio=config.get("train_ratio", 0.7),
        min_train_bars=config.get("min_train_bars", 500),
        min_test_bars=config.get("min_test_bars", 200),
    )

    if not periods:
        raise RuntimeError("Failed to create walk-forward periods.")

    eval_cfg = EvalConfig(**config.get("eval", {}))
    harness = EvaluationHarness(eval_cfg)

    agents_config = config.get("agents", {})
    experiments = config.get("experiments", {})
    seeds = config.get("seeds", [42])

    results: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "start_time": start_time.isoformat(),
        "config_path": config_path,
        "config": config,
        "data_info": data_info,
        "experiments": {},
    }

    for exp_key, exp_spec in experiments.items():
        exp_name = exp_spec.get("name", exp_key)
        agent_names = exp_spec.get("agents", [])
        logger.info("Running %s (%s) with agents: %s", exp_key, exp_name, agent_names)

        agent_records: Dict[str, List[Dict[str, Any]]] = {}
        agent_summaries: Dict[str, Dict[str, Any]] = {}

        for agent_name in agent_names:
            spec = agents_config.get(agent_name)
            if not spec:
                logger.warning("Missing agent spec: %s", agent_name)
                continue

            combined_records: List[Dict[str, Any]] = []
            for seed in seeds:
                agent = build_agent(agent_name, spec, seed=seed)
                if agent is None:
                    break
                combined_records.extend(evaluate_agent(harness, agent, periods, seed))

            if combined_records:
                agent_records[agent_name] = combined_records
                agent_summaries[agent_name] = summarize_records(combined_records)

        comparisons = compute_comparisons(agent_records, agent_names[0]) if agent_names else {}
        table = build_summary_table(agent_summaries)

        exp_dir = os.path.join(run_dir, exp_key)
        os.makedirs(exp_dir, exist_ok=True)
        table.to_csv(os.path.join(exp_dir, "summary_table.csv"), index=False)

        exp_results = {
            "name": exp_name,
            "agents": agent_summaries,
            "records": agent_records,
            "comparisons": comparisons,
        }
        results["experiments"][exp_key] = exp_results

        save_json(os.path.join(exp_dir, "results.json"), exp_results)

    results["end_time"] = datetime.now().isoformat()
    save_json(os.path.join(run_dir, "results.json"), results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal ToM experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/tom_minimal.json",
        help="Path to ToM config JSON",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=not args.quiet)

    config_path = args.config
    config = load_config(config_path)

    results = run_experiments(config, config_path)
    logger.info("ToM experiments complete: %s", results.get("experiment_id"))


if __name__ == "__main__":
    main()
