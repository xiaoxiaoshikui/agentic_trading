#!/usr/bin/env python
"""
ToM Experiment Results Analyzer
================================

Quick analysis of the latest ToM experiment results.
Generates tables and key statistics for the paper.
"""

import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def find_latest_results(base_dir="experiments/results_tom"):
    """Find the most recent experiment results directory."""
    dirs = sorted(glob.glob(os.path.join(base_dir, "tom_minimal_*")))
    if not dirs:
        print("No experiment results found!")
        return None
    latest = dirs[-1]
    results_path = os.path.join(latest, "results.json")
    if not os.path.exists(results_path):
        print(f"results.json not found in {latest}")
        print(f"Contents: {os.listdir(latest)}")
        return None
    return results_path


def analyze_results(results_path: str):
    """Analyze and display comprehensive experiment results."""
    with open(results_path, "r") as f:
        results = json.load(f)

    print("=" * 70)
    print(f"实验ID: {results.get('experiment_id', 'N/A')}")
    print(f"开始时间: {results.get('start_time', 'N/A')}")
    print(f"结束时间: {results.get('end_time', 'N/A')}")
    print(f"配置: {results.get('config_path', 'N/A')}")
    print("=" * 70)

    experiments = results.get("experiments", {})

    for exp_key, exp_data in experiments.items():
        exp_name = exp_data.get("name", exp_key)
        agents = exp_data.get("agents", {})
        comparisons = exp_data.get("comparisons", {})

        print(f"\n{'=' * 70}")
        print(f"  {exp_key}: {exp_name}")
        print(f"{'=' * 70}")

        # Summary table
        if agents:
            rows = []
            for agent_name, summary in agents.items():
                rows.append({
                    "Agent": agent_name,
                    "Sharpe": f"{summary.get('mean_sharpe', 0):.3f} ± {summary.get('std_sharpe', 0):.3f}",
                    "PnL": f"${summary.get('mean_pnl', 0):,.2f}",
                    "Win%": f"{summary.get('mean_win_rate', 0) * 100:.1f}%",
                    "MaxDD": f"{summary.get('mean_max_drawdown', 0) * 100:.1f}%",
                    "PF": f"{summary.get('mean_profit_factor', 0):.2f}",
                    "Trades": f"{summary.get('mean_trades', 0):.0f}",
                    "Samples": f"{summary.get('n_samples', 0)} (raw: {summary.get('n_raw_samples', 0)})",
                })
            df = pd.DataFrame(rows)
            print("\n性能摘要 (Performance Summary):")
            print(df.to_string(index=False))

        # Comparisons (statistical tests)
        if comparisons:
            print("\n统计比较 (Statistical Comparisons vs baseline):")
            for agent_name, comp in comparisons.items():
                sig = comp.get("significance", "n/a")
                p = comp.get("p_value", 1.0)
                d = comp.get("effect_size_d", 0.0)
                t = comp.get("t_stat", 0.0)
                n = comp.get("n_pairs", 0)
                baseline = comp.get("baseline", "?")
                star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
                print(f"  {agent_name} vs {baseline}: "
                      f"t={t:.3f}, p={p:.4f}{star}, d={d:.3f}, "
                      f"sig={sig}, n_pairs={n}")

        # Per-period detail
        records = exp_data.get("records", {})
        if records:
            print("\n逐期详情 (Per-Period Detail):")
            for agent_name, recs in records.items():
                print(f"\n  [{agent_name}]")
                for rec in recs:
                    print(f"    Period {rec['period']} Seed {rec['seed']}: "
                          f"Sharpe={rec['sharpe']:.3f}, "
                          f"PnL=${rec['total_pnl']:,.2f}, "
                          f"WR={rec['win_rate'] * 100:.1f}%, "
                          f"Trades={rec['total_trades']}")

    # Key findings
    print(f"\n{'=' * 70}")
    print("  关键发现 (Key Findings)")
    print(f"{'=' * 70}")

    e1 = experiments.get("E1", {})
    e1_agents = e1.get("agents", {})
    if e1_agents:
        agent_sharpes = {k: v.get("mean_sharpe", 0) for k, v in e1_agents.items()}
        best = max(agent_sharpes, key=agent_sharpes.get)
        print(f"\n  E1 最佳Agent: {best} (Sharpe={agent_sharpes[best]:.3f})")

        tom_sharpe = agent_sharpes.get("ToM-Full", 0)
        tech_sharpe = agent_sharpes.get("Technical", 0)
        llm_sharpe = agent_sharpes.get("LLM", 0)

        if tech_sharpe != 0:
            tom_vs_tech = (tom_sharpe - tech_sharpe) / abs(tech_sharpe) * 100
            print(f"  ToM vs Technical: {tom_vs_tech:+.1f}% (Sharpe改进)")
        if llm_sharpe != 0:
            tom_vs_llm = (tom_sharpe - llm_sharpe) / abs(llm_sharpe) * 100
            print(f"  ToM vs LLM: {tom_vs_llm:+.1f}% (Sharpe改进)")

        # Check if LLM had zero metrics (API error indicator)
        llm_summary = e1_agents.get("LLM", {})
        if llm_summary.get("mean_trades", 0) == 0 and llm_summary.get("mean_pnl", 0) == 0:
            print("  ⚠️ 警告: LLM Agent交易数=0, 可能存在API调用问题!")

    e1_comp = e1.get("comparisons", {})
    for name, comp in e1_comp.items():
        if comp.get("p_value", 1.0) < 0.05:
            print(f"  ✅ {name} vs {comp['baseline']}: 统计显著 (p={comp['p_value']:.4f})")

    print()


def generate_latex_table(results_path: str):
    """Generate LaTeX table for the paper."""
    with open(results_path, "r") as f:
        results = json.load(f)

    e1 = results.get("experiments", {}).get("E1", {})
    agents = e1.get("agents", {})
    comparisons = e1.get("comparisons", {})

    print("\n% LaTeX Table: E1 Main Performance")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{E1: Main Performance Comparison (Walk-Forward Evaluation)}")
    print("\\label{tab:e1_main}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("Agent & Sharpe & PnL (\\$) & Win \\% & Max DD & PF & Trades \\\\")
    print("\\midrule")

    for agent_name, summary in agents.items():
        s = summary.get("mean_sharpe", 0)
        s_std = summary.get("std_sharpe", 0)
        pnl = summary.get("mean_pnl", 0)
        wr = summary.get("mean_win_rate", 0) * 100
        dd = summary.get("mean_max_drawdown", 0) * 100
        pf = summary.get("mean_profit_factor", 0)
        tr = summary.get("mean_trades", 0)

        # Check for significance
        comp = comparisons.get(agent_name, {})
        sig = ""
        p = comp.get("p_value", 1.0)
        if p < 0.01:
            sig = "$^{***}$"
        elif p < 0.05:
            sig = "$^{**}$"
        elif p < 0.1:
            sig = "$^{*}$"

        print(f"{agent_name}{sig} & {s:.3f}$\\pm${s_std:.3f} & "
              f"{pnl:,.0f} & {wr:.1f} & {dd:.1f} & {pf:.2f} & {tr:.0f} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None

    if path is None:
        path = find_latest_results()

    if path is None:
        print("未找到实验结果，请先运行实验。")
        sys.exit(1)

    print(f"分析文件: {path}\n")
    analyze_results(path)
    generate_latex_table(path)
