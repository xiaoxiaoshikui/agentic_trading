import json
import os
import glob
import pandas as pd
from datetime import datetime
from typing import List, Dict

def analyze_performance():
    # 改为只读取今天的文件
    log_file = "data/history/analysis_log_2025-12-09.jsonl"
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("Empty log file")
        return

    df = pd.DataFrame(records)
    
    # 基础统计
    total_scans = len(df)
    
    # 信号统计
    math_signals = df['signal'].value_counts().to_dict()
    final_actions = df['action'].value_counts().to_dict()
    
    # 筛选出高分潜力股 (Action=FLAT 但分数 >= 45)
    high_potential = df[
        (df['action'] == 'FLAT') & 
        (df['score'] >= 45)
    ].sort_values(by='score', ascending=False)

    print("\n" + "="*60)
    print(f"Agentic Trading - RDNT Analysis (2025-12-09)")
    print("="*60)
    print(f"Time Range: {datetime.fromtimestamp(records[0]['timestamp_ts'])} to {datetime.fromtimestamp(records[-1]['timestamp_ts'])}")
    print(f"Total Scans: {total_scans}")
    print("-" * 60)
    
    print("\n1. Signal Overview")
    print(f"   Math Engine: {math_signals}")
    print(f"   Final Actions: {final_actions}")
    
    print("\n2. Top Potential Signals (Missed Opportunities?)")
    print("   (Signals with Score >= 45 but not executed)")
    print("-" * 60)
    
    if not high_potential.empty:
        # 只看前 10 个最高分的
        for i, row in high_potential.head(10).iterrows():
            time_str = datetime.fromtimestamp(row['timestamp_ts']).strftime("%H:%M:%S")
            print(f"   Time: {time_str} | Score: {row['score']:.1f}/100 | Signal: {row['signal']}")
            print(f"      Conf: {row['confidence']:.2f} | Indicators: {row['confirmations']} confirmed")
            # 打印关键指标原因
            indicators = row.get('indicators', {})
            vol = indicators.get('volume', {}).get('value', 'N/A')
            rsi = indicators.get('rsi', {}).get('value', 'N/A')
            print(f"      Key Data: {vol} | {rsi}")
            print(f"      Reason: {row.get('reason', 'N/A')}")
            print("      ---")
    else:
        print("   No high potential signals found.")

    print("\n" + "="*60)

if __name__ == "__main__":
    analyze_performance()
