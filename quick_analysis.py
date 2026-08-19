# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import glob

# 加载所有日志
all_records = []
for f in sorted(glob.glob('data/history/*.jsonl')):
    with open(f, encoding='utf-8') as fp:
        for line in fp:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

# 统计
total = len(all_records)
signals = [r for r in all_records if r.get('signal') != 'FLAT']
trades = [r for r in all_records if r.get('action') != 'FLAT']
blocked = [r for r in all_records if r.get('signal') != 'FLAT' and r.get('action') == 'FLAT']

print("=" * 60)
print("策略分析报告")
print("=" * 60)
print(f"总记录数: {total}")
print(f"技术信号数: {len(signals)} ({len(signals)/total*100:.1f}%)")
print(f"执行交易数: {len(trades)} ({len(trades)/total*100:.1f}%)")
print(f"AI拦截数: {len(blocked)} ({len(blocked)/len(signals)*100:.1f}% 的信号被拦截)")

# 分析信号方向
long_signals = [r for r in signals if r.get('signal') == 'LONG']
short_signals = [r for r in signals if r.get('signal') == 'SHORT']
print(f"\n信号方向分布:")
print(f"  LONG: {len(long_signals)} ({len(long_signals)/len(signals)*100:.1f}%)")
print(f"  SHORT: {len(short_signals)} ({len(short_signals)/len(signals)*100:.1f}%)")

# 分析评分分布
scores = [r.get('score', 0) for r in trades]
if scores:
    print(f"\n执行交易评分:")
    print(f"  平均分: {sum(scores)/len(scores):.1f}")
    print(f"  最高分: {max(scores)}")
    print(f"  最低分: {min(scores)}")

# 分析被拦截原因
print(f"\n被拦截的信号 (前10条):")
for r in blocked[:10]:
    ts = r.get('timestamp', '')[:16]
    sig = r.get('signal')
    score = r.get('score', 0)
    reason = r.get('reason', '')[:50]
    print(f"  {ts} | {sig} | 分数:{score} | {reason}...")

# 分析执行的交易
print(f"\n执行的交易:")
for r in trades:
    ts = r.get('timestamp', '')[:16]
    action = r.get('action')
    score = r.get('score', 0)
    reason = r.get('reason', '')[:50]
    print(f"  {ts} | {action} | 分数:{score} | {reason}...")
