import json
import os
import glob
import pandas as pd
from datetime import datetime

def load_logs(log_dir: str = "data/history") -> list:
    """加载并按时间排序所有日志"""
    files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    files.sort()
    all_records = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # 确保按时间戳排序
    all_records.sort(key=lambda x: x['timestamp_ts'])
    return all_records

def verify_decisions():
    records = load_logs()
    if not records:
        print("📭 暂无日志数据")
        return

    # 转换为 DataFrame 以便处理
    df = pd.DataFrame(records)
    
    # 我们只关心被拦截的信号 (Math!=FLAT, Action==FLAT)
    interceptions = df[
        (df['signal'] != 'FLAT') & 
        (df['action'] == 'FLAT')
    ].copy()

    if interceptions.empty:
        print("✅ 没有拦截记录，所有数学信号都被执行了（或本身就是FLAT）")
        return

    print(f"🔍 开始验证 {len(interceptions)} 次 AI 拦截决策...")
    print(f"🕒 验证逻辑: 检查信号发出后 1~4 小时内的最大价格变动")
    
    correct_blocks = 0
    missed_opportunities = 0
    neutral_blocks = 0
    
    results = []

    for idx, row in interceptions.iterrows():
        signal_time = row['timestamp_ts']
        entry_price = row['entry_price']
        signal_type = row['signal'] # LONG or SHORT
        
        # 寻找该信号之后 1-4 小时内的记录
        future_window = df[
            (df['timestamp_ts'] > signal_time) & 
            (df['timestamp_ts'] <= signal_time + 4 * 3600) # 4小时窗口
        ]
        
        if future_window.empty:
            continue # 数据不够，没法验证（可能是刚发生的）

        # 计算未来最大盈亏幅度
        if signal_type == "LONG":
            max_price = future_window['entry_price'].max()
            price_change_pct = (max_price - entry_price) / entry_price * 100
        else: # SHORT
            min_price = future_window['entry_price'].min()
            price_change_pct = (entry_price - min_price) / entry_price * 100
            
        # 判定标准
        # 如果涨幅超过 1% (对于15m级别)，算作踏空 (Missed Opportunity)
        # 如果涨幅微弱 (< 0.5%) 或亏损，算作正确拦截 (Correct Block)
        
        status = "UNKNOWN"
        if price_change_pct > 1.0:
            status = "❌ 踏空 (AI 错了)"
            missed_opportunities += 1
        elif price_change_pct < 0.2:
            status = "✅ 避险 (AI 对了)"
            correct_blocks += 1
        else:
            status = "😐 无效波动"
            neutral_blocks += 1
            
        results.append({
            "Time": datetime.fromtimestamp(signal_time).strftime("%H:%M"),
            "Signal": signal_type,
            "Price": entry_price,
            "Max_Profit_Potential": f"{price_change_pct:.2f}%",
            "Verdict": status,
            "AI_Reason": row.get('reason', '')[:30] + "..."
        })

    # 输出结果表格
    res_df = pd.DataFrame(results)
    
    if res_df.empty:
        print("⚠️ 数据不足以验证（可能因为是最近产生的信号，还没走出未来行情）")
        return

    print("\n" + "="*80)
    print("🤖 AI 拦截验证报告")
    print("="*80)
    print(res_df.to_string(index=False))
    
    print("\n📈 总结统计:")
    print(f"✅ AI 正确拦截 (帮你避坑): {correct_blocks} 次")
    print(f"❌ AI 错误拦截 (导致踏空): {missed_opportunities} 次")
    print(f"😐 拦截无效波动 (省手续费): {neutral_blocks} 次")
    
    total_valid = correct_blocks + missed_opportunities + neutral_blocks
    if total_valid > 0:
        accuracy = (correct_blocks + neutral_blocks) / total_valid * 100
        print(f"\n🏆 AI 防守成功率: {accuracy:.1f}%")
        print("(注: '防守成功' = 避坑 + 避免无效交易)")

if __name__ == "__main__":
    verify_decisions()
