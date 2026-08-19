import logging
import time
import json
import os
import glob
import pandas as pd
from datetime import datetime
from src.local_llm import LocalStrategyArchitect

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_recent_context():
    """读取最新的交易日志，生成市场摘要和问题描述"""
    # 找到最新的日志文件
    list_of_files = glob.glob('data/history/*.jsonl') 
    if not list_of_files:
        return None, None
    latest_file = max(list_of_files, key=os.path.getctime)
    
    logger.info(f"正在分析日志文件: {latest_file}")
    
    records = []
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
    if not records:
        return None, None
        
    df = pd.DataFrame(records)
    
    # 1. 寻找错失的机会 (Score >= 45 但没开单)
    missed_ops = df[(df['action'] == 'FLAT') & (df['score'] >= 45)]
    
    # 2. 构建市场摘要 (取最近 5 条记录的平均状态)
    recent_records = df.tail(5)
    last_record = recent_records.iloc[-1]
    
    indicators = last_record.get('indicators', {})
    rsi_val = indicators.get('rsi', {}).get('value', 'N/A')
    vol_val = indicators.get('volume', {}).get('value', 'N/A')
    
    market_summary = f"""
    最新分析时间: {datetime.fromtimestamp(last_record['timestamp_ts'])}
    当前信号: {last_record['signal']} (Conf: {last_record['confidence']})
    关键指标:
    - {rsi_val}
    - {vol_val}
    - Regime: {last_record.get('regime', 'Unknown')}
    """
    
    # 3. 构建问题描述
    issue_desc = "当前策略表现正常。"
    if not missed_ops.empty:
        top_miss = missed_ops.sort_values('score', ascending=False).iloc[0]
        issue_desc = f"""
        发现错失的交易机会! 
        在 {datetime.fromtimestamp(top_miss['timestamp_ts'])}，系统给出了 {top_miss['score']} 分的评价，但未达到开仓门槛。
        当时 RSI 为 {top_miss.get('indicators', {}).get('rsi', {}).get('value')}，
        成交量为 {top_miss.get('indicators', {}).get('volume', {}).get('value')}。
        需要调整策略以捕捉此类高分信号 (目标分数 > 60)。
        """
    
    performance_metrics = {
        "missed_count": len(missed_ops),
        "issue": issue_desc.strip()
    }
    
    return market_summary, performance_metrics

def main():
    print("=" * 60)
    print("🧬 Agentic Trading - 策略进化引擎 (基于 Qwen3)")
    print("=" * 60)
    
    architect = LocalStrategyArchitect(model_name="qwen3:8b")
    
    # 1. 从历史数据加载上下文
    market_summary, performance_metrics = load_recent_context()
    
    if not market_summary:
        print("❌ 未找到历史日志，无法进行基于数据的进化。")
        print("请先运行主程序一段时间生成数据。")
        return

    print("正在请求 AI 架构师优化策略...")
    print(f"市场环境: {market_summary.strip()}")
    print(f"诊断问题: {performance_metrics['issue']}")
    print("-" * 60)
    
    start_time = time.time()
    success = architect.optimize_strategy(market_summary, performance_metrics)
    duration = time.time() - start_time
    
    if success:
        print(f"\n✅ 策略进化成功! (耗时: {duration:.1f}s)")
        print("新的策略代码已写入 src/dynamic_strategy.py")
        print("请重启主程序以应用新策略。")
    else:
        print("\n❌ 策略进化失败，请检查 Ollama 服务或日志。")

if __name__ == "__main__":
    main()
