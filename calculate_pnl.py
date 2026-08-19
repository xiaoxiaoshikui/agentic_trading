"""
收益率计算器
分析 Dry Run 模式下的模拟交易表现
"""

import json
import os
import sys
import glob
from datetime import datetime
from typing import List, Dict
from binance.client import Client
import pandas as pd

# 修复 Windows 控制台编码
sys.stdout.reconfigure(encoding='utf-8')

# 配置
INITIAL_CAPITAL = 10000  # 初始资金 $10,000
LEVERAGE = 10            # 杠杆倍数
FEE_RATE = 0.0004        # 手续费率 (0.04% taker)


def load_logs(log_dir: str = "data/history") -> List[Dict]:
    """加载所有日志"""
    files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    files.sort()
    records = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def get_price_data(symbol: str = "BTCUSDT", days: int = 7) -> pd.DataFrame:
    """获取历史价格数据用于验证"""
    client = Client()
    klines = client.futures_klines(symbol=symbol, interval="1m", limit=1000)
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df


def simulate_trade(trade: Dict, price_data: pd.DataFrame, lookforward_minutes: int = 240) -> Dict:
    """
    模拟单笔交易结果
    检查后续 4 小时内是否触及止盈或止损
    """
    try:
        trade_time = datetime.fromtimestamp(trade['timestamp_ts'])
        entry_price = float(trade['entry_price'])
        stop_loss = float(trade['stop_loss'])
        take_profit = float(trade['take_profit'])
        position_size = float(trade['position_size'])
        action = trade['action']
        
        # 找到交易时间之后的 K 线
        future_data = price_data[price_data.index >= trade_time].head(lookforward_minutes)
        
        if future_data.empty:
            return {
                "status": "PENDING",
                "pnl": 0,
                "pnl_percent": 0,
                "exit_price": entry_price,
                "exit_reason": "数据不足"
            }
        
        # 逐根 K 线检查是否触及止盈或止损
        for idx, candle in future_data.iterrows():
            if action == "LONG":
                # 先检查止损（价格可能先触及止损再反弹）
                if candle['low'] <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price * position_size * LEVERAGE
                    return {
                        "status": "LOSS",
                        "pnl": pnl * INITIAL_CAPITAL - (entry_price * position_size * FEE_RATE * 2),
                        "pnl_percent": (stop_loss - entry_price) / entry_price * 100 * LEVERAGE,
                        "exit_price": stop_loss,
                        "exit_reason": "止损触发"
                    }
                # 再检查止盈
                if candle['high'] >= take_profit:
                    pnl = (take_profit - entry_price) / entry_price * position_size * LEVERAGE
                    return {
                        "status": "WIN",
                        "pnl": pnl * INITIAL_CAPITAL - (entry_price * position_size * FEE_RATE * 2),
                        "pnl_percent": (take_profit - entry_price) / entry_price * 100 * LEVERAGE,
                        "exit_price": take_profit,
                        "exit_reason": "止盈触发"
                    }
                    
            elif action == "SHORT":
                # 先检查止损
                if candle['high'] >= stop_loss:
                    pnl = (entry_price - stop_loss) / entry_price * position_size * LEVERAGE
                    return {
                        "status": "LOSS",
                        "pnl": pnl * INITIAL_CAPITAL - (entry_price * position_size * FEE_RATE * 2),
                        "pnl_percent": (entry_price - stop_loss) / entry_price * 100 * LEVERAGE,
                        "exit_price": stop_loss,
                        "exit_reason": "止损触发"
                    }
                # 再检查止盈
                if candle['low'] <= take_profit:
                    pnl = (entry_price - take_profit) / entry_price * position_size * LEVERAGE
                    return {
                        "status": "WIN",
                        "pnl": pnl * INITIAL_CAPITAL - (entry_price * position_size * FEE_RATE * 2),
                        "pnl_percent": (entry_price - take_profit) / entry_price * 100 * LEVERAGE,
                        "exit_price": take_profit,
                        "exit_reason": "止盈触发"
                    }
        
        # 4 小时内未触及止盈止损，按最后价格平仓
        last_price = float(future_data.iloc[-1]['close'])
        if action == "LONG":
            pnl_percent = (last_price - entry_price) / entry_price * 100 * LEVERAGE
            pnl = (last_price - entry_price) / entry_price * position_size * LEVERAGE * INITIAL_CAPITAL
        else:
            pnl_percent = (entry_price - last_price) / entry_price * 100 * LEVERAGE
            pnl = (entry_price - last_price) / entry_price * position_size * LEVERAGE * INITIAL_CAPITAL
            
        pnl -= (entry_price * position_size * FEE_RATE * 2)  # 扣手续费
        
        return {
            "status": "WIN" if pnl > 0 else "LOSS",
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "exit_price": last_price,
            "exit_reason": "超时平仓 (4h)"
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "pnl": 0,
            "pnl_percent": 0,
            "exit_price": 0,
            "exit_reason": str(e)
        }


def filter_duplicate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    模拟真实持仓状态，过滤重复开仓
    规则：
    1. 开仓后，同方向信号不再开仓
    2. 持仓期间（4小时内）不允许同向开仓
    3. 允许反向开仓（相当于平仓后反手）
    """
    if trades.empty:
        return trades
    
    filtered_trades = []
    current_position = None  # 当前持仓: {'direction': 'LONG'/'SHORT', 'open_time': timestamp}
    HOLD_DURATION = 4 * 60 * 60  # 4小时持仓时间（秒）
    
    for _, trade in trades.iterrows():
        trade_time = trade['timestamp_ts']
        trade_direction = trade['action']
        
        # 检查当前是否有持仓
        if current_position is not None:
            time_held = trade_time - current_position['open_time']
            
            # 持仓超过4小时，自动平仓
            if time_held >= HOLD_DURATION:
                current_position = None
            # 同方向信号，跳过（不重复开仓）
            elif trade_direction == current_position['direction']:
                continue
            # 反向信号，平仓并反手
            else:
                current_position = None
        
        # 开新仓
        filtered_trades.append(trade)
        current_position = {
            'direction': trade_direction,
            'open_time': trade_time
        }
    
    return pd.DataFrame(filtered_trades)


def calculate_performance():
    """计算整体表现（模拟真实持仓状态，避免重复开仓）"""
    print("\n" + "=" * 70)
    print("💰 Agentic Trading - 收益率分析报告 (模拟实盘)")
    print("=" * 70)
    
    # 1. 加载日志
    records = load_logs()
    if not records:
        print("❌ 未找到日志文件")
        return
    
    df = pd.DataFrame(records)
    
    # 2. 筛选实际触发的交易信号
    all_trades = df[df['action'].isin(['LONG', 'SHORT'])].copy()
    
    if all_trades.empty:
        print("📭 暂无交易记录")
        return
    
    # 3. 模拟真实持仓状态，过滤重复开仓
    trades = filter_duplicate_trades(all_trades)
    
    print(f"📊 分析时间: {datetime.fromtimestamp(records[0]['timestamp_ts']).strftime('%Y-%m-%d %H:%M')} 至 {datetime.fromtimestamp(records[-1]['timestamp_ts']).strftime('%Y-%m-%d %H:%M')}")
    print(f"📈 总信号数: {len(df)} | 原始交易信号: {len(all_trades)} | 实际开仓: {len(trades)}")
    
    # 3. 获取价格数据
    print("\n⏳ 正在获取价格数据验证交易...")
    try:
        price_data = get_price_data()
    except Exception as e:
        print(f"❌ 获取价格数据失败: {e}")
        print("\n📝 使用理论计算 (基于止盈止损设置)...")
        calculate_theoretical_performance(trades)
        return
    
    # 4. 模拟每笔交易
    results = []
    for _, trade in trades.iterrows():
        result = simulate_trade(trade, price_data)
        result['time'] = datetime.fromtimestamp(trade['timestamp_ts']).strftime('%H:%M')
        result['action'] = trade['action']
        result['entry'] = trade['entry_price']
        result['score'] = trade.get('score', 0)
        results.append(result)
    
    # 5. 汇总统计
    results_df = pd.DataFrame(results)
    
    wins = len(results_df[results_df['status'] == 'WIN'])
    losses = len(results_df[results_df['status'] == 'LOSS'])
    pending = len(results_df[results_df['status'] == 'PENDING'])
    total_pnl = results_df['pnl'].sum()
    total_pnl_percent = results_df['pnl_percent'].sum()
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    # 计算最大回撤
    cumulative = results_df['pnl'].cumsum()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak)
    max_drawdown = drawdown.min()
    
    # 平均盈亏
    avg_win = results_df[results_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = results_df[results_df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
    profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
    
    print("\n" + "-" * 70)
    print("📊 交易明细:")
    print("-" * 70)
    print(f"{'时间':<8} {'方向':<6} {'入场价':<12} {'出场价':<12} {'盈亏':<12} {'结果':<10}")
    print("-" * 70)
    
    for r in results:
        pnl_str = f"${r['pnl']:.2f}" if r['pnl'] != 0 else "-"
        status_emoji = "✅" if r['status'] == 'WIN' else "❌" if r['status'] == 'LOSS' else "⏳"
        print(f"{r['time']:<8} {r['action']:<6} ${r['entry']:<11.2f} ${r['exit_price']:<11.2f} {pnl_str:<12} {status_emoji} {r['exit_reason']}")
    
    print("\n" + "=" * 70)
    print("📈 绩效汇总:")
    print("=" * 70)
    print(f"💵 初始资金:     ${INITIAL_CAPITAL:,.2f}")
    print(f"💰 总盈亏:       ${total_pnl:,.2f} ({total_pnl_percent:+.2f}%)")
    print(f"📊 最终资金:     ${INITIAL_CAPITAL + total_pnl:,.2f}")
    print(f"📉 最大回撤:     ${max_drawdown:,.2f}")
    print()
    print(f"🎯 胜率:         {win_rate:.1f}% ({wins}胜 / {losses}负)")
    print(f"📊 盈亏比:       {profit_factor:.2f}")
    print(f"💹 平均盈利:     ${avg_win:,.2f}")
    print(f"💸 平均亏损:     ${avg_loss:,.2f}")
    print()
    print(f"⚙️  杠杆:         {LEVERAGE}x")
    print(f"💳 手续费率:     {FEE_RATE * 100:.2f}%")
    print("=" * 70)
    
    # 收益率
    roi = total_pnl / INITIAL_CAPITAL * 100
    print(f"\n🏆 投资回报率 (ROI): {roi:+.2f}%")
    print("=" * 70)


def calculate_theoretical_performance(trades: pd.DataFrame):
    """
    理论收益计算 (当无法获取实时数据时)
    假设所有交易都按 1:3 风险回报比例平仓
    """
    print("\n📊 理论收益分析 (基于风险回报比)")
    print("-" * 50)
    
    # 假设胜率 50%，止盈触发
    assumed_win_rate = 0.5
    total_trades = len(trades)
    
    # 计算理论盈亏
    total_risk = 0
    total_reward = 0
    
    for _, trade in trades.iterrows():
        entry = float(trade['entry_price'])
        sl = float(trade['stop_loss'])
        tp = float(trade['take_profit'])
        size = float(trade['position_size'])
        
        if trade['action'] == 'LONG':
            risk = (entry - sl) / entry * size * LEVERAGE
            reward = (tp - entry) / entry * size * LEVERAGE
        else:
            risk = (sl - entry) / entry * size * LEVERAGE
            reward = (entry - tp) / entry * size * LEVERAGE
        
        total_risk += abs(risk)
        total_reward += abs(reward)
    
    avg_risk = total_risk / total_trades * INITIAL_CAPITAL
    avg_reward = total_reward / total_trades * INITIAL_CAPITAL
    
    # 假设不同胜率下的预期收益
    print(f"\n假设 {total_trades} 笔交易:")
    print(f"  平均风险: ${avg_risk:.2f}")
    print(f"  平均收益: ${avg_reward:.2f}")
    print(f"  风险回报比: 1:{avg_reward/avg_risk:.1f}")
    
    print("\n📈 不同胜率下的预期收益:")
    for wr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        wins = int(total_trades * wr)
        losses = total_trades - wins
        expected_pnl = wins * avg_reward - losses * avg_risk
        print(f"  胜率 {wr*100:.0f}%: ${expected_pnl:+,.2f} ({expected_pnl/INITIAL_CAPITAL*100:+.2f}%)")


if __name__ == "__main__":
    calculate_performance()
