"""
多时间框架回测脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from binance.client import Client
from src.advanced_risk import AdvancedRiskManager
import src.dynamic_strategy as dynamic_strategy


def get_historical_data(client, symbol, interval, batches=12):
    """获取历史K线数据"""
    all_klines = []
    end_time = None
    
    for i in range(batches):
        try:
            if end_time:
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=1000, endTime=end_time)
            else:
                klines = client.futures_klines(symbol=symbol, interval=interval, limit=1000)
            
            if not klines:
                break
                
            all_klines = klines + all_klines
            end_time = klines[0][0] - 1
            print(f"    Batch {i+1}: {len(klines)} candles")
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = df[col].astype(float)
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    
    return df.drop_duplicates().sort_index()


def run_multi_timeframe_backtest(
    symbol="BTCUSDT",
    initial_balance=10000,
    risk_per_trade=0.01,
    min_confirmations=2,
    mtf_required=2  # 至少几个时间框架一致
):
    """
    运行多时间框架回测
    """
    print("=" * 60)
    print("Multi-Timeframe Backtest")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Initial Balance: ${initial_balance:,}")
    print(f"Risk per trade: {risk_per_trade*100}%")
    print(f"Min confirmations: {min_confirmations}")
    print(f"MTF required: {mtf_required}/3")
    print("=" * 60)
    
    # 连接币安
    client = Client('', '')
    
    # 获取多时间框架数据
    print("\nFetching 15m data...")
    df_15m = get_historical_data(client, symbol, '15m', 12)
    print(f"  Total: {len(df_15m)} candles")
    
    print("\nFetching 1h data...")
    df_1h = get_historical_data(client, symbol, '1h', 6)
    print(f"  Total: {len(df_1h)} candles")
    
    print("\nFetching 4h data...")
    df_4h = get_historical_data(client, symbol, '4h', 3)
    print(f"  Total: {len(df_4h)} candles")
    
    print(f"\nData range: {df_15m.index[0]} to {df_15m.index[-1]}")
    print(f"Days: {(df_15m.index[-1] - df_15m.index[0]).days}")
    
    # 回测变量
    balance = initial_balance
    position = None
    trades = []
    equity_curve = [balance]
    peak = balance
    max_dd = 0
    
    print("\nRunning backtest...")
    print("This may take a few minutes...")
    
    # 遍历15m数据
    total_bars = len(df_15m)
    for i in range(300, total_bars):
        # 进度显示
        if i % 1000 == 0:
            progress = (i - 300) / (total_bars - 300) * 100
            print(f"  Progress: {progress:.1f}% ({len(trades)} trades)")
        
        current_time = df_15m.index[i]
        current_price = float(df_15m['close'].iloc[i])
        current_low = float(df_15m['low'].iloc[i])
        current_high = float(df_15m['high'].iloc[i])
        
        # 检查持仓止损止盈
        if position:
            if position['side'] == 'LONG':
                if current_low <= position['sl']:
                    # 止损
                    pnl = (position['sl'] - position['entry']) * position['size']
                    pnl *= 0.999  # 手续费
                    balance += pnl
                    trades.append({
                        'pnl': pnl, 
                        'win': False, 
                        'side': 'LONG', 
                        'exit': 'SL',
                        'entry_price': position['entry'],
                        'exit_price': position['sl']
                    })
                    position = None
                    
                elif current_high >= position['tp']:
                    # 止盈
                    pnl = (position['tp'] - position['entry']) * position['size']
                    pnl *= 0.999
                    balance += pnl
                    trades.append({
                        'pnl': pnl, 
                        'win': True, 
                        'side': 'LONG', 
                        'exit': 'TP',
                        'entry_price': position['entry'],
                        'exit_price': position['tp']
                    })
                    position = None
                    
            else:  # SHORT
                if current_high >= position['sl']:
                    # 止损
                    pnl = (position['entry'] - position['sl']) * position['size']
                    pnl *= 0.999
                    balance += pnl
                    trades.append({
                        'pnl': pnl, 
                        'win': False, 
                        'side': 'SHORT', 
                        'exit': 'SL',
                        'entry_price': position['entry'],
                        'exit_price': position['sl']
                    })
                    position = None
                    
                elif current_low <= position['tp']:
                    # 止盈
                    pnl = (position['entry'] - position['tp']) * position['size']
                    pnl *= 0.999
                    balance += pnl
                    trades.append({
                        'pnl': pnl, 
                        'win': True, 
                        'side': 'SHORT', 
                        'exit': 'TP',
                        'entry_price': position['entry'],
                        'exit_price': position['tp']
                    })
                    position = None
        
        # 无持仓时检查新信号
        if position is None and balance > 0:
            # 15m 信号 - 使用 dynamic_strategy
            df_15m_slice = df_15m.iloc[:i+1].copy()
            df_15m_slice['funding_rate'] = 0.0
            df_15m_slice['open_interest'] = 0.0
            sig_15m = dynamic_strategy.calculate_signal(df_15m_slice)
            atr = df_15m_slice['atr'].iloc[-1] if 'atr' in df_15m_slice.columns else current_price * 0.01
            
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.01
            
            # 1h 信号
            df_1h_slice = df_1h[df_1h.index <= current_time].copy()
            if len(df_1h_slice) >= 200:
                df_1h_slice['funding_rate'] = 0.0
                df_1h_slice['open_interest'] = 0.0
                sig_1h = dynamic_strategy.calculate_signal(df_1h_slice)
                sig_1h_side = sig_1h.side
            else:
                sig_1h_side = 'FLAT'
            
            # 4h 信号
            df_4h_slice = df_4h[df_4h.index <= current_time].copy()
            if len(df_4h_slice) >= 200:
                df_4h_slice['funding_rate'] = 0.0
                df_4h_slice['open_interest'] = 0.0
                sig_4h = dynamic_strategy.calculate_signal(df_4h_slice)
                sig_4h_side = sig_4h.side
            else:
                sig_4h_side = 'FLAT'
            
            # 统计多时间框架一致性
            signals = [sig_15m.side, sig_1h_side, sig_4h_side]
            long_count = signals.count('LONG')
            short_count = signals.count('SHORT')
            
            # 开仓条件：至少 mtf_required 个时间框架一致
            if long_count >= mtf_required and sig_15m.side == 'LONG' and sig_15m.confidence >= 0.5:
                # 开多
                sl = current_price - atr * 2
                tp = current_price + atr * 4
                risk_amount = balance * risk_per_trade
                size = risk_amount / (current_price - sl)
                
                position = {
                    'side': 'LONG', 
                    'entry': current_price, 
                    'sl': sl, 
                    'tp': tp, 
                    'size': size
                }
                
            elif short_count >= mtf_required and sig_15m.side == 'SHORT' and sig_15m.confidence >= 0.5:
                # 开空
                sl = current_price + atr * 2
                tp = current_price - atr * 4
                risk_amount = balance * risk_per_trade
                size = risk_amount / (sl - current_price)
                
                position = {
                    'side': 'SHORT', 
                    'entry': current_price, 
                    'sl': sl, 
                    'tp': tp, 
                    'size': size
                }
        
        # 记录权益
        equity_curve.append(balance)
        
        # 计算回撤
        if balance > peak:
            peak = balance
        dd = peak - balance
        if dd > max_dd:
            max_dd = dd
    
    # 打印结果
    print_results(trades, initial_balance, balance, max_dd, equity_curve)
    
    return trades, equity_curve


def print_results(trades, initial_balance, final_balance, max_dd, equity_curve):
    """打印回测结果"""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    if not trades:
        print("No trades executed!")
        return
    
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    total_pnl = final_balance - initial_balance
    
    print(f"\nTrade Statistics:")
    print(f"  Total trades:   {len(trades)}")
    print(f"  Winning trades: {len(wins)}")
    print(f"  Losing trades:  {len(losses)}")
    print(f"  Win rate:       {len(wins)/len(trades)*100:.1f}%")
    
    print(f"\nProfit/Loss:")
    print(f"  Total PnL:      ${total_pnl:.2f} ({total_pnl/initial_balance*100:.1f}%)")
    print(f"  Max drawdown:   ${max_dd:.2f} ({max_dd/initial_balance*100:.1f}%)")
    
    if wins and losses:
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        print(f"  Profit factor:  {pf:.2f}")
        print(f"  Avg win:        ${gross_profit/len(wins):.2f}")
        print(f"  Avg loss:       ${gross_loss/len(losses):.2f}")
        print(f"  Best trade:     ${max(t['pnl'] for t in trades):.2f}")
        print(f"  Worst trade:    ${min(t['pnl'] for t in trades):.2f}")
    
    # 多空统计
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']
    print(f"\nLong/Short:")
    print(f"  Long trades:    {len(long_trades)}")
    print(f"  Short trades:   {len(short_trades)}")
    
    # 评分
    print("\n" + "=" * 60)
    score = 0
    wr = len(wins)/len(trades)
    
    if wr >= 0.45:
        score += 2
        print("[+2] Win rate >= 45%")
    else:
        print("[ 0] Win rate < 45%")
    
    if wins and losses:
        if pf >= 1.3:
            score += 2
            print("[+2] Profit factor >= 1.3")
        else:
            print("[ 0] Profit factor < 1.3")
    
    if max_dd/initial_balance <= 0.25:
        score += 2
        print("[+2] Max drawdown <= 25%")
    else:
        print("[ 0] Max drawdown > 25%")
    
    if total_pnl > 0:
        score += 2
        print("[+2] Profitable")
    else:
        print("[ 0] Not profitable")
    
    print("=" * 60)
    ratings = {8: 'EXCELLENT', 6: 'GOOD', 4: 'AVERAGE', 2: 'BELOW AVG', 0: 'POOR'}
    for threshold, rating in sorted(ratings.items(), reverse=True):
        if score >= threshold:
            print(f"RATING: {rating} ({score}/8)")
            break
    print("=" * 60)


if __name__ == "__main__":
    run_multi_timeframe_backtest(
        symbol="BTCUSDT",
        initial_balance=10000,
        risk_per_trade=0.01,
        min_confirmations=2,
        mtf_required=2
    )
