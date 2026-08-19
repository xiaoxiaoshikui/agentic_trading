"""
参数优化回测
测试多组参数找到最佳配置
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from binance.client import Client
from src.advanced_strategy import generate_advanced_signal


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
        except Exception:
            break
    
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = df[col].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df.drop_duplicates().sort_index()


def run_backtest_with_params(
    df_15m, df_1h, df_4h,
    sl_mult, tp_mult, min_conf, mtf_req, conf_threshold,
    initial_balance=10000
):
    """用指定参数运行回测"""
    balance = initial_balance
    position = None
    trades = []
    peak = balance
    max_dd = 0
    
    for i in range(300, len(df_15m)):
        current_time = df_15m.index[i]
        current_price = float(df_15m['close'].iloc[i])
        current_low = float(df_15m['low'].iloc[i])
        current_high = float(df_15m['high'].iloc[i])
        
        # 检查持仓
        if position:
            if position['side'] == 'LONG':
                if current_low <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['size'] * 0.999
                    balance += pnl
                    trades.append({'pnl': pnl, 'win': False})
                    position = None
                elif current_high >= position['tp']:
                    pnl = (position['tp'] - position['entry']) * position['size'] * 0.999
                    balance += pnl
                    trades.append({'pnl': pnl, 'win': True})
                    position = None
            else:
                if current_high >= position['sl']:
                    pnl = (position['entry'] - position['sl']) * position['size'] * 0.999
                    balance += pnl
                    trades.append({'pnl': pnl, 'win': False})
                    position = None
                elif current_low <= position['tp']:
                    pnl = (position['entry'] - position['tp']) * position['size'] * 0.999
                    balance += pnl
                    trades.append({'pnl': pnl, 'win': True})
                    position = None
        
        # 检查新信号
        if position is None and balance > 0:
            df_15m_slice = df_15m.iloc[:i+1]
            sig_15m, atr = generate_advanced_signal(df_15m_slice, min_confirmations=min_conf)
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # 多时间框架
            df_1h_slice = df_1h[df_1h.index <= current_time]
            df_4h_slice = df_4h[df_4h.index <= current_time]
            
            sig_1h_side = 'FLAT'
            sig_4h_side = 'FLAT'
            
            if len(df_1h_slice) >= 200:
                sig_1h, _ = generate_advanced_signal(df_1h_slice, min_confirmations=min_conf)
                sig_1h_side = sig_1h.side
            
            if len(df_4h_slice) >= 200:
                sig_4h, _ = generate_advanced_signal(df_4h_slice, min_confirmations=min_conf)
                sig_4h_side = sig_4h.side
            
            signals = [sig_15m.side, sig_1h_side, sig_4h_side]
            long_count = signals.count('LONG')
            short_count = signals.count('SHORT')
            
            if long_count >= mtf_req and sig_15m.side == 'LONG' and sig_15m.confidence >= conf_threshold:
                sl = current_price - atr * sl_mult
                tp = current_price + atr * tp_mult
                size = (balance * 0.01) / (current_price - sl)
                position = {'side': 'LONG', 'entry': current_price, 'sl': sl, 'tp': tp, 'size': size}
                
            elif short_count >= mtf_req and sig_15m.side == 'SHORT' and sig_15m.confidence >= conf_threshold:
                sl = current_price + atr * sl_mult
                tp = current_price - atr * tp_mult
                size = (balance * 0.01) / (sl - current_price)
                position = {'side': 'SHORT', 'entry': current_price, 'sl': sl, 'tp': tp, 'size': size}
        
        if balance > peak:
            peak = balance
        dd = peak - balance
        if dd > max_dd:
            max_dd = dd
    
    # 计算结果
    if not trades:
        return None
    
    wins = [t for t in trades if t['win']]
    total_pnl = balance - initial_balance
    win_rate = len(wins) / len(trades)
    
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in trades if not t['win']))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'pnl_pct': total_pnl / initial_balance * 100,
        'max_dd': max_dd / initial_balance * 100,
        'profit_factor': pf
    }


def main():
    print("=" * 70)
    print("Parameter Optimization")
    print("=" * 70)
    
    # 获取数据
    client = Client('', '')
    
    print("\nFetching data...")
    df_15m = get_historical_data(client, 'BTCUSDT', '15m', 12)
    df_1h = get_historical_data(client, 'BTCUSDT', '1h', 6)
    df_4h = get_historical_data(client, 'BTCUSDT', '4h', 3)
    print(f"Data: {len(df_15m)} x 15m, {len(df_1h)} x 1h, {len(df_4h)} x 4h")
    
    # 参数组合
    param_sets = [
        # (sl_mult, tp_mult, min_conf, mtf_req, conf_threshold, name)
        (2.0, 4.0, 2, 2, 0.5, "Baseline"),
        (1.5, 3.0, 2, 2, 0.5, "Tighter SL/TP"),
        (1.5, 4.5, 2, 2, 0.5, "Tight SL, Wide TP"),
        (2.0, 3.0, 2, 2, 0.5, "Lower RR"),
        (2.0, 4.0, 3, 2, 0.5, "More confirmations"),
        (2.0, 4.0, 2, 3, 0.5, "All 3 MTF required"),
        (2.0, 4.0, 2, 2, 0.6, "Higher confidence"),
        (2.0, 4.0, 3, 3, 0.6, "Strict all"),
        (1.5, 3.0, 3, 2, 0.6, "Tight + Strict"),
        (2.5, 5.0, 2, 2, 0.5, "Wider SL/TP"),
        (1.0, 2.0, 2, 2, 0.5, "Very tight 1:2"),
        (1.5, 2.25, 2, 2, 0.5, "RR 1:1.5"),
    ]
    
    results = []
    
    print("\nTesting parameter combinations...")
    print("-" * 70)
    
    for i, (sl, tp, mc, mtf, conf, name) in enumerate(param_sets):
        print(f"[{i+1}/{len(param_sets)}] Testing: {name}...", end=" ")
        
        result = run_backtest_with_params(
            df_15m, df_1h, df_4h,
            sl_mult=sl, tp_mult=tp, 
            min_conf=mc, mtf_req=mtf, 
            conf_threshold=conf
        )
        
        if result:
            results.append({
                'name': name,
                'sl': sl, 'tp': tp, 'min_conf': mc, 
                'mtf_req': mtf, 'conf': conf,
                **result
            })
            print(f"Trades: {result['trades']}, PnL: {result['pnl_pct']:.1f}%, WR: {result['win_rate']*100:.1f}%")
        else:
            print("No trades")
    
    # 排序结果
    print("\n" + "=" * 70)
    print("RESULTS RANKED BY PROFIT")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Name':<20} {'Trades':<8} {'Win%':<8} {'PnL%':<10} {'MaxDD%':<10} {'PF':<8}")
    print("-" * 70)
    
    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1:<5} {r['name']:<20} {r['trades']:<8} {r['win_rate']*100:<8.1f} {r['pnl_pct']:<10.1f} {r['max_dd']:<10.1f} {r['profit_factor']:<8.2f}")
    
    # 最佳参数
    best = results_sorted[0]
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"Name:            {best['name']}")
    print(f"SL multiplier:   {best['sl']}x ATR")
    print(f"TP multiplier:   {best['tp']}x ATR")
    print(f"Min confirmations: {best['min_conf']}")
    print(f"MTF required:    {best['mtf_req']}/3")
    print(f"Confidence:      {best['conf']}")
    print("-" * 70)
    print(f"Total trades:    {best['trades']}")
    print(f"Win rate:        {best['win_rate']*100:.1f}%")
    print(f"Total PnL:       {best['pnl_pct']:.1f}%")
    print(f"Max drawdown:    {best['max_dd']:.1f}%")
    print(f"Profit factor:   {best['profit_factor']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
