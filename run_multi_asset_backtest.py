"""
Multi-Asset Backtest Script
Tests dynamic_strategy.py across multiple symbols simultaneously
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
import src.dynamic_strategy as dynamic_strategy
from src.multi_asset_simulator import MultiAssetSimulator, print_multi_asset_report
from src.advanced_risk import AdvancedRiskManager


# Top symbols to backtest
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


def get_historical_data(client, symbol, interval, batches=6):
    """Fetch historical klines data"""
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
            print(f"    {symbol} batch {i+1}: {len(klines)} candles")
        except Exception as e:
            print(f"    {symbol} error: {e}")
            break
    
    if not all_klines:
        return None
        
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = df[col].astype(float)
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df.drop_duplicates().sort_index()
    
    # Add required columns for dynamic_strategy
    df['funding_rate'] = 0.0
    df['open_interest'] = 0.0
    
    return df


def run_multi_asset_backtest(
    symbols=None,
    initial_balance=10000,
    max_positions=3,
    capital_per_position=0.3,
    interval="15m"
):
    """
    Run multi-asset backtest
    
    Args:
        symbols: List of symbols to trade
        initial_balance: Starting capital
        max_positions: Maximum simultaneous positions
        capital_per_position: Fraction of capital per position (0.0-1.0)
        interval: Timeframe to use
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
        
    print("=" * 60)
    print("Multi-Asset Backtest")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Initial Balance: ${initial_balance:,}")
    print(f"Max Positions: {max_positions}")
    print(f"Capital per Position: {capital_per_position*100}%")
    print(f"Interval: {interval}")
    print("=" * 60)
    
    # Connect to Binance
    client = Client('', '')
    
    # Fetch data for all symbols
    print("\nFetching historical data...")
    data = {}
    for symbol in symbols:
        print(f"  {symbol}...")
        df = get_historical_data(client, symbol, interval, batches=6)
        if df is not None and len(df) >= 300:
            data[symbol] = df
            print(f"    Total: {len(df)} candles")
        else:
            print(f"    Skipped (insufficient data)")
    
    if not data:
        print("No data available!")
        return None
        
    # Find common time range
    min_time = max(df.index[0] for df in data.values())
    max_time = min(df.index[-1] for df in data.values())
    
    # Align all dataframes to common time range
    for symbol in data:
        data[symbol] = data[symbol][(data[symbol].index >= min_time) & (data[symbol].index <= max_time)]
    
    # Get common timestamps
    common_times = data[list(data.keys())[0]].index
    
    print(f"\nBacktest period: {min_time} to {max_time}")
    print(f"Common bars: {len(common_times)}")
    
    # Initialize simulator
    simulator = MultiAssetSimulator(
        initial_capital=initial_balance,
        max_positions=max_positions,
        capital_per_position=capital_per_position
    )
    
    risk_manager = AdvancedRiskManager()
    
    print("\nRunning backtest...")
    print("This may take a few minutes...")
    
    # Main backtest loop
    total_bars = len(common_times)
    
    for i in range(200, total_bars):
        # Progress
        if i % 500 == 0:
            progress = (i - 200) / (total_bars - 200) * 100
            print(f"  Progress: {progress:.1f}% | Trades: {simulator.total_trades} | Capital: ${simulator.capital:.2f}")
        
        current_time = common_times[i]
        current_timestamp = current_time.timestamp()
        
        # Get current prices for all symbols
        prices = {}
        for symbol, df in data.items():
            if current_time in df.index:
                prices[symbol] = float(df.loc[current_time, 'close'])
        
        # Check existing positions for SL/TP
        closed = simulator.check_positions(prices, current_timestamp)
        for trade in closed:
            emoji = "✅" if trade['win'] else "❌"
            print(f"  {emoji} Closed {trade['symbol']} {trade['direction']}: ${trade['pnl']:+.2f} ({trade['reason']})")
        
        # Check for new signals on each symbol
        for symbol, df in data.items():
            # Skip if already have position or can't open new one
            if not simulator.can_open_position(symbol):
                continue
            
            # Get data up to current bar
            df_slice = df.iloc[:i+1].copy()
            
            if len(df_slice) < 200:
                continue
            
            # Generate signal using dynamic_strategy
            try:
                signal = dynamic_strategy.calculate_signal(df_slice)
            except Exception as e:
                continue
            
            # Check if we have a valid signal
            if signal.side == "FLAT" or signal.confidence < 0.3:
                continue
            
            current_price = prices.get(symbol)
            if not current_price:
                continue
            
            # Calculate ATR for position sizing
            atr = df_slice['atr'].iloc[-1] if 'atr' in df_slice.columns else current_price * 0.01
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.01
            
            # Calculate stop loss and take profit
            if signal.side == "LONG":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 4)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 4)
            
            # Calculate position size
            try:
                plan = risk_manager.calculate_position(
                    balance=simulator.get_position_capital(),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    signal_confidence=signal.confidence,
                    market_regime="trending",
                    atr=atr,
                    atr_percent=atr/current_price
                )
            except Exception:
                continue
            
            if plan.position_size <= 0:
                continue
            
            # Open position
            opened = simulator.open_position(
                symbol=symbol,
                direction=signal.side,
                entry_price=current_price,
                position_size=plan.position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_time=current_timestamp
            )
            
            if opened:
                print(f"  🆕 {symbol} {signal.side} @ ${current_price:.2f} (conf: {signal.confidence:.2f})")
    
    # Close any remaining positions at end
    final_prices = {}
    for symbol, df in data.items():
        final_prices[symbol] = float(df.iloc[-1]['close'])
    
    for symbol in list(simulator.positions.keys()):
        if symbol in final_prices:
            simulator.close_position(symbol, final_prices[symbol], "End of Backtest")
    
    # Generate and print report
    report = simulator.get_report()
    print_multi_asset_report(report)
    
    return report, simulator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Asset Backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to trade")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--max-positions", type=int, default=3, help="Max simultaneous positions")
    parser.add_argument("--capital-per-pos", type=float, default=0.3, help="Capital fraction per position")
    parser.add_argument("--interval", default="15m", help="Timeframe")
    
    args = parser.parse_args()
    
    run_multi_asset_backtest(
        symbols=args.symbols,
        initial_balance=args.balance,
        max_positions=args.max_positions,
        capital_per_position=args.capital_per_pos,
        interval=args.interval
    )
