"""
Evolution Testing Script
Tests the LLM strategy evolution by:
1. Backing up current strategy
2. Running backtest on current strategy
3. Triggering evolution
4. Running backtest on new strategy
5. Comparing results
"""
import sys
import os
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from binance.client import Client
from src.local_llm import LocalStrategyArchitect
from src.multi_asset_simulator import MultiAssetSimulator, print_multi_asset_report
from src.advanced_risk import AdvancedRiskManager
import src.dynamic_strategy as dynamic_strategy
import importlib


STRATEGY_FILE = "src/dynamic_strategy.py"
BACKUP_DIR = "strategy_backups"
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def backup_strategy(suffix="before"):
    """Backup current strategy with timestamp"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"dynamic_strategy_{timestamp}_{suffix}.py")
    shutil.copy(STRATEGY_FILE, backup_path)
    print(f"📦 Strategy backed up to: {backup_path}")
    return backup_path


def save_evolution_result(result: dict, strategy_path: str):
    """Save evolution result metadata"""
    meta_path = strategy_path.replace(".py", "_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Trades: {result.get('total_trades', 0)}\n")
        f.write(f"Win Rate: {result.get('win_rate', 0):.1%}\n")
        f.write(f"Total PnL: ${result.get('total_pnl', 0):.2f}\n")
        f.write(f"ROI: {result.get('roi_percent', 0):.1f}%\n")
        f.write(f"Max Drawdown: {result.get('max_drawdown_percent', 0):.1f}%\n")
        f.write(f"Profit Factor: {result.get('profit_factor', 0):.2f}\n")
    print(f"📝 Metadata saved to: {meta_path}")


def get_historical_data(client, symbol, interval="15m", batches=4):
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
        except Exception as e:
            print(f"    Error fetching {symbol}: {e}")
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
    df['funding_rate'] = 0.0
    df['open_interest'] = 0.0
    
    return df


def run_quick_backtest(data: dict, strategy_module) -> dict:
    """Run a quick backtest on the given strategy"""
    simulator = MultiAssetSimulator(
        initial_capital=10000.0,
        max_positions=3,
        capital_per_position=0.3
    )
    risk_manager = AdvancedRiskManager()
    
    symbols = list(data.keys())
    
    # Find common time range
    min_time = max(df.index[0] for df in data.values())
    max_time = min(df.index[-1] for df in data.values())
    
    for symbol in data:
        data[symbol] = data[symbol][(data[symbol].index >= min_time) & (data[symbol].index <= max_time)]
    
    common_times = data[symbols[0]].index
    total_bars = len(common_times)
    
    for i in range(200, total_bars):
        current_time = common_times[i]
        current_timestamp = current_time.timestamp()
        
        # Get prices
        prices = {}
        for symbol, df in data.items():
            if current_time in df.index:
                prices[symbol] = float(df.loc[current_time, 'close'])
        
        # Check positions
        simulator.check_positions(prices, current_timestamp)
        
        # Check each symbol
        for symbol, df in data.items():
            if not simulator.can_open_position(symbol):
                continue
            
            df_slice = df.iloc[:i+1].copy()
            if len(df_slice) < 200:
                continue
            
            try:
                signal = strategy_module.calculate_signal(df_slice)
            except Exception:
                continue
            
            if signal.side == "FLAT" or signal.confidence < 0.3:
                continue
            
            current_price = prices.get(symbol)
            if not current_price:
                continue
            
            atr = df_slice['atr'].iloc[-1] if 'atr' in df_slice.columns else current_price * 0.01
            if pd.isna(atr) or atr <= 0:
                atr = current_price * 0.01
            
            if signal.side == "LONG":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 4)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 4)
            
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
            
            simulator.open_position(
                symbol=symbol,
                direction=signal.side,
                entry_price=current_price,
                position_size=plan.position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_time=current_timestamp
            )
    
    # Close remaining positions
    final_prices = {s: float(data[s].iloc[-1]['close']) for s in symbols}
    for symbol in list(simulator.positions.keys()):
        if symbol in final_prices:
            simulator.close_position(symbol, final_prices[symbol], "End of Backtest")
    
    return simulator.get_report()


def compare_results(before: dict, after: dict):
    """Compare two backtest results"""
    print("\n" + "=" * 60)
    print("         EVOLUTION COMPARISON")
    print("=" * 60)
    
    metrics = [
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Total PnL", "total_pnl", "${:+.2f}"),
        ("ROI", "roi_percent", "{:+.1f}%"),
        ("Max Drawdown", "max_drawdown_percent", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
    ]
    
    print(f"\n{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 60)
    
    improvements = 0
    
    for name, key, fmt in metrics:
        before_val = before.get(key, 0)
        after_val = after.get(key, 0)
        
        # Handle percentage formatting
        if "rate" in key.lower():
            before_str = f"{before_val:.1%}"
            after_str = f"{after_val:.1%}"
            change = (after_val - before_val) * 100
            change_str = f"{change:+.1f}pp"
        elif "pnl" in key.lower():
            before_str = f"${before_val:.2f}"
            after_str = f"${after_val:.2f}"
            change = after_val - before_val
            change_str = f"${change:+.2f}"
        elif "percent" in key.lower():
            before_str = f"{before_val:.1f}%"
            after_str = f"{after_val:.1f}%"
            change = after_val - before_val
            change_str = f"{change:+.1f}%"
        else:
            before_str = fmt.format(before_val)
            after_str = fmt.format(after_val)
            change = after_val - before_val
            change_str = f"{change:+.2f}"
        
        # Determine if improvement (considering metric direction)
        is_better = False
        if key in ["win_rate", "total_pnl", "roi_percent", "profit_factor"]:
            is_better = after_val > before_val
        elif key == "max_drawdown_percent":
            is_better = after_val < before_val
        
        if is_better:
            improvements += 1
            emoji = "🟢"
        elif after_val == before_val:
            emoji = "⚪"
        else:
            emoji = "🔴"
        
        print(f"{name:<20} {before_str:<15} {after_str:<15} {emoji} {change_str}")
    
    print("-" * 60)
    
    # Overall verdict
    if improvements >= 4:
        verdict = "✅ EVOLUTION SUCCESS - Strategy improved!"
    elif improvements >= 2:
        verdict = "⚠️ MIXED RESULTS - Some improvements"
    else:
        verdict = "❌ EVOLUTION FAILED - Consider reverting"
    
    print(f"\n{verdict}")
    print("=" * 60)
    
    return improvements >= 3


def main():
    print("=" * 60)
    print("     STRATEGY EVOLUTION TEST")
    print("=" * 60)
    
    # Step 1: Backup current strategy (BEFORE)
    before_path = backup_strategy("before")
    
    # Step 2: Fetch test data
    print("\n📊 Fetching test data...")
    client = Client('', '')
    
    data = {}
    for symbol in TEST_SYMBOLS:
        print(f"   {symbol}...", end=" ")
        df = get_historical_data(client, symbol)
        if df is not None and len(df) >= 300:
            data[symbol] = df
            print(f"{len(df)} candles")
        else:
            print("skipped")
    
    if len(data) < 2:
        print("❌ Not enough data for testing")
        return
    
    # Step 3: Backtest BEFORE evolution
    print("\n🔬 Running backtest on CURRENT strategy...")
    before_result = run_quick_backtest(data.copy(), dynamic_strategy)
    print(f"   Trades: {before_result['total_trades']} | Win: {before_result['win_rate']:.1%} | PnL: ${before_result['total_pnl']:.2f}")
    
    # Step 4: Trigger evolution
    print("\n🧬 Triggering strategy evolution...")
    architect = LocalStrategyArchitect(model_name="qwen3:8b")
    
    evolution_context = f"""
EVOLUTION TEST
==============
Testing on: {TEST_SYMBOLS}
Current Performance:
- Trades: {before_result['total_trades']}
- Win Rate: {before_result['win_rate']:.1%}
- PnL: ${before_result['total_pnl']:.2f}
- Max Drawdown: {before_result['max_drawdown_percent']:.1f}%

Goal: Improve win rate and reduce drawdown while maintaining profitability.
"""
    
    performance = {
        "win_rate": before_result['win_rate'] * 100,
        "issue": "Testing evolution capability",
    }
    
    success = architect.optimize_strategy(evolution_context, performance)
    
    if not success:
        print("❌ Evolution failed!")
        return
    
    print("✅ Evolution completed!")
    
    # Step 5: Save evolved strategy (AFTER)
    after_path = backup_strategy("after")
    
    # Step 6: Reload and backtest AFTER evolution
    print("\n🔬 Running backtest on NEW strategy...")
    importlib.reload(dynamic_strategy)
    after_result = run_quick_backtest(data.copy(), dynamic_strategy)
    print(f"   Trades: {after_result['total_trades']} | Win: {after_result['win_rate']:.1%} | PnL: ${after_result['total_pnl']:.2f}")
    
    # Step 7: Save metadata for both versions
    save_evolution_result(before_result, before_path)
    save_evolution_result(after_result, after_path)
    
    # Step 8: Compare results
    is_better = compare_results(before_result, after_result)
    
    # Step 9: Option to revert
    if not is_better:
        print(f"\n💡 To revert to previous strategy:")
        print(f"   copy {before_path} {STRATEGY_FILE}")
    else:
        print(f"\n💾 Strategies saved in: {BACKUP_DIR}/")
        print(f"   Before: {os.path.basename(before_path)}")
        print(f"   After:  {os.path.basename(after_path)}")


if __name__ == "__main__":
    main()
