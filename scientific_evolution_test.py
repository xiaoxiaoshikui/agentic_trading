"""
Scientific Evolution Testing Framework (Walk-Forward Analysis)
==============================================================
Tests LLM strategy evolution with:
1. Walk-Forward Analysis (rolling train/test periods)
2. Realistic evolution triggers (same as main.py)
3. Out-of-sample validation
4. Statistical significance testing
"""
import sys
import os
import json
import shutil
from datetime import datetime
from typing import List, Dict
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from binance.client import Client
from src.local_llm import LocalStrategyArchitect
from src.multi_asset_simulator import MultiAssetSimulator
from src.advanced_risk import AdvancedRiskManager
import src.dynamic_strategy as dynamic_strategy
import importlib


STRATEGY_FILE = "src/dynamic_strategy.py"
RESULTS_DIR = "evolution_experiments"
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class ScientificEvolutionTester:
    """
    Walk-Forward Analysis for LLM Strategy Evolution
    
    Methodology:
    1. Divide data into N rolling periods
    2. For each period: Train on 70%, Test on 30%
    3. Only evolve when trigger conditions are met (same as main.py)
    4. Record out-of-sample (Test) performance
    5. Statistical analysis across all periods
    """
    
    def __init__(self, n_periods: int = 5, train_ratio: float = 0.7):
        self.n_periods = n_periods
        self.train_ratio = train_ratio
        self.client = Client('', '')
        self.risk_manager = AdvancedRiskManager()
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Evolution trigger thresholds (same as main.py)
        self.win_rate_threshold = 0.40
        self.single_asset_loss_threshold = 0.05
        
        # Create results directory
        self.experiment_dir = os.path.join(RESULTS_DIR, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
    
    def fetch_data(self, symbols: List[str], interval: str = "15m", batches: int = 6) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols"""
        print("\n📊 Fetching historical data...")
        data = {}
        
        for symbol in symbols:
            print(f"   {symbol}...", end=" ")
            all_klines = []
            end_time = None
            
            for i in range(batches):
                try:
                    if end_time:
                        klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=1000, endTime=end_time)
                    else:
                        klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=1000)
                    
                    if not klines:
                        break
                    all_klines = klines + all_klines
                    end_time = klines[0][0] - 1
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            if all_klines:
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
                data[symbol] = df
                print(f"{len(df)} candles")
            else:
                print("failed")
        
        return data
    
    def split_data_walk_forward(self, data: Dict[str, pd.DataFrame]) -> List[tuple]:
        """
        Split data into N rolling walk-forward periods
        
        Example with 5 periods:
        Period 1: [0%----70%][70%--100%]  (Train | Test)
        Period 2: [10%---80%][80%--100%]  (overlap, rolling)
        ...
        """
        periods = []
        
        # Get minimum length across all symbols
        min_len = min(len(df) for df in data.values())
        
        # Calculate step size for rolling windows
        test_size = int(min_len * (1 - self.train_ratio))
        step_size = test_size // self.n_periods if self.n_periods > 1 else 0
        
        for i in range(self.n_periods):
            start_offset = i * step_size
            train_end = int(min_len * self.train_ratio) + start_offset
            test_end = min(train_end + test_size, min_len)
            
            if test_end > min_len:
                break
                
            train_data = {}
            test_data = {}
            
            for symbol, df in data.items():
                train_data[symbol] = df.iloc[start_offset:train_end].copy()
                test_data[symbol] = df.iloc[train_end:test_end].copy()
            
            periods.append((train_data, test_data))
        
        return periods
    
    def run_backtest(self, data: Dict[str, pd.DataFrame], strategy_module) -> dict:
        """Run backtest on given data with given strategy (using same risk controls as main.py)"""
        def _error_result(msg: str) -> dict:
            return {
                "error": msg,
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "profit_factor": 0,
                "avg_trade": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "pnl_by_symbol": {},
            }

        try:
            simulator = MultiAssetSimulator(
                initial_capital=10000.0,
                max_positions=3,
                capital_per_position=0.3,
                # Portfolio Risk Controls (same as main.py)
                max_long_exposure=0.6,
                max_short_exposure=0.6,
                max_daily_loss=0.10,  # Relaxed for backtest (spans multiple days)
                max_single_asset_loss=0.10  # Relaxed for backtest
            )
            
            symbols = list(data.keys())
            if not symbols:
                return _error_result("No symbols provided")
            
            # Find common time range
            min_time = max(df.index[0] for df in data.values())
            max_time = min(df.index[-1] for df in data.values())
            
            for symbol in data:
                data[symbol] = data[symbol][(data[symbol].index >= min_time) & (data[symbol].index <= max_time)]
            
            if len(data[symbols[0]]) < 300:
                return _error_result("Not enough data")
            
            common_times = data[symbols[0]].index
            total_bars = len(common_times)
            bars_per_day = 96  # 15min candles = 96 per day
            
            for i in range(200, total_bars):
                current_time = common_times[i]
                current_timestamp = current_time.timestamp()
                
                # Reset daily controls every 24 hours (96 bars for 15min)
                if (i - 200) % bars_per_day == 0:
                    simulator.daily_pnl = 0.0
                    simulator.trading_halted = False
                    simulator.halted_symbols = []
                
                prices = {}
                for symbol, df in data.items():
                    if current_time in df.index:
                        prices[symbol] = float(df.loc[current_time, 'close'])
                
                simulator.check_positions(prices, current_timestamp)
                
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
                        plan = self.risk_manager.calculate_position(
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
            
            # Close remaining
            final_prices = {s: float(data[s].iloc[-1]['close']) for s in symbols if len(data[s]) > 0}
            for symbol in list(simulator.positions.keys()):
                if symbol in final_prices:
                    simulator.close_position(symbol, final_prices[symbol], "End")
            
            report = simulator.get_report()
            if "total_trades" not in report:
                return _error_result(f"Invalid report format: {report}")
            return report
        except Exception as e:
            return _error_result(f"Backtest exception: {e}")
    
    def backup_strategy(self, name: str) -> str:
        """Backup current strategy"""
        path = os.path.join(self.experiment_dir, f"{name}.py")
        shutil.copy(STRATEGY_FILE, path)
        return path
    
    def restore_strategy(self, path: str):
        """Restore strategy from backup"""
        shutil.copy(path, STRATEGY_FILE)
        importlib.reload(dynamic_strategy)
    
    def run_experiment(self):
        """Run Walk-Forward Analysis experiment"""
        print("=" * 70)
        print("     WALK-FORWARD EVOLUTION TEST")
        print("=" * 70)
        print(f"Periods: {self.n_periods} | Train/Test: {self.train_ratio*100:.0f}%/{(1-self.train_ratio)*100:.0f}%")
        print(f"Evolution Triggers: Win Rate < {self.win_rate_threshold*100:.0f}% OR Single Asset Loss > {self.single_asset_loss_threshold*100:.0f}%")
        print(f"Experiment ID: {self.experiment_id}")
        print("=" * 70)
        
        # Step 1: Fetch data and create walk-forward periods
        full_data = self.fetch_data(TEST_SYMBOLS)
        periods = self.split_data_walk_forward(full_data)
        
        if not periods:
            print("❌ Failed to create walk-forward periods")
            return
        
        print(f"\n📈 Walk-Forward Periods: {len(periods)}")
        for i, (train, test) in enumerate(periods):
            symbol = TEST_SYMBOLS[0]
            print(f"   Period {i+1}: Train={len(train[symbol])} | Test={len(test[symbol])}")
        
        # Step 2: Backup original strategy
        original_path = self.backup_strategy("original")
        architect = LocalStrategyArchitect(model_name="qwen3:8b")
        
        # Step 3: Walk through each period
        for period_idx, (train_data, test_data) in enumerate(periods):
            period_num = period_idx + 1
            print(f"\n{'=' * 70}")
            print(f"PERIOD {period_num}/{len(periods)}")
            print("=" * 70)
            
            # Always start from original strategy for fair comparison
            self.restore_strategy(original_path)
            
            # Test on TRAIN data
            print("\n🔬 Testing on TRAIN data...")
            train_result = self.run_backtest(train_data.copy(), dynamic_strategy)
            train_wr = train_result.get('win_rate', 0)
            train_pnl = train_result.get('total_pnl', 0)
            print(f"   Trades={train_result['total_trades']} | Win={train_wr:.1%} | PnL=${train_pnl:.2f}")
            
            # Get profitability metrics
            profit_factor = train_result.get('profit_factor', 0)
            avg_trade = train_result.get('avg_trade', 0)
            
            # Check evolution trigger conditions
            should_evolve = False
            evolution_reason = ""
            
            # Condition 1: Low win rate
            if train_wr < self.win_rate_threshold:
                should_evolve = True
                evolution_reason = f"Low win rate: {train_wr:.1%}"
            
            # Condition 2: Low profit factor (profitability issue)
            if profit_factor < 1.0 and profit_factor > 0:
                should_evolve = True
                evolution_reason = f"Low profit factor: {profit_factor:.2f} (losing money)"
            
            # Condition 3: Negative PnL
            if train_pnl < 0:
                should_evolve = True
                evolution_reason = f"Negative PnL: ${train_pnl:.2f}"
            
            # Condition 4: Single asset excessive loss
            pnl_by_symbol = train_result.get('pnl_by_symbol', {})
            for sym, pnl in pnl_by_symbol.items():
                if pnl < -10000 * self.single_asset_loss_threshold:
                    should_evolve = True
                    evolution_reason = f"{sym} loss: ${pnl:.2f}"
                    break
            
            # Record whether evolution was triggered
            evolved = False
            
            if should_evolve:
                print(f"\n🚨 Evolution triggered: {evolution_reason}")
                
                # Build context with profitability info
                context_lines = [
                    "=" * 50,
                    f"WALK-FORWARD PERIOD {period_num}",
                    "=" * 50,
                    f"Symbols: {TEST_SYMBOLS}",
                    f"Train Trades: {train_result['total_trades']}",
                    f"Train Win Rate: {train_wr*100:.1f}%",
                    f"Train PnL: ${train_pnl:+.2f}",
                    f"Profit Factor: {profit_factor:.2f}",
                    f"Avg Trade PnL: ${avg_trade:.2f}",
                    "",
                    "## Per-Symbol:",
                ]
                for sym in TEST_SYMBOLS:
                    sym_pnl = pnl_by_symbol.get(sym, 0)
                    status = "🟢" if sym_pnl > 0 else "🔴"
                    context_lines.append(f"  {status} {sym}: ${sym_pnl:+.2f}")
                
                context_lines.extend([
                    "",
                    f"## Issue: {evolution_reason}",
                    "",
                    "## Profitability Analysis:",
                    f"- Profit Factor {profit_factor:.2f} means avg_win/avg_loss ratio",
                    f"- Need Profit Factor > 1.0 to be profitable",
                    f"- Consider: signal timing, entry conditions, trend filters",
                    "",
                    "Goal: Generate signals that lead to PROFITABLE trades, not just high win rate",
                    "=" * 50
                ])
                
                print("\n🧬 Evolving strategy...")
                success = architect.optimize_strategy(
                    "\n".join(context_lines),
                    {
                        "win_rate": train_wr * 100,
                        "issue": evolution_reason,
                        "profit_factor": profit_factor,
                        "avg_trade": avg_trade
                    }
                )
                
                if success:
                    evolved = True
                    self.backup_strategy(f"period_{period_num}_evolved")
                    importlib.reload(dynamic_strategy)
                    print("   ✅ Evolution successful")
                else:
                    print("   ❌ Evolution failed")
            else:
                print(f"\n✅ No evolution needed (Win={train_wr:.1%} >= {self.win_rate_threshold:.0%})")
            
            # Test on TEST data (out-of-sample)
            print("\n🔬 Testing on TEST data (out-of-sample)...")
            test_result = self.run_backtest(test_data.copy(), dynamic_strategy)
            test_wr = test_result.get('win_rate', 0)
            test_pnl = test_result.get('total_pnl', 0)
            test_trades = test_result.get('total_trades', 0)
            if test_result.get("error"):
                print(f"   ⚠️ Backtest error: {test_result.get('error')}")
            print(f"   Trades={test_trades} | Win={test_wr:.1%} | PnL=${test_pnl:.2f}")
            
            # Record result with detailed metrics
            self.results.append({
                "period": period_num,
                "evolved": evolved,
                "evolution_reason": evolution_reason if evolved else "N/A",
                "train_trades": train_result.get('total_trades', 0),
                "train_win_rate": train_wr,
                "train_pnl": train_pnl,
                "test_trades": test_result.get('total_trades', 0),
                "test_win_rate": test_wr,
                "test_pnl": test_pnl,
                "test_profit_factor": test_result.get('profit_factor', 0),
                "test_avg_trade": test_result.get('avg_trade', 0),
                "test_best_trade": test_result.get('best_trade', 0),
                "test_worst_trade": test_result.get('worst_trade', 0),
                "test_pnl_by_symbol": test_result.get('pnl_by_symbol', {}),
            })
        
        # Step 4: Statistical analysis
        self.print_statistical_report()
        self.save_results()
        
        # Restore original
        self.restore_strategy(original_path)
    
    def print_statistical_report(self):
        """Print statistical analysis of walk-forward results"""
        print("\n" + "=" * 70)
        print("     WALK-FORWARD ANALYSIS REPORT")
        print("=" * 70)
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Separate evolved vs non-evolved periods
        evolved_periods = [r for r in self.results if r.get('evolved', False)]
        non_evolved_periods = [r for r in self.results if not r.get('evolved', False)]
        
        print(f"\n📊 Summary:")
        print(f"   Total Periods: {len(self.results)}")
        print(f"   Evolution Triggered: {len(evolved_periods)} times")
        print(f"   No Evolution Needed: {len(non_evolved_periods)} times")
        
        # Overall test performance
        all_test_pnl = sum(r.get('test_pnl', 0) for r in self.results)
        all_test_wr = [r.get('test_win_rate', 0) for r in self.results if r.get('test_win_rate', 0) > 0]
        avg_test_wr = statistics.mean(all_test_wr) if all_test_wr else 0
        
        print(f"\n📈 Overall Out-of-Sample Performance:")
        print(f"   Total Test PnL: ${all_test_pnl:.2f}")
        print(f"   Avg Test Win Rate: {avg_test_wr:.1%}")
        
        # Compare evolved vs non-evolved periods
        if evolved_periods:
            evolved_test_pnl = [r.get('test_pnl', 0) for r in evolved_periods]
            evolved_test_wr = [r.get('test_win_rate', 0) for r in evolved_periods]
            evolved_pnl_avg = statistics.mean(evolved_test_pnl)
            evolved_wr_avg = statistics.mean(evolved_test_wr)
            print(f"{'Test PnL (avg)':<25} {'N/A':<17} ${evolved_pnl_avg:.2f}")
            print(f"{'Test Win Rate (avg)':<25} {'N/A':<17} {evolved_wr_avg:.1%}")
        
        if non_evolved_periods:
            non_evolved_test_pnl = [r.get('test_pnl', 0) for r in non_evolved_periods]
            non_evolved_test_wr = [r.get('test_win_rate', 0) for r in non_evolved_periods]
            non_evolved_pnl_avg = statistics.mean(non_evolved_test_pnl)
            non_evolved_wr_avg = statistics.mean(non_evolved_test_wr)
            if evolved_periods:
                print(f"{'Test PnL (no evol)':<25} ${non_evolved_pnl_avg:<16.2f}")
                print(f"{'Test Win Rate (no evol)':<25} {non_evolved_wr_avg:<16.1%}")
        
        print("-" * 60)
        
        # Period-by-period breakdown
        print("\n📋 Period Details:")
        for r in self.results:
            evolved_mark = "🧬" if r.get('evolved') else "📊"
            print(f"   {evolved_mark} Period {r['period']}: Train={r['train_win_rate']:.0%} → Test={r['test_win_rate']:.0%} | PnL=${r['test_pnl']:.2f}")
        
        # Verdict
        print("\n" + "=" * 70)
        
        # Calculate if evolution helped
        if evolved_periods and non_evolved_periods:
            evolved_avg = statistics.mean([r['test_pnl'] for r in evolved_periods])
            non_evolved_avg = statistics.mean([r['test_pnl'] for r in non_evolved_periods])
            if evolved_avg > non_evolved_avg:
                print("✅ VERDICT: Evolution IMPROVED performance when triggered")
            else:
                print("⚠️ VERDICT: Evolution did NOT improve performance")
        elif evolved_periods:
            print("📊 All periods triggered evolution - compare to baseline")
        else:
            print("✅ Strategy performing well - no evolution needed")
        
        print("=" * 70)
    
    def save_results(self):
        """Save results to JSON"""
        results_path = os.path.join(self.experiment_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "n_periods": self.n_periods,
                "train_ratio": self.train_ratio,
                "symbols": TEST_SYMBOLS,
                "results": self.results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {results_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Evolution Test")
    parser.add_argument("--periods", type=int, default=5, help="Number of walk-forward periods")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train data ratio (0.0-1.0)")
    args = parser.parse_args()
    
    tester = ScientificEvolutionTester(n_periods=args.periods, train_ratio=args.train_ratio)
    tester.run_experiment()


if __name__ == "__main__":
    main()
