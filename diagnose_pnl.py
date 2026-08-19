"""Diagnose why all PnL are negative."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Load results
rdir = Path("experiments/results_tom/tom_minimal_20260209_085645")
with open(rdir / "results.json") as f:
    results = json.load(f)

print("=" * 60)
print("1. PER-PERIOD REGIME ANALYSIS")
print("=" * 60)

e1 = results["experiments"]["E1"]
for agent_name in ["Technical", "ToM-Full", "LLM"]:
    raw = e1["records"][agent_name]
    print(f"\n--- {agent_name} ---")
    for p in sorted(set(r["period"] for r in raw)):
        p_data = [r for r in raw if r["period"] == p]
        sr = np.mean([r["sharpe"] for r in p_data])
        pnl = np.mean([r["total_pnl"] for r in p_data])
        wr = np.mean([r["win_rate"] for r in p_data])
        trades = np.mean([r["total_trades"] for r in p_data])
        print(f"  P{p}: SR={sr:+7.2f}  PnL=${pnl:+8.2f}  WR={wr:.1%}  Trades={trades:.0f}")
    total_pnl = np.mean([r["total_pnl"] for r in raw])
    print(f"  AVG: PnL=${total_pnl:+8.2f}")

# 2. Load and analyze the actual price data
print("\n" + "=" * 60)
print("2. PRICE DATA REGIMES")
print("=" * 60)

# Find the data file
data_files = list(rdir.glob("*.parquet"))
if data_files:
    df = pd.read_parquet(data_files[0])
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Determine price column
    price_col = None
    for col in ["close", "Close", "price"]:
        if col in df.columns:
            price_col = col
            break
    
    if price_col:
        prices = df[price_col].values
        n = len(prices)
        print(f"\nTotal bars: {n}")
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"First price: ${prices[0]:.2f}")
        print(f"Last price: ${prices[-1]:.2f}")
        print(f"Overall return: {(prices[-1]/prices[0]-1)*100:.1f}%")
        
        # Walk-forward periods (config: 5 periods, warmup=200)
        n_periods = 5
        warmup = 200
        period_len = n // n_periods  # 10000/5 = 2000 bars per period
        test_bars = period_len - warmup  # 1800 test bars? or different split
        
        # Actually check the config
        cfg = results["config"]
        print(f"\nConfig: n_bars={cfg['n_bars']}, n_periods={cfg['n_periods']}, warmup={cfg['warmup_bars']}")
        
        # Each period gets n_bars/n_periods = 2000 bars
        # warmup = 200 bars, so test = 1800 bars? 
        # But from results, trades~86 per period => decision every ~21 bars?
        # Let's check actual period boundaries
        for p in range(n_periods):
            start = p * period_len
            end = (p + 1) * period_len
            test_start = start + warmup
            p_prices = prices[test_start:end]
            warmup_prices = prices[start:test_start]
            if len(p_prices) > 0:
                ret = (p_prices[-1] / p_prices[0] - 1) * 100
                vol = np.std(np.diff(p_prices) / p_prices[:-1]) * 100
                print(f"  Period {p+1}: bars {test_start}-{end}, "
                      f"price ${p_prices[0]:.0f}->${p_prices[-1]:.0f}, "
                      f"return={ret:+.1f}%, vol={vol:.2f}%")
    else:
        print("No price column found!")
else:
    print("No parquet files found, checking snapshots...")
    snap_files = list(rdir.glob("*snapshot*"))
    print(f"Snapshots: {snap_files}")

# 3. Check trade direction distribution
print("\n" + "=" * 60)
print("3. TRADE DIRECTION ANALYSIS")
print("=" * 60)

for agent_name in ["Technical", "ToM-Full"]:
    raw = e1["records"][agent_name]
    for p in [1, 4]:  # one bear (1) and one bull (4) period
        p_data = [r for r in raw if r["period"] == p]
        if p_data:
            # Check if we have trade details
            sample = p_data[0]
            keys = list(sample.keys())
            print(f"\n{agent_name} P{p} keys: {keys}")
            # Check for long/short breakdown
            for k in ["n_longs", "n_shorts", "long_pnl", "short_pnl", "trades"]:
                if k in sample:
                    print(f"  {k}: {sample[k]}")
            break

# 4. Look at harness to understand what constitutes a "trade"
print("\n" + "=" * 60)
print("4. HARNESS EVALUATION LOGIC CHECK")
print("=" * 60)

# Check the evaluation harness for potential issues
import inspect
import importlib.util
spec = importlib.util.spec_from_file_location("harness", "src/tom/evaluation/harness.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    harness = mod.EvaluationHarness
    print(f"EPSILON={harness.EPSILON}, CONF_JITTER={harness.CONF_JITTER}, SLIPPAGE={harness.SLIPPAGE}")
except Exception as e:
    print(f"Could not load harness: {e}")

# 5. Check if the strategy is mostly going LONG (bias issue)
print("\n" + "=" * 60)
print("5. WIN RATE BY PERIOD (regime correlation)")
print("=" * 60)
for agent_name in ["Technical", "ToM-Full"]:
    print(f"\n{agent_name}:")
    raw = e1["records"][agent_name]
    for p in sorted(set(r["period"] for r in raw)):
        p_data = [r for r in raw if r["period"] == p]
        wr = np.mean([r["win_rate"] for r in p_data])
        pf = np.mean([r["profit_factor"] for r in p_data])
        sr = np.mean([r["sharpe"] for r in p_data])
        print(f"  P{p}: WR={wr:.1%}, PF={pf:.2f}, SR={sr:+.2f}")
