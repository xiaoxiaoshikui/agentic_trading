"""
Data Loader with Caching
========================

Handles data loading, caching, and walk-forward splitting.
Ensures reproducibility by caching downloaded data.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import requests

try:
    from binance.client import Client
except Exception:  # pragma: no cover - optional for cached-only workflows
    Client = None

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader with caching for reproducible experiments.

    Features:
    - Downloads data from Binance Futures
    - Caches data locally for reproducibility
    - Provides walk-forward data splits
    """

    def __init__(self, cache_dir: str = "experiments/data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.client = Client("", "") if Client is not None else None
        self.session = requests.Session()

    def get_cache_path(self, symbol: str, interval: str, n_bars: int, end_time_ms: Optional[int] = None) -> str:
        """Generate cache file path"""
        cache_key = f"{symbol}_{interval}_{n_bars}"
        if end_time_ms:
            end_key = pd.to_datetime(end_time_ms, unit="ms", utc=True).strftime("%Y%m%d%H%M")
            cache_key = f"{cache_key}_end{end_key}"
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")

    def load_data(
        self,
        symbols: List[str],
        interval: str = "15m",
        n_bars: int = 6000,  # ~62 days of 15m data
        force_download: bool = False,
        end_time: Optional[Any] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for all symbols.

        Args:
            symbols: List of trading pairs
            interval: Kline interval (e.g., "15m", "1h")
            n_bars: Number of bars to fetch
            force_download: If True, re-download even if cached

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}

        end_time_ms = self._normalize_end_time(end_time)
        if end_time_ms is None:
            logger.warning("end_time not set; results may drift over time. Set end_time for reproducibility.")

        for symbol in symbols:
            cache_path = self.get_cache_path(symbol, interval, n_bars, end_time_ms=end_time_ms)

            if os.path.exists(cache_path) and not force_download:
                logger.info(f"Loading cached data for {symbol}")
                df = pd.read_parquet(cache_path)
            else:
                logger.info(f"Downloading data for {symbol}...")
                df = self._download_data(symbol, interval, n_bars, end_time_ms=end_time_ms)
                if df is not None and len(df) > 0:
                    df.to_parquet(cache_path)
                    logger.info(f"Cached {len(df)} bars for {symbol}")

            if df is not None and len(df) > 0:
                data[symbol] = df

        return data

    def _download_data(
        self,
        symbol: str,
        interval: str,
        n_bars: int,
        end_time_ms: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Download historical data from Binance"""
        all_klines = []
        end_time = end_time_ms
        bars_per_request = 1000
        n_requests = (n_bars // bars_per_request) + 1

        for i in range(n_requests):
            try:
                if self.client is not None:
                    if end_time:
                        klines = self.client.futures_klines(
                            symbol=symbol,
                            interval=interval,
                            limit=bars_per_request,
                            endTime=end_time
                        )
                    else:
                        klines = self.client.futures_klines(
                            symbol=symbol,
                            interval=interval,
                            limit=bars_per_request
                        )
                else:
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": bars_per_request,
                    }
                    if end_time:
                        params["endTime"] = int(end_time)
                    response = self.session.get(
                        "https://fapi.binance.com/fapi/v1/klines",
                        params=params,
                        timeout=30,
                    )
                    response.raise_for_status()
                    klines = response.json()

                if not klines:
                    break

                all_klines = klines + all_klines
                end_time = klines[0][0] - 1

                if len(all_klines) >= n_bars:
                    break

            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                break

        if not all_klines:
            return None

        # Convert to DataFrame
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'n_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(all_klines, columns=cols)

        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.drop_duplicates().sort_index()

        # Add placeholder columns for advanced features
        df['funding_rate'] = 0.0
        df['open_interest'] = 0.0

        # Keep only needed columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'funding_rate', 'open_interest']]

        return df.iloc[-n_bars:] if len(df) > n_bars else df

    @staticmethod
    def _normalize_end_time(end_time: Optional[Any]) -> Optional[int]:
        if end_time is None:
            return None
        if isinstance(end_time, (int, float)):
            return int(end_time)
        try:
            ts = pd.to_datetime(end_time, utc=True)
            return int(ts.timestamp() * 1000)
        except Exception:
            return None

    def create_walk_forward_splits(
        self,
        data: Dict[str, pd.DataFrame],
        n_periods: int,
        train_ratio: float,
        min_train_bars: int = 500,
        min_test_bars: int = 200
    ) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        """
        Create walk-forward train/test splits.

        Walk-forward analysis:
        - Divides data into N rolling periods
        - Each period: expanding train window + fixed test window
        - Periods advance by one test window each time (true walk-forward)

        Args:
            data: Dictionary of symbol -> DataFrame
            n_periods: Number of walk-forward periods
            train_ratio: Fraction of each window for training
            min_train_bars: Minimum bars required for training
            min_test_bars: Minimum bars required for testing

        Returns:
            List of (train_data, test_data) tuples
        """
        if not data:
            return []

        # Find minimum length across symbols
        min_len = min(len(df) for df in data.values())
        logger.info(f"Minimum data length across symbols: {min_len} bars")

        if n_periods <= 0:
            logger.warning("n_periods must be positive")
            return []

        if min_len < min_train_bars + min_test_bars:
            logger.warning(
                f"Insufficient data: min_len={min_len}. "
                f"Required: train>={min_train_bars}, test>={min_test_bars}"
            )
            return []

        # Use full dataset: compute test window from ratio, then assign remainder to training
        test_size = int(min_len * (1 - train_ratio) / n_periods) if n_periods > 0 else 0
        if test_size < min_test_bars:
            test_size = min_test_bars
        train_size = min_len - (n_periods * test_size)

        if train_size < min_train_bars or test_size < min_test_bars:
            logger.warning(
                f"Insufficient data after sizing: train={train_size}, test={test_size}. "
                f"Required: train>={min_train_bars}, test>={min_test_bars}"
            )
            return []

        max_periods = 1 + (min_len - train_size - test_size) // test_size
        periods_to_run = min(n_periods, max_periods)
        if periods_to_run < n_periods:
            logger.info(f"Reducing periods from {n_periods} to {periods_to_run} due to data limits")

        periods = []

        for i in range(periods_to_run):
            train_end = train_size + i * test_size
            test_end = train_end + test_size

            if test_end > min_len:
                logger.info(f"Stopping at period {i+1}/{periods_to_run} - insufficient data")
                break

            train_data = {}
            test_data = {}

            for symbol, df in data.items():
                train_data[symbol] = df.iloc[:train_end].copy()
                test_data[symbol] = df.iloc[train_end:test_end].copy()

            periods.append((train_data, test_data))

            logger.debug(
                f"Period {i+1}: train[0:{train_end}] "
                f"test[{train_end}:{test_end}]"
            )

        logger.info(f"Created {len(periods)} walk-forward periods")
        return periods

    def get_data_info(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary information about loaded data"""
        info = {
            "symbols": list(data.keys()),
            "n_symbols": len(data),
        }

        for symbol, df in data.items():
            info[symbol] = {
                "n_bars": len(df),
                "start_date": str(df.index[0]),
                "end_date": str(df.index[-1]),
                "columns": list(df.columns),
            }

        return info


def save_data_snapshot(
    data: Dict[str, pd.DataFrame],
    path: str,
    metadata: Optional[Dict] = None
):
    """
    Save a snapshot of data for reproducibility.

    Useful for paper: save exact data used in experiments.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "data": {}
    }

    for symbol, df in data.items():
        snapshot["data"][symbol] = {
            "n_bars": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "checksum": hashlib.md5(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()
        }

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Also save parquet files
    base_dir = os.path.dirname(path)
    for symbol, df in data.items():
        df.to_parquet(os.path.join(base_dir, f"{symbol}.parquet"))
