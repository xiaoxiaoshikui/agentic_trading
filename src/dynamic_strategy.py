import pandas as pd
import numpy as np
import ta
from dataclasses import dataclass
from typing import List

@dataclass
class Signal:
    side: str
    confidence: float
    reason: str


# Required columns for signal calculation
REQUIRED_COLUMNS: List[str] = ['open', 'high', 'low', 'close', 'volume']
MIN_ROWS_FOR_INDICATORS: int = 201  # Need at least 200 for EMA-200


def validate_dataframe(df: pd.DataFrame) -> tuple:
    """
    Validate DataFrame before processing.

    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"

    if df.empty:
        return False, "DataFrame is empty"

    if len(df) < MIN_ROWS_FOR_INDICATORS:
        return False, f"Insufficient data: {len(df)} rows (need {MIN_ROWS_FOR_INDICATORS})"

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check for NaN in critical columns
    if df['close'].isna().all():
        return False, "All close prices are NaN"

    return True, ""


def calculate_signal(df: pd.DataFrame) -> Signal:
    try:
        # Validate DataFrame first
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            return Signal(side="FLAT", confidence=0.0, reason=error_msg)

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Add optional columns with defaults
        if 'funding_rate' not in df.columns:
            df['funding_rate'] = 0.0
        if 'open_interest' not in df.columns:
            df['open_interest'] = 0.0

        # Calculate key indicators
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bollinger_upper'] = bb.bollinger_hband()
        df['bollinger_lower'] = bb.bollinger_lband()
        df['volume_rolling_mean'] = df['volume'].rolling(window=20).mean()
        df['open_interest_rolling_mean'] = df['open_interest'].rolling(window=20).mean()

        # Get latest values with NaN checks
        close = float(df['close'].iloc[-1])
        ema50 = float(df['ema_50'].iloc[-1])
        ema200 = float(df['ema_200'].iloc[-1])
        rsi = float(df['rsi'].iloc[-1])
        bollinger_upper = float(df['bollinger_upper'].iloc[-1])
        bollinger_lower = float(df['bollinger_lower'].iloc[-1])
        volume = float(df['volume'].iloc[-1])
        volume_rolling_mean = float(df['volume_rolling_mean'].iloc[-1])

        # Check for NaN values in critical indicators
        if not np.isfinite(close) or not np.isfinite(ema50) or not np.isfinite(ema200):
            return Signal(side="FLAT", confidence=0.0, reason="NaN in critical indicators")

        if not np.isfinite(rsi):
            rsi = 50.0  # Default neutral RSI

        if not np.isfinite(volume_rolling_mean) or volume_rolling_mean == 0:
            volume_rolling_mean = volume + 1e-9  # Prevent division by zero

        open_interest = float(df['open_interest'].iloc[-1])
        open_interest_rolling_mean = float(df['open_interest_rolling_mean'].iloc[-1])
        funding_rate = float(df['funding_rate'].iloc[-1])

        # Handle NaN in optional indicators
        if not np.isfinite(open_interest):
            open_interest = 0.0
        if not np.isfinite(open_interest_rolling_mean) or open_interest_rolling_mean == 0:
            open_interest_rolling_mean = open_interest + 1e-9
        if not np.isfinite(funding_rate):
            funding_rate = 0.0

        # Initialize variables
        side = "FLAT"
        confidence = 0.0
        reasons = []

        # Bullish trend detection
        if close > ema50 and close > ema200:
            reasons.append("Price above EMA 50 and EMA 200")
            confidence += 0.4

            if rsi > 50 and rsi < 70:
                reasons.append(f"RSI in healthy range ({rsi:.1f})")
                confidence += 0.3

            if volume > volume_rolling_mean * 2:
                reasons.append("High volume")
                confidence += 0.2

            if funding_rate > 0 and open_interest > open_interest_rolling_mean:
                reasons.append("Positive funding rate and rising open interest")
                confidence += 0.1

            if confidence >= 0.5:
                side = "LONG"

        # Bearish trend detection
        elif close < ema50 and close < ema200:
            reasons.append("Price below EMA 50 and EMA 200")
            confidence += 0.4

            if rsi < 50 and rsi > 30:
                reasons.append(f"RSI in healthy range ({rsi:.1f})")
                confidence += 0.3

            if volume > volume_rolling_mean * 2:
                reasons.append("High volume")
                confidence += 0.2

            if funding_rate < 0 and open_interest > open_interest_rolling_mean:
                reasons.append("Negative funding rate and rising open interest")
                confidence += 0.1

            if confidence >= 0.5:
                side = "SHORT"

        # Mean reversion (optional)
        else:
            if close > bollinger_upper and rsi > 70:
                reasons.append("Overbought in Bollinger Bands")
                confidence += 0.3

                if volume > volume_rolling_mean * 2:
                    reasons.append("High volume")
                    confidence += 0.2

                if funding_rate > 0:
                    reasons.append("Positive funding rate")
                    confidence += 0.1

                if confidence >= 0.5:
                    side = "SHORT"

            elif close < bollinger_lower and rsi < 30:
                reasons.append("Oversold in Bollinger Bands")
                confidence += 0.3

                if volume > volume_rolling_mean * 2:
                    reasons.append("High volume")
                    confidence += 0.2

                if funding_rate < 0:
                    reasons.append("Negative funding rate")
                    confidence += 0.1

                if confidence >= 0.5:
                    side = "LONG"

        reason = "; ".join(reasons) if reasons else "No clear signal"
        return Signal(side=side, confidence=min(confidence, 1.0), reason=reason)

    except (KeyError, IndexError) as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Data access error: {e}")
    except (ValueError, TypeError) as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Calculation error: {e}")
    except Exception as e:
        return Signal(side="FLAT", confidence=0.0, reason=f"Unexpected error: {type(e).__name__}: {e}")