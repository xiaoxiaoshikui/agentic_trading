"""
Market Analyst Agent
====================
Responsibility: Real-time market state analysis, output structured market report
"""

import pandas as pd
import ta
from dataclasses import dataclass, field
from typing import List, Dict
from .base import AgentBase
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Market State Report"""
    # Market regime
    regime: str = "unknown"  # trending, ranging, volatile, crisis
    direction: str = "neutral"  # bullish, bearish, neutral
    strength: float = 0.0  # 0.0-1.0 trend strength
    
    # Key price levels
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Anomaly detection
    anomalies: List[str] = field(default_factory=list)
    
    # Raw indicators
    indicators: Dict[str, float] = field(default_factory=dict)
    
    # Composite score (-1 to +1, negative=bearish, positive=bullish)
    score: float = 0.0
    
    # Recommendation
    recommendation: str = ""


class MarketAnalyst(AgentBase):
    """
    Market Analyst Agent
    
    Input: Kline data, funding rate, open interest
    Output: MarketState (market state judgment)
    """
    
    def __init__(self):
        super().__init__("MarketAnalyst")
        
        # Configuration parameters
        self.config = {
            "ema_short": 10,
            "ema_long": 20,
            "rsi_period": 14,
            "atr_period": 14,
            "volatility_threshold": 1.5,  # ATR > 1.5x average = high volatility
            "funding_extreme": 0.0005,    # Funding rate extreme threshold (0.05%)
            "oi_divergence_threshold": 0.05,  # OI divergence threshold (5%)
        }
    
    def process(
        self, 
        df: pd.DataFrame, 
        funding_rate: float = 0.0, 
        open_interest: float = 0.0,
        oi_change_24h: float = 0.0
    ) -> MarketState:
        """
        Analyze market state
        
        Args:
            df: Kline data (must contain close, high, low, volume)
            funding_rate: Current funding rate
            open_interest: Current open interest
            oi_change_24h: 24h OI change rate
        
        Returns:
            MarketState: Market state report
        """
        state = MarketState()
        
        try:
            # 1. Calculate technical indicators
            indicators = self._calculate_indicators(df)
            state.indicators = indicators
            
            # 2. Detect market regime
            state.regime = self._detect_regime(df, indicators)
            
            # 3. Detect direction
            state.direction, state.strength = self._detect_direction(indicators)
            
            # 4. Detect anomaly signals
            state.anomalies = self._detect_anomalies(
                indicators, funding_rate, oi_change_24h, df
            )
            
            # 5. Find key price levels
            state.support_levels, state.resistance_levels = self._find_key_levels(df)
            
            # 6. Calculate composite score
            state.score = self._calculate_score(state, funding_rate, oi_change_24h)
            
            # 7. Generate recommendation
            state.recommendation = self._generate_recommendation(state)
            
            self.log_info(f"Market: {state.regime} | Direction: {state.direction} | Score: {state.score:.2f}")
            
        except Exception as e:
            self.log_error(f"Analysis failed: {e}")
            state.recommendation = f"Analysis error: {str(e)}"
        
        return state
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # EMA
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=self.config['ema_short'])
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=self.config['ema_long'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.config['rsi_period'])
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.config['atr_period']
        )
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'])
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Get latest values
        latest = df.iloc[-1]
        indicators['close'] = latest['close']
        indicators['ema_short'] = latest['ema_short']
        indicators['ema_long'] = latest['ema_long']
        indicators['rsi'] = latest['rsi']
        indicators['atr'] = latest['atr']
        indicators['atr_avg'] = df['atr'].rolling(20).mean().iloc[-1]
        indicators['bb_upper'] = latest['bb_upper']
        indicators['bb_lower'] = latest['bb_lower']
        indicators['bb_mid'] = latest['bb_mid']
        indicators['macd'] = latest['macd']
        indicators['macd_signal'] = latest['macd_signal']
        indicators['macd_hist'] = latest['macd_hist']
        indicators['volume'] = latest['volume']
        indicators['volume_avg'] = df['volume'].rolling(20).mean().iloc[-1]
        
        # Price change
        indicators['price_change_1h'] = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1) if len(df) > 12 else 0
        indicators['price_change_24h'] = (df['close'].iloc[-1] / df['close'].iloc[-96] - 1) if len(df) > 96 else 0
        
        return indicators
    
    def _detect_regime(self, df: pd.DataFrame, indicators: Dict) -> str:
        """
        Detect market regime
        - trending: Trend market (momentum strategies work)
        - ranging: Range-bound market (mean reversion works)
        - volatile: High volatility (trade cautiously)
        - crisis: Crisis mode (stop trading)
        """
        atr_ratio = indicators['atr'] / indicators['atr_avg'] if indicators['atr_avg'] > 0 else 1
        
        # Crisis detection: volatility > 2.5x average
        if atr_ratio > 2.5:
            return "crisis"
        
        # High volatility detection
        if atr_ratio > self.config['volatility_threshold']:
            return "volatile"
        
        # Trending vs Ranging using EMA slope
        ema_slope = (indicators['ema_short'] - indicators['ema_long']) / indicators['close']
        
        if abs(ema_slope) > 0.005:  # EMA slope > 0.5%
            return "trending"
        else:
            return "ranging"
    
    def _detect_direction(self, indicators: Dict) -> tuple:
        """Detect market direction and strength"""
        score = 0
        
        # EMA direction
        if indicators['close'] > indicators['ema_short'] > indicators['ema_long']:
            score += 2
        elif indicators['close'] < indicators['ema_short'] < indicators['ema_long']:
            score -= 2
        
        # RSI direction
        rsi = indicators['rsi']
        if rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1
        
        # MACD direction
        if indicators['macd_hist'] > 0:
            score += 1
        else:
            score -= 1
        
        # Determine direction
        if score >= 2:
            direction = "bullish"
        elif score <= -2:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Strength (0-1)
        strength = min(abs(score) / 4, 1.0)
        
        return direction, strength
    
    def _detect_anomalies(
        self, 
        indicators: Dict, 
        funding_rate: float, 
        oi_change: float,
        df: pd.DataFrame
    ) -> List[str]:
        """Detect market anomalies"""
        anomalies = []
        
        # 1. Extreme funding rate
        if funding_rate > self.config['funding_extreme']:
            anomalies.append(f"High funding rate ({funding_rate*100:.3f}%) - Longs crowded")
        elif funding_rate < -self.config['funding_extreme']:
            anomalies.append(f"Low funding rate ({funding_rate*100:.3f}%) - Shorts crowded")
        
        # 2. OI Divergence
        price_change = indicators.get('price_change_24h', 0)
        if abs(oi_change) > self.config['oi_divergence_threshold']:
            if oi_change > 0 and abs(price_change) < 0.01:
                anomalies.append(f"OI Divergence: OI+{oi_change*100:.1f}% price flat - Big move coming")
            elif oi_change < 0 and abs(price_change) < 0.01:
                anomalies.append(f"OI Divergence: OI-{abs(oi_change)*100:.1f}% price flat - Money exiting")
        
        # 3. Extreme RSI
        rsi = indicators['rsi']
        if rsi > 80:
            anomalies.append(f"RSI extremely overbought ({rsi:.0f})")
        elif rsi < 20:
            anomalies.append(f"RSI extremely oversold ({rsi:.0f})")
        
        # 4. Volume anomaly
        vol_ratio = indicators['volume'] / indicators['volume_avg'] if indicators['volume_avg'] > 0 else 1
        if vol_ratio > 3:
            anomalies.append(f"Abnormal volume spike ({vol_ratio:.1f}x)")
        
        # 5. Bollinger Band breakout
        close = indicators['close']
        if close > indicators['bb_upper']:
            anomalies.append("Price broke above upper Bollinger Band")
        elif close < indicators['bb_lower']:
            anomalies.append("Price broke below lower Bollinger Band")
        
        return anomalies
    
    def _find_key_levels(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Find support and resistance levels"""
        recent = df.tail(lookback)
        
        # Simplified: use recent highs and lows
        highs = recent['high'].nlargest(3).tolist()
        lows = recent['low'].nsmallest(3).tolist()
        
        return lows, highs  # support, resistance
    
    def _calculate_score(
        self, 
        state: MarketState, 
        funding_rate: float, 
        oi_change: float
    ) -> float:
        """
        Calculate composite score (-1 to +1)
        Negative = bearish, Positive = bullish
        """
        score = 0.0
        
        # 1. Technical direction (weight 40%)
        if state.direction == "bullish":
            score += 0.4 * state.strength
        elif state.direction == "bearish":
            score -= 0.4 * state.strength
        
        # 2. Funding rate reversal signal (weight 30%)
        # High funding = longs crowded = bearish
        if abs(funding_rate) > self.config['funding_extreme']:
            if funding_rate > 0:
                score -= 0.3  # Longs crowded, go short
            else:
                score += 0.3  # Shorts crowded, go long
        
        # 3. RSI mean reversion (weight 20%)
        rsi = state.indicators.get('rsi', 50)
        if rsi < 30:
            score += 0.2 * (30 - rsi) / 30  # Oversold, bullish
        elif rsi > 70:
            score -= 0.2 * (rsi - 70) / 30  # Overbought, bearish
        
        # 4. OI Divergence (weight 10%)
        price_change = state.indicators.get('price_change_24h', 0)
        if abs(oi_change) > 0.05 and abs(price_change) < 0.01:
            if oi_change > 0:
                score += 0.1  # Whales accumulating
            else:
                score -= 0.1  # Whales distributing
        
        return max(-1.0, min(1.0, score))
    
    def _generate_recommendation(self, state: MarketState) -> str:
        """Generate trading recommendation"""
        if state.regime == "crisis":
            return "WARNING: Crisis mode - Stop trading, wait for stability"
        
        if state.regime == "volatile":
            return "HIGH VOLATILITY: Reduce position size, strict stop loss"
        
        if state.score > 0.5:
            return f"STRONG LONG (score: {state.score:.2f}) - Recommend long"
        elif state.score > 0.2:
            return f"LEAN LONG (score: {state.score:.2f}) - Consider long"
        elif state.score < -0.5:
            return f"STRONG SHORT (score: {state.score:.2f}) - Recommend short"
        elif state.score < -0.2:
            return f"LEAN SHORT (score: {state.score:.2f}) - Consider short"
        else:
            return f"NEUTRAL (score: {state.score:.2f}) - Stay flat"
    
    def generate_llm_context(self, state: MarketState) -> str:
        """
        Generate rich context string for LLM strategy evolution.
        This context helps the LLM understand current market conditions
        and generate better alpha strategies.
        """
        lines = [
            "=" * 50,
            "MARKET ANALYSIS CONTEXT FOR STRATEGY EVOLUTION",
            "=" * 50,
            "",
            f"## Market Regime: {state.regime.upper()}",
            f"- Direction: {state.direction} (strength: {state.strength:.2f})",
            f"- Composite Score: {state.score:+.2f} (-1=bearish, +1=bullish)",
            "",
            "## Technical Indicators:",
        ]
        
        ind = state.indicators
        if ind:
            lines.extend([
                f"- Price: ${ind.get('close', 0):.2f}",
                f"- EMA({self.config['ema_short']}): ${ind.get('ema_short', 0):.2f}",
                f"- EMA({self.config['ema_long']}): ${ind.get('ema_long', 0):.2f}",
                f"- RSI({self.config['rsi_period']}): {ind.get('rsi', 0):.1f}",
                f"- ATR: {ind.get('atr', 0):.4f} (avg: {ind.get('atr_avg', 0):.4f})",
                f"- MACD Histogram: {ind.get('macd_hist', 0):.4f}",
                f"- BB Upper/Mid/Lower: {ind.get('bb_upper', 0):.2f} / {ind.get('bb_mid', 0):.2f} / {ind.get('bb_lower', 0):.2f}",
                f"- Price Change 1h: {ind.get('price_change_1h', 0)*100:+.2f}%",
                f"- Price Change 24h: {ind.get('price_change_24h', 0)*100:+.2f}%",
                f"- Volume Ratio: {ind.get('volume', 0) / ind.get('volume_avg', 1):.2f}x avg",
            ])
        
        if state.anomalies:
            lines.extend([
                "",
                "## ANOMALIES DETECTED:",
            ])
            for anomaly in state.anomalies:
                lines.append(f"  ⚠️ {anomaly}")
        
        if state.support_levels or state.resistance_levels:
            lines.extend([
                "",
                "## Key Price Levels:",
                f"- Support: {[f'${x:.2f}' for x in state.support_levels[:3]]}",
                f"- Resistance: {[f'${x:.2f}' for x in state.resistance_levels[:3]]}",
            ])
        
        lines.extend([
            "",
            f"## Recommendation: {state.recommendation}",
            "",
            "=" * 50,
            "STRATEGY EVOLUTION HINTS:",
            "- Consider the market regime when choosing indicators",
            f"- Current regime '{state.regime}' suggests: " + self._get_regime_hint(state.regime),
            "- Anomalies may indicate reversals or breakouts",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def _get_regime_hint(self, regime: str) -> str:
        """Get strategy hints based on market regime"""
        hints = {
            "trending": "Use momentum indicators (EMA crossover, MACD). Avoid mean reversion.",
            "ranging": "Use mean reversion (RSI extremes, Bollinger Bands). Avoid trend following.",
            "volatile": "Use wider stops, smaller positions. ATR-based exits work well.",
            "crisis": "Stay flat or use very small positions. Capital preservation priority.",
        }
        return hints.get(regime, "Adapt to current conditions.")
