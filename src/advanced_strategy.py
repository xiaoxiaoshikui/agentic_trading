"""
高级交易策略模块
多指标确认 + 市场状态识别 + 多时间框架分析
目标: 接近专业对冲基金水平
"""

import logging
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Side = Literal["LONG", "SHORT", "FLAT"]
MarketRegime = Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"]


@dataclass
class AdvancedSignal:
    """高级交易信号"""
    side: Side
    confidence: float  # 0-1
    regime: MarketRegime
    reason: str
    entry_type: str  # "breakout", "pullback", "reversal"
    indicators: Dict[str, Any]
    confirmations: int  # 确认指标数量
    

# ============== 高级数据计算 ==============

def calc_funding_rate_signal(funding_rate: float) -> Tuple[str, float]:
    """
    资金费率信号
    极高的资金费率往往意味着反转
    """
    if funding_rate > 0.05:  # 极度看多情绪 -> 做空
        return "SHORT", 0.8
    elif funding_rate < -0.05:  # 极度看空情绪 -> 做多
        return "LONG", 0.8
    return "NEUTRAL", 0.0


def calc_long_short_ratio_signal(ls_ratio: float) -> Tuple[str, float]:
    """
    多空比信号 (反向指标)
    散户都在做多时(>2.5)，往往是顶部
    散户都在做空时(<0.5)，往往是底部
    """
    if ls_ratio > 2.5:
        return "SHORT", 0.7
    elif ls_ratio < 0.6:
        return "LONG", 0.7
    return "NEUTRAL", 0.0


# ============== 技术指标计算 ==============

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 相对强弱指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD 指标"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带"""
    middle = calc_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 真实波动幅度"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ADX 趋势强度指标"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # +DM 和 -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # TR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di


def calc_volume_profile(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """成交量比率"""
    return df["volume"] / df["volume"].rolling(period).mean()


def calc_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    """支撑阻力位计算"""
    recent = df.tail(lookback)
    
    # 简化版: 使用最高最低点
    resistance = recent["high"].max()
    support = recent["low"].min()
    
    # 中间价位
    pivot = (resistance + support + recent["close"].iloc[-1]) / 3
    
    return {
        "resistance": resistance,
        "support": support,
        "pivot": pivot,
        "r1": pivot + (pivot - support),
        "s1": pivot - (resistance - pivot)
    }


# ============== 市场状态识别 ==============

def identify_market_regime(df: pd.DataFrame) -> Tuple[MarketRegime, float]:
    """
    识别市场状态
    返回: (状态, 强度)
    """
    if len(df) < 50:
        return "RANGING", 0.0
    
    close = df["close"]
    
    # 计算 ADX 判断趋势强度
    adx, plus_di, minus_di = calc_adx(df)
    current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    current_plus_di = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 0
    current_minus_di = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 0
    
    # 计算波动率
    atr = calc_atr(df)
    atr_percent = float(atr.iloc[-1] / close.iloc[-1] * 100) if close.iloc[-1] > 0 else 0
    
    # 判断状态
    if current_adx > 25:
        # 趋势市场
        if current_plus_di > current_minus_di:
            return "TRENDING_UP", current_adx / 100
        else:
            return "TRENDING_DOWN", current_adx / 100
    elif atr_percent > 4:
        # 高波动
        return "VOLATILE", atr_percent / 10
    else:
        # 震荡市场
        return "RANGING", 1 - (current_adx / 50)


# ============== 多指标确认系统 ==============

def get_indicator_signals(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    获取所有指标信号
    返回各指标的独立判断
    """
    if len(df) < 200:
        return {}
    
    close = df["close"]
    signals = {}
    
    # 1. EMA 交叉 (趋势跟随)
    ema_fast = calc_ema(close, 20)  # 加快反应速度: 50 -> 20
    ema_slow = calc_ema(close, 50)  # 加快反应速度: 200 -> 50
    ema_trend = calc_ema(close, 200) # 长期趋势参考
    
    current_price = float(close.iloc[-1])
    trend_filter = "LONG" if current_price > ema_trend.iloc[-1] else "SHORT"
    
    ema_signal = "LONG" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "SHORT"
    ema_strength = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
    signals["ema_cross"] = {
        "signal": ema_signal,
        "strength": min(ema_strength, 5) / 5,  # 归一化到 0-1
        "value": f"EMA20={ema_fast.iloc[-1]:.2f}, EMA50={ema_slow.iloc[-1]:.2f}",
        "trend_filter": trend_filter
    }
    
    # 2. RSI (动量反转 + 趋势动量)
    rsi = calc_rsi(close)
    current_rsi = float(rsi.iloc[-1])
    
    if current_rsi < 30:
        rsi_signal = "LONG"  # 超卖反转
        rsi_strength = (30 - current_rsi) / 30
    elif current_rsi > 70:
        rsi_signal = "SHORT"  # 超买反转
        rsi_strength = (current_rsi - 70) / 30
    elif 50 < current_rsi < 70 and trend_filter == "LONG":
        rsi_signal = "LONG"  # 趋势动量
        rsi_strength = 0.5
    elif 30 < current_rsi < 50 and trend_filter == "SHORT":
        rsi_signal = "SHORT" # 趋势动量
        rsi_strength = 0.5
    else:
        rsi_signal = "FLAT"
        rsi_strength = 0
        
    signals["rsi"] = {
        "signal": rsi_signal,
        "strength": rsi_strength,
        "value": f"RSI={current_rsi:.1f}"
    }
    
    # 3. MACD
    macd_line, signal_line, histogram = calc_macd(close)
    macd_signal = "LONG" if histogram.iloc[-1] > 0 else "SHORT"
    # MACD 交叉确认
    macd_cross = "LONG" if (histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0) else \
                 "SHORT" if (histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0) else None
    signals["macd"] = {
        "signal": macd_signal,
        "strength": min(abs(float(histogram.iloc[-1])) / float(close.iloc[-1]) * 1000, 1),
        "value": f"MACD={macd_line.iloc[-1]:.2f}, Signal={signal_line.iloc[-1]:.2f}",
        "cross": macd_cross
    }
    
    # 4. 布林带 (突破 + 均值回归)
    bb_upper, bb_middle, bb_lower = calc_bollinger_bands(close)
    current_price = float(close.iloc[-1])
    bb_position = (current_price - float(bb_lower.iloc[-1])) / (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1]))
    
    if bb_position < 0.05: # 极度超卖，可能反弹
        bb_signal = "LONG"
        bb_strength = 0.8
    elif bb_position > 0.95: # 极度超买，可能回调
        bb_signal = "SHORT" 
        bb_strength = 0.8
    elif bb_position > 0.6 and trend_filter == "LONG": # 趋势向上，价格在上方
        bb_signal = "LONG"
        bb_strength = 0.5
    elif bb_position < 0.4 and trend_filter == "SHORT": # 趋势向下，价格在下方
        bb_signal = "SHORT"
        bb_strength = 0.5
    else:
        bb_signal = "FLAT"
        bb_strength = 0
        
    signals["bollinger"] = {
        "signal": bb_signal,
        "strength": bb_strength,
        "value": f"Price={current_price:.2f}, BB%={bb_position*100:.1f}%"
    }
    
    # 5. 成交量确认
    volume_ratio = calc_volume_profile(df)
    current_vol_ratio = float(volume_ratio.iloc[-1])
    vol_signal = "CONFIRM" if current_vol_ratio > 1.2 else "WEAK" if current_vol_ratio < 0.8 else "NEUTRAL"
    signals["volume"] = {
        "signal": vol_signal,
        "strength": min(current_vol_ratio / 2, 1),
        "value": f"Vol Ratio={current_vol_ratio:.2f}"
    }
    
    # 6. ADX 趋势强度
    adx, plus_di, minus_di = calc_adx(df)
    current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
    adx_signal = "TRENDING" if current_adx > 25 else "RANGING"
    signals["adx"] = {
        "signal": adx_signal,
        "strength": min(current_adx / 50, 1),
        "value": f"ADX={current_adx:.1f}, +DI={plus_di.iloc[-1]:.1f}, -DI={minus_di.iloc[-1]:.1f}"
    }
    
    return signals


def count_confirmations(signals: Dict[str, Dict], target_side: Side) -> int:
    """计算确认指标数量"""
    count = 0
    for name, data in signals.items():
        if name == "volume":
            if data["signal"] == "CONFIRM":
                count += 1
        elif name == "adx":
            if data["signal"] == "TRENDING":
                count += 1
        elif data["signal"] == target_side:
            count += 1
    return count


# ============== 高级信号生成 ==============

def generate_advanced_signal(
    df: pd.DataFrame,
    min_confirmations: int = 3,
    extra_data: Dict[str, float] = None  # 支持传入资金费率等额外数据
) -> Tuple[AdvancedSignal, float]:
    """
    生成高级交易信号
    
    Args:
        df: K线数据
        min_confirmations: 最少需要的确认指标数
        extra_data: 额外市场数据 (funding_rate, long_short_ratio 等)
        
    Returns:
        (AdvancedSignal, ATR值)
    """
    if len(df) < 200:
        return AdvancedSignal(
            side="FLAT",
            confidence=0.0,
            regime="RANGING",
            reason="数据不足",
            entry_type="none",
            indicators={},
            confirmations=0
        ), np.nan
    
    # 1. 识别市场状态
    regime, regime_strength = identify_market_regime(df)
    
    # 2. 获取所有指标信号
    indicators = get_indicator_signals(df)
    
    # 3. 计算 ATR
    atr = calc_atr(df)
    atr_value = float(atr.iloc[-1])
    
    # 4. 获取支撑阻力
    sr_levels = calc_support_resistance(df)
    current_price = float(df["close"].iloc[-1])
    
    # 5. 额外数据过滤
    extra_signal = "NEUTRAL"
    if extra_data:
        fr = extra_data.get("funding_rate", 0)
        ls = extra_data.get("long_short_ratio", 1)
        
        fr_sig, _ = calc_funding_rate_signal(fr)
        ls_sig, _ = calc_long_short_ratio_signal(ls)
        
        if fr_sig == ls_sig and fr_sig != "NEUTRAL":
            extra_signal = fr_sig
    
    # 6. 价格行为分析 (Price Action)
    # 检查突破
    pa_signal = "FLAT"
    pa_reason = ""
    
    resistance = sr_levels["resistance"]
    support = sr_levels["support"]
    pivot = sr_levels["pivot"]
    
    # 突破关键位
    if current_price > resistance * 0.995: # 接近或突破阻力
        if regime == "TRENDING_UP":
            pa_signal = "LONG"
            pa_reason = "上升趋势突破阻力位"
    elif current_price < support * 1.005: # 接近或突破支撑
        if regime == "TRENDING_DOWN":
            pa_signal = "SHORT"
            pa_reason = "下降趋势跌破支撑位"
            
    # 回踩均值
    ema_20 = calc_ema(df["close"], 20).iloc[-1]
    if regime == "TRENDING_UP" and current_price < ema_20 * 1.002 and current_price > ema_20 * 0.998:
        pa_signal = "LONG"
        pa_reason = "上升趋势回踩 EMA20"
    elif regime == "TRENDING_DOWN" and current_price > ema_20 * 0.998 and current_price < ema_20 * 1.002:
        pa_signal = "SHORT"
        pa_reason = "下降趋势反弹 EMA20"
        
    # 7. 综合决策
    long_confirmations = count_confirmations(indicators, "LONG")
    short_confirmations = count_confirmations(indicators, "SHORT")
    
    side = "FLAT"
    confidence = 0.0
    reason = ""
    entry_type = "wait"
    
    # 优先价格行为信号
    if pa_signal == "LONG":
        if long_confirmations >= min_confirmations - 1: # PA信号只需要较少确认
            side = "LONG"
            confidence = 0.8
            reason = f"{pa_reason} + {long_confirmations}指标"
            entry_type = "price_action"
    elif pa_signal == "SHORT":
        if short_confirmations >= min_confirmations - 1:
            side = "SHORT"
            confidence = 0.8
            reason = f"{pa_reason} + {short_confirmations}指标"
            entry_type = "price_action"
            
    # 如果没有 PA 信号，使用指标共振
    if side == "FLAT":
        if regime == "TRENDING_UP" and long_confirmations >= min_confirmations:
            side = "LONG"
            confidence = 0.6 + (long_confirmations * 0.05)
            reason = f"趋势跟随: {long_confirmations}指标确认"
            entry_type = "trend_follow"
        elif regime == "TRENDING_DOWN" and short_confirmations >= min_confirmations:
            side = "SHORT"
            confidence = 0.6 + (short_confirmations * 0.05)
            reason = f"趋势跟随: {short_confirmations}指标确认"
            entry_type = "trend_follow"
        elif regime == "RANGING":
            # 震荡区间做反转
            if current_price < support * 1.01 and long_confirmations >= min_confirmations:
                side = "LONG"
                confidence = 0.7
                reason = "震荡区间底部反转"
                entry_type = "reversal"
            elif current_price > resistance * 0.99 and short_confirmations >= min_confirmations:
                side = "SHORT"
                confidence = 0.7
                reason = "震荡区间顶部反转"
                entry_type = "reversal"
    
    # 额外数据过滤
    if extra_signal != "NEUTRAL":
        if side == extra_signal:
            confidence = min(confidence + 0.15, 1.0)
            reason += f" + 宏观数据支持({extra_signal})"
        elif side != "FLAT" and side != extra_signal:
            confidence *= 0.6 # 宏观数据反对，降低置信度
            reason += f" (宏观数据反对: {extra_signal})"
            if confidence < 0.5: # 如果置信度太低，取消交易
                side = "FLAT"
                reason = f"宏观数据({extra_signal})否定了交易"
    
    # 8. 构建信号
    signal = AdvancedSignal(
        side=side,
        confidence=round(confidence, 2),
        regime=regime,
        reason=reason,
        entry_type=entry_type,
        indicators={
            **indicators,
            "support_resistance": sr_levels,
            "regime_strength": regime_strength,
            "extra_data": extra_data
        },
        confirmations=long_confirmations if side == "LONG" else short_confirmations if side == "SHORT" else 0
    )
    
    return signal, atr_value


# ============== 多时间框架分析 ==============

def analyze_multi_timeframe(
    data_15m: pd.DataFrame,
    data_1h: pd.DataFrame,
    data_4h: pd.DataFrame
) -> Dict[str, Any]:
    """
    多时间框架分析
    
    Args:
        data_15m: 15分钟数据
        data_1h: 1小时数据
        data_4h: 4小时数据
        
    Returns:
        多时间框架分析结果
    """
    results = {}
    
    # 各时间框架信号
    for name, df in [("15m", data_15m), ("1h", data_1h), ("4h", data_4h)]:
        if df is not None and len(df) >= 200:
            signal, _ = generate_advanced_signal(df, min_confirmations=2)
            results[name] = {
                "side": signal.side,
                "confidence": signal.confidence,
                "regime": signal.regime
            }
        else:
            results[name] = {"side": "FLAT", "confidence": 0, "regime": "UNKNOWN"}
    
    # 计算时间框架一致性
    sides = [r["side"] for r in results.values() if r["side"] != "FLAT"]
    
    if len(sides) >= 2:
        # 多数时间框架一致
        long_count = sides.count("LONG")
        short_count = sides.count("SHORT")
        
        if long_count >= 2:
            consensus = "LONG"
            alignment = long_count / 3
        elif short_count >= 2:
            consensus = "SHORT"
            alignment = short_count / 3
        else:
            consensus = "FLAT"
            alignment = 0
    else:
        consensus = "FLAT"
        alignment = 0
    
    results["consensus"] = {
        "side": consensus,
        "alignment": alignment,
        "description": f"{consensus} 信号在 {int(alignment*3)}/3 时间框架确认"
    }
    
    return results


# ============== 入场优化 ==============

def optimize_entry(
    signal: AdvancedSignal,
    current_price: float,
    atr: float
) -> Dict[str, Any]:
    """
    优化入场点
    
    Returns:
        入场优化建议
    """
    sr = signal.indicators.get("support_resistance", {})
    
    if signal.side == "LONG":
        # 做多：寻找支撑位附近入场
        ideal_entry = max(
            current_price - atr * 0.5,  # 回调半个 ATR
            sr.get("support", current_price * 0.98)
        )
        stop_loss = ideal_entry - atr * 2
        take_profit = ideal_entry + atr * 4
        
    elif signal.side == "SHORT":
        # 做空：寻找阻力位附近入场
        ideal_entry = min(
            current_price + atr * 0.5,
            sr.get("resistance", current_price * 1.02)
        )
        stop_loss = ideal_entry + atr * 2
        take_profit = ideal_entry - atr * 4
        
    else:
        return {
            "action": "WAIT",
            "reason": "无明确信号"
        }
    
    return {
        "action": signal.side,
        "current_price": current_price,
        "ideal_entry": round(ideal_entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_reward": round(abs(take_profit - ideal_entry) / abs(ideal_entry - stop_loss), 2),
        "entry_type": signal.entry_type,
        "confidence": signal.confidence
    }
