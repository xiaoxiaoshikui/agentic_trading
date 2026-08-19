import requests
import logging
import re
import json
import ast
import ta
import pandas as pd
import numpy as np
import importlib.util
import traceback
import sys
import os
from typing import Any, Dict, Tuple, Optional
from src.strategy_memory import StrategyMemory
from src.binance_client import BinanceFuturesClient, klines_to_df
from src.config import load_config
from src.advanced_risk import AdvancedRiskManager
from src.multi_asset_simulator import MultiAssetSimulator

logger = logging.getLogger(__name__)

class LocalStrategyArchitect:
    """
    本地策略架构师 (基于 Ollama/Qwen3)
    负责：
    1. 读取 dynamic_strategy.py
    2. 根据市场反馈生成优化建议
    3. 直接重写策略代码
    4. [新] 代码沙箱验证与自动修复
    """
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b", reviewer_model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model_name
        self.reviewer_model = reviewer_model  # 代码审查模型
        self.host = host
        self.strategy_file = "src/dynamic_strategy.py"
        self.memory = StrategyMemory()
        
        # 回测数据缓存
        self.backtest_data: Optional[pd.DataFrame] = None
        self._init_backtest_data()

    def _init_backtest_data(self):
        """初始化回测数据 (真实 K 线 + 尝试获取真实 Funding/OI)"""
        try:
            cfg = load_config()
            client = BinanceFuturesClient(cfg.api_key, cfg.api_secret)
            symbol = cfg.default_symbol or "BTCUSDT"
            
            logger.info(f"⏳ 正在拉取回测数据 ({symbol} 15m)...")
            # 拉取 1000 根 K 线 (约 10 天)
            klines = client.get_klines(symbol, "15m", limit=1000)
            if klines:
                df = klines_to_df(klines)
                
                # --- 尝试获取真实的高级数据 ---
                # 注意: 币安 API 历史数据接口可能受限，且频率不同
                try:
                    # 1. 资金费率 (通常 8h 一次)
                    funding_hist = client.get_funding_rate_history(symbol, limit=100)
                    if funding_hist:
                        f_df = pd.DataFrame(funding_hist)
                        f_df['fundingTime'] = pd.to_datetime(f_df['fundingTime'], unit='ms')
                        f_df['fundingRate'] = f_df['fundingRate'].astype(float)
                        f_df = f_df.set_index('fundingTime').sort_index()
                        
                        # 将 funding rate 合并到主 df (使用 merge_asof 或重采样)
                        # 简单起见，我们创建一个临时的 Series 并重采样
                        # 注意：这里为了简化，我们只做 ffill
                        df['funding_rate'] = 0.0 # 默认
                        # 实际对齐比较复杂，这里简化处理：
                        # 如果没有精确对齐，就用最近的值
                        last_funding = float(funding_hist[-1]['fundingRate'])
                        df['funding_rate'] = last_funding # 暂时用最新的填充，避免 look-ahead bias 太难处理
                        # 理想情况是按时间戳 merge，但 15m vs 8h 需要处理
                        
                    else:
                        df['funding_rate'] = 0.0
                        
                    # 2. 持仓量 (尝试获取 15m 粒度的 OI)
                    oi_hist = client.get_open_interest_history(symbol, period="15m", limit=500)
                    if oi_hist:
                        oi_df = pd.DataFrame(oi_hist)
                        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                        oi_df['sumOpenInterest'] = oi_df['sumOpenInterest'].astype(float) # 这里的单位通常是币
                        
                        # Merge based on timestamp (open_time)
                        # df['open_time'] is already datetime
                        df = pd.merge(df, oi_df[['timestamp', 'sumOpenInterest']], 
                                      left_on='open_time', right_on='timestamp', 
                                      how='left')
                        df.rename(columns={'sumOpenInterest': 'open_interest'}, inplace=True)
                        df['open_interest'] = df['open_interest'].ffill().fillna(0)
                        df.drop(columns=['timestamp'], inplace=True)
                    else:
                        df['open_interest'] = 0.0
                        
                except Exception as advanced_err:
                    logger.warning(f"获取高级历史数据失败，使用默认值: {advanced_err}")
                    df['funding_rate'] = 0.0
                    df['open_interest'] = 0.0
                
                self.backtest_data = df
                logger.info(f"✅ 回测数据加载完成: {len(df)} 条")
            else:
                logger.warning("❌ 无法获取回测数据")
        except Exception as e:
            logger.error(f"初始化回测数据失败: {e}")

    def _clean_response(self, text: str) -> str:
        """清洗模型响应，移除 <think> 标签和 markdown"""
        # 移除 <think>...</think> 块
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # 提取 python 代码块
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果没有代码块，尝试移除 markdown 标记后返回
        return text.replace("```python", "").replace("```", "").strip()

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    def _extract_calculate_signal_for_prompt(self, code: str, fallback_max_chars: int = 2500) -> str:
        if not code:
            return ""
        m = re.search(r"^\s*def\s+calculate_signal\s*\(.*?\):\s*(?:\n.*?)(?=\n\s*def\s+|\Z)", code, re.DOTALL | re.MULTILINE)
        if m:
            return m.group(0).strip()
        return self._truncate_text(code, fallback_max_chars)

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if json_match:
            candidate = json_match.group(1).strip()
        else:
            any_block = re.search(r"```\s*(.*?)```", text, re.DOTALL)
            candidate = any_block.group(1).strip() if any_block else text

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        raw = candidate[start:end + 1].strip()
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        try:
            obj = ast.literal_eval(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        try:
            py_raw = raw
            py_raw = re.sub(r"\btrue\b", "True", py_raw, flags=re.IGNORECASE)
            py_raw = re.sub(r"\bfalse\b", "False", py_raw, flags=re.IGNORECASE)
            py_raw = re.sub(r"\bnull\b", "None", py_raw, flags=re.IGNORECASE)
            obj = ast.literal_eval(py_raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _normalize_strategy_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "ema_fast": 50,
            "ema_slow": 200,
            "rsi_window": 14,
            "rsi_long_min": 50.0,
            "rsi_long_max": 70.0,
            "rsi_short_min": 30.0,
            "rsi_short_max": 50.0,
            "volume_window": 20,
            "volume_mult": 2.0,
            "oi_window": 20,
            "funding_window": 96,
            "funding_z": 1.0,
            "min_confidence": 0.55,
            "use_mean_reversion": False,
            "weights": {
                "trend": 0.40,
                "rsi": 0.30,
                "volume": 0.15,
                "funding": 0.10,
                "oi": 0.05,
            },
        }

        out: Dict[str, Any] = {**defaults, **(spec or {})}

        def _to_int(v: Any, lo: int, hi: int, d: int) -> int:
            try:
                vv = int(float(v))
            except Exception:
                return d
            return max(lo, min(hi, vv))

        def _to_float(v: Any, lo: float, hi: float, d: float) -> float:
            try:
                vv = float(v)
            except Exception:
                return d
            return max(lo, min(hi, vv))

        out["ema_fast"] = _to_int(out.get("ema_fast"), 10, 120, defaults["ema_fast"])
        out["ema_slow"] = _to_int(out.get("ema_slow"), 50, 400, defaults["ema_slow"])
        if out["ema_slow"] <= out["ema_fast"]:
            out["ema_slow"] = min(400, out["ema_fast"] + 50)

        out["rsi_window"] = _to_int(out.get("rsi_window"), 5, 50, defaults["rsi_window"])
        out["volume_window"] = _to_int(out.get("volume_window"), 5, 100, defaults["volume_window"])
        out["oi_window"] = _to_int(out.get("oi_window"), 1, 100, defaults["oi_window"])
        out["funding_window"] = _to_int(out.get("funding_window"), 5, 400, defaults["funding_window"])

        out["rsi_long_min"] = _to_float(out.get("rsi_long_min"), 40.0, 65.0, defaults["rsi_long_min"])
        out["rsi_long_max"] = _to_float(out.get("rsi_long_max"), 55.0, 85.0, defaults["rsi_long_max"])
        if out["rsi_long_max"] <= out["rsi_long_min"]:
            out["rsi_long_max"] = min(85.0, out["rsi_long_min"] + 10.0)

        out["rsi_short_min"] = _to_float(out.get("rsi_short_min"), 10.0, 45.0, defaults["rsi_short_min"])
        out["rsi_short_max"] = _to_float(out.get("rsi_short_max"), 25.0, 60.0, defaults["rsi_short_max"])
        if out["rsi_short_max"] <= out["rsi_short_min"]:
            out["rsi_short_max"] = min(60.0, out["rsi_short_min"] + 10.0)

        out["volume_mult"] = _to_float(out.get("volume_mult"), 1.0, 5.0, defaults["volume_mult"])
        out["funding_z"] = _to_float(out.get("funding_z"), 0.2, 5.0, defaults["funding_z"])
        out["min_confidence"] = _to_float(out.get("min_confidence"), 0.3, 0.9, defaults["min_confidence"])
        out["use_mean_reversion"] = bool(out.get("use_mean_reversion", defaults["use_mean_reversion"]))

        weights = out.get("weights")
        if not isinstance(weights, dict):
            weights = defaults["weights"].copy()
        normed: Dict[str, float] = {}
        for k in defaults["weights"].keys():
            normed[k] = _to_float(weights.get(k), 0.0, 1.0, defaults["weights"][k])
        s = sum(normed.values())
        if s <= 0:
            normed = defaults["weights"].copy()
            s = sum(normed.values())
        for k in normed:
            normed[k] = float(normed[k] / s)
        out["weights"] = normed

        return out

    def _compile_strategy_code(self, spec: Dict[str, Any]) -> str:
        spec = self._normalize_strategy_spec(spec)
        w = spec["weights"]

        lines = [
            "import pandas as pd",
            "import numpy as np",
            "import ta",
            "from dataclasses import dataclass",
            "",
            f"STRATEGY_SPEC = {repr(spec)}",
            "",
            "@dataclass",
            "class Signal:",
            "    side: str",
            "    confidence: float",
            "    reason: str",
            "",
            "def calculate_signal(df: pd.DataFrame) -> Signal:",
            "    try:",
            "        if 'funding_rate' not in df.columns:",
            "            df['funding_rate'] = 0.0",
            "        if 'open_interest' not in df.columns:",
            "            df['open_interest'] = 0.0",
            "",
            f"        min_bars = max({int(spec['ema_slow'])}, {int(spec['rsi_window'])}, {int(spec['volume_window'])}, {int(spec['oi_window'])} + 2, {int(spec['funding_window'])} + 2, 50)",
            "        if len(df) < min_bars:",
            "            return Signal(side='FLAT', confidence=0.0, reason='Insufficient data')",
            "",
            "        close = float(df['close'].iloc[-1])",
            f"        ema_fast = float(ta.trend.ema_indicator(df['close'], window=int({int(spec['ema_fast'])})).iloc[-1])",
            f"        ema_slow = float(ta.trend.ema_indicator(df['close'], window=int({int(spec['ema_slow'])})).iloc[-1])",
            f"        rsi = float(ta.momentum.rsi(df['close'], window=int({int(spec['rsi_window'])})).iloc[-1])",
            "        if not np.isfinite(ema_fast) or not np.isfinite(ema_slow) or not np.isfinite(rsi):",
            "            return Signal(side='FLAT', confidence=0.0, reason='NaN indicator')",
            "",
            f"        vol_ma = float(df['volume'].rolling(int({int(spec['volume_window'])})).mean().iloc[-1])",
            "        vol_ratio = float(df['volume'].iloc[-1] / (vol_ma + 1e-9))",
            "",
            "        oi = df['open_interest'].astype(float)",
            f"        oi_mom = float(oi.pct_change(int({int(spec['oi_window'])})).iloc[-1])",
            "        if not np.isfinite(oi_mom):",
            "            oi_mom = 0.0",
            "",
            "        fr = df['funding_rate'].astype(float)",
            f"        fr_mean = float(fr.rolling(int({int(spec['funding_window'])})).mean().iloc[-1])",
            f"        fr_std = float(fr.rolling(int({int(spec['funding_window'])})).std().iloc[-1])",
            "        if not np.isfinite(fr_std):",
            "            fr_std = 0.0",
            "        fr_z = float((fr.iloc[-1] - fr_mean) / (fr_std + 1e-9))",
            "        if not np.isfinite(fr_z):",
            "            fr_z = 0.0",
            "",
            "        long_score = 0.0",
            "        short_score = 0.0",
            "        reasons = []",
            "",
            "        uptrend = close > ema_fast and ema_fast > ema_slow",
            "        downtrend = close < ema_fast and ema_fast < ema_slow",
            "",
            "        if uptrend:",
            f"            long_score += {float(w['trend'])}",
            "            reasons.append('Uptrend')",
            "        elif downtrend:",
            f"            short_score += {float(w['trend'])}",
            "            reasons.append('Downtrend')",
            "",
            f"        if rsi > {float(spec['rsi_long_min'])} and rsi < {float(spec['rsi_long_max'])}:",
            f"            long_score += {float(w['rsi'])}",
            "            reasons.append('RSI bull')",
            f"        if rsi > {float(spec['rsi_short_min'])} and rsi < {float(spec['rsi_short_max'])}:",
            f"            short_score += {float(w['rsi'])}",
            "            reasons.append('RSI bear')",
            "",
            f"        if vol_ratio > {float(spec['volume_mult'])}:",
            "            if uptrend:",
            f"                long_score += {float(w['volume'])}",
            "                reasons.append('Volume spike')",
            "            elif downtrend:",
            f"                short_score += {float(w['volume'])}",
            "                reasons.append('Volume spike')",
            "",
            f"        if abs(fr_z) > {float(spec['funding_z'])}:",
            "            if fr_z > 0:",
            f"                short_score += {float(w['funding'])}",
            "                reasons.append('Crowded longs')",
            "            else:",
            f"                long_score += {float(w['funding'])}",
            "                reasons.append('Crowded shorts')",
            "",
            "        if oi_mom > 0:",
            "            if uptrend:",
            f"                long_score += {float(w['oi'])}",
            "                reasons.append('OI rising')",
            "            elif downtrend:",
            f"                short_score += {float(w['oi'])}",
            "                reasons.append('OI rising')",
            "",
            f"        if {bool(spec['use_mean_reversion'])}:",
            "            bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)",
            "            upper = float(bb.bollinger_hband().iloc[-1])",
            "            lower = float(bb.bollinger_lband().iloc[-1])",
            "            if np.isfinite(upper) and np.isfinite(lower):",
            "                if close > upper and rsi > 70:",
            f"                    short_score += {float(w['volume'])}",
            "                    reasons.append('BB overbought')",
            "                elif close < lower and rsi < 30:",
            f"                    long_score += {float(w['volume'])}",
            "                    reasons.append('BB oversold')",
            "",
            "        side = 'FLAT'",
            "        conf = 0.0",
            f"        min_conf = float({float(spec['min_confidence'])})",
            "        if long_score >= min_conf and long_score > short_score:",
            "            side = 'LONG'",
            "            conf = min(1.0, long_score)",
            "        elif short_score >= min_conf and short_score > long_score:",
            "            side = 'SHORT'",
            "            conf = min(1.0, short_score)",
            "",
            "        reason = '; '.join(reasons) if reasons else 'No edge'",
            "        return Signal(side=side, confidence=float(conf), reason=reason)",
            "",
            "    except Exception as e:",
            "        return Signal(side='FLAT', confidence=0.0, reason='Error: ' + str(e))",
            "",
        ]

        return "\n".join(lines)

    def _split_train_test(self, df: pd.DataFrame, test_ratio: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df is None or len(df) < 400:
            return df, df
        split = int(len(df) * (1.0 - test_ratio))
        split = max(300, min(len(df) - 100, split))
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()
        return train, test

    def _backtest_strategy_aligned(self, code: str, df: pd.DataFrame, symbol: str = "BTCUSDT") -> Tuple[bool, Dict[str, Any]]:
        """Risk-aligned backtest using MultiAssetSimulator + AdvancedRiskManager on a single symbol."""
        if df is None or len(df) < 300:
            return False, {"error": "Not enough data"}

        temp_file = "src/temp_aligned_backtest_strategy.py"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            spec = importlib.util.spec_from_file_location("aligned_backtest_strat", temp_file)
            if spec is None or spec.loader is None:
                return False, {"error": "Cannot load strategy module"}
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            sim = MultiAssetSimulator(
                initial_capital=10000.0,
                max_positions=3,
                capital_per_position=0.3,
                max_long_exposure=0.6,
                max_short_exposure=0.6,
                max_daily_loss=0.03,
                max_single_asset_loss=0.05,
            )
            risk = AdvancedRiskManager()

            start_idx = 200
            total_bars = len(df)
            for i in range(start_idx, total_bars):
                row = df.iloc[i]

                current_time = df.index[i]
                current_ts = float(current_time.timestamp()) if hasattr(current_time, "timestamp") else float(i)
                current_price = float(row["close"])

                sim.check_positions({symbol: current_price}, current_ts)

                if not sim.can_open_position(symbol):
                    continue

                df_slice = df.iloc[: i + 1].copy()
                if len(df_slice) < start_idx:
                    continue

                try:
                    signal = module.calculate_signal(df_slice)
                except Exception:
                    continue

                if getattr(signal, "side", "FLAT") == "FLAT" or float(getattr(signal, "confidence", 0.0)) < 0.3:
                    continue

                if not sim.can_open_position(symbol, getattr(signal, "side", None)):
                    continue

                atr = None
                if "atr" in df_slice.columns:
                    atr = float(df_slice["atr"].iloc[-1])
                if atr is None or not np.isfinite(atr) or atr <= 0:
                    try:
                        atr_series = ta.volatility.average_true_range(df_slice["high"], df_slice["low"], df_slice["close"], window=14)
                        atr = float(atr_series.iloc[-1])
                    except Exception:
                        atr = current_price * 0.01

                if getattr(signal, "side", "FLAT") == "LONG":
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 4)
                else:
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 4)

                try:
                    plan = risk.calculate_position(
                        balance=sim.get_position_capital(),
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        signal_confidence=float(getattr(signal, "confidence", 0.0)),
                        market_regime="trending",
                        atr=atr,
                        atr_percent=atr / current_price if current_price > 0 else 0.02,
                    )
                except Exception:
                    continue

                if getattr(plan, "position_size", 0) <= 0:
                    continue

                sim.open_position(
                    symbol=symbol,
                    direction=getattr(signal, "side", "FLAT"),
                    entry_price=current_price,
                    position_size=float(plan.position_size),
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    current_time=current_ts,
                )

            final_price = float(df.iloc[-1]["close"])
            for sym in list(sim.positions.keys()):
                sim.close_position(sym, final_price, "End")

            report = sim.get_report()
            if "error" in report:
                return False, report
            return True, report

        except Exception as e:
            return False, {"error": f"Aligned backtest error: {str(e)}"}
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _review_code_logic(self, code: str) -> Tuple[bool, str]:
        """
        使用 DeepSeek-R1 审查代码逻辑
        Returns: (是否通过, 审查意见)
        """
        logger.info(f"🕵️ 使用 {self.reviewer_model} 进行量化逻辑审查...")
        
        review_prompt = f"""
你是一个严格的**量化策略风控专家**。请审查以下 Python 交易策略代码。

【审查核心标准 - 必须严格执行】
1. **严禁未来函数 (Future Look-ahead)**:
   - 检查是否使用了 `shift(-1)`, `iloc[i+1]` 等“偷看未来”的操作。
   - 所有指标计算只能依赖当前及过去的数据。
2. **数学稳健性 (Robustness)**:
   - 检查除法操作是否处理了分母为零的情况 (如 `+ 1e-9` 或 `replace(0, ...)`).
   - `lower_band` 等价格相关指标不能乘以错误的系数 (如 `lower * 1.8` 是严重错误)。
3. **Alpha 逻辑合理性**:
   - 检查资金费率 (Funding Rate) 和持仓量 (OI) 的使用是否符合金融逻辑。
   - 例如：资金费率极高通常意味着多头拥挤，可能是反向信号。
4. **代码完整性**:
   - 必须处理异常 (try-except)。
   - 必须返回标准的 `Signal` 对象。

【待审查代码】
{code}

【输出格式】
- 如果代码**安全且逻辑自洽**，只输出 "PASS"。
- 如果发现隐患，请列出具体问题点 (如: "第15行存在除零风险", "使用了未来函数")。
"""

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.reviewer_model,
                    "prompt": review_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=180
            )

            if response.status_code != 200:
                logger.warning(f"审查 API 失败，跳过审查: {response.text}")
                return True, "审查跳过"

            result = response.json()
            review_feedback = result.get("response", "").strip()
            review_feedback = re.sub(r"<think>.*?</think>", "", review_feedback, flags=re.DOTALL).strip()

            if "PASS" in review_feedback and len(review_feedback) < 20:
                logger.info("✅ 代码逻辑审查通过")
                return True, ""

            logger.warning(f"❌ 代码审查未通过: {review_feedback}")
            return False, review_feedback

        except Exception as e:
            logger.warning(f"审查过程出错: {e}")
            return True, "审查出错跳过"

    def _verify_code(self, code: str) -> Tuple[bool, str]:
        """
        沙箱验证代码 (Runtime Check)
        Returns: (是否通过, 错误信息)
        """
        temp_file = "src/temp_strategy_check.py"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            spec = importlib.util.spec_from_file_location("check_strategy", temp_file)
            if spec is None or spec.loader is None:
                return False, "无法加载模块规范"

            module = importlib.util.module_from_spec(spec)
            sys.modules["check_strategy"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "calculate_signal"):
                return False, "缺少 calculate_signal 函数"

            return True, ""

        except Exception as e:
            return False, f"语法或运行时错误: {str(e)}"
        finally:
            if "check_strategy" in sys.modules:
                del sys.modules["check_strategy"]
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _backtest_strategy(self, code: str) -> Tuple[bool, str]:
        """
        Fast vectorized backtest on historical data
        Returns: (is_profitable, report_string)
        """
        if self.backtest_data is None or len(self.backtest_data) < 100:
            return True, "Backtest skipped: insufficient data"

        temp_file = "src/temp_backtest_strategy.py"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            spec = importlib.util.spec_from_file_location("backtest_strat", temp_file)
            if spec is None or spec.loader is None:
                return False, "Backtest error: cannot load module"
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            df = self.backtest_data.copy()
            signals = []

            window_size = 100
            for i in range(window_size, len(df) - 1):
                window_df = df.iloc[i-window_size:i+1].copy()
                try:
                    sig = module.calculate_signal(window_df)
                    signals.append({
                        'idx': i,
                        'side': sig.side,
                        'confidence': sig.confidence,
                        'next_return': (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
                    })
                except Exception:
                    continue

            if not signals:
                return False, "Backtest failed: no signals generated"

            total_return = 0.0
            trade_returns = []
            wins = 0
            trades = 0

            for s in signals:
                if s['side'] == 'FLAT':
                    continue
                trades += 1
                ret = s['next_return'] if s['side'] == 'LONG' else -s['next_return']
                ret *= s['confidence']
                total_return += ret
                trade_returns.append(ret)
                if ret > 0:
                    wins += 1

            win_rate = (wins / trades * 100) if trades > 0 else 0

            gross_profit = float(sum(r for r in trade_returns if r > 0))
            gross_loss = float(-sum(r for r in trade_returns if r < 0))
            profit_factor = float(gross_profit / (gross_loss + 1e-9)) if trades > 0 else 0.0

            fee_per_trade = 0.0002
            net_return = float(total_return - trades * fee_per_trade)
            trade_rate = float(trades / max(1, len(signals)))

            report = (
                f"Backtest ({len(signals)} bars): Return {total_return:.4f}, Net {net_return:.4f}, "
                f"Trades {trades}, Win rate {win_rate:.2f}%, PF {profit_factor:.2f}, TradeRate {trade_rate:.3f}"
            )

            return net_return > 0, report

        except Exception as e:
            return False, f"Backtest error: {str(e)}"
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

    def _evaluate_candidate(self, code: str, do_review: bool = True) -> Dict:
        """评估单个候选策略，返回详细指标"""
        metrics = {
            "valid": False, 
            "profit": -999.0, 
            "net_profit": -999.0,
            "win_rate": 0.0, 
            "profit_factor": 0.0,
            "trades": 0,
            "trade_rate": 0.0,
            "score": -999.0,
            "train_score": -999.0,
            "test_score": -999.0,
            "train_report": "",
            "test_report": "",
            "error": "",
            "report": ""
        }
        
        # 1. 逻辑审查
        if do_review:
            is_logic_valid, review_msg = self._review_code_logic(code)
            if not is_logic_valid:
                metrics["error"] = f"逻辑审查失败: {review_msg}"
                return metrics
            
        # 2. 沙箱验证
        is_runtime_valid, runtime_msg = self._verify_code(code)
        if not is_runtime_valid:
            metrics["error"] = f"运行时错误: {runtime_msg}"
            return metrics
            
        # 3. 光速回测
        is_profitable, report_str = self._backtest_strategy(code)
        
        # Parse backtest report
        # Format: "Backtest (X bars): Return 0.1234, Net 0.1200, Trades 10, Win rate 65.00%, PF 1.30, TradeRate 0.100"
        try:
            profit_match = re.search(r"Return ([-\d\.]+)", report_str)
            net_match = re.search(r"Net ([-\d\.]+)", report_str)
            trades_match = re.search(r"Trades (\d+)", report_str)
            win_match = re.search(r"Win rate ([\d\.]+)%", report_str)
            pf_match = re.search(r"PF ([\d\.]+)", report_str)
            tr_match = re.search(r"TradeRate ([\d\.]+)", report_str)
            
            if profit_match:
                metrics["profit"] = float(profit_match.group(1))
            if net_match:
                metrics["net_profit"] = float(net_match.group(1))
            else:
                metrics["net_profit"] = metrics["profit"]
            if trades_match:
                metrics["trades"] = int(trades_match.group(1))
            if win_match:
                metrics["win_rate"] = float(win_match.group(1))
            if pf_match:
                metrics["profit_factor"] = float(pf_match.group(1))
            if tr_match:
                metrics["trade_rate"] = float(tr_match.group(1))
            
            metrics["valid"] = True
            metrics["report"] = report_str

            # Scoring: prefer higher net return and higher PF, penalize overtrading.
            # Note: net_profit is a per-bar return proxy (not compounded).
            pf_component = max(0.0, metrics["profit_factor"] - 1.0)
            overtrade_penalty = max(0.0, metrics["trade_rate"] - 0.12)
            metrics["score"] = float(metrics["net_profit"] + 0.10 * pf_component - 0.20 * overtrade_penalty)
            
            # 如果回测亏损，虽然 valid=True (代码能跑)，但在选拔中会被降权
            if not is_profitable:
                metrics["error"] = "回测未盈利"
                metrics["score"] -= 0.50
                return metrics

            # 4. 风控对齐回测 + OOS(train/test)
            if self.backtest_data is not None and len(self.backtest_data) >= 400:
                train_df, test_df = self._split_train_test(self.backtest_data)

                ok_tr, tr_report = self._backtest_strategy_aligned(code, train_df)
                ok_te, te_report = self._backtest_strategy_aligned(code, test_df)

                def _score_from_report(rep: Dict[str, Any], bars: int) -> float:
                    if not rep or "error" in rep:
                        return -999.0
                    roi = float(rep.get("roi_percent", 0.0)) / 100.0
                    pf = float(rep.get("profit_factor", 0.0))
                    dd = float(rep.get("max_drawdown_percent", 0.0)) / 100.0
                    trades = int(rep.get("total_trades", 0))
                    trade_rate = float(trades / max(1, bars))
                    return float(roi + 0.05 * max(0.0, pf - 1.0) - 0.20 * dd - 0.05 * max(0.0, trade_rate - 0.02))

                metrics["train_report"] = str(tr_report)
                metrics["test_report"] = str(te_report)
                metrics["train_score"] = _score_from_report(tr_report, len(train_df) if train_df is not None else 0)
                metrics["test_score"] = _score_from_report(te_report, len(test_df) if test_df is not None else 0)

                if ok_te and metrics["test_score"] > -900:
                    metrics["score"] = float(metrics["test_score"])
                
        except Exception as e:
            metrics["error"] = f"解析回测结果失败: {e}"
            
        return metrics

    def _save_new_strategy(self, code: str) -> bool:
        """保存新策略代码"""
        try:
            with open(self.strategy_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info("✅ 新策略代码已保存")
            return True
        except Exception as e:
            logger.error(f"保存策略失败: {e}")
            return False

    def _read_current_strategy(self) -> str:
        """读取当前策略源码"""
        try:
            with open(self.strategy_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"无法读取策略文件: {e}")
            return ""

    def optimize_strategy(self, market_context: str, performance_metrics: Dict) -> bool:
        """
        Core method: Tournament Optimization
        Generate multiple strategies -> Backtest -> Survival of the fittest
        
        Args:
            market_context: Rich market analysis from MarketAnalyst.generate_llm_context()
            performance_metrics: dict with 'win_rate', 'issue', etc.
        """
        current_code = self._read_current_strategy()
        if not current_code:
            return False
        
        best_past_strategy = self.memory.retrieve_relevant_strategy(market_context)
        memory_context = ""
        if best_past_strategy:
            memory_context = f"\n## Historical Reference (similar market conditions):\n```python\n{best_past_strategy['code']}\n```\n"

        # Extract profitability metrics
        profit_factor = performance_metrics.get('profit_factor', 0)
        avg_trade = performance_metrics.get('avg_trade', 0)
        
        prompt_market_context = self._truncate_text(market_context, 2200)
        prompt_strategy_code = self._extract_calculate_signal_for_prompt(current_code, fallback_max_chars=2200)

        base_prompt = f"""You are a quantitative researcher. Propose a StrategySpec (JSON) for a safe strategy template.

{prompt_market_context}

## Current Strategy Code (NEEDS IMPROVEMENT)
```python
{prompt_strategy_code}
```

## Recent Performance
- Win Rate: {performance_metrics.get('win_rate', 0):.1f}%
- Profit Factor: {profit_factor:.2f} (avg_win / avg_loss ratio)
- Avg Trade PnL: ${avg_trade:.2f}
- Issue: {performance_metrics.get('issue', 'None')}

## Risk Management Context (DO NOT MODIFY - just for awareness)
- Stop Loss: 2 × ATR (fixed by risk module)
- Take Profit: 4 × ATR (fixed by risk module)
- This means: You need signals where price moves 4×ATR BEFORE hitting 2×ATR stop
- Breakeven Win Rate needed: ~33% (with 2:1 reward/risk)

## Profitability Goal
- Current Profit Factor: {profit_factor:.2f}
- REQUIRED: Profit Factor > 1.27 for profitability with 44% win rate
- FOCUS: Generate signals that MAXIMIZE PROFIT FACTOR, not just win rate
- STRATEGY: Look for strong trend signals, avoid choppy/ranging markets

## Available Data Columns
`df` contains: `close`, `high`, `low`, `volume`, `funding_rate`, `open_interest`

## Alpha Discovery Guidelines
1. **Vectorized Computation**: Use `numpy`/`pandas` for momentum, volatility, money flow.
2. **Dynamic Thresholds**: Use `rolling(N).mean()` instead of hardcoded numbers.
3. **Market Microstructure**: Leverage `funding_rate` and `open_interest` for crowd positioning.
4. **Regime Adaptation**: Consider the market regime mentioned above when designing logic.
5. **Trend Confirmation**: Only signal when trend is strong enough to reach 4×ATR target.

## Code Safety Checklist (must pass strict reviewer)
1. **NO Future Look-ahead**: Do NOT use `shift(-1)`, `iloc[i+1]`, any forward shift, or any logic that peeks future bars.
2. **NaN-Safe / Robustness**:
   - Any `pct_change`, `diff`, `rolling`, indicator output may be NaN at the beginning; guard with `fillna`, `dropna` logic or early return FLAT.
   - Any division must protect denominator with `+ 1e-9` (or equivalent).
3. **Missing Columns**: If `funding_rate` / `open_interest` columns are missing, default them to 0.0 (do not crash).
4. **Threshold Sanity**:
   - Funding rate is typically small (decimal). Use dynamic thresholds (rolling mean/std) rather than tiny hard-coded values.
   - Avoid magic numbers; prefer volatility/regime-scaled conditions.
5. **Completeness**:
   - `calculate_signal(data)` MUST be wrapped with `try/except` and return a valid `Signal` in all branches.

## Output Requirements
Return ONLY a single JSON object (no markdown, no python code) with this schema:
```json
{
  "ema_fast": 50,
  "ema_slow": 200,
  "rsi_window": 14,
  "rsi_long_min": 50,
  "rsi_long_max": 70,
  "rsi_short_min": 30,
  "rsi_short_max": 50,
  "volume_window": 20,
  "volume_mult": 2.0,
  "oi_window": 20,
  "funding_window": 96,
  "funding_z": 1.0,
  "min_confidence": 0.55,
  "use_mean_reversion": false,
  "weights": {"trend": 0.4, "rsi": 0.3, "volume": 0.15, "funding": 0.1, "oi": 0.05}
}
```
Rules:
- Only use these keys.
- Windows must be integers; thresholds must be numbers.
- `weights` values must be non-negative; they will be normalized to sum to 1.

{memory_context}

## Task
Propose a StrategySpec JSON that improves profitability and avoids overtrading under the risk constraints.
"""
        
        # 锦标赛配置
        candidates_count = 3  # 参赛选手数量
        candidates = []
        last_failure_feedback = ""
        
        logger.info(f"🏆 开启策略进化锦标赛 (选手: {candidates_count}名)...")
        
        for i in range(candidates_count):
            logger.info(f"🤖 生成选手 #{i+1} ...")
            
            # 使用不同的 temperature 增加多样性
            # 选手1: 稳健 (0.2); 选手2: 均衡 (0.4); 选手3: 激进 (0.6)
            temp = 0.2 + (i * 0.2)
            
            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": base_prompt
                        + ("\n\nPrevious attempt feedback: " + last_failure_feedback if last_failure_feedback else "")
                        + f"\n\n(variant #{i+1}: propose a different parameter set within the JSON schema)",
                        "stream": False,
                        "options": {"temperature": temp}
                    },
                    timeout=240  # Increased for complex code generation
                )
                
                if response.status_code != 200:
                    continue
                
                generated_text = response.json().get("response", "")
                spec = self._extract_json_object(generated_text)
                if spec is None:
                    last_failure_feedback = "Output was not valid JSON. Return ONLY one JSON object with the schema." 
                    score_card = {
                        "valid": False,
                        "profit": -999.0,
                        "win_rate": 0.0,
                        "error": "Invalid StrategySpec JSON",
                        "report": "",
                        "id": i + 1,
                    }
                    candidates.append(score_card)
                    logger.warning(f"   ❌ 选手 #{i+1} 淘汰: {score_card['error']}")
                    continue

                spec = self._normalize_strategy_spec(spec)
                code = self._compile_strategy_code(spec)
                
                # 评估选手
                score_card = self._evaluate_candidate(code, do_review=False)
                score_card["code"] = code
                score_card["spec"] = spec
                score_card["id"] = i + 1
                
                candidates.append(score_card)
                
                if score_card.get("valid") and not score_card.get("error"):
                    last_failure_feedback = ""
                    logger.info(f"   ✅ 选手 #{i+1} 成绩: 收益 {score_card['profit']:.2%} | 胜率 {score_card['win_rate']:.0%}")
                else:
                    last_failure_feedback = score_card.get("error", "")
                    logger.warning(f"   ❌ 选手 #{i+1} 淘汰: {score_card.get('error', '')}")
                    
            except Exception as e:
                logger.error(f"选手 #{i+1} 生成失败: {e}")

        # 选拔冠军
        # 筛选出 valid 的候选人
        valid_candidates = [c for c in candidates if c.get("valid") and not c.get("error")]
        
        if not valid_candidates:
            logger.error("❌ 所有选手均未通过选拔，进化失败")
            return False
            
        # 按综合得分排序 (优先 OOS test_score)
        best_candidate = max(valid_candidates, key=lambda x: x.get("score", x["profit"]))

        # Only run strict LLM logic review on champion to reduce timeouts
        is_logic_valid, review_msg = self._review_code_logic(best_candidate["code"])
        if not is_logic_valid:
            logger.warning(f"⚠️ 冠军未通过审查，尝试下一名: {review_msg}")
            remaining = [c for c in valid_candidates if c is not best_candidate]
            remaining_sorted = sorted(remaining, key=lambda x: x.get("score", x["profit"]), reverse=True)
            picked = None
            for cand in remaining_sorted:
                ok, msg = self._review_code_logic(cand["code"])
                if ok:
                    picked = cand
                    break
                logger.warning(f"⚠️ 候选未通过审查: {msg}")
            if picked is None:
                logger.error("❌ 所有候选均未通过审查")
                return False
            best_candidate = picked
        
        # 门槛检查 (如果冠军也是亏损严重的，就不录用)
        if best_candidate.get("net_profit", best_candidate["profit"]) < -0.01:
            logger.warning(f"⚠️ 冠军选手表现不佳 (收益 {best_candidate.get('net_profit', best_candidate['profit']):.2%})，放弃本次进化")
            return False
            
        logger.info(f"👑 锦标赛冠军诞生: 选手 #{best_candidate['id']}")
        logger.info(f"   🏆 最终成绩: {best_candidate['report']}")
        
        return self._save_new_strategy(best_candidate["code"])

    def auto_fix_code(self, file_path: str, error_traceback: str) -> bool:
        """
        自动修复代码错误
        Args:
            file_path: 出错的文件路径
            error_traceback: 完整的错误堆栈
        Returns:
            是否修复成功
        """
        logger.info(f"🔧 尝试自动修复: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                current_code = f.read()
        except Exception as e:
            logger.error(f"无法读取文件: {e}")
            return False
        
        fix_prompt = f"""
你是一个 Python 代码修复专家。以下代码运行时出错，请修复它。

【出错文件】
{file_path}

【错误信息】
{error_traceback}

【当前代码】
```python
{current_code}
```

【任务】
1. 分析错误原因
2. 修复代码中的 bug
3. 确保修复后的代码能正常运行
4. 只输出修复后的完整代码，从 import 语句开始

只输出 Python 代码，不要解释。
"""
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": fix_prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.text}")
                return False
            
            result = response.json()
            generated_text = result.get("response", "")
            
            # 清洗响应（处理 <think> 和 markdown）
            new_code = self._clean_response(generated_text)
            
            # 保存修复后的代码
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            
            logger.info(f"✅ 代码已自动修复: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"自动修复失败: {e}")
            return False
