"""
实时分析器
整合高级信号 + 多智能体决策 + 网络情报
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from .advanced_strategy import (
    generate_advanced_signal, 
    get_indicator_signals,
    identify_market_regime,
    optimize_entry
)
from .advanced_risk import AdvancedRiskManager, RiskProfile
from .multi_agent_trading import MultiAgentTradingSystem
from .web_intelligence import WebIntelligenceAgent
from .recorder import AnalysisRecorder

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAnalysis:
    """实时分析结果"""
    # 高级信号
    signal: str  # LONG/SHORT/FLAT
    confidence: float
    regime: str  # 市场状态
    confirmations: int
    
    # 指标详情
    indicators: Dict[str, Any]
    
    # 仓位计划
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    
    # 多智能体决策
    agent_decision: Dict[str, Any]
    
    # 网络情报
    web_intelligence: Dict[str, Any] = None
    
    # 多时间框架分析 (新增)
    multi_timeframe: Dict[str, Any] = None
    
    # 综合建议
    action: str = "FLAT"
    reason: str = ""
    score: float = 0.0  # 0-100


class RealtimeAnalyzer:
    """
    实时分析器
    整合:
    - 高级技术信号 (6指标)
    - 市场状态识别
    - 动态风险管理
    - 多智能体决策
    - 网络情报 (GPT Web Search)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        risk_profile: RiskProfile = None,
        enable_multi_agent: bool = True,
        enable_web_search: bool = True,
        min_confirmations: int = 3
    ):
        self.model = model
        self.api_key = api_key
        self.min_confirmations = min_confirmations
        self.enable_multi_agent = enable_multi_agent
        self.enable_web_search = enable_web_search
        
        # 风险管理器
        self.risk_manager = AdvancedRiskManager(risk_profile or RiskProfile())
        
        # 多智能体系统 (可选)
        self.multi_agent = None
        if enable_multi_agent:
            try:
                self.multi_agent = MultiAgentTradingSystem(model, api_key)
                logger.info("多智能体系统已启用")
            except Exception as e:
                logger.warning(f"多智能体系统初始化失败: {e}")
        
        # 网络情报智能体 (可选)
        self.web_intel = None
        if enable_web_search and api_key:
            try:
                self.web_intel = WebIntelligenceAgent(model, api_key)
                logger.info("网络情报系统已启用 (GPT Web Search)")
            except Exception as e:
                logger.warning(f"网络情报系统初始化失败: {e}")
        
        # 历史记录器
        self.recorder = AnalysisRecorder()
        
        logger.info("实时分析器初始化完成")
    
    def analyze(
        self,
        df: pd.DataFrame,
        balance: float,
        current_position: Optional[Dict] = None,
        symbol: str = "BTCUSDT"
    ) -> RealtimeAnalysis:
        """
        执行实时分析
        
        Args:
            df: K线数据
            balance: 账户余额
            current_position: 当前持仓 (如有)
            symbol: 交易对
            
        Returns:
            RealtimeAnalysis 完整分析结果
        """
        logger.info(f"开始实时分析 {symbol}...")
        
        # 1️⃣ 计算高级信号
        advanced_signal, atr = generate_advanced_signal(df, self.min_confirmations)
        
        # 2️⃣ 获取详细指标
        indicators = get_indicator_signals(df)
        
        # 3️⃣ 市场状态
        regime, regime_strength = identify_market_regime(df)
        
        # 4️⃣ 当前价格
        current_price = float(df["close"].iloc[-1])
        atr_percent = (atr / current_price * 100) if current_price > 0 and not pd.isna(atr) else 2
        
        # 5️⃣ 计算仓位和风控
        if advanced_signal.side != "FLAT":
            # 计算止损止盈
            if advanced_signal.side == "LONG":
                stop_loss = current_price - atr * 2
                take_profit = current_price + atr * 4
            else:
                stop_loss = current_price + atr * 2
                take_profit = current_price - atr * 4
            
            # 动态仓位计算
            position_plan = self.risk_manager.calculate_position(
                balance=balance,
                entry_price=current_price,
                stop_loss=stop_loss,
                signal_confidence=advanced_signal.confidence,
                market_regime=regime,
                atr=atr,
                atr_percent=atr_percent
            )
            
            # 入场优化
            entry_opt = optimize_entry(advanced_signal, current_price, atr)
        else:
            stop_loss = 0
            take_profit = 0
            position_plan = self.risk_manager._empty_position()
            entry_opt = {"action": "WAIT"}
        
        # 5. 获取网络情报 (如果启用)
        web_intelligence = {}
        if self.enable_web_search and self.web_intel:
            logger.info("🌐 获取网络情报...")
            try:
                web_intelligence = self.web_intel.get_comprehensive_analysis(symbol)
            except Exception as e:
                logger.error(f"网络情报获取失败: {e}")
                
        # 6. 多智能体决策 (如果启用)
        agent_decision = {}
        if self.multi_agent and advanced_signal.confidence >= 0.4:
            try:
                # 准备市场数据 (包含网络情报)
                market_data = self._prepare_market_data(
                    df, indicators, regime, current_price, atr, balance, symbol, web_intelligence
                )
                agent_result = self.multi_agent.analyze(market_data)
                agent_decision = agent_result.get("decision", {})
            except Exception as e:
                logger.warning(f"多智能体分析失败: {e}")
        
        # 8️⃣ 综合决策 (包含网络情报)
        final_action, final_reason, score = self._make_final_decision(
            advanced_signal, agent_decision, position_plan, current_position, web_intelligence
        )
        
        # 构建结果
        result = RealtimeAnalysis(
            signal=advanced_signal.side,
            confidence=advanced_signal.confidence,
            regime=regime,
            confirmations=advanced_signal.confirmations,
            indicators=indicators,
            position_size=position_plan.position_size,
            entry_price=entry_opt.get("ideal_entry", current_price),
            stop_loss=position_plan.stop_loss,
            take_profit=position_plan.take_profit,
            risk_reward=position_plan.risk_reward_ratio,
            agent_decision=agent_decision,
            web_intelligence=web_intelligence,
            action=final_action,
            reason=final_reason,
            score=score
        )
        
        self._log_analysis(result, symbol)
        
        # 记录历史
        self.recorder.record_analysis(result)
        
        return result
    
    def _prepare_market_data(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        regime: str,
        price: float,
        atr: float,
        balance: float,
        symbol: str,
        web_intelligence: Dict = None  # 新增参数
    ) -> Dict[str, Any]:
        """准备多智能体分析数据"""
        return {
            "symbol": symbol,
            "price": price,
            "ema_fast": indicators.get("ema_cross", {}).get("value", ""),
            "ema_slow": indicators.get("ema_cross", {}).get("value", ""),
            "atr": atr,
            "rsi": float(indicators.get("rsi", {}).get("value", "RSI=50").split("=")[1]) if "rsi" in indicators else 50,
            "macd": indicators.get("macd", {}).get("value", ""),
            "bollinger": indicators.get("bollinger", {}).get("value", ""),
            "volume_ratio": indicators.get("volume", {}).get("strength", 1.0),
            "adx": indicators.get("adx", {}).get("value", ""),
            "regime": regime,
            "balance": balance,
            "web_intelligence": web_intelligence or {},  # 传递网络情报
            "indicators_summary": {
                name: {
                    "signal": data.get("signal"),
                    "strength": data.get("strength")
                }
                for name, data in indicators.items()
            }
        }
    
    def _make_final_decision(
        self,
        signal,
        agent_decision: Dict,
        position_plan,
        current_position: Optional[Dict],
        web_intelligence: Dict = None
    ) -> tuple:
        """
        综合决策
        整合高级信号 + 多智能体意见 + 网络情报
        """
        score = 0
        reasons = []
        
        # 基础信号得分 (35分)
        if signal.side != "FLAT":
            score += signal.confidence * 35
            reasons.append(f"技术信号: {signal.side} (确认数: {signal.confirmations})")
        
        # 多智能体得分 (25分)
        if agent_decision:
            agent_action = agent_decision.get("action", "FLAT")
            agent_conf = agent_decision.get("confidence", 0)
            
            if agent_action == signal.side:
                score += agent_conf * 25
                reasons.append(f"多智能体确认: {agent_action}")
            elif agent_action == "FLAT":
                score -= 10
                reasons.append("多智能体建议观望")
        
        # 网络情报得分 (15分) - 新增
        if web_intelligence:
            overall = web_intelligence.get("overall_assessment", {})
            web_direction = overall.get("direction", "neutral")
            web_conf = overall.get("confidence", 0.5)
            
            if signal.side == "LONG" and web_direction == "bullish":
                score += web_conf * 15
                reasons.append("🌐 网络情报看多")
            elif signal.side == "SHORT" and web_direction == "bearish":
                score += web_conf * 15
                reasons.append("🌐 网络情报看空")
            elif web_direction == "neutral":
                pass  # 不加分
            else:
                score -= 5  # 网络情报与信号相反
                reasons.append(f"⚠️ 网络情报{web_direction}，与信号相反")
        
        # 仓位质量得分 (15分)
        score += position_plan.position_score * 15
        
        # 风险回报得分 (10分)
        if position_plan.risk_reward_ratio >= 2:
            score += 10
        elif position_plan.risk_reward_ratio >= 1.5:
            score += 5
        
        # 最终决策
        if score >= 70 and signal.side != "FLAT":
            action = signal.side
            reason = f"强信号 ({score:.0f}分): " + " | ".join(reasons)
        elif score >= 50 and signal.side != "FLAT":
            action = signal.side
            reason = f"中等信号 ({score:.0f}分): " + " | ".join(reasons)
        else:
            action = "FLAT"
            reason = f"信号不足 ({score:.0f}分): 建议观望"
        
        # 检查是否有持仓冲突
        if current_position:
            # 兼容字典和 SimulatedPosition 对象
            if hasattr(current_position, 'direction'):
                pos_side = current_position.direction  # SimulatedPosition
            elif isinstance(current_position, dict):
                pos_side = current_position.get("side", "")
            else:
                pos_side = ""
            if pos_side and pos_side != action and action != "FLAT":
                reason += f" [注意: 当前持有{pos_side}仓位]"
        
        return action, reason, score
    
    def analyze_multi_timeframe(
        self,
        mtf_data: Dict[str, pd.DataFrame],
        primary_interval: str = "15m"
    ) -> Dict[str, Any]:
        """
        多时间框架分析
        
        Args:
            mtf_data: {interval: DataFrame} 多时间框架数据
            primary_interval: 主时间框架
            
        Returns:
            多时间框架分析结果
        """
        results = {}
        
        for interval, df in mtf_data.items():
            if df is not None and len(df) >= 200:
                signal, _ = generate_advanced_signal(df, min_confirmations=2)
                results[interval] = {
                    "side": signal.side,
                    "confidence": signal.confidence,
                    "regime": signal.regime
                }
            else:
                results[interval] = {
                    "side": "FLAT",
                    "confidence": 0,
                    "regime": "UNKNOWN"
                }
        
        # 计算时间框架一致性
        sides = [r["side"] for r in results.values() if r["side"] != "FLAT"]
        
        if len(sides) >= 2:
            long_count = sides.count("LONG")
            short_count = sides.count("SHORT")
            total = len(sides)
            
            if long_count >= 2:
                consensus = "LONG"
                alignment = long_count / total
            elif short_count >= 2:
                consensus = "SHORT"
                alignment = short_count / total
            else:
                consensus = "FLAT"
                alignment = 0
        else:
            consensus = "FLAT"
            alignment = 0
        
        results["consensus"] = {
            "side": consensus,
            "alignment": round(alignment, 2),
            "aligned_count": max(sides.count("LONG"), sides.count("SHORT")) if sides else 0,
            "total_count": len(mtf_data),
            "description": f"{consensus} 在 {int(alignment * len(mtf_data))}/{len(mtf_data)} 时间框架确认"
        }
        
        logger.info(f"📊 多时间框架: {results['consensus']['description']}")
        
        return results
    
    def _log_analysis(self, result: RealtimeAnalysis, symbol: str):
        """记录分析结果"""
        logger.info("=" * 60)
        logger.info(f"📊 实时分析结果 - {symbol}")
        logger.info("=" * 60)
        logger.info(f"信号: {result.signal} | 置信度: {result.confidence:.2f}")
        logger.info(f"市场状态: {result.regime} | 确认指标: {result.confirmations}")
        
        # 多时间框架结果
        if result.multi_timeframe:
            mtf = result.multi_timeframe.get("consensus", {})
            logger.info(f"⏱️ 多时间框架: {mtf.get('description', 'N/A')}")
        
        logger.info(f"仓位: {result.position_size:.6f} | 止损: {result.stop_loss:.2f} | 止盈: {result.take_profit:.2f}")
        logger.info(f"风险回报比: {result.risk_reward:.2f}")
        logger.info(f"综合评分: {result.score:.0f}/100")
        logger.info(f"最终建议: {result.action}")
        logger.info(f"原因: {result.reason}")
        logger.info("=" * 60)


def create_realtime_analyzer(
    model: str = "gpt-4o-mini",
    enable_multi_agent: bool = True,
    enable_web_search: bool = True
) -> RealtimeAnalyzer:
    """创建实时分析器"""
    return RealtimeAnalyzer(
        model=model,
        enable_multi_agent=enable_multi_agent,
        enable_web_search=enable_web_search
    )
