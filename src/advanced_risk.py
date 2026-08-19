"""
高级风险管理模块
动态仓位管理 + 波动率自适应 + 凯利公式
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """风险配置"""
    max_risk_per_trade: float = 0.01      # 最大单笔风险 1%
    max_daily_risk: float = 0.03          # 最大日风险 3%
    max_open_positions: int = 3           # 最大持仓数
    max_correlation_exposure: float = 0.5 # 最大相关性敞口
    volatility_scaling: bool = True       # 是否使用波动率缩放
    kelly_fraction: float = 0.5           # 凯利比例 (半凯利更保守)


@dataclass
class PositionPlan:
    """仓位计划"""
    position_size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_percent: float
    leverage: float
    risk_reward_ratio: float
    position_score: float  # 仓位评分 0-1


class AdvancedRiskManager:
    """
    高级风险管理器
    特点:
    - 动态仓位调整
    - 波动率自适应
    - 凯利公式优化
    - 最大回撤控制
    """
    
    def __init__(self, profile: RiskProfile = None):
        self.profile = profile or RiskProfile()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_rate = 0.5  # 初始假设胜率
        self.avg_win = 1.0   # 平均盈利
        self.avg_loss = 1.0  # 平均亏损
        self.trade_history = []
    
    def calculate_position(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        market_regime: str,
        atr: float,
        atr_percent: float = None
    ) -> PositionPlan:
        """
        计算最优仓位
        
        Args:
            balance: 账户余额
            entry_price: 入场价格
            stop_loss: 止损价格
            signal_confidence: 信号置信度 (0-1)
            market_regime: 市场状态
            atr: ATR 值
            atr_percent: ATR 占价格百分比
        """
        # 1. 基础风险金额
        base_risk = balance * self.profile.max_risk_per_trade
        
        # 2. 根据置信度调整
        confidence_multiplier = self._get_confidence_multiplier(signal_confidence)
        
        # 3. 根据市场状态调整
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # 4. 波动率调整 (高波动降低仓位)
        if self.profile.volatility_scaling and atr_percent:
            volatility_multiplier = self._get_volatility_multiplier(atr_percent)
        else:
            volatility_multiplier = 1.0
        
        # 5. 凯利公式优化 (如果有足够历史数据)
        kelly_multiplier = self._get_kelly_multiplier()
        
        # 6. 日内风险限制
        daily_multiplier = self._get_daily_risk_multiplier(balance)
        
        # 综合调整
        adjusted_risk = (
            base_risk 
            * confidence_multiplier 
            * regime_multiplier 
            * volatility_multiplier 
            * kelly_multiplier
            * daily_multiplier
        )
        
        # 计算仓位
        price_risk = abs(entry_price - stop_loss)
        
        # 对于低价币，设置最小价格风险为入场价的 1%
        min_price_risk = entry_price * 0.01
        if price_risk < min_price_risk:
            price_risk = min_price_risk
            logger.warning(f"价格风险过小，调整为最小值: {price_risk:.6f}")
        
        if price_risk <= 0:
            logger.warning("无效的止损设置")
            return self._empty_position()
        
        position_size = adjusted_risk / price_risk
        
        # 仓位上限保护：最大仓位不超过账户余额的10倍（考虑杠杆）
        max_position_value = balance * 10
        max_position_size = max_position_value / entry_price if entry_price > 0 else 0
        if position_size > max_position_size:
            logger.warning(f"仓位过大，限制从 {position_size:.2f} 到 {max_position_size:.2f}")
            position_size = max_position_size
        
        position_value = position_size * entry_price
        leverage = position_value / balance if balance > 0 else 0
        
        # 止盈计算 (根据风险回报比)
        target_rr = self._get_target_risk_reward(market_regime, signal_confidence)
        if entry_price > stop_loss:  # LONG
            take_profit = entry_price + (price_risk * target_rr)
        else:  # SHORT
            take_profit = entry_price - (price_risk * target_rr)
        
        # 仓位评分
        position_score = self._calculate_position_score(
            signal_confidence, regime_multiplier, volatility_multiplier
        )
        
        # 动态精度：根据价格大小决定小数位数
        if entry_price < 0.1:
            price_precision = 6  # 低价币用6位小数
        elif entry_price < 10:
            price_precision = 4
        else:
            price_precision = 2
        
        plan = PositionPlan(
            position_size=round(position_size, 6),
            stop_loss=round(stop_loss, price_precision),
            take_profit=round(take_profit, price_precision),
            risk_amount=round(adjusted_risk, 2),
            risk_percent=round(adjusted_risk / balance * 100, 2) if balance > 0 else 0,
            leverage=round(leverage, 2),
            risk_reward_ratio=round(target_rr, 2),
            position_score=round(position_score, 2)
        )
        
        logger.info(
            f"仓位计划: Size={plan.position_size}, SL={plan.stop_loss}, "
            f"TP={plan.take_profit}, Risk={plan.risk_percent}%, RR={plan.risk_reward_ratio}"
        )
        
        return plan
    
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """置信度乘数"""
        if confidence >= 0.8:
            return 1.2  # 高置信度，增加仓位
        elif confidence >= 0.6:
            return 1.0
        elif confidence >= 0.4:
            return 0.7
        else:
            return 0.5  # 低置信度，减少仓位
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """市场状态乘数"""
        multipliers = {
            "TRENDING_UP": 1.2,    # 趋势市场可以增加仓位
            "TRENDING_DOWN": 1.2,
            "RANGING": 0.8,        # 震荡市场减少仓位
            "VOLATILE": 0.5,       # 高波动大幅减少仓位
        }
        return multipliers.get(regime, 0.8)
    
    def _get_volatility_multiplier(self, atr_percent: float) -> float:
        """波动率乘数"""
        # ATR 占价格比例越高，仓位越小
        if atr_percent > 5:
            return 0.4
        elif atr_percent > 3:
            return 0.6
        elif atr_percent > 2:
            return 0.8
        elif atr_percent > 1:
            return 1.0
        else:
            return 1.2  # 低波动可以增加仓位
    
    def _get_kelly_multiplier(self) -> float:
        """凯利公式乘数"""
        if len(self.trade_history) < 20:
            return 1.0  # 数据不足，使用默认
        
        # 计算凯利比例
        # Kelly% = W - [(1-W) / R]
        # W = 胜率, R = 盈亏比
        if self.avg_loss == 0:
            return 1.0
        
        r = self.avg_win / self.avg_loss
        kelly = self.win_rate - ((1 - self.win_rate) / r)
        
        # 使用半凯利 (更保守)
        kelly = kelly * self.profile.kelly_fraction
        
        # 限制范围
        kelly = max(0.2, min(kelly, 1.5))
        
        return kelly
    
    def _get_daily_risk_multiplier(self, balance: float) -> float:
        """日内风险限制"""
        if self.daily_pnl < 0:
            # 已经亏损，减少后续仓位
            daily_loss_percent = abs(self.daily_pnl) / balance
            if daily_loss_percent >= self.profile.max_daily_risk:
                return 0.0  # 停止交易
            elif daily_loss_percent >= self.profile.max_daily_risk * 0.7:
                return 0.5
            elif daily_loss_percent >= self.profile.max_daily_risk * 0.5:
                return 0.7
        return 1.0
    
    def _get_target_risk_reward(self, regime: str, confidence: float) -> float:
        """目标风险回报比"""
        base_rr = 2.0
        
        # 趋势市场可以追求更高 RR
        if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            base_rr = 2.5
        elif regime == "RANGING":
            base_rr = 1.5  # 震荡市场降低预期
        
        # 高置信度可以设置更高目标
        if confidence >= 0.8:
            base_rr *= 1.2
        
        return base_rr
    
    def _calculate_position_score(
        self, 
        confidence: float, 
        regime_mult: float, 
        vol_mult: float
    ) -> float:
        """计算仓位综合评分"""
        return confidence * regime_mult * vol_mult
    
    def _empty_position(self) -> PositionPlan:
        """返回空仓位"""
        return PositionPlan(
            position_size=0,
            stop_loss=0,
            take_profit=0,
            risk_amount=0,
            risk_percent=0,
            leverage=0,
            risk_reward_ratio=0,
            position_score=0
        )
    
    def update_trade_result(self, pnl: float, win: bool):
        """更新交易结果"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.trade_history.append({
            "pnl": pnl,
            "win": win
        })
        
        # 更新统计
        if len(self.trade_history) >= 10:
            wins = [t for t in self.trade_history if t["win"]]
            losses = [t for t in self.trade_history if not t["win"]]
            
            self.win_rate = len(wins) / len(self.trade_history)
            
            if wins:
                self.avg_win = sum(t["pnl"] for t in wins) / len(wins)
            if losses:
                self.avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
    
    def reset_daily(self):
        """重置日内统计"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def should_trade(self, balance: float) -> tuple:
        """
        是否应该交易
        Returns: (可以交易, 原因)
        """
        # 检查日内风险
        if balance > 0:
            daily_loss_percent = abs(self.daily_pnl) / balance if self.daily_pnl < 0 else 0
            if daily_loss_percent >= self.profile.max_daily_risk:
                return False, f"已达日内最大亏损 ({daily_loss_percent*100:.1f}%)"
        
        # 检查交易次数 (可选)
        if self.daily_trades >= 10:
            return False, "日内交易次数过多"
        
        return True, "可以交易"


# 创建默认风险管理器
default_risk_manager = AdvancedRiskManager()
