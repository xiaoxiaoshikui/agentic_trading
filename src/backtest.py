"""
回测系统
验证策略有效性
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from .advanced_strategy import generate_advanced_signal, AdvancedSignal
from .advanced_risk import AdvancedRiskManager, RiskProfile

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    entry_time: datetime
    exit_time: datetime = None
    side: str = ""
    entry_price: float = 0
    exit_price: float = 0
    position_size: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    pnl: float = 0
    pnl_percent: float = 0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    total_pnl: float = 0
    total_pnl_percent: float = 0
    max_drawdown: float = 0
    max_drawdown_percent: float = 0
    sharpe_ratio: float = 0
    profit_factor: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    avg_trade: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    avg_holding_time: float = 0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """
    策略回测器
    """
    
    def __init__(
        self,
        initial_balance: float = 10000,
        commission: float = 0.001,  # 0.1% 手续费
        slippage: float = 0.0005,   # 0.05% 滑点
        risk_profile: RiskProfile = None
    ):
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = AdvancedRiskManager(risk_profile or RiskProfile())
    
    def run(
        self,
        df: pd.DataFrame,
        min_confirmations: int = 3
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            df: K线数据 (需包含 open, high, low, close, volume)
            min_confirmations: 最小确认指标数
            
        Returns:
            回测结果
        """
        if len(df) < 300:
            logger.error("数据不足以进行回测")
            return BacktestResult()
        
        # 初始化
        balance = self.initial_balance
        position = None  # 当前持仓
        trades = []
        equity_curve = [balance]
        peak_balance = balance
        max_drawdown = 0
        
        # 遍历数据
        for i in range(200, len(df)):
            current_data = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            current_price = float(current_bar["close"])
            current_time = current_bar.name if isinstance(current_bar.name, datetime) else datetime.now()
            
            # 检查是否有持仓需要管理
            if position:
                # 检查止损
                if position["side"] == "LONG":
                    if float(current_bar["low"]) <= position["stop_loss"]:
                        # 触发止损
                        exit_price = position["stop_loss"] * (1 - self.slippage)
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                        pnl -= abs(pnl) * self.commission
                        
                        trade = Trade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            side="LONG",
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            position_size=position["size"],
                            stop_loss=position["stop_loss"],
                            take_profit=position["take_profit"],
                            pnl=pnl,
                            pnl_percent=pnl / balance * 100,
                            exit_reason="stop_loss"
                        )
                        trades.append(trade)
                        balance += pnl
                        position = None
                        
                    elif float(current_bar["high"]) >= position["take_profit"]:
                        # 触发止盈
                        exit_price = position["take_profit"] * (1 - self.slippage)
                        pnl = (exit_price - position["entry_price"]) * position["size"]
                        pnl -= abs(pnl) * self.commission
                        
                        trade = Trade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            side="LONG",
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            position_size=position["size"],
                            stop_loss=position["stop_loss"],
                            take_profit=position["take_profit"],
                            pnl=pnl,
                            pnl_percent=pnl / balance * 100,
                            exit_reason="take_profit"
                        )
                        trades.append(trade)
                        balance += pnl
                        position = None
                        
                elif position["side"] == "SHORT":
                    if float(current_bar["high"]) >= position["stop_loss"]:
                        # 触发止损
                        exit_price = position["stop_loss"] * (1 + self.slippage)
                        pnl = (position["entry_price"] - exit_price) * position["size"]
                        pnl -= abs(pnl) * self.commission
                        
                        trade = Trade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            side="SHORT",
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            position_size=position["size"],
                            stop_loss=position["stop_loss"],
                            take_profit=position["take_profit"],
                            pnl=pnl,
                            pnl_percent=pnl / balance * 100,
                            exit_reason="stop_loss"
                        )
                        trades.append(trade)
                        balance += pnl
                        position = None
                        
                    elif float(current_bar["low"]) <= position["take_profit"]:
                        # 触发止盈
                        exit_price = position["take_profit"] * (1 + self.slippage)
                        pnl = (position["entry_price"] - exit_price) * position["size"]
                        pnl -= abs(pnl) * self.commission
                        
                        trade = Trade(
                            entry_time=position["entry_time"],
                            exit_time=current_time,
                            side="SHORT",
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            position_size=position["size"],
                            stop_loss=position["stop_loss"],
                            take_profit=position["take_profit"],
                            pnl=pnl,
                            pnl_percent=pnl / balance * 100,
                            exit_reason="take_profit"
                        )
                        trades.append(trade)
                        balance += pnl
                        position = None
            
            # 没有持仓时，检查新信号
            if position is None:
                signal, atr = generate_advanced_signal(current_data, min_confirmations)
                
                if signal.side != "FLAT" and signal.confidence >= 0.5:
                    # 计算仓位
                    atr_percent = atr / current_price * 100 if current_price > 0 else 2
                    
                    if signal.side == "LONG":
                        stop_loss = current_price - atr * 2
                        take_profit = current_price + atr * 4
                    else:
                        stop_loss = current_price + atr * 2
                        take_profit = current_price - atr * 4
                    
                    plan = self.risk_manager.calculate_position(
                        balance=balance,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        signal_confidence=signal.confidence,
                        market_regime=signal.regime,
                        atr=atr,
                        atr_percent=atr_percent
                    )
                    
                    if plan.position_size > 0:
                        # 开仓
                        entry_price = current_price * (1 + self.slippage) if signal.side == "LONG" else current_price * (1 - self.slippage)
                        
                        position = {
                            "side": signal.side,
                            "entry_price": entry_price,
                            "size": plan.position_size,
                            "stop_loss": plan.stop_loss,
                            "take_profit": plan.take_profit,
                            "entry_time": current_time
                        }
            
            # 记录权益
            equity_curve.append(balance)
            
            # 计算回撤
            if balance > peak_balance:
                peak_balance = balance
            drawdown = peak_balance - balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算统计数据
        result = self._calculate_statistics(trades, equity_curve, max_drawdown)
        
        return result
    
    def _calculate_statistics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        max_drawdown: float
    ) -> BacktestResult:
        """计算回测统计"""
        if not trades:
            return BacktestResult(equity_curve=equity_curve)
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        
        # 盈利因子
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 夏普比率 (简化版)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        
        # 平均持仓时间
        holding_times = []
        for t in trades:
            if t.exit_time and t.entry_time:
                delta = t.exit_time - t.entry_time
                holding_times.append(delta.total_seconds() / 3600)  # 小时
        avg_holding = np.mean(holding_times) if holding_times else 0
        
        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades) if trades else 0,
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl / self.initial_balance * 100, 2),
            max_drawdown=round(max_drawdown, 2),
            max_drawdown_percent=round(max_drawdown / self.initial_balance * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            avg_win=round(np.mean([t.pnl for t in winning_trades]), 2) if winning_trades else 0,
            avg_loss=round(np.mean([t.pnl for t in losing_trades]), 2) if losing_trades else 0,
            avg_trade=round(np.mean([t.pnl for t in trades]), 2),
            best_trade=round(max(t.pnl for t in trades), 2),
            worst_trade=round(min(t.pnl for t in trades), 2),
            avg_holding_time=round(avg_holding, 2),
            trades=trades,
            equity_curve=equity_curve
        )


def print_backtest_report(result: BacktestResult):
    """打印回测报告"""
    print("\n" + "="*60)
    print("                    回测报告")
    print("="*60)
    
    print(f"\n📊 交易统计:")
    print(f"   总交易次数: {result.total_trades}")
    print(f"   盈利交易: {result.winning_trades}")
    print(f"   亏损交易: {result.losing_trades}")
    print(f"   胜率: {result.win_rate*100:.1f}%")
    
    print(f"\n💰 收益分析:")
    print(f"   总收益: ${result.total_pnl:.2f} ({result.total_pnl_percent:.1f}%)")
    print(f"   最大回撤: ${result.max_drawdown:.2f} ({result.max_drawdown_percent:.1f}%)")
    print(f"   盈利因子: {result.profit_factor:.2f}")
    print(f"   夏普比率: {result.sharpe_ratio:.2f}")
    
    print(f"\n📈 交易明细:")
    print(f"   平均盈利: ${result.avg_win:.2f}")
    print(f"   平均亏损: ${result.avg_loss:.2f}")
    print(f"   平均交易: ${result.avg_trade:.2f}")
    print(f"   最佳交易: ${result.best_trade:.2f}")
    print(f"   最差交易: ${result.worst_trade:.2f}")
    print(f"   平均持仓: {result.avg_holding_time:.1f} 小时")
    
    print("\n" + "="*60)
    
    # 评级
    score = 0
    if result.win_rate >= 0.5:
        score += 2
    if result.profit_factor >= 1.5:
        score += 2
    if result.sharpe_ratio >= 1.0:
        score += 2
    if result.max_drawdown_percent <= 20:
        score += 2
    if result.total_pnl_percent > 0:
        score += 2
    
    ratings = {10: "⭐⭐⭐⭐⭐ 优秀", 8: "⭐⭐⭐⭐ 良好", 6: "⭐⭐⭐ 中等", 4: "⭐⭐ 一般", 0: "⭐ 需改进"}
    for threshold, rating in sorted(ratings.items(), reverse=True):
        if score >= threshold:
            print(f"\n策略评级: {rating}")
            break
    
    print("="*60 + "\n")
