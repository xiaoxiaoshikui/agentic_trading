"""
Multi-Asset Trading Simulator
Supports simultaneous positions across multiple symbols
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Single position"""
    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    open_time: float
    allocated_capital: float  # Capital allocated to this position
    confidence: float = 0.0


@dataclass
class MultiAssetSimulator:
    """
    Multi-Asset Trading Simulator
    - Manages multiple positions simultaneously
    - Allocates capital across positions
    - Tracks PnL per asset and total portfolio
    - Portfolio-level risk controls
    """
    initial_capital: float = 10000.0
    max_positions: int = 5  # Maximum simultaneous positions
    capital_per_position: float = 0.2  # 20% of capital per position
    leverage: int = 10
    fee_rate: float = 0.0004
    max_hold_hours: int = 4
    
    # Portfolio Risk Controls
    max_long_exposure: float = 0.6   # Max 60% in long positions
    max_short_exposure: float = 0.6  # Max 60% in short positions
    max_daily_loss: float = 0.03     # Stop trading if daily loss > 3%
    max_single_asset_loss: float = 0.05  # Stop trading asset if loss > 5%

    # State
    capital: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict, init=False)
    total_trades: int = field(default=0, init=False)
    wins: int = field(default=0, init=False)
    losses: int = field(default=0, init=False)
    total_pnl: float = field(default=0.0, init=False)
    max_drawdown: float = field(default=0.0, init=False)
    peak_capital: float = field(init=False)
    trade_history: List[dict] = field(default_factory=list, init=False)
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict, init=False)
    daily_pnl: float = field(default=0.0, init=False)
    trading_halted: bool = field(default=False, init=False)
    halted_symbols: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.daily_pnl = 0.0
        self.trading_halted = False
        self.halted_symbols = []

    def get_available_capital(self) -> float:
        """Get capital available for new positions"""
        allocated = sum(p.allocated_capital for p in self.positions.values())
        return self.capital - allocated

    def get_position_capital(self) -> float:
        """Get capital to allocate per new position"""
        return self.capital * self.capital_per_position

    def get_exposure(self) -> dict:
        """Calculate current portfolio exposure"""
        long_exposure = 0.0
        short_exposure = 0.0
        
        for pos in self.positions.values():
            exposure = pos.allocated_capital / self.initial_capital
            if pos.direction == "LONG":
                long_exposure += exposure
            else:
                short_exposure += exposure
        
        return {
            "long": long_exposure,
            "short": short_exposure,
            "gross": long_exposure + short_exposure,
            "net": long_exposure - short_exposure
        }

    def can_open_position(self, symbol: str, direction: str = None) -> bool:
        """
        Check if we can open a new position with portfolio risk controls
        
        Args:
            symbol: Trading symbol
            direction: LONG or SHORT (optional, for exposure check)
        """
        # Basic checks
        if symbol in self.positions:
            return False  # Already have position in this symbol
        if len(self.positions) >= self.max_positions:
            return False  # Max positions reached
        if self.get_available_capital() < self.get_position_capital() * 0.5:
            return False  # Not enough capital
        
        # Portfolio risk checks
        if self.trading_halted:
            logger.warning("🛑 Trading halted due to daily loss limit")
            return False
        
        if symbol in self.halted_symbols:
            logger.warning(f"🛑 {symbol} halted due to excessive losses")
            return False
        
        # Check daily loss
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss:
            self.trading_halted = True
            logger.warning(f"🛑 Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        # Check single asset loss
        if symbol in self.pnl_by_symbol:
            if self.pnl_by_symbol[symbol] < -self.initial_capital * self.max_single_asset_loss:
                self.halted_symbols.append(symbol)
                logger.warning(f"🛑 {symbol} loss limit reached: ${self.pnl_by_symbol[symbol]:.2f}")
                return False
        
        # Check directional exposure (if direction provided)
        if direction:
            exposure = self.get_exposure()
            new_exposure = self.capital_per_position
            
            if direction == "LONG":
                if exposure["long"] + new_exposure > self.max_long_exposure:
                    logger.warning(f"🛑 Long exposure limit: {exposure['long']*100:.0f}% + {new_exposure*100:.0f}% > {self.max_long_exposure*100:.0f}%")
                    return False
            else:
                if exposure["short"] + new_exposure > self.max_short_exposure:
                    logger.warning(f"🛑 Short exposure limit: {exposure['short']*100:.0f}% + {new_exposure*100:.0f}% > {self.max_short_exposure*100:.0f}%")
                    return False
        
        return True

    def check_positions(self, prices: Dict[str, float], current_time: float) -> List[dict]:
        """
        Check all positions for stop loss, take profit, or timeout
        
        Args:
            prices: Dict of {symbol: current_price}
            current_time: Current timestamp
            
        Returns:
            List of closed position results
        """
        results = []
        symbols_to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]
            time_held = (current_time - pos.open_time) / 3600

            close_reason = None
            exit_price = current_price

            if pos.direction == "LONG":
                if current_price <= pos.stop_loss:
                    close_reason = "Stop Loss"
                    exit_price = pos.stop_loss
                elif current_price >= pos.take_profit:
                    close_reason = "Take Profit"
                    exit_price = pos.take_profit
            else:  # SHORT
                if current_price >= pos.stop_loss:
                    close_reason = "Stop Loss"
                    exit_price = pos.stop_loss
                elif current_price <= pos.take_profit:
                    close_reason = "Take Profit"
                    exit_price = pos.take_profit

            if time_held >= self.max_hold_hours and not close_reason:
                close_reason = f"Timeout ({self.max_hold_hours}h)"

            if close_reason:
                symbols_to_close.append((symbol, exit_price, close_reason))

        for symbol, exit_price, reason in symbols_to_close:
            result = self._close_position(symbol, exit_price, reason)
            if result:
                results.append(result)

        return results

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[dict]:
        """Close a position and calculate PnL"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # Calculate PnL
        if pos.direction == "LONG":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price

        gross_pnl = price_diff * pos.position_size

        # Fees
        notional_value = exit_price * pos.position_size
        total_fees = (pos.entry_price * pos.position_size + notional_value) * self.fee_rate
        net_pnl = gross_pnl - total_fees

        # Calculate percentage return
        pnl_percent = (net_pnl / pos.allocated_capital) * 100 if pos.allocated_capital > 0 else 0

        # Update state
        self.capital += net_pnl
        self.total_pnl += net_pnl
        self.daily_pnl += net_pnl  # Track daily PnL for circuit breaker
        self.total_trades += 1

        # Track PnL by symbol
        if symbol not in self.pnl_by_symbol:
            self.pnl_by_symbol[symbol] = 0
        self.pnl_by_symbol[symbol] += net_pnl

        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        # Update drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        drawdown = (self.peak_capital - self.capital) / self.peak_capital * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        result = {
            "symbol": symbol,
            "direction": pos.direction,
            "entry": pos.entry_price,
            "exit": exit_price,
            "size": pos.position_size,
            "pnl": net_pnl,
            "pnl_percent": pnl_percent,
            "reason": reason,
            "win": net_pnl > 0,
            "confidence": pos.confidence
        }

        self.trade_history.append(result)
        del self.positions[symbol]

        return result

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        current_time: float,
        confidence: float = 0.0
    ) -> bool:
        """Open a new position"""
        if not self.can_open_position(symbol):
            return False

        allocated = self.get_position_capital()

        self.positions[symbol] = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            open_time=current_time,
            allocated_capital=allocated,
            confidence=confidence
        )

        logger.info(f"🆕 [{symbol}] Opened {direction} @ ${entry_price:.2f} | Size: {position_size:.6f}")
        logger.info(f"   🛑 SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")
        return True

    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> Optional[dict]:
        """Manually close a position"""
        return self._close_position(symbol, exit_price, reason)

    def get_status(self) -> str:
        """Get portfolio status summary"""
        win_rate = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
        roi = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        pos_list = ", ".join([f"{s}:{p.direction}" for s, p in self.positions.items()]) or "None"

        return (
            f"💵 Capital: ${self.capital:,.2f} | ROI: {roi:+.2f}% | "
            f"Trades: {self.total_trades} | Win: {win_rate:.1f}% | "
            f"Positions: {len(self.positions)}/{self.max_positions} [{pos_list}]"
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in this symbol"""
        return symbol in self.positions

    def is_corrupted(self) -> bool:
        """Check for data corruption"""
        if math.isnan(self.capital) or math.isinf(self.capital):
            return True
        if self.capital < 0 or self.capital > self.initial_capital * 1000:
            return True
        return False

    def reset(self):
        """Reset simulator to initial state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        self.trade_history = []
        self.pnl_by_symbol = {}
        # Reset daily risk controls
        self.daily_pnl = 0.0
        self.trading_halted = False
        self.halted_symbols = []

    def get_report(self) -> dict:
        """Generate performance report"""
        if self.total_trades == 0:
            return {"error": "No trades"}

        win_rate = self.wins / self.total_trades
        roi = (self.capital - self.initial_capital) / self.initial_capital * 100

        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Best and worst
        pnls = [t['pnl'] for t in self.trade_history]
        
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "roi_percent": roi,
            "max_drawdown_percent": self.max_drawdown,
            "profit_factor": profit_factor,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "avg_trade": sum(pnls) / len(pnls) if pnls else 0,
            "pnl_by_symbol": self.pnl_by_symbol,
            "final_capital": self.capital
        }


def print_multi_asset_report(report: dict):
    """Print formatted report"""
    print("\n" + "=" * 60)
    print("         MULTI-ASSET BACKTEST REPORT")
    print("=" * 60)

    if "error" in report:
        print(f"Error: {report['error']}")
        return

    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades:  {report['total_trades']}")
    print(f"   Wins/Losses:   {report['wins']}/{report['losses']}")
    print(f"   Win Rate:      {report['win_rate']*100:.1f}%")

    print(f"\n💰 Performance:")
    print(f"   Total PnL:     ${report['total_pnl']:.2f} ({report['roi_percent']:.1f}%)")
    print(f"   Final Capital: ${report['final_capital']:.2f}")
    print(f"   Max Drawdown:  {report['max_drawdown_percent']:.1f}%")
    print(f"   Profit Factor: {report['profit_factor']:.2f}")

    print(f"\n📈 Trade Details:")
    print(f"   Best Trade:    ${report['best_trade']:.2f}")
    print(f"   Worst Trade:   ${report['worst_trade']:.2f}")
    print(f"   Avg Trade:     ${report['avg_trade']:.2f}")

    if report['pnl_by_symbol']:
        print(f"\n🏆 PnL by Symbol:")
        sorted_pnl = sorted(report['pnl_by_symbol'].items(), key=lambda x: x[1], reverse=True)
        for symbol, pnl in sorted_pnl:
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"   {emoji} {symbol}: ${pnl:+.2f}")

    # Rating
    score = 0
    if report['win_rate'] >= 0.45:
        score += 2
    if report['profit_factor'] >= 1.3:
        score += 2
    if report['max_drawdown_percent'] <= 25:
        score += 2
    if report['roi_percent'] > 0:
        score += 2

    ratings = {8: "⭐⭐⭐⭐⭐ EXCELLENT", 6: "⭐⭐⭐⭐ GOOD", 4: "⭐⭐⭐ AVERAGE", 2: "⭐⭐ BELOW AVG", 0: "⭐ POOR"}
    print("\n" + "=" * 60)
    for threshold, rating in sorted(ratings.items(), reverse=True):
        if score >= threshold:
            print(f"RATING: {rating} ({score}/8)")
            break
    print("=" * 60)
