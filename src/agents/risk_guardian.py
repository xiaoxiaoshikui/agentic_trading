"""
Risk Guardian Agent
===================
Responsibility: Real-time risk monitoring with emergency intervention
Authority: Can force close positions, halt trading
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from .base import AgentBase
from .market_analyst import MarketState
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Risk Assessment Report"""
    # Risk level
    level: str = "low"  # low, medium, high, critical
    
    # Risk score (0-100)
    score: int = 0
    
    # Current exposure
    exposure: float = 0.0  # 0.0-1.0 (percentage of total capital)
    
    # Warning messages
    warnings: List[str] = field(default_factory=list)
    
    # Recommended action
    recommended_action: str = "none"  # none, reduce_position, halt_new_trades, close_all
    
    # Allow opening position
    can_open_position: bool = True
    
    # Position size multiplier
    position_multiplier: float = 1.0  # 0.0-1.0
    
    # Detailed metrics
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Position:
    """Position Information"""
    symbol: str
    side: str  # LONG / SHORT
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: int
    entry_time: datetime


class RiskGuardian(AgentBase):
    """
    Risk Guardian Agent
    
    Core responsibilities:
    1. Assess current risk level
    2. Limit position size
    3. Emergency intervention when needed
    """
    
    def __init__(self):
        super().__init__("RiskGuardian")
        
        # Risk control parameters
        self.config = {
            # Max exposure (of total capital)
            "max_exposure": 0.3,
            
            # Max loss per trade
            "max_loss_per_trade": 0.02,  # 2%
            
            # Max daily loss
            "max_daily_loss": 0.05,  # 5%
            
            # Max consecutive losses
            "max_consecutive_losses": 3,
            
            # Max drawdown
            "max_drawdown": 0.15,  # 15%
            
            # Volatility adjustment
            "high_volatility_reduction": 0.5,  # Halve position in high volatility
            "crisis_mode_reduction": 0.0,  # No positions in crisis mode
            
            # Max holding time (hours)
            "max_holding_hours": 24,
            
            # Funding rate risk threshold
            "funding_rate_risk_threshold": 0.001,  # 0.1%
        }
        
        # State tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
    
    def process(
        self,
        capital: float,
        position: Optional[Position],
        market_state: MarketState,
        funding_rate: float = 0.0,
        recent_trades: Optional[List[Dict]] = None
    ) -> RiskAssessment:
        """
        Assess current risk
        
        Args:
            capital: Current capital
            position: Current position
            market_state: Market state (from MarketAnalyst)
            funding_rate: Current funding rate
            recent_trades: Recent trade records
        
        Returns:
            RiskAssessment: Risk assessment report
        """
        assessment = RiskAssessment()
        
        try:
            # Reset daily stats
            self._check_daily_reset()
            
            # Update trade stats
            if recent_trades:
                self._update_trade_stats(recent_trades)
            
            # 1. Calculate current exposure
            assessment.exposure = self._calculate_exposure(position, capital)
            assessment.metrics['exposure'] = assessment.exposure
            
            # 2. Check various risks
            risk_score = 0
            
            # 2.1 Market risk
            market_risk, market_warnings = self._assess_market_risk(market_state)
            risk_score += market_risk
            assessment.warnings.extend(market_warnings)
            
            # 2.2 Position risk
            position_risk, position_warnings = self._assess_position_risk(position, capital)
            risk_score += position_risk
            assessment.warnings.extend(position_warnings)
            
            # 2.3 Funding rate risk
            funding_risk, funding_warnings = self._assess_funding_risk(funding_rate, position)
            risk_score += funding_risk
            assessment.warnings.extend(funding_warnings)
            
            # 2.4 Historical performance risk
            performance_risk, performance_warnings = self._assess_performance_risk()
            risk_score += performance_risk
            assessment.warnings.extend(performance_warnings)
            
            # 3. Determine risk level
            assessment.score = min(risk_score, 100)
            assessment.level = self._score_to_level(assessment.score)
            
            # 4. Decide action
            assessment.recommended_action = self._decide_action(assessment)
            assessment.can_open_position = self._can_open_position(assessment, market_state)
            assessment.position_multiplier = self._calculate_position_multiplier(assessment, market_state)
            
            # Record metrics
            assessment.metrics.update({
                'risk_score': assessment.score,
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'market_regime': market_state.regime
            })
            
            self.log_info(f"Risk: {assessment.level} (score: {assessment.score}) | Position mult: {assessment.position_multiplier:.2f}")
            
            if assessment.warnings:
                for w in assessment.warnings:
                    self.log_warning(w)
        
        except Exception as e:
            self.log_error(f"Risk assessment failed: {e}")
            assessment.level = "high"
            assessment.can_open_position = False
            assessment.warnings.append(f"Assessment error: {str(e)}")
        
        return assessment
    
    def _check_daily_reset(self):
        """Check if daily stats need to be reset"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            self.log_info("Daily stats reset")
    
    def _update_trade_stats(self, recent_trades: List[Dict]):
        """Update trade statistics"""
        # Simplified: count recent consecutive losses
        losses = 0
        for trade in reversed(recent_trades[-10:]):
            if trade.get('pnl', 0) < 0:
                losses += 1
            else:
                break
        self.consecutive_losses = losses
    
    def _calculate_exposure(self, position: Optional[Position], capital: float) -> float:
        """Calculate current exposure"""
        if position is None or capital <= 0:
            return 0.0
        
        position_value = position.size * position.current_price
        exposure = position_value / capital
        return min(exposure, 1.0)
    
    def _assess_market_risk(self, market_state: MarketState) -> tuple:
        """Assess market risk"""
        risk = 0
        warnings = []
        
        # Market regime risk
        regime_risk = {
            "trending": 10,
            "ranging": 20,
            "volatile": 40,
            "crisis": 80
        }
        risk += regime_risk.get(market_state.regime, 30)
        
        if market_state.regime == "crisis":
            warnings.append("Market in crisis mode")
        elif market_state.regime == "volatile":
            warnings.append("High market volatility")
        
        # Anomaly signals
        for anomaly in market_state.anomalies:
            risk += 5
            warnings.append(anomaly)
        
        return risk, warnings
    
    def _assess_position_risk(self, position: Optional[Position], capital: float) -> tuple:
        """Assess position risk"""
        risk = 0
        warnings = []
        
        if position is None:
            return 0, []
        
        # Unrealized loss
        if position.unrealized_pnl < 0:
            loss_pct = abs(position.unrealized_pnl) / capital
            if loss_pct > 0.05:
                risk += 30
                warnings.append(f"Large unrealized loss: {loss_pct*100:.1f}%")
            elif loss_pct > 0.02:
                risk += 15
        
        # Holding time too long
        if position.entry_time:
            holding_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
            if holding_hours > self.config['max_holding_hours']:
                risk += 10
                warnings.append(f"Position held too long: {holding_hours:.1f}h")
        
        # Leverage too high
        if position.leverage > 10:
            risk += 20
            warnings.append(f"High leverage: {position.leverage}x")
        
        return risk, warnings
    
    def _assess_funding_risk(self, funding_rate: float, position: Optional[Position]) -> tuple:
        """Assess funding rate risk"""
        risk = 0
        warnings = []
        
        if position is None:
            return 0, []
        
        # Check if funding rate is adverse to position
        threshold = self.config['funding_rate_risk_threshold']
        
        if position.side == "LONG" and funding_rate > threshold:
            risk += 15
            warnings.append(f"Long but high funding: {funding_rate*100:.3f}%")
        elif position.side == "SHORT" and funding_rate < -threshold:
            risk += 15
            warnings.append(f"Short but negative funding: {funding_rate*100:.3f}%")
        
        return risk, warnings
    
    def _assess_performance_risk(self) -> tuple:
        """Assess historical performance risk"""
        risk = 0
        warnings = []
        
        # Daily loss
        if self.daily_pnl < -self.config['max_daily_loss']:
            risk += 40
            warnings.append(f"Large daily loss: {self.daily_pnl*100:.1f}%")
        
        # Consecutive losses
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            risk += 30
            warnings.append(f"Consecutive losses: {self.consecutive_losses}")
        
        return risk, warnings
    
    def _score_to_level(self, score: int) -> str:
        """Convert risk score to level"""
        if score >= 70:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 30:
            return "medium"
        else:
            return "low"
    
    def _decide_action(self, assessment: RiskAssessment) -> str:
        """Decide recommended action"""
        if assessment.level == "critical":
            return "close_all"
        elif assessment.level == "high":
            return "halt_new_trades"
        elif assessment.level == "medium":
            return "reduce_position"
        else:
            return "none"
    
    def _can_open_position(self, assessment: RiskAssessment, market_state: MarketState) -> bool:
        """Determine if opening position is allowed"""
        # No positions in crisis mode
        if market_state.regime == "crisis":
            return False
        
        # No positions in high risk
        if assessment.level in ["critical", "high"]:
            return False
        
        # Exposure at limit
        if assessment.exposure >= self.config['max_exposure']:
            return False
        
        return True
    
    def _calculate_position_multiplier(self, assessment: RiskAssessment, market_state: MarketState) -> float:
        """Calculate position size multiplier"""
        multiplier = 1.0
        
        # Adjust by market state
        if market_state.regime == "crisis":
            multiplier *= self.config['crisis_mode_reduction']
        elif market_state.regime == "volatile":
            multiplier *= self.config['high_volatility_reduction']
        
        # Adjust by risk level
        level_multiplier = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.3,
            "critical": 0.0
        }
        multiplier *= level_multiplier.get(assessment.level, 0.5)
        
        # Adjust by consecutive losses
        if self.consecutive_losses > 0:
            multiplier *= max(0.3, 1 - self.consecutive_losses * 0.2)
        
        return max(0.0, min(1.0, multiplier))
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily PnL"""
        self.daily_pnl += pnl_change
    
    def record_trade_result(self, is_win: bool):
        """Record trade result"""
        self.daily_trades += 1
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
