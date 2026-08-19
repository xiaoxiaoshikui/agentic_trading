"""
Orchestrator
============
Coordinates all Agents and manages workflow
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .market_analyst import MarketAnalyst, MarketState
from .risk_guardian import RiskGuardian, RiskAssessment, Position

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Trading Decision"""
    # Whether trading is allowed
    can_trade: bool = False
    
    # Recommended direction
    direction: str = "FLAT"  # LONG, SHORT, FLAT
    
    # Confidence level
    confidence: float = 0.0
    
    # Position size multiplier (0-1)
    position_multiplier: float = 1.0
    
    # Decision reasons
    reasons: list = field(default_factory=list)
    
    # Market state
    market_state: Optional[MarketState] = None
    
    # Risk assessment
    risk_assessment: Optional[RiskAssessment] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class Orchestrator:
    """
    Orchestrator
    
    Coordinates MarketAnalyst and RiskGuardian, outputs final trading decision
    """
    
    def __init__(self):
        self.market_analyst = MarketAnalyst()
        self.risk_guardian = RiskGuardian()
        self.logger = logging.getLogger("Orchestrator")
        
        # State
        self.last_decision: Optional[TradingDecision] = None
        self.decision_history: list = []
    
    def analyze(
        self,
        df,  # Kline data
        capital: float,
        position: Optional[Position] = None,
        funding_rate: float = 0.0,
        open_interest: float = 0.0,
        oi_change_24h: float = 0.0,
        recent_trades: Optional[list] = None
    ) -> TradingDecision:
        """
        Comprehensive analysis, output trading decision
        
        Args:
            df: Kline data
            capital: Current capital
            position: Current position
            funding_rate: Funding rate
            open_interest: Open interest
            oi_change_24h: 24h OI change rate
            recent_trades: Recent trade records
        
        Returns:
            TradingDecision: Final trading decision
        """
        decision = TradingDecision()
        
        try:
            # Step 1: Market analysis
            self.logger.info("Market Analyst analyzing...")
            market_state = self.market_analyst.process(
                df=df,
                funding_rate=funding_rate,
                open_interest=open_interest,
                oi_change_24h=oi_change_24h
            )
            decision.market_state = market_state
            
            # Step 2: Risk assessment
            self.logger.info("Risk Guardian assessing...")
            risk_assessment = self.risk_guardian.process(
                capital=capital,
                position=position,
                market_state=market_state,
                funding_rate=funding_rate,
                recent_trades=recent_trades
            )
            decision.risk_assessment = risk_assessment
            
            # Step 3: Make decision
            decision = self._make_decision(decision, market_state, risk_assessment)
            
            # Record history
            self.last_decision = decision
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
            # Log summary
            self._log_decision(decision)
            
        except Exception as e:
            self.logger.error("Analysis failed: %s", e)
            decision.can_trade = False
            decision.reasons.append(f"Analysis error: {str(e)}")
        
        return decision
    
    def _make_decision(
        self, 
        decision: TradingDecision,
        market_state: MarketState,
        risk_assessment: RiskAssessment
    ) -> TradingDecision:
        """Combine market analysis and risk assessment to make final decision"""
        
        # 1. Check if trading is allowed
        if not risk_assessment.can_open_position:
            decision.can_trade = False
            decision.direction = "FLAT"
            decision.reasons.append(f"Risk control: {risk_assessment.recommended_action}")
            return decision
        
        if market_state.regime == "crisis":
            decision.can_trade = False
            decision.direction = "FLAT"
            decision.reasons.append("Market in crisis mode")
            return decision
        
        # 2. Determine direction based on market score
        score = market_state.score
        
        if score > 0.3:
            decision.direction = "LONG"
            decision.confidence = min(abs(score), 0.9)
            decision.reasons.append(f"Bullish market (score: {score:.2f})")
        elif score < -0.3:
            decision.direction = "SHORT"
            decision.confidence = min(abs(score), 0.9)
            decision.reasons.append(f"Bearish market (score: {score:.2f})")
        else:
            decision.direction = "FLAT"
            decision.confidence = 0.0
            decision.reasons.append(f"Unclear signal (score: {score:.2f})")
        
        # 3. Apply risk multiplier
        decision.position_multiplier = risk_assessment.position_multiplier
        decision.reasons.append(f"Position multiplier: {decision.position_multiplier:.2f}")
        
        # 4. Add anomaly signals
        if market_state.anomalies:
            decision.reasons.extend(market_state.anomalies)
        
        # 5. Final judgment
        decision.can_trade = (
            decision.direction != "FLAT" and 
            decision.confidence > 0.3 and
            decision.position_multiplier > 0.1
        )
        
        return decision
    
    def _log_decision(self, decision: TradingDecision):
        """Log decision summary"""
        market = decision.market_state
        risk = decision.risk_assessment
        
        self.logger.info("=" * 50)
        self.logger.info("Trading Decision Summary")
        self.logger.info("=" * 50)
        
        if market:
            self.logger.info("Market: %s | Direction: %s | Score: %.2f", 
                           market.regime, market.direction, market.score)
        
        if risk:
            self.logger.info("Risk: %s | Score: %d | Position mult: %.2f",
                           risk.level, risk.score, risk.position_multiplier)
        
        self.logger.info("Decision: %s", "CAN TRADE" if decision.can_trade else "NO TRADE")
        self.logger.info("Direction: %s | Confidence: %.2f", decision.direction, decision.confidence)
        
        if decision.reasons:
            self.logger.info("Reasons:")
            for r in decision.reasons:
                self.logger.info("  - %s", r)
        
        self.logger.info("=" * 50)
    
    def update_trade_result(self, pnl: float, is_win: bool):
        """Update trade result (for risk stats)"""
        self.risk_guardian.update_daily_pnl(pnl)
        self.risk_guardian.record_trade_result(is_win)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "last_decision": self.last_decision,
            "risk_guardian": {
                "daily_pnl": self.risk_guardian.daily_pnl,
                "consecutive_losses": self.risk_guardian.consecutive_losses,
                "daily_trades": self.risk_guardian.daily_trades
            }
        }
