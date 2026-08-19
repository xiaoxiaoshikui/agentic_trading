"""
Multi-Agent System for LLM Strategy Evolution
=============================================
MarketAnalyst provides context for LLM to evolve better alpha strategies.
Risk control is handled by AdvancedRiskManager (not duplicated here).
"""

from .base import AgentBase, AgentMessage
from .market_analyst import MarketAnalyst, MarketState

__all__ = [
    "AgentBase",
    "AgentMessage",
    "MarketAnalyst",
    "MarketState",
]
