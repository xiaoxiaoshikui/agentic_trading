"""
Agent Base Class and Message Definitions
"""

from dataclasses import dataclass, field
from typing import Any, Dict
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    sender: str
    topic: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0=normal, 1=important, 2=urgent


class AgentBase(ABC):
    """Base class for all Agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing logic, must be implemented by subclass"""
    
    def emit(self, topic: str, payload: Dict[str, Any], priority: int = 0) -> AgentMessage:
        """Emit a message"""
        msg = AgentMessage(
            sender=self.name,
            topic=topic,
            payload=payload,
            priority=priority
        )
        self.logger.debug("Emit [%s]: %s", topic, payload)
        return msg
    
    def log_info(self, message: str):
        self.logger.info("[%s] %s", self.name, message)
    
    def log_warning(self, message: str):
        self.logger.warning("[%s] %s", self.name, message)
    
    def log_error(self, message: str):
        self.logger.error("[%s] %s", self.name, message)
