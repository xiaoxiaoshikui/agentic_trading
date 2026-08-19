"""
A2A (Agent-to-Agent) 通信协议
自建实现，遵循 Google A2A 协议规范
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """A2A 消息类型"""
    REQUEST = "request"          # 请求分析
    RESPONSE = "response"        # 响应分析
    BROADCAST = "broadcast"      # 广播给所有智能体
    DELEGATE = "delegate"        # 委派任务
    CONSENSUS = "consensus"      # 共识请求
    VOTE = "vote"               # 投票


class AgentCapability(str, Enum):
    """智能体能力声明"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_MANAGEMENT = "risk_management"
    DECISION_MAKING = "decision_making"
    TOOL_EXECUTION = "tool_execution"


@dataclass
class A2AMessage:
    """
    A2A 消息格式
    遵循 Google A2A 协议规范
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""  # "all" 表示广播
    message_type: MessageType = MessageType.REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 关联请求ID
    requires_response: bool = False
    priority: int = 1  # 1-5, 5最高
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            from_agent=data.get("from", ""),
            to_agent=data.get("to", ""),
            message_type=MessageType(data.get("type", "request")),
            content=data.get("content", {}),
            correlation_id=data.get("correlation_id"),
            requires_response=data.get("requires_response", False),
            priority=data.get("priority", 1)
        )


@dataclass 
class AgentCard:
    """
    智能体名片 (A2A Discovery)
    用于智能体发现和能力声明
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    supported_message_types: List[MessageType]
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "supported_messages": [m.value for m in self.supported_message_types],
            "version": self.version
        }


class A2ARouter:
    """
    A2A 消息路由器
    管理智能体间的消息传递
    """
    
    def __init__(self):
        self.agents: Dict[str, "AgentCard"] = {}
        self.message_queue: List[A2AMessage] = []
        self.message_handlers: Dict[str, callable] = {}
        self.conversation_history: List[A2AMessage] = []
    
    def register_agent(self, card: AgentCard, handler: callable = None):
        """注册智能体"""
        self.agents[card.agent_id] = card
        if handler:
            self.message_handlers[card.agent_id] = handler
        logger.info(f"A2A: 智能体 '{card.name}' 已注册")
    
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.message_handlers:
                del self.message_handlers[agent_id]
    
    def discover_agents(self, capability: AgentCapability = None) -> List[AgentCard]:
        """发现智能体（可按能力筛选）"""
        if capability is None:
            return list(self.agents.values())
        return [
            card for card in self.agents.values() 
            if capability in card.capabilities
        ]
    
    def send_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """
        发送消息
        返回响应（如果需要）
        """
        self.conversation_history.append(message)
        logger.debug(f"A2A: {message.from_agent} -> {message.to_agent}: {message.message_type.value}")
        
        # 广播消息
        if message.to_agent == "all":
            responses = []
            for agent_id, handler in self.message_handlers.items():
                if agent_id != message.from_agent:
                    response = handler(message)
                    if response:
                        responses.append(response)
            return responses if responses else None
        
        # 单播消息
        if message.to_agent in self.message_handlers:
            handler = self.message_handlers[message.to_agent]
            response = handler(message)
            if response:
                self.conversation_history.append(response)
            return response
        
        logger.warning(f"A2A: 目标智能体 '{message.to_agent}' 未找到")
        return None
    
    def broadcast(self, from_agent: str, content: Dict[str, Any], 
                  msg_type: MessageType = MessageType.BROADCAST) -> List[A2AMessage]:
        """广播消息给所有智能体"""
        message = A2AMessage(
            from_agent=from_agent,
            to_agent="all",
            message_type=msg_type,
            content=content
        )
        return self.send_message(message) or []
    
    def request_consensus(self, from_agent: str, topic: str, 
                         options: List[str]) -> Dict[str, Any]:
        """
        请求共识投票
        返回投票结果
        """
        message = A2AMessage(
            from_agent=from_agent,
            to_agent="all",
            message_type=MessageType.CONSENSUS,
            content={
                "topic": topic,
                "options": options
            },
            requires_response=True
        )
        
        responses = self.send_message(message) or []
        
        # 统计投票
        votes = {}
        for response in responses:
            if isinstance(response, A2AMessage):
                vote = response.content.get("vote")
                if vote:
                    votes[vote] = votes.get(vote, 0) + 1
        
        # 确定结果
        if votes:
            winner = max(votes.items(), key=lambda x: x[1])
            return {
                "topic": topic,
                "result": winner[0],
                "votes": votes,
                "consensus_reached": winner[1] > len(responses) / 2
            }
        
        return {"topic": topic, "result": None, "consensus_reached": False}
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        if not self.conversation_history:
            return "No conversation yet"
        
        summary = []
        for msg in self.conversation_history[-10:]:  # 最近10条
            summary.append(f"[{msg.from_agent} -> {msg.to_agent}] {msg.message_type.value}")
        
        return "\n".join(summary)


# 全局路由器
a2a_router = A2ARouter()
