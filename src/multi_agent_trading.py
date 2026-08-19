"""
自建多智能体交易系统
集成 MCP 工具 + A2A 协议 + OpenAI GPT
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .mcp_tools import mcp_registry, MCPTool
from .a2a_protocol import (
    a2a_router, A2AMessage, MessageType, 
    AgentCard, AgentCapability
)

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    交易智能体基类
    使用 GPT-5.1 + MCP 工具 + A2A 通信
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        capabilities: List[AgentCapability],
        tools: List[str],  # MCP 工具名称列表
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.tool_names = tools
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        
        # 注册到 A2A 路由器
        self._register_to_a2a()
        
        logger.info(f"智能体 '{name}' 初始化完成")
    
    def _register_to_a2a(self):
        """注册到 A2A 网络"""
        card = AgentCard(
            agent_id=self.agent_id,
            name=self.name,
            description=self.role,
            capabilities=self.capabilities,
            supported_message_types=[
                MessageType.REQUEST,
                MessageType.RESPONSE,
                MessageType.CONSENSUS,
                MessageType.VOTE
            ]
        )
        a2a_router.register_agent(card, self._handle_message)
    
    def _handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """处理 A2A 消息"""
        if message.message_type == MessageType.REQUEST:
            # 处理分析请求
            result = self.analyze(message.content)
            return A2AMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type=MessageType.RESPONSE,
                content=result,
                correlation_id=message.id
            )
        
        elif message.message_type == MessageType.CONSENSUS:
            # 处理共识投票
            vote = self._vote(message.content)
            return A2AMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type=MessageType.VOTE,
                content={"vote": vote},
                correlation_id=message.id
            )
        
        return None
    
    def _vote(self, consensus_request: Dict[str, Any]) -> str:
        """参与共识投票"""
        # 默认实现：基于分析结果投票
        options = consensus_request.get("options", [])
        if options:
            return options[0]  # 子类应重写此方法
        return "FLAT"
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词 - 子类重写"""
        return f"你是{self.name}，{self.role}。"
    
    def _get_tools(self) -> List[Dict]:
        """获取 MCP 工具（OpenAI 格式）"""
        tools = []
        for name in self.tool_names:
            tool = mcp_registry.get_tool(name)
            if tool:
                tools.append(tool.to_openai_format())
        return tools
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析"""
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._format_context(context)}
            ]
            
            # 获取工具
            tools = self._get_tools()
            
            # 调用 LLM
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )
            
            # 处理工具调用
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                # 执行工具调用
                tool_results = self._execute_tool_calls(assistant_message.tool_calls)
                
                # 将工具结果加入对话
                messages.append(assistant_message.model_dump())
                for tool_call, result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                
                # 再次调用获取最终分析
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )
                content = final_response.choices[0].message.content
            else:
                content = assistant_message.content
            
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"智能体 '{self.name}' 分析失败: {e}")
            return {"error": str(e)}
    
    def _execute_tool_calls(self, tool_calls) -> List[tuple]:
        """执行 MCP 工具调用"""
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            try:
                import json
                args = json.loads(tool_call.function.arguments)
                result = mcp_registry.execute_tool(func_name, **args)
                results.append((tool_call, result))
                logger.debug(f"MCP 工具 '{func_name}' 执行成功")
            except Exception as e:
                results.append((tool_call, {"error": str(e)}))
                logger.error(f"MCP 工具 '{func_name}' 执行失败: {e}")
        return results
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """格式化上下文"""
        lines = ["请分析以下市场数据："]
        for key, value in context.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("\n请以 JSON 格式输出你的分析结果。")
        return "\n".join(lines)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """解析响应"""
        try:
            import json
            # 处理 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except (json.JSONDecodeError, ValueError, IndexError, KeyError):
            return {"raw_response": content}
    
    def send_to(self, target_agent: str, content: Dict[str, Any]) -> Optional[Dict]:
        """发送 A2A 消息给其他智能体"""
        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=target_agent,
            message_type=MessageType.REQUEST,
            content=content,
            requires_response=True
        )
        response = a2a_router.send_message(message)
        if response and isinstance(response, A2AMessage):
            return response.content
        return None


# ============== 专业智能体 ==============

class TechnicalAnalystAgent(TradingAgent):
    """技术分析智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(
            agent_id="technical_analyst",
            name="技术分析师",
            role="专业技术分析师，擅长图表分析和技术指标解读",
            capabilities=[AgentCapability.TECHNICAL_ANALYSIS, AgentCapability.TOOL_EXECUTION],
            tools=["analyze_indicators"],
            model=model,
            api_key=api_key
        )
    
    def _get_system_prompt(self) -> str:
        return """你是专业的加密货币技术分析师。

你的职责：
1. 分析价格趋势（使用 EMA 交叉）
2. 评估市场动量（使用 RSI）
3. 判断波动性（使用 ATR）
4. 给出交易信号（LONG/SHORT/FLAT）

使用 analyze_indicators 工具获取指标数据。

输出格式（JSON）：
{
    "signal": "LONG" | "SHORT" | "FLAT",
    "confidence": 0.0-1.0,
    "trend": "bullish" | "bearish" | "neutral",
    "key_levels": {"support": 价格, "resistance": 价格},
    "reason": "分析理由"
}"""

    def _vote(self, consensus_request: Dict[str, Any]) -> str:
        """基于技术分析投票"""
        topic = consensus_request.get("topic", "")
        if "direction" in topic.lower():
            # 如果有上下文数据，进行快速分析
            data = consensus_request.get("context", {})
            ema_fast = data.get("ema_fast", 0)
            ema_slow = data.get("ema_slow", 0)
            if ema_fast > ema_slow:
                return "LONG"
            elif ema_fast < ema_slow:
                return "SHORT"
        return "FLAT"


class SentimentAnalystAgent(TradingAgent):
    """市场情绪分析智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(
            agent_id="sentiment_analyst",
            name="情绪分析师",
            role="市场情绪专家，擅长解读市场心理和资金流向",
            capabilities=[AgentCapability.SENTIMENT_ANALYSIS, AgentCapability.TOOL_EXECUTION],
            tools=["assess_sentiment"],
            model=model,
            api_key=api_key
        )
    
    def _get_system_prompt(self) -> str:
        return """你是市场情绪分析专家。

你的职责：
1. 评估市场情绪（基于提供的新闻、恐慌指数、社交媒体数据）
2. 分析大户/鲸鱼动向（基于链上数据）
3. 识别市场极端情绪（FUD 或 FOMO）
4. 给出基于情绪的交易建议

【关键指令】
在调用 `assess_sentiment` 工具时，你必须从输入数据中提取 `volume_ratio` 并作为参数传入。
如果数据中包含 `funding_rate` 或 `long_short_ratio`，也请一并传入。

重要：你收到的输入数据包含最新的【网络搜索情报】，包括实时新闻、鲸鱼异动和社交情绪。请务必基于这些【具体信息】进行分析，不要泛泛而谈。引用具体的新闻或数据来支持你的观点。

输出格式（JSON）：
{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "fear_greed": "fear" | "neutral" | "greed",
    "contrarian_signal": "LONG" | "SHORT" | "NONE",
    "key_drivers": ["驱动因素1", "驱动因素2"],
    "warning": "警告信息（如有）",
    "reason": "分析理由（引用具体新闻/数据）"
}"""


class RiskManagerAgent(TradingAgent):
    """风险管理智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(
            agent_id="risk_manager",
            name="风险管理专家",
            role="保守的风险管理专家，确保资金安全",
            capabilities=[AgentCapability.RISK_MANAGEMENT, AgentCapability.TOOL_EXECUTION],
            tools=["calculate_position", "evaluate_risk"],
            model=model,
            api_key=api_key
        )
    
    def _get_system_prompt(self) -> str:
        return """你是风险管理专家，资金安全是首要任务。

你的职责：
1. 评估市场风险水平
2. 计算合适的仓位大小
3. 设定止损和止盈
4. 决定是否适合交易

使用 calculate_position 和 evaluate_risk 工具。

原则：
- 每笔交易风险不超过账户 1%
- 高波动时减少仓位
- 极端风险时建议观望

输出格式（JSON）：
{
    "trade_allowed": true | false,
    "risk_level": "low" | "moderate" | "high" | "extreme",
    "position_size": 数量,
    "stop_loss": 价格,
    "take_profit": 价格,
    "risk_reward_ratio": 比率,
    "reason": "风控理由"
}"""


class CoordinatorAgent(TradingAgent):
    """协调决策智能体"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        super().__init__(
            agent_id="coordinator",
            name="交易协调者",
            role="整合各专家意见，做出最终交易决策",
            capabilities=[AgentCapability.DECISION_MAKING],
            tools=[],
            model=model,
            api_key=api_key
        )
    
    def _get_system_prompt(self) -> str:
        return """你是交易决策协调者。

你的职责：
1. 整合技术分析、情绪分析、风险评估
2. 解决专家意见冲突
3. 做出最终交易决策

决策原则：
1. 技术+情绪一致 → 提高置信度
2. 存在分歧 → 优先风险管理意见
3. 风险过高 → 建议观望
4. 极端市场 → 可考虑反向信号

输出格式（JSON）：
{
    "action": "LONG" | "SHORT" | "FLAT",
    "confidence": 0.0-1.0,
    "position_size": 建议仓位,
    "entry_price": 入场价格范围,
    "stop_loss": 止损价,
    "take_profit": 止盈价,
    "reason": "决策理由",
    "dissent": "不同意见（如有）"
}"""
    
    def coordinate(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """协调各智能体分析结果"""
        context = {
            "task": "coordinate_decision",
            "technical_analysis": analyses.get("technical", {}),
            "sentiment_analysis": analyses.get("sentiment", {}),
            "risk_assessment": analyses.get("risk", {})
        }
        return self.analyze(context)


# ============== 多智能体系统 ==============

class MultiAgentTradingSystem:
    """
    多智能体交易系统
    集成 MCP + A2A + GPT-5.1
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.api_key = api_key
        
        # 创建智能体
        self.technical = TechnicalAnalystAgent(model, api_key)
        self.sentiment = SentimentAnalystAgent(model, api_key)
        self.risk = RiskManagerAgent(model, api_key)
        self.coordinator = CoordinatorAgent(model, api_key)
        
        logger.info("多智能体交易系统初始化完成")
    
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多智能体分析
        
        流程：
        1. 并行收集各专家分析 (可通过 A2A)
        2. 协调者整合决策
        3. 可选：共识投票
        """
        logger.info("开始多智能体分析...")
        
        # 1. 收集各智能体分析
        analyses = {}
        
        # 技术分析
        logger.info("  → 技术分析中...")
        analyses["technical"] = self.technical.analyze(market_data)
        
        # 情绪分析
        logger.info("  → 情绪分析中...")
        analyses["sentiment"] = self.sentiment.analyze(market_data)
        
        # 风险评估
        logger.info("  → 风险评估中...")
        risk_context = {**market_data, "balance": market_data.get("balance", 10000)}
        analyses["risk"] = self.risk.analyze(risk_context)
        
        # 2. 协调者整合决策
        logger.info("  → 协调决策中...")
        final_decision = self.coordinator.coordinate(analyses)
        
        # 3. 可选：A2A 共识投票
        if self._should_vote(analyses):
            logger.info("  → 执行共识投票...")
            consensus = a2a_router.request_consensus(
                from_agent="coordinator",
                topic="trading_direction",
                options=["LONG", "SHORT", "FLAT"]
            )
            final_decision["consensus"] = consensus
        
        logger.info(f"多智能体决策完成: {final_decision.get('action', 'UNKNOWN')}")
        
        return {
            "decision": final_decision,
            "analyses": analyses,
            "conversation": a2a_router.get_conversation_summary()
        }
    
    def _should_vote(self, analyses: Dict) -> bool:
        """判断是否需要共识投票"""
        # 当分析结果存在分歧时进行投票
        signals = []
        if "signal" in analyses.get("technical", {}):
            signals.append(analyses["technical"]["signal"])
        if "sentiment" in analyses.get("sentiment", {}):
            s = analyses["sentiment"]["sentiment"]
            signals.append("LONG" if s == "bullish" else "SHORT" if s == "bearish" else "FLAT")
        
        # 信号不一致时投票
        return len(set(signals)) > 1


# ============== 便捷函数 ==============

def create_trading_system(model: str = "gpt-4o-mini") -> MultiAgentTradingSystem:
    """创建多智能体交易系统"""
    return MultiAgentTradingSystem(model=model)
