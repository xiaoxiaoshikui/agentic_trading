"""
MCP 工具层
遵循 Model Context Protocol 标准定义交易工具
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Callable
import json


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        return self.function(**kwargs)


class MCPToolRegistry:
    """MCP 工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self._register_trading_tools()
    
    def _register_trading_tools(self):
        """注册交易相关工具"""
        
        # 1. 技术指标分析工具
        self.register(MCPTool(
            name="analyze_indicators",
            description="分析技术指标，返回趋势方向和信号强度",
            parameters={
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "当前价格"},
                    "ema_fast": {"type": "number", "description": "快速EMA值"},
                    "ema_slow": {"type": "number", "description": "慢速EMA值"},
                    "atr": {"type": "number", "description": "ATR值"},
                    "rsi": {"type": "number", "description": "RSI值", "default": 50}
                },
                "required": ["price", "ema_fast", "ema_slow", "atr"]
            },
            function=self._analyze_indicators
        ))
        
        # 2. 仓位计算工具
        self.register(MCPTool(
            name="calculate_position",
            description="根据风险参数计算仓位大小",
            parameters={
                "type": "object",
                "properties": {
                    "balance": {"type": "number", "description": "账户余额"},
                    "risk_percent": {"type": "number", "description": "风险比例(0-1)"},
                    "entry_price": {"type": "number", "description": "入场价格"},
                    "stop_loss": {"type": "number", "description": "止损价格"}
                },
                "required": ["balance", "risk_percent", "entry_price", "stop_loss"]
            },
            function=self._calculate_position
        ))
        
        # 3. 市场情绪工具
        self.register(MCPTool(
            name="assess_sentiment",
            description="评估市场情绪状态",
            parameters={
                "type": "object",
                "properties": {
                    "volume_ratio": {"type": "number", "description": "成交量比率"},
                    "funding_rate": {"type": "number", "description": "资金费率", "default": 0},
                    "long_short_ratio": {"type": "number", "description": "多空比", "default": 1}
                },
                "required": ["volume_ratio"]
            },
            function=self._assess_sentiment
        ))
        
        # 4. 风险评估工具
        self.register(MCPTool(
            name="evaluate_risk",
            description="评估当前市场风险水平",
            parameters={
                "type": "object",
                "properties": {
                    "atr": {"type": "number", "description": "ATR值"},
                    "price": {"type": "number", "description": "当前价格"},
                    "recent_volatility": {"type": "number", "description": "近期波动率", "default": 0}
                },
                "required": ["atr", "price"]
            },
            function=self._evaluate_risk
        ))
    
    def register(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> MCPTool:
        """获取工具"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[MCPTool]:
        """获取所有工具"""
        return list(self.tools.values())
    
    def get_openai_tools(self) -> List[Dict]:
        """获取 OpenAI 格式的工具列表"""
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """执行指定工具"""
        tool = self.get_tool(name)
        if tool:
            return tool.execute(**kwargs)
        return {"error": f"Tool {name} not found"}
    
    # ============== 工具实现 ==============
    
    @staticmethod
    def _analyze_indicators(price: float, ema_fast: float, ema_slow: float, 
                           atr: float, rsi: float = 50) -> Dict[str, Any]:
        """技术指标分析"""
        # 趋势判断
        if ema_fast > ema_slow * 1.001:
            trend = "bullish"
            trend_strength = min((ema_fast - ema_slow) / ema_slow * 100, 10)
        elif ema_fast < ema_slow * 0.999:
            trend = "bearish"
            trend_strength = min((ema_slow - ema_fast) / ema_slow * 100, 10)
        else:
            trend = "neutral"
            trend_strength = 0
        
        # RSI 信号
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
        
        # 波动率
        volatility = (atr / price * 100) if price > 0 else 0
        volatility_level = "high" if volatility > 3 else "low" if volatility < 1 else "normal"
        
        # 综合信号
        signal = "LONG" if trend == "bullish" and rsi_signal != "overbought" else \
                 "SHORT" if trend == "bearish" and rsi_signal != "oversold" else "FLAT"
        
        return {
            "trend": trend,
            "trend_strength": round(trend_strength, 2),
            "rsi_signal": rsi_signal,
            "volatility": round(volatility, 2),
            "volatility_level": volatility_level,
            "signal": signal
        }
    
    @staticmethod
    def _calculate_position(balance: float, risk_percent: float, 
                           entry_price: float, stop_loss: float) -> Dict[str, Any]:
        """仓位计算"""
        risk_amount = balance * risk_percent
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk <= 0:
            return {"error": "Invalid stop loss", "position_size": 0}
        
        position_size = risk_amount / price_risk
        position_value = position_size * entry_price
        leverage_used = position_value / balance if balance > 0 else 0
        
        return {
            "position_size": round(position_size, 6),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "leverage_used": round(leverage_used, 2),
            "risk_reward_info": "Based on 1% risk per trade"
        }
    
    @staticmethod
    def _assess_sentiment(volume_ratio: float, funding_rate: float = 0, 
                         long_short_ratio: float = 1) -> Dict[str, Any]:
        """市场情绪评估"""
        # 成交量信号
        if volume_ratio > 2:
            volume_signal = "extremely_high"
        elif volume_ratio > 1.5:
            volume_signal = "high"
        elif volume_ratio < 0.5:
            volume_signal = "low"
        else:
            volume_signal = "normal"
        
        # 资金费率信号
        if funding_rate > 0.05:
            funding_signal = "extremely_bullish_crowd"
        elif funding_rate > 0.01:
            funding_signal = "bullish_crowd"
        elif funding_rate < -0.01:
            funding_signal = "bearish_crowd"
        else:
            funding_signal = "neutral"
        
        # 多空比信号
        if long_short_ratio > 2:
            crowd = "extremely_long"
            contrarian = "SHORT"
        elif long_short_ratio < 0.5:
            crowd = "extremely_short"
            contrarian = "LONG"
        else:
            crowd = "balanced"
            contrarian = "NONE"
        
        # 整体情绪
        sentiment = "fear" if volume_signal == "extremely_high" and funding_rate < 0 else \
                    "greed" if funding_rate > 0.03 else "neutral"
        
        return {
            "volume_signal": volume_signal,
            "funding_signal": funding_signal,
            "crowd_position": crowd,
            "contrarian_signal": contrarian,
            "overall_sentiment": sentiment
        }
    
    @staticmethod
    def _evaluate_risk(atr: float, price: float, recent_volatility: float = 0) -> Dict[str, Any]:
        """风险评估"""
        # ATR 相对波动率
        atr_percent = (atr / price * 100) if price > 0 else 0
        
        # 风险等级
        if atr_percent > 5:
            risk_level = "extreme"
            recommended_position = 0.5  # 减半仓位
        elif atr_percent > 3:
            risk_level = "high"
            recommended_position = 0.7
        elif atr_percent > 1.5:
            risk_level = "moderate"
            recommended_position = 1.0
        else:
            risk_level = "low"
            recommended_position = 1.0
        
        # 建议止损距离
        recommended_sl_atr = 2.0 if risk_level in ["extreme", "high"] else 1.5
        recommended_tp_atr = recommended_sl_atr * 2  # 1:2 风险回报
        
        return {
            "risk_level": risk_level,
            "atr_percent": round(atr_percent, 2),
            "position_multiplier": recommended_position,
            "recommended_sl_atr": recommended_sl_atr,
            "recommended_tp_atr": recommended_tp_atr,
            "trade_allowed": risk_level != "extreme"
        }


# 全局工具注册表
mcp_registry = MCPToolRegistry()
