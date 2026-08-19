import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StrategyMemory:
    """
    策略记忆库
    负责保存、检索和管理历史优秀策略
    """
    def __init__(self, filepath: str = "data/strategies_memory.json"):
        self.filepath = filepath
        self._ensure_file_exists()
        
    def _ensure_file_exists(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"strategies": []}, f, indent=2)

    def load_strategies(self) -> List[Dict]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("strategies", [])
        except Exception as e:
            logger.error(f"无法读取策略记忆: {e}")
            return []

    def save_strategy(self, code: str, performance: Dict, market_summary: str, tags: List[str] = None):
        """保存表现优秀的策略"""
        strategies = self.load_strategies()
        
        # 避免重复保存完全相同的代码
        for s in strategies:
            if s["code"] == code:
                # 更新表现数据
                if performance["win_rate"] > s["performance"]["win_rate"]:
                    s["performance"] = performance
                    s["updated_at"] = datetime.now().isoformat()
                    self._write_file(strategies)
                    logger.info("🧠 更新了现有策略的记忆记录")
                return

        new_entry = {
            "id": f"strat_{int(datetime.now().timestamp())}",
            "code": code,
            "performance": performance,
            "market_summary": market_summary,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }
        
        strategies.append(new_entry)
        # 只保留最近/最好的 50 个策略
        if len(strategies) > 50:
            strategies.sort(key=lambda x: x["performance"].get("win_rate", 0), reverse=True)
            strategies = strategies[:50]
            
        self._write_file(strategies)
        logger.info("🧠 已将当前策略存入记忆库")

    def _get_win_rate(self, strat: Dict) -> float:
        """安全获取胜率，兼容多种数据格式"""
        # 格式1: {"performance": {"win_rate": 0.6}}
        if "performance" in strat and isinstance(strat["performance"], dict):
            return strat["performance"].get("win_rate", 0)
        # 格式2: {"win_rate": 0.6} (黄金模板格式)
        if "win_rate" in strat:
            return strat["win_rate"]
        return 0
    
    def retrieve_relevant_strategy(self, current_market_summary: str) -> Optional[Dict]:
        """
        检索最相关的策略（目前简化为返回胜率最高的策略）
        未来可以用向量搜索做语义匹配
        """
        strategies = self.load_strategies()
        if not strategies:
            return None
            
        # 简单逻辑：返回胜率最高的策略（带容错）
        try:
            best_strat = max(strategies, key=lambda x: self._get_win_rate(x))
            return best_strat
        except Exception as e:
            logger.warning(f"检索策略失败: {e}")
            return strategies[0] if strategies else None

    def _write_file(self, strategies: List[Dict]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"strategies": strategies}, f, indent=2, ensure_ascii=False)
