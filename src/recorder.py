"""
交易历史记录器
负责保存每一次的分析结果、GPT决策和后续的市场表现
用于后续的 AI 策略评估和优化
"""

import json
import os
from datetime import datetime
from typing import Dict, Any
from dataclasses import asdict

class AnalysisRecorder:
    def __init__(self, data_dir: str = "data/history"):
        self.data_dir = data_dir
        self._ensure_dir()
        self.current_file = self._get_log_filename()
    
    def _ensure_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _get_log_filename(self) -> str:
        """按日期生成日志文件名"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.data_dir, f"analysis_log_{date_str}.jsonl")
    
    def record_analysis(self, analysis_result: Any):
        """
        记录一次完整的分析
        
        Args:
            analysis_result: RealtimeAnalysis 对象
        """
        try:
            # 转换为字典
            if hasattr(analysis_result, '__dataclass_fields__'):
                data = asdict(analysis_result)
            else:
                data = analysis_result
            
            # 添加时间戳
            record = {
                "timestamp": datetime.now().isoformat(),
                "timestamp_ts": datetime.now().timestamp(),
                **data
            }
            
            # 写入文件 (JSONL 格式，一行一个 JSON)
            with open(self.current_file, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
                
        except Exception as e:
            print(f"❌ 记录分析历史失败: {e}")

    def load_history(self, date_str: str = None) -> list:
        """读取历史记录进行分析"""
        filename = self._get_log_filename() if not date_str else os.path.join(self.data_dir, f"analysis_log_{date_str}.jsonl")
        
        records = []
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
        return records
