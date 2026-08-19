"""
数据工具模块
K线数据处理
"""

from typing import List
import pandas as pd


def klines_to_df(klines: List[list]) -> pd.DataFrame:
    """
    将币安 K 线数据转换为 DataFrame
    
    Args:
        klines: 币安 API 返回的 K 线数据
        
    Returns:
        处理后的 DataFrame，包含 open, high, low, close, volume 列
    """
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df