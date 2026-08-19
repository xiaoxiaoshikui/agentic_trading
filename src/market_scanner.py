import logging
from typing import List, Dict, Any
from binance.client import Client
import pandas as pd

logger = logging.getLogger(__name__)

class MarketScanner:
    """
    市场扫描器 - 自动寻找交易机会
    策略:
    1. 过滤掉非 USDT 交易对
    2. 过滤掉稳定币 (USDC, FDUSD, DAI)
    3. 过滤掉杠杆代币 (UP, DOWN)
    4. 评分 = 24h成交量 * abs(24h涨跌幅)
    5. 返回评分最高的 Top N 币种
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)
        
    def get_top_opportunities(self, limit: int = 5, min_volume_usdt: float = 50000000) -> List[Dict[str, Any]]:
        """
        获取最佳交易机会
        
        Args:
            limit: 返回数量
            min_volume_usdt: 最小 24h 成交额 (USDT)
            
        Returns:
            List[Dict]: 包含 symbol, price, change, volume, score
        """
        try:
            # 获取所有期货交易对的 24h 统计数据
            tickers = self.client.futures_ticker()
            
            candidates = []
            
            # 黑名单
            blacklist = ["USDCUSDT", "FDUSDUSDT", "DAIUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT"]
            
            for t in tickers:
                symbol = t['symbol']
                
                # 1. 必须是 USDT 本位
                if not symbol.endswith("USDT"):
                    continue
                    
                # 2. 过滤黑名单
                if symbol in blacklist:
                    continue
                    
                # 3. 过滤杠杆代币 (通常包含 UP 或 DOWN，且不是普通代币)
                # 简单过滤: 如果包含 UP/DOWN 且不在白名单内 (这里简化处理，手动观察通常带 UPUSDT)
                if "UPUSDT" in symbol or "DOWNUSDT" in symbol:
                    continue
                
                quote_volume = float(t['quoteVolume'])  # 24h 成交额
                price_change_percent = float(t['priceChangePercent'])
                last_price = float(t['lastPrice'])
                
                # 4. 最小流动性过滤
                if quote_volume < min_volume_usdt:
                    continue
                    
                # 5. 计算原始评分: 波动率 * 流动性
                vol_in_million = quote_volume / 1000000
                raw_score = vol_in_million * abs(price_change_percent)
                
                candidates.append({
                    "symbol": symbol,
                    "price": last_price,
                    "change_percent": price_change_percent,
                    "volume_24h": quote_volume,
                    "raw_score": raw_score
                })
                
            # 按原始评分降序排序，先取前20名进行精细筛选
            df = pd.DataFrame(candidates)
            if df.empty:
                logger.warning("没有找到符合条件的交易对")
                return []
                
            df = df.sort_values(by="raw_score", ascending=False).head(20)
            
            logger.info("正在获取 Top 20 币种的 1h K线数据以确认短期爆发力...")
            
            final_candidates = []
            
            for index, row in df.iterrows():
                symbol = row['symbol']
                try:
                    # 获取最近的 1h K线 (取最近2根即可，因为要看这1小时的变化)
                    # K线格式: [Open time, Open, High, Low, Close, Volume, ...]
                    klines = self.client.futures_klines(symbol=symbol, interval="1h", limit=2)
                    
                    if not klines or len(klines) < 2:
                        continue
                        
                    #上一根K线（已完成的1小时）
                    last_k = klines[-2] 
                    # 当前K线（正在进行的）
                    curr_k = klines[-1]
                    
                    # 价格
                    price_1h_ago = float(last_k[4]) # 上一根收盘价
                    current_price = float(curr_k[4]) # 当前价
                    
                    # 计算 1h 涨跌幅
                    change_1h = (current_price - price_1h_ago) / price_1h_ago
                    
                    # 计算 1h 波动率 (High - Low) / Low
                    high_1h = float(curr_k[2])
                    low_1h = float(curr_k[3])
                    volatility_1h = (high_1h - low_1h) / low_1h if low_1h > 0 else 0
                    
                    # 新评分: 侧重短期爆发
                    # 只有上涨的才给正分（如果你只做多）
                    # Score = 1h涨幅 * 波动率 * 原始热度(权重降低)
                    
                    # 这里给个简单的逻辑: 
                    # 如果 1h 是跌的，直接降分
                    if change_1h < 0:
                        short_term_score = 0
                    else:
                        # 爆发力 = 涨幅 * 波动率
                        short_term_score = (change_1h * 100) * (volatility_1h * 100)
                    
                    # 综合评分: 30% 看 24h热度, 70% 看 1h爆发力
                    # 为了量级统一，这里做个简单处理
                    # row['raw_score'] 通常很大，normalize 一下
                    
                    final_candidates.append({
                        "symbol": symbol,
                        "price": current_price,
                        "change_percent_24h": row['change_percent'],
                        "change_percent_1h": change_1h * 100,
                        "volume_24h": row['volume_24h'],
                        "short_term_score": short_term_score,
                        "raw_score_24h": row['raw_score']
                    })
                    
                except Exception as e:
                    logger.warning(f"获取 {symbol} 1h 数据失败: {e}")
                    continue
            
            # 最终排序
            final_df = pd.DataFrame(final_candidates)
            if final_df.empty:
                 return []

            final_df = final_df.sort_values(by="short_term_score", ascending=False)
            
            # 归一化最终分数
            if not final_df.empty:
                max_score = final_df['short_term_score'].iloc[0]
                if max_score > 0:
                    final_df['score'] = (final_df['short_term_score'] / max_score) * 100
                else:
                    final_df['score'] = 0
            
            results = final_df.head(limit).to_dict('records')
            
            logger.info(f"精细扫描完成 (侧重1h爆发力):")
            for i, res in enumerate(results):
                logger.info(f"{i+1}. {res['symbol']}: 1h涨幅 {res['change_percent_1h']:+.2f}% | 24h涨幅 {res['change_percent_24h']:+.2f}% | 爆发力 {res['score']:.1f}/100")
                
            return results
            
        except Exception as e:
            logger.error(f"扫描市场失败: {e}")
            return []

if __name__ == "__main__":
    # 简单的测试运行
    logging.basicConfig(level=logging.INFO)
    scanner = MarketScanner()
    scanner.get_top_opportunities(limit=5)
