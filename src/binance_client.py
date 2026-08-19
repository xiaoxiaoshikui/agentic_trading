import logging
from typing import List, Dict, Any, Optional
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
import pandas as pd

logger = logging.getLogger(__name__)


def klines_to_df(klines: List) -> pd.DataFrame:
    """
    Convert raw klines list to DataFrame
    """
    if not klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(
        klines, 
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    )
    
    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
        
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


class BinanceFuturesClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)

    # ---------- Market data ----------

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict[str, Any]]:
        logger.debug("Fetching klines: %s %s limit=%d", symbol, interval, limit)
        return self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    def get_multi_timeframe_klines(
        self, 
        symbol: str, 
        intervals: List[str] = ["15m", "1h", "4h"],
        limit: int = 300
    ) -> Dict[str, List]:
        """
        获取多时间框架K线数据
        
        Args:
            symbol: 交易对
            intervals: 时间周期列表
            limit: 每个周期获取的K线数量
            
        Returns:
            {interval: klines} 字典
        """
        result = {}
        for interval in intervals:
            try:
                klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
                result[interval] = klines
                logger.debug(f"获取 {interval} K线: {len(klines)} 根")
            except Exception as e:
                logger.warning(f"获取 {interval} K线失败: {e}")
                result[interval] = []
        return result

    def get_price(self, symbol: str) -> float:
        ticker = self.client.futures_symbol_ticker(symbol=symbol)
        if not ticker or "price" not in ticker:
            raise ValueError(f"Invalid ticker response for {symbol}: {ticker}")
        return float(ticker["price"])

    # ---------- Account / position ----------

    def get_balance(self, asset: str = "USDT") -> float:
        balances = self.client.futures_account_balance()
        if not balances:
            logger.warning("Empty balance response from Binance")
            return 0.0
        for b in balances:
            if not isinstance(b, dict):
                continue
            if b.get("asset") == asset:
                try:
                    return float(b.get("balance", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid balance value for {asset}: {b.get('balance')}")
                    return 0.0
        return 0.0

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        positions = self.client.futures_position_information(symbol=symbol)
        if not positions:
            return None
        p = positions[0]
        # Validate required keys exist
        if "positionAmt" not in p:
            logger.warning(f"Invalid position response for {symbol}: missing 'positionAmt'")
            return None
        try:
            if float(p["positionAmt"]) == 0:
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid positionAmt value for {symbol}: {e}")
            return None
        return p

    # ---------- Advanced Data ----------
    
    def get_funding_rate(self, symbol: str) -> float:
        """获取最新资金费率"""
        try:
            funding = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding and isinstance(funding, list) and len(funding) > 0:
                rate = funding[-1].get('fundingRate')
                if rate is not None:
                    rate_float = float(rate)
                    # Validate funding rate is in reasonable range (-0.5% to +0.5%)
                    if abs(rate_float) > 0.01:
                        logger.warning(f"Unusual funding rate for {symbol}: {rate_float}")
                    return rate_float
            return 0.0
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"获取资金费率失败 (数据错误): {e}")
            return 0.0
        except Exception as e:
            logger.warning(f"获取资金费率失败: {e}")
            return 0.0
            
    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取历史资金费率"""
        try:
            return self.client.futures_funding_rate(symbol=symbol, limit=limit)
        except Exception as e:
            logger.warning(f"获取历史资金费率失败: {e}")
            return []

    def get_open_interest(self, symbol: str) -> float:
        """获取持仓量 (以币为单位)"""
        try:
            oi = self.client.futures_open_interest(symbol=symbol)
            if not oi or not isinstance(oi, dict):
                logger.warning(f"Invalid open interest response for {symbol}")
                return 0.0
            value = oi.get('openInterest')
            if value is not None:
                return float(value)
            return 0.0
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"获取持仓量失败 (数据错误): {e}")
            return 0.0
        except Exception as e:
            logger.warning(f"获取持仓量失败: {e}")
            return 0.0
            
    def get_open_interest_history(self, symbol: str, period: str, limit: int = 100) -> List[Dict]:
        """获取历史持仓量"""
        try:
            return self.client.futures_open_interest_hist(symbol=symbol, period=period, limit=limit)
        except Exception as e:
            logger.warning(f"获取历史持仓量失败: {e}")
            return []

    # ---------- Trading ----------
    
    def set_leverage(self, symbol: str, leverage: int):
        logger.info("Setting leverage %s to %dx", symbol, leverage)
        self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    def market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> Dict[str, Any]:
        logger.info("Placing market order: %s %s qty=%f", symbol, side, quantity)
        return self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if side.upper() == "LONG" else SIDE_SELL,
            type="MARKET",
            quantity=quantity,
            reduceOnly=reduce_only,
        )

    def place_bracket(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: float,
    ):
        """
        Simple implementation:
        - open market order
        - place SL and TP as separate orders

        (You can improve this later.)
        """
        direction = side.upper()
        logger.info(
            "Placing bracket order %s: qty=%.4f sl=%.2f tp=%.2f",
            direction, quantity, stop_loss, take_profit
        )
        self.market_order(symbol, direction, quantity)

        # Stop loss
        self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if direction == "LONG" else SIDE_BUY,
            type="STOP_MARKET",
            stopPrice=stop_loss,
            closePosition=True,
        )

        # Take profit
        self.client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if direction == "LONG" else SIDE_BUY,
            type="TAKE_PROFIT_MARKET",
            stopPrice=take_profit,
            closePosition=True,
        )