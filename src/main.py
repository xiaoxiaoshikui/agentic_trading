"""
AgenticTrading - 高级多智能体交易系统
仅使用高级策略: 多指标确认 + 市场状态识别 + 动态风控 + 多智能体决策
"""

import time
import argparse
import logging
import importlib
import traceback
import re
import math
from dataclasses import dataclass, field
from typing import Optional

from .config import load_config
from .binance_client import BinanceFuturesClient, klines_to_df
from .realtime_analyzer import RealtimeAnalyzer
from .advanced_risk import AdvancedRiskManager
from .local_llm import LocalStrategyArchitect
from .agents import MarketAnalyst
from .market_scanner import MarketScanner
from .multi_asset_simulator import MultiAssetSimulator
from .llm_advisor import LLMAdvisor
import src.dynamic_strategy as dynamic_strategy  # 引入动态策略模块

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# Dry Run 模拟交易器
# ============================================================
@dataclass
class SimulatedPosition:
    """模拟持仓"""
    direction: str  # LONG / SHORT
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    open_time: float  # timestamp


@dataclass
class DryRunSimulator:
    """
    Dry Run 模拟交易器
    真实模拟持仓状态和资金变化
    """
    initial_capital: float = 10000.0
    leverage: int = 10
    fee_rate: float = 0.0004
    max_hold_hours: int = 4

    # 状态
    capital: float = field(init=False)
    position: Optional[SimulatedPosition] = field(default=None, init=False)
    total_trades: int = field(default=0, init=False)
    wins: int = field(default=0, init=False)
    losses: int = field(default=0, init=False)
    consecutive_losses: int = field(default=0, init=False)  # 连续亏损计数
    total_pnl: float = field(default=0.0, init=False)
    max_drawdown: float = field(default=0.0, init=False)
    peak_capital: float = field(init=False)
    last_trade_time: float = field(init=False)  # 上次交易时间

    def __post_init__(self):
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.last_trade_time = time.time()  # 初始化为当前时间

    def check_position(self, current_price: float, current_time: float, logger) -> Optional[dict]:
        """
        检查当前持仓是否触发止盈/止损/超时
        返回平仓结果或 None
        """
        if self.position is None:
            return None

        pos = self.position
        time_held = (current_time - pos.open_time) / 3600  # 小时

        # 检查止盈止损
        if pos.direction == "LONG":
            if current_price <= pos.stop_loss:
                return self._close_position(pos.stop_loss, "止损触发", logger)
            elif current_price >= pos.take_profit:
                return self._close_position(pos.take_profit, "止盈触发", logger)
        else:  # SHORT
            if current_price >= pos.stop_loss:
                return self._close_position(pos.stop_loss, "止损触发", logger)
            elif current_price <= pos.take_profit:
                return self._close_position(pos.take_profit, "止盈触发", logger)

        # 检查超时
        if time_held >= self.max_hold_hours:
            return self._close_position(current_price, f"超时平仓({self.max_hold_hours}h)", logger)

        return None

    def _close_position(self, exit_price: float, reason: str, logger) -> dict:
        """平仓并计算盈亏"""
        pos = self.position

        # 计算 PnL (USDT)
        # 公式: (平仓价 - 开仓价) * 数量 [做多]
        #       (开仓价 - 平仓价) * 数量 [做空]
        if pos.direction == "LONG":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price
            
        gross_pnl = price_diff * pos.position_size
        
        # 计算百分比收益率 (相对于保证金)
        # 保证金 = (Entry * Size) / Leverage
        margin = (pos.entry_price * pos.position_size) / self.leverage
        pnl_percent = (gross_pnl / margin) if margin > 0 else 0

        # 实际盈亏（扣除手续费）
        # 手续费通常是名义价值的万分之几
        notional_value = exit_price * pos.position_size
        fees = notional_value * self.fee_rate # 平仓手续费 (开仓费已在开仓时扣除，或这里简化为双边)
        # 简单起见，这里扣双边费率
        total_fees = (pos.entry_price * pos.position_size + notional_value) * self.fee_rate
        
        net_pnl = gross_pnl - total_fees

        # 更新状态
        self.capital += net_pnl
        self.total_pnl += net_pnl
        self.total_trades += 1
        self.last_trade_time = time.time()  # 更新交易时间

        if net_pnl > 0:
            self.wins += 1
            self.consecutive_losses = 0  # 重置连败
            emoji = "✅"
        else:
            self.losses += 1
            self.consecutive_losses += 1 # 增加连败
            emoji = "❌"

        # 更新最大回撤
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        drawdown = (self.peak_capital - self.capital) / self.peak_capital * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        result = {
            "direction": pos.direction,
            "entry": pos.entry_price,
            "exit": exit_price,
            "pnl": net_pnl,
            "pnl_percent": pnl_percent * 100,
            "reason": reason
        }

        logger.info(f"{emoji} 平仓: {pos.direction} | 入场: ${pos.entry_price:.6f} → 出场: ${exit_price:.6f}")
        logger.info(f"   💰 盈亏: ${net_pnl:+.2f} ({pnl_percent*100:+.2f}%) | 原因: {reason}")
        logger.info(f"   ⚠️ 连续亏损: {self.consecutive_losses} 次")

        self.position = None
        return result

    def open_position(self, direction: str, entry_price: float, position_size: float,
                      stop_loss: float, take_profit: float, current_time: float, logger) -> bool:
        """
        开仓
        如果已有持仓，返回 False
        """
        if self.position is not None:
            # 已有持仓
            if self.position.direction == direction:
                logger.info(f"⏭️ 跳过: 已持有 {direction} 仓位")
                return False
            else:
                # 反向信号，先平仓
                logger.info(f"🔄 反向信号，先平当前仓位...")
                self._close_position(entry_price, "反向平仓", logger)

        # 开新仓
        self.position = SimulatedPosition(
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            open_time=current_time
        )

        logger.info(f"🆕 开仓: {direction} @ ${entry_price:.2f}")
        logger.info(f"   🛑 止损: ${stop_loss:.6f} | 🎯 止盈: ${take_profit:.6f}")
        return True

    def get_status(self) -> str:
        """获取当前状态摘要"""
        win_rate = self.wins / self.total_trades * 100 if self.total_trades > 0 else 0
        roi = (self.capital - self.initial_capital) / self.initial_capital * 100

        pos_status = "空仓"
        if self.position:
            pos_status = f"{self.position.direction} @ ${self.position.entry_price:.2f}"

        return (
            f"💵 资金: ${self.capital:,.2f} | ROI: {roi:+.2f}% | "
            f"交易: {self.total_trades} | 胜率: {win_rate:.1f}% | "
            f"持仓: {pos_status}"
        )

    def has_position(self) -> bool:
        return self.position is not None

    def get_position_direction(self) -> Optional[str]:
        return self.position.direction if self.position else None

    def is_corrupted(self) -> bool:
        """检测数据是否损坏 (nan/inf/溢出)"""
        # 检测 nan/inf
        if (math.isnan(self.capital) or math.isinf(self.capital) or
            math.isnan(self.total_pnl) or math.isinf(self.total_pnl)):
            return True
        # 检测数值溢出（资金不应超过初始资金的1000倍）
        if self.capital > self.initial_capital * 1000:
            return True
        # 检测负资金
        if self.capital < 0:
            return True
        return False
    
    def reset(self):
        """重置模拟器到初始状态"""
        self.capital = self.initial_capital
        self.position = None
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.consecutive_losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        self.last_trade_time = time.time()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="AgenticTrading – Advanced Multi-Agent Trading Bot")
    parser.add_argument("--symbol", default=None, help="交易对, 如 BTCUSDT")
    parser.add_argument("--interval", default=None, help="K线周期, 如 15m")
    parser.add_argument("--dry-run", action="store_true", help="干跑模式(不实际下单)")
    parser.add_argument("--sleep", type=int, default=60, help="分析间隔(秒)")
    parser.add_argument("--no-multi-agent", action="store_true", help="禁用多智能体系统")
    parser.add_argument("--min-score", type=int, default=60, help="最低交易评分(0-100)")
    parser.add_argument("--min-confirmations", type=int, default=3, help="最少确认指标数")
    parser.add_argument("--max-idle-hours", type=float, default=4.0, help="最长空闲时间(小时)，超时触发进化")
    parser.add_argument("--auto-scan", action="store_true", help="自动扫描最佳币种")
    parser.add_argument("--scan-interval", type=int, default=30, help="扫描间隔(分钟)")
    # Multi-asset mode
    parser.add_argument("--multi-asset", action="store_true", help="多资产同时交易模式")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], help="多资产模式的交易对列表")
    parser.add_argument("--max-positions", type=int, default=3, help="最大同时持仓数")
    parser.add_argument("--capital-per-pos", type=float, default=0.3, help="每个仓位占用资金比例")
    return parser.parse_args()


def main():
    cfg = load_config()
    args = parse_args()
    setup_logging(cfg.log_level)

    logger = logging.getLogger(__name__)

    symbol = args.symbol or cfg.default_symbol
    interval = args.interval or cfg.default_interval
    dry_run = args.dry_run or cfg.dry_run

    logger.info("=" * 60)
    logger.info("🚀 AgenticTrading - 高级多智能体交易系统")
    logger.info("=" * 60)
    logger.info(f"交易对: {symbol} | 周期: {interval}")
    logger.info(f"干跑模式: {dry_run} | 多智能体: {not args.no_multi_agent}")
    logger.info(f"最低评分: {args.min_score} | 最少确认: {args.min_confirmations}")
    logger.info(f"僵尸检测: {args.max_idle_hours} 小时")
    logger.info("=" * 60)

    # 初始化币安客户端
    client = BinanceFuturesClient(cfg.api_key, cfg.api_secret, testnet=cfg.testnet)

    # 初始化实时分析器
    # 强制禁用 LLM 和多智能体，避免 429 错误
    enable_multi_agent = False # not args.no_multi_agent and cfg.openai_api_key
    
    analyzer = RealtimeAnalyzer(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        enable_multi_agent=enable_multi_agent,
        min_confirmations=args.min_confirmations
    )
    
    # 同时也强制关闭 analyzer 内部的 web search 开关 (如果它是根据 enable_multi_agent 自动推断的，或者有独立开关)
    analyzer.enable_web_search = False
    analyzer.enable_multi_agent = False

    risk_manager = AdvancedRiskManager()

    # Initialize LLM Advisor for alpha discovery
    llm_advisor = None
    if cfg.enable_llm:
        try:
            llm_advisor = LLMAdvisor()
            logger.info("LLM Advisor initialized for alpha discovery")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM Advisor: {e}")
    else:
        logger.info("LLM Advisor disabled")

    logger.info(f"多智能体: {'✅ 启用' if enable_multi_agent else '❌ 禁用'}")
    
    # 根据模式选择交易循环
    if args.multi_asset:
        logger.info(f"🌐 多资产模式: {args.symbols}")
        logger.info(f"   最大持仓: {args.max_positions} | 每仓资金: {args.capital_per_pos*100}%")
        logger.info("=" * 60)
        run_multi_asset_loop(client, risk_manager, args, interval, dry_run, logger)
    else:
        logger.info("系统启动完成，开始交易循环...")
        run_trading_loop(client, analyzer, risk_manager, llm_advisor, args, symbol, interval, dry_run, logger)


def _merge_signals(base_signal, llm_advice, logger):
    """
    Merge base trading signal with LLM advice based on confidence levels.
    
    Args:
        base_signal: Signal object from strategy
        llm_advice: LLMAdvice object
        logger: Logger instance
        
    Returns:
        tuple: (final_action, final_confidence)
    """
    base_action = base_signal.side
    base_confidence = base_signal.confidence
    llm_action = llm_advice.action
    llm_confidence = llm_advice.confidence
    
    # Decision logic based on LLM confidence
    if llm_confidence >= 0.7:
        # High confidence: override base signal
        final_action = llm_action
        final_confidence = llm_confidence
        logger.info(f"🔄 LLM override (high conf): {base_action} → {final_action}")
        
    elif llm_confidence >= 0.5:
        # Medium confidence: only override if LLM suggests FLAT
        if llm_action == "FLAT":
            final_action = "FLAT"
            final_confidence = llm_confidence
            logger.info(f"🔄 LLM override (medium conf): {base_action} → FLAT")
        else:
            final_action = base_action
            final_confidence = base_confidence
            logger.info(f"✅ Keep base signal (LLM medium conf, not FLAT)")
            
    else:
        # Low confidence: keep base signal, log LLM opinion
        final_action = base_action
        final_confidence = base_confidence
        logger.info(f"✅ Keep base signal (LLM low conf: {llm_action})")
    
    return final_action, final_confidence


def run_trading_loop(client, analyzer, risk_manager, llm_advisor, args, symbol, interval, dry_run, logger):
    """主交易循环"""

    # 多时间框架配置
    mtf_intervals = ["15m", "1h", "4h"]

    # 初始化 Dry Run 模拟器
    simulator = DryRunSimulator() if dry_run else None
    
    # 初始化 AI 架构师
    architect = LocalStrategyArchitect(model_name="qwen3:8b")
    
    # 初始化 MarketAnalyst (提供 LLM 进化的市场分析上下文)
    market_analyst = MarketAnalyst()
    logger.info("MarketAnalyst initialized for LLM context generation")
    
    # 初始化 MarketScanner (自动选币)
    scanner = None
    last_scan_time = 0
    current_symbol = symbol
    if args.auto_scan:
        scanner = MarketScanner()
        logger.info(f"🔍 MarketScanner enabled - scan interval: {args.scan_interval} minutes")

    while True:
        try:
            current_time = time.time()
            
            # --- 自动选币扫描 ---
            if scanner and (current_time - last_scan_time) > (args.scan_interval * 60):
                logger.info("🔍 Scanning market for best opportunities...")
                opportunities = scanner.get_top_opportunities(limit=3)
                
                if opportunities:
                    best = opportunities[0]
                    if best['symbol'] != current_symbol:
                        # 如果有持仓，先平仓再切换
                        if simulator and simulator.has_position():
                            logger.info(f"📤 Closing position before switching to {best['symbol']}")
                            price = client.get_price(current_symbol)
                            simulator._close_position(price, "Switch symbol", logger)
                        
                        current_symbol = best['symbol']
                        logger.info(f"🎯 Switched to {current_symbol} (1h: {best['change_percent_1h']:+.2f}%, score: {best['score']:.1f})")
                    else:
                        logger.info(f"✅ Staying with {current_symbol} (still best)")
                
                last_scan_time = current_time
            
            # --- 自动进化触发器 ---
            evolution_triggered = False
            evolution_reason = ""
            
            if simulator:
                # 条件1: 连续亏损
                if simulator.consecutive_losses >= 2:
                    evolution_triggered = True
                    evolution_reason = f"连续亏损 {simulator.consecutive_losses} 次"
                
                # 条件2: 僵尸策略检测 (超过设定的空闲时间)
                elif (current_time - simulator.last_trade_time) > (args.max_idle_hours * 3600):
                    evolution_triggered = True
                    evolution_reason = f"策略僵尸化 ({args.max_idle_hours}小时无交易)"
            
            if evolution_triggered:
                logger.warning(f"🚨 触发自动进化! 原因: {evolution_reason}")
                logger.info("🧠 正在请求 AI 架构师优化策略...")
                
                # Use MarketAnalyst context if available, otherwise use basic summary
                market_context = getattr(architect, 'last_market_context', None)
                if not market_context:
                    market_context = f"Symbol: {symbol}\nTrigger: {evolution_reason}\nMarket conditions may have changed."
                
                performance = {
                    "win_rate": (simulator.wins / simulator.total_trades * 100) if simulator.total_trades > 0 else 0,
                    "issue": evolution_reason,
                }
                
                success = architect.optimize_strategy(market_context, performance)
                if success:
                    logger.info("♻️ 策略已更新，正在热重载...")
                    importlib.reload(dynamic_strategy)
                    logger.info("✅ 新策略已加载生效!")
                    if simulator:
                        simulator.consecutive_losses = 0
                        simulator.last_trade_time = current_time # 重置时间防止重复触发
                else:
                    logger.error("❌ 策略进化失败")
            # --------------------

            # 1. 获取多时间框架市场数据
            logger.info(f"\n{'='*60}")
            logger.info(f"⏱️ 获取多时间框架数据: {mtf_intervals}")

            mtf_klines = client.get_multi_timeframe_klines(current_symbol, mtf_intervals, limit=300)
            mtf_data = {k: klines_to_df(v) if v else None for k, v in mtf_klines.items()}

            # 主时间框架数据
            df = mtf_data.get(interval, mtf_data.get("15m"))

            # 2. 获取账户信息
            current_price = client.get_price(current_symbol)
            
            # --- 获取高级数据 (Funding & OI) ---
            try:
                funding_rate = client.get_funding_rate(current_symbol)
                open_interest = client.get_open_interest(current_symbol)
                # 将数据注入 DataFrame (广播到整列)
                df['funding_rate'] = funding_rate
                df['open_interest'] = open_interest
                logger.info(f"📊 高级数据: Funding={funding_rate:.6f} | OI={open_interest:.2f}")
            except Exception as e:
                logger.warning(f"无法获取高级数据: {e}")
                df['funding_rate'] = 0.0
                df['open_interest'] = 0.0
                
            # --------------------------------

            if dry_run:
                # 🔧 NaN 检测与自愈
                if simulator.is_corrupted():
                    logger.error("🚨 检测到数据损坏 (nan/inf)！触发自动修复...")
                    
                    # 触发 AI 修复风控模块
                    fix_summary = "风控计算产生了 nan/inf 值，可能是止损价与入场价过于接近导致除零。请检查 calculate_position 方法。"
                    fix_perf = {
                        "win_rate": 0,
                        "issue": "position_size 或 capital 变成 nan，需要增加最小价格风险保护。",
                        "is_error_fix": True
                    }
                    
                    architect.optimize_strategy(fix_summary, fix_perf)
                    
                    # 重置模拟器
                    simulator.reset()
                    logger.info("♻️ 模拟器已重置，资金恢复到初始状态")
                    continue  # 跳过本轮，重新开始
                
                # 检查模拟持仓是否触发平仓
                simulator.check_position(current_price, current_time, logger)

                balance = simulator.capital
                position = simulator.position  # 模拟持仓

                # 显示模拟状态
                logger.info(f"📊 [模拟实盘] {simulator.get_status()}")
            else:
                balance = client.get_balance("USDT")
                position = client.get_position(symbol)

            logger.info(f"📈 {current_symbol} 当前价格: ${current_price:.6f}")
            logger.info(f"💰 账户余额: ${balance:.2f}")

            # 3. 多时间框架分析
            mtf_result = analyzer.analyze_multi_timeframe(mtf_data, interval)
            mtf_consensus = mtf_result.get("consensus", {})

            # 4. 实时高级分析 (使用动态策略)
            # ==========================================
            # 使用动态加载的 dynamic_strategy
            dynamic_signal = dynamic_strategy.calculate_signal(df)
            
            # --- 自动错误修复机制 ---
            # 扩展检测逻辑：包含 "Error:" 或 "計算異常"
            if dynamic_signal.reason.startswith("Error:") or "計算異常" in dynamic_signal.reason:
                logger.error(f"🚨 策略运行报错: {dynamic_signal.reason}")
                logger.info("🔧 正在请求 AI 紧急修复代码 Bug...")
                
                error_summary = f"策略代码运行时发生致命错误，请立即修复。\n错误详情: {dynamic_signal.reason}"
                error_perf = {
                    "win_rate": 0,
                    "issue": "代码存在 Bug，导致无法计算信号。请检查变量作用域、未定义变量或除零错误。",
                    "is_error_fix": True
                }
                
                # 尝试修复
                if architect.optimize_strategy(error_summary, error_perf):
                    logger.info("✅ Bug 已修复，正在重载策略...")
                    importlib.reload(dynamic_strategy)
                    # 重新计算一次
                    dynamic_signal = dynamic_strategy.calculate_signal(df)
                else:
                    logger.error("❌ 自动修复失败，请人工检查！")
            # -----------------------
            
            logger.info(f"🧠 [LLM Strategy] Signal: {dynamic_signal.side} | Confidence: {dynamic_signal.confidence:.2f}")
            
            # ==========================================
            # MarketAnalyst: Generate context for LLM evolution
            # ==========================================
            _funding = df['funding_rate'].iloc[-1] if 'funding_rate' in df.columns else 0.0
            _oi = df['open_interest'].iloc[-1] if 'open_interest' in df.columns else 0.0
            
            market_state = market_analyst.process(df, _funding, _oi, oi_change_24h=0.0)
            logger.info(f"📊 [MarketAnalyst] Regime: {market_state.regime} | Score: {market_state.score:+.2f}")
            
            # Store context for next evolution cycle
            if hasattr(architect, 'last_market_context'):
                architect.last_market_context = market_analyst.generate_llm_context(market_state)
            
            # ==========================================
            # LLM Alpha Discovery Integration
            # ==========================================
            llm_advice = None
            if llm_advisor:
                try:
                    market_context = market_analyst.generate_llm_context(market_state)
                    llm_advice = llm_advisor.get_advice(market_context, dynamic_signal.side, market_state.indicators)
                    logger.info(f"🧠 [LLM Advisor] Action: {llm_advice.action} | Confidence: {llm_advice.confidence:.2f}")
                    logger.info(f"   Reason: {llm_advice.reason}")
                except Exception as e:
                    logger.warning(f"LLM advice failed: {e}")
                    llm_advice = None
            
            # ==========================================
            # SIMPLIFIED DECISION: LLM Alpha + Risk Manager
            # ==========================================
            final_action = dynamic_signal.side
            final_confidence = dynamic_signal.confidence
            
            # Merge LLM advice if available
            if llm_advice:
                final_action, final_confidence = _merge_signals(dynamic_signal, llm_advice, logger)
            
            # Only trade if confidence > threshold
            if final_action != "FLAT" and final_confidence > 0.3:
                current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
                
                risk_plan = risk_manager.calculate_position(
                    balance=balance,
                    entry_price=current_price,
                    stop_loss=current_price - (2 * current_atr) if final_action == "LONG" else current_price + (2 * current_atr),
                    signal_confidence=final_confidence,
                    market_regime=market_state.regime,
                    atr=current_atr,
                    atr_percent=current_atr/current_price
                )
                
                logger.info(f"🎯 Final: {final_action} | Size: {risk_plan.position_size:.6f} | Conf: {final_confidence:.2f}")

                if dry_run:
                    simulator.open_position(
                        direction=final_action,
                        entry_price=current_price,
                        position_size=risk_plan.position_size,
                        stop_loss=risk_plan.stop_loss,
                        take_profit=risk_plan.take_profit,
                        current_time=current_time,
                        logger=logger
                    )
                else:
                    logger.info(f"Would execute {final_action} trade")
            else:
                if dry_run and simulator.has_position():
                    logger.info(f"⏳ Holding: {simulator.get_position_direction()}")
                else:
                    logger.info("⏳ No signal")

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.exception(f"分析过程出错: {e}")
            
            # 🔧 自动修复机制
            if architect and "src/" in error_traceback:
                # 从 traceback 中提取出错的文件路径
                file_match = re.search(r'File "([^"]+src[/\\][^"]+\.py)"', error_traceback)
                if file_match:
                    error_file = file_match.group(1)
                    logger.info(f"🤖 检测到代码错误，尝试自动修复: {error_file}")
                    if architect.auto_fix_code(error_file, error_traceback):
                        logger.info("♻️ 修复完成，重新加载模块...")
                        try:
                            if "strategy_memory" in error_file:
                                from src import strategy_memory
                                importlib.reload(strategy_memory)
                            elif "dynamic_strategy" in error_file:
                                importlib.reload(dynamic_strategy)
                            elif "local_llm" in error_file:
                                from src import local_llm
                                importlib.reload(local_llm)
                            logger.info("✅ 模块已重新加载")
                        except Exception as reload_err:
                            logger.error(f"重载模块失败: {reload_err}")

        time.sleep(args.sleep)


def run_multi_asset_loop(client, risk_manager, args, interval, dry_run, logger):
    """Multi-asset trading loop - trades multiple symbols simultaneously"""
    
    symbols = args.symbols
    
    # Initialize multi-asset simulator
    simulator = MultiAssetSimulator(
        initial_capital=10000.0,
        max_positions=args.max_positions,
        capital_per_position=args.capital_per_pos
    ) if dry_run else None
    
    # Initialize AI architect
    architect = LocalStrategyArchitect(model_name="qwen3:8b")
    
    logger.info(f"🌐 Multi-Asset Mode Started")
    logger.info(f"   Symbols: {symbols}")
    
    while True:
        try:
            current_time = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"⏱️ Multi-Asset Scan: {symbols}")
            
            # Get current prices for all symbols
            prices = {}
            for symbol in symbols:
                try:
                    prices[symbol] = client.get_price(symbol)
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
            
            # Check existing positions for SL/TP
            if simulator:
                closed = simulator.check_positions(prices, current_time)
                for trade in closed:
                    emoji = "✅" if trade['win'] else "❌"
                    logger.info(f"{emoji} Closed {trade['symbol']} {trade['direction']}: ${trade['pnl']:+.2f} ({trade['reason']})")
                
                logger.info(f"📊 Portfolio: {simulator.get_status()}")
            
            # Scan each symbol for signals
            for symbol in symbols:
                if symbol not in prices:
                    continue
                
                # Skip if already have position
                if simulator and simulator.has_position(symbol):
                    pos = simulator.get_position(symbol)
                    if pos:
                        logger.info(f"   [{symbol}] Holding {pos.direction} @ ${pos.entry_price:.2f}")
                    continue
                
                # Get market data
                try:
                    klines = client.get_klines(symbol, interval, limit=300)
                    df = klines_to_df(klines)
                    df['funding_rate'] = 0.0
                    df['open_interest'] = 0.0
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
                
                # Generate signal
                try:
                    signal = dynamic_strategy.calculate_signal(df)
                except Exception as e:
                    logger.warning(f"Signal error for {symbol}: {e}")
                    continue
                
                current_price = prices[symbol]
                
                # Check if we have a valid signal
                if signal.side == "FLAT" or signal.confidence < 0.3:
                    logger.info(f"   [{symbol}] ${current_price:.2f} | {signal.side} (conf: {signal.confidence:.2f})")
                    continue
                
                # Portfolio risk check with direction
                if simulator and not simulator.can_open_position(symbol, signal.side):
                    continue
                
                # Calculate position
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.01
                if atr <= 0:
                    atr = current_price * 0.01
                
                if signal.side == "LONG":
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 4)
                else:
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 4)
                
                try:
                    plan = risk_manager.calculate_position(
                        balance=simulator.get_position_capital() if simulator else 3000,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        signal_confidence=signal.confidence,
                        market_regime="trending",
                        atr=atr,
                        atr_percent=atr/current_price
                    )
                except Exception as e:
                    logger.warning(f"Risk calc error for {symbol}: {e}")
                    continue
                
                if plan.position_size <= 0:
                    continue
                
                # Open position
                logger.info(f"   🎯 [{symbol}] {signal.side} Signal @ ${current_price:.2f} (conf: {signal.confidence:.2f})")
                
                if dry_run and simulator:
                    simulator.open_position(
                        symbol=symbol,
                        direction=signal.side,
                        entry_price=current_price,
                        position_size=plan.position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        current_time=current_time
                    )
                else:
                    logger.info(f"   Would execute {signal.side} trade on {symbol}")
            
            # Strategy evolution trigger (based on multi-asset performance analysis)
            if simulator and simulator.total_trades >= 10:
                win_rate = simulator.wins / simulator.total_trades
                should_evolve = False
                evolution_reason = ""
                
                # Condition 1: Overall win rate too low
                if win_rate < 0.40:
                    should_evolve = True
                    evolution_reason = f"Portfolio win rate: {win_rate*100:.1f}%"
                
                # Condition 2: Any symbol consistently losing
                for sym, pnl in simulator.pnl_by_symbol.items():
                    if pnl < -simulator.initial_capital * 0.05:  # Lost >5% on one symbol
                        should_evolve = True
                        evolution_reason = f"{sym} lost ${abs(pnl):.2f}"
                        break
                
                if should_evolve:
                    logger.warning(f"🚨 Triggering multi-asset evolution: {evolution_reason}")
                    
                    # Build comprehensive multi-asset context
                    context_lines = [
                        "=" * 50,
                        "MULTI-ASSET PERFORMANCE ANALYSIS",
                        "=" * 50,
                        f"\nSymbols Traded: {symbols}",
                        f"Total Trades: {simulator.total_trades}",
                        f"Overall Win Rate: {win_rate*100:.1f}%",
                        f"Total PnL: ${simulator.total_pnl:+.2f}",
                        f"Max Drawdown: {simulator.max_drawdown:.1f}%",
                        "",
                        "## Per-Symbol Breakdown:"
                    ]
                    
                    # Analyze each symbol's performance
                    for sym in symbols:
                        sym_pnl = simulator.pnl_by_symbol.get(sym, 0)
                        sym_trades = [t for t in simulator.trade_history if t['symbol'] == sym]
                        sym_wins = sum(1 for t in sym_trades if t['win'])
                        sym_wr = sym_wins / len(sym_trades) * 100 if sym_trades else 0
                        
                        status = "🟢" if sym_pnl > 0 else "🔴"
                        context_lines.append(f"  {status} {sym}: PnL ${sym_pnl:+.2f} | Trades: {len(sym_trades)} | Win: {sym_wr:.0f}%")
                    
                    context_lines.extend([
                        "",
                        "## Evolution Guidance:",
                        "- The strategy must work across ALL symbols above",
                        "- Focus on improving weak performers without hurting strong ones",
                        "- Consider: Are losing symbols more volatile? Different regime?",
                        f"- Primary Issue: {evolution_reason}",
                        "=" * 50
                    ])
                    
                    market_context = "\n".join(context_lines)
                    
                    performance = {
                        "win_rate": win_rate * 100,
                        "issue": evolution_reason,
                        "pnl_by_symbol": simulator.pnl_by_symbol.copy()
                    }
                    
                    if architect.optimize_strategy(market_context, performance):
                        importlib.reload(dynamic_strategy)
                        logger.info("✅ Multi-asset strategy evolved!")
                        # Reset tracking for fair comparison
                        simulator.pnl_by_symbol = {s: 0 for s in symbols}
            
        except Exception as e:
            logger.exception(f"Multi-asset loop error: {e}")
        
        time.sleep(args.sleep)


def log_dry_run_trade(analysis, logger):
    """记录干跑交易"""
    logger.info("[DRY RUN] 建议交易:")
    logger.info(f"  📍 方向: {analysis.action}")
    logger.info(f"  📦 仓位: {analysis.position_size:.6f}")
    logger.info(f"  💵 入场: ${analysis.entry_price:.6f}")
    logger.info(f"  🛑 止损: ${analysis.stop_loss:.6f}")
    logger.info(f"  🎯 止盈: ${analysis.take_profit:.6f}")
    logger.info(f"  ⚖️ 风险回报: {analysis.risk_reward:.2f}")
    logger.info(f"  💡 原因: {analysis.reason}")


def execute_trade(client, symbol, analysis, logger):
    """执行实际交易"""
    try:
        if analysis.action == "LONG":
            logger.info(f"🟢 执行开多: {symbol}")
            # TODO: 实际下单逻辑
            # order = client.create_order(
            #     symbol=symbol,
            #     side="BUY",
            #     type="MARKET",
            #     quantity=analysis.position_size
            # )
            # 设置止损止盈
            # client.create_stop_loss(...)
            # client.create_take_profit(...)
            
        elif analysis.action == "SHORT":
            logger.info(f"🔴 执行开空: {symbol}")
            # TODO: 实际下单逻辑
            
    except Exception as e:
        logger.error(f"❌ 下单失败: {e}")


if __name__ == "__main__":
    main()