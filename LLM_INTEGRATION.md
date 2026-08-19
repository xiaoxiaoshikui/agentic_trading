# ChatGPT 集成说明

## ✅ 已完成的集成

您的交易机器人现在可以使用 ChatGPT 来辅助交易决策！

### 🎯 工作原理

1. **基础策略** (EMA 双均线) 生成初始信号
2. **LLM 顾问** 分析市场状况并给出建议
3. **智能合并** 结合两者的意见做出最终决策

### 📊 LLM 决策逻辑

- **AGREE**: LLM 同意基础信号 → 执行基础信号
- **高置信度 (≥70%)**: LLM 覆盖基础信号
- **中置信度 (≥50%)**: 仅在 LLM 建议 FLAT 时覆盖
- **低置信度 (<50%)**: 保持基础信号，记录 LLM 意见

---

## 🔧 使用步骤

### 1. 更新 .env 文件

打开 `.env` 文件，添加你的配置：

```bash
# Binance API (必需)
BINANCE_API_KEY=你的币安API密钥
BINANCE_API_SECRET=你的币安API密钥

# OpenAI API (必需 - 启用LLM时)
OPENAI_API_KEY=你的OpenAI_API密钥
OPENAI_MODEL=gpt-4o-mini
ENABLE_LLM=true
```

**重要配置:**
- `OPENAI_API_KEY`: 你的 ChatGPT API 密钥
- `OPENAI_MODEL`: 推荐 `gpt-4o-mini` (性价比高) 或 `gpt-4` (更强大)
- `ENABLE_LLM`: 设为 `true` 启用 LLM

### 2. 运行机器人

#### 启用 LLM 模式 (推荐)
```powershell
.venv\Scripts\python.exe -m src.main --symbol BTCUSDT --interval 15m --dry-run
```

#### 仅基础策略模式
将 `.env` 中的 `ENABLE_LLM=false`，然后运行：
```powershell
.venv\Scripts\python.exe -m src.main --symbol BTCUSDT --interval 15m --dry-run
```

---

## 📝 日志示例

启用 LLM 后，你会看到这样的日志：

```
2025-12-07 14:52:22 [INFO] __main__: Starting AgenticTrading bot: symbol=BTCUSDT interval=15m dry_run=True
2025-12-07 14:52:22 [INFO] __main__: Initializing LLM advisor with model: gpt-4o-mini
2025-12-07 14:52:22 [INFO] src.llm_advisor: LLM Advisor initialized with model: gpt-4o-mini
2025-12-07 14:52:23 [INFO] src.agent: LLM advisor enabled for trading decisions

2025-12-07 14:52:24 [INFO] src.strategy: Signal: LONG (ema_fast=95234.56, ema_slow=94123.45, atr=345.67)
2025-12-07 14:52:26 [INFO] src.agent: LLM advice: action=AGREE, confidence=0.75, reason=EMA crossover confirms uptrend, low volatility suggests stable entry
2025-12-07 14:52:26 [INFO] src.agent: [DRY RUN] Would place LONG order: OrderPlan(quantity=0.002, stop_loss=94500.00, take_profit=96500.00)
```

---

## ⚙️ 高级配置

### 调整 LLM 模型

在 `.env` 中修改：

```bash
# 性价比高 (推荐)
OPENAI_MODEL=gpt-4o-mini

# 更强大但更贵
OPENAI_MODEL=gpt-4

# 便宜但效果一般
OPENAI_MODEL=gpt-3.5-turbo
```

### 修改决策逻辑

编辑 `src/agent.py` 中的 `_merge_signals()` 方法来调整：
- 置信度阈值 (目前: 0.7 高, 0.5 中)
- 覆盖规则
- 风险偏好

---

## 💡 LLM 提示词自定义

编辑 `src/llm_advisor.py` 中的 `_get_system_prompt()` 方法来自定义 LLM 的行为：

```python
def _get_system_prompt(self) -> str:
    return """你是一个专业的加密货币交易顾问...
    
    可以在这里添加:
    - 特定市场规则
    - 风险偏好
    - 交易风格指导
    - 禁止交易的条件
    """
```

---

## 🔍 测试建议

1. **先干跑测试**: 使用 `--dry-run` 观察 LLM 的决策
2. **对比模式**: 关闭 LLM 和启用 LLM 分别运行，对比结果
3. **小仓位**: 真实交易时从最小仓位开始
4. **监控 API 成本**: ChatGPT API 按 token 计费，注意成本

---

## 📊 成本估算

以 `gpt-4o-mini` 为例：
- 每次决策约消耗 ~500 tokens
- 成本约 $0.0003 per 决策
- 每小时运行一次 ≈ $0.007/小时 ≈ $5/月

---

## 🐛 故障排除

### LLM 未启用
```
[WARNING] LLM enabled but OPENAI_API_KEY not set. Running without LLM.
```
→ 检查 `.env` 文件中的 `OPENAI_API_KEY`

### API 错误
```
[ERROR] Error calling LLM: ...
```
→ 检查 API key 是否有效，账户是否有余额

### LLM 总是回退到基础信号
→ 可能是 LLM 置信度太低，检查日志中的 confidence 值

---

## 🎓 进阶应用

### 1. 添加更多市场数据
在 `llm_advisor.py` 的 `_build_context()` 中添加：
- 资金费率
- 交易量分析
- 新闻/情绪指标
- 链上数据

### 2. 多模型集成
创建多个 LLM advisor，投票决策

### 3. 强化学习
使用历史数据训练 LLM 的决策权重

---

## 📞 支持

遇到问题？检查：
1. `.env` 配置是否正确
2. API key 是否有效
3. 网络连接是否正常
4. 查看详细日志: `LOG_LEVEL=DEBUG`

祝交易顺利！🚀
