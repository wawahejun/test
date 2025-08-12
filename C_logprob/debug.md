

## 我的整体思路
1. **厘清链路**：模型前向 → `jiuge.cpp` 取 logits → 截断逻辑 → `log_softmax` → Python 侧 logprobs / PPL 计算。
2. **逐级排查**：
   - 确认 Python 侧计算（`simple_logprobs_test.py`、`quick_ppl_test.py`）逻辑正确。
   - 检查 `log_softmax` 实现是否数值稳定、符合标准。
   - 聚焦截断逻辑：验证不同阈值策略对 logprobs 分布与 PPL 的影响。
3. **实验对比**：依次实现 IQR → 百分位数 → 均值±5σ 三种策略并编译测试，记录分布和 PPL 指标。
4. **收敛方案**：当发现仍有 ~48% token 同一 logprob 时，说明**仅靠简单阈值仍不足**，需考虑：
   - 仅处理绝对异常值（`INF`、`NaN`）。
   - 对 logits 按 layernorm/温度缩放后再截断。
   - 结合 soft clipping / tanh 压缩而非硬截断。
   - 审查模型本身是否输出极端值。

## 目前确认 **没问题** 的部分
- Python 端脚本（logprobs 统计 & PPL 计算）逻辑正确；概率和≈1，说明 `log_softmax` 输出合法。
- `log_softmax` 实现采用 **max-shift + exp + sum + log** 标准稳定算法，无溢出/下溢问题。
- 编译链已打通：`build_jiuge_logprobs.sh` / `xmake` 可成功生成 `libinfinicore_infer.so` 并导出 `inferBatchWithLogprobs`。

## 目前确认 **有问题** 的部分
- **截断过于激进**：无论 IQR、百分位数还是均值±5σ，仍导致约 48% token 拥有相同 logprob → 词汇表利用率<2%。
- **PPL 极大**：截断导致的大量 -X.87 常数 logprob 令困惑度失真，第三条样本直接溢出报 `math range error`。

## 下一步计划
1. **移除/极大放宽** 截断，仅在检测到 `NaN`/`±INF` 时替换为安全值；或使用 **soft clipping**（如 `tanh`）而非硬阈值。
2. **加入调试日志**：记录截断前后 logits 的 min/max/σ，判断真实分布是否异常。
3. **检查模型权重与量化**：如果原始 logits 非常极端，需考虑重新量化或调整 fp16/bf16 精度。
4. **小规模数据集验证**：在 wikitext2 上跑 PPL，确保变更能实质降低困惑度并减少重复 logprob占比。
5. 发现问题根源在模型前向计算的GEMM操作

---
若对此方案有任何疑问或需要进一步细化，请告诉我！
        