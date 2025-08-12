# 推理引擎统一 API 与多模型适配方案

---

## 1. 背景
目前 InfiniCore-Infer 框架已支持 **九格 (Jiuge)** 模型，通过 `inferBatch` 返回采样 token，但无法输出 `logprobs`，导致 PPL 评测需使用模拟数据。
随着业务需要接入 **MoE、RWKV 乃至更多结构**，亟需:
1. **统一对外推理接口**，隔离模型差异;
2. **原生支持返回 logprobs**，便于 PPL / NLL 评测与对齐;
3. **保持向后兼容**，现有服务/脚本零改动即可继续运行。

## 2. 方案目标
- 对外仅暴露 *两* 个 C API: `inferBatch` 与 `inferBatchWithLogprobs`。
- 内部为每种模型实现对应 *适配层*，但签名保持一致。
- Python / Rust / 其他调用方始终只链接 `libinfinicore_infer.so`，无需关心模型类型。

## 3. 核心设计
```
┌────────────────────────┐       ┌──────────────────────────┐
│ Python / Rust Engine   │──────▶│ 统一 C API (inferBatch*) │
└────────────────────────┘       └──────────────────────────┘
                                         ▲
                    ┌────────────────────┼─────────────────────┐
                    │                    │                     │
             ┌──────────────┐   ┌──────────────┐      ┌──────────────┐
             │ Jiuge Kernel │   │  MoE Kernel  │      │ RWKV Kernel  │
             └──────────────┘   └──────────────┘      └──────────────┘
```

### 3.1 统一 C API (头文件 `infinicore_infer.h`)
```c
#ifdef __cplusplus
extern "C" {
#endif

// 仅返回生成 token
int32_t inferBatch(ModelHandle* model,
                   const uint32_t* tokens,
                   const uint32_t* req_lens,
                   KVCache* kvs,
                   float temperature,
                   uint32_t topk,
                   float topp,
                   uint32_t* output);

// 返回 token + 对应 logprob (float32)
int32_t inferBatchWithLogprobs(ModelHandle* model,
                               const uint32_t* tokens,
                               const uint32_t* req_lens,
                               KVCache* kvs,
                               float temperature,
                               uint32_t topk,
                               float topp,
                               uint32_t* output,
                               float* logprobs_out);

#ifdef __cplusplus
}
#endif
```

*注意*: `inferBatchWithLogprobs` 仅比旧函数多一个 `float* logprobs_out`。

### 3.2 模型适配层
- **Jiuge**: 现有 `jiuge.cpp` 内实现 `inferBatch*`，新增 `logf(prob)` 写入缓冲即可。
- **MoE**: 新增 `moe_model.cpp`，内部包含 router、专家调度等逻辑，但导出同名 API。
- **RWKV**: 新增 `rwkv_model.cpp`，循环状态向量存储到 `KVCache`，同样导出 API。

所有实现文件在 `src/` 子目录中，但最终链接到同一个 `libinfinicore_infer.so`。

### 3.3 Python 包装 (`libinfinicore_infer.py`)
```python
infer_batch = dll.inferBatch
infer_batch_with_logprobs = dll.inferBatchWithLogprobs
infer_batch_with_logprobs.argtypes = infer_batch.argtypes + [ctypes.POINTER(ctypes.c_float)]
```
- `InfiniCoreInferEngine.generate(return_logprobs=False)` 若为 `True` 则调用新接口。

## 4. 编译 & 链接
- `xmake.lua` 中为每个模型实现创建 target: `jiuge_impl`, `moe_impl`, `rwkv_impl`。
- 末端 `libinfinicore_infer` target `add_deps(jiuge_impl, moe_impl, rwkv_impl)`。
- 单库导出统一符号，客户端只链接一次。

## 5. 性能与兼容
- 旧服务继续使用 `inferBatch`，无感知升级；
- 评测脚本/新功能可切换 `inferBatchWithLogprobs` 获取真实 NLL；
- 新增模型只在内部扩展，不影响公共 API，持续演进成本低。

## 6. 实施步骤
1. 在 `include/` 添加或更新 `infinicore_infer.h`。
2. 在 `src/` 分别实现或修改 `jiuge.cpp`, `moe_model.cpp`, `rwkv_model.cpp`。
3. 更新 `xmake.lua` 目标依赖并编译生成新 so。
4. 调整 `libinfinicore_infer.py` 绑定与 `ppl_test.py` 逻辑。
5. 运行 `quick_ppl_test.py` 验证 logprobs 对齐。


---
