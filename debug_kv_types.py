#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask
from libinfinicore_infer import POINTER, KVCacheCStruct

print("=== 调试KVCache类型问题 ===")

# 加载模型
model = JiugeForCauslLM(
    "/home/shared/models/jiuge9G4B",
    device=DeviceType.DEVICE_TYPE_CPU,
    ndev=1
)

print("模型加载完成")

# 创建KV缓存
kv_cache = model.create_kv_cache()
print(f"kv_cache类型: {type(kv_cache)}")
print(f"kv_cache.data()类型: {type(kv_cache.data())}")

# 创建推理任务
infer_task = InferTask(
    id=0,
    tokens=[1, 2, 3],
    max_tokens=10,
    temperature=1.0,
    topk=1,
    topp=1.0,
    end_tokens=2
)

# 绑定KV缓存
infer_task.bind_kvcache(kv_cache)
print(f"infer_task.kvcache()类型: {type(infer_task.kvcache())}")
print(f"infer_task.kvcache().data()类型: {type(infer_task.kvcache().data())}")

# 检查是否是POINTER(KVCacheCStruct)类型
kv_data = infer_task.kvcache().data()
print(f"kv_data是否为POINTER(KVCacheCStruct): {isinstance(kv_data, POINTER(KVCacheCStruct))}")

# 尝试创建JiugeBatchedTask
try:
    from jiuge import JiugeBatchedTask
    batch_task = JiugeBatchedTask([infer_task])
    print("JiugeBatchedTask创建成功")
    print(f"kv_cache_ptrs[0]类型: {type(batch_task.kv_cache_ptrs[0])}")
except Exception as e:
    print(f"JiugeBatchedTask创建失败: {e}")

model.destroy_model_instance()
print("测试完成")