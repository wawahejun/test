#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask

def debug_kv_cache():
    print("=== 调试KV缓存类型 ===")
    
    # 加载模型
    model = JiugeForCauslLM(
        "/home/shared/models/jiuge9G4B",
        device=DeviceType.DEVICE_TYPE_CPU
    )
    
    # 创建KV缓存
    kv_cache = model.create_kv_cache()
    print(f"KV缓存类型: {type(kv_cache)}")
    print(f"KV缓存有data方法: {hasattr(kv_cache, 'data')}")
    
    if hasattr(kv_cache, 'data'):
        data = kv_cache.data()
        print(f"data()返回类型: {type(data)}")
    
    # 创建简单的推理任务
    tokens = [1, 2, 3]  # 简单的token序列
    infer_task = InferTask(
        id=0,
        tokens=tokens,
        max_tokens=10,
        temperature=1.0,
        topk=1,
        topp=1.0,
        end_tokens=[2]
    )
    
    # 绑定KV缓存
    infer_task.bind_kvcache(kv_cache)
    
    # 检查kvcache()返回的类型
    bound_cache = infer_task.kvcache()
    print(f"绑定后kvcache()返回类型: {type(bound_cache)}")
    print(f"绑定后kvcache()有data方法: {hasattr(bound_cache, 'data')}")
    
    # 清理
    kv_cache.drop(model)
    model.destroy_model_instance()
    
if __name__ == "__main__":
    debug_kv_cache()