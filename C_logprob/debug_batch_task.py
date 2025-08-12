#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType, JiugeBatchedTask
from infer_task import InferTask

def debug_batch_task():
    print("=== 调试批处理任务 ===")
    
    # 加载模型
    model = JiugeForCauslLM(
        "/home/shared/models/jiuge9G4B",
        device=DeviceType.DEVICE_TYPE_CPU
    )
    
    # 创建KV缓存
    kv_cache = model.create_kv_cache()
    
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
    
    try:
        # 创建批处理任务
        print("创建JiugeBatchedTask...")
        batch_task = JiugeBatchedTask([infer_task])
        print("JiugeBatchedTask创建成功")
        
        # 检查kv_cache_ptrs
        print(f"kv_cache_ptrs长度: {len(batch_task.kv_cache_ptrs)}")
        print(f"kv_cache_ptrs[0]类型: {type(batch_task.kv_cache_ptrs[0])}")
        
    except Exception as e:
        print(f"创建JiugeBatchedTask时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    kv_cache.drop(model)
    model.destroy_model_instance()
    
if __name__ == "__main__":
    debug_batch_task()