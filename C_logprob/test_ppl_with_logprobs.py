#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
import numpy as np

# 添加脚本路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask

def calculate_ppl_simple():
    print("=== Jiuge模型PPL测试（使用修复后的logprobs） ===")
    
    # 加载模型
    model_path = '/home/shared/models/jiuge9G4B'
    print(f"加载模型: {model_path}")
    
    model = JiugeForCauslLM(
        model_dir_path=model_path,
        device=DeviceType.DEVICE_TYPE_CPU,
        ndev=1,
        max_tokens=512
    )
    
    print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
    
    # 测试文本列表
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language."
    ]
    
    total_log_prob = 0.0
    total_tokens = 0
    valid_results = 0
    
    for i, text in enumerate(test_texts):
        print(f"\n--- 测试文本 {i+1}/{len(test_texts)} ---")
        print(f"  处理文本: {text[:50]}...")
        
        try:
            # 编码文本
            tokens = model.tokenizer.encode(text, add_special_tokens=True)
            print(f"  Token数量: {len(tokens)}")
            
            if len(tokens) < 2:
                print(f"  ✗ Token数量太少，跳过")
                continue
            
            text_log_prob = 0.0
            
            # 对每个位置计算logprob
            for pos in range(1, len(tokens)):
                context_tokens = tokens[:pos]
                target_token = tokens[pos]
                
                print(f"    位置{pos}: token={target_token}, logprob=", end="")
                
                # 创建推理任务
                infer_task = InferTask(
                    id=0,
                    tokens=context_tokens,
                    max_tokens=len(context_tokens) + 1,
                    temperature=1.0,
                    topk=1,
                    topp=1.0,
                    end_tokens=model.eos_token_id
                )
                
                # 创建和绑定KV缓存
                kv_cache = model.create_kv_cache()
                infer_task.bind_kvcache(kv_cache)
                
                # 推理获取logprobs
                output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task])
                
                # 获取目标token的logprob
                if target_token < len(logprobs):
                    target_logprob = logprobs[target_token]
                    print(f"{target_logprob:.4f}")
                    text_log_prob += target_logprob
                else:
                    print(f"token超出范围")
                    break
                
                # 清理KV缓存
                kv_cache.drop(model)
            
            # 计算该文本的PPL
            text_ppl = math.exp(-text_log_prob / (len(tokens) - 1))
            print(f"  ✓ 文本PPL: {text_ppl:.4f}")
            
            total_log_prob += text_log_prob
            total_tokens += len(tokens) - 1
            valid_results += 1
            
        except Exception as e:
            print(f"  ✗ 处理错误: {e}")
            continue
    
    # 计算总体PPL
    if valid_results > 0:
        overall_ppl = math.exp(-total_log_prob / total_tokens)
        print(f"\n=== 测试结果 ===")
        print(f"有效文本数: {valid_results}/{len(test_texts)}")
        print(f"总token数: {total_tokens}")
        print(f"平均负对数概率: {-total_log_prob / total_tokens:.4f}")
        print(f"整体PPL: {overall_ppl:.4f}")
    else:
        print(f"\n✗ 没有有效的测试结果")
    
    # 清理模型
    model.destroy_model_instance()
    print("\n测试完成")

if __name__ == '__main__':
    calculate_ppl_simple()