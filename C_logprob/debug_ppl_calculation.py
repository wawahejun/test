#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试PPL计算问题
检查logprob值和PPL计算是否正确
"""

import sys
import os
import math
import numpy as np

# 添加脚本路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask

def debug_ppl_calculation():
    print("=== 调试PPL计算问题 ===")
    
    # 加载模型
    model = JiugeForCauslLM(
        model_dir_path='/home/shared/models/jiuge9G4B',
        device=DeviceType.DEVICE_TYPE_CPU,
        ndev=1,
        max_tokens=512
    )
    
    print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
    
    # 简单测试文本
    test_text = "你好世界"
    tokens = model.tokenizer.encode(test_text, add_special_tokens=True)
    print(f"\n测试文本: {test_text}")
    print(f"Token序列: {tokens}")
    print(f"Token数量: {len(tokens)}")
    
    if len(tokens) < 2:
        print("Token数量太少，无法测试")
        return
    
    print(f"\n=== 详细logprob分析 ===")
    
    total_log_prob = 0.0
    valid_positions = 0
    
    for pos in range(1, len(tokens)):
        context_tokens = tokens[:pos]
        target_token = tokens[pos]
        
        print(f"\n--- 位置 {pos} ---")
        print(f"上下文tokens: {context_tokens}")
        print(f"目标token: {target_token}")
        
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
        
        try:
            # 推理获取logprobs
            output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task])
            
            print(f"输出token: {output_tokens}")
            print(f"Logprobs数组长度: {len(logprobs)}")
            
            # 转换为numpy数组进行分析
            logprobs_array = np.array(logprobs)
            print(f"Logprobs范围: [{logprobs_array.min():.4f}, {logprobs_array.max():.4f}]")
            print(f"Logprobs均值: {logprobs_array.mean():.4f}")
            print(f"Logprobs标准差: {logprobs_array.std():.4f}")
            
            # 检查目标token的logprob
            if target_token < len(logprobs):
                target_logprob = logprobs[target_token]
                print(f"目标token {target_token} 的logprob: {target_logprob:.4f}")
                
                # 检查logprob是否合理
                if np.isfinite(target_logprob):
                    total_log_prob += target_logprob
                    valid_positions += 1
                    
                    # 计算对应的概率
                    prob = math.exp(target_logprob)
                    print(f"对应概率: {prob:.6e}")
                    
                    # 检查是否为log概率
                    if target_logprob > 0:
                        print(f"⚠️  警告: logprob > 0，这不应该发生！")
                    
                    if target_logprob < -1000:
                        print(f"⚠️  警告: logprob过小 ({target_logprob:.4f})，可能导致数值问题")
                        
                else:
                    print(f"✗ 目标token的logprob无效: {target_logprob}")
            else:
                print(f"✗ 目标token {target_token} 超出词汇表范围")
            
            # 验证概率和是否接近1
            probs = np.exp(logprobs_array)
            prob_sum = probs.sum()
            print(f"概率和: {prob_sum:.6f}")
            
            if abs(prob_sum - 1.0) > 0.01:
                print(f"⚠️  警告: 概率和不接近1，可能存在问题")
            
        except Exception as e:
            print(f"✗ 推理错误: {e}")
        finally:
            # 清理KV缓存
            kv_cache.drop(model)
    
    # 计算PPL
    if valid_positions > 0:
        avg_nll = -total_log_prob / valid_positions
        ppl = math.exp(avg_nll)
        
        print(f"\n=== PPL计算结果 ===")
        print(f"有效位置数: {valid_positions}")
        print(f"总log概率: {total_log_prob:.4f}")
        print(f"平均负log似然: {avg_nll:.4f}")
        print(f"PPL: {ppl:.4f}")
        
        # 检查PPL是否合理
        if ppl > 10000:
            print(f"⚠️  警告: PPL值过大 ({ppl:.4f})，可能存在计算问题")
        elif ppl < 1:
            print(f"⚠️  警告: PPL值小于1，这不应该发生")
        else:
            print(f"✓ PPL值在合理范围内")
    else:
        print(f"\n✗ 没有有效的位置用于计算PPL")
    
    # 清理模型
    model.destroy_model_instance()
    print(f"\n调试完成")

if __name__ == '__main__':
    debug_ppl_calculation()