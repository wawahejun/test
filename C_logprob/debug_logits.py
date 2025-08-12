#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试logits和logprobs的详细脚本
"""

import sys
import os
import numpy as np
from ctypes import c_uint, c_float

# 添加InfiniCore-Infer脚本路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask

def debug_logits_detailed():
    """详细调试logits和logprobs"""
    print("=== 详细logits调试 ===")
    
    # 初始化模型
    model_path = "/home/shared/models/jiuge9G4B"
    print(f"加载模型: {model_path}")
    
    model = JiugeForCauslLM(
        model_dir_path=model_path,
        device=DeviceType.DEVICE_TYPE_CPU,
        ndev=1,
        max_tokens=512
    )
    
    print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
    
    # 准备测试数据 - 使用更简单的输入
    test_texts = [
        "你",
        "你好",
        "hello",
        "the"
    ]
    
    for test_text in test_texts:
        print(f"\n--- 测试文本: '{test_text}' ---")
        tokens = model.tokenizer.encode(test_text, add_special_tokens=True)
        print(f"Token序列: {tokens}")
        
        if len(tokens) < 1:
            print("Token序列为空，跳过")
            continue
            
        # 使用所有tokens作为上下文，预测下一个token
        context_tokens = tokens
        
        print(f"上下文tokens: {context_tokens}")
        
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
            print("开始推理...")
            # 使用logprobs功能进行推理
            output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task])
            
            print(f"推理成功!")
            print(f"输出token: {output_tokens}")
            print(f"Logprobs长度: {len(logprobs)}")
            
            if len(logprobs) == model.meta.dvoc:
                # 转换为numpy数组
                logprobs_array = np.array(logprobs)
                
                # 检查nan和inf值
                nan_count = np.isnan(logprobs_array).sum()
                inf_count = np.isinf(logprobs_array).sum()
                finite_count = np.isfinite(logprobs_array).sum()
                
                print(f"NaN值数量: {nan_count}")
                print(f"Inf值数量: {inf_count}")
                print(f"有限值数量: {finite_count}")
                
                if finite_count > 0:
                    finite_values = logprobs_array[np.isfinite(logprobs_array)]
                    print(f"有限值范围: [{finite_values.min():.4f}, {finite_values.max():.4f}]")
                    print(f"有限值均值: {finite_values.mean():.4f}")
                    
                    # 检查前10个和后10个值
                    print(f"前10个logprobs: {logprobs_array[:10]}")
                    print(f"后10个logprobs: {logprobs_array[-10:]}")
                    
                    # 找到最大概率的token
                    if finite_count > 0:
                        max_idx = np.nanargmax(logprobs_array)
                        print(f"最大logprob token: {max_idx}, 值: {logprobs_array[max_idx]:.4f}")
                        
                        # 尝试解码这个token
                        try:
                            decoded = model.tokenizer.decode([max_idx])
                            print(f"最大概率token解码: '{decoded}'")
                        except:
                            print(f"无法解码token {max_idx}")
                    
                    # 验证概率和
                    if finite_count == len(logprobs_array):
                        prob_sum = np.exp(logprobs_array).sum()
                        print(f"概率和: {prob_sum:.6f} (应该接近1.0)")
                    else:
                        print("由于存在非有限值，无法计算概率和")
                else:
                    print("所有logprobs值都不是有限的!")
                    
            else:
                print(f"✗ Logprobs长度不匹配: 期望{model.meta.dvoc}, 实际{len(logprobs)}")
                
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 清理资源
            model.drop_kv_cache(kv_cache.data())
    
    # 清理模型
    model.destroy_model_instance()
    print("\n调试完成")

if __name__ == '__main__':
    debug_logits_detailed()