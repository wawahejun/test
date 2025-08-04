#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的logprobs分布测试
验证自适应截断策略是否生效
"""

import sys
import os
import numpy as np
from collections import Counter

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM
from infer_task import InferTask

def test_simple_logprobs():
    """简单测试logprobs分布"""
    model_path = "/home/shared/models/jiuge9G4B"
    
    print("=== 简单Logprobs分布测试 ===")
    print(f"模型路径: {model_path}")
    print()
    
    # 加载模型
    print("正在加载模型...")
    model = JiugeForCauslLM(model_path)
    print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
    print()
    
    # 简单测试文本
    test_text = "Hello"
    
    # 分词
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.encode(test_text)
    print(f"测试文本: '{test_text}'")
    print(f"分词结果: {tokens}")
    
    if len(tokens) < 1:
        print("tokens数量不足")
        return
    
    # 使用第一个token作为输入
    input_tokens = tokens[:1]
    print(f"输入tokens: {input_tokens}")
    
    try:
        # 创建推理任务
        task = InferTask(
            id="simple_test",
            tokens=input_tokens,
            max_tokens=len(input_tokens) + 1,
            temperature=1.0,
            topk=1,
            topp=1.0,
            end_tokens=[2]
        )
        
        # 创建KV缓存
        kv_cache = model.create_kv_cache()
        task.bind_kvcache(kv_cache)
        
        tasks = [task]
        
        # 执行推理获取logprobs
        print("执行推理...")
        output_tokens, logprobs_flat = model.batch_infer_one_round_with_logprobs(tasks)
        
        # 检查是否有logprobs
        if logprobs_flat is not None and len(logprobs_flat) > 0:
            # logprobs_flat是一维数组，需要重新整形为 [batch_size, vocab_size]
            vocab_size = model.meta.dvoc
            batch_size = len(tasks)
            logprobs_array = np.array(logprobs_flat).reshape(batch_size, vocab_size)
            
            # 取第一个任务的logprobs
            logprobs = logprobs_array[0]
            print(f"获取到logprobs，长度: {len(logprobs)}")
            
            # 转换为numpy数组进行分析
            logprobs_array = np.array(logprobs)
            
            print(f"\n=== Logprobs基本统计 ===")
            print(f"最小值: {np.min(logprobs_array):.4f}")
            print(f"最大值: {np.max(logprobs_array):.4f}")
            print(f"均值: {np.mean(logprobs_array):.4f}")
            print(f"标准差: {np.std(logprobs_array):.4f}")
            print(f"中位数: {np.median(logprobs_array):.4f}")
            
            # 检查唯一值
            unique_values, counts = np.unique(logprobs_array, return_counts=True)
            print(f"\n=== 唯一值分析 ===")
            print(f"唯一值数量: {len(unique_values)}")
            print(f"总token数量: {len(logprobs_array)}")
            print(f"唯一值比例: {len(unique_values)/len(logprobs_array)*100:.2f}%")
            
            # 找出最常见的值
            sorted_indices = np.argsort(counts)[::-1]
            print(f"\n=== 最常见的logprob值 (前10个) ===")
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                value = unique_values[idx]
                count = counts[idx]
                percentage = count / len(logprobs_array) * 100
                print(f"{i+1}. 值: {value:.4f}, 数量: {count}, 占比: {percentage:.2f}%")
            
            # 检查是否有大量相同值（截断的迹象）
            max_count = np.max(counts)
            max_percentage = max_count / len(logprobs_array) * 100
            print(f"\n=== 截断检测 ===")
            print(f"单一值最大占比: {max_percentage:.2f}%")
            if max_percentage > 50:
                print("⚠️  警告: 超过50%的tokens具有相同logprob值，可能存在过度截断")
            elif max_percentage > 30:
                print("⚠️  注意: 超过30%的tokens具有相同logprob值，截断可能过于激进")
            else:
                print("✅ 截断程度合理")
            
            # 概率和检查
            probs = np.exp(logprobs_array)
            prob_sum = np.sum(probs)
            print(f"\n=== 概率分布检查 ===")
            print(f"所有token概率和: {prob_sum:.8f}")
            if abs(prob_sum - 1.0) < 0.01:
                print("✅ 概率分布正常")
            else:
                print(f"⚠️  概率和偏离1.0: {abs(prob_sum - 1.0):.8f}")
            
        else:
            print("❌ 未获取到logprobs")
        
        # 清理KV缓存
        model.drop_kv_cache(kv_cache.data())
        
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理模型
    del model
    print("\n模型资源已清理")

if __name__ == "__main__":
    test_simple_logprobs()