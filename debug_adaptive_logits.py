#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试自适应logits截断策略的效果
分析新的logits分布和PPL计算
"""

import sys
import os
import json
import numpy as np
from collections import Counter

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM
from infer_task import InferTask

def analyze_logprobs_distribution(logprobs):
    """分析logprobs的分布情况"""
    logprobs = np.array(logprobs)
    
    # 基本统计
    stats = {
        'min': float(np.min(logprobs)),
        'max': float(np.max(logprobs)),
        'mean': float(np.mean(logprobs)),
        'std': float(np.std(logprobs)),
        'median': float(np.median(logprobs))
    }
    
    # 分位数分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = float(np.percentile(logprobs, p))
    
    # 值分布统计
    unique_values, counts = np.unique(logprobs, return_counts=True)
    value_distribution = {}
    total_count = len(logprobs)
    
    # 找出最常见的值
    sorted_indices = np.argsort(counts)[::-1]  # 降序排列
    top_values = []
    for i in sorted_indices[:10]:  # 前10个最常见的值
        value = float(unique_values[i])
        count = int(counts[i])
        percentage = count / total_count * 100
        top_values.append({
            'value': value,
            'count': count,
            'percentage': percentage
        })
    
    stats['unique_values_count'] = len(unique_values)
    stats['top_values'] = top_values
    
    # 检查是否有大量相同值（截断的迹象）
    max_count = np.max(counts)
    max_percentage = max_count / total_count * 100
    stats['max_single_value_percentage'] = max_percentage
    
    return stats

def test_adaptive_clipping():
    """测试自适应截断策略的效果"""
    model_path = "/home/shared/models/jiuge9G4B"
    
    print("=== 自适应Logits截断策略调试 ===")
    print(f"模型路径: {model_path}")
    print()
    
    # 加载模型
    print("正在加载模型...")
    model = JiugeForCauslLM(model_path)
    print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
    print()
    
    # 测试文本
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is",
        "Python programming",
        "Natural language processing"
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"=== 测试文本 {i}: '{text}' ===")
        
        # 分词 - 使用transformers tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokens = tokenizer.encode(text)
        print(f"分词结果: {tokens} ({len(tokens)} tokens)")
        
        if len(tokens) < 2:
            print("跳过：tokens数量不足")
            continue
        
        # 获取第一个预测的logprobs
        input_tokens = tokens[:-1]
        target_token = tokens[-1]
        
        print(f"输入tokens: {input_tokens}")
        print(f"目标token: {target_token}")
        
        # 执行推理获取logprobs
        try:
            # 创建推理任务
            task = InferTask(
                id=f"test_{i}",
                tokens=input_tokens,
                max_tokens=len(input_tokens) + 1,
                temperature=1.0,
                topk=1,
                topp=1.0,
                end_tokens=[2]  # EOS token
            )
            tasks = [task]
            
            # 执行推理
            model.batch_infer_one_round_with_logprobs(tasks)
            output_token = task.output_tokens[0] if task.output_tokens else None
            logprobs = task.logprobs if hasattr(task, 'logprobs') and task.logprobs else None
            
            if logprobs is None:
                print("未获取到logprobs")
                continue
            print(f"输出token: {output_token}")
            print(f"Logprobs数组长度: {len(logprobs)}")
            
            # 分析logprobs分布
            stats = analyze_logprobs_distribution(logprobs)
            
            print(f"\n--- Logprobs分布统计 ---")
            print(f"最小值: {stats['min']:.4f}")
            print(f"最大值: {stats['max']:.4f}")
            print(f"均值: {stats['mean']:.4f}")
            print(f"标准差: {stats['std']:.4f}")
            print(f"中位数: {stats['median']:.4f}")
            print(f"唯一值数量: {stats['unique_values_count']}")
            print(f"单一值最大占比: {stats['max_single_value_percentage']:.2f}%")
            
            print(f"\n--- 分位数分析 ---")
            for p in [1, 5, 25, 50, 75, 95, 99]:
                print(f"P{p}: {stats[f'p{p}']:.4f}")
            
            print(f"\n--- 最常见的logprob值 (前5个) ---")
            for j, item in enumerate(stats['top_values'][:5]):
                print(f"{j+1}. 值: {item['value']:.4f}, 数量: {item['count']}, 占比: {item['percentage']:.2f}%")
            
            # 目标token的logprob
            target_logprob = logprobs[target_token]
            target_prob = np.exp(target_logprob)
            print(f"\n--- 目标Token分析 ---")
            print(f"目标token {target_token} 的logprob: {target_logprob:.4f}")
            print(f"目标token {target_token} 的概率: {target_prob:.8f}")
            
            # 计算PPL
            ppl = np.exp(-target_logprob)
            print(f"单token PPL: {ppl:.4f}")
            
            # 检查概率和
            probs = np.exp(logprobs)
            prob_sum = np.sum(probs)
            print(f"所有token概率和: {prob_sum:.8f}")
            
            # 保存结果
            result = {
                'text': text,
                'tokens': tokens,
                'input_tokens': input_tokens,
                'target_token': target_token,
                'output_token': output_token,
                'target_logprob': target_logprob,
                'target_prob': target_prob,
                'single_token_ppl': ppl,
                'prob_sum': prob_sum,
                'logprobs_stats': stats
            }
            results.append(result)
            
        except Exception as e:
            print(f"推理失败: {e}")
        
        print("\n" + "="*60 + "\n")
    
    # 保存详细结果
    output_file = "/home/wawahejun/reasoning/test/results/adaptive_clipping_debug.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': model_path,
            'vocab_size': model.meta.dvoc,
            'test_results': results,
            'summary': {
                'total_tests': len(results),
                'avg_unique_values': np.mean([r['logprobs_stats']['unique_values_count'] for r in results]) if results else 0,
                'avg_max_single_value_percentage': np.mean([r['logprobs_stats']['max_single_value_percentage'] for r in results]) if results else 0,
                'avg_prob_sum': np.mean([r['prob_sum'] for r in results]) if results else 0
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存到: {output_file}")
    
    # 清理模型
    del model
    print("模型资源已清理")

if __name__ == "__main__":
    test_adaptive_clipping()