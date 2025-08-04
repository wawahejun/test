#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试原始logits分布，检查模型前向计算是否正常
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM
from infer_task import InferTask

def debug_raw_logits():
    """调试原始logits分布"""
    print("=== 原始Logits调试 ===")
    
    # 模型路径
    model_path = "/home/shared/models/jiuge9G4B"
    print(f"模型路径: {model_path}")
    
    try:
        # 加载模型
        print("\n正在加载模型...")
        model = JiugeForCauslLM(model_path)
        vocab_size = model.meta.dvoc
        print(f"模型加载完成，词汇表大小: {vocab_size}")
        
        # 测试文本
        test_text = "Hello"
        print(f"\n测试文本: '{test_text}'")
        
        # 分词
        tokens = model.tokenizer.encode(test_text)
        print(f"分词结果: {tokens}")
        
        if len(tokens) < 1:
            print("⚠️  警告: 分词结果为空")
            return
        
        # 创建推理任务
        task = InferTask(
            id=0,
            tokens=tokens[:1],  # 使用第一个token
            max_tokens=10,
            temperature=1.0,
            topk=50,
            topp=0.9,
            end_tokens=[2]  # EOS token
        )
        kv_cache = model.create_kv_cache()
        task.bind_kvcache(kv_cache)
        
        print(f"输入tokens: {task.tokens}")
        
        print("\n执行推理...")
        # 获取logprobs
        tasks = [task]
        output_tokens, logprobs_flat = model.batch_infer_one_round_with_logprobs(tasks)
        
        if not logprobs_flat:
            print("❌ 错误: 未获取到logprobs")
            return
            
        # logprobs_flat是一维数组，需要重塑为(batch_size, vocab_size)
        logprobs = np.array(logprobs_flat).reshape(len(tasks), vocab_size)
        logprobs = logprobs[0]  # 取第一个任务的logprobs
        print(f"获取到logprobs，长度: {len(logprobs)}")
        
        # 检查logprobs是否合理
        print("\n=== 原始Logprobs分析 ===")
        
        # 基本统计
        finite_mask = np.isfinite(logprobs)
        finite_logprobs = logprobs[finite_mask]
        
        print(f"有限值数量: {len(finite_logprobs)} / {len(logprobs)} ({100*len(finite_logprobs)/len(logprobs):.2f}%)")
        
        if len(finite_logprobs) == 0:
            print("❌ 错误: 所有logprobs都是非有限值")
            return
            
        print(f"最小值: {np.min(finite_logprobs):.4f}")
        print(f"最大值: {np.max(finite_logprobs):.4f}")
        print(f"均值: {np.mean(finite_logprobs):.4f}")
        print(f"标准差: {np.std(finite_logprobs):.4f}")
        print(f"中位数: {np.median(finite_logprobs):.4f}")
        
        # 检查数值范围是否合理
        if np.max(finite_logprobs) > 100:
            print("⚠️  警告: logprobs最大值异常大")
        if np.min(finite_logprobs) < -1000:
            print("⚠️  警告: logprobs最小值异常小")
        if np.abs(np.mean(finite_logprobs)) > 1000:
            print("⚠️  警告: logprobs均值异常")
            
        # 检查概率分布
        probs = np.exp(logprobs)
        prob_sum = np.sum(probs[finite_mask])
        print(f"\n概率和: {prob_sum:.8f}")
        
        if abs(prob_sum - 1.0) > 0.01:
            print("⚠️  警告: 概率和偏离1.0较多")
        else:
            print("✅ 概率分布正常")
            
        # 检查唯一值分布
        unique_logprobs = np.unique(finite_logprobs)
        print(f"\n唯一值数量: {len(unique_logprobs)}")
        print(f"唯一值比例: {100*len(unique_logprobs)/len(finite_logprobs):.2f}%")
        
        # 检查最常见值
        from collections import Counter
        counter = Counter(finite_logprobs)
        most_common = counter.most_common(5)
        
        print("\n最常见的logprob值:")
        for i, (value, count) in enumerate(most_common, 1):
            percentage = 100 * count / len(finite_logprobs)
            print(f"{i}. 值: {value:.4f}, 数量: {count}, 占比: {percentage:.2f}%")
            
        # 检查是否存在过度集中
        max_percentage = 100 * most_common[0][1] / len(finite_logprobs)
        if max_percentage > 30:
            print(f"\n⚠️  警告: {max_percentage:.2f}%的tokens具有相同logprob值")
        else:
            print(f"\n✅ logprobs分布相对均匀，最大占比: {max_percentage:.2f}%")
            
        # 尝试计算简单PPL
        if len(tokens) > 1:
            target_token = tokens[1]
            target_logprob = logprobs[target_token]
            
            print(f"\n=== PPL计算测试 ===")
            print(f"目标token: {target_token}")
            print(f"目标logprob: {target_logprob:.4f}")
            
            if np.isfinite(target_logprob) and target_logprob < 0:
                ppl = np.exp(-target_logprob)
                print(f"单token PPL: {ppl:.4f}")
                
                if ppl > 10000:
                    print("⚠️  警告: PPL值异常高")
                elif ppl < 1:
                    print("⚠️  警告: PPL值异常低")
                else:
                    print("✅ PPL值在合理范围内")
            else:
                print("❌ 错误: 目标logprob无效，无法计算PPL")
        else:
            print("\n=== PPL计算测试 ===")
            print("⚠️  警告: 只有一个token，无法计算PPL")
            
    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'kv_cache' in locals():
                model.drop_kv_cache(kv_cache)
            if 'model' in locals():
                model.destroy_model_instance()
            print("\n模型资源已清理")
        except:
            pass

if __name__ == "__main__":
    debug_raw_logits()