#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型权重和GEMM操作调试脚本
用于深入分析导致logits异常的根本原因
"""

import sys
import os
import numpy as np
import torch
import json
from pathlib import Path

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM
from infer_task import InferTask

def check_model_weights_integrity():
    """检查模型权重文件的完整性和数值分布"""
    print("\n=== 模型权重完整性检查 ===")
    
    model_path = "/home/shared/models/jiuge9G4B"
    
    try:
        # 检查模型文件结构
        model_dir = Path(model_path)
        if not model_dir.exists():
            print(f"❌ 错误: 模型目录不存在: {model_path}")
            return False
            
        print(f"模型目录: {model_path}")
        
        # 列出模型文件
        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        print(f"找到权重文件: {len(model_files)}个")
        
        for file in model_files[:3]:  # 只显示前3个文件
            print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
            
        # 检查配置文件
        config_file = model_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"模型配置: {config.get('model_type', 'unknown')} - {config.get('hidden_size', 'unknown')}维")
        else:
            print("⚠️  警告: 未找到config.json文件")
            
        return True
        
    except Exception as e:
        print(f"❌ 检查模型文件时出错: {e}")
        return False

def analyze_output_embedding_weights():
    """分析输出嵌入层权重的数值分布"""
    print("\n=== 输出嵌入层权重分析 ===")
    
    model_path = "/home/shared/models/jiuge9G4B"
    
    try:
        # 加载模型以访问权重
        print("正在加载模型...")
        model = JiugeForCauslLM(model_path)
        
        # 检查是否可以直接访问权重
        if hasattr(model, 'weights') and hasattr(model.weights, 'w_out_embd'):
            print("✅ 可以访问输出嵌入层权重")
            
            # 这里需要从C++层获取权重数据，但目前接口可能不支持
            # 我们先检查模型的基本信息
            vocab_size = model.meta.dvoc
            hidden_size = model.meta.d
            
            print(f"词汇表大小: {vocab_size}")
            print(f"隐藏层维度: {hidden_size}")
            print(f"预期权重矩阵形状: ({vocab_size}, {hidden_size})")
            
        else:
            print("⚠️  警告: 无法直接访问权重数据")
            
        # 清理资源
        model.destroy_model_instance()
        
    except Exception as e:
        print(f"❌ 分析权重时出错: {e}")
        import traceback
        traceback.print_exc()

def debug_gemm_computation():
    """调试GEMM计算过程"""
    print("\n=== GEMM计算调试 ===")
    
    model_path = "/home/shared/models/jiuge9G4B"
    
    try:
        # 加载模型
        print("正在加载模型...")
        model = JiugeForCauslLM(model_path)
        vocab_size = model.meta.dvoc
        hidden_size = model.meta.d
        
        print(f"模型参数: vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        # 准备测试输入
        test_text = "Hello"
        tokens = model.tokenizer.encode(test_text)
        print(f"测试文本: '{test_text}' -> tokens: {tokens}")
        
        if len(tokens) < 1:
            print("❌ 错误: 分词结果为空")
            return
            
        # 创建推理任务
        task = InferTask(
            id=0,
            tokens=tokens[:1],
            max_tokens=10,
            temperature=1.0,
            topk=50,
            topp=0.9,
            end_tokens=[2]
        )
        kv_cache = model.create_kv_cache()
        task.bind_kvcache(kv_cache)
        
        print(f"输入tokens: {task.tokens}")
        
        # 执行推理并获取logprobs
        print("\n执行推理...")
        tasks = [task]
        output_tokens, logprobs_flat = model.batch_infer_one_round_with_logprobs(tasks)
        
        if not logprobs_flat:
            print("❌ 错误: 未获取到logprobs")
            return
            
        # 分析logprobs
        logprobs = np.array(logprobs_flat).reshape(len(tasks), vocab_size)
        logprobs = logprobs[0]
        
        print(f"\n=== Logprobs统计信息 ===")
        print(f"形状: {logprobs.shape}")
        print(f"数据类型: {logprobs.dtype}")
        print(f"最小值: {np.min(logprobs):.2e}")
        print(f"最大值: {np.max(logprobs):.2e}")
        print(f"均值: {np.mean(logprobs):.2e}")
        print(f"标准差: {np.std(logprobs):.2e}")
        
        # 检查异常值
        finite_count = np.sum(np.isfinite(logprobs))
        inf_count = np.sum(np.isinf(logprobs))
        nan_count = np.sum(np.isnan(logprobs))
        
        print(f"\n=== 数值有效性检查 ===")
        print(f"有限值: {finite_count}/{len(logprobs)} ({finite_count/len(logprobs)*100:.1f}%)")
        print(f"无穷值: {inf_count}")
        print(f"NaN值: {nan_count}")
        
        # 检查数值范围合理性
        if finite_count > 0:
            finite_logprobs = logprobs[np.isfinite(logprobs)]
            if np.min(finite_logprobs) < -1000 or np.max(finite_logprobs) > 100:
                print("⚠️  警告: logprobs数值范围异常")
                print(f"   正常范围应该在[-100, 0]之间")
                print(f"   当前范围: [{np.min(finite_logprobs):.2e}, {np.max(finite_logprobs):.2e}]")
            else:
                print("✅ logprobs数值范围正常")
                
        # 检查概率分布
        try:
            probs = np.exp(logprobs)
            prob_sum = np.sum(probs)
            print(f"\n=== 概率分布检查 ===")
            print(f"概率和: {prob_sum:.8f}")
            if abs(prob_sum - 1.0) < 1e-6:
                print("✅ 概率分布归一化正常")
            else:
                print(f"⚠️  警告: 概率分布未正确归一化，偏差: {abs(prob_sum - 1.0):.2e}")
        except:
            print("❌ 错误: 无法计算概率分布（数值溢出）")
            
        # 清理资源
        model.drop_kv_cache(kv_cache)
        model.destroy_model_instance()
        
    except Exception as e:
        print(f"❌ GEMM调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def check_numerical_precision():
    """检查数值精度问题"""
    print("\n=== 数值精度检查 ===")
    
    # 检查float32的数值范围
    print(f"Float32数值范围:")
    print(f"  最小正数: {np.finfo(np.float32).tiny:.2e}")
    print(f"  最大数: {np.finfo(np.float32).max:.2e}")
    print(f"  机器精度: {np.finfo(np.float32).eps:.2e}")
    
    # 模拟大数值的log_softmax计算
    print(f"\n=== 大数值log_softmax测试 ===")
    
    # 创建一些测试logits
    test_cases = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),  # 正常范围
        np.array([100.0, 200.0, 300.0], dtype=np.float32),  # 较大值
        np.array([1000.0, 2000.0, 3000.0], dtype=np.float32),  # 很大值
        np.array([1e10, 2e10, 3e10], dtype=np.float32),  # 极大值
    ]
    
    for i, logits in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: logits = {logits}")
        
        try:
            # 数值稳定的log_softmax
            max_logit = np.max(logits)
            shifted_logits = logits - max_logit
            log_sum_exp = max_logit + np.log(np.sum(np.exp(shifted_logits)))
            log_probs = logits - log_sum_exp
            
            print(f"  max_logit: {max_logit:.2e}")
            print(f"  log_sum_exp: {log_sum_exp:.2e}")
            print(f"  log_probs: {log_probs}")
            print(f"  概率和: {np.sum(np.exp(log_probs)):.8f}")
            
            if np.any(np.isinf(log_probs)) or np.any(np.isnan(log_probs)):
                print("  ⚠️  警告: 出现无穷或NaN值")
            else:
                print("  ✅ 计算正常")
                
        except Exception as e:
            print(f"  ❌ 计算失败: {e}")

def main():
    """主函数"""
    print("=== 模型权重和GEMM操作深度调试 ===")
    print("目标: 找出导致logits异常的根本原因")
    
    # 1. 检查模型文件完整性
    if not check_model_weights_integrity():
        print("\n❌ 模型文件检查失败，终止调试")
        return
        
    # 2. 分析输出嵌入层权重
    analyze_output_embedding_weights()
    
    # 3. 调试GEMM计算过程
    debug_gemm_computation()
    
    # 4. 检查数值精度问题
    check_numerical_precision()
    
    print("\n=== 调试完成 ===")
    print("请查看上述输出，重点关注:")
    print("1. 模型文件是否完整")
    print("2. logprobs的数值范围是否异常")
    print("3. 概率分布是否正确归一化")
    print("4. 是否存在数值精度问题")

if __name__ == "__main__":
    main()