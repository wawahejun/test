#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始GEMM操作调试脚本
专门检查从GPU拷贝的原始logits数值
"""

import sys
import os
import numpy as np
import ctypes
from ctypes import c_float, c_uint, POINTER

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM
from infer_task import InferTask
from libinfinicore_infer import (
    create_kv_cache,
    drop_kv_cache,
    infer_batch_with_logprobs,
    KVCacheCStruct
)

def debug_raw_gemm_output():
    """直接调试GEMM操作的原始输出"""
    print("=== 原始GEMM输出调试 ===")
    
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
            
        # 使用底层C接口直接调用
        input_tokens = (c_uint * 1)(tokens[0])
        req_lens = (c_uint * 1)(1)
        req_pos = (c_uint * 1)(0)
        
        # 创建KV cache
        kv_cache_ptr = create_kv_cache(model.model_instance)
        kv_caches = (ctypes.POINTER(KVCacheCStruct) * 1)(kv_cache_ptr)
        
        # 采样参数
        temperature = (c_float * 1)(1.0)
        topk = (c_uint * 1)(50)
        topp = (c_float * 1)(0.9)
        
        # 输出缓冲区
        output = (c_uint * 1)()
        logprobs_out = (c_float * vocab_size)()
        
        print("\n执行底层推理...")
        
        # 直接调用C接口
        infer_batch_with_logprobs(
            model.model_instance,
            input_tokens, 1,  # tokens, ntok
            req_lens, 1, req_pos,  # req_lens, nreq, req_pos
            kv_caches,
            temperature, topk, topp,
            output, logprobs_out
        )
        
        print("推理完成，分析原始logprobs...")
        
        # 转换为numpy数组进行分析
        logprobs_array = np.array([logprobs_out[i] for i in range(vocab_size)])
        
        print(f"\n=== 原始Logprobs分析 ===")
        print(f"数组长度: {len(logprobs_array)}")
        print(f"数据类型: {logprobs_array.dtype}")
        
        # 基本统计
        finite_mask = np.isfinite(logprobs_array)
        finite_count = np.sum(finite_mask)
        inf_count = np.sum(np.isinf(logprobs_array))
        nan_count = np.sum(np.isnan(logprobs_array))
        
        print(f"\n=== 数值有效性 ===")
        print(f"有限值: {finite_count}/{len(logprobs_array)} ({finite_count/len(logprobs_array)*100:.1f}%)")
        print(f"无穷值: {inf_count}")
        print(f"NaN值: {nan_count}")
        
        if finite_count > 0:
            finite_logprobs = logprobs_array[finite_mask]
            print(f"\n=== 有限值统计 ===")
            print(f"最小值: {np.min(finite_logprobs):.6e}")
            print(f"最大值: {np.max(finite_logprobs):.6e}")
            print(f"均值: {np.mean(finite_logprobs):.6e}")
            print(f"标准差: {np.std(finite_logprobs):.6e}")
            print(f"中位数: {np.median(finite_logprobs):.6e}")
            
            # 检查数值范围
            min_val = np.min(finite_logprobs)
            max_val = np.max(finite_logprobs)
            
            print(f"\n=== 数值范围检查 ===")
            if min_val < -1000 or max_val > 100:
                print(f"⚠️  警告: 数值范围异常")
                print(f"   当前范围: [{min_val:.2e}, {max_val:.2e}]")
                print(f"   正常范围应该在: [-100, 0]")
                
                # 检查是否所有值都是相同的异常值
                unique_values = np.unique(finite_logprobs)
                print(f"   唯一值数量: {len(unique_values)}")
                if len(unique_values) < 100:  # 如果唯一值太少
                    print(f"   前10个唯一值: {unique_values[:10]}")
                    
            else:
                print("✅ 数值范围正常")
                
        # 检查概率分布
        try:
            if finite_count > 0:
                # 只对有限值计算概率
                max_logprob = np.max(finite_logprobs)
                # 为了避免溢出，先减去最大值
                shifted_logprobs = finite_logprobs - max_logprob
                probs = np.exp(shifted_logprobs)
                prob_sum = np.sum(probs)
                
                print(f"\n=== 概率分布检查 ===")
                print(f"最大logprob: {max_logprob:.6e}")
                print(f"概率和: {prob_sum:.8f}")
                
                if prob_sum > 0:
                    normalized_probs = probs / prob_sum
                    entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-12))
                    print(f"熵值: {entropy:.4f}")
                    print(f"最大概率: {np.max(normalized_probs):.6f}")
                    
                    if entropy < 1.0:
                        print("⚠️  警告: 熵值过低，分布过于集中")
                    elif entropy > 15.0:
                        print("⚠️  警告: 熵值过高，分布过于平均")
                    else:
                        print("✅ 熵值正常")
                        
        except Exception as e:
            print(f"❌ 概率分布计算失败: {e}")
            
        # 分析最常见的值
        print(f"\n=== 值分布分析 ===")
        unique_vals, counts = np.unique(logprobs_array, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]  # 按出现次数降序排列
        
        print("最常见的logprob值:")
        for i in range(min(5, len(unique_vals))):
            idx = sorted_indices[i]
            val = unique_vals[idx]
            count = counts[idx]
            percentage = count / len(logprobs_array) * 100
            print(f"  {i+1}. 值: {val:.6e}, 出现次数: {count}, 占比: {percentage:.2f}%")
            
        # 清理资源
        drop_kv_cache(model.model_instance, kv_cache_ptr)
        model.destroy_model_instance()
        
        print("\n=== 分析结论 ===")
        if finite_count == len(logprobs_array) and min_val > -1000 and max_val < 100:
            print("✅ logprobs数值正常")
        else:
            print("❌ logprobs数值异常，可能的原因:")
            print("   1. 模型权重文件损坏")
            print("   2. GEMM操作数值溢出")
            print("   3. 模型量化问题")
            print("   4. 硬件计算精度问题")
            
    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def compare_with_reference_model():
    """与参考模型进行对比（如果有的话）"""
    print("\n=== 参考模型对比 ===")
    
    # 这里可以加载一个已知正常的模型进行对比
    # 或者使用PyTorch/Transformers库加载相同的模型
    print("暂时跳过参考模型对比（需要额外的模型文件）")

def main():
    """主函数"""
    print("=== 原始GEMM操作深度调试 ===")
    print("目标: 检查GEMM操作直接输出的logits数值")
    
    # 调试原始GEMM输出
    debug_raw_gemm_output()
    
    # 与参考模型对比
    compare_with_reference_model()
    
    print("\n=== 调试建议 ===")
    print("如果发现logprobs数值异常，建议:")
    print("1. 检查模型文件完整性（重新下载）")
    print("2. 检查模型加载过程中的数值精度设置")
    print("3. 检查GEMM操作的输入（隐藏状态）是否正常")
    print("4. 考虑使用不同的数值精度（float16 vs float32）")

if __name__ == "__main__":
    main()