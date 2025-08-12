#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用JiugeForCauslLM计算PPL，无需启动HTTP服务器
"""

import os
import sys
import argparse
import math
import time
from typing import List, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask, KVCache

DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
}

def calculate_ppl_direct(model: JiugeForCauslLM, text: str) -> float:
    """
    直接使用模型计算单个文本的PPL
    
    Args:
        model: 模型实例
        text: 输入文本
        
    Returns:
        float: PPL值
    """
    try:
        # 编码文本
        tokens = model.tokenizer.encode(text)
        if len(tokens) <= 1:
            return float('inf')
        
        total_log_prob = 0.0
        token_count = 0
        
        # 逐个token计算logprobs
        for i in range(1, len(tokens)):
            context_tokens = tokens[:i]
            target_token = tokens[i]
            
            # 创建推理任务
            task = InferTask(
                id=f"ppl_task_{i}",
                tokens=context_tokens,
                max_tokens=1,
                temperature=1.0,
                topk=50,
                topp=1.0,
                end_tokens=model.eos_token_id
            )
            task.bind_kvcache(KVCache(model))
            
            try:
                # 执行推理获取logits
                tasks = [task]
                from libinfinicore_infer import DataType, destroy_logits_output
                import ctypes
                import numpy as np
                
                # 尝试使用FP32获得更高精度
                output_tokens, logits_output = model.batch_infer_with_logits(tasks, DataType.INFINI_DTYPE_F32)
                
                if logits_output and hasattr(logits_output, 'contents'):
                    contents = logits_output.contents
                    batch_size = contents.batch_size
                    seq_len = contents.sequence_length
                    vocab_size = contents.vocab_size
                    
                    print(f"[DEBUG] Token {i}: logits结构 - batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}, 目标token={target_token}")
                    
                    if contents.logits and vocab_size > target_token:
                        # 获取logits数据
                        total_elements = batch_size * seq_len * vocab_size
                        logits_ptr = ctypes.cast(contents.logits, ctypes.POINTER(ctypes.c_float))
                        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(total_elements,))
                        
                        # 获取最后一个token位置的logits（用于预测下一个token）
                        last_token_start = (seq_len - 1) * vocab_size
                        last_token_logits = logits_array[last_token_start:last_token_start + vocab_size]
                        
                        print(f"[DEBUG] Token {i}: logits范围=[{np.min(last_token_logits):.3f}, {np.max(last_token_logits):.3f}]")
                        
                        # 检查logits分布的方差
                        logits_std = np.std(last_token_logits)
                        logits_mean = np.mean(last_token_logits)
                        print(f"[DEBUG] Token {i}: logits统计 - mean={logits_mean:.3f}, std={logits_std:.3f}")
                        
                        # 数值稳定化：限制logits范围以避免溢出
                        # 将异常大的logits值clamp到合理范围
                        last_token_logits = np.clip(last_token_logits, -50.0, 50.0)
                        print(f"[DEBUG] Token {i}: clamp后logits范围=[{np.min(last_token_logits):.3f}, {np.max(last_token_logits):.3f}]")
                        
                        # 检查clamp后的分布
                        clamp_std = np.std(last_token_logits)
                        clamp_mean = np.mean(last_token_logits)
                        print(f"[DEBUG] Token {i}: clamp后统计 - mean={clamp_mean:.3f}, std={clamp_std:.3f}")
                        
                        # 检查logits是否有效
                        if np.all(np.isfinite(last_token_logits)):
                            # 计算softmax概率
                            max_logit = np.max(last_token_logits)
                            exp_logits = np.exp(last_token_logits - max_logit)
                            sum_exp = np.sum(exp_logits)
                            
                            if sum_exp > 0 and np.isfinite(sum_exp):
                                probs = exp_logits / sum_exp
                                target_prob = probs[target_token]
                                
                                print(f"[DEBUG] Token {i}: target_token={target_token}, 原始logit={last_token_logits[target_token]:.3f}")
                                
                                # 显示top-5概率用于调试
                                top5_indices = np.argsort(probs)[-5:][::-1]
                                top5_probs = [(int(idx), float(probs[idx])) for idx in top5_indices]
                                print(f"[DEBUG] Token {i}: Top-5概率: {top5_probs}")
                                
                                if target_prob > 1e-30 and target_prob < 1.0:  # 降低阈值，避免log(0)
                                    log_prob = math.log(target_prob)
                                    total_log_prob += log_prob
                                    token_count += 1
                                    print(f"[DEBUG] Token {i}: 真实概率={target_prob:.6f}, log_prob={log_prob:.6f}")
                                else:
                                    print(f"[DEBUG] Token {i}: 概率过小({target_prob:.2e})，使用fallback")
                                    fallback_prob = 1.0 / len(model.tokenizer)
                                    total_log_prob += math.log(fallback_prob)
                                    token_count += 1
                            else:
                                print(f"[DEBUG] Token {i}: softmax计算异常，sum_exp={sum_exp}，使用fallback")
                                fallback_prob = 1.0 / len(model.tokenizer)
                                total_log_prob += math.log(fallback_prob)
                                token_count += 1
                        else:
                            print(f"[DEBUG] Token {i}: logits包含无效值，使用fallback")
                            fallback_prob = 1.0 / len(model.tokenizer)
                            total_log_prob += math.log(fallback_prob)
                            token_count += 1
                    else:
                        print(f"[DEBUG] Token {i}: logprobs指针无效或target_token超出范围，使用fallback")
                        fallback_prob = 1.0 / len(model.tokenizer)
                        total_log_prob += math.log(fallback_prob)
                        token_count += 1
                    
                    # 清理logits_output
                    destroy_logits_output(logits_output)
                else:
                    print(f"[DEBUG] Token {i}: 未获取到有效的logits_output，使用fallback")
                    fallback_prob = 1.0 / len(model.tokenizer)
                    total_log_prob += math.log(fallback_prob)
                    token_count += 1
                    
            finally:
                # 清理KV缓存
                if task._kv_cache:
                    task._kv_cache.drop(model)
        
        if token_count == 0:
            return float('inf')
            
        # 计算PPL
        avg_log_prob = total_log_prob / token_count
        ppl = math.exp(-avg_log_prob)
        return ppl
        
    except Exception as e:
        print(f"计算PPL时出错: {e}")
        return float('inf')

def load_wikitext2_data(data_path: str, max_samples: int = None) -> List[str]:
    """
    加载WikiText-2数据集
    
    Args:
        data_path: 数据文件路径
        max_samples: 最大样本数量
        
    Returns:
        List[str]: 文本样本列表
    """
    texts = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('='):  # 跳过标题行
                    texts.append(line)
                    if max_samples and len(texts) >= max_samples:
                        break
    except FileNotFoundError:
        print(f"数据文件未找到: {data_path}")
        return []
    
    return texts

def main():
    parser = argparse.ArgumentParser(description='直接计算PPL，无需HTTP服务器')
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=list(DEVICE_TYPE_MAP.keys()),
                       help='设备类型')
    parser.add_argument('--data-path', type=str, 
                       default='/home/wawahejun/reasoning/c_reasoning/datasets/wikitext-2/wiki.test.tokens',
                       help='WikiText-2数据文件路径')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='最大测试样本数量')
    parser.add_argument('--text', type=str, default=None,
                       help='直接测试指定文本')
    parser.add_argument('--quick-test', action='store_true',
                       help='运行快速测试（使用预定义的简单句子）')
    
    args = parser.parse_args()
    
    print("=== 直接PPL计算器 ===\n")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {args.device}")
    print()
    
    # 加载模型
    print("正在加载模型...")
    start_time = time.time()
    
    device_type = DEVICE_TYPE_MAP[args.device]
    model = JiugeForCauslLM(
        model_dir_path=args.model_path,
        device=device_type,
        ndev=1
    )
    
    load_time = time.time() - start_time
    print(f"模型加载完成 (耗时: {load_time:.2f}秒)")
    print(f"最大上下文长度: {model.max_context_len()}")
    print(f"词汇表大小: {len(model.tokenizer)}")
    print()
    
    if args.text:
        # 测试单个文本
        print(f"测试文本: '{args.text}'")
        start_time = time.time()
        ppl = calculate_ppl_direct(model, args.text)
        calc_time = time.time() - start_time
        print(f"PPL: {ppl:.4f} (耗时: {calc_time:.2f}秒)")
    elif args.quick_test:
        # 快速测试模式
        test_sentences = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "I love machine learning",
            "This is a simple test sentence",
            "Python is a great programming language"
        ]
        
        print("=== 快速测试模式 ===")
        print(f"测试 {len(test_sentences)} 个简单句子...\n")
        
        ppls = []
        for i, sentence in enumerate(test_sentences, 1):
            print(f"测试句子 {i}: '{sentence}'")
            start_time = time.time()
            ppl = calculate_ppl_direct(model, sentence)
            calc_time = time.time() - start_time
            
            if not math.isinf(ppl):
                ppls.append(ppl)
                print(f"  PPL: {ppl:.4f} (耗时: {calc_time:.3f}秒)")
            else:
                print(f"  PPL: 计算失败 (耗时: {calc_time:.3f}秒)")
            print()
        
        if ppls:
            avg_ppl = sum(ppls) / len(ppls)
            print(f"快速测试结果:")
            print(f"  成功: {len(ppls)}/{len(test_sentences)} 个句子")
            print(f"  平均PPL: {avg_ppl:.4f}")
            print(f"  PPL范围: [{min(ppls):.4f}, {max(ppls):.4f}]")
        else:
            print("所有测试句子的PPL计算都失败了")
    else:
        # 批量测试
        print(f"数据路径: {args.data_path}")
        print(f"最大样本数: {args.max_samples}")
        print()
        
        # 加载数据
        texts = load_wikitext2_data(args.data_path, args.max_samples)
        if not texts:
            print("❌ 无法加载数据")
            return
            
        print(f"加载了 {len(texts)} 个文本样本")
        print()
        
        # 计算PPL
        print(f"开始计算PPL，共 {len(texts)} 个文本样本...")
        start_time = time.time()
        
        ppls = []
        success_count = 0
        
        for i, text in enumerate(texts, 1):
            ppl = calculate_ppl_direct(model, text)
            if not math.isinf(ppl):
                ppls.append(ppl)
                success_count += 1
                print(f"  样本 {i}/{len(texts)}: PPL = {ppl:.4f}")
            else:
                print(f"  样本 {i}/{len(texts)}: PPL计算失败")
        
        calc_time = time.time() - start_time
        
        if ppls:
            avg_ppl = sum(ppls) / len(ppls)
            min_ppl = min(ppls)
            max_ppl = max(ppls)
            
            print(f"\n PPL计算完成:")
            print(f"   成功样本: {success_count}/{len(texts)} ({success_count/len(texts)*100:.1f}%)")
            print(f"   失败样本: {len(texts) - success_count}")
            print(f"   平均PPL: {avg_ppl:.4f}")
            print(f"   PPL范围: [{min_ppl:.4f}, {max_ppl:.4f}]")
        else:
            print("\n所有样本的PPL计算都失败了")
            
        print(f"\n⏱PPL计算耗时: {calc_time:.2f}秒")
        print(f" 平均每样本耗时: {calc_time/len(texts):.3f}秒")
        
        print("\n=== 计算完成 ===")
        if ppls:
            print(f"最终平均PPL: {avg_ppl:.4f}")
    
    print("\n提示: 使用 --quick-test 参数可以快速测试几个简单句子")

if __name__ == "__main__":
    main()