#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的PPL调试脚本，检查多个方面：
1. 不同模型的PPL表现
2. 原始logits vs 截断后logits
3. 数据集大小对PPL的影响
4. InfiniCore-Infer实现问题
"""

import sys
import os
import math
import json
import time
import numpy as np
from typing import List, Dict, Any

# 添加路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

try:
    from jiuge import JiugeForCauslLM, DeviceType
    from infer_task import InferTask
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def load_test_data(limit=None):
    """加载测试数据"""
    data_path = '/home/wawahejun/reasoning/test/datasets/wikitext2_processed.json'
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 修复数据格式问题：data是包含texts数组的对象
        if isinstance(data, dict) and 'texts' in data:
            texts = [text for text in data['texts'] if len(text.strip()) > 10]
        elif isinstance(data, list):
            texts = [item['text'] if isinstance(item, dict) else item for item in data if len(str(item).strip()) > 10]
        else:
            print(f"未知的数据格式: {type(data)}")
            return []
        
        if limit:
            texts = texts[:limit]
            
        print(f"加载了 {len(texts)} 个文本样本")
        return texts
    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_ppl_with_debug(model, text: str, debug_info: Dict) -> Dict:
    """计算PPL并收集调试信息"""
    try:
        # 分词
        tokens = model.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 2:
            return None
            
        total_neg_log_prob = 0.0
        token_count = 0
        logprob_stats = []
        
        # 逐个计算每个token的logprob
        for i in range(len(tokens) - 1):
            input_tokens = tokens[:i+1]
            target_token = tokens[i+1]
            
            # 创建推理任务
            infer_task_obj = InferTask(
                id=0,
                tokens=input_tokens,
                max_tokens=len(input_tokens) + 1,
                temperature=1.0,
                topk=1,
                topp=1.0,
                end_tokens=model.eos_token_id
            )
            
            # 创建和绑定KV缓存
            kv_cache = model.create_kv_cache()
            infer_task_obj.bind_kvcache(kv_cache)
            
            # 执行推理获取logprobs
            output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task_obj])
            
            if logprobs and target_token < len(logprobs):
                target_logprob = logprobs[target_token]
                total_neg_log_prob += -target_logprob
                token_count += 1
                
                # 收集统计信息
                logprob_stats.append({
                    'position': i,
                    'target_token': target_token,
                    'logprob': target_logprob,
                    'prob': math.exp(target_logprob),
                    'logprobs_min': min(logprobs),
                    'logprobs_max': max(logprobs),
                    'logprobs_mean': np.mean(logprobs)
                })
            
            # 清理KV缓存
            del kv_cache
        
        if token_count == 0:
            return None
            
        avg_neg_log_prob = total_neg_log_prob / token_count
        ppl = math.exp(avg_neg_log_prob)
        
        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'token_count': token_count,
            'total_neg_log_prob': total_neg_log_prob,
            'avg_neg_log_prob': avg_neg_log_prob,
            'ppl': ppl,
            'logprob_stats': logprob_stats
        }
        
    except Exception as e:
        print(f"计算PPL失败: {e}")
        return None

def test_model_ppl(model_path: str, model_name: str, test_texts: List[str], debug_info: Dict) -> Dict:
    """测试特定模型的PPL表现"""
    print(f"\n=== 测试模型: {model_name} ===")
    
    try:
        # 加载模型
        print("正在加载模型...")
        model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=DeviceType.DEVICE_TYPE_CPU,
            ndev=1,
            max_tokens=128
        )
        
        print(f"模型加载完成，词汇表大小: {model.meta.dvoc}")
        
        results = []
        valid_ppls = []
        
        for i, text in enumerate(test_texts):
            print(f"处理样本 {i+1}/{len(test_texts)}...")
            
            result = calculate_ppl_with_debug(model, text, debug_info)
            if result:
                results.append(result)
                valid_ppls.append(result['ppl'])
                
                # 打印每个样本的结果
                print(f"  PPL: {result['ppl']:.4f}, tokens: {result['token_count']}")
        
        # 计算总体统计
        if valid_ppls:
            avg_ppl = np.mean(valid_ppls)
            median_ppl = np.median(valid_ppls)
            min_ppl = np.min(valid_ppls)
            max_ppl = np.max(valid_ppls)
            
            print(f"\n{model_name} 结果统计:")
            print(f"  有效样本数: {len(valid_ppls)}")
            print(f"  平均PPL: {avg_ppl:.4f}")
            print(f"  中位数PPL: {median_ppl:.4f}")
            print(f"  最小PPL: {min_ppl:.4f}")
            print(f"  最大PPL: {max_ppl:.4f}")
        
        # 清理模型
        del model
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'vocab_size': model.meta.dvoc if 'model' in locals() else 0,
            'valid_samples': len(valid_ppls),
            'results': results,
            'statistics': {
                'avg_ppl': avg_ppl if valid_ppls else 0,
                'median_ppl': median_ppl if valid_ppls else 0,
                'min_ppl': min_ppl if valid_ppls else 0,
                'max_ppl': max_ppl if valid_ppls else 0
            } if valid_ppls else {}
        }
        
    except Exception as e:
        print(f"测试模型 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return {'model_name': model_name, 'error': str(e)}

def main():
    """主函数"""
    print("=== 全面PPL调试分析 ===")
    
    # 调试信息收集
    debug_info = {
        'timestamp': time.time(),
        'logits_clipping_range': '[-20, 20]',  # 当前截断范围
        'test_purpose': 'comprehensive_ppl_debug'
    }
    
    # 测试不同数据集大小
    dataset_sizes = [3, 10, 20]
    
    # 可用的模型路径
    models_to_test = [
        ('/home/shared/models/jiuge9G4B', 'jiuge9G4B'),
        # 如果有其他模型，可以添加
        # ('/home/shared/models/FM9G_70B_SFT_MHA', 'FM9G_70B_SFT_MHA')
    ]
    
    all_results = []
    
    for dataset_size in dataset_sizes:
        print(f"\n{'='*50}")
        print(f"测试数据集大小: {dataset_size}")
        print(f"{'='*50}")
        
        # 加载测试数据
        test_texts = load_test_data(limit=dataset_size)
        if not test_texts:
            print(f"无法加载数据集大小为 {dataset_size} 的测试数据")
            continue
        
        dataset_results = {
            'dataset_size': dataset_size,
            'models': []
        }
        
        # 测试每个模型
        for model_path, model_name in models_to_test:
            if os.path.exists(model_path):
                result = test_model_ppl(model_path, model_name, test_texts, debug_info)
                dataset_results['models'].append(result)
            else:
                print(f"模型路径不存在: {model_path}")
        
        all_results.append(dataset_results)
    
    # 保存结果
    output_file = '/home/wawahejun/reasoning/test/results/comprehensive_ppl_debug.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'debug_info': debug_info,
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")
    
    # 打印总结
    print(f"\n{'='*50}")
    print("调试总结:")
    print(f"{'='*50}")
    
    for dataset_result in all_results:
        dataset_size = dataset_result['dataset_size']
        print(f"\n数据集大小 {dataset_size}:")
        
        for model_result in dataset_result['models']:
            model_name = model_result['model_name']
            if 'error' in model_result:
                print(f"  {model_name}: 错误 - {model_result['error']}")
            elif 'statistics' in model_result and model_result['statistics']:
                stats = model_result['statistics']
                print(f"  {model_name}: 平均PPL={stats['avg_ppl']:.4f}, 中位数PPL={stats['median_ppl']:.4f}")
            else:
                print(f"  {model_name}: 无有效结果")

if __name__ == "__main__":
    main()