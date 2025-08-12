#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用WikiText-2数据集测试模型PPL
对比jiuge9G4B和FM9G_70B_SFT_MHA模型的PPL表现
"""

import sys
import os
import math
import json
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

try:
    from jiuge import JiugeForCauslLM, DeviceType
    from infer_task import InferTask
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def load_wikitext_data(file_path: str) -> list:
    """
    加载处理后的WikiText-2数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['texts']
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []

def calculate_model_ppl(model_path: str, model_name: str, test_texts: list) -> dict:
    """
    计算指定模型的PPL
    """
    print(f"\n=== 测试模型: {model_name} ===")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        return {'error': f'模型路径不存在: {model_path}'}
    
    try:
        start_time = time.time()
        
        # 加载模型
        print("正在加载模型...")
        model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=DeviceType.DEVICE_TYPE_CPU,
            ndev=1,
            max_tokens=512
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}s")
        print(f"词汇表大小: {model.meta.dvoc}")
        
        results = []
        total_tokens = 0
        total_neg_log_prob = 0.0
        successful_samples = 0
        
        for i, text in enumerate(test_texts):
            print(f"\n处理样本 {i+1}/{len(test_texts)} (长度: {len(text)} 字符)")
            
            try:
                # 分词
                tokens = model.tokenizer.encode(text, add_special_tokens=True)
                print(f"分词结果: {len(tokens)} tokens")
                
                if len(tokens) <= 1:
                    print("跳过：tokens太少")
                    continue
                
                # 计算该文本的PPL
                text_neg_log_prob = 0.0
                valid_tokens = 0
                
                for j in range(len(tokens) - 1):
                    input_tokens = tokens[:j+1]
                    target_token = tokens[j+1]
                    
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
                    
                    try:
                        # 执行推理获取logprobs
                        output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task_obj])
                        
                        if logprobs and len(logprobs) > 0 and target_token < len(logprobs):
                            target_logprob = logprobs[target_token]
                            
                            # 检查logprob是否合理
                            if not math.isnan(target_logprob) and not math.isinf(target_logprob):
                                text_neg_log_prob += -target_logprob
                                valid_tokens += 1
                            else:
                                print(f"  Token {j+1}: 无效logprob {target_logprob}")
                        else:
                            print(f"  Token {j+1}: 未获取到有效logprobs")
                    
                    except Exception as e:
                        print(f"  Token {j+1}: 推理失败 {e}")
                        continue
                
                if valid_tokens > 0:
                    text_ppl = math.exp(text_neg_log_prob / valid_tokens)
                    print(f"样本PPL: {text_ppl:.4f} (基于 {valid_tokens} 个有效tokens)")
                    
                    results.append({
                        'sample_id': i + 1,
                        'text_length': len(text),
                        'token_count': valid_tokens,
                        'ppl': text_ppl,
                        'neg_log_prob': text_neg_log_prob,
                        'avg_neg_log_prob': text_neg_log_prob / valid_tokens
                    })
                    
                    total_tokens += valid_tokens
                    total_neg_log_prob += text_neg_log_prob
                    successful_samples += 1
                else:
                    print("样本处理失败：没有有效的logprobs")
                    
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
        
        # 计算整体统计
        if total_tokens > 0 and successful_samples > 0:
            overall_ppl = math.exp(total_neg_log_prob / total_tokens)
            individual_ppls = [r['ppl'] for r in results]
            avg_ppl = sum(individual_ppls) / len(individual_ppls)
            
            print(f"\n=== {model_name} 最终结果 ===")
            print(f"成功处理样本: {successful_samples}/{len(test_texts)}")
            print(f"总tokens: {total_tokens}")
            print(f"平均负对数概率: {total_neg_log_prob / total_tokens:.4f}")
            print(f"整体PPL: {overall_ppl:.4f}")
            print(f"平均PPL: {avg_ppl:.4f}")
            print(f"PPL范围: {min(individual_ppls):.4f} - {max(individual_ppls):.4f}")
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'load_time': load_time,
                'vocab_size': model.meta.dvoc,
                'successful_samples': successful_samples,
                'total_samples': len(test_texts),
                'total_tokens': total_tokens,
                'overall_ppl': overall_ppl,
                'average_ppl': avg_ppl,
                'ppl_range': [min(individual_ppls), max(individual_ppls)],
                'avg_neg_log_prob': total_neg_log_prob / total_tokens,
                'sample_results': results
            }
        else:
            print(f"\n{model_name} 测试失败：没有有效结果")
            return {'error': '没有有效结果', 'model_name': model_name}
            
    except Exception as e:
        print(f"模型 {model_name} 测试失败: {e}")
        return {'error': str(e), 'model_name': model_name}
    finally:
        # 清理资源
        try:
            if 'model' in locals():
                del model
            print(f"\n{model_name} 模型资源已清理")
        except:
            pass

def main():
    # 加载WikiText-2数据
    data_file = "/home/wawahejun/reasoning/test/datasets/wikitext2_processed.json"
    
    if not os.path.exists(data_file):
        print(f"错误：数据文件不存在: {data_file}")
        print("请先运行 process_wikitext.py 处理数据")
        return
    
    print("=== WikiText-2 PPL 对比测试 ===")
    test_texts = load_wikitext_data(data_file)
    
    if not test_texts:
        print("加载测试数据失败")
        return
    
    print(f"加载了 {len(test_texts)} 个测试样本")
    
    # 测试模型配置
    models_to_test = [
        {
            'name': 'jiuge9G4B',
            'path': '/home/shared/models/jiuge9G4B'
        },
        {
            'name': 'FM9G_70B_SFT_MHA',
            'path': '/home/shared/models/FM9G_70B_SFT_MHA'
        }
    ]
    
    # 测试结果
    all_results = []
    
    for model_config in models_to_test:
        result = calculate_model_ppl(
            model_config['path'], 
            model_config['name'], 
            test_texts
        )
        all_results.append(result)
    
    # 保存对比结果
    comparison_data = {
        'dataset': 'WikiText-2 (Processed)',
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sample_count': len(test_texts),
        'models': all_results
    }
    
    output_file = "/home/wawahejun/reasoning/test/results/wikitext_ppl_comparison.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 对比结果总结 ===")
    for result in all_results:
        if 'error' not in result:
            print(f"{result['model_name']}: PPL = {result['overall_ppl']:.4f}")
        else:
            print(f"{result['model_name']}: 测试失败 - {result['error']}")
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()