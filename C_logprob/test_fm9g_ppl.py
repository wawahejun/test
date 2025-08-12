#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FM9G_70B_SFT_MHA模型PPL测试脚本
测试是否是jiuge模型本身导致的PPL异常问题
"""

import sys
import os
import math
import json
from pathlib import Path

# 添加路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

try:
    from jiuge import JiugeForCauslLM, DeviceType
    from infer_task import InferTask
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def calculate_ppl_with_logprobs(model_path: str, test_texts: list) -> dict:
    """
    使用logprobs功能计算PPL
    """
    print(f"加载模型: {model_path}")
    
    try:
        # 加载模型
        model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=DeviceType.DEVICE_TYPE_CPU,
            ndev=1,
            max_tokens=512
        )
        print(f"模型加载成功")
        print(f"词汇表大小: {model.meta.dvoc}")
        
        results = []
        total_tokens = 0
        total_neg_log_prob = 0.0
        
        for i, text in enumerate(test_texts):
            print(f"\n处理文本 {i+1}/{len(test_texts)}: {text[:50]}...")
            
            try:
                
                # 分词
                tokens = model.tokenizer.encode(text, add_special_tokens=True)
                print(f"分词结果: {len(tokens)} tokens")
                
                if len(tokens) <= 1:
                    print("跳过：tokens太少")
                    continue
                
                # 逐个token推理并获取logprobs
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
                    
                    # 执行推理获取logprobs
                    output_tokens, logprobs = model.batch_infer_one_round_with_logprobs([infer_task_obj])
                    
                    if logprobs and len(logprobs) > 0:
                        # 检查目标token的logprob
                        if target_token < len(logprobs):
                            target_logprob = logprobs[target_token]
                            
                            # 检查logprob是否合理
                            if not math.isnan(target_logprob) and not math.isinf(target_logprob):
                                text_neg_log_prob += -target_logprob
                                valid_tokens += 1
                                
                                if j < 3:  # 打印前几个token的详细信息
                                    print(f"  Token {j+1}: {target_token} -> logprob: {target_logprob:.4f}")
                            else:
                                print(f"  Token {j+1}: {target_token} -> 无效logprob: {target_logprob}")
                        else:
                            print(f"  Token {j+1}: {target_token} -> token超出logprobs范围")
                    else:
                        print(f"  Token {j+1}: 未获取到logprobs")
                
                if valid_tokens > 0:
                    text_ppl = math.exp(text_neg_log_prob / valid_tokens)
                    print(f"文本PPL: {text_ppl:.4f} (基于 {valid_tokens} 个有效tokens)")
                    
                    results.append({
                        'text': text[:100],
                        'ppl': text_ppl,
                        'tokens': valid_tokens,
                        'neg_log_prob': text_neg_log_prob
                    })
                    
                    total_tokens += valid_tokens
                    total_neg_log_prob += text_neg_log_prob
                else:
                    print("文本处理失败：没有有效的logprobs")
                    
            except Exception as e:
                print(f"处理文本时出错: {e}")
                continue
        
        # 计算整体PPL
        if total_tokens > 0:
            overall_ppl = math.exp(total_neg_log_prob / total_tokens)
            print(f"\n=== 最终结果 ===")
            print(f"总tokens: {total_tokens}")
            print(f"平均负对数概率: {total_neg_log_prob / total_tokens:.4f}")
            print(f"整体PPL: {overall_ppl:.4f}")
            
            if results:
                individual_ppls = [r['ppl'] for r in results]
                avg_ppl = sum(individual_ppls) / len(individual_ppls)
                print(f"平均PPL: {avg_ppl:.4f}")
                print(f"PPL范围: {min(individual_ppls):.4f} - {max(individual_ppls):.4f}")
            
            return {
                'overall_ppl': overall_ppl,
                'total_tokens': total_tokens,
                'avg_neg_log_prob': total_neg_log_prob / total_tokens,
                'results': results
            }
        else:
            print("\n没有有效的结果")
            return {'error': '没有有效的tokens'}
            
    except Exception as e:
        print(f"模型加载或处理失败: {e}")
        return {'error': str(e)}
    finally:
        # 清理资源
        try:
            if 'model' in locals():
                del model
            print("\n模型资源已清理")
        except:
            pass

def main():
    # 测试文本
    test_texts = [
        "你好世界，这是一个测试。",
        "人工智能是计算机科学的一个分支。",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "今天天气很好，适合出去散步。",
        "Natural language processing enables computers to understand human language."
    ]
    
    model_path = "/home/shared/models/FM9G_70B_SFT_MHA"
    
    print("=== FM9G_70B_SFT_MHA模型PPL测试 ===")
    print(f"模型路径: {model_path}")
    print(f"测试文本数量: {len(test_texts)}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        return
    
    # 计算PPL
    results = calculate_ppl_with_logprobs(model_path, test_texts)
    
    # 保存结果
    output_file = "/home/wawahejun/reasoning/test/results/fm9g_ppl_test.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()