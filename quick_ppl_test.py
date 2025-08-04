#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速PPL测试 - 使用少量样本验证功能
"""

import sys
import os
import math
import json
import time

# 添加路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

try:
    from jiuge import JiugeForCauslLM, DeviceType
    from infer_task import InferTask
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def quick_ppl_test(model_path: str, model_name: str) -> dict:
    """
    快速PPL测试 - 只测试几个简短文本
    """
    print(f"\n=== 快速测试模型: {model_name} ===")
    print(f"模型路径: {model_path}")
    
    # 简短的测试文本
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is"
    ]
    
    try:
        start_time = time.time()
        
        # 加载模型
        print("正在加载模型...")
        model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=DeviceType.DEVICE_TYPE_CPU,
            ndev=1,
            max_tokens=128
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}s")
        print(f"词汇表大小: {model.meta.dvoc}")
        
        results = []
        
        for i, text in enumerate(test_texts):
            print(f"\n处理文本 {i+1}: '{text}'")
            
            try:
                # 分词
                tokens = model.tokenizer.encode(text, add_special_tokens=True)
                print(f"分词结果: {tokens} ({len(tokens)} tokens)")
                
                if len(tokens) <= 1:
                    print("跳过：tokens太少")
                    continue
                
                # 只计算前几个token的PPL
                max_tokens_to_test = min(5, len(tokens) - 1)
                text_neg_log_prob = 0.0
                valid_tokens = 0
                
                for j in range(max_tokens_to_test):
                    input_tokens = tokens[:j+1]
                    target_token = tokens[j+1]
                    
                    print(f"  Token {j+1}: input={input_tokens}, target={target_token}")
                    
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
                        
                        if logprobs and len(logprobs) > target_token:
                            target_logprob = logprobs[target_token]
                            print(f"    logprob[{target_token}] = {target_logprob:.4f}")
                            
                            # 检查logprob是否合理
                            if not math.isnan(target_logprob) and not math.isinf(target_logprob):
                                text_neg_log_prob += -target_logprob
                                valid_tokens += 1
                                print(f"    累计neg_log_prob: {text_neg_log_prob:.4f}")
                            else:
                                print(f"    无效logprob: {target_logprob}")
                        else:
                            print(f"    未获取到target_token={target_token}的logprob")
                    
                    except Exception as e:
                        print(f"    推理失败: {e}")
                        continue
                
                if valid_tokens > 0:
                    text_ppl = math.exp(text_neg_log_prob / valid_tokens)
                    print(f"文本PPL: {text_ppl:.4f} (基于 {valid_tokens} 个tokens)")
                    
                    results.append({
                        'text': text,
                        'tokens': tokens,
                        'valid_tokens': valid_tokens,
                        'ppl': text_ppl,
                        'neg_log_prob': text_neg_log_prob,
                        'avg_neg_log_prob': text_neg_log_prob / valid_tokens
                    })
                else:
                    print("文本处理失败：没有有效的logprobs")
                    
            except Exception as e:
                print(f"处理文本时出错: {e}")
                continue
        
        # 计算整体统计
        if results:
            ppls = [r['ppl'] for r in results]
            avg_ppl = sum(ppls) / len(ppls)
            
            print(f"\n=== {model_name} 快速测试结果 ===")
            print(f"成功处理文本: {len(results)}/{len(test_texts)}")
            print(f"平均PPL: {avg_ppl:.4f}")
            print(f"PPL范围: {min(ppls):.4f} - {max(ppls):.4f}")
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'load_time': load_time,
                'vocab_size': model.meta.dvoc,
                'successful_texts': len(results),
                'total_texts': len(test_texts),
                'average_ppl': avg_ppl,
                'ppl_range': [min(ppls), max(ppls)],
                'results': results
            }
        else:
            print(f"\n{model_name} 快速测试失败：没有有效结果")
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
    print("=== 快速PPL测试 ===")
    
    # 测试jiuge模型
    jiuge_result = quick_ppl_test(
        '/home/shared/models/jiuge9G4B',
        'jiuge9G4B'
    )
    
    # 保存结果
    output_file = "/home/wawahejun/reasoning/test/results/quick_ppl_test.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    test_data = {
        'test_type': 'Quick PPL Test',
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'jiuge_result': jiuge_result
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 快速测试总结 ===")
    if 'error' not in jiuge_result:
        print(f"jiuge9G4B: 平均PPL = {jiuge_result['average_ppl']:.4f}")
    else:
        print(f"jiuge9G4B: 测试失败 - {jiuge_result['error']}")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()