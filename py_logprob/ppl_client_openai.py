#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用OpenAI兼容接口计算PPL的客户端
"""

import requests
import json
import math
import argparse
import time
from typing import List, Tuple

def calculate_ppl_via_openai_api(server_url: str, text: str, model: str = "jiuge") -> float:
    """
    通过OpenAI兼容API计算单个文本的PPL
    
    Args:
        server_url: 服务器URL
        text: 输入文本
        model: 模型名称
        
    Returns:
        float: PPL值
    """
    try:
        # 调用completions接口获取logprobs
        response = requests.post(
            f"{server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "prompt": text,
                "max_tokens": 1,  # PPL计算模式
                "logprobs": 5,    # 返回top-5概率
                "temperature": 1.0
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"API请求失败: {response.status_code} - {response.text}")
            return float('inf')
        
        result = response.json()
        
        # 提取logprobs数据
        if "choices" not in result or len(result["choices"]) == 0:
            print("响应中没有choices数据")
            return float('inf')
        
        choice = result["choices"][0]
        if "logprobs" not in choice:
            print("响应中没有logprobs数据")
            return float('inf')
        
        logprobs_data = choice["logprobs"]
        token_logprobs = logprobs_data.get("token_logprobs", [])
        
        if not token_logprobs:
            print("没有获取到token logprobs")
            return float('inf')
        
        # 计算平均负对数似然
        total_log_prob = sum(token_logprobs)
        avg_log_prob = total_log_prob / len(token_logprobs)
        
        # 计算PPL
        ppl = math.exp(-avg_log_prob)
        
        return ppl
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return float('inf')
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return float('inf')
    except Exception as e:
        print(f"计算PPL时出错: {e}")
        return float('inf')

def calculate_ppl_batch_via_api(server_url: str, texts: List[str], model: str = "jiuge") -> Tuple[float, List[float]]:
    """
    批量计算PPL
    
    Args:
        server_url: 服务器URL
        texts: 文本列表
        model: 模型名称
        
    Returns:
        Tuple[float, List[float]]: (平均PPL, 每个文本的PPL列表)
    """
    print(f"开始通过API计算PPL，共 {len(texts)} 个文本样本...")
    
    ppl_values = []
    failed_count = 0
    
    for i, text in enumerate(texts):
        try:
            ppl = calculate_ppl_via_openai_api(server_url, text, model)
            
            if ppl != float('inf') and not math.isnan(ppl) and ppl > 0:
                ppl_values.append(ppl)
                print(f"  样本 {i+1}/{len(texts)}: PPL = {ppl:.4f}")
            else:
                failed_count += 1
                print(f"  样本 {i+1}/{len(texts)}: PPL计算失败 (值: {ppl})")
                
        except Exception as e:
            failed_count += 1
            print(f"  样本 {i+1}/{len(texts)}: PPL计算异常: {e}")
        
        # 每10个样本显示一次进度
        if (i + 1) % 10 == 0:
            current_avg = sum(ppl_values) / len(ppl_values) if ppl_values else float('inf')
            print(f"进度: {i+1}/{len(texts)}, 当前平均PPL: {current_avg:.4f}")
    
    if not ppl_values:
        print("所有样本的PPL计算都失败了")
        return float('inf'), []
    
    avg_ppl = sum(ppl_values) / len(ppl_values)
    success_rate = len(ppl_values) / len(texts) * 100
    
    print(f"\n PPL计算完成:")
    print(f"   成功样本: {len(ppl_values)}/{len(texts)} ({success_rate:.1f}%)")
    print(f"   失败样本: {failed_count}")
    print(f"   平均PPL: {avg_ppl:.4f}")
    print(f"   PPL范围: [{min(ppl_values):.4f}, {max(ppl_values):.4f}]")
    
    return avg_ppl, ppl_values

def load_wikitext2_data(data_path: str, max_samples: int = None) -> List[str]:
    """
    加载WikiText-2数据集
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 过滤空行和标题行
        texts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=') and len(line) > 10:
                texts.append(line)
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"加载了 {len(texts)} 个文本样本")
        return texts
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []

def test_api_connection(server_url: str) -> bool:
    """
    测试API连接
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ API服务器连接正常: {server_url}")
            return True
        else:
            print(f"❌ API服务器响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='通过OpenAI兼容API计算PPL')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                       help='API服务器URL')
    parser.add_argument('--model', type=str, default='jiuge',
                       help='模型名称')
    parser.add_argument('--data-path', type=str, 
                       default='/home/wawahejun/reasoning/c_reasoning/datasets/wikitext-2/wiki.test.tokens',
                       help='WikiText-2数据文件路径')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='最大测试样本数量')
    parser.add_argument('--output-file', type=str, default=None,
                       help='结果输出文件路径')
    
    args = parser.parse_args()
    
    print("=== 通过OpenAI兼容API计算PPL ===\n")
    print(f"API服务器: {args.server_url}")
    print(f"模型: {args.model}")
    print(f"数据路径: {args.data_path}")
    print(f"最大样本数: {args.max_samples}")
    print()
    
    # 1. 测试API连接
    if not test_api_connection(args.server_url):
        print("请确保API服务器正在运行")
        return
    
    # 2. 加载数据
    start_time = time.time()
    texts = load_wikitext2_data(args.data_path, args.max_samples)
    load_time = time.time() - start_time
    print(f"⏱数据加载耗时: {load_time:.2f}秒\n")
    
    if not texts:
        print("没有加载到有效的文本数据")
        return
    
    # 3. 计算PPL
    start_time = time.time()
    avg_ppl, ppl_values = calculate_ppl_batch_via_api(args.server_url, texts, args.model)
    calc_time = time.time() - start_time
    
    print(f"\n⏱PPL计算耗时: {calc_time:.2f}秒")
    print(f" 平均每样本耗时: {calc_time/len(texts):.3f}秒")
    
    # 4. 保存结果
    if args.output_file:
        result_data = {
            'server_url': args.server_url,
            'model': args.model,
            'data_path': args.data_path,
            'total_samples': len(texts),
            'successful_samples': len(ppl_values),
            'average_ppl': avg_ppl,
            'ppl_values': ppl_values,
            'load_time': load_time,
            'calc_time': calc_time
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {args.output_file}")
    
    print("\n=== 测试完成 ===")
    print(f"最终平均PPL: {avg_ppl:.4f}")
    
    # 5. 简单测试
    print("\n=== 简单测试 ===")
    test_text = "The quick brown fox jumps over the lazy dog."
    test_ppl = calculate_ppl_via_openai_api(args.server_url, test_text, args.model)
    print(f"测试文本: '{test_text}'")
    print(f"测试PPL: {test_ppl:.4f}")

if __name__ == "__main__":
    main()