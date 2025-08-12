#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPL对比测试脚本
支持多种推理框架的PPL对比测试：
- InfiniCore-Infer (自研C++引擎)
- InfiniM (自研Rust引擎) 
- Ollama (官方框架)
- vLLM (官方框架)

使用方法:
    python ppl_comparison.py --config config.json --model jiuge9G4B --dataset sample
"""

import argparse
import json
import math
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

# 导入自定义模块
from openai_adapter import create_infinicore_adapter, OpenAIAdapter


@dataclass
class ComparisonResult:
    """对比测试结果"""
    framework: str
    model_name: str
    dataset_name: str
    ppl: float
    token_count: int
    processing_time: float
    error: Optional[str] = None
    details: Optional[Dict] = None


class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        
    def is_available(self) -> bool:
        """检查Ollama是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def calculate_ppl(self, model_name: str, text: str) -> Dict[str, Any]:
        """使用Ollama计算PPL
        
        注意：Ollama不直接支持logprobs，这里使用近似方法
        """
        try:
            # Ollama的generate接口
            payload = {
                "model": model_name,
                "prompt": text,
                "stream": False,
                "options": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "num_predict": 1  # 只预测一个token
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                # 由于Ollama不提供logprobs，我们无法计算真实的PPL
                # 这里返回一个占位符结果
                return {
                    'ppl': float('nan'),  # 表示无法计算
                    'token_count': len(text.split()),  # 近似token数
                    'processing_time': end_time - start_time,
                    'error': 'Ollama does not support logprobs calculation',
                    'response': data.get('response', '')
                }
            else:
                return {
                    'error': f'Ollama request failed: {response.status_code}'
                }
                
        except Exception as e:
            return {
                'error': f'Ollama calculation failed: {str(e)}'
            }


class VLLMClient:
    """vLLM客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def is_available(self) -> bool:
        """检查vLLM是否可用"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [model['id'] for model in data.get('data', [])]
        except:
            pass
        return []
    
    def calculate_ppl(self, model_name: str, text: str) -> Dict[str, Any]:
        """使用vLLM计算PPL"""
        try:
            # 使用OpenAI兼容的completions接口
            payload = {
                "model": model_name,
                "prompt": text,
                "max_tokens": 1,
                "logprobs": 5,  # 返回top-5 logprobs
                "temperature": 1.0,
                "top_p": 1.0
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # 从响应中提取logprobs
                choices = data.get('choices', [])
                if choices and 'logprobs' in choices[0]:
                    logprobs_data = choices[0]['logprobs']
                    token_logprobs = logprobs_data.get('token_logprobs', [])
                    
                    if token_logprobs:
                        # 计算平均负对数似然
                        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
                        if valid_logprobs:
                            avg_nll = -sum(valid_logprobs) / len(valid_logprobs)
                            ppl = math.exp(avg_nll)
                            
                            return {
                                'ppl': ppl,
                                'token_count': len(valid_logprobs),
                                'processing_time': end_time - start_time,
                                'avg_nll': avg_nll,
                                'logprobs': valid_logprobs
                            }
                
                return {
                    'error': 'No valid logprobs in vLLM response',
                    'response': data
                }
            else:
                return {
                    'error': f'vLLM request failed: {response.status_code}'
                }
                
        except Exception as e:
            return {
                'error': f'vLLM calculation failed: {str(e)}'
            }


class PPLComparator:
    """PPL对比器"""
    
    def __init__(self, config_path: str):
        """初始化对比器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化客户端
        self.ollama_client = None
        self.vllm_client = None
        self.infinicore_adapter = None
        
        # 检查并初始化可用的框架
        self._initialize_frameworks()
    
    def _initialize_frameworks(self):
        """初始化可用的推理框架"""
        # 初始化Ollama
        if self.config['benchmark_frameworks']['ollama']['enabled']:
            ollama_url = self.config['benchmark_frameworks']['ollama']['url']
            self.ollama_client = OllamaClient(ollama_url)
            if not self.ollama_client.is_available():
                print(f"警告: Ollama不可用 ({ollama_url})")
                self.ollama_client = None
        
        # 初始化vLLM
        if self.config['benchmark_frameworks']['vllm']['enabled']:
            vllm_url = self.config['benchmark_frameworks']['vllm']['url']
            self.vllm_client = VLLMClient(vllm_url)
            if not self.vllm_client.is_available():
                print(f"警告: vLLM不可用 ({vllm_url})")
                self.vllm_client = None
        
        # 初始化InfiniCore-Infer
        if self.config['engines']['infinicore']['enabled']:
            try:
                self.infinicore_adapter = create_infinicore_adapter()
                print("InfiniCore-Infer适配器初始化成功")
            except Exception as e:
                print(f"警告: InfiniCore-Infer初始化失败: {e}")
                self.infinicore_adapter = None
    
    def load_dataset(self, dataset_name: str) -> List[str]:
        """加载数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            文本列表
        """
        dataset_config = self.config['datasets'].get(dataset_name)
        if not dataset_config:
            raise ValueError(f"未找到数据集配置: {dataset_name}")
        
        dataset_path = dataset_config['path']
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.endswith('.json'):
                dataset = json.load(f)
            else:
                dataset = [line.strip() for line in f if line.strip()]
        
        # 限制样本数量
        max_samples = self.config['test_settings']['max_samples_per_dataset']
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]
        
        # 处理字典格式的数据
        texts = []
        for item in dataset:
            if isinstance(item, dict):
                text = item.get('text', item.get('content', str(item)))
            else:
                text = str(item)
            texts.append(text)
        
        return texts
    
    def test_infinicore(self, model_name: str, texts: List[str]) -> ComparisonResult:
        """测试InfiniCore-Infer"""
        if not self.infinicore_adapter:
            return ComparisonResult(
                framework="InfiniCore-Infer",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error="InfiniCore-Infer不可用"
            )
        
        try:
            # 获取模型路径
            model_config = None
            for model in self.config['models']['test_models']:
                if model['name'] == model_name and 'infinicore' in model['compatible_engines']:
                    model_config = model
                    break
            
            if not model_config:
                return ComparisonResult(
                    framework="InfiniCore-Infer",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型 {model_name} 不支持InfiniCore-Infer"
                )
            
            model_path = os.path.join(self.config['models']['base_dir'], model_config['path'])
            
            # 加载模型
            load_result = self.infinicore_adapter.load_model(model_path)
            if not load_result['success']:
                return ComparisonResult(
                    framework="InfiniCore-Infer",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型加载失败: {load_result['error']}"
                )
            
            # 计算PPL
            total_ppl = 0.0
            total_tokens = 0
            valid_samples = 0
            start_time = time.time()
            
            for text in texts:
                ppl_result = self.infinicore_adapter.calculate_ppl(text)
                
                if 'error' not in ppl_result and not math.isnan(ppl_result['ppl']) and not math.isinf(ppl_result['ppl']):
                    total_ppl += ppl_result['ppl']
                    total_tokens += ppl_result['token_count']
                    valid_samples += 1
            
            end_time = time.time()
            
            if valid_samples == 0:
                return ComparisonResult(
                    framework="InfiniCore-Infer",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=end_time - start_time,
                    error="没有有效的样本"
                )
            
            avg_ppl = total_ppl / valid_samples
            
            return ComparisonResult(
                framework="InfiniCore-Infer",
                model_name=model_name,
                dataset_name="",
                ppl=avg_ppl,
                token_count=total_tokens,
                processing_time=end_time - start_time,
                details={
                    'valid_samples': valid_samples,
                    'total_samples': len(texts)
                }
            )
            
        except Exception as e:
            return ComparisonResult(
                framework="InfiniCore-Infer",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error=str(e)
            )
    
    def test_ollama(self, model_name: str, texts: List[str]) -> ComparisonResult:
        """测试Ollama"""
        if not self.ollama_client:
            return ComparisonResult(
                framework="Ollama",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error="Ollama不可用"
            )
        
        try:
            # 获取Ollama模型名称映射
            ollama_model_name = self.config['benchmark_frameworks']['ollama']['models'].get(model_name)
            if not ollama_model_name:
                return ComparisonResult(
                    framework="Ollama",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型 {model_name} 在Ollama中没有对应的映射"
                )
            
            # 检查模型是否可用
            available_models = self.ollama_client.list_models()
            if ollama_model_name not in available_models:
                return ComparisonResult(
                    framework="Ollama",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型 {ollama_model_name} 在Ollama中不可用"
                )
            
            # 计算PPL（注意：Ollama不支持真实的PPL计算）
            start_time = time.time()
            
            # 由于Ollama不支持logprobs，我们只能返回一个占位符结果
            end_time = time.time()
            
            return ComparisonResult(
                framework="Ollama",
                model_name=model_name,
                dataset_name="",
                ppl=float('nan'),  # 表示无法计算
                token_count=sum(len(text.split()) for text in texts),
                processing_time=end_time - start_time,
                error="Ollama不支持logprobs，无法计算真实PPL"
            )
            
        except Exception as e:
            return ComparisonResult(
                framework="Ollama",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error=str(e)
            )
    
    def test_vllm(self, model_name: str, texts: List[str]) -> ComparisonResult:
        """测试vLLM"""
        if not self.vllm_client:
            return ComparisonResult(
                framework="vLLM",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error="vLLM不可用"
            )
        
        try:
            # 获取vLLM模型名称映射
            vllm_model_name = self.config['benchmark_frameworks']['vllm']['models'].get(model_name)
            if not vllm_model_name:
                return ComparisonResult(
                    framework="vLLM",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型 {model_name} 在vLLM中没有对应的映射"
                )
            
            # 检查模型是否可用
            available_models = self.vllm_client.list_models()
            if vllm_model_name not in available_models:
                return ComparisonResult(
                    framework="vLLM",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=0,
                    error=f"模型 {vllm_model_name} 在vLLM中不可用"
                )
            
            # 计算PPL
            total_ppl = 0.0
            total_tokens = 0
            valid_samples = 0
            start_time = time.time()
            
            for text in texts:
                ppl_result = self.vllm_client.calculate_ppl(vllm_model_name, text)
                
                if 'error' not in ppl_result and not math.isnan(ppl_result.get('ppl', float('nan'))):
                    total_ppl += ppl_result['ppl']
                    total_tokens += ppl_result['token_count']
                    valid_samples += 1
            
            end_time = time.time()
            
            if valid_samples == 0:
                return ComparisonResult(
                    framework="vLLM",
                    model_name=model_name,
                    dataset_name="",
                    ppl=float('inf'),
                    token_count=0,
                    processing_time=end_time - start_time,
                    error="没有有效的样本"
                )
            
            avg_ppl = total_ppl / valid_samples
            
            return ComparisonResult(
                framework="vLLM",
                model_name=model_name,
                dataset_name="",
                ppl=avg_ppl,
                token_count=total_tokens,
                processing_time=end_time - start_time,
                details={
                    'valid_samples': valid_samples,
                    'total_samples': len(texts)
                }
            )
            
        except Exception as e:
            return ComparisonResult(
                framework="vLLM",
                model_name=model_name,
                dataset_name="",
                ppl=float('inf'),
                token_count=0,
                processing_time=0,
                error=str(e)
            )
    
    def run_comparison(self, model_name: str, dataset_name: str) -> List[ComparisonResult]:
        """运行对比测试
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            
        Returns:
            对比结果列表
        """
        print(f"开始对比测试: 模型={model_name}, 数据集={dataset_name}")
        
        # 加载数据集
        try:
            texts = self.load_dataset(dataset_name)
            print(f"数据集加载成功，共 {len(texts)} 个样本")
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return []
        
        results = []
        
        # 测试InfiniCore-Infer
        print("\n测试 InfiniCore-Infer...")
        infinicore_result = self.test_infinicore(model_name, texts)
        infinicore_result.dataset_name = dataset_name
        results.append(infinicore_result)
        print(f"InfiniCore-Infer 结果: PPL={infinicore_result.ppl:.4f}, 错误={infinicore_result.error}")
        
        # 测试Ollama
        print("\n测试 Ollama...")
        ollama_result = self.test_ollama(model_name, texts)
        ollama_result.dataset_name = dataset_name
        results.append(ollama_result)
        print(f"Ollama 结果: PPL={ollama_result.ppl:.4f}, 错误={ollama_result.error}")
        
        # 测试vLLM
        print("\n测试 vLLM...")
        vllm_result = self.test_vllm(model_name, texts)
        vllm_result.dataset_name = dataset_name
        results.append(vllm_result)
        print(f"vLLM 结果: PPL={vllm_result.ppl:.4f}, 错误={vllm_result.error}")
        
        return results
    
    def cleanup(self):
        """清理资源"""
        if self.infinicore_adapter:
            self.infinicore_adapter.cleanup()


def save_comparison_results(results: List[ComparisonResult], output_path: str):
    """保存对比结果
    
    Args:
        results: 对比结果列表
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为可序列化的格式
    serializable_results = []
    for result in results:
        serializable_results.append({
            'framework': result.framework,
            'model_name': result.model_name,
            'dataset_name': result.dataset_name,
            'ppl': result.ppl if not math.isnan(result.ppl) and not math.isinf(result.ppl) else None,
            'token_count': result.token_count,
            'processing_time': result.processing_time,
            'error': result.error,
            'details': result.details
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': serializable_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"对比结果已保存到: {output_path}")


def print_comparison_summary(results: List[ComparisonResult]):
    """打印对比结果摘要
    
    Args:
        results: 对比结果列表
    """
    print("\n=== PPL 对比结果摘要 ===")
    print(f"{'框架':<15} {'模型':<15} {'数据集':<15} {'PPL':<10} {'Token数':<10} {'时间(s)':<10} {'状态':<20}")
    print("-" * 100)
    
    for result in results:
        ppl_str = f"{result.ppl:.4f}" if not math.isnan(result.ppl) and not math.isinf(result.ppl) else "N/A"
        status = "成功" if result.error is None else "失败"
        
        print(f"{result.framework:<15} {result.model_name:<15} {result.dataset_name:<15} "
              f"{ppl_str:<10} {result.token_count:<10} {result.processing_time:<10.2f} {status:<20}")
        
        if result.error:
            print(f"  错误: {result.error}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='PPL对比测试')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--model', required=True, help='模型名称')
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--output', help='结果输出文件路径')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    try:
        # 创建对比器
        comparator = PPLComparator(args.config)
        
        # 运行对比测试
        results = comparator.run_comparison(args.model, args.dataset)
        
        # 打印结果摘要
        print_comparison_summary(results)
        
        # 保存结果
        if args.output:
            save_comparison_results(results, args.output)
        else:
            # 默认保存路径
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = f"./results/comparison_{args.model}_{args.dataset}_{timestamp}.json"
            save_comparison_results(results, output_path)
        
        # 清理资源
        comparator.cleanup()
        
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()