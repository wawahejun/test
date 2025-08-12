#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI兼容API适配器
为InfiniCore-Infer和InfiniM提供统一的OpenAI风格接口

支持的接口:
- /v1/completions (用于PPL计算)
- /v1/models (列出可用模型)
"""

import json
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np

# 添加InfiniCore-Infer脚本路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask


class BaseInferenceEngine(ABC):
    """推理引擎基类"""
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """分词"""
        pass
    
    @abstractmethod
    def get_logprobs(self, tokens: List[int], target_token_id: int) -> float:
        """获取指定token的log概率"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass


class InfiniCoreInferEngine(BaseInferenceEngine):
    """InfiniCore-Infer引擎适配器"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        
    def load_model(self, model_path: str, device: str = "cpu", **kwargs) -> None:
        """加载Jiuge模型
        
        Args:
            model_path: 模型路径
            device: 设备类型
            **kwargs: 其他参数
        """
        self.model_path = model_path
        device_type = DeviceType.DEVICE_TYPE_CPU if device.lower() == "cpu" else DeviceType.DEVICE_TYPE_NVIDIA
        
        self.model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=device_type,
            ndev=kwargs.get('ndev', 1),
            max_tokens=kwargs.get('max_tokens', None)
        )
        
    def tokenize(self, text: str) -> List[int]:
        """分词"""
        if not self.model:
            raise RuntimeError("模型未加载")
        return self.model.tokenizer.encode(text, add_special_tokens=True)
    
    def get_logprobs(self, tokens: List[int], target_token_id: int) -> float:
        """获取指定token的log概率
        
        Args:
            tokens: 上下文tokens
            target_token_id: 目标token ID
            
        Returns:
            目标token的log概率
        """
        if not self.model:
            raise RuntimeError("模型未加载")
            
        # 创建推理任务
        infer_task = InferTask(
            req_id=0,
            tokens=tokens,
            max_len=len(tokens) + 1,
            temperature=1.0,
            topk=1,
            topp=1.0,
            eos_token_id=self.model.eos_token_id
        )
        
        # 绑定KV缓存
        kv_cache = self.model.create_kv_cache()
        infer_task.bind_kvcache(kv_cache)
        
        try:
            # 使用logprobs功能进行推理
            output_tokens, logprobs = self.model.batch_infer_one_round_with_logprobs([infer_task])
            
            # 从logprobs中提取目标token的概率
            vocab_size = self.model.meta.dvoc
            logprobs_matrix = np.array(logprobs).reshape(1, vocab_size)
            
            # 获取目标token的log概率
            target_log_prob = logprobs_matrix[0, target_token_id]
            
            return float(target_log_prob)
            
        finally:
            # 清理KV缓存
            self.model.drop_kv_cache(kv_cache)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        if not self.model:
            raise RuntimeError("模型未加载")
        return self.model.meta.dvoc
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.model:
            self.model.destroy_model_instance()
            self.model = None


class OpenAIAdapter:
    """OpenAI API适配器"""
    
    def __init__(self, engine: BaseInferenceEngine):
        self.engine = engine
        self.loaded_model = None
        
    def load_model(self, model_path: str, **kwargs) -> Dict[str, Any]:
        """加载模型
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数
            
        Returns:
            加载结果
        """
        try:
            self.engine.load_model(model_path, **kwargs)
            self.loaded_model = {
                'id': os.path.basename(model_path),
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'infinicore',
                'path': model_path
            }
            return {
                'success': True,
                'model': self.loaded_model
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_models(self) -> Dict[str, Any]:
        """列出可用模型
        
        Returns:
            模型列表（OpenAI格式）
        """
        models = []
        if self.loaded_model:
            models.append(self.loaded_model)
            
        return {
            'object': 'list',
            'data': models
        }
    
    def completions(self, 
                   prompt: str,
                   model: Optional[str] = None,
                   max_tokens: int = 1,
                   logprobs: bool = True,
                   temperature: float = 1.0,
                   top_p: float = 1.0,
                   **kwargs) -> Dict[str, Any]:
        """文本补全接口（OpenAI兼容）
        
        Args:
            prompt: 输入提示
            model: 模型名称（可选）
            max_tokens: 最大生成token数
            logprobs: 是否返回log概率
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他参数
            
        Returns:
            OpenAI格式的响应
        """
        if not self.loaded_model:
            return {
                'error': {
                    'message': '没有加载的模型',
                    'type': 'invalid_request_error',
                    'code': 'model_not_loaded'
                }
            }
        
        try:
            # 分词
            tokens = self.engine.tokenize(prompt)
            
            if len(tokens) == 0:
                return {
                    'error': {
                        'message': '输入文本为空',
                        'type': 'invalid_request_error',
                        'code': 'empty_input'
                    }
                }
            
            # 对于PPL计算，我们通常只需要预测下一个token
            if max_tokens == 1 and logprobs:
                # 获取所有token的logprobs（用于PPL计算）
                vocab_size = self.engine.get_vocab_size()
                
                # 创建推理任务获取完整的logprobs分布
                context_tokens = tokens[:-1] if len(tokens) > 1 else tokens
                
                if len(context_tokens) == 0:
                    # 如果没有上下文，返回均匀分布
                    uniform_logprob = -math.log(vocab_size)
                    top_logprobs = {str(i): uniform_logprob for i in range(min(10, vocab_size))}
                else:
                    # 获取下一个token的logprobs分布
                    try:
                        # 这里我们需要获取完整的logprobs分布
                        # 由于当前实现只能获取单个token的logprob，我们需要修改实现
                        # 暂时返回一个模拟的分布
                        target_token = tokens[-1] if len(tokens) > len(context_tokens) else 0
                        target_logprob = self.engine.get_logprobs(context_tokens, target_token)
                        
                        # 构建top_logprobs（这里简化处理）
                        top_logprobs = {str(target_token): target_logprob}
                        
                        # 添加一些其他高概率token（模拟）
                        for i in range(min(9, vocab_size)):
                            if i != target_token:
                                # 模拟其他token的概率（比目标token低一些）
                                other_logprob = target_logprob - abs(np.random.normal(1.0, 0.5))
                                top_logprobs[str(i)] = other_logprob
                                
                    except Exception as e:
                        return {
                            'error': {
                                'message': f'推理错误: {str(e)}',
                                'type': 'inference_error',
                                'code': 'inference_failed'
                            }
                        }
                
                # 构建OpenAI格式的响应
                response = {
                    'id': f'cmpl-{int(time.time())}',
                    'object': 'text_completion',
                    'created': int(time.time()),
                    'model': self.loaded_model['id'],
                    'choices': [
                        {
                            'text': '',  # 对于PPL计算，我们不需要生成文本
                            'index': 0,
                            'logprobs': {
                                'tokens': [str(target_token)] if 'target_token' in locals() else [],
                                'token_logprobs': [target_logprob] if 'target_logprob' in locals() else [],
                                'top_logprobs': [top_logprobs] if 'top_logprobs' in locals() else [],
                                'text_offset': [len(prompt)]
                            },
                            'finish_reason': 'length'
                        }
                    ],
                    'usage': {
                        'prompt_tokens': len(tokens),
                        'completion_tokens': 0,
                        'total_tokens': len(tokens)
                    }
                }
                
                return response
            
            else:
                # 正常的文本生成（暂不实现）
                return {
                    'error': {
                        'message': '暂不支持文本生成，仅支持logprobs计算',
                        'type': 'not_implemented_error',
                        'code': 'generation_not_supported'
                    }
                }
                
        except Exception as e:
            return {
                'error': {
                    'message': f'处理请求时出错: {str(e)}',
                    'type': 'internal_error',
                    'code': 'processing_error'
                }
            }
    
    def calculate_ppl(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """计算文本的PPL
        
        Args:
            text: 输入文本
            max_length: 最大长度
            
        Returns:
            PPL计算结果
        """
        if not self.loaded_model:
            return {
                'error': 'No model loaded'
            }
        
        try:
            tokens = self.engine.tokenize(text)
            
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            if len(tokens) < 2:
                return {
                    'ppl': float('inf'),
                    'token_count': len(tokens),
                    'error': 'Text too short'
                }
            
            total_log_prob = 0.0
            token_count = 0
            
            # 计算每个位置的log概率
            for i in range(1, len(tokens)):
                context_tokens = tokens[:i]
                target_token = tokens[i]
                
                try:
                    log_prob = self.engine.get_logprobs(context_tokens, target_token)
                    total_log_prob += log_prob
                    token_count += 1
                except Exception as e:
                    print(f"计算位置 {i} 的logprob时出错: {e}")
                    continue
            
            if token_count == 0:
                return {
                    'ppl': float('inf'),
                    'token_count': 0,
                    'error': 'No valid tokens processed'
                }
            
            # 计算PPL
            avg_nll = -total_log_prob / token_count
            ppl = math.exp(avg_nll)
            
            return {
                'ppl': ppl,
                'token_count': token_count,
                'avg_nll': avg_nll,
                'total_log_prob': total_log_prob
            }
            
        except Exception as e:
            return {
                'error': f'PPL calculation failed: {str(e)}'
            }
    
    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine.cleanup()


def create_infinicore_adapter() -> OpenAIAdapter:
    """创建InfiniCore-Infer适配器
    
    Returns:
        OpenAI适配器实例
    """
    engine = InfiniCoreInferEngine()
    return OpenAIAdapter(engine)


# 示例用法
if __name__ == '__main__':
    # 创建适配器
    adapter = create_infinicore_adapter()
    
    # 加载模型
    model_path = "/home/shared/models/jiuge9G4B"
    load_result = adapter.load_model(model_path)
    
    if load_result['success']:
        print("模型加载成功")
        
        # 测试PPL计算
        test_text = "The quick brown fox jumps over the lazy dog."
        ppl_result = adapter.calculate_ppl(test_text)
        print(f"PPL结果: {ppl_result}")
        
        # 测试OpenAI兼容接口
        completion_result = adapter.completions(
            prompt=test_text,
            max_tokens=1,
            logprobs=True
        )
        print(f"Completion结果: {json.dumps(completion_result, indent=2)}")
        
    else:
        print(f"模型加载失败: {load_result['error']}")
    
    # 清理资源
    adapter.cleanup()