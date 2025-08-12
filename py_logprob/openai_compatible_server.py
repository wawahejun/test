#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI兼容的推理服务器
支持logprobs参数，用于PPL计算
"""

import os
import sys
import argparse
import json
import uuid
import time
import math
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jiuge import JiugeForCauslLM, DeviceType
from infer_task import InferTask, KVCache

# 设备类型映射
DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
}

# 请求模型定义
class CompletionRequest(BaseModel):
    model: str = "jiuge"
    prompt: str
    max_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    logprobs: Optional[int] = None  # 返回top-k个token的概率
    echo: bool = False
    stop: Optional[List[str]] = None
    stream: bool = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "jiuge"
    messages: List[ChatMessage]
    max_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[List[str]] = None
    stream: bool = False

# 全局模型实例
model_instance = None

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI兼容的推理服务器")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_TYPE_MAP.keys(),
        default="cpu",
        help="设备类型 (默认: cpu)"
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="设备数量 (默认: 1)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    return parser.parse_args()

def calculate_token_logprobs(model: JiugeForCauslLM, tokens: List[int], max_logprobs: int = 5) -> List[Dict[str, Any]]:
    """
    计算每个token位置的logprobs
    
    Args:
        model: 模型实例
        tokens: token序列
        max_logprobs: 返回的最大logprobs数量
        
    Returns:
        List[Dict]: 每个位置的logprobs信息
    """
    if len(tokens) <= 1:
        return []
    
    logprobs_list = []
    
    try:
        # 逐个token计算logprobs
        for i in range(1, len(tokens)):
            context_tokens = tokens[:i]
            target_token = tokens[i]
            
            # 创建推理任务
            task = InferTask(
                id=f"logprob_task_{i}",
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
                logits_output = model.batch_infer_with_logits(tasks)
                
                if logits_output and len(logits_output) > 0:
                    # 获取logits并计算概率
                    logits = logits_output[0]  # 第一个任务的logits
                    
                    # 计算softmax概率
                    import numpy as np
                    logits_np = np.array(logits, dtype=np.float32)
                    
                    # 数值稳定的softmax
                    max_logit = np.max(logits_np)
                    exp_logits = np.exp(logits_np - max_logit)
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # 计算log概率
                    log_probs = np.log(probs + 1e-10)  # 避免log(0)
                    
                    # 获取top-k概率
                    top_indices = np.argsort(probs)[::-1][:max_logprobs]
                    
                    # 构建logprobs字典
                    top_logprobs = {}
                    for idx in top_indices:
                        token_str = model.tokenizer.decode([int(idx)])
                        top_logprobs[token_str] = float(log_probs[idx])
                    
                    # 获取目标token的概率
                    target_token_str = model.tokenizer.decode([target_token])
                    target_logprob = float(log_probs[target_token]) if target_token < len(log_probs) else -float('inf')
                    
                    logprobs_info = {
                        "token": target_token_str,
                        "logprob": target_logprob,
                        "top_logprobs": top_logprobs
                    }
                    
                else:
                    # 使用fallback方法
                    target_token_str = model.tokenizer.decode([target_token])
                    fallback_prob = 1.0 / len(model.tokenizer)  # 均匀分布假设
                    logprobs_info = {
                        "token": target_token_str,
                        "logprob": math.log(fallback_prob),
                        "top_logprobs": {target_token_str: math.log(fallback_prob)}
                    }
                    
                logprobs_list.append(logprobs_info)
                
            finally:
                # 清理KV缓存
                if task._kv_cache:
                    task._kv_cache.drop(model)
                    
    except Exception as e:
        print(f"计算logprobs时出错: {e}")
        # 返回fallback结果
        for i in range(1, len(tokens)):
            target_token = tokens[i]
            target_token_str = model.tokenizer.decode([target_token])
            fallback_prob = 1.0 / len(model.tokenizer)
            logprobs_info = {
                "token": target_token_str,
                "logprob": math.log(fallback_prob),
                "top_logprobs": {target_token_str: math.log(fallback_prob)}
            }
            logprobs_list.append(logprobs_info)
    
    return logprobs_list

def create_completion_response(request_id: str, model_name: str, prompt: str, 
                             completion_text: str, logprobs_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    创建OpenAI格式的completion响应
    """
    choice = {
        "text": completion_text,
        "index": 0,
        "finish_reason": "length"
    }
    
    if logprobs_data:
        choice["logprobs"] = {
            "tokens": [item["token"] for item in logprobs_data],
            "token_logprobs": [item["logprob"] for item in logprobs_data],
            "top_logprobs": [item["top_logprobs"] for item in logprobs_data]
        }
    
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(model_instance.tokenizer.encode(prompt)),
            "completion_tokens": len(model_instance.tokenizer.encode(completion_text)),
            "total_tokens": len(model_instance.tokenizer.encode(prompt + completion_text))
        }
    }

# 创建FastAPI应用
app = FastAPI(title="OpenAI兼容推理服务器", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    global model_instance
    args = parse_args()
    
    print(f"正在加载模型: {args.model_path}")
    print(f"设备: {args.device}, 设备数量: {args.ndev}")
    
    device_type = DEVICE_TYPE_MAP[args.device]
    model_instance = JiugeForCauslLM(
        model_dir_path=args.model_path,
        device=device_type,
        ndev=args.ndev
    )
    
    print("模型加载完成")
    print(f"最大上下文长度: {model_instance.max_context_len()}")
    print(f"词汇表大小: {len(model_instance.tokenizer)}")

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI兼容的completions接口
    """
    try:
        request_id = f"cmpl-{uuid.uuid4().hex}"
        
        # 编码输入文本
        tokens = model_instance.tokenizer.encode(request.prompt, add_special_tokens=True)
        
        if request.max_tokens == 1 and request.logprobs is not None:
            # PPL计算模式：只需要logprobs，不需要生成新token
            logprobs_data = calculate_token_logprobs(model_instance, tokens, request.logprobs or 5)
            
            response = create_completion_response(
                request_id=request_id,
                model_name=request.model,
                prompt=request.prompt,
                completion_text="",  # PPL模式不生成新文本
                logprobs_data=logprobs_data
            )
            
            return JSONResponse(content=response)
        
        else:
            # 正常生成模式
            # 创建推理任务
            task = InferTask(tokens, max_tokens=request.max_tokens)
            task.bind_kvcache(KVCache(model_instance))
            
            try:
                # 执行推理
                tasks = [task]
                results = model_instance.batch_infer_one_round(tasks)
                
                if results and len(results) > 0:
                    generated_tokens = results[0]
                    completion_text = model_instance.tokenizer.decode(generated_tokens)
                    
                    # 如果需要logprobs，计算生成token的概率
                    logprobs_data = None
                    if request.logprobs is not None:
                        all_tokens = tokens + generated_tokens
                        logprobs_data = calculate_token_logprobs(model_instance, all_tokens, request.logprobs)
                        # 只返回生成部分的logprobs
                        logprobs_data = logprobs_data[-len(generated_tokens):] if logprobs_data else []
                    
                    response = create_completion_response(
                        request_id=request_id,
                        model_name=request.model,
                        prompt=request.prompt,
                        completion_text=completion_text,
                        logprobs_data=logprobs_data
                    )
                    
                    return JSONResponse(content=response)
                
                else:
                    raise HTTPException(status_code=500, detail="推理失败")
                    
            finally:
                # 清理KV缓存
                if task._kv_cache:
                    task._kv_cache.drop(model_instance)
        
    except Exception as e:
        print(f"Completions请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI兼容的chat completions接口
    """
    try:
        # 将chat消息转换为prompt
        prompt_parts = []
        for message in request.messages:
            if message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
            elif message.role == "system":
                prompt_parts.append(f"System: {message.content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # 转换为completion请求
        completion_request = CompletionRequest(
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            stream=request.stream
        )
        
        # 调用completions接口
        completion_response = await completions(completion_request)
        
        # 转换为chat格式
        completion_data = completion_response.body.decode('utf-8')
        completion_json = json.loads(completion_data)
        
        chat_response = {
            "id": completion_json["id"].replace("cmpl-", "chatcmpl-"),
            "object": "chat.completion",
            "created": completion_json["created"],
            "model": completion_json["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_json["choices"][0]["text"]
                },
                "finish_reason": completion_json["choices"][0]["finish_reason"]
            }],
            "usage": completion_json["usage"]
        }
        
        # 如果有logprobs，添加到响应中
        if "logprobs" in completion_json["choices"][0]:
            chat_response["choices"][0]["logprobs"] = completion_json["choices"][0]["logprobs"]
        
        return JSONResponse(content=chat_response)
        
    except Exception as e:
        print(f"Chat completions请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """
    列出可用模型
    """
    return JSONResponse(content={
        "object": "list",
        "data": [{
            "id": "jiuge",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "infinicore"
        }]
    })

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return JSONResponse(content={"status": "healthy"})

if __name__ == "__main__":
    args = parse_args()
    print(f"启动OpenAI兼容服务器...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"模型: {args.model_path}")
    print(f"设备: {args.device}")
    
    uvicorn.run(app, host=args.host, port=args.port)