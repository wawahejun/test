#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速PyTorch测试脚本
用于快速验证TinyLlama模型的基本功能和中间输出

使用方法:
    python quick_pytorch_test.py
    python quick_pytorch_test.py --save-outputs  # 保存中间输出用于对比
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse
import time

def test_model_loading(model_path='/home/shared/models/9G7B_MHA'):
    """测试模型加载"""
    print("=== 测试1: 模型加载 ===")
    print(f"模型路径: {model_path}")
    
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        
        print(f"   模型加载成功")
        print(f"   耗时: {load_time:.2f}s")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"   设备: {next(model.parameters()).device}")
        print(f"   数据类型: {next(model.parameters()).dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"  模型加载失败: {e}")
        return None, None

def test_basic_inference(model, tokenizer):
    """测试基本推理功能"""
    print("\n=== 测试2: 基本推理 ===")
    
    # 测试文本
    test_text = "Robert Boulter is an English film"
    print(f"输入文本: {test_text}")
    
    try:
        # Tokenize
        inputs = tokenizer(test_text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        print(f"Token IDs: {inputs['input_ids'].tolist()[0]}")
        
        # 推理
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            logits = outputs.logits
            print(f"   推理成功")
            print(f"   耗时: {inference_time*1000:.2f}ms")
            print(f"   输出形状: {logits.shape}")
            print(f"   前5个logits: {logits[0, -1, :5].cpu().numpy()}")
            
            # 预测下一个token
            next_token_id = torch.argmax(logits[0, -1, :]).item()
            next_token = tokenizer.decode([next_token_id])
            print(f"   预测下一个token: {next_token_id} -> '{next_token}'")
            
        return True
        
    except Exception as e:
        print(f"  推理失败: {e}")
        return False

def load_wikitext_data_for_test(file_path: str, tokenizer, max_length: int = 512, max_sequences: int = None):
    """加载WikiText-2数据集用于测试"""
    print(f"加载测试数据集: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤空行和只包含标题标记的行
    valid_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and len(line) > 10:
            valid_lines.append(line)
    
    print(f"有效文本行数: {len(valid_lines)}")
    
    # 将所有有效行合并为一个文本
    text = ' '.join(valid_lines)
    
    # 使用tokenizer编码文本
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"总token数: {len(token_ids)}")
    
    # 分割为固定长度的序列
    sequences = []
    for i in range(0, len(token_ids) - max_length + 1, max_length // 2):
        seq = token_ids[i:i + max_length]
        if len(seq) == max_length:
            sequences.append(seq)
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
    
    print(f"生成测试序列数: {len(sequences)}")
    return sequences

def test_ppl_calculation(model, tokenizer):
    """测试PPL计算（使用WikiText-2数据集和模型自身tokenizer编码）"""
    print("\n=== 测试3: PPL计算 ===")
    
    # 加载WikiText-2测试数据
    data_path = '/home/wawahejun/reasoning/c_reasoning/datasets/wikitext-2/wiki.test.tokens'
    sequences = load_wikitext_data_for_test(data_path, tokenizer, max_length=512, max_sequences=200)
    
    if not sequences:
        print("  数据加载失败")
        return None
    
    try:
        device = next(model.parameters()).device
        total_loss = 0.0
        total_tokens = 0
        
        print(f"开始计算PPL，测试序列数: {len(sequences)}")
        
        with torch.no_grad():
            start_time = time.time()
            
            for i, seq in enumerate(sequences):
                # 准备输入、目标和掩码
                input_ids = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
                target_ids = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, device=device)  # 无padding时全为1
                
                # 前向传播
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 计算损失
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += target_ids.numel()
            
            # 计算平均困惑度
            avg_loss = total_loss / total_tokens
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            calc_time = time.time() - start_time
            
            print(f"   PPL计算成功（使用WikiText-2数据集）")
            print(f"   耗时: {calc_time*1000:.2f}ms")
            print(f"   平均损失: {avg_loss:.4f}")
            print(f"   困惑度: {ppl:.4f}")
            print(f"   处理序列数: {len(sequences)}")
            print(f"   总token数: {total_tokens}")
            
        return ppl
        
    except Exception as e:
        print(f"  PPL计算失败: {e}")
        return None

def extract_and_save_intermediate_outputs(model, tokenizer, save_outputs=False):
    """提取并保存中间输出"""
    print("\n=== 测试4: 中间输出提取 ===")
    
    # 使用与InfiniCore-Infer相同的测试序列
    test_tokens = [1, 4755, 350, 5059, 357, 338, 385, 4223, 2706, 1919]
    print(f"测试序列: {test_tokens}")
    
    try:
        device = next(model.parameters()).device
        input_ids = torch.tensor(test_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        intermediate_outputs = {}
        
        with torch.no_grad():
            # 1. 嵌入层输出
            embeddings = model.model.embed_tokens(input_ids)
            intermediate_outputs['embeddings'] = embeddings.cpu().numpy()
            print(f"嵌入层输出形状: {embeddings.shape}")
            print(f"前3个嵌入值: {embeddings[0, 0, :3].cpu().numpy()}")
            
            # 2. 逐层transformer输出 - 使用完整前向传播
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True, return_dict=True)
                
                # 获取所有隐藏状态
                all_hidden_states = outputs.hidden_states
                
                # 保存前3层的输出
                for i in range(min(3, len(all_hidden_states)-1)):
                    layer_output = all_hidden_states[i+1]  # 跳过embedding层
                    intermediate_outputs[f'layer_{i}_output'] = layer_output.cpu().numpy()
                    print(f"第{i}层输出形状: {layer_output.shape}")
                    print(f"第{i}层前3个值: {layer_output[0, 0, :3].cpu().numpy()}")
                
                hidden_states = all_hidden_states[-1]  # 最后一层的输出
            
            # 3. 最终归一化和logits
            final_hidden = model.model.norm(hidden_states)
            intermediate_outputs['final_norm'] = final_hidden.cpu().numpy()
            
            logits = model.lm_head(final_hidden)
            intermediate_outputs['final_logits'] = logits.cpu().numpy()
            
            print(f"最终logits形状: {logits.shape}")
            print(f"最后一个位置前5个logits: {logits[0, -1, :5].cpu().numpy()}")
            
            # 4. 计算概率分布
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            print(f"\nTop-5预测:")
            for i, (prob, idx) in enumerate(zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy())):
                token = tokenizer.decode([idx])
                print(f"  {i+1}. Token {idx} ('{token}'): {prob:.4f}")
        
        if save_outputs:
            output_file = 'pytorch_intermediate_outputs.npz'
            np.savez(output_file, **intermediate_outputs)
            print(f"\n  中间输出已保存到: {output_file}")
            
            # 保存文本格式的摘要
            summary_file = 'pytorch_outputs_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("PyTorch中间输出摘要\n")
                f.write("=" * 30 + "\n")
                f.write(f"测试序列: {test_tokens}\n")
                f.write(f"嵌入层形状: {intermediate_outputs['embeddings'].shape}\n")
                f.write(f"最终logits形状: {intermediate_outputs['final_logits'].shape}\n")
                f.write(f"最后位置前5个logits: {intermediate_outputs['final_logits'][0, -1, :5]}\n")
                
                f.write("\nTop-5预测:\n")
                for i, (prob, idx) in enumerate(zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy())):
                    token = tokenizer.decode([idx])
                    f.write(f"  {i+1}. Token {idx} ('{token}'): {prob:.4f}\n")
            
            print(f"  输出摘要已保存到: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"  中间输出提取失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='快速PyTorch测试')
    parser.add_argument('--save-outputs', action='store_true', help='保存中间输出文件')
    parser.add_argument('--model', type=str, default='tinyllama', choices=['9G7B_MHA', 'jiuge9G4B', 'tinyllama'], help='选择要测试的模型')
    
    args = parser.parse_args()
    
    # 根据选择设置模型路径
    if args.model == '9G7B_MHA':
        model_path = '/home/shared/models/9G7B_MHA'
    elif args.model == 'jiuge9G4B':
        model_path = '/home/shared/models/jiuge9G4B'
    elif args.model == 'tinyllama':
        model_path = '/home/shared/models/TinyLlama'
    else:
        model_path = '/home/shared/models/TinyLlama'  # 默认值
    
    print(" PyTorch快速测试开始")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    
    # 测试1: 模型加载
    print(f"选择的模型: {args.model}")
    print(f"模型路径: {model_path}")
    model, tokenizer = test_model_loading(model_path)
    if model is None:
        return
    
    # 测试2: 基本推理
    if not test_basic_inference(model, tokenizer):
        return
    
    # 测试3: PPL计算
    ppl = test_ppl_calculation(model, tokenizer)
    if ppl is None:
        return
    
    # 测试4: 中间输出提取
    if not extract_and_save_intermediate_outputs(model, tokenizer, args.save_outputs):
        return
    
    print("\n  所有测试通过！")
    print("\n  关键结果:")
    print(f"   模型: 9G7B_MHA")
    print(f"   设备: {next(model.parameters()).device}")
    print(f"   测试PPL: {ppl:.4f}")
    
    if args.save_outputs:
        print("1. 使用保存的中间输出文件与InfiniCore-Infer进行对比")
        print("2. 重点关注embeddings和final_logits的数值差异")

if __name__ == '__main__':
    main()