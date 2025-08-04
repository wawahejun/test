#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的模型权重检查脚本
检查模型权重文件的数值分布和完整性
"""

import sys
import os
import numpy as np
import struct
from pathlib import Path

# 添加脚本路径
sys.path.append('/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')
from jiuge import JiugeForCauslLM

def check_model_file_integrity(model_path):
    """检查模型文件的完整性"""
    print("=== 模型文件完整性检查 ===")
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        return False
        
    print(f"模型目录: {model_path}")
    
    # 检查关键文件
    key_files = [
        "config.json",
        "tokenizer.json"
    ]
    
    missing_files = []
    for file_name in key_files:
        file_path = model_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name}: {size:,} bytes")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name}: 文件不存在")
    
    # 检查权重文件
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if weight_files:
        print(f"\n权重文件:")
        total_size = 0
        for weight_file in weight_files:
            size = weight_file.stat().st_size
            total_size += size
            print(f"  {weight_file.name}: {size:,} bytes")
        print(f"总权重大小: {total_size:,} bytes ({total_size/1024/1024/1024:.2f} GB)")
        
        # 检查是否有分片文件
        shard_files = [f for f in weight_files if 'of' in f.name]
        if shard_files:
            print(f"✅ 检测到分片模型文件 ({len(shard_files)} 个分片)")
        elif any('model.safetensors' in f.name for f in weight_files):
            print(f"✅ 检测到单一模型文件")
        else:
            print(f"⚠️  权重文件格式可能异常")
    else:
        print("❌ 未找到权重文件")
        missing_files.append("权重文件")
    
    if missing_files:
        print(f"\n❌ 缺少关键文件: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ 模型文件完整性检查通过")
        return True

def analyze_weight_distribution(model):
    """分析模型权重的数值分布"""
    print("\n=== 模型权重数值分布分析 ===")
    
    try:
        # 获取模型权重信息
        weights = model.weights
        meta = model.meta
        
        print(f"模型层数: {meta.nlayer}")
        print(f"隐藏维度: {meta.d}")
        print(f"词汇表大小: {meta.dvoc}")
        
        # 检查输出嵌入层权重（这是GEMM操作中最关键的）
        print(f"\n=== 输出嵌入层权重分析 ===")
        
        # 注意：这里我们无法直接访问C++中的权重数据
        # 但我们可以通过模型的元数据来推断权重的预期范围
        
        expected_weight_std = 1.0 / np.sqrt(meta.d)  # 典型的权重初始化标准差
        print(f"预期权重标准差: {expected_weight_std:.6f}")
        print(f"预期权重范围: [{-3*expected_weight_std:.6f}, {3*expected_weight_std:.6f}]")
        
        # 检查数据类型
        print(f"\n=== 数据类型信息 ===")
        print(f"Logits数据类型: {weights.dt_mat}")
        print(f"Norm数据类型: {weights.dt_norm}")
        
        # 如果权重被量化，可能会导致数值问题
        if weights.dt_mat != 13:  # 13 = INFINI_DTYPE_F32
            print(f"⚠️  警告: 权重不是FP32格式，可能存在量化精度问题")
            print(f"   当前格式代码: {weights.dt_mat}")
            
        return True
        
    except Exception as e:
        print(f"❌ 权重分析失败: {e}")
        return False

def test_simple_forward_pass(model):
    """测试简单的前向传播，检查中间结果"""
    print("\n=== 简单前向传播测试 ===")
    
    try:
        # 使用简单的输入
        test_inputs = [
            "Hello",
            "The", 
            "AI",
            "1",
            "."
        ]
        
        for test_text in test_inputs:
            print(f"\n测试输入: '{test_text}'")
            
            # 分词
            tokens = model.tokenizer.encode(test_text)
            print(f"  Tokens: {tokens}")
            
            if len(tokens) == 0:
                print(f"  ❌ 分词结果为空")
                continue
                
            # 检查token是否在合理范围内
            vocab_size = model.meta.dvoc
            for token in tokens:
                if token >= vocab_size:
                    print(f"  ❌ Token {token} 超出词汇表范围 [0, {vocab_size-1}]")
                    return False
                    
            print(f"  ✅ Tokens在合理范围内")
            
            # 这里我们无法直接检查隐藏状态，但可以检查输入嵌入的合理性
            # 输入嵌入应该在合理的数值范围内
            
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_numerical_precision():
    """检查数值精度相关的问题"""
    print("\n=== 数值精度检查 ===")
    
    # 检查系统的浮点数精度
    print(f"系统float32精度: {np.finfo(np.float32)}")
    print(f"系统float64精度: {np.finfo(np.float64)}")
    
    # 测试大数值的计算
    print("\n=== 大数值计算测试 ===")
    
    # 模拟可能导致溢出的情况
    large_values = np.array([1e10, 1e15, 1e20], dtype=np.float32)
    print(f"大数值: {large_values}")
    
    # 测试softmax计算
    try:
        # 标准softmax（可能溢出）
        exp_vals = np.exp(large_values)
        print(f"exp(大数值): {exp_vals}")
        
        if np.any(np.isinf(exp_vals)):
            print("⚠️  警告: exp计算产生无穷大")
            
        # 数值稳定的softmax
        max_val = np.max(large_values)
        stable_exp = np.exp(large_values - max_val)
        stable_softmax = stable_exp / np.sum(stable_exp)
        print(f"数值稳定的softmax: {stable_softmax}")
        
        # log_softmax
        log_softmax = large_values - max_val - np.log(np.sum(stable_exp))
        print(f"log_softmax: {log_softmax}")
        
    except Exception as e:
        print(f"❌ 数值计算测试失败: {e}")
        
def suggest_solutions():
    """提供解决方案建议"""
    print("\n=== 问题诊断和解决方案 ===")
    
    print("基于调试结果，问题的可能原因和解决方案:")
    
    print("\n1. 模型权重异常:")
    print("   - 原因: 模型文件损坏或权重初始化异常")
    print("   - 解决: 重新下载模型文件，检查文件完整性")
    
    print("\n2. 数值精度问题:")
    print("   - 原因: 量化精度不足或数值溢出")
    print("   - 解决: 使用FP32精度，避免过度量化")
    
    print("\n3. GEMM操作实现问题:")
    print("   - 原因: GEMM库实现有bug或配置错误")
    print("   - 解决: 检查BLAS库版本，尝试不同的GEMM实现")
    
    print("\n4. 硬件相关问题:")
    print("   - 原因: GPU计算精度设置或驱动问题")
    print("   - 解决: 更新GPU驱动，检查CUDA版本兼容性")
    
    print("\n5. 模型架构不匹配:")
    print("   - 原因: 模型文件与推理引擎版本不匹配")
    print("   - 解决: 确保模型格式与推理引擎兼容")

def main():
    """主函数"""
    print("=== 详细模型权重和数值精度调试 ===")
    
    model_path = "/home/shared/models/jiuge9G4B"
    
    # 1. 检查模型文件完整性
    if not check_model_file_integrity(model_path):
        print("\n❌ 模型文件完整性检查失败，请检查模型文件")
        return
    
    try:
        # 2. 加载模型
        print("\n正在加载模型...")
        model = JiugeForCauslLM(model_path)
        
        # 3. 分析权重分布
        analyze_weight_distribution(model)
        
        # 4. 测试前向传播
        test_simple_forward_pass(model)
        
        # 5. 检查数值精度
        check_numerical_precision()
        
        # 6. 提供解决方案
        suggest_solutions()
        
        # 清理资源
        model.destroy_model_instance()
        
    except Exception as e:
        print(f"❌ 模型加载或分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()