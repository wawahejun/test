#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理WikiText-2数据集，提取有效文本用于PPL测试
"""

import json
import os
import re
from pathlib import Path

def process_wikitext_file(file_path: str, max_samples: int = 50) -> list:
    """
    处理WikiText文件，提取有效的文本段落
    """
    print(f"处理文件: {file_path}")
    
    texts = []
    current_paragraph = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    if current_paragraph:
                        # 结束当前段落
                        paragraph_text = ' '.join(current_paragraph)
                        if is_valid_paragraph(paragraph_text):
                            texts.append(paragraph_text)
                            if len(texts) >= max_samples:
                                break
                        current_paragraph = []
                    continue
                
                # 跳过标题行（以=开头）
                if line.startswith('='):
                    continue
                
                # 添加到当前段落
                current_paragraph.append(line)
        
        # 处理最后一个段落
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            if is_valid_paragraph(paragraph_text) and len(texts) < max_samples:
                texts.append(paragraph_text)
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return []
    
    print(f"提取了 {len(texts)} 个有效段落")
    return texts

def is_valid_paragraph(text: str) -> bool:
    """
    判断段落是否有效用于PPL测试
    """
    # 基本长度检查
    if len(text) < 50 or len(text) > 500:
        return False
    
    # 检查<unk>标记的比例
    unk_count = text.count('<unk>')
    word_count = len(text.split())
    if word_count == 0 or unk_count / word_count > 0.1:  # 超过10%的<unk>标记
        return False
    
    # 检查是否包含足够的字母字符
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / len(text) < 0.7:  # 字母字符少于70%
        return False
    
    # 检查是否包含过多的特殊符号
    special_chars = sum(1 for c in text if c in '@-')
    if special_chars / len(text) > 0.05:  # 特殊符号超过5%
        return False
    
    return True

def clean_text(text: str) -> str:
    """
    清理文本，移除不必要的标记
    """
    # 移除<unk>标记
    text = re.sub(r'<unk>', '', text)
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊符号组合
    text = re.sub(r'@-@', '-', text)
    
    return text.strip()

def main():
    wikitext_dir = "/home/wawahejun/reasoning/test/datasets/wikitext-2"
    
    # 检查目录是否存在
    if not os.path.exists(wikitext_dir):
        print(f"错误：WikiText-2目录不存在: {wikitext_dir}")
        return
    
    # 处理测试集
    test_file = os.path.join(wikitext_dir, "wiki.test.tokens")
    if not os.path.exists(test_file):
        print(f"错误：测试文件不存在: {test_file}")
        return
    
    print("=== 处理WikiText-2测试集 ===")
    texts = process_wikitext_file(test_file, max_samples=50)
    
    if not texts:
        print("没有提取到有效文本")
        return
    
    # 清理文本
    cleaned_texts = []
    for text in texts:
        cleaned = clean_text(text)
        if len(cleaned) > 30:  # 清理后仍然有足够长度
            cleaned_texts.append(cleaned)
    
    print(f"清理后剩余 {len(cleaned_texts)} 个文本")
    
    # 保存处理后的数据
    output_data = {
        "dataset_name": "WikiText-2 Test Set (Processed)",
        "description": "从WikiText-2测试集提取的有效文本段落，用于PPL评测",
        "sample_count": len(cleaned_texts),
        "processing_info": {
            "max_samples": 50,
            "min_length": 50,
            "max_length": 500,
            "max_unk_ratio": 0.1,
            "min_alpha_ratio": 0.7
        },
        "texts": cleaned_texts
    }
    
    output_file = "/home/wawahejun/reasoning/test/datasets/wikitext2_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成，数据已保存到: {output_file}")
    
    # 显示一些样本
    print("\n=== 样本预览 ===")
    for i, text in enumerate(cleaned_texts[:3]):
        print(f"样本 {i+1} (长度: {len(text)}): {text[:100]}...")
    
    print(f"\n数据集统计:")
    print(f"- 总样本数: {len(cleaned_texts)}")
    print(f"- 平均长度: {sum(len(t) for t in cleaned_texts) / len(cleaned_texts):.1f} 字符")
    print(f"- 长度范围: {min(len(t) for t in cleaned_texts)} - {max(len(t) for t in cleaned_texts)} 字符")

if __name__ == "__main__":
    main()