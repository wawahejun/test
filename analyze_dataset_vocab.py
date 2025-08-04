#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据集的词汇分布，检查是否与PPL异常有关
"""

import sys
import os
import json
import re
from collections import Counter
from typing import Set, List, Dict

# 添加路径
sys.path.insert(0, '/home/wawahejun/reasoning/c_reasoning/InfiniCore-Infer/scripts')

try:
    from jiuge import JiugeForCauslLM, DeviceType
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

def load_dataset():
    """加载数据集"""
    data_path = '/home/wawahejun/reasoning/test/datasets/wikitext2_processed.json'
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'texts' in data:
            texts = data['texts']
        else:
            print(f"数据格式错误: {type(data)}")
            return []
        
        print(f"加载了 {len(texts)} 个文本样本")
        return texts
    except Exception as e:
        print(f"加载数据失败: {e}")
        return []

def analyze_text_vocabulary(texts: List[str]) -> Dict:
    """分析文本的词汇分布"""
    print("\n=== 分析数据集词汇分布 ===")
    
    # 合并所有文本
    all_text = ' '.join(texts)
    
    # 基本统计
    total_chars = len(all_text)
    total_words = len(all_text.split())
    
    # 词汇统计（简单按空格分割）
    words = all_text.lower().split()
    word_counter = Counter(words)
    unique_words = len(word_counter)
    
    # 字符统计
    char_counter = Counter(all_text.lower())
    unique_chars = len(char_counter)
    
    # 最常见的词汇
    most_common_words = word_counter.most_common(20)
    
    print(f"总字符数: {total_chars:,}")
    print(f"总词数: {total_words:,}")
    print(f"唯一词汇数: {unique_words:,}")
    print(f"唯一字符数: {unique_chars}")
    print(f"词汇重复率: {(total_words - unique_words) / total_words * 100:.2f}%")
    
    print(f"\n最常见的20个词汇:")
    for word, count in most_common_words:
        print(f"  '{word}': {count}次")
    
    return {
        'total_chars': total_chars,
        'total_words': total_words,
        'unique_words': unique_words,
        'unique_chars': unique_chars,
        'word_repetition_rate': (total_words - unique_words) / total_words * 100,
        'most_common_words': most_common_words
    }

def analyze_model_tokenization(texts: List[str]) -> Dict:
    """分析模型的分词结果"""
    print("\n=== 分析模型分词结果 ===")
    
    model_path = '/home/shared/models/jiuge9G4B'
    
    try:
        # 加载模型（仅用于分词）
        print("正在加载模型...")
        model = JiugeForCauslLM(
            model_dir_path=model_path,
            device=DeviceType.DEVICE_TYPE_CPU,
            ndev=1,
            max_tokens=128
        )
        
        print(f"模型词汇表大小: {model.meta.dvoc}")
        
        # 分析前几个样本的分词
        sample_texts = texts[:5]  # 只分析前5个样本
        tokenization_results = []
        all_tokens = []
        
        for i, text in enumerate(sample_texts):
            tokens = model.tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
            
            tokenization_results.append({
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'token_count': len(tokens),
                'tokens_preview': tokens[:10],  # 前10个token
                'unique_tokens_in_text': len(set(tokens))
            })
            
            print(f"样本 {i+1}: {len(tokens)} tokens, 唯一tokens: {len(set(tokens))}")
        
        # 统计所有tokens
        token_counter = Counter(all_tokens)
        unique_tokens_used = len(token_counter)
        total_tokens = len(all_tokens)
        
        print(f"\n分词统计:")
        print(f"总token数: {total_tokens}")
        print(f"使用的唯一token数: {unique_tokens_used}")
        print(f"模型词汇表利用率: {unique_tokens_used / model.meta.dvoc * 100:.4f}%")
        
        # 最常见的tokens
        most_common_tokens = token_counter.most_common(10)
        print(f"\n最常见的10个tokens:")
        for token_id, count in most_common_tokens:
            print(f"  Token {token_id}: {count}次")
        
        # 清理模型
        del model
        
        return {
            'model_vocab_size': model.meta.dvoc if 'model' in locals() else 0,
            'total_tokens': total_tokens,
            'unique_tokens_used': unique_tokens_used,
            'vocab_utilization_rate': unique_tokens_used / (model.meta.dvoc if 'model' in locals() else 1) * 100,
            'tokenization_results': tokenization_results,
            'most_common_tokens': most_common_tokens
        }
        
    except Exception as e:
        print(f"模型分词分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def main():
    """主函数"""
    print("=== 数据集词汇分析 ===")
    
    # 加载数据集
    texts = load_dataset()
    if not texts:
        print("无法加载数据集")
        return
    
    # 分析文本词汇
    text_vocab_analysis = analyze_text_vocabulary(texts)
    
    # 分析模型分词
    model_tokenization_analysis = analyze_model_tokenization(texts)
    
    # 保存结果
    results = {
        'dataset_info': {
            'sample_count': len(texts),
            'avg_text_length': sum(len(text) for text in texts) / len(texts)
        },
        'text_vocabulary_analysis': text_vocab_analysis,
        'model_tokenization_analysis': model_tokenization_analysis
    }
    
    output_file = '/home/wawahejun/reasoning/test/results/dataset_vocab_analysis.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n分析结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")
    
    # 打印总结
    print(f"\n{'='*50}")
    print("分析总结:")
    print(f"{'='*50}")
    print(f"数据集样本数: {len(texts)}")
    print(f"平均文本长度: {results['dataset_info']['avg_text_length']:.1f} 字符")
    print(f"数据集唯一词汇数: {text_vocab_analysis['unique_words']:,}")
    
    if 'error' not in model_tokenization_analysis:
        print(f"模型词汇表大小: {model_tokenization_analysis['model_vocab_size']:,}")
        print(f"实际使用的token数: {model_tokenization_analysis['unique_tokens_used']:,}")
        print(f"词汇表利用率: {model_tokenization_analysis['vocab_utilization_rate']:.4f}%")
        
        # 分析可能的问题
        utilization_rate = model_tokenization_analysis['vocab_utilization_rate']
        if utilization_rate < 1.0:
            print(f"\n⚠️  词汇表利用率极低 ({utilization_rate:.4f}%)，这可能导致:")
            print("   1. 大量未使用的token具有异常的logprob值")
            print("   2. 模型对数据集外的token处理不当")
            print("   3. PPL计算时包含了大量噪声token")
    else:
        print(f"模型分词分析失败: {model_tokenization_analysis['error']}")

if __name__ == "__main__":
    main()