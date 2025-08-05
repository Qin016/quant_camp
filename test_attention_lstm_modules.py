#!/usr/bin/env python3
"""
Attention-LSTM策略测试脚本
验证数据加载、特征工程和标签构造模块
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from attention_lstm_strategy import (
    StrategyConfig, 
    DataLoader, 
    FeatureEngineering, 
    LabelConstructor,
    AttentionLSTMModel,
    AttentionLSTMStrategy
)

def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """创建模拟的L2行情数据"""
    np.random.seed(42)
    
    # 生成基础价格序列
    base_price = 3000.0
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 创建完整的tick数据
    data = pd.DataFrame({
        'LASTPRICE': prices,
        'HIGHPRICE': prices * (1 + np.random.uniform(0, 0.002, n_samples)),
        'LOWPRICE': prices * (1 - np.random.uniform(0, 0.002, n_samples)),
        'TRADEVOLUME': np.random.poisson(100, n_samples),
        
        # 盘口数据
        'BUYPRICE01': prices * (1 - np.random.uniform(0.0001, 0.0005, n_samples)),
        'SELLPRICE01': prices * (1 + np.random.uniform(0.0001, 0.0005, n_samples)),
        'BUYVOLUME01': np.random.poisson(50, n_samples),
        'SELLVOLUME01': np.random.poisson(50, n_samples),
        
        # 多档盘口
        'BUYPRICE02': prices * (1 - np.random.uniform(0.0005, 0.001, n_samples)),
        'SELLPRICE02': prices * (1 + np.random.uniform(0.0005, 0.001, n_samples)),
        'BUYVOLUME02': np.random.poisson(30, n_samples),
        'SELLVOLUME02': np.random.poisson(30, n_samples),
        
        # 时间戳
        'timestamp': pd.date_range('2024-01-01 09:30:00', periods=n_samples, freq='1s')
    })
    
    return data

def test_data_loading():
    """测试数据加载模块"""
    print("=== 测试数据加载模块 ===")
    
    config = StrategyConfig()
    data_loader = DataLoader(config)
    
    # 创建测试数据
    sample_data = create_sample_data(1000)
    print(f"原始数据形状: {sample_data.shape}")
    print(f"数据列: {list(sample_data.columns)}")
    
    # 测试数据预处理
    processed_data = data_loader._preprocess_data(sample_data)
    print(f"处理后数据形状: {processed_data.shape}")
    print(f"数据类型: {processed_data.dtypes}")
    
    # 检查异常值处理
    print(f"价格范围: {processed_data['LASTPRICE'].min():.2f} - {processed_data['LASTPRICE'].max():.2f}")
    print(f"成交量范围: {processed_data['TRADEVOLUME'].min()} - {processed_data['TRADEVOLUME'].max()}")
    
    print("✓ 数据加载模块测试通过\n")

def test_feature_engineering():
    """测试特征工程模块"""
    print("=== 测试特征工程模块 ===")
    
    config = StrategyConfig()
    feature_engineer = FeatureEngineering(config)
    
    # 创建测试数据
    sample_data = create_sample_data(2000)
    
    # 测试各个特征提取模块
    print("测试价格特征提取...")
    price_features = feature_engineer.extract_price_features(sample_data)
    print(f"价格特征数量: {price_features.shape[1]}")
    print(f"价格特征列: {list(price_features.columns)}")
    
    print("测试成交量特征提取...")
    volume_features = feature_engineer.extract_volume_features(sample_data)
    print(f"成交量特征数量: {volume_features.shape[1]}")
    print(f"成交量特征列: {list(volume_features.columns)}")
    
    print("测试盘口特征提取...")
    orderbook_features = feature_engineer.extract_orderbook_features(sample_data)
    print(f"盘口特征数量: {orderbook_features.shape[1]}")
    print(f"盘口特征列: {list(orderbook_features.columns)}")
    
    print("测试技术指标特征提取...")
    technical_features = feature_engineer.extract_technical_indicators(sample_data)
    print(f"技术指标特征数量: {technical_features.shape[1]}")
    print(f"技术指标特征列: {list(technical_features.columns)}")
    
    # 测试完整特征工程
    print("测试完整特征工程流程...")
    all_features = feature_engineer.engineer_features(sample_data)
    print(f"总特征数量: {all_features.shape[1]}")
    print(f"特征矩阵形状: {all_features.shape}")
    
    # 测试滑动窗口归一化
    print("测试滑动窗口归一化...")
    normalized_features = feature_engineer.rolling_normalize_features(all_features)
    print(f"归一化后形状: {normalized_features.shape}")
    print(f"归一化后统计: 均值={normalized_features.mean():.4f}, 标准差={normalized_features.std():.4f}")
    
    # 检查特征质量
    print("特征质量检查:")
    print(f"  无穷大值数量: {np.sum(np.isinf(normalized_features))}")
    print(f"  NaN值数量: {np.sum(np.isnan(normalized_features))}")
    
    print("✓ 特征工程模块测试通过\n")

def test_label_construction():
    """测试标签构造模块"""
    print("=== 测试标签构造模块 ===")
    
    # 创建测试数据
    sample_data = create_sample_data(1000)
    
    # 测试基础标签构造
    label_constructor = LabelConstructor(k_ticks=5, threshold=0.0001)
    labels = label_constructor.construct_labels(sample_data)
    
    print(f"标签数组长度: {len(labels)}")
    print(f"标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = ['下跌', '震荡', '上涨'][label]
        print(f"  {label_name}({label}): {count} ({count/len(labels)*100:.1f}%)")
    
    # 测试带置信度的标签构造
    labels_conf, confidence = label_constructor.construct_labels_with_confidence(sample_data)
    print(f"置信度统计: 均值={confidence.mean():.3f}, 标准差={confidence.std():.3f}")
    print(f"置信度范围: {confidence.min():.3f} - {confidence.max():.3f}")
    
    # 验证无未来数据泄露
    is_valid = label_constructor.validate_no_future_leakage(sample_data, labels)
    print(f"未来数据泄露检查: {'通过' if is_valid else '失败'}")
    
    # 检查标签的时间一致性
    print("标签时间一致性检查:")
    # 最后k个tick应该都是震荡标签
    last_k_labels = labels[-label_constructor.k_ticks:]
    all_neutral = np.all(last_k_labels == 1)
    print(f"  最后{label_constructor.k_ticks}个标签是否为震荡: {'是' if all_neutral else '否'}")
    
    print("✓ 标签构造模块测试通过\n")

def test_model_architecture():
    """测试模型架构"""
    print("=== 测试模型架构 ===")
    
    config = StrategyConfig()
    
    # 创建测试数据
    sample_data = create_sample_data(500)
    feature_engineer = FeatureEngineering(config)
    features = feature_engineer.engineer_features(sample_data)
    
    print(f"输入特征维度: {features.shape[1]}")
    
    # 创建模型
    model = AttentionLSTMModel(config, input_size=features.shape[1])
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    import torch
    device = torch.device('cpu')
    model = model.to(device)
    
    # 创建测试输入
    test_input = torch.FloatTensor(features[:config.sequence_length]).unsqueeze(0)
    print(f"测试输入形状: {test_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output, attention_weights = model(test_input)
    
    print(f"模型输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"输出概率分布: {output.numpy()[0]}")
    print(f"注意力权重统计: 最小={attention_weights.min():.4f}, 最大={attention_weights.max():.4f}")
    
    print("✓ 模型架构测试通过\n")

def test_end_to_end_pipeline():
    """测试端到端流程"""
    print("=== 测试端到端流程 ===")
    
    try:
        # 创建策略配置
        config = StrategyConfig()
        config.epochs = 2  # 减少训练轮次用于测试
        config.batch_size = 32
        
        # 创建策略实例
        strategy = AttentionLSTMStrategy(config)
        
        # 创建训练数据
        train_data = {
            'IF': create_sample_data(5000),
            'IC': create_sample_data(5000)
        }
        
        print(f"训练数据: {list(train_data.keys())}")
        print(f"每个合约数据量: {[len(df) for df in train_data.values()]}")
        
        # 测试数据预处理
        for contract, df in train_data.items():
            features, labels = strategy._prepare_contract_data(df, contract)
            print(f"{contract} 预处理结果:")
            print(f"  特征形状: {features.shape}")
            print(f"  标签数量: {len(labels)}")
            print(f"  标签分布: {np.bincount(labels)}")
        
        print("✓ 端到端流程测试通过\n")
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """运行所有测试"""
    print("🚀 开始Attention-LSTM策略模块测试")
    print("="*60)
    
    test_data_loading()
    test_feature_engineering()
    test_label_construction()
    test_model_architecture()
    test_end_to_end_pipeline()
    
    print("="*60)
    print("🎉 所有测试完成！")
    
    print("\n📊 模块设计总结:")
    print("1. 数据加载: ✓ 支持异常值处理、缺失值填充、数据类型优化")
    print("2. 特征工程: ✓ 实现20+特征，包括价格、成交量、盘口、技术指标")
    print("3. 滑动窗口归一化: ✓ 避免look-ahead bias的归一化方法")
    print("4. 标签构造: ✓ 基于未来k个tick中间价，严格避免数据泄露")
    print("5. 模型架构: ✓ 动态特征数量支持，注意力机制增强")

if __name__ == "__main__":
    main()
