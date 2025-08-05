#!/usr/bin/env python3
"""
Attention-LSTM 模型训练脚本
=============================

独立的模型训练脚本，用于训练 Attention-LSTM 模型

使用方法:
python train_attention_lstm.py

作者: AI Assistant
版本: 1.0
"""

import os
import sys
from attention_lstm_strategy import AttentionLSTMStrategy, StrategyConfig

def main():
    """主训练函数"""
    print("=" * 50)
    print("Attention-LSTM 模型训练")
    print("=" * 50)
    
    # 创建策略配置
    config = StrategyConfig(
        # 模型参数
        sequence_length=60,
        hidden_size=128,
        num_layers=2,
        attention_size=64,
        dropout=0.2,
        
        # 训练参数
        batch_size=32,          # 减小批次大小以适应内存
        learning_rate=0.001,
        num_epochs=20,          # 减少训练轮数进行快速测试
        patience=5,
        
        # 阈值参数
        signal_threshold=0.4,
        confidence_threshold=0.5
    )
    
    # 创建策略实例
    strategy = AttentionLSTMStrategy(config)
    
    # 检查训练数据目录
    train_dir = './future_L2/train'
    if not os.path.exists(train_dir):
        print(f"错误: 训练目录不存在 {train_dir}")
        print("请确保训练数据存在于 future_L2/train/ 目录下")
        sys.exit(1)
    
    # 开始训练
    print("开始模型训练...")
    try:
        success = strategy.train_model(data_dir=train_dir, save_model=True)
        
        if success:
            print("✅ 模型训练成功完成!")
            print(f"模型已保存到: {strategy.config.model_save_path}/attention_lstm_model.pth")
            
            # 测试模型加载
            print("\n测试模型加载...")
            if strategy.load_model():
                print("✅ 模型加载测试成功!")
            else:
                print("❌ 模型加载测试失败!")
        else:
            print("❌ 模型训练失败!")
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n训练脚本执行完成!")

if __name__ == '__main__':
    main()
