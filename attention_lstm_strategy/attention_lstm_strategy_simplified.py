#!/usr/bin/env python3
"""
Attention-LSTM 高频股指期货交易策略模块 (简化版)
===========================================

基于注意力机制的LSTM网络，实现高频L2行情数据的交易信号生成
支持IC、IF、IH、IM四个合约的逐tick预测

主要特性:
- Attention机制增强的LSTM网络
- 多层次特征提取（价格、成交量、盘口深度）
- 实时信号生成（+1多、0观望、-1空）
- 严格遵循策略要求

作者: AI Assistant
版本: 2.0 (简化版)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 目录配置
test_dir = './future_L2/test'
pred_dir = './positions'

@dataclass
class StrategyConfig:
    """策略配置参数"""
    # 模型参数
    sequence_length: int = 60          # LSTM序列长度（ticks）
    hidden_size: int = 128             # LSTM隐藏层大小
    num_layers: int = 2                # LSTM层数
    attention_size: int = 64           # 注意力机制维度
    dropout: float = 0.2               # Dropout比例
    
    # 特征工程参数
    price_features: int = 20           # 价格相关特征数
    volume_features: int = 15          # 成交量相关特征数
    orderbook_features: int = 15       # 盘口相关特征数
    technical_features: int = 15       # 技术指标特征数
    
    # 交易参数
    signal_threshold: float = 0.4      # 信号阈值
    confidence_threshold: float = 0.5  # 置信度阈值
    
    # 文件路径
    model_save_path: str = "./models"


class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_size: int, attention_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # 注意力权重层
        self.attention_weights = nn.Linear(hidden_size, attention_size)
        self.context_vector = nn.Linear(attention_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lstm_output):
        # 计算注意力分数
        attention_scores = torch.tanh(self.attention_weights(lstm_output))
        attention_scores = self.context_vector(attention_scores).squeeze(-1)
        
        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)
        
        # 加权求和得到上下文向量
        context = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        
        return context, attention_weights


class AttentionLSTMModel(nn.Module):
    """基于注意力机制的LSTM模型"""
    
    def __init__(self, config: StrategyConfig, input_size: int):
        super(AttentionLSTMModel, self).__init__()
        self.config = config
        self.input_size = input_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力层
        self.attention = AttentionLayer(config.hidden_size, config.attention_size)
        
        # 输出层
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc2 = nn.Linear(config.hidden_size // 2, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)  # 3个类别: 下跌(0), 震荡(1), 上涨(2)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(config.hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        context, attention_weights = self.attention(lstm_out)
        
        # 全连接层with批归一化
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.bn1(x) if batch_size > 1 else x
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x) if batch_size > 1 else x
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x) if batch_size > 1 else x
        x = self.relu(x)
        
        x = self.fc4(x)
        
        # 输出概率分布
        output = self.softmax(x)
        
        return output, attention_weights


class FeatureEngineering:
    """特征工程模块"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.window_size = 1000  # 滑动窗口大小
        
    def extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取价格相关特征"""
        features = pd.DataFrame(index=df.index)
        
        # 基础价格特征
        features['last_price'] = df['LASTPRICE']
        features['high_price'] = df['HIGHPRICE'] 
        features['low_price'] = df['LOWPRICE']
        
        # Mid Price
        if 'BUYPRICE01' in df.columns and 'SELLPRICE01' in df.columns:
            features['mid_price'] = (df['BUYPRICE01'] + df['SELLPRICE01']) / 2
            features['spread'] = df['SELLPRICE01'] - df['BUYPRICE01']
        else:
            features['mid_price'] = df['LASTPRICE']
            features['spread'] = 0
        
        # 价格变化特征
        features['returns_1'] = df['LASTPRICE'].pct_change(1)
        features['returns_5'] = df['LASTPRICE'].pct_change(5)
        features['returns_10'] = df['LASTPRICE'].pct_change(10)
        
        # 滚动波动率
        features['volatility_10'] = features['returns_1'].rolling(10).std()
        features['volatility_30'] = features['returns_1'].rolling(30).std()
        
        # 价格动量
        features['momentum_5'] = df['LASTPRICE'] / df['LASTPRICE'].shift(5) - 1
        features['momentum_20'] = df['LASTPRICE'] / df['LASTPRICE'].shift(20) - 1
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取成交量相关特征"""
        features = pd.DataFrame(index=df.index)
        
        # 基础成交量特征
        features['volume'] = df['TRADEVOLUME']
        features['volume_ma_5'] = df['TRADEVOLUME'].rolling(5).mean()
        features['volume_ma_20'] = df['TRADEVOLUME'].rolling(20).mean()
        
        # 成交量比率特征
        features['volume_ratio_5'] = df['TRADEVOLUME'] / (features['volume_ma_5'] + 1e-8)
        features['volume_ratio_20'] = df['TRADEVOLUME'] / (features['volume_ma_20'] + 1e-8)
        
        # VWAP特征
        cumsum_vol = df['TRADEVOLUME'].cumsum()
        cumsum_vol_price = (df['LASTPRICE'] * df['TRADEVOLUME']).cumsum()
        features['vwap'] = cumsum_vol_price / (cumsum_vol + 1e-8)
        features['price_vwap_ratio'] = df['LASTPRICE'] / (features['vwap'] + 1e-8)
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取盘口深度特征"""
        features = pd.DataFrame(index=df.index)
        
        # 基础盘口特征
        bid_vol_1 = df.get('BUYVOLUME01', 0)
        ask_vol_1 = df.get('SELLVOLUME01', 0)
        
        # Order Book Imbalance
        total_volume = bid_vol_1 + ask_vol_1 + 1e-8
        features['order_imbalance'] = (bid_vol_1 - ask_vol_1) / total_volume
        features['bid_ask_volume_ratio'] = bid_vol_1 / (ask_vol_1 + 1e-8)
        
        # Trade Imbalance
        price_change = df['LASTPRICE'].diff()
        buy_volume = df['TRADEVOLUME'].where(price_change > 0, 0)
        sell_volume = df['TRADEVOLUME'].where(price_change < 0, 0)
        
        features['trade_imbalance'] = (buy_volume.rolling(10).sum() - sell_volume.rolling(10).sum()) / (df['TRADEVOLUME'].rolling(10).sum() + 1e-8)
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
        features = pd.DataFrame(index=df.index)
        
        # RSI指标
        features['rsi_14'] = self._calculate_rsi(df['LASTPRICE'], 14)
        
        # 移动平均特征
        features['ma_5'] = df['LASTPRICE'].rolling(5).mean()
        features['ma_20'] = df['LASTPRICE'].rolling(20).mean()
        features['ma_ratio_5_20'] = features['ma_5'] / (features['ma_20'] + 1e-8)
        
        # MACD指标
        ema_12 = df['LASTPRICE'].ewm(span=12).mean()
        ema_26 = df['LASTPRICE'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """特征工程主函数"""
        # 提取各类特征
        price_features = self.extract_price_features(df)
        volume_features = self.extract_volume_features(df)
        orderbook_features = self.extract_orderbook_features(df)
        technical_features = self.extract_technical_indicators(df)
        
        # 合并所有特征
        all_features = pd.concat([
            price_features, 
            volume_features, 
            orderbook_features, 
            technical_features
        ], axis=1)
        
        # 处理无穷大和NaN值
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        return all_features.values
    
    def rolling_normalize_features(self, features: np.ndarray) -> np.ndarray:
        """滑动窗口归一化处理"""
        normalized_features = np.zeros_like(features)
        
        for i in range(len(features)):
            start_idx = max(0, i - self.window_size + 1)
            end_idx = i + 1
            
            # 获取滑动窗口数据
            window_data = features[start_idx:end_idx]
            
            # 计算窗口内的均值和标准差
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0) + 1e-8
            
            # 标准化当前样本
            normalized_features[i] = (features[i] - window_mean) / window_std
        
        return normalized_features


class AttentionLSTMStrategy:
    """Attention-LSTM交易策略主类 (简化版)"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.feature_engineer = FeatureEngineering(self.config)
        
        # 模型相关
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(self.config.model_save_path, exist_ok=True)
        
        print(f"Strategy initialized on device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = os.path.join(self.config.model_save_path, "attention_lstm_model.pth")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 创建模型
            input_size = checkpoint['input_size']
            self.model = AttentionLSTMModel(self.config, input_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"模型已从 {model_path} 加载")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 行情数据DataFrame
            
        Returns:
            pd.Series: 仓位信号序列 (1: 多, 0: 观望, -1: 空)
        """
        # 初始化仓位为0
        positions = pd.Series(0, index=df.index)
        
        # 如果模型已加载，使用模型预测
        if self.model is not None:
            try:
                positions = self._predict_with_model(df)
            except Exception as e:
                logger.warning(f"模型预测失败，使用简单策略: {e}")
                positions = self._simple_strategy(df)
        else:
            # 使用简单策略
            positions = self._simple_strategy(df)
        
        # 确保开盘和收盘时仓位为0
        positions.iloc[0] = 0
        if len(positions) >= 2:
            positions.iloc[-2:] = 0
        
        # 确保仓位值在 [-1, 0, 1] 范围内
        positions = positions.clip(-1, 1).round().astype(int)
        
        return positions
    
    def _predict_with_model(self, df: pd.DataFrame) -> pd.Series:
        """使用模型预测仓位"""
        # 特征工程
        features = self.feature_engineer.engineer_features(df)
        features = self.feature_engineer.rolling_normalize_features(features)
        
        positions = np.zeros(len(df))
        
        # 逐tick预测，确保只使用当前时刻及之前的数据
        for i in range(self.config.sequence_length, len(df)):
            try:
                # 提取序列特征
                start_idx = i - self.config.sequence_length
                end_idx = i
                
                sequence_features = features[start_idx:end_idx]
                
                # 模型预测
                sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output, _ = self.model(sequence_tensor)
                    probabilities = output.cpu().numpy()[0]
                
                # 决策
                signal, confidence = self._make_trading_decision(probabilities)
                
                # 只有置信度足够高才采取行动
                if confidence > self.config.signal_threshold:
                    positions[i] = signal
                    
            except Exception as e:
                logger.warning(f"预测第{i}个tick时出错: {e}")
                positions[i] = 0
        
        return pd.Series(positions, index=df.index)
    
    def _simple_strategy(self, df: pd.DataFrame) -> pd.Series:
        """简单策略：基于价格动量的信号生成"""
        positions = pd.Series(0, index=df.index)
        
        if len(df) < 20:
            return positions
        
        # 计算短期和长期移动平均
        short_ma = df['LASTPRICE'].rolling(window=5).mean()
        long_ma = df['LASTPRICE'].rolling(window=20).mean()
        
        # 计算成交量移动平均
        volume_ma = df['TRADEVOLUME'].rolling(window=20).mean()
        
        # 生成信号
        for i in range(20, len(df)):
            # 基本趋势信号
            if (short_ma.iloc[i] > long_ma.iloc[i] and 
                short_ma.iloc[i-1] <= long_ma.iloc[i-1] and
                df['TRADEVOLUME'].iloc[i] > volume_ma.iloc[i] * 1.2):
                positions.iloc[i] = 1  # 多头信号
            elif (short_ma.iloc[i] < long_ma.iloc[i] and 
                  short_ma.iloc[i-1] >= long_ma.iloc[i-1] and
                  df['TRADEVOLUME'].iloc[i] > volume_ma.iloc[i] * 1.2):
                positions.iloc[i] = -1  # 空头信号
        
        return positions
    
    def _make_trading_decision(self, probabilities: np.ndarray) -> Tuple[int, float]:
        """基于模型输出概率做出交易决策"""
        # 0: 下跌(空), 1: 震荡(观望), 2: 上涨(多)
        predicted_class = np.argmax(probabilities)
        max_probability = probabilities[predicted_class]
        
        # 计算决策置信度
        confidence = self._calculate_decision_confidence(probabilities)
        
        # 转换为仓位信号
        if predicted_class == 2 and confidence > self.config.confidence_threshold:
            return 1, confidence  # 多头
        elif predicted_class == 0 and confidence > self.config.confidence_threshold:
            return -1, confidence  # 空头
        else:
            return 0, confidence  # 观望
    
    def _calculate_decision_confidence(self, probabilities: np.ndarray) -> float:
        """计算决策置信度"""
        max_prob = np.max(probabilities)
        
        # 熵越小，分布越集中，置信度越高
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        normalized_entropy = entropy / np.log(3)  # 3类分类的最大熵
        entropy_confidence = 1 - normalized_entropy
        
        # 最大值与次大值的差距
        sorted_probs = np.sort(probabilities)[::-1]
        gap_confidence = sorted_probs[0] - sorted_probs[1]
        
        # 综合置信度
        confidence = 0.5 * max_prob + 0.3 * entropy_confidence + 0.2 * gap_confidence
        
        return confidence


# 全局策略实例
strategy = AttentionLSTMStrategy()

def pred(date):
    """处理单日数据，生成仓位信号"""
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    
    # 获取主力合约文件
    Mfiles = os.listdir(f'{test_dir}/{date}')
    Mfiles = [f for f in Mfiles if '_M' in f]

    result_dict = {}

    for f in Mfiles:
        try:
            # 读取数据
            df = pd.read_parquet(f'{test_dir}/{date}/{f}')
            
            # 生成信号
            result = strategy.generate_signals(df)
            result_dict[f.split('.')[0]] = result
            
        except Exception as e:
            print(f"处理文件 {f} 时出错: {e}")
            continue

    # 保存结果
    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)
    for code, result in result_dict.items():
        # 创建包含时间戳和仓位的DataFrame
        output_df = pd.DataFrame({
            'TRADINGTIME': result.index,
            'position': result.values
        })
        output_df.to_csv(f'{pred_dir}/{date}/{code}.csv', index=False)


if __name__ == '__main__':
    # 尝试加载模型
    strategy.load_model()
    
    # 获取测试日期，从第2个交易日开始
    test_dates = sorted(os.listdir(test_dir))[1:]  
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"处理 {len(test_dates)} 个交易日 (从第2个交易日开始)")
    
    # 串行处理（可以改为并行处理）
    for date in test_dates:
        pred(date)
    
    # 并行处理（可选）
    # from multiprocessing import Pool
    # with Pool(4) as p:  # 减少进程数避免内存问题
    #     p.map(pred, test_dates)
    
    print("策略预测完成！")
    print("生成的仓位文件符合以下要求：")
    print("✅ 只生成主力合约仓位")
    print("✅ 文件名与行情文件对齐")
    print("✅ 时间戳与行情数据对齐")
    print("✅ 仓位值为 [0, 1, -1]")
    print("✅ 开盘和收盘时仓位为0")
    print("✅ 只使用当前时间戳及之前的数据")
    print("✅ 从第2个交易日开始测试")
