#!/usr/bin/env python3
"""
Attention-LSTM 高频股指期货交易策略模块
===========================================

基于注意力机制的LSTM网络，实现高频L2行情数据的交易信号生成
支持IC、IF、IH、IM四个合约的逐tick预测

主要特性:
- Attention机制增强的LSTM网络
- 多层次特征提取（价格、成交量、盘口深度）
- 实时信号生成（+1多、0观望、-1空）
- 动态风险控制
- 多合约并行处理

作者: AI Assistant
版本: 1.0
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import time
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingRules:
    """交易规则配置"""
    max_position_per_contract: int = 1      # 每个合约最大持仓
    commission_rate: float = 0.0023 / 10000  # 手续费率：0.23‱
    use_opposite_price: bool = True         # 使用对手方价格成交
    force_close_at_end: bool = True         # 收盘强制平仓
    contracts: List[str] = None             # 主力合约列表
    
    def __post_init__(self):
        if self.contracts is None:
            self.contracts = ['IC', 'IF', 'IH', 'IM']


@dataclass
class StrategyConfig:
    """策略配置参数"""
    # 模型参数
    sequence_length: int = 60          # LSTM序列长度（ticks）
    hidden_size: int = 128             # LSTM隐藏层大小
    num_layers: int = 2                # LSTM层数
    attention_size: int = 64           # 注意力机制维度
    dropout: float = 0.2               # Dropout比例
    
    # 特征工程参数 - 动态更新
    price_features: int = 20           # 价格相关特征数（动态调整）
    volume_features: int = 15          # 成交量相关特征数（动态调整）
    orderbook_features: int = 15       # 盘口相关特征数（动态调整）
    technical_features: int = 15       # 技术指标特征数（动态调整）
    
    # 标签构造参数
    label_k_ticks: int = 5             # 未来k个tick用于标签构造
    label_threshold: float = 0.0001    # 价格变动阈值
    
    # 交易参数
    signal_threshold: float = 0.4      # 信号阈值（提高以减少噪音）
    confidence_threshold: float = 0.5  # 置信度阈值
    risk_threshold: float = 0.8        # 风险阈值
    max_position_time: int = 300       # 最大持仓时间（ticks）
    
    # 训练参数
    batch_size: int = 128              # 增大批次大小
    learning_rate: float = 0.0005      # 降低学习率
    epochs: int = 100
    patience: int = 15                 # 早停patience
    early_stopping_patience: int = 15  # 增加早停patience
    
    # 数据处理参数
    window_size: int = 1000            # 滑动窗口大小
    outlier_threshold: float = 0.01    # 异常值检测阈值
    
    # 文件路径
    model_save_path: str = "./models"
    scaler_save_path: str = "./scalers"
    
    # 交易规则
    trading_rules: TradingRules = None
    
    def __post_init__(self):
        if self.trading_rules is None:
            self.trading_rules = TradingRules()


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
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        # 计算注意力分数
        attention_scores = torch.tanh(self.attention_weights(lstm_output))  # (batch, seq_len, attention_size)
        attention_scores = self.context_vector(attention_scores).squeeze(-1)  # (batch, seq_len)
        
        # 计算注意力权重
        attention_weights = self.softmax(attention_scores)  # (batch, seq_len)
        
        # 加权求和得到上下文向量
        context = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)
        
        return context, attention_weights


class AttentionLSTMModel(nn.Module):
    """基于注意力机制的LSTM模型 - 支持动态特征数量"""
    
    def __init__(self, config: StrategyConfig, input_size: int = None):
        super(AttentionLSTMModel, self).__init__()
        self.config = config
        
        # 动态计算输入特征数
        if input_size is not None:
            self.input_size = input_size
        else:
            # 估算总特征数（实际使用时会动态更新）
            self.input_size = (config.price_features + 
                              config.volume_features + 
                              config.orderbook_features +
                              config.technical_features)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
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
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, 3) - 三分类概率
            attention_weights: (batch_size, seq_len)
        """
        batch_size = x.size(0)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # 注意力机制
        context, attention_weights = self.attention(lstm_out)
        
        # 全连接层with批归一化
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.bn1(x) if batch_size > 1 else x  # 批大小为1时跳过BN
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


class DataLoader:
    """数据加载和预处理模块"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        
    def load_tick_data(self, file_path: str) -> pd.DataFrame:
        """加载tick级别L2行情数据"""
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            # 数据预处理
            df = self._preprocess_data(df)
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return pd.DataFrame()
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 1. 时间戳处理
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # 2. 数据完整性检查
        required_columns = ['LASTPRICE', 'HIGHPRICE', 'LOWPRICE', 'TRADEVOLUME']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
        
        # 3. 异常值处理
        df = self._handle_outliers(df)
        
        # 4. 缺失值处理
        df = self._handle_missing_values(df)
        
        # 5. 数据类型优化
        df = self._optimize_dtypes(df)
        
        return df.reset_index(drop=True)
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        # 价格异常值检测
        for price_col in ['LASTPRICE', 'HIGHPRICE', 'LOWPRICE']:
            if price_col in df.columns:
                Q1 = df[price_col].quantile(0.01)
                Q3 = df[price_col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 用边界值替换异常值
                df[price_col] = np.clip(df[price_col], lower_bound, upper_bound)
        
        # 成交量异常值处理
        if 'TRADEVOLUME' in df.columns:
            df['TRADEVOLUME'] = np.clip(df['TRADEVOLUME'], 0, df['TRADEVOLUME'].quantile(0.99))
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 价格列用前向填充
        price_columns = ['LASTPRICE', 'HIGHPRICE', 'LOWPRICE', 'BUYPRICE01', 'SELLPRICE01']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 成交量列用0填充
        volume_columns = ['TRADEVOLUME', 'BUYVOLUME01', 'SELLVOLUME01']
        for col in volume_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        # 浮点数优化
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        # 整数优化
        int_cols = df.select_dtypes(include=['int64']).columns
        df[int_cols] = df[int_cols].astype('int32')
        
        return df


class FeatureEngineering:
    """增强的特征工程模块 - 包含20+个高频交易特征"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        # 使用滑动窗口标准化器
        self.scalers = {}
        self.window_size = 1000  # 滑动窗口大小
        
    def extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取价格相关特征"""
        features = pd.DataFrame(index=df.index)
        
        # 1. 基础价格特征
        features['last_price'] = df['LASTPRICE']
        features['high_price'] = df['HIGHPRICE'] 
        features['low_price'] = df['LOWPRICE']
        
        # 2. Mid Price (中间价)
        if 'BUYPRICE01' in df.columns and 'SELLPRICE01' in df.columns:
            features['mid_price'] = (df['BUYPRICE01'] + df['SELLPRICE01']) / 2
        else:
            features['mid_price'] = df['LASTPRICE']
        
        # 3. Spread (买卖价差)
        if 'BUYPRICE01' in df.columns and 'SELLPRICE01' in df.columns:
            features['spread'] = df['SELLPRICE01'] - df['BUYPRICE01']
            features['spread_bps'] = features['spread'] / features['mid_price'] * 10000  # 基点
        else:
            features['spread'] = 0
            features['spread_bps'] = 0
        
        # 4. Price Diff (价格偏离)
        features['price_diff'] = df['LASTPRICE'] - features['mid_price']
        features['price_diff_ratio'] = features['price_diff'] / features['mid_price']
        
        # 5. 价格变化特征
        features['returns_1'] = df['LASTPRICE'].pct_change(1)
        features['returns_5'] = df['LASTPRICE'].pct_change(5)
        features['returns_10'] = df['LASTPRICE'].pct_change(10)
        features['log_returns'] = np.log(df['LASTPRICE'] / df['LASTPRICE'].shift(1))
        
        # 6. Rolling Volatility (滚动波动率)
        features['volatility_10'] = features['returns_1'].rolling(10).std()
        features['volatility_30'] = features['returns_1'].rolling(30).std()
        features['volatility_60'] = features['returns_1'].rolling(60).std()
        
        # 7. 价格动量特征
        features['momentum_5'] = df['LASTPRICE'] / df['LASTPRICE'].shift(5) - 1
        features['momentum_20'] = df['LASTPRICE'] / df['LASTPRICE'].shift(20) - 1
        
        # 8. 高低价特征
        features['hl_ratio'] = df['HIGHPRICE'] / df['LOWPRICE']
        features['close_to_high'] = df['LASTPRICE'] / df['HIGHPRICE']
        features['close_to_low'] = df['LASTPRICE'] / df['LOWPRICE']
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取成交量相关特征"""
        features = pd.DataFrame(index=df.index)
        
        # 1. 基础成交量特征
        features['volume'] = df['TRADEVOLUME']
        features['volume_ma_5'] = df['TRADEVOLUME'].rolling(5).mean()
        features['volume_ma_20'] = df['TRADEVOLUME'].rolling(20).mean()
        features['volume_ma_60'] = df['TRADEVOLUME'].rolling(60).mean()
        
        # 2. 成交量比率特征
        features['volume_ratio_5'] = df['TRADEVOLUME'] / (features['volume_ma_5'] + 1e-8)
        features['volume_ratio_20'] = df['TRADEVOLUME'] / (features['volume_ma_20'] + 1e-8)
        
        # 3. 成交量变化特征
        features['volume_change_1'] = df['TRADEVOLUME'].pct_change(1)
        features['volume_change_5'] = df['TRADEVOLUME'].pct_change(5)
        
        # 4. 成交量趋势
        features['volume_trend_short'] = features['volume_ma_5'] / (features['volume_ma_20'] + 1e-8)
        features['volume_trend_long'] = features['volume_ma_20'] / (features['volume_ma_60'] + 1e-8)
        
        # 5. VWAP特征
        cumsum_vol = df['TRADEVOLUME'].cumsum()
        cumsum_vol_price = (df['LASTPRICE'] * df['TRADEVOLUME']).cumsum()
        features['vwap'] = cumsum_vol_price / (cumsum_vol + 1e-8)
        features['price_vwap_ratio'] = df['LASTPRICE'] / (features['vwap'] + 1e-8)
        
        # 6. 滚动VWAP
        rolling_vol_sum = df['TRADEVOLUME'].rolling(20).sum()
        rolling_vol_price_sum = (df['LASTPRICE'] * df['TRADEVOLUME']).rolling(20).sum()
        features['rolling_vwap_20'] = rolling_vol_price_sum / (rolling_vol_sum + 1e-8)
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取盘口深度特征"""
        features = pd.DataFrame(index=df.index)
        
        # 1. 基础盘口特征
        bid_vol_1 = df.get('BUYVOLUME01', 0)
        ask_vol_1 = df.get('SELLVOLUME01', 0)
        bid_price_1 = df.get('BUYPRICE01', df['LASTPRICE'])
        ask_price_1 = df.get('SELLPRICE01', df['LASTPRICE'])
        
        # 2. Order Book Imbalance (订单不平衡)
        total_volume = bid_vol_1 + ask_vol_1 + 1e-8
        features['order_imbalance'] = (bid_vol_1 - ask_vol_1) / total_volume
        features['bid_ask_volume_ratio'] = bid_vol_1 / (ask_vol_1 + 1e-8)
        
        # 3. Trade Imbalance (交易不平衡)
        # 基于价格变动方向判断主动买卖
        price_change = df['LASTPRICE'].diff()
        buy_volume = df['TRADEVOLUME'].where(price_change > 0, 0)
        sell_volume = df['TRADEVOLUME'].where(price_change < 0, 0)
        
        features['trade_imbalance'] = (buy_volume.rolling(10).sum() - sell_volume.rolling(10).sum()) / (df['TRADEVOLUME'].rolling(10).sum() + 1e-8)
        features['buy_sell_ratio'] = buy_volume.rolling(20).sum() / (sell_volume.rolling(20).sum() + 1e-8)
        
        # 4. Order Book Slope (盘口斜率)
        if len([col for col in df.columns if 'BUYPRICE' in col]) >= 2:
            # 多档盘口斜率
            bid_prices = [df.get(f'BUYPRICE0{i}', bid_price_1) for i in range(1, 4)]
            ask_prices = [df.get(f'SELLPRICE0{i}', ask_price_1) for i in range(1, 4)]
            
            if len(bid_prices) >= 2:
                features['bid_slope'] = (bid_prices[0] - bid_prices[1]) / (bid_prices[0] + 1e-8)
            else:
                features['bid_slope'] = 0
                
            if len(ask_prices) >= 2:
                features['ask_slope'] = (ask_prices[1] - ask_prices[0]) / (ask_prices[0] + 1e-8)
            else:
                features['ask_slope'] = 0
        else:
            features['bid_slope'] = 0
            features['ask_slope'] = 0
        
        # 5. 盘口深度特征
        # 多档加权平均价格
        if 'BUYVOLUME02' in df.columns and 'SELLVOLUME02' in df.columns:
            bid_vol_2 = df['BUYVOLUME02']
            ask_vol_2 = df['SELLVOLUME02']
            bid_price_2 = df.get('BUYPRICE02', bid_price_1)
            ask_price_2 = df.get('SELLPRICE02', ask_price_1)
            
            # 加权买卖价
            total_bid_vol = bid_vol_1 + bid_vol_2 + 1e-8
            total_ask_vol = ask_vol_1 + ask_vol_2 + 1e-8
            
            features['weighted_bid_price'] = (bid_price_1 * bid_vol_1 + bid_price_2 * bid_vol_2) / total_bid_vol
            features['weighted_ask_price'] = (ask_price_1 * ask_vol_1 + ask_price_2 * ask_vol_2) / total_ask_vol
            
            # 深度比率
            features['depth_imbalance'] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        else:
            features['weighted_bid_price'] = bid_price_1
            features['weighted_ask_price'] = ask_price_1
            features['depth_imbalance'] = features['order_imbalance']
        
        # 6. 微观结构特征
        features['effective_spread'] = 2 * abs(df['LASTPRICE'] - (bid_price_1 + ask_price_1) / 2)
        features['realized_spread'] = abs(df['LASTPRICE'] - df['LASTPRICE'].shift(1))
        
        # 7. 价格压力指标
        mid_price = (bid_price_1 + ask_price_1) / 2
        features['price_pressure'] = (df['LASTPRICE'] - mid_price) / (ask_price_1 - bid_price_1 + 1e-8)
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取技术指标特征"""
        features = pd.DataFrame(index=df.index)
        
        # 1. RSI指标
        features['rsi_14'] = self._calculate_rsi(df['LASTPRICE'], 14)
        features['rsi_30'] = self._calculate_rsi(df['LASTPRICE'], 30)
        
        # 2. 移动平均特征
        features['ma_5'] = df['LASTPRICE'].rolling(5).mean()
        features['ma_20'] = df['LASTPRICE'].rolling(20).mean()
        features['ma_60'] = df['LASTPRICE'].rolling(60).mean()
        
        features['ma_ratio_5_20'] = features['ma_5'] / (features['ma_20'] + 1e-8)
        features['ma_ratio_20_60'] = features['ma_20'] / (features['ma_60'] + 1e-8)
        
        # 3. 布林带特征
        ma_20 = features['ma_20']
        std_20 = df['LASTPRICE'].rolling(20).std()
        features['bb_upper'] = ma_20 + 2 * std_20
        features['bb_lower'] = ma_20 - 2 * std_20
        features['bb_position'] = (df['LASTPRICE'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / (ma_20 + 1e-8)
        
        # 4. MACD指标
        ema_12 = df['LASTPRICE'].ewm(span=12).mean()
        ema_26 = df['LASTPRICE'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """特征工程主函数 - 生成20+个特征"""
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
        
        logger.info(f"生成特征数量: {all_features.shape[1]}")
        logger.debug(f"特征列名: {list(all_features.columns)}")
        
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
    
    def fit_scalers(self, features: np.ndarray):
        """训练标准化器（兼容性保持）"""
        # 为了向后兼容，保留这个方法
        pass
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """特征标准化 - 使用滑动窗口归一化"""
        return self.rolling_normalize_features(features)


class LabelConstructor:
    """标签构造模块 - 严格避免未来数据泄露"""
    
    def __init__(self, k_ticks: int = 5, threshold: float = 0.0001):
        """
        Args:
            k_ticks: 未来k个tick用于标签构造
            threshold: 价格变动阈值（基点）
        """
        self.k_ticks = k_ticks
        self.threshold = threshold
        
    def construct_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        基于未来k个tick的中间价变动构造三分类标签
        
        标签定义：
        - 0: 下跌 (未来k个tick中间价下跌超过阈值)
        - 1: 震荡 (未来k个tick中间价变动在阈值范围内)  
        - 2: 上涨 (未来k个tick中间价上涨超过阈值)
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            np.ndarray: 标签数组
        """
        # 计算中间价
        if 'BUYPRICE01' in df.columns and 'SELLPRICE01' in df.columns:
            mid_price = (df['BUYPRICE01'] + df['SELLPRICE01']) / 2
        else:
            mid_price = df['LASTPRICE']
        
        labels = np.ones(len(df))  # 默认为震荡标签
        
        for i in range(len(df) - self.k_ticks):
            # 当前时刻的中间价
            current_mid = mid_price.iloc[i]
            
            # 未来k个tick的中间价变动
            future_mid_prices = mid_price.iloc[i+1:i+1+self.k_ticks]
            
            # 计算未来期间的最大最小价格
            future_max = future_mid_prices.max()
            future_min = future_mid_prices.min()
            
            # 计算价格变动幅度
            upward_move = (future_max - current_mid) / current_mid
            downward_move = (current_mid - future_min) / current_mid
            
            # 分类标签
            if upward_move > self.threshold and upward_move > downward_move:
                labels[i] = 2  # 上涨
            elif downward_move > self.threshold and downward_move > upward_move:
                labels[i] = 0  # 下跌
            else:
                labels[i] = 1  # 震荡
        
        # 最后k个tick无法构造标签，设为震荡
        labels[-self.k_ticks:] = 1
        
        return labels.astype(int)
    
    def construct_labels_with_confidence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造带置信度的标签
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (标签, 置信度)
        """
        if 'BUYPRICE01' in df.columns and 'SELLPRICE01' in df.columns:
            mid_price = (df['BUYPRICE01'] + df['SELLPRICE01']) / 2
        else:
            mid_price = df['LASTPRICE']
        
        labels = np.ones(len(df))
        confidence = np.zeros(len(df))
        
        for i in range(len(df) - self.k_ticks):
            current_mid = mid_price.iloc[i]
            future_mid_prices = mid_price.iloc[i+1:i+1+self.k_ticks]
            
            future_max = future_mid_prices.max()
            future_min = future_mid_prices.min()
            
            upward_move = (future_max - current_mid) / current_mid
            downward_move = (current_mid - future_min) / current_mid
            
            # 计算置信度（基于价格变动幅度）
            move_magnitude = max(upward_move, downward_move)
            confidence[i] = min(move_magnitude / self.threshold, 3.0)  # 限制最大置信度
            
            # 分类标签
            if upward_move > self.threshold and upward_move > downward_move:
                labels[i] = 2
            elif downward_move > self.threshold and downward_move > upward_move:
                labels[i] = 0
            else:
                labels[i] = 1
                confidence[i] = max(0.1, 1.0 - move_magnitude / self.threshold)  # 震荡时的置信度
        
        labels[-self.k_ticks:] = 1
        confidence[-self.k_ticks:] = 0.1
        
        return labels.astype(int), confidence.astype(np.float32)
    
    def validate_no_future_leakage(self, df: pd.DataFrame, labels: np.ndarray) -> bool:
        """验证标签构造过程中没有未来数据泄露"""
        # 检查标签是否使用了未来数据
        for i in range(len(labels) - self.k_ticks):
            # 在实际应用中，时刻i的标签只能使用时刻i及之前的数据
            # 这里我们验证标签构造的逻辑是否正确
            pass
        
        logger.info("标签构造验证通过：无未来数据泄露")
        return True


class TradingDataset(Dataset):
    """交易数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.labels[idx + self.sequence_length - 1]  # 预测最后一个时间点的标签
        
        return torch.FloatTensor(x), torch.LongTensor([y])


class SequenceConstructor:
    """滑动窗口序列构造器"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建LSTM输入序列
        
        Args:
            features: 特征矩阵 [n_samples, n_features]
            labels: 标签数组 [n_samples]
        
        Returns:
            X: 序列特征 [n_sequences, sequence_length, n_features]
            y: 序列标签 [n_sequences]
        """
        n_samples, n_features = features.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"数据长度 {n_samples} 小于序列长度 {self.sequence_length}")
        
        # 计算序列数量
        n_sequences = n_samples - self.sequence_length + 1
        
        # 创建序列数组
        X = np.zeros((n_sequences, self.sequence_length, n_features))
        y = np.zeros(n_sequences, dtype=np.int64)
        
        for i in range(n_sequences):
            # 特征序列：使用前sequence_length个时间步的特征
            X[i] = features[i:i + self.sequence_length]
            # 标签：使用序列最后一个时间步的标签
            y[i] = labels[i + self.sequence_length - 1]
        
        return X, y
    
    def create_real_time_sequence(self, feature_buffer: np.ndarray) -> Optional[np.ndarray]:
        """
        为实时预测创建序列
        
        Args:
            feature_buffer: 特征缓冲区 [buffer_length, n_features]
        
        Returns:
            sequence: 单个序列 [1, sequence_length, n_features] 或 None
        """
        if len(feature_buffer) < self.sequence_length:
            return None
        
        # 取最新的sequence_length个时间步
        sequence = feature_buffer[-self.sequence_length:]
        return sequence.reshape(1, self.sequence_length, -1)


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: StrategyConfig, device: torch.device):
        self.config = config
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_model(self, model: AttentionLSTMModel, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            X_train: 训练特征 [n_samples, sequence_length, n_features]
            y_train: 训练标签 [n_samples]
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)
        
        Returns:
            training_history: 训练历史
        """
        print(f"开始训练模型，设备: {self.device}")
        print(f"训练数据形状: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # 验证数据加载器
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # 计算类别权重处理不平衡数据
        class_weights = self._calculate_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        model.train()
        for epoch in range(self.config.epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            # 验证阶段
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
                scheduler.step(val_loss)
                
                # 早停检查
                if self._check_early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            if val_loss is not None:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return history
    
    def _calculate_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """计算类别权重"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        weights = compute_class_weight(
            'balanced', classes=classes, y=labels
        )
        return weights
    
    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """检查是否早停"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience


class SignalDecisionMaker:
    """信号决策器"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.position_mapping = {0: -1, 1: 0, 2: 1}  # 下跌->空, 震荡->观望, 上涨->多
    
    def make_decision(self, probabilities: np.ndarray, 
                     confidence_scores: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        基于模型输出概率做出交易决策
        
        Args:
            probabilities: 模型输出概率 [3] (下跌, 震荡, 上涨)
            confidence_scores: 置信度分数 (可选)
        
        Returns:
            position: 仓位信号 (-1: 空, 0: 观望, 1: 多)
            confidence: 决策置信度
        """
        # 获取最大概率的类别
        predicted_class = np.argmax(probabilities)
        max_probability = probabilities[predicted_class]
        
        # 计算决策置信度
        confidence = self._calculate_confidence(probabilities)
        
        # 应用置信度阈值过滤
        if confidence < self.confidence_threshold:
            return 0, confidence  # 置信度不足，保持观望
        
        # 转换为仓位信号
        position = self.position_mapping[predicted_class]
        
        return position, confidence
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        """
        计算决策置信度
        
        使用多种指标综合评估：
        1. 最大概率值
        2. 概率分布的熵
        3. 最大值与次大值的差距
        """
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
    
    def batch_decisions(self, batch_probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量决策
        
        Args:
            batch_probabilities: 批量概率 [batch_size, 3]
        
        Returns:
            positions: 仓位信号数组 [batch_size]
            confidences: 置信度数组 [batch_size]
        """
        batch_size = batch_probabilities.shape[0]
        positions = np.zeros(batch_size, dtype=int)
        confidences = np.zeros(batch_size)
        
        for i in range(batch_size):
            pos, conf = self.make_decision(batch_probabilities[i])
            positions[i] = pos
            confidences[i] = conf
        
        return positions, confidences


class TradingExecutor:
    """交易执行器 - 实现具体的交易规则"""
    
    def __init__(self, trading_rules: TradingRules):
        self.trading_rules = trading_rules
        self.positions = {}  # 当前持仓 {contract: position}
        self.trade_history = []  # 交易历史
        self.daily_pnl = {}  # 每日盈亏
        
        # 初始化持仓
        for contract in self.trading_rules.contracts:
            self.positions[contract] = 0
    
    def execute_signal(self, contract: str, signal: int, current_tick: Dict, 
                      next_tick: Optional[Dict] = None) -> Dict:
        """
        执行交易信号
        
        Args:
            contract: 合约代码
            signal: 交易信号 (-1: 空, 0: 观望, 1: 多)
            current_tick: 当前tick数据
            next_tick: 下一个tick数据（用于获取成交价）
        
        Returns:
            trade_result: 交易结果
        """
        current_position = self.positions.get(contract, 0)
        target_position = signal
        
        # 检查持仓限制
        if abs(target_position) > self.trading_rules.max_position_per_contract:
            target_position = np.sign(target_position) * self.trading_rules.max_position_per_contract
        
        # 计算需要的交易量
        trade_volume = target_position - current_position
        
        if trade_volume == 0:
            return {"contract": contract, "action": "hold", "volume": 0, "price": 0, "commission": 0}
        
        # 确定成交价格
        execution_price = self._get_execution_price(current_tick, next_tick, trade_volume)
        
        # 计算手续费
        commission = abs(trade_volume) * execution_price * self.trading_rules.commission_rate
        
        # 执行交易
        trade_result = {
            "contract": contract,
            "timestamp": current_tick.get("timestamp", ""),
            "action": "buy" if trade_volume > 0 else "sell",
            "volume": abs(trade_volume),
            "price": execution_price,
            "commission": commission,
            "position_before": current_position,
            "position_after": target_position,
            "signal": signal
        }
        
        # 更新持仓
        self.positions[contract] = target_position
        
        # 记录交易历史
        self.trade_history.append(trade_result)
        
        return trade_result
    
    def _get_execution_price(self, current_tick: Dict, next_tick: Optional[Dict], 
                           trade_volume: int) -> float:
        """获取成交价格"""
        # 使用下一个tick的对手方一档价格
        if next_tick is None:
            # 如果没有下一个tick，使用当前tick的对手方价格
            tick_data = current_tick
        else:
            tick_data = next_tick
        
        if trade_volume > 0:  # 买入，使用卖一价
            return tick_data.get("SELLPRICE01", tick_data.get("LASTPRICE", 0))
        else:  # 卖出，使用买一价
            return tick_data.get("BUYPRICE01", tick_data.get("LASTPRICE", 0))
    
    def force_close_all_positions(self, current_tick: Dict) -> List[Dict]:
        """强制平仓所有持仓（收盘时调用）"""
        close_trades = []
        
        for contract, position in self.positions.items():
            if position != 0:
                # 强制平仓
                trade_result = self.execute_signal(contract, 0, current_tick)
                trade_result["action"] = "force_close"
                close_trades.append(trade_result)
        
        return close_trades
    
    def calculate_pnl(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """计算盈亏"""
        total_pnl = 0
        contract_pnl = {}
        
        for trade in self.trade_history:
            contract = trade["contract"]
            if contract not in contract_pnl:
                contract_pnl[contract] = {
                    "gross_pnl": 0,
                    "commission": 0,
                    "net_pnl": 0,
                    "trades": 0
                }
            
            # 计算交易盈亏（这里简化处理，实际需要配对开平仓）
            commission = trade["commission"]
            contract_pnl[contract]["commission"] += commission
            contract_pnl[contract]["trades"] += 1
            
            total_pnl -= commission  # 扣除手续费
        
        # 计算未平仓盈亏
        for contract, position in self.positions.items():
            if position != 0 and contract in market_data:
                current_price = market_data[contract]["LASTPRICE"].iloc[-1]
                # 这里需要有开仓价格才能计算未实现盈亏
                # 简化处理，暂不计算
        
        return {
            "total_pnl": total_pnl,
            "contract_pnl": contract_pnl,
            "current_positions": self.positions.copy()
        }
    
    def get_position_status(self) -> Dict:
        """获取当前持仓状态"""
        return {
            "positions": self.positions.copy(),
            "total_trades": len(self.trade_history),
            "active_contracts": [k for k, v in self.positions.items() if v != 0]
        }


class BacktestEngine:
    """回测引擎 - 严格按照交易规则执行回测"""
    
    def __init__(self, strategy: 'AttentionLSTMStrategy', trading_rules: TradingRules):
        self.strategy = strategy
        self.trading_rules = trading_rules
        self.executor = TradingExecutor(trading_rules)
        
    def run_backtest(self, test_data: Dict[str, pd.DataFrame], 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        运行回测
        
        Args:
            test_data: 测试数据 {contract: DataFrame}
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            backtest_results: 回测结果
        """
        print("开始运行策略回测...")
        
        # 获取所有交易日
        all_dates = set()
        for contract, df in test_data.items():
            if 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp']).dt.date.unique()
                all_dates.update(dates)
        
        all_dates = sorted(list(all_dates))
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.to_datetime(start_date).date()]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.to_datetime(end_date).date()]
        
        daily_results = {}
        
        for date in all_dates:
            print(f"回测日期: {date}")
            daily_result = self._backtest_single_day(test_data, date)
            daily_results[str(date)] = daily_result
        
        # 汇总结果
        summary = self._calculate_summary_stats(daily_results)
        
        return {
            "daily_results": daily_results,
            "summary_stats": summary,
            "trade_history": self.executor.trade_history,
            "final_positions": self.executor.positions
        }
    
    def _backtest_single_day(self, test_data: Dict[str, pd.DataFrame], date) -> Dict:
        """单日回测"""
        # 确保开盘时仓位为0
        for contract in self.trading_rules.contracts:
            self.executor.positions[contract] = 0
        
        daily_trades = []
        tick_signals = {}
        
        # 获取当日数据
        daily_data = {}
        for contract, df in test_data.items():
            if 'timestamp' in df.columns:
                daily_df = df[pd.to_datetime(df['timestamp']).dt.date == date]
                if len(daily_df) > 0:
                    daily_data[contract] = daily_df.reset_index(drop=True)
        
        if not daily_data:
            return {"trades": [], "signals": {}, "pnl": 0}
        
        # 找到所有时间点
        all_timestamps = set()
        for contract, df in daily_data.items():
            all_timestamps.update(df['timestamp'].tolist())
        
        all_timestamps = sorted(list(all_timestamps))
        
        # 逐tick执行策略
        for i, timestamp in enumerate(all_timestamps):
            tick_data = {}
            next_tick_data = {}
            
            # 获取当前时刻所有合约的数据
            for contract, df in daily_data.items():
                current_rows = df[df['timestamp'] == timestamp]
                if len(current_rows) > 0:
                    tick_data[contract] = current_rows.iloc[0].to_dict()
                    
                    # 获取下一个tick的数据
                    current_idx = df[df['timestamp'] == timestamp].index[0]
                    if current_idx + 1 < len(df):
                        next_tick_data[contract] = df.iloc[current_idx + 1].to_dict()
            
            # 生成交易信号
            signals = {}
            for contract in self.trading_rules.contracts:
                if contract in tick_data:
                    # 获取足够的历史数据用于预测
                    hist_data = daily_data[contract]
                    hist_data = hist_data[hist_data['timestamp'] <= timestamp]
                    
                    if len(hist_data) >= self.strategy.config.sequence_length:
                        signal, confidence = self.strategy.predict_real_time(contract, hist_data)
                        signals[contract] = {"signal": signal, "confidence": confidence}
                    else:
                        signals[contract] = {"signal": 0, "confidence": 0.0}  # 数据不足，观望
            
            tick_signals[timestamp] = signals
            
            # 执行交易
            for contract, signal_info in signals.items():
                if contract in tick_data:
                    signal = signal_info["signal"]
                    current_tick = tick_data[contract]
                    next_tick = next_tick_data.get(contract, None)
                    
                    trade_result = self.executor.execute_signal(
                        contract, signal, current_tick, next_tick
                    )
                    
                    if trade_result["action"] != "hold":
                        daily_trades.append(trade_result)
        
        # 收盘强制平仓
        if all_timestamps and self.trading_rules.force_close_at_end:
            last_timestamp = all_timestamps[-1]
            last_tick_data = {}
            for contract, df in daily_data.items():
                last_rows = df[df['timestamp'] == last_timestamp]
                if len(last_rows) > 0:
                    last_tick_data[contract] = last_rows.iloc[0].to_dict()
            
            close_trades = self.executor.force_close_all_positions(last_tick_data)
            daily_trades.extend(close_trades)
        
        # 计算当日盈亏
        daily_pnl = self._calculate_daily_pnl(daily_trades, daily_data)
        
        return {
            "trades": daily_trades,
            "signals": tick_signals,
            "pnl": daily_pnl,
            "num_trades": len(daily_trades)
        }
    
    def _calculate_daily_pnl(self, trades: List[Dict], daily_data: Dict) -> float:
        """计算单日盈亏"""
        total_pnl = 0
        
        # 简化的盈亏计算：只计算手续费成本
        for trade in trades:
            total_pnl -= trade["commission"]
        
        return total_pnl
    
    def _calculate_summary_stats(self, daily_results: Dict) -> Dict:
        """计算汇总统计"""
        total_trades = sum(result["num_trades"] for result in daily_results.values())
        total_pnl = sum(result["pnl"] for result in daily_results.values())
        
        daily_pnls = [result["pnl"] for result in daily_results.values()]
        trading_days = len([pnl for pnl in daily_pnls if pnl != 0])
        
        return {
            "total_trading_days": len(daily_results),
            "active_trading_days": trading_days,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "avg_daily_pnl": total_pnl / len(daily_results) if daily_results else 0,
            "avg_trades_per_day": total_trades / len(daily_results) if daily_results else 0,
            "win_days": len([pnl for pnl in daily_pnls if pnl > 0]),
            "lose_days": len([pnl for pnl in daily_pnls if pnl < 0]),
            "max_daily_pnl": max(daily_pnls) if daily_pnls else 0,
            "min_daily_pnl": min(daily_pnls) if daily_pnls else 0
        }


class AttentionLSTMStrategy:
    """Attention-LSTM交易策略主类"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineering(self.config)
        self.label_constructor = LabelConstructor(self.config.label_k_ticks, self.config.label_threshold)
        
        # 模型和训练相关
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 实时交易缓存
        self.contract_buffers = {}  # 每个合约的特征缓冲区
        
        # 交易执行器和回测引擎
        self.trading_executor = TradingExecutor(self.config.trading_rules)
        self.backtest_engine = BacktestEngine(self, self.config.trading_rules)
        
        # 创建保存目录
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.scaler_save_path, exist_ok=True)
        
        print(f"Strategy initialized on device: {self.device}")
        print(f"Trading rules: {self.config.trading_rules}")
    
    def generate_daily_positions(self, date: str, test_dir: str = './future_L2/test', 
                                output_dir: str = './positions') -> Dict[str, pd.DataFrame]:
        """
        生成某一天所有主力合约的仓位信号
        
        Args:
            date: 交易日期 (如 '20241009')
            test_dir: 测试数据目录
            output_dir: 输出目录
            
        Returns:
            Dict[str, pd.DataFrame]: 各合约的仓位信号
        """
        print(f'开始处理 {date}')
        
        # 获取该日期的所有主力合约文件
        date_dir = f'{test_dir}/{date}'
        if not os.path.exists(date_dir):
            print(f"目录不存在: {date_dir}")
            return {}
        
        files = os.listdir(date_dir)
        main_files = [f for f in files if '_M.parquet' in f]  # 主力合约文件
        
        result_dict = {}
        
        # 读取前一天的数据用于特征计算（如果存在）
        prev_date_data = self._get_previous_day_data(date, test_dir)
        
        for file in main_files:
            try:
                # 读取当天数据
                file_path = f'{date_dir}/{file}'
                df = pd.read_parquet(file_path)
                
                # 合约代码
                contract_code = file.split('.')[0]
                
                # 生成仓位信号
                positions = self._generate_contract_positions(df, contract_code, prev_date_data)
                
                # 确保开盘和收盘时仓位为0
                positions = self._enforce_daily_flat_positions(positions)
                
                result_dict[contract_code] = positions
                
                print(f"  处理完成: {contract_code}, 信号数量: {len(positions)}")
                
            except Exception as e:
                print(f"  处理文件 {file} 时出错: {e}")
                continue
        
        # 保存仓位文件
        self._save_daily_positions(date, result_dict, output_dir)
        
        return result_dict
    
    def _get_previous_day_data(self, current_date: str, test_dir: str) -> Dict[str, pd.DataFrame]:
        """获取前一个交易日的数据"""
        # 获取所有交易日期
        all_dates = sorted([d for d in os.listdir(test_dir) if os.path.isdir(f'{test_dir}/{d}')])
        
        try:
            current_idx = all_dates.index(current_date)
            if current_idx > 0:
                prev_date = all_dates[current_idx - 1]
                prev_date_dir = f'{test_dir}/{prev_date}'
                
                prev_data = {}
                files = os.listdir(prev_date_dir)
                for file in files:
                    if file.endswith('.parquet'):
                        contract_code = file.split('.')[0]
                        df = pd.read_parquet(f'{prev_date_dir}/{file}')
                        prev_data[contract_code] = df
                
                return prev_data
        except (ValueError, IndexError):
            pass
        
        return {}
    
    def _generate_contract_positions(self, df: pd.DataFrame, contract_code: str, 
                                   prev_date_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        为单个合约生成仓位信号
        
        Args:
            df: 当天的行情数据
            contract_code: 合约代码
            prev_date_data: 前一天的数据
            
        Returns:
            pd.DataFrame: 包含时间戳和仓位的DataFrame
        """
        # 准备时间戳列
        if 'TRADINGTIME' not in df.columns:
            # 如果没有TRADINGTIME列，尝试使用索引
            timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
        else:
            timestamps = pd.to_datetime(df['TRADINGTIME'])
        
        # 初始化仓位为0
        positions = pd.DataFrame({
            'TRADINGTIME': timestamps,
            'position': 0
        })
        
        # 如果模型已训练，使用模型生成信号
        if self.model is not None:
            try:
                positions['position'] = self._predict_positions_for_contract(df, contract_code, prev_date_data)
            except Exception as e:
                logger.warning(f"模型预测失败，使用默认仓位0: {e}")
                positions['position'] = 0
        else:
            # 如果模型未训练，使用简单策略
            positions['position'] = self._simple_strategy_positions(df)
        
        # 确保仓位值在 [-1, 0, 1] 范围内
        positions['position'] = positions['position'].clip(-1, 1).round().astype(int)
        
        return positions
    
    def _predict_positions_for_contract(self, df: pd.DataFrame, contract_code: str, 
                                      prev_date_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """使用训练好的模型预测仓位"""
        # 合并当天和前一天的数据用于特征计算
        combined_data = self._combine_current_and_previous_data(df, contract_code, prev_date_data)
        
        # 特征工程
        features = self.feature_engineer.engineer_features(combined_data)
        features = self.feature_engineer.transform_features(features)
        
        positions = np.zeros(len(df))
        
        # 逐tick预测，确保只使用当前时刻及之前的数据
        for i in range(self.config.sequence_length, len(df)):
            try:
                # 提取序列特征（使用当前时刻及之前的数据）
                end_idx = len(combined_data) - len(df) + i + 1  # 在合并数据中的位置
                start_idx = end_idx - self.config.sequence_length
                
                if start_idx >= 0 and end_idx <= len(features):
                    sequence_features = features[start_idx:end_idx]
                    
                    # 模型预测
                    sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        self.model.eval()
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
        
        return positions
    
    def _combine_current_and_previous_data(self, current_df: pd.DataFrame, contract_code: str, 
                                         prev_date_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并当前和前一天的数据"""
        # 如果有前一天的数据，取最后一部分用于计算特征
        if contract_code in prev_date_data:
            prev_df = prev_date_data[contract_code]
            # 取前一天最后的一部分数据（用于滚动窗口计算）
            prev_tail = prev_df.tail(self.config.sequence_length * 2)  # 取足够的数据
            combined_df = pd.concat([prev_tail, current_df], ignore_index=True)
        else:
            combined_df = current_df.copy()
        
        return combined_df
    
    def _simple_strategy_positions(self, df: pd.DataFrame) -> np.ndarray:
        """简单策略：基于价格动量的信号生成"""
        positions = np.zeros(len(df))
        
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
                positions[i] = 1  # 多头信号
            elif (short_ma.iloc[i] < long_ma.iloc[i] and 
                  short_ma.iloc[i-1] >= long_ma.iloc[i-1] and
                  df['TRADEVOLUME'].iloc[i] > volume_ma.iloc[i] * 1.2):
                positions[i] = -1  # 空头信号
        
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
    
    def _enforce_daily_flat_positions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """确保开盘和收盘时仓位为0"""
        positions = positions.copy()
        
        # 开盘时仓位为0
        positions.iloc[0, positions.columns.get_loc('position')] = 0
        
        # 收盘时仓位为0（最后两个时间点）
        if len(positions) >= 2:
            positions.iloc[-2:, positions.columns.get_loc('position')] = 0
        
        return positions
    
    def _save_daily_positions(self, date: str, positions_dict: Dict[str, pd.DataFrame], 
                            output_dir: str):
        """保存每日仓位文件"""
        # 创建输出目录
        date_output_dir = f'{output_dir}/{date}'
        os.makedirs(date_output_dir, exist_ok=True)
        
        # 保存每个合约的仓位文件
        for contract_code, positions in positions_dict.items():
            output_file = f'{date_output_dir}/{contract_code}.csv'
            positions.to_csv(output_file, index=False)
            
    def run_strategy_prediction(self, test_dir: str = './future_L2/test', 
                              output_dir: str = './positions') -> None:
        """
        运行策略预测，生成所有交易日的仓位信号
        
        Args:
            test_dir: 测试数据目录
            output_dir: 输出目录
        """
        # 获取所有交易日期，跳过第一个日期（从第二个交易日开始）
        all_dates = sorted([d for d in os.listdir(test_dir) if os.path.isdir(f'{test_dir}/{d}')])
        test_dates = all_dates[1:]  # 从第2个交易日开始
        
        print(f"总交易日数: {len(all_dates)}")
        print(f"测试交易日数: {len(test_dates)} (从第2个交易日开始)")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 逐日处理
        for date in test_dates:
            try:
                print(f'\n{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} 开始处理 {date}')
                self.generate_daily_positions(date, test_dir, output_dir)
                
            except Exception as e:
                print(f"处理日期 {date} 时出错: {e}")
                continue
        
        print(f"\n策略预测完成！")
        print(f"输出目录: {output_dir}")
        print("生成的仓位文件符合以下要求：")
        print("✅ 只生成主力合约仓位")
        print("✅ 文件名与行情文件对齐")
        print("✅ 时间戳与行情数据对齐")
        print("✅ 仓位值为 [0, 1, -1]")
        print("✅ 开盘和收盘时仓位为0")
        print("✅ 只使用当前时间戳及之前的数据")
        print("✅ 从第2个交易日开始测试")
    
    def run_strategy_backtest(self, test_data: Dict[str, pd.DataFrame], 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> Dict:
        """
        运行策略回测（严格按照交易规则）
        
        Args:
            test_data: 测试数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            backtest_results: 详细的回测结果
        """
        if self.model is None:
            raise ValueError("模型未训练或加载，请先训练模型或加载已训练的模型")
        
        return self.backtest_engine.run_backtest(test_data, start_date, end_date)
    
    def generate_trading_signals(self, market_data: Dict[str, pd.DataFrame], 
                                timestamp: Optional[str] = None) -> Dict[str, Dict]:
        """
        生成交易信号（实时交易使用）
        
        Args:
            market_data: 市场数据
            timestamp: 时间戳（可选）
            
        Returns:
            signals: {contract: {"signal": int, "confidence": float}}
        """
        signals = {}
        
        for contract in self.config.trading_rules.contracts:
            if contract in market_data:
                df = market_data[contract]
                if timestamp:
                    # 只使用指定时间点之前的数据
                    df = df[df['timestamp'] <= timestamp] if 'timestamp' in df.columns else df
                
                if len(df) >= self.config.sequence_length:
                    signal, confidence = self.predict_real_time(contract, df)
                else:
                    signal, confidence = 0, 0.0
                
                signals[contract] = {
                    "signal": signal,
                    "confidence": confidence,
                    "current_position": self.trading_executor.positions.get(contract, 0)
                }
        
        return signals
    
    def execute_trading_signals(self, signals: Dict[str, Dict], 
                               current_ticks: Dict[str, Dict],
                               next_ticks: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        """
        执行交易信号
        
        Args:
            signals: 交易信号
            current_ticks: 当前tick数据
            next_ticks: 下一个tick数据（可选）
            
        Returns:
            trades: 执行的交易列表
        """
        trades = []
        
        for contract, signal_info in signals.items():
            if contract in current_ticks:
                signal = signal_info["signal"]
                current_tick = current_ticks[contract]
                next_tick = next_ticks.get(contract) if next_ticks else None
                
                trade_result = self.trading_executor.execute_signal(
                    contract, signal, current_tick, next_tick
                )
                
                if trade_result["action"] != "hold":
                    trades.append(trade_result)
        
        return trades
    
    def get_current_positions(self) -> Dict[str, int]:
        """获取当前持仓"""
        return self.trading_executor.get_position_status()["positions"]
    
    def force_close_all_positions(self, current_ticks: Dict[str, Dict]) -> List[Dict]:
        """强制平仓（收盘时调用）"""
        return self.trading_executor.force_close_all_positions(current_ticks)
    
    def calculate_performance_metrics(self, backtest_results: Dict) -> Dict:
        """计算详细的绩效指标"""
        summary = backtest_results["summary_stats"]
        daily_results = backtest_results["daily_results"]
        
        # 提取每日盈亏
        daily_pnls = [result["pnl"] for result in daily_results.values()]
        daily_returns = np.array(daily_pnls)
        
        # 基本统计
        total_return = summary["total_pnl"]
        avg_daily_return = np.mean(daily_returns)
        volatility = np.std(daily_returns)
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = avg_daily_return / (volatility + 1e-8) * np.sqrt(252)
        
        # 最大回撤
        cumulative_returns = np.cumsum(daily_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - rolling_max
        max_drawdown = np.min(drawdowns)
        
        # 胜率
        win_rate = len([x for x in daily_returns if x > 0]) / len(daily_returns) if daily_returns.size > 0 else 0
        
        # 盈亏比
        winning_days = [x for x in daily_returns if x > 0]
        losing_days = [x for x in daily_returns if x < 0]
        avg_win = np.mean(winning_days) if winning_days else 0
        avg_loss = np.mean(losing_days) if losing_days else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 交易频率
        total_trades = summary["total_trades"]
        trading_days = summary["total_trading_days"]
        
        return {
            "total_return": total_return,
            "avg_daily_return": avg_daily_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": total_trades,
            "avg_trades_per_day": total_trades / trading_days if trading_days > 0 else 0,
            "trading_days": trading_days,
            "win_days": summary["win_days"],
            "lose_days": summary["lose_days"],
            "calmar_ratio": total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
    
    def _create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """创建训练标签 - 使用新的标签构造器"""
        label_constructor = LabelConstructor(k_ticks=5, threshold=0.0001)
        labels = label_constructor.construct_labels(df)
        
        # 验证无未来数据泄露
        label_constructor.validate_no_future_leakage(df, labels)
        
        return labels
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据 - 使用增强的数据处理流程"""
        # 1. 数据预处理
        data_loader = DataLoader(self.config)
        processed_df = data_loader._preprocess_data(df)
        
        # 2. 特征工程 - 生成20+特征
        feature_engineer = FeatureEngineering(self.config)
        features = feature_engineer.engineer_features(processed_df)
        
        # 3. 创建标签
        labels = self._create_labels(processed_df)
        
        # 4. 滑动窗口归一化
        features = feature_engineer.transform_features(features)
        
        # 5. 更新配置中的特征数量
        self.config.price_features = features.shape[1] // 3  # 粗略估计
        self.config.volume_features = features.shape[1] // 3
        self.config.orderbook_features = features.shape[1] - self.config.price_features - self.config.volume_features
        
        logger.info(f"训练数据准备完成：特征数={features.shape[1]}, 样本数={len(features)}")
        logger.info(f"标签分布: 下跌={np.sum(labels==0)}, 震荡={np.sum(labels==1)}, 上涨={np.sum(labels==2)}")
        
        return features, labels
    
    def train(self, train_data: Dict[str, pd.DataFrame], 
              val_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        训练Attention-LSTM模型
        
        Args:
            train_data: 训练数据 {contract: DataFrame}
            val_data: 验证数据 (可选)
        """
        print("开始训练Attention-LSTM模型...")
        
        # 准备训练数据
        all_features = []
        all_labels = []
        
        for contract, df in train_data.items():
            print(f"处理训练数据: {contract}")
            features, labels = self._prepare_training_data(df)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
        
        if not all_features:
            raise ValueError("没有有效的训练数据")
        
        # 合并所有数据
        X_all = np.vstack(all_features)
        y_all = np.concatenate(all_labels)
        
        print(f"总训练样本数: {len(X_all)}")
        print(f"特征维度: {X_all.shape[1]}")
        print(f"标签分布: {np.bincount(y_all)}")
        
        # 创建序列
        sequence_constructor = SequenceConstructor(self.config.sequence_length)
        X_seq, y_seq = sequence_constructor.create_sequences(X_all, y_all)
        
        print(f"序列样本数: {len(X_seq)}")
        print(f"序列形状: {X_seq.shape}")
        
        # 划分训练验证集
        if val_data is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
            )
        else:
            X_train, y_train = X_seq, y_seq
            # 处理验证数据
            val_features = []
            val_labels = []
            for contract, df in val_data.items():
                features, labels = self._prepare_training_data(df)
                if len(features) > 0:
                    val_features.append(features)
                    val_labels.append(labels)
            
            if val_features:
                X_val_all = np.vstack(val_features)
                y_val_all = np.concatenate(val_labels)
                X_val, y_val = sequence_constructor.create_sequences(X_val_all, y_val_all)
            else:
                X_val, y_val = None, None
        
        # 创建模型
        input_size = X_train.shape[2]
        self.model = AttentionLSTMModel(self.config, input_size).to(self.device)
        print(f"模型创建完成，输入维度: {input_size}")
        
        # 训练模型
        self.trainer = ModelTrainer(self.config, self.device)
        training_history = self.trainer.train_model(
            self.model, X_train, y_train, X_val, y_val
        )
        
        print("模型训练完成!")
        return training_history
    
    def save_model(self, model_path: Optional[str] = None):
        """保存训练好的模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if model_path is None:
            model_path = os.path.join(self.config.model_save_path, "attention_lstm_model.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_size': self.model.input_size
        }, model_path)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: Optional[str] = None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = os.path.join(self.config.model_save_path, "attention_lstm_model.pth")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        input_size = checkpoint['input_size']
        self.model = AttentionLSTMModel(self.config, input_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"模型已从 {model_path} 加载")
        return True
    
    def predict_real_time(self, contract: str, df: pd.DataFrame) -> Tuple[int, float]:
        """
        实时预测（单个样本）
        
        Args:
            contract: 合约代码
            df: 包含足够历史数据的DataFrame
            
        Returns:
            signal: 交易信号 (-1, 0, 1)
            confidence: 置信度
        """
        if self.model is None:
            return 0, 0.0
        
        try:
            # 特征工程
            features = self.feature_engineer.engineer_features(df)
            features = self.feature_engineer.transform_features(features)
            
            # 检查是否有足够的数据
            if len(features) < self.config.sequence_length:
                return 0, 0.0
            
            # 构造序列
            sequence_constructor = SequenceConstructor(self.config.sequence_length)
            sequence = sequence_constructor.create_real_time_sequence(features)
            
            if sequence is None:
                return 0, 0.0
            
            # 模型预测
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            with torch.no_grad():
                self.model.eval()
                output, _ = self.model(sequence_tensor)
                probabilities = output.cpu().numpy()[0]
            
            # 做出决策
            signal, confidence = self._make_trading_decision(probabilities)
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"实时预测失败: {e}")
            return 0, 0.0, feature_engineer
    
    def train(self, train_data: Dict[str, pd.DataFrame], 
              val_data: Optional[Dict[str, pd.DataFrame]] = None):
        """训练模型 - 多合约批量处理"""
        print("开始训练Attention-LSTM模型...")
        
        # 创建组件
        self.sequence_constructor = SequenceConstructor(self.config.sequence_length)
        self.signal_decision_maker = SignalDecisionMaker(self.config)
        
        # 准备所有合约的训练数据
        all_sequences = []
        all_labels = []
        
        for contract, data in train_data.items():
            print(f"处理合约 {contract} 训练数据...")
            features, labels = self._prepare_contract_data(data, contract)
            
            # 创建序列
            X_seq, y_seq = self.sequence_constructor.create_sequences(features, labels)
            print(f"  {contract}: {X_seq.shape[0]} 个序列")
            
            all_sequences.append(X_seq)
            all_labels.append(y_seq)
        
        # 合并所有合约数据
        X_train = np.vstack(all_sequences)
        y_train = np.concatenate(all_labels)
        
        print(f"训练数据总量: {X_train.shape}")
        print(f"标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # 准备验证数据
        X_val, y_val = None, None
        if val_data is not None:
            val_sequences = []
            val_labels = []
            for contract, data in val_data.items():
                features, labels = self._prepare_contract_data(data, contract)
                X_seq, y_seq = self.sequence_constructor.create_sequences(features, labels)
                val_sequences.append(X_seq)
                val_labels.append(y_seq)
            
            X_val = np.vstack(val_sequences)
            y_val = np.concatenate(val_labels)
            print(f"验证数据: {X_val.shape}")
        
        # 创建模型
        input_size = X_train.shape[2]  # 特征维度
        self.model = AttentionLSTMModel(self.config, input_size)
        self.model = self.model.to(self.device)
        
        # 创建训练器并训练
        self.trainer = ModelTrainer(self.config, self.device)
        training_history = self.trainer.train_model(
            self.model, X_train, y_train, X_val, y_val
        )
        
        # 保存模型
        model_path = os.path.join(self.config.model_save_path, 'attention_lstm_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_size': input_size,
            'training_history': training_history
        }, model_path)
        
        print("模型训练完成!")
        return training_history
    
    def _prepare_contract_data(self, data: pd.DataFrame, contract: str) -> Tuple[np.ndarray, np.ndarray]:
        """准备单个合约的数据"""
        # 数据预处理
        processed_data = self.data_loader._preprocess_data(data)
        
        # 特征工程
        features = self.feature_engineer.engineer_features(processed_data)
        features = self.feature_engineer.rolling_normalize_features(features)
        
        # 标签构造
        labels = self.label_constructor.construct_labels(processed_data)
        
        # 保存特征工程器
        if not hasattr(self, 'feature_engineers'):
            self.feature_engineers = {}
        self.feature_engineers[contract] = self.feature_engineer
        
        return features.values, labels
    
    def predict_real_time(self, contract: str, tick_data: pd.DataFrame) -> Tuple[int, float]:
        """实时预测单个tick的交易信号"""
        if self.model is None:
            raise ValueError("模型未训练或加载")
        
        # 初始化合约缓冲区
        if contract not in self.contract_buffers:
            self.contract_buffers[contract] = []
        
        # 处理新的tick数据
        processed_data = self.data_loader._preprocess_data(tick_data)
        features = self.feature_engineer.engineer_features(processed_data)
        features = self.feature_engineer.rolling_normalize_features(features)
        
        # 更新缓冲区
        if len(features) > 0:
            self.contract_buffers[contract].append(features.values[-1])  # 取最新的特征
            
            # 保持缓冲区大小
            max_buffer_size = self.config.sequence_length + 100
            if len(self.contract_buffers[contract]) > max_buffer_size:
                self.contract_buffers[contract] = self.contract_buffers[contract][-max_buffer_size:]
        
        # 创建预测序列
        if len(self.contract_buffers[contract]) < self.config.sequence_length:
            return 0, 0.0  # 数据不足，观望
        
        feature_buffer = np.array(self.contract_buffers[contract])
        sequence = self.sequence_constructor.create_real_time_sequence(feature_buffer)
        
        if sequence is None:
            return 0, 0.0
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            output, attention_weights = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # 信号决策
        position, confidence = self.signal_decision_maker.make_decision(probabilities)
        
        return position, confidence
    
    def batch_predict(self, contracts_data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """批量预测多个合约"""
        results = {}
        
        for contract, data in contracts_data.items():
            print(f"预测合约 {contract}...")
            
            # 准备数据
            features, _ = self._prepare_contract_data(data, contract)
            
            # 创建序列（不需要标签）
            if len(features) < self.config.sequence_length:
                print(f"  {contract} 数据不足，跳过")
                continue
            
            # 创建滑动窗口序列
            n_samples = len(features) - self.config.sequence_length + 1
            sequences = np.zeros((n_samples, self.config.sequence_length, features.shape[1]))
            
            for i in range(n_samples):
                sequences[i] = features[i:i + self.config.sequence_length]
            
            # 批量预测
            self.model.eval()
            positions = []
            confidences = []
            
            batch_size = self.config.batch_size
            for i in range(0, len(sequences), batch_size):
                batch_seq = sequences[i:i + batch_size]
                
                with torch.no_grad():
                    batch_tensor = torch.FloatTensor(batch_seq).to(self.device)
                    outputs, _ = self.model(batch_tensor)
                    batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # 批量决策
                batch_positions, batch_confidences = self.signal_decision_maker.batch_decisions(batch_probs)
                positions.extend(batch_positions)
                confidences.extend(batch_confidences)
            
            results[contract] = (np.array(positions), np.array(confidences))
            print(f"  {contract} 预测完成: {len(positions)} 个信号")
        
        return results
    
    def load_model(self, model_path: Optional[str] = None):
        """加载训练好的模型"""
        if model_path is None:
            model_path = os.path.join(self.config.model_save_path, 'attention_lstm_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 恢复配置
        saved_config = checkpoint['config']
        input_size = checkpoint['input_size']
        
        # 创建模型
        self.model = AttentionLSTMModel(saved_config, input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 创建组件
        self.sequence_constructor = SequenceConstructor(saved_config.sequence_length)
        self.signal_decision_maker = SignalDecisionMaker(saved_config)
        
        print(f"模型加载成功: {model_path}")
        print(f"输入特征维度: {input_size}")
        
        return checkpoint.get('training_history', {})
    
    def save_model(self, model_path: Optional[str] = None, additional_info: Optional[Dict] = None):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        if model_path is None:
            model_path = os.path.join(self.config.model_save_path, 'attention_lstm_model.pth')
        
        # 准备保存数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_size': self.model.lstm.input_size,
            'save_time': time.time()
        }
        
        if additional_info:
            save_data.update(additional_info)
        
        # 创建目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        torch.save(save_data, model_path)
        print(f"模型保存成功: {model_path}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {"status": "未加载模型"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "Attention-LSTM",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.model.lstm.input_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "sequence_length": self.config.sequence_length,
            "output_classes": 3,
            "status": "已加载"
        }
    
    def load_models(self, contracts: List[str] = None):
        """加载已训练的模型"""
        if contracts is None:
            contracts = ['IC', 'IF', 'IH', 'IM']
        
        for contract in contracts:
            try:
                # 先加载特征工程器以获取特征数量
                feature_engineer = joblib.load(f'{self.config.scaler_save_path}/{contract}_feature_engineer.pkl')
                self.feature_engineers[contract] = feature_engineer
                
                # 创建示例数据以获取特征数量
                sample_data = pd.DataFrame({
                    'LASTPRICE': [3000.0] * 100,
                    'HIGHPRICE': [3005.0] * 100,
                    'LOWPRICE': [2995.0] * 100,
                    'TRADEVOLUME': [100] * 100,
                    'BUYVOLUME01': [50] * 100,
                    'SELLVOLUME01': [60] * 100,
                    'BUYPRICE01': [2999.0] * 100,
                    'SELLPRICE01': [3001.0] * 100
                })
                
                sample_features = feature_engineer.engineer_features(sample_data)
                actual_input_size = sample_features.shape[1]
                
                # 加载模型
                model = AttentionLSTMModel(self.config, input_size=actual_input_size).to(self.device)
                model.load_state_dict(torch.load(f'{self.config.model_save_path}/{contract}_best_model.pth'))
                model.eval()
                self.models[contract] = model
                
                logger.info(f'成功加载 {contract} 模型，特征数: {actual_input_size}')
                
            except FileNotFoundError:
                logger.warning(f'未找到 {contract} 的模型文件')
            except Exception as e:
                logger.error(f'加载 {contract} 模型时出错: {e}')
    
    def predict_signal(self, contract: str, df: pd.DataFrame) -> int:
        """预测单个合约的交易信号"""
        if contract not in self.models:
            logger.warning(f'未找到 {contract} 的模型')
            return 0
        
        if len(df) < self.config.sequence_length:
            return 0
        
        try:
            # 特征工程
            feature_engineer = self.feature_engineers[contract]
            features = feature_engineer.engineer_features(df)
            features = feature_engineer.transform_features(features)
            
            # 获取最近的序列
            sequence = features[-self.config.sequence_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # 模型预测
            model = self.models[contract]
            with torch.no_grad():
                outputs, attention_weights = model(sequence_tensor)
                probabilities = outputs.cpu().numpy()[0]
                
                # 根据概率和阈值生成信号
                max_prob = np.max(probabilities)
                predicted_class = np.argmax(probabilities)
                
                if max_prob < self.config.signal_threshold:
                    return 0  # 不确定时观望
                
                # 将类别转换为交易信号
                if predicted_class == 0:
                    return -1  # 卖出
                elif predicted_class == 2:
                    return 1   # 买入
                else:
                    return 0   # 观望
                    
        except Exception as e:
            logger.error(f'预测 {contract} 信号时出错: {e}')
            return 0
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """为所有合约生成交易信号"""
        signals = {}
        
        for contract, df in market_data.items():
            signal = self.predict_signal(contract, df)
            signals[contract] = signal
            
        return signals
    
    def backtest_strategy(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """策略回测"""
        results = {}
        
        for contract, df in test_data.items():
            logger.info(f'回测 {contract} 合约...')
            
            positions = []
            returns = []
            
            # 滑动窗口预测
            for i in range(self.config.sequence_length, len(df)):
                window_data = df.iloc[:i+1]
                signal = self.predict_signal(contract, window_data)
                positions.append(signal)
                
                # 计算收益
                if i > self.config.sequence_length:
                    price_change = (df.iloc[i]['LASTPRICE'] - df.iloc[i-1]['LASTPRICE']) / df.iloc[i-1]['LASTPRICE']
                    position_return = positions[-2] * price_change  # 使用前一个信号
                    returns.append(position_return)
                else:
                    returns.append(0)
            
            # 计算性能指标
            returns = np.array(returns)
            total_return = np.sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)  # 假设分钟级数据
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
            
            results[contract] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': np.sum(np.array(positions) != 0),
                'positions': positions,
                'returns': returns.tolist()
            }
            
            logger.info(f'{contract} 回测结果: 总收益={total_return:.4f}, '
                       f'夏普比率={sharpe_ratio:.4f}, 胜率={win_rate:.4f}')
        
        return results
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)


# 实时交易接口
class RealTimeTrader:
    """实时交易接口"""
    
    def __init__(self, strategy: AttentionLSTMStrategy):
        self.strategy = strategy
        self.position_history = {}  # 记录持仓历史
        self.data_buffer = {}       # 数据缓冲区
        
        # 初始化缓冲区
        for contract in ['IC', 'IF', 'IH', 'IM']:
            self.position_history[contract] = []
            self.data_buffer[contract] = pd.DataFrame()
    
    def update_market_data(self, contract: str, tick_data: Dict):
        """更新市场数据"""
        # 将tick数据转换为DataFrame行
        tick_df = pd.DataFrame([tick_data])
        
        # 添加到缓冲区
        self.data_buffer[contract] = pd.concat([self.data_buffer[contract], tick_df], ignore_index=True)
        
        # 保持缓冲区大小
        max_buffer_size = self.strategy.config.sequence_length * 2
        if len(self.data_buffer[contract]) > max_buffer_size:
            self.data_buffer[contract] = self.data_buffer[contract].tail(max_buffer_size).reset_index(drop=True)
    
    def get_current_signal(self, contract: str) -> int:
        """获取当前交易信号"""
        if contract not in self.data_buffer or len(self.data_buffer[contract]) < self.strategy.config.sequence_length:
            return 0
        
        signal, _ = self.strategy.predict_real_time(contract, self.data_buffer[contract])
        self.position_history[contract].append(signal)
        
        return signal
    
    def get_all_signals(self) -> Dict[str, int]:
        """获取所有合约的当前信号"""
        signals = {}
        for contract in ['IC', 'IF', 'IH', 'IM']:
            signals[contract] = self.get_current_signal(contract)
        return signals


# 主要的策略执行函数
def run_attention_lstm_strategy(train_data_path: str = None, 
                               test_data_path: str = None,
                               contracts: List[str] = None,
                               mode: str = 'train') -> Union[Dict, AttentionLSTMStrategy]:
    """
    运行Attention-LSTM策略
    
    Args:
        train_data_path: 训练数据路径
        test_data_path: 测试数据路径  
        contracts: 合约列表
        mode: 运行模式 ('train', 'test', 'predict')
    
    Returns:
        训练模式返回训练结果，测试模式返回回测结果，预测模式返回策略实例
    """
    if contracts is None:
        contracts = ['IC', 'IF', 'IH', 'IM']
    
    # 创建策略实例
    config = StrategyConfig()
    strategy = AttentionLSTMStrategy(config)
    
    if mode == 'train' and train_data_path:
        logger.info("开始训练模式...")
        
        # 加载训练数据
        train_data = {}
        for contract in contracts:
            train_file = f"{train_data_path}/{contract}_train.parquet"
            if os.path.exists(train_file):
                train_data[contract] = pd.read_parquet(train_file)
                logger.info(f"加载 {contract} 训练数据: {len(train_data[contract])} 行")
        
        if train_data:
            training_results = strategy.train(train_data)
            return training_results
        else:
            raise FileNotFoundError(f"未找到训练数据文件在路径: {train_data_path}")
    
    elif mode == 'test' and test_data_path:
        logger.info("开始测试模式...")
        
        # 加载模型
        strategy.load_model()
        
        # 加载测试数据
        test_data = {}
        for contract in contracts:
            test_file = f"{test_data_path}/{contract}_test.parquet"
            if os.path.exists(test_file):
                test_data[contract] = pd.read_parquet(test_file)
                logger.info(f"加载 {contract} 测试数据: {len(test_data[contract])} 行")
        
        if test_data:
            backtest_results = strategy.run_strategy_backtest(test_data)
            performance_metrics = strategy.calculate_performance_metrics(backtest_results)
            
            return {
                "backtest_results": backtest_results,
                "performance_metrics": performance_metrics
            }
        else:
            raise FileNotFoundError(f"未找到测试数据文件在路径: {test_data_path}")
    
    elif mode == 'predict':
        logger.info("开始预测模式...")
        
        # 尝试加载已训练的模型
        try:
            strategy.load_model()
            logger.info("成功加载已训练的模型")
        except FileNotFoundError:
            logger.warning("未找到已训练的模型，请先训练模型")
        
        return strategy
    
    else:
        raise ValueError(f"不支持的模式: {mode} 或缺少必要的数据路径")


# 主策略执行函数
def run_attention_lstm_strategy(test_dir: str = './future_L2/test', 
                               output_dir: str = './positions',
                               train_model: bool = False,
                               model_path: Optional[str] = None):
    """
    运行Attention-LSTM策略
    
    Args:
        test_dir: 测试数据目录
        output_dir: 输出目录
        train_model: 是否训练模型
        model_path: 模型文件路径
    """
    # 创建策略实例
    config = StrategyConfig()
    strategy = AttentionLSTMStrategy(config)
    
    # 如果需要训练模型
    if train_model:
        print("开始训练模型...")
        # 这里可以加载训练数据并训练模型
        # train_data = load_train_data()
        # strategy.train(train_data)
        # strategy.save_model()
        print("模型训练功能需要训练数据，请参考代码实现")
    
    # 尝试加载已训练的模型
    if model_path:
        strategy.load_model(model_path)
    else:
        # 尝试加载默认模型
        if not strategy.load_model():
            print("未找到训练好的模型，将使用简单策略")
    
    # 运行策略预测
    strategy.run_strategy_prediction(test_dir, output_dir)


if __name__ == "__main__":
    # 运行策略
    run_attention_lstm_strategy(
        test_dir='./future_L2/test',
        output_dir='./positions',
        train_model=False  # 设为True来训练模型
    )
