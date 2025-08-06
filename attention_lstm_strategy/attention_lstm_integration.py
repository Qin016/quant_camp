#!/usr/bin/env python3
"""
Attention-LSTM策略集成模块
==========================

将Attention-LSTM深度学习策略集成到现有的交易框架中
提供与strategy.py兼容的接口

主要功能:
- 与现有框架无缝集成
- 实时信号生成
- 批量处理支持
- 性能优化

使用方法:
from attention_lstm_integration import AttentionLSTMIntegration

# 初始化策略
lstm_strategy = AttentionLSTMIntegration()

# 生成信号
positions = lstm_strategy.generate_signals_for_day(df, contract='IF')
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import time

# 导入核心策略模块
try:
    from attention_lstm_strategy import (
        AttentionLSTMStrategy, 
        RealTimeTrader,
        StrategyConfig,
        run_attention_lstm_strategy
    )
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入Attention-LSTM模块: {e}")
    print("将使用传统策略作为后备方案")
    LSTM_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLSTMIntegration:
    """Attention-LSTM策略集成类"""
    
    def __init__(self, 
                 model_path: str = "./models",
                 use_fallback: bool = True,
                 contracts: List[str] = None):
        """
        初始化LSTM策略集成
        
        Args:
            model_path: 模型文件路径
            use_fallback: 是否使用后备策略
            contracts: 支持的合约列表
        """
        self.model_path = model_path
        self.use_fallback = use_fallback
        self.contracts = contracts or ['IC', 'IF', 'IH', 'IM']
        
        # 策略状态
        self.lstm_strategy = None
        self.real_time_trader = None
        self.is_initialized = False
        self.fallback_mode = False
        
        # 性能缓存
        self.feature_cache = {}
        self.signal_cache = {}
        
        # 初始化策略
        self._initialize_strategy()
    
    def _initialize_strategy(self):
        """初始化策略"""
        if not LSTM_AVAILABLE:
            logger.warning("LSTM模块不可用，启用后备模式")
            self.fallback_mode = True
            return
        
        try:
            # 检查模型文件是否存在
            model_files_exist = all(
                os.path.exists(f"{self.model_path}/{contract}_best_model.pth") 
                for contract in self.contracts
            )
            
            if not model_files_exist:
                logger.warning("模型文件不完整，尝试训练新模型...")
                self._train_models_if_needed()
            
            # 加载策略
            self.lstm_strategy = run_attention_lstm_strategy(mode='predict')
            self.real_time_trader = RealTimeTrader(self.lstm_strategy)
            self.is_initialized = True
            
            logger.info("Attention-LSTM策略初始化成功")
            
        except Exception as e:
            logger.error(f"LSTM策略初始化失败: {e}")
            if self.use_fallback:
                logger.info("启用后备策略模式")
                self.fallback_mode = True
            else:
                raise
    
    def _train_models_if_needed(self):
        """如果需要则训练模型"""
        train_data_dir = "./future_L2/train"
        
        if not os.path.exists(train_data_dir):
            logger.warning("训练数据目录不存在，无法训练模型")
            return
        
        logger.info("开始训练Attention-LSTM模型...")
        try:
            results = run_attention_lstm_strategy(
                train_data_path=train_data_dir,
                mode='train',
                contracts=self.contracts
            )
            logger.info(f"模型训练完成: {results}")
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _fallback_strategy(self, df: pd.DataFrame, contract: str = 'IF') -> pd.Series:
        """后备策略：使用简化的技术指标策略"""
        logger.info(f"使用后备策略处理 {contract}")
        
        if len(df) < 100:
            return pd.Series(0, index=df.index)
        
        # 简化的技术指标策略
        try:
            # 价格特征
            close = df['LASTPRICE']
            volume = df.get('TRADEVOLUME', pd.Series(index=df.index, dtype=float))
            
            # 移动平均
            ma_short = close.rolling(20).mean()
            ma_long = close.rolling(50).mean()
            
            # 价格动量
            momentum = close.pct_change(10)
            
            # 成交量过滤
            vol_ma = volume.rolling(20).mean()
            vol_filter = volume > vol_ma * 1.5
            
            # 生成信号
            signals = pd.Series(0, index=df.index)
            
            # 买入条件
            buy_condition = (
                (close > ma_short) & 
                (ma_short > ma_long) & 
                (momentum > 0.001) & 
                vol_filter
            )
            
            # 卖出条件
            sell_condition = (
                (close < ma_short) & 
                (ma_short < ma_long) & 
                (momentum < -0.001) & 
                vol_filter
            )
            
            signals[buy_condition] = 1
            signals[sell_condition] = -1
            
            return signals
            
        except Exception as e:
            logger.error(f"后备策略执行失败: {e}")
            return pd.Series(0, index=df.index)
    
    def generate_signals_for_day(self, 
                                df: pd.DataFrame, 
                                contract: str = 'IF') -> pd.Series:
        """
        为单个交易日生成信号
        
        Args:
            df: 交易数据
            contract: 合约代码
            
        Returns:
            pd.Series: 仓位信号序列
        """
        if self.fallback_mode or not self.is_initialized:
            return self._fallback_strategy(df, contract)
        
        try:
            # 检查数据质量
            if len(df) < self.lstm_strategy.config.sequence_length:
                logger.warning(f"数据长度不足: {len(df)} < {self.lstm_strategy.config.sequence_length}")
                return self._fallback_strategy(df, contract)
            
            # 生成信号序列
            positions = []
            
            # 批量处理以提高效率
            batch_size = max(100, len(df) // 10)
            
            for i in range(len(df)):
                if i < self.lstm_strategy.config.sequence_length:
                    positions.append(0)
                else:
                    # 获取当前窗口数据
                    window_data = df.iloc[:i+1]
                    
                    # 生成信号
                    signal = self.lstm_strategy.predict_signal(contract, window_data)
                    positions.append(signal)
                
                # 进度提示
                if i % batch_size == 0:
                    logger.debug(f"处理进度: {i}/{len(df)} ({i/len(df)*100:.1f}%)")
            
            result = pd.Series(positions, index=df.index)
            logger.info(f"{contract} 信号生成完成: 长仓={sum(result==1)}, 短仓={sum(result==-1)}, 观望={sum(result==0)}")
            
            return result
            
        except Exception as e:
            logger.error(f"LSTM信号生成失败 {contract}: {e}")
            return self._fallback_strategy(df, contract)
    
    def generate_realtime_signal(self, 
                                df: pd.DataFrame, 
                                contract: str = 'IF') -> int:
        """
        生成实时交易信号
        
        Args:
            df: 最新的市场数据
            contract: 合约代码
            
        Returns:
            int: 交易信号 (1=买入, 0=观望, -1=卖出)
        """
        if self.fallback_mode or not self.is_initialized:
            # 使用后备策略的最后一个信号
            signals = self._fallback_strategy(df, contract)
            return int(signals.iloc[-1]) if len(signals) > 0 else 0
        
        try:
            # 更新实时交易器的数据
            if len(df) > 0:
                latest_tick = df.iloc[-1].to_dict()
                self.real_time_trader.update_market_data(contract, latest_tick)
            
            # 获取信号
            signal = self.real_time_trader.get_current_signal(contract)
            return signal
            
        except Exception as e:
            logger.error(f"实时信号生成失败 {contract}: {e}")
            return 0
    
    def batch_process_contracts(self, 
                               data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        批量处理多个合约
        
        Args:
            data_dict: 合约数据字典 {contract: dataframe}
            
        Returns:
            Dict[str, pd.Series]: 信号字典 {contract: signals}
        """
        results = {}
        
        for contract, df in data_dict.items():
            logger.info(f"处理合约 {contract}...")
            start_time = time.time()
            
            signals = self.generate_signals_for_day(df, contract)
            results[contract] = signals
            
            elapsed = time.time() - start_time
            logger.info(f"{contract} 处理完成，耗时: {elapsed:.2f}秒")
        
        return results
    
    def get_strategy_info(self) -> Dict:
        """获取策略信息"""
        info = {
            'strategy_name': 'Attention-LSTM',
            'version': '1.0',
            'fallback_mode': self.fallback_mode,
            'is_initialized': self.is_initialized,
            'supported_contracts': self.contracts,
            'model_path': self.model_path
        }
        
        if self.is_initialized and not self.fallback_mode:
            config = self.lstm_strategy.config
            info.update({
                'sequence_length': config.sequence_length,
                'hidden_size': config.hidden_size,
                'attention_size': config.attention_size,
                'signal_threshold': config.signal_threshold
            })
        
        return info
    
    def validate_models(self) -> Dict[str, bool]:
        """验证模型文件完整性"""
        model_status = {}
        
        for contract in self.contracts:
            model_file = f"{self.model_path}/{contract}_best_model.pth"
            scaler_file = f"./scalers/{contract}_feature_engineer.pkl"
            
            model_status[contract] = (
                os.path.exists(model_file) and 
                os.path.exists(scaler_file)
            )
        
        return model_status


# 便捷函数：直接替换原始策略函数
def generate_signals_lstm(df: pd.DataFrame, 
                         contract: str = 'IF',
                         **kwargs) -> pd.Series:
    """
    LSTM策略信号生成函数
    
    可以直接替换原有的generate_signals函数
    
    Args:
        df: 市场数据
        contract: 合约代码
        **kwargs: 其他参数（兼容性）
        
    Returns:
        pd.Series: 仓位信号
    """
    # 使用全局策略实例（如果存在）
    global _global_lstm_strategy
    
    if '_global_lstm_strategy' not in globals():
        _global_lstm_strategy = AttentionLSTMIntegration()
    
    return _global_lstm_strategy.generate_signals_for_day(df, contract)


def generate_signals_optimized_lstm(df: pd.DataFrame,
                                   contract: str = 'IF',
                                   **kwargs) -> pd.Series:
    """
    优化版LSTM策略函数
    
    兼容原有generate_signals_optimized函数接口
    
    Args:
        df: 市场数据
        contract: 合约代码
        **kwargs: 其他参数（向后兼容）
        
    Returns:
        pd.Series: 仓位信号
    """
    return generate_signals_lstm(df, contract, **kwargs)


# 测试和演示函数
def test_lstm_integration():
    """测试LSTM集成模块"""
    print("=== Attention-LSTM 集成测试 ===")
    
    # 初始化策略
    strategy = AttentionLSTMIntegration()
    
    # 显示策略信息
    info = strategy.get_strategy_info()
    print("策略信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 验证模型
    model_status = strategy.validate_models()
    print("\n模型状态:")
    for contract, status in model_status.items():
        print(f"  {contract}: {'✓' if status else '✗'}")
    
    # 生成模拟数据进行测试
    print("\n生成测试数据...")
    np.random.seed(42)
    test_data = pd.DataFrame({
        'LASTPRICE': 3000 + np.cumsum(np.random.randn(1000) * 0.1),
        'HIGHPRICE': lambda x: x['LASTPRICE'] + np.random.rand(1000) * 2,
        'LOWPRICE': lambda x: x['LASTPRICE'] - np.random.rand(1000) * 2,
        'TRADEVOLUME': np.random.randint(50, 200, 1000),
        'BUYVOLUME01': np.random.randint(20, 100, 1000),
        'SELLVOLUME01': np.random.randint(20, 100, 1000)
    })
    
    # 修正高低价格
    test_data['HIGHPRICE'] = test_data['LASTPRICE'] + np.random.rand(1000) * 2
    test_data['LOWPRICE'] = test_data['LASTPRICE'] - np.random.rand(1000) * 2
    
    print(f"测试数据形状: {test_data.shape}")
    
    # 测试信号生成
    print("\n测试信号生成...")
    start_time = time.time()
    
    signals = strategy.generate_signals_for_day(test_data, 'IF')
    
    elapsed = time.time() - start_time
    print(f"信号生成完成，耗时: {elapsed:.2f}秒")
    
    # 分析信号
    signal_counts = signals.value_counts().sort_index()
    print("\n信号分布:")
    signal_names = {-1: '卖出', 0: '观望', 1: '买入'}
    for signal, count in signal_counts.items():
        name = signal_names.get(signal, f'未知({signal})')
        print(f"  {name}: {count} ({count/len(signals)*100:.1f}%)")
    
    # 测试实时信号
    print("\n测试实时信号...")
    for i in range(5):
        real_time_signal = strategy.generate_realtime_signal(test_data.iloc[:500+i*100], 'IF')
        print(f"  实时信号 {i+1}: {real_time_signal}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_lstm_integration()
