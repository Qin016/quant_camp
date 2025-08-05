#!/usr/bin/env python3
"""
测试交易规则实现
验证Attention-LSTM策略的交易规则符合要求
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append('.')

from attention_lstm_strategy import (
    StrategyConfig,
    TradingRules,
    TradingExecutor,
    BacktestEngine,
    AttentionLSTMStrategy
)

def create_mock_market_data(n_ticks: int = 1000) -> pd.DataFrame:
    """创建模拟的市场数据"""
    np.random.seed(42)
    
    # 生成基础价格序列
    base_price = 3000.0
    price_changes = np.random.normal(0, 0.001, n_ticks)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # 创建完整的tick数据
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 09:30:00', periods=n_ticks, freq='1s'),
        'LASTPRICE': prices,
        'HIGHPRICE': prices * (1 + np.random.uniform(0, 0.001, n_ticks)),
        'LOWPRICE': prices * (1 - np.random.uniform(0, 0.001, n_ticks)),
        'TRADEVOLUME': np.random.poisson(100, n_ticks),
        
        # 盘口数据
        'BUYPRICE01': prices * (1 - np.random.uniform(0.0001, 0.0003, n_ticks)),
        'SELLPRICE01': prices * (1 + np.random.uniform(0.0001, 0.0003, n_ticks)),
        'BUYVOLUME01': np.random.poisson(50, n_ticks),
        'SELLVOLUME01': np.random.poisson(50, n_ticks),
    })
    
    return data

def test_trading_rules_config():
    """测试交易规则配置"""
    print("=== 测试交易规则配置 ===")
    
    # 创建交易规则
    trading_rules = TradingRules(
        max_position_per_contract=1,
        commission_rate=0.0023 / 10000,  # 0.23‱
        use_opposite_price=True,
        force_close_at_end=True,
        contracts=['IC', 'IF', 'IH', 'IM']
    )
    
    print(f"最大持仓: {trading_rules.max_position_per_contract}")
    print(f"手续费率: {trading_rules.commission_rate * 10000:.4f}‱")
    print(f"使用对手方价格: {trading_rules.use_opposite_price}")
    print(f"收盘强制平仓: {trading_rules.force_close_at_end}")
    print(f"支持合约: {trading_rules.contracts}")
    
    assert trading_rules.max_position_per_contract == 1, "最大持仓应为1"
    assert abs(trading_rules.commission_rate - 0.0023 / 10000) < 1e-10, "手续费率错误"
    assert trading_rules.force_close_at_end == True, "应启用收盘强制平仓"
    
    print("✓ 交易规则配置测试通过\n")

def test_trading_executor():
    """测试交易执行器"""
    print("=== 测试交易执行器 ===")
    
    trading_rules = TradingRules()
    executor = TradingExecutor(trading_rules)
    
    # 测试初始状态
    print("初始持仓状态:")
    for contract in trading_rules.contracts:
        print(f"  {contract}: {executor.positions[contract]}")
        assert executor.positions[contract] == 0, f"{contract}初始持仓应为0"
    
    # 模拟tick数据
    current_tick = {
        'timestamp': '2024-01-01 09:30:00',
        'LASTPRICE': 3000.0,
        'BUYPRICE01': 2999.5,
        'SELLPRICE01': 3000.5,
        'BUYVOLUME01': 50,
        'SELLVOLUME01': 60
    }
    
    next_tick = {
        'timestamp': '2024-01-01 09:30:01',
        'LASTPRICE': 3001.0,
        'BUYPRICE01': 3000.5,
        'SELLPRICE01': 3001.5,
        'BUYVOLUME01': 45,
        'SELLVOLUME01': 55
    }
    
    # 测试买入信号
    print("\\n测试买入信号...")
    trade_result = executor.execute_signal('IF', 1, current_tick, next_tick)
    print(f"交易结果: {trade_result}")
    
    assert trade_result['action'] == 'buy', "应为买入操作"
    assert trade_result['volume'] == 1, "交易量应为1"
    assert trade_result['price'] == next_tick['SELLPRICE01'], "买入应使用卖一价"
    assert executor.positions['IF'] == 1, "买入后持仓应为1"
    
    # 测试手续费计算
    expected_commission = 1 * trade_result['price'] * trading_rules.commission_rate
    assert abs(trade_result['commission'] - expected_commission) < 1e-8, "手续费计算错误"
    print(f"手续费: {trade_result['commission']:.6f}")
    
    # 测试卖出信号
    print("\\n测试卖出信号...")
    trade_result = executor.execute_signal('IF', -1, current_tick, next_tick)
    print(f"交易结果: {trade_result}")
    
    assert trade_result['action'] == 'sell', "应为卖出操作"
    assert trade_result['volume'] == 2, "从多1到空1，交易量应为2"
    assert executor.positions['IF'] == -1, "卖出后持仓应为-1"
    
    # 测试观望信号
    print("\\n测试观望信号...")
    trade_result = executor.execute_signal('IF', -1, current_tick, next_tick)
    print(f"交易结果: {trade_result}")
    
    assert trade_result['action'] == 'hold', "应为观望操作"
    assert trade_result['volume'] == 0, "观望时交易量应为0"
    assert executor.positions['IF'] == -1, "观望后持仓不变"
    
    print("✓ 交易执行器测试通过\\n")

def test_force_close():
    """测试强制平仓功能"""
    print("=== 测试强制平仓功能 ===")
    
    trading_rules = TradingRules()
    executor = TradingExecutor(trading_rules)
    
    # 设置一些持仓
    executor.positions['IC'] = 1
    executor.positions['IF'] = -1
    executor.positions['IH'] = 0
    executor.positions['IM'] = 1
    
    print("收盘前持仓:")
    for contract, position in executor.positions.items():
        print(f"  {contract}: {position}")
    
    # 模拟收盘tick
    close_tick = {
        'timestamp': '2024-01-01 15:00:00',
        'LASTPRICE': 3000.0,
        'BUYPRICE01': 2999.5,
        'SELLPRICE01': 3000.5
    }
    
    # 执行强制平仓
    close_trades = executor.force_close_all_positions(close_tick)
    
    print(f"\\n强制平仓交易数量: {len(close_trades)}")
    for trade in close_trades:
        print(f"  {trade['contract']}: {trade['action']} {trade['volume']} @ {trade['price']}")
    
    # 验证所有持仓都被平掉
    print("\\n收盘后持仓:")
    for contract, position in executor.positions.items():
        print(f"  {contract}: {position}")
        assert position == 0, f"{contract}收盘后持仓应为0"
    
    # 验证强制平仓的交易数量
    expected_trades = 3  # IC、IF、IM需要平仓，IH已经是0
    assert len(close_trades) == expected_trades, f"应有{expected_trades}笔强制平仓交易"
    
    print("✓ 强制平仓功能测试通过\\n")

def test_position_limits():
    """测试持仓限制"""
    print("=== 测试持仓限制 ===")
    
    trading_rules = TradingRules(max_position_per_contract=1)
    executor = TradingExecutor(trading_rules)
    
    current_tick = {
        'timestamp': '2024-01-01 09:30:00',
        'LASTPRICE': 3000.0,
        'BUYPRICE01': 2999.5,
        'SELLPRICE01': 3000.5
    }
    
    # 测试超过限制的信号
    print("测试持仓限制...")
    
    # 尝试超过最大持仓
    trade_result = executor.execute_signal('IF', 2, current_tick)  # 尝试持仓2
    print(f"信号2的执行结果: 目标持仓={trade_result['position_after']}")
    
    # 应该被限制为最大持仓1
    assert executor.positions['IF'] == 1, "持仓应被限制为1"
    assert trade_result['position_after'] == 1, "目标持仓应被限制为1"
    
    # 尝试超过最小持仓
    trade_result = executor.execute_signal('IC', -2, current_tick)  # 尝试持仓-2
    print(f"信号-2的执行结果: 目标持仓={trade_result['position_after']}")
    
    # 应该被限制为最小持仓-1
    assert executor.positions['IC'] == -1, "持仓应被限制为-1"
    assert trade_result['position_after'] == -1, "目标持仓应被限制为-1"
    
    print("✓ 持仓限制测试通过\\n")

def test_commission_calculation():
    """测试手续费计算"""
    print("=== 测试手续费计算 ===")
    
    trading_rules = TradingRules(commission_rate=0.0023 / 10000)
    executor = TradingExecutor(trading_rules)
    
    # 不同价格的tick数据
    test_cases = [
        {'price': 3000.0, 'volume': 1},
        {'price': 4000.0, 'volume': 1},
        {'price': 2500.0, 'volume': 1},
    ]
    
    print("手续费计算测试:")
    for i, case in enumerate(test_cases):
        tick_data = {
            'timestamp': f'2024-01-01 09:30:0{i}',
            'LASTPRICE': case['price'],
            'SELLPRICE01': case['price'] * 1.0001
        }
        
        trade_result = executor.execute_signal(f'Test{i}', 1, tick_data)
        
        expected_commission = case['volume'] * trade_result['price'] * trading_rules.commission_rate
        actual_commission = trade_result['commission']
        
        print(f"  价格: {case['price']}, 成交价: {trade_result['price']:.4f}")
        print(f"  预期手续费: {expected_commission:.8f}")
        print(f"  实际手续费: {actual_commission:.8f}")
        print(f"  手续费率: {actual_commission/trade_result['price']*10000:.4f}‱")
        
        assert abs(actual_commission - expected_commission) < 1e-10, "手续费计算错误"
        print()
    
    print("✓ 手续费计算测试通过\\n")

def test_opposite_price_execution():
    """测试对手方价格成交"""
    print("=== 测试对手方价格成交 ===")
    
    trading_rules = TradingRules(use_opposite_price=True)
    executor = TradingExecutor(trading_rules)
    
    current_tick = {
        'timestamp': '2024-01-01 09:30:00',
        'LASTPRICE': 3000.0,
        'BUYPRICE01': 2999.0,
        'SELLPRICE01': 3001.0
    }
    
    next_tick = {
        'timestamp': '2024-01-01 09:30:01',
        'LASTPRICE': 3000.5,
        'BUYPRICE01': 2999.5,
        'SELLPRICE01': 3001.5
    }
    
    # 测试买入使用卖一价
    print("测试买入使用下一个tick的卖一价...")
    trade_result = executor.execute_signal('IF', 1, current_tick, next_tick)
    print(f"买入成交价: {trade_result['price']}")
    print(f"下一tick卖一价: {next_tick['SELLPRICE01']}")
    assert trade_result['price'] == next_tick['SELLPRICE01'], "买入应使用下一tick的卖一价"
    
    # 测试卖出使用买一价
    print("\\n测试卖出使用下一个tick的买一价...")
    trade_result = executor.execute_signal('IF', 0, current_tick, next_tick)  # 平仓
    print(f"卖出成交价: {trade_result['price']}")
    print(f"下一tick买一价: {next_tick['BUYPRICE01']}")
    assert trade_result['price'] == next_tick['BUYPRICE01'], "卖出应使用下一tick的买一价"
    
    # 测试没有下一tick时使用当前tick价格
    print("\\n测试没有下一tick时使用当前tick价格...")
    trade_result = executor.execute_signal('IC', 1, current_tick, None)
    print(f"买入成交价: {trade_result['price']}")
    print(f"当前tick卖一价: {current_tick['SELLPRICE01']}")
    assert trade_result['price'] == current_tick['SELLPRICE01'], "没有下一tick时应使用当前tick的卖一价"
    
    print("✓ 对手方价格成交测试通过\\n")

def main():
    """运行所有测试"""
    print("🚀 开始交易规则测试")
    print("="*60)
    
    test_trading_rules_config()
    test_trading_executor()
    test_force_close()
    test_position_limits()
    test_commission_calculation()
    test_opposite_price_execution()
    
    print("="*60)
    print("🎉 所有交易规则测试通过！")
    
    print("\\n📋 交易规则总结:")
    print("1. ✓ 每个合约最大持仓1个单位")
    print("2. ✓ 使用下一个tick的对手方一档价格成交")
    print("3. ✓ 手续费按0.23‱计算")
    print("4. ✓ 收盘时强制平仓所有持仓")
    print("5. ✓ 支持IC、IF、IH、IM四个主力合约")
    print("6. ✓ 严格的持仓限制和风险控制")

if __name__ == "__main__":
    main()
