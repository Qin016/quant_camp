import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


def simple_mean_reversion_strategy(df, window=60, entry_threshold=2.0, exit_threshold=0.5):
    """
    简单的均值回归策略
    
    参数:
    df: 行情数据DataFrame
    window: 移动平均窗口期
    entry_threshold: 入场阈值（标准差倍数）
    exit_threshold: 出场阈值（标准差倍数）
    
    返回:
    position: 仓位序列 (1=多头, -1=空头, 0=空仓)
    """
    data = df.copy()
    
    # 计算移动平均和标准差
    data['price'] = data['LASTPRICE']
    data['ma'] = data['price'].rolling(window=window).mean()
    data['std'] = data['price'].rolling(window=window).std()
    
    # 计算价格偏离
    data['deviation'] = (data['price'] - data['ma']) / data['std']
    
    # 初始化仓位
    positions = np.zeros(len(data))
    current_pos = 0
    
    for i in range(window, len(data)):
        if pd.isna(data['deviation'].iloc[i]):
            continue
            
        dev = data['deviation'].iloc[i]
        
        # 空仓时的入场信号
        if current_pos == 0:
            if dev > entry_threshold:  # 价格过高，做空
                current_pos = -1
            elif dev < -entry_threshold:  # 价格过低，做多
                current_pos = 1
        
        # 持仓时的出场信号
        elif current_pos == 1:  # 持多仓
            if dev > -exit_threshold:  # 回归到均值附近，平仓
                current_pos = 0
        
        elif current_pos == -1:  # 持空仓
            if dev < exit_threshold:  # 回归到均值附近，平仓
                current_pos = 0
        
        positions[i] = current_pos
    
    return pd.Series(positions, index=data.index, name='position')


def momentum_strategy(df, short_window=20, long_window=60, volume_threshold=1.2):
    """
    简单的动量策略
    
    参数:
    df: 行情数据DataFrame
    short_window: 短期均线窗口
    long_window: 长期均线窗口
    volume_threshold: 成交量阈值
    """
    data = df.copy()
    
    # 计算短期和长期移动平均
    data['short_ma'] = data['LASTPRICE'].rolling(window=short_window).mean()
    data['long_ma'] = data['LASTPRICE'].rolling(window=long_window).mean()
    data['volume_ma'] = data['TRADEVOLUME'].rolling(window=long_window).mean()
    
    # 初始化仓位
    positions = np.zeros(len(data))
    current_pos = 0
    
    for i in range(long_window, len(data)):
        if pd.isna(data['short_ma'].iloc[i]) or pd.isna(data['long_ma'].iloc[i]):
            continue
            
        short_ma = data['short_ma'].iloc[i]
        long_ma = data['long_ma'].iloc[i]
        volume = data['TRADEVOLUME'].iloc[i]
        volume_ma = data['volume_ma'].iloc[i]
        
        # 成交量过滤
        volume_filter = volume > volume_threshold * volume_ma if volume_ma > 0 else False
        
        # 空仓时的入场信号
        if current_pos == 0:
            if short_ma > long_ma and volume_filter:  # 上涨趋势，做多
                current_pos = 1
            elif short_ma < long_ma and volume_filter:  # 下跌趋势，做空
                current_pos = -1
        
        # 持仓时的出场信号
        elif current_pos == 1:  # 持多仓
            if short_ma <= long_ma:  # 趋势反转，平仓
                current_pos = 0
        
        elif current_pos == -1:  # 持空仓
            if short_ma >= long_ma:  # 趋势反转，平仓
                current_pos = 0
        
        positions[i] = current_pos
    
    return pd.Series(positions, index=data.index, name='position')


def bollinger_bands_strategy(df, window=40, num_std=2.0):
    """
    布林带策略
    
    参数:
    df: 行情数据DataFrame
    window: 移动平均窗口
    num_std: 标准差倍数
    """
    data = df.copy()
    
    # 计算布林带
    data['ma'] = data['LASTPRICE'].rolling(window=window).mean()
    data['std'] = data['LASTPRICE'].rolling(window=window).std()
    data['upper_band'] = data['ma'] + num_std * data['std']
    data['lower_band'] = data['ma'] - num_std * data['std']
    
    # 初始化仓位
    positions = np.zeros(len(data))
    current_pos = 0
    
    for i in range(window, len(data)):
        if pd.isna(data['upper_band'].iloc[i]) or pd.isna(data['lower_band'].iloc[i]):
            continue
            
        price = data['LASTPRICE'].iloc[i]
        upper = data['upper_band'].iloc[i]
        lower = data['lower_band'].iloc[i]
        ma = data['ma'].iloc[i]
        
        # 空仓时的入场信号
        if current_pos == 0:
            if price > upper:  # 突破上轨，做空（均值回归）
                current_pos = -1
            elif price < lower:  # 突破下轨，做多（均值回归）
                current_pos = 1
        
        # 持仓时的出场信号
        elif current_pos == 1:  # 持多仓
            if price >= ma:  # 回到中轨附近，平仓
                current_pos = 0
        
        elif current_pos == -1:  # 持空仓
            if price <= ma:  # 回到中轨附近，平仓
                current_pos = 0
        
        positions[i] = current_pos
    
    return pd.Series(positions, index=data.index, name='position')


def process_single_contract(df, strategy_type='bollinger'):
    """
    处理单个合约的策略信号生成
    
    参数:
    df: 行情数据DataFrame
    strategy_type: 策略类型 ('mean_reversion', 'momentum', 'bollinger')
    """
    # 确保开盘和收盘时仓位为0
    df = df.copy()
    
    # 生成策略信号
    if strategy_type == 'mean_reversion':
        positions = simple_mean_reversion_strategy(df)
    elif strategy_type == 'momentum':
        positions = momentum_strategy(df)
    elif strategy_type == 'bollinger':
        positions = bollinger_bands_strategy(df)
    else:
        # 默认使用布林带策略
        positions = bollinger_bands_strategy(df)
    
    # 确保开盘和收盘时仓位为0
    if len(positions) > 0:
        positions.iloc[0] = 0   # 开盘时仓位为0
        positions.iloc[-1] = 0  # 收盘时仓位为0
    
    return positions


def generate_daily_signals(date, test_dir='./future_L2/test', strategy_type='bollinger'):
    """
    生成某一天的所有主力合约信号
    
    参数:
    date: 交易日期 (如 '20241009')
    test_dir: 测试数据目录
    strategy_type: 策略类型
    """
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} 开始处理 {date}')
    
    # 获取该日期的所有主力合约文件
    date_dir = f'{test_dir}/{date}'
    if not os.path.exists(date_dir):
        print(f"目录不存在: {date_dir}")
        return {}
    
    files = os.listdir(date_dir)
    main_files = [f for f in files if '_M.parquet' in f]  # 主力合约文件
    
    result_dict = {}
    
    for file in main_files:
        try:
            # 读取数据
            file_path = f'{date_dir}/{file}'
            df = pd.read_parquet(file_path)
            
            # 生成信号
            positions = process_single_contract(df, strategy_type)
            
            # 保存结果（不包含文件扩展名）
            contract_code = file.split('.')[0]
            result_dict[contract_code] = positions
            
            print(f"  处理完成: {contract_code}, 信号数量: {len(positions)}")
            
        except Exception as e:
            print(f"  处理文件 {file} 时出错: {e}")
            continue
    
    return result_dict


def save_signals(date, signals_dict, output_dir='./positions'):
    """
    保存信号到文件
    
    参数:
    date: 交易日期
    signals_dict: 信号字典
    output_dir: 输出目录
    """
    date_output_dir = f'{output_dir}/{date}'
    os.makedirs(date_output_dir, exist_ok=True)
    
    for contract_code, positions in signals_dict.items():
        output_file = f'{date_output_dir}/{contract_code}.csv'
        positions.to_csv(output_file, header=False)
        print(f"  保存信号: {output_file}")


def run_strategy(test_dir='./future_L2/test', output_dir='./positions', strategy_type='bollinger'):
    """
    运行完整策略
    
    参数:
    test_dir: 测试数据目录
    output_dir: 输出目录
    strategy_type: 策略类型
    """
    # 获取所有测试日期（从第2个交易日开始）
    all_dates = sorted(os.listdir(test_dir))
    if len(all_dates) < 2:
        print("测试数据不足，需要至少2个交易日")
        return
    
    test_dates = all_dates[1:]  # 从第2个交易日开始
    print(f"策略类型: {strategy_type}")
    print(f"测试日期范围: {test_dates[0]} 到 {test_dates[-1]} (共{len(test_dates)}个交易日)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个交易日
    for date in test_dates:
        try:
            # 生成信号
            signals = generate_daily_signals(date, test_dir, strategy_type)
            
            # 保存信号
            if signals:
                save_signals(date, signals, output_dir)
                print(f"完成日期: {date}, 处理合约数: {len(signals)}")
            else:
                print(f"日期 {date} 没有生成任何信号")
                
        except Exception as e:
            print(f"处理日期 {date} 时出错: {e}")
            continue
    
    print("策略运行完成!")


if __name__ == '__main__':
    # 运行策略
    # 可选策略类型: 'mean_reversion', 'momentum', 'bollinger'
    run_strategy(
        test_dir='./future_L2/test',
        output_dir='./positions',
        strategy_type='bollinger'  # 使用布林带策略
    )
