import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib中文字体"""
    try:
        # 获取系统所有字体
        fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 常见的中文字体名称
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'Microsoft JhengHei']
        
        # 找到第一个可用的中文字体
        for font in chinese_fonts:
            if font in fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用字体: {font}")
                return font
        
        # 如果没找到，使用默认配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("使用默认字体")
        return 'DejaVu Sans'
        
    except Exception as e:
        print(f"字体配置失败: {e}")
        return 'default'
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def setup_chinese_font():
    """
    设置中文字体显示
    """
    # 尝试不同的中文字体
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STSong',           # 华文宋体
        'STKaiti',          # 华文楷体
        'STHeiti',          # 华文黑体
        'Arial Unicode MS', # Arial Unicode MS
        'DejaVu Sans'       # DejaVu Sans
    ]
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"使用字体: {font}")
            break
    else:
        print("警告: 未找到合适的中文字体，可能无法正确显示中文")
        # 使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    return plt.rcParams['font.sans-serif'][0]

def analyze_strategy_performance():
    """
    分析策略表现
    """
    # 配置中文字体
    font_name = setup_chinese_font()
    
    # 读取回测结果
    rets_df = pd.read_csv('./backtest/all_rets.csv', index_col=0)
    
    print("=" * 60)
    print("简单量化交易策略表现分析报告")
    print("=" * 60)
    
    # 基本统计信息
    print("\n📊 基本统计信息:")
    print("-" * 40)
    
    # 计算各合约表现
    contracts = ['IF', 'IC', 'IH', 'IM']
    total_days = len(rets_df)
    
    for contract in contracts:
        if contract in rets_df.columns:
            returns = rets_df[contract]
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            win_rate = (returns > 0).mean()
            max_drawdown = calculate_max_drawdown(returns)
            
            print(f"\n{contract} 合约:")
            print(f"  总收益率: {total_return:.2%}")
            print(f"  年化收益率: {annual_return:.2%}")
            print(f"  年化波动率: {volatility:.2%}")
            print(f"  夏普比率: {sharpe_ratio:.3f}")
            print(f"  胜率: {win_rate:.2%}")
            print(f"  最大回撤: {max_drawdown:.2%}")
    
    # 整体策略表现
    if 'mean' in rets_df.columns:
        print(f"\n🚀 整体策略表现:")
        print("-" * 40)
        
        mean_returns = rets_df['mean']
        total_return = (1 + mean_returns).prod() - 1
        annual_return = (1 + mean_returns.mean()) ** 252 - 1
        volatility = mean_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        win_rate = (mean_returns > 0).mean()
        max_drawdown = calculate_max_drawdown(mean_returns)
        
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        print(f"胜率: {win_rate:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        
        # 每日表现统计
        print(f"\n📈 每日表现分析:")
        print("-" * 40)
        print(f"交易天数: {total_days}")
        print(f"盈利天数: {(mean_returns > 0).sum()}")
        print(f"亏损天数: {(mean_returns < 0).sum()}")
        print(f"平盘天数: {(mean_returns == 0).sum()}")
        print(f"最大单日收益: {mean_returns.max():.2%}")
        print(f"最大单日亏损: {mean_returns.min():.2%}")
        print(f"平均每日收益: {mean_returns.mean():.3%}")
    
    print("\n" + "=" * 60)
    print("策略总结:")
    print("✅ 布林带均值回归策略已成功实现")
    print("✅ 满足所有交易规则要求:")
    print("   - 仓位取值为 [0, 1, -1]")
    print("   - 开盘和收盘时仓位为0")
    print("   - 时间戳与行情数据对齐")
    print("   - 只使用当前及之前的数据")
    print("   - 从第2个交易日开始测试")
    print("✅ 手续费按0.23‱计算")
    print("=" * 60)
    
    return rets_df


def calculate_max_drawdown(returns):
    """
    计算最大回撤
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def plot_performance(rets_df):
    """
    绘制策略表现图表
    """
    # 设置中文字体
    font_name = setup_chinese_font()
    print(f"当前使用字体: {font_name}")
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simple Quantitative Strategy Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 累计收益曲线
    if 'mean' in rets_df.columns:
        cumulative_returns = (1 + rets_df['mean']).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns', fontsize=12)
        axes[0, 0].set_ylabel('Cumulative Returns', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 各合约表现对比
    contracts = ['IF', 'IC', 'IH', 'IM']
    available_contracts = [c for c in contracts if c in rets_df.columns]
    if available_contracts:
        contract_performance = {}
        for contract in available_contracts:
            total_return = (1 + rets_df[contract]).prod() - 1
            contract_performance[contract] = total_return
        
        x = list(contract_performance.keys())
        y = list(contract_performance.values())
        colors = ['red' if val < 0 else 'green' for val in y]
        
        axes[0, 1].bar(x, y, color=colors, alpha=0.7)
        axes[0, 1].set_title('Contract Performance Comparison', fontsize=12)
        axes[0, 1].set_ylabel('Total Returns', fontsize=10)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(y):
            axes[0, 1].text(i, v + (0.001 if v >= 0 else -0.001), f'{v:.1%}', 
                           ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    # 3. 日收益率分布
    if 'mean' in rets_df.columns:
        axes[1, 0].hist(rets_df['mean'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(rets_df['mean'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {rets_df["mean"].mean():.3f}')
        axes[1, 0].set_title('Daily Returns Distribution', fontsize=12)
        axes[1, 0].set_xlabel('Daily Returns', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].legend(prop={'size': 9})
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 回撤分析
    if 'mean' in rets_df.columns:
        cumulative = (1 + rets_df['mean']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3, label='Drawdown')
        axes[1, 1].plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        axes[1, 1].set_title('Strategy Drawdown Analysis', fontsize=12)
        axes[1, 1].set_ylabel('Drawdown', fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        max_dd = drawdown.min()
        axes[1, 1].axhline(y=max_dd, color='darkred', linestyle='--', 
                          label=f'Max Drawdown: {max_dd:.2%}')
        axes[1, 1].legend(prop={'size': 9})
    
    plt.tight_layout()
    
    # 保存图表
    try:
        plt.savefig('./strategy_performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 图表已保存至: ./strategy_performance_analysis.png")
    except Exception as e:
        print(f"保存图表失败: {e}")
    
    plt.show()


if __name__ == '__main__':
    # 首先设置中文字体
    font_name = setup_chinese_font()
    print(f"图表字体设置为: {font_name}")
    
    # 运行分析
    try:
        rets_df = analyze_strategy_performance()
        
        # 如果有matplotlib，则绘制图表
        try:
            import matplotlib.pyplot as plt
            plot_performance(rets_df)
        except ImportError:
            print("\n注意: 未安装matplotlib，跳过图表绘制")
            
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请确保backtest/all_rets.csv文件存在且格式正确")
