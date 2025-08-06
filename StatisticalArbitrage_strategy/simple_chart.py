import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_font():
    """配置matplotlib字体，优先使用支持英文的字体"""
    try:
        # 设置为英文字体，避免中文字体问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("Using English font: DejaVu Sans")
        return True
    except Exception as e:
        print(f"Font setup failed: {e}")
        return False

def create_simple_performance_chart():
    """创建简单的策略表现图表"""
    
    # 设置字体
    setup_font()
    
    try:
        # 读取回测结果
        rets_df = pd.read_csv('./backtest/all_rets.csv', index_col=0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quantitative Strategy Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 累计收益率曲线
        if 'mean' in rets_df.columns:
            cumulative_returns = (1 + rets_df['mean']).cumprod()
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Cumulative Returns Curve', fontsize=12)
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
            axes[0, 1].set_ylabel('Total Return Rate', fontsize=10)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(y):
                axes[0, 1].text(i, v + (0.001 if v >= 0 else -0.001), f'{v:.1%}', 
                               ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        
        # 3. 日收益率分布
        if 'mean' in rets_df.columns:
            axes[1, 0].hist(rets_df['mean'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            mean_return = rets_df['mean'].mean()
            axes[1, 0].axvline(mean_return, color='red', linestyle='--', 
                              label=f'Mean: {mean_return:.4f}')
            axes[1, 0].set_title('Daily Returns Distribution', fontsize=12)
            axes[1, 0].set_xlabel('Daily Return Rate', fontsize=10)
            axes[1, 0].set_ylabel('Frequency', fontsize=10)
            axes[1, 0].legend(fontsize=9)
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
            axes[1, 1].set_ylabel('Drawdown Rate', fontsize=10)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            max_dd = drawdown.min()
            axes[1, 1].axhline(y=max_dd, color='darkred', linestyle='--', 
                              label=f'Max Drawdown: {max_dd:.2%}')
            axes[1, 1].legend(fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = './strategy_performance_english.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved successfully: {output_file}")
        
        # 显示统计信息
        if 'mean' in rets_df.columns:
            print("\n" + "="*50)
            print("STRATEGY PERFORMANCE SUMMARY")
            print("="*50)
            mean_ret = rets_df['mean']
            print(f"Total Trading Days: {len(mean_ret)}")
            print(f"Total Return: {(mean_ret + 1).prod() - 1:.2%}")
            print(f"Annual Return: {(mean_ret.mean() + 1)**252 - 1:.2%}")
            print(f"Volatility: {mean_ret.std() * np.sqrt(252):.2%}")
            print(f"Sharpe Ratio: {(mean_ret.mean() * 252) / (mean_ret.std() * np.sqrt(252)):.3f}")
            print(f"Max Drawdown: {drawdown.min():.2%}")
            print(f"Win Rate: {(mean_ret > 0).mean():.2%}")
            print("="*50)
        
        plt.show()
        
    except FileNotFoundError:
        print("Error: backtest/all_rets.csv file not found!")
        print("Please run backtest.py first to generate results.")
    except Exception as e:
        print(f"Error creating chart: {e}")

if __name__ == '__main__':
    create_simple_performance_chart()
