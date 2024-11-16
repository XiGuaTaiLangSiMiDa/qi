import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import os
from datetime import datetime

class Plotter:
    def __init__(self, results_dir: str = 'backtest_results'):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 设置matplotlib样式
        plt.style.use('default')

    def plot_analysis(self, df: pd.DataFrame, returns: pd.DataFrame,
                     metrics: Dict, symbol: str) -> str:
        """绘制分析图表"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建子图
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 价格和信号
            ax1 = plt.subplot(3, 2, 1)
            df['close'].plot(ax=ax1, label='Price')
            signals = returns[returns['signal'] == 1].index
            ax1.scatter(signals, df.loc[signals, 'close'], 
                       color='green', marker='^', label='Buy Signal')
            ax1.set_title(f'{symbol} Price and Signals')
            ax1.legend()
            
            # 2. 累积收益
            ax2 = plt.subplot(3, 2, 2)
            returns['cumulative_return'].plot(ax=ax2)
            ax2.set_title('Cumulative Returns')
            
            # 3. RSI和Stochastic RSI
            ax3 = plt.subplot(3, 2, 3)
            df['rsi'].plot(ax=ax3, label='RSI')
            df['stoch_k'].plot(ax=ax3, label='Stoch K')
            df['stoch_d'].plot(ax=ax3, label='Stoch D')
            ax3.axhline(y=30, color='r', linestyle='--')
            ax3.axhline(y=70, color='r', linestyle='--')
            ax3.set_title('RSI and Stochastic RSI')
            ax3.legend()
            
            # 4. MACD
            ax4 = plt.subplot(3, 2, 4)
            df['macd'].plot(ax=ax4, label='MACD')
            df['macd_signal'].plot(ax=ax4, label='Signal')
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_title('MACD')
            ax4.legend()
            
            # 5. 收益分布
            ax5 = plt.subplot(3, 2, 5)
            returns[returns['signal'] == 1]['return'].hist(
                ax=ax5, bins=50, density=True)
            ax5.set_title('Return Distribution')
            
            # 6. 性能指标
            ax6 = plt.subplot(3, 2, 6)
            metrics_text = '\n'.join([
                f"总交易次数: {metrics['total_trades']}",
                f"胜率: {metrics['win_rate']:.2%}",
                f"平均收益: {metrics['avg_return']:.2%}",
                f"夏普比率: {metrics['sharpe_ratio']:.2f}",
                f"最大回撤: {metrics['max_drawdown']:.2%}",
                f"总收益: {metrics['total_return']:.2%}"
            ])
            ax6.text(0.1, 0.5, metrics_text, fontsize=12)
            ax6.axis('off')
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join(
                self.results_dir, 
                f'analysis_{symbol.replace("/", "_")}_{timestamp}.png'
            )
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"绘制图表失败: {str(e)}")
            return ""
