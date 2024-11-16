import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats

class BacktestAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.report_template = self._load_report_template()
        
    def _load_report_template(self) -> str:
        """加载报告模板"""
        try:
            with open('optimization_report_template.md', 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"加载报告模板失败: {str(e)}")
            return ""

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """计算性能指标"""
        trades = pd.DataFrame(results)
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': len(trades[trades['return'] > 0]) / len(trades),
            'avg_return': trades['return'].mean(),
            'max_drawdown': self._calculate_max_drawdown(trades),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades),
            'sortino_ratio': self._calculate_sortino_ratio(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'avg_win': trades[trades['return'] > 0]['return'].mean(),
            'avg_loss': trades[trades['return'] < 0]['return'].mean(),
            'max_win': trades['return'].max(),
            'max_loss': trades['return'].min(),
            'win_loss_ratio': abs(trades[trades['return'] > 0]['return'].mean() / 
                                trades[trades['return'] < 0]['return'].mean()),
            'avg_holding_time': trades['holding_time'].mean()
        }
        
        return metrics

    def _calculate_max_drawdown(self, trades: pd.DataFrame) -> float:
        """计算最大回撤"""
        cumulative = (1 + trades['return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_sharpe_ratio(self, trades: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        returns = trades['return']
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(self, trades: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        returns = trades['return']
        excess_returns = returns - risk_free_rate/252
        downside_std = returns[returns < 0].std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0

    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """计算盈亏比"""
        profits = trades[trades['return'] > 0]['return'].sum()
        losses = abs(trades[trades['return'] < 0]['return'].sum())
        return profits / losses if losses != 0 else float('inf')

    def analyze_parameter_sensitivity(self, results: List[Dict]) -> Dict:
        """分析参数敏感度"""
        df = pd.DataFrame(results)
        sensitivity = {}
        
        for param in ['rsi_lower', 'rsi_upper', 'macd_threshold', 'volume_multiplier']:
            if param in df.columns:
                correlation = stats.spearmanr(df[param], df['return'])
                sensitivity[param] = {
                    'correlation': correlation.correlation,
                    'p_value': correlation.pvalue
                }
        
        return sensitivity

    def generate_report(self, optimization_results: Dict, metrics: Dict, sensitivity: Dict) -> str:
        """生成优化报告"""
        report = self.report_template
        
        # 填充交易统计
        report = report.replace("总交易次数：", f"总交易次数：{metrics['total_trades']}")
        report = report.replace("胜率：", f"胜率：{metrics['win_rate']:.2%}")
        report = report.replace("平均收益：", f"平均收益：{metrics['avg_return']:.2%}")
        report = report.replace("最大回撤：", f"最大回撤：{metrics['max_drawdown']:.2%}")
        report = report.replace("夏普比率：", f"夏普比率：{metrics['sharpe_ratio']:.2f}")
        
        # 填充最优参数
        best_params = optimization_results['best_params']
        report = report.replace("RSI\n   - 下限：", f"RSI\n   - 下限：{best_params['rsi_lower']}")
        report = report.replace("上限：", f"上限：{best_params['rsi_upper']}")
        report = report.replace("MACD\n   - 阈值：", f"MACD\n   - 阈值：{best_params['macd_threshold']}")
        
        # 添加参数敏感度分析
        sensitivity_text = "\n### 参数敏感度分析\n\n"
        for param, stats in sensitivity.items():
            sensitivity_text += f"- {param}:\n"
            sensitivity_text += f"  * 相关系数: {stats['correlation']:.3f}\n"
            sensitivity_text += f"  * 显著性: {stats['p_value']:.3f}\n"
        
        report = report.replace("### 6. 补充说明", sensitivity_text + "\n### 6. 补充说明")
        
        return report

    def plot_performance(self, results: List[Dict], save_path: str):
        """绘制性能分析图表"""
        trades = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 收益分布
        sns.histplot(trades['return'], kde=True, ax=axes[0,0])
        axes[0,0].set_title('收益分布')
        
        # 累积收益
        cumulative = (1 + trades['return']).cumprod()
        cumulative.plot(ax=axes[0,1])
        axes[0,1].set_title('累积收益')
        
        # 参数相关性热图
        param_cols = ['rsi_lower', 'rsi_upper', 'macd_threshold', 'volume_multiplier']
        correlation = trades[param_cols + ['return']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=axes[1,0])
        axes[1,0].set_title('参数相关性')
        
        # 胜率随时间变化
        trades['win'] = trades['return'] > 0
        win_rate = trades['win'].rolling(20).mean()
        win_rate.plot(ax=axes[1,1])
        axes[1,1].set_title('胜率变化(20期移动平均)')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    # 示例使用
    analyzer = BacktestAnalyzer('optimization_results')
    
    # 加载优化结果
    with open('optimization_results/latest_optimization.json', 'r') as f:
        optimization_results = json.load(f)
    
    # 计算指标
    metrics = analyzer.calculate_metrics(optimization_results['trades'])
    
    # 分析参数敏感度
    sensitivity = analyzer.analyze_parameter_sensitivity(optimization_results['trades'])
    
    # 生成报告
    report = analyzer.generate_report(optimization_results, metrics, sensitivity)
    
    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'optimization_results/optimization_report_{timestamp}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 绘制图表
    plot_path = f'optimization_results/performance_analysis_{timestamp}.png'
    analyzer.plot_performance(optimization_results['trades'], plot_path)
    
    print(f"分析报告已保存到: {report_path}")
    print(f"性能分析图表已保存到: {plot_path}")

if __name__ == "__main__":
    main()
