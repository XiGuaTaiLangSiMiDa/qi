import json
import os
from datetime import datetime
from typing import Dict, Optional

from data_loader import DataLoader
from indicators import calculate_indicators, generate_signals
from metrics import calculate_returns, calculate_metrics
from plotter import Plotter

class BacktestAnalyzer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.plotter = Plotter()
        self.results_dir = 'backtest_results'
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def get_default_params(self) -> Dict:
        """获取默认参数"""
        return {
            'rsi': {
                'period': 14,
                'lower': 30,
                'upper': 40
            },
            'stoch_rsi': {
                'period': 14,
                'k': 3,
                'd': 3,
                'lower': 20
            },
            'macd': {
                'fast': 12,
                'slow': 26,
                'signal': 9,
                'threshold': 0
            },
            'bollinger': {
                'period': 20,
                'std': 2.0
            },
            'obv': {
                'period': 20,
                'threshold': 1.5
            }
        }

    def save_results(self, symbol: str, params: Dict, metrics: Dict):
        """保存回测结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results = {
                'symbol': symbol,
                'params': params,
                'metrics': metrics,
                'timestamp': timestamp
            }
            
            filepath = os.path.join(
                self.results_dir, 
                f'results_{symbol.replace("/", "_")}_{timestamp}.json'
            )
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存结果失败: {str(e)}")

    def run_backtest(self, symbol: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """运行回测"""
        if params is None:
            params = self.get_default_params()
        
        try:
            print(f"\n开始分析 {symbol}...")
            
            # 加载数据
            df = self.data_loader.load_data(symbol)
            if df is None:
                return None
            
            # 计算指标
            df = calculate_indicators(df, params)
            if df is None:
                return None
            
            # 生成信号
            signals = generate_signals(df, params)
            if signals is None:
                return None
            
            # 计算收益
            returns = calculate_returns(df, signals)
            if returns is None:
                return None
            
            # 计算指标
            metrics = calculate_metrics(returns)
            if metrics is None:
                return None
            
            # 绘制分析图表
            plot_path = self.plotter.plot_analysis(df, returns, metrics, symbol)
            if plot_path:
                print(f"分析图表已保存到: {plot_path}")
            
            # 保存结果
            self.save_results(symbol, params, metrics)
            
            return metrics
            
        except Exception as e:
            print(f"回测失败: {str(e)}")
            return None

def main():
    try:
        analyzer = BacktestAnalyzer()
        
        # 获取所有可用的交易对
        symbols = analyzer.data_loader.get_available_symbols()
        print(f"找到 {len(symbols)} 个交易对的数据")
        
        # 运行回测
        for symbol in symbols:
            metrics = analyzer.run_backtest(symbol)
            
            if metrics:
                print(f"\n{symbol} 回测结果:")
                print(f"总交易次数: {metrics['total_trades']}")
                print(f"胜率: {metrics['win_rate']:.2%}")
                print(f"平均收益: {metrics['avg_return']:.2%}")
                print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
                print(f"最大回撤: {metrics['max_drawdown']:.2%}")
                print(f"总收益: {metrics['total_return']:.2%}")
                
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
