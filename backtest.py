import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple
from sklearn.model_selection import ParameterGrid
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm

class IndicatorOptimizer:
    def __init__(self):
        self.exchange = ccxt.binance({
            'timeout': 30000,
            'enableRateLimit': True
        })
        
        # 回测参数
        self.test_period = 90  # 回测天数
        self.min_trades = 30   # 最小交易次数
        self.results_dir = 'optimization_results'
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def fetch_historical_data(self, symbol: str, timeframe: str = '4h', days: int = 90) -> pd.DataFrame:
        """获取历史数据"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            print(f"获取{symbol}历史数据时发生错误: {str(e)}")
            return None

    def calculate_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # Volume indicators
            df['volume_ema'] = ta.trend.ema_indicator(df['volume'], window=20)
            
            # 价格变化率
            df['price_change'] = df['close'].pct_change()
            
            # 波动率
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            return df
        except Exception as e:
            print(f"计算技术指标时发生错误: {str(e)}")
            return None

    def generate_signals(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """生成交易信号"""
        df['signal'] = 0
        
        # RSI信号
        rsi_buy = (df['rsi'] >= params['rsi_lower']) & (df['rsi'] <= params['rsi_upper'])
        
        # MACD信号
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'] < params['macd_threshold'])
        
        # 布林带信号
        bb_buy = df['close'] <= df['bb_lower']
        
        # 成交量信号
        volume_buy = df['volume'] > df['volume_ema'] * params['volume_multiplier']
        
        # 综合信号
        df.loc[rsi_buy & macd_buy & (bb_buy | volume_buy), 'signal'] = 1
        
        return df

    def calculate_returns(self, df: pd.DataFrame, holding_period: int = 24) -> Tuple[float, float, int]:
        """计算回报率"""
        signals = df[df['signal'] == 1].index
        total_trades = len(signals)
        if total_trades < self.min_trades:
            return 0, 0, 0
        
        profits = []
        for signal in signals:
            entry_price = df.loc[signal, 'close']
            if signal + holding_period >= len(df):
                continue
            exit_price = df.loc[signal + holding_period, 'close']
            profit = (exit_price - entry_price) / entry_price
            profits.append(profit)
        
        if not profits:
            return 0, 0, 0
            
        win_rate = len([p for p in profits if p > 0]) / len(profits)
        avg_return = np.mean(profits)
        
        return win_rate, avg_return, total_trades

    def optimize_parameters(self, symbol: str) -> Dict:
        """优化参数"""
        # 参数网格
        param_grid = {
            'rsi_lower': [25, 30, 35],
            'rsi_upper': [40, 45, 50],
            'macd_threshold': [-0.1, -0.05, 0],
            'volume_multiplier': [1.3, 1.5, 1.7]
        }
        
        # 获取历史数据
        df = self.fetch_historical_data(symbol)
        if df is None:
            return None
            
        results = []
        grid = ParameterGrid(param_grid)
        
        print(f"\n优化 {symbol} 的参数...")
        for params in tqdm(grid):
            df_analysis = self.calculate_indicators(df.copy(), params)
            if df_analysis is None:
                continue
                
            df_signals = self.generate_signals(df_analysis, params)
            win_rate, avg_return, total_trades = self.calculate_returns(df_signals)
            
            if total_trades >= self.min_trades:
                results.append({
                    'params': params,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_trades': total_trades,
                    'score': win_rate * avg_return * total_trades
                })
        
        if not results:
            return None
            
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        best_result = results[0]
        
        # 保存优化结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(self.results_dir, f'optimization_{symbol.replace("/", "_")}_{timestamp}.json')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol,
                'best_params': best_result['params'],
                'performance': {
                    'win_rate': best_result['win_rate'],
                    'avg_return': best_result['avg_return'],
                    'total_trades': best_result['total_trades']
                },
                'all_results': results[:10],  # 保存前10个结果
                'optimization_time': timestamp
            }, f, indent=2)
        
        return best_result

    def plot_performance(self, symbol: str, params: Dict):
        """绘制性能分析图表"""
        df = self.fetch_historical_data(symbol)
        if df is None:
            return
            
        df = self.calculate_indicators(df, params)
        df = self.generate_signals(df, params)
        
        plt.figure(figsize=(15, 10))
        
        # 价格和信号
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='Price')
        plt.scatter(df[df['signal'] == 1].index, df[df['signal'] == 1]['close'], 
                   color='green', marker='^', label='Buy Signal')
        plt.title(f'{symbol} Price and Signals')
        plt.legend()
        
        # RSI
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['rsi'], label='RSI')
        plt.axhline(y=params['rsi_lower'], color='r', linestyle='--')
        plt.axhline(y=params['rsi_upper'], color='r', linestyle='--')
        plt.title('RSI')
        plt.legend()
        
        # MACD
        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['macd'], label='MACD')
        plt.plot(df.index, df['macd_signal'], label='Signal')
        plt.axhline(y=params['macd_threshold'], color='r', linestyle='--')
        plt.title('MACD')
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.results_dir, f'performance_{symbol.replace("/", "_")}_{timestamp}.png'))
        plt.close()

def main():
    # 测试用的交易对
    test_symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT',
        'SOL/USDT', 'ADA/USDT', 'XRP/USDT'
    ]
    
    optimizer = IndicatorOptimizer()
    
    print("开始参数优化...")
    
    for symbol in test_symbols:
        print(f"\n分析 {symbol}...")
        best_result = optimizer.optimize_parameters(symbol)
        
        if best_result:
            print(f"\n{symbol} 最优参数:")
            print(f"参数: {best_result['params']}")
            print(f"胜率: {best_result['win_rate']:.2%}")
            print(f"平均收益: {best_result['avg_return']:.2%}")
            print(f"交易次数: {best_result['total_trades']}")
            
            # 绘制性能分析图表
            optimizer.plot_performance(symbol, best_result['params'])
        else:
            print(f"{symbol} 优化失败")

if __name__ == "__main__":
    main()
