import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
import sys

class DataCollector:
    def __init__(self):
        print("初始化数据收集器...")
        try:
            self.exchange = ccxt.binance({
                'timeout': 30000,
                'enableRateLimit': True,
                'proxies': {
                    'http': 'http://127.0.0.1:7890',
                    'https': 'http://127.0.0.1:7890',
                }
            })
            
            # 测试连接
            print("测试交易所连接...")
            self.exchange.load_markets()
            print("连接成功")
            
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            sys.exit(1)
        
        self.data_dir = 'historical_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")

    def get_top_symbols(self, quote: str = 'USDT', 
                       min_volume: float = 1000000, 
                       limit: int = 10) -> List[str]:
        """获取交易量最大的交易对"""
        try:
            print("\n获取市场数据...")
            tickers = self.exchange.fetch_tickers()
            
            # 筛选和排序
            valid_tickers = []
            for symbol, ticker in tickers.items():
                if symbol.endswith(f'/{quote}') and \
                   ticker.get('quoteVolume', 0) > min_volume:
                    valid_tickers.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume']
                    })
            
            # 按交易量排序
            valid_tickers.sort(key=lambda x: x['volume'], reverse=True)
            
            # 取前N个
            top_symbols = [t['symbol'] for t in valid_tickers[:limit]]
            
            print(f"找到 {len(top_symbols)} 个符合条件的交易对")
            return top_symbols
            
        except Exception as e:
            print(f"获取交易对列表失败: {str(e)}")
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', 
                   days: int = 90) -> Optional[pd.DataFrame]:
        """获取历史K线数据"""
        try:
            print(f"\n获取 {symbol} 的历史数据...")
            
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
            # 检查缓存
            filename = f"{symbol.replace('/', '_')}_{timeframe}_{days}d.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                # 如果文件不超过1小时，直接使用
                if (datetime.now() - file_time).total_seconds() < 3600:
                    print(f"使用缓存数据: {filepath}")
                    return pd.read_csv(filepath)
            
            # 获取新数据
            print(f"从交易所获取数据...")
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )
            
            if not ohlcv:
                print(f"未获取到 {symbol} 的数据")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                            'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 保存数据
            df.to_csv(filepath, index=False)
            print(f"数据已保存到: {filepath}")
            print(f"获取到 {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            print(f"获取 {symbol} 的数据失败: {str(e)}")
            return None

    def collect_data(self, symbols: Optional[List[str]] = None, 
                    timeframe: str = '4h', days: int = 90):
        """收集多个交易对的数据"""
        if symbols is None:
            print("\n获取交易量最大的交易对...")
            symbols = self.get_top_symbols()
        
        if not symbols:
            print("未找到符合条件的交易对")
            return
        
        print(f"\n开始收集 {len(symbols)} 个交易对的数据...")
        
        success_count = 0
        for i, symbol in enumerate(symbols, 1):
            print(f"\n处理进度: {i}/{len(symbols)}")
            df = self.fetch_ohlcv(symbol, timeframe, days)
            
            if df is not None:
                success_count += 1
            
            # 添加延时避免请求过快
            time.sleep(1)
        
        print(f"\n数据收集完成")
        print(f"成功: {success_count}")
        print(f"失败: {len(symbols) - success_count}")

def main():
    try:
        collector = DataCollector()
        collector.collect_data()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
