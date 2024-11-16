import pandas as pd
import os
from typing import Optional

class DataLoader:
    def __init__(self, data_dir: str = 'historical_data'):
        self.data_dir = data_dir

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载历史数据"""
        try:
            files = [f for f in os.listdir(self.data_dir) 
                    if f.startswith(symbol.replace('/', '_'))]
            
            if not files:
                raise ValueError(f"未找到 {symbol} 的历史数据")
                
            df = pd.read_csv(os.path.join(self.data_dir, files[0]))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')
            
        except Exception as e:
            print(f"加载 {symbol} 的数据失败: {str(e)}")
            return None

    def get_available_symbols(self) -> list:
        """获取所有可用的交易对"""
        try:
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.endswith('.csv')]
            symbols = [f.replace('_4h_90d.csv', '').replace('_', '/') 
                      for f in data_files]
            return symbols
        except Exception as e:
            print(f"获取交易对列表失败: {str(e)}")
            return []
