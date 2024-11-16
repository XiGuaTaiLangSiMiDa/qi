import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import time
import json
import os
import sys
from typing import List, Dict, Tuple

class CryptoAnalyzer:
    def __init__(self):
        self.config = self.load_config()
        self.exchange = ccxt.binance({
            'timeout': self.config['api']['timeout'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': self.config['api']['recv_window']
            }
        })
        
        if self.config['proxy']['enabled']:
            self.exchange.proxies = {
                'http': self.config['proxy']['http'],
                'https': self.config['proxy']['https']
            }
            print(f"使用代理: {self.config['proxy']['http']}")
        
        self.indicators_config = self.config['analysis']['indicators']
        self.cache_dir = self.config['paths']['cache_dir']
        self.results_dir = self.config['paths']['results_dir']
        self.min_volume = self.config['analysis']['min_volume']
        self.timeframe = self.config['analysis']['timeframe']
        self.lookback_periods = self.config['analysis']['lookback_periods']
        
        for directory in [self.cache_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_config(self) -> Dict:
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def get_cache_path(self, symbol: str) -> str:
        """获取缓存文件路径"""
        safe_symbol = symbol.replace('/', '_')
        return os.path.join(self.cache_dir, f"{safe_symbol}.json")

    def load_cache(self, symbol: str) -> Dict:
        """加载缓存数据"""
        cache_path = self.get_cache_path(symbol)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                if time.time() - cache_data['timestamp'] < self.config['analysis']['cache_expiry']:
                    return cache_data
            except Exception as e:
                print(f"读取缓存文件出错: {str(e)}", file=sys.stderr)
        return None

    def save_cache(self, symbol: str, data: Dict):
        """保存数据到缓存"""
        try:
            cache_path = self.get_cache_path(symbol)
            data['timestamp'] = time.time()
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"保存缓存文件出错: {str(e)}", file=sys.stderr)

    def fetch_all_tickers(self) -> List[Dict]:
        """获取所有交易对的行情数据"""
        cache_path = os.path.join(self.cache_dir, 'tickers.json')
        
        # 尝试从缓存加载
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                if time.time() - cache_data['timestamp'] < 300:  # 5分钟缓存
                    print("使用缓存的市场数据")
                    return cache_data['tickers']
            except Exception as e:
                print(f"读取市场数据缓存出错: {str(e)}", file=sys.stderr)

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"正在获取市场数据... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(1)
                tickers = self.exchange.fetch_tickers()
                print(f"成功获取 {len(tickers)} 个交易对数据")
                
                valid_tickers = []
                for symbol, ticker in tickers.items():
                    if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > self.min_volume:
                        valid_tickers.append({
                            'symbol': symbol,
                            'last': float(ticker['last']),
                            'volume': float(ticker['quoteVolume']),
                            'change': float(ticker.get('percentage', 0))
                        })
                
                # 保存到缓存
                cache_data = {
                    'tickers': valid_tickers,
                    'timestamp': time.time()
                }
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f)
                
                print(f"其中符合交易量条件的有 {len(valid_tickers)} 个交易对")
                return valid_tickers
            except Exception as e:
                print(f"获取市场数据时发生错误: {str(e)}", file=sys.stderr)
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("达到最大重试次数，放弃获取数据")
                    return []

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """获取K线数据"""
        # 尝试从缓存加载
        cache_data = self.load_cache(symbol)
        if cache_data and 'ohlcv' in cache_data:
            print(f"使用缓存的 {symbol} K线数据")
            df = pd.DataFrame(cache_data['ohlcv'])
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"获取 {symbol} 的K线数据... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(0.5)
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=self.timeframe,
                    limit=100
                )
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 确保数值列为float类型
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_columns] = df[numeric_columns].astype(float)
                
                # 保存到缓存
                cache_data = {
                    'ohlcv': df.to_dict(orient='records'),
                    'timestamp': time.time()
                }
                self.save_cache(symbol, cache_data)
                
                return df
            except Exception as e:
                print(f"获取{symbol}的K线数据时发生错误: {str(e)}", file=sys.stderr)
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("达到最大重试次数，放弃获取数据")
                    return None

    def calculate_stoch_rsi(self, close_prices: pd.Series, config: Dict) -> pd.Series:
        """计算Stochastic RSI"""
        # 计算RSI
        rsi = ta.momentum.rsi(close_prices, window=config['rsi_period'])
        
        # 计算RSI的随机指标
        stoch_rsi = pd.Series(index=close_prices.index)
        for i in range(config['stoch_period'], len(rsi)):
            window = rsi[i-config['stoch_period']:i]
            if not window.empty:
                min_rsi = window.min()
                max_rsi = window.max()
                if max_rsi - min_rsi != 0:
                    stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
                else:
                    stoch_rsi[i] = 50  # 如果最大最小值相同，设为中值
        
        # 计算K线和D线
        k = stoch_rsi.rolling(window=config['k_period']).mean()
        d = k.rolling(window=config['d_period']).mean()
        
        return k, d

    def calculate_obv(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """计算On Balance Volume (OBV)"""
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])
        obv_ema = ta.trend.ema_indicator(obv, window=config['ema_period'])
        return obv, obv_ema

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], 
                                      window=self.indicators_config['rsi']['period'])
            
            # Stochastic RSI
            df['stoch_k'], df['stoch_d'] = self.calculate_stoch_rsi(
                df['close'], 
                self.indicators_config['stoch_rsi']
            )
            
            # MACD
            macd = ta.trend.MACD(
                df['close'],
                window_fast=self.indicators_config['macd']['fast_period'],
                window_slow=self.indicators_config['macd']['slow_period'],
                window_sign=self.indicators_config['macd']['signal_period']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df['close'],
                window=self.indicators_config['bollinger']['period'],
                window_dev=self.indicators_config['bollinger']['std_dev']
            )
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # OBV
            df['obv'], df['obv_ema'] = self.calculate_obv(
                df, 
                self.indicators_config['obv']
            )
            
            # Volume EMA
            df['volume_ema'] = ta.trend.ema_indicator(
                df['volume'], 
                window=self.indicators_config['volume']['period']
            )
            
            return df
        except Exception as e:
            print(f"计算技术指标时发生错误: {str(e)}", file=sys.stderr)
            return None

    def analyze_potential_breakout(self, df: pd.DataFrame) -> Tuple[float, float, str, int, str]:
        """分析潜在突破可能性"""
        try:
            if len(df) < 20:
                return 0, 0, "数据不足", 0, "无法估算"

            latest = df.iloc[-1]
            signals = []
            probabilities = []
            
            # RSI信号
            rsi_config = self.indicators_config['rsi']
            if rsi_config['lower'] <= latest['rsi'] <= rsi_config['upper']:
                signals.append("RSI超卖区间")
                probabilities.append(rsi_config['weight'])
            
            # Stochastic RSI信号
            stoch_config = self.indicators_config['stoch_rsi']
            if (latest['stoch_k'] <= stoch_config['lower'] and 
                latest['stoch_k'] > latest['stoch_d']):
                signals.append("Stoch RSI超卖区域金叉")
                probabilities.append(stoch_config['weight'])
            
            # MACD信号
            macd_config = self.indicators_config['macd']
            if (latest['macd'] > latest['macd_signal'] and 
                latest['macd'] < macd_config['threshold']):
                signals.append("MACD金叉即将形成")
                probabilities.append(macd_config['weight'])
            
            # 布林带信号
            if latest['close'] <= latest['bb_lower']:
                signals.append("价格触及布林带下轨")
                probabilities.append(self.indicators_config['bollinger']['weight'])
            
            # OBV信号
            obv_config = self.indicators_config['obv']
            if latest['obv'] > latest['obv_ema'] * obv_config['threshold']:
                signals.append("OBV放量突破")
                probabilities.append(obv_config['weight'])
            
            # 成交量信号
            volume_config = self.indicators_config['volume']
            if latest['volume'] > latest['volume_ema'] * volume_config['multiplier']:
                signals.append("成交量放大")
                probabilities.append(volume_config['weight'])
            
            if not probabilities:
                return 0, 0, "无明显信号", 0, "无法估算"
            
            probability = np.mean(probabilities)
            current_price = latest['close']
            potential_target = current_price * (1 + probability * 0.2)
            
            # 估算达到目标价格的时间
            estimated_hours, time_desc = self.estimate_target_time(df, current_price, potential_target)
            
            return probability, potential_target, ", ".join(signals), estimated_hours, time_desc
        except Exception as e:
            print(f"分析突破可能性时发生错误: {str(e)}", file=sys.stderr)
            return 0, 0, f"分析错误: {str(e)}", 0, "无法估算"

    def estimate_target_time(self, df: pd.DataFrame, current_price: float, target_price: float) -> Tuple[int, str]:
        """估算达到目标价格的时间"""
        try:
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            recent_price_change = df['price_change'].tail(20).mean()
            recent_volatility = df['volatility'].tail(20).mean()
            
            required_change = (target_price / current_price) - 1
            
            if recent_price_change > 0:
                estimated_periods = abs(required_change) / (recent_price_change + recent_volatility)
            else:
                estimated_periods = abs(required_change) / recent_volatility

            estimated_hours = int(estimated_periods * 4)
            
            if estimated_hours < 24:
                time_desc = f"{estimated_hours}小时"
            else:
                estimated_days = estimated_hours / 24
                if estimated_days < 7:
                    time_desc = f"{estimated_days:.1f}天"
                else:
                    time_desc = f"{estimated_days/7:.1f}周"
            
            return estimated_hours, time_desc
            
        except Exception as e:
            print(f"估算目标时间时发生错误: {str(e)}", file=sys.stderr)
            return 0, "无法估算"

    def save_scan_results(self, results: List[Dict]):
        """保存扫描结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scan_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            data = {
                'scan_time': datetime.now().isoformat(),
                'results': results,
                'total_opportunities': len(results)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n分析结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存分析结果时发生错误: {str(e)}", file=sys.stderr)

    def scan_market(self) -> List[Dict]:
        """扫描市场寻找潜在机会"""
        try:
            results = []
            tickers = self.fetch_all_tickers()
            
            if not tickers:
                print("未获取到市场数据")
                return results

            print(f"开始分析 {len(tickers)} 个交易对...")
            
            for ticker in tickers:
                symbol = ticker['symbol']
                df = self.fetch_ohlcv(symbol)
                
                if df is None or len(df) < self.lookback_periods:
                    continue
                    
                df = self.calculate_indicators(df)
                if df is None:
                    continue

                probability, target_price, signals, estimated_hours, time_desc = self.analyze_potential_breakout(df)
                
                if probability >= 0.6:  # 只返回概率大于60%的结果
                    results.append({
                        'symbol': symbol,
                        'current_price': ticker['last'],
                        'target_price': target_price,
                        'probability': probability,
                        'signals': signals,
                        '24h_volume': ticker['volume'],
                        'estimated_time': time_desc
                    })
            
            # 按概率排序
            results.sort(key=lambda x: x['probability'], reverse=True)
            return results
        except Exception as e:
            print(f"扫描市场时发生错误: {str(e)}", file=sys.stderr)
            return []

def main():
    analyzer = CryptoAnalyzer()
    
    print("开始扫描市场...")
    results = analyzer.scan_market()
    
    if not results:
        print("未找到符合条件的交易机会")
        return

    # 保存分析结果
    analyzer.save_scan_results(results)

    # 打印分析结果
    print("\n潜在暴涨机会：")
    print("-" * 100)
    for r in results:
        print(f"币种: {r['symbol']}")
        print(f"当前价格: ${r['current_price']:.4f}")
        print(f"预计目标价: ${r['target_price']:.4f} (涨幅: {((r['target_price']/r['current_price'])-1)*100:.2f}%)")
        print(f"概率: {r['probability']*100:.1f}%")
        print(f"预计达到时间: {r['estimated_time']}")
        print(f"信号: {r['signals']}")
        print(f"24h成交量: ${r['24h_volume']:,.0f}")
        print("-" * 100)

if __name__ == "__main__":
    main()
