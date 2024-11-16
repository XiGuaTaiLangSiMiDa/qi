import pandas as pd
import ta
from typing import Dict

def calculate_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """计算技术指标"""
    try:
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], 
                                  window=params['rsi']['period'])
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(
            df['close'],
            window=params['stoch_rsi']['period'],
            smooth1=params['stoch_rsi']['k'],
            smooth2=params['stoch_rsi']['d']
        )
        df['stoch_k'] = stoch_rsi.stochrsi_k()
        df['stoch_d'] = stoch_rsi.stochrsi_d()
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_fast=params['macd']['fast'],
            window_slow=params['macd']['slow'],
            window_sign=params['macd']['signal']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            df['close'],
            window=params['bollinger']['period'],
            window_dev=params['bollinger']['std']
        )
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # OBV
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ema'] = ta.trend.ema_indicator(
            df['obv'], 
            window=params['obv']['period']
        )
        
        return df
        
    except Exception as e:
        print(f"计算指标失败: {str(e)}")
        return None

def generate_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    """生成交易信号"""
    try:
        signals = pd.Series(0, index=df.index)
        
        # RSI信号
        rsi_signal = (df['rsi'] >= params['rsi']['lower']) & \
                    (df['rsi'] <= params['rsi']['upper'])
        
        # Stochastic RSI信号
        stoch_signal = (df['stoch_k'] <= params['stoch_rsi']['lower']) & \
                      (df['stoch_k'] > df['stoch_d'])
        
        # MACD信号
        macd_signal = (df['macd'] > df['macd_signal']) & \
                     (df['macd'] < params['macd']['threshold'])
        
        # 布林带信号
        bb_signal = df['close'] <= df['bb_lower']
        
        # OBV信号
        obv_signal = df['obv'] > df['obv_ema'] * params['obv']['threshold']
        
        # 组合信号
        signals[rsi_signal & (macd_signal | stoch_signal) & \
               (bb_signal | obv_signal)] = 1
        
        return signals
        
    except Exception as e:
        print(f"生成信号失败: {str(e)}")
        return None
