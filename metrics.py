import pandas as pd
import numpy as np
from typing import Dict, Optional

def calculate_returns(df: pd.DataFrame, signals: pd.Series,
                     holding_period: int = 24) -> Optional[pd.DataFrame]:
    """计算收益率"""
    try:
        returns = pd.DataFrame(index=df.index)
        returns['signal'] = signals
        returns['entry_price'] = df['close']
        returns['exit_price'] = df['close'].shift(-holding_period)
        returns['return'] = np.where(
            returns['signal'] == 1,
            (returns['exit_price'] - returns['entry_price']) / returns['entry_price'],
            0
        )
        returns['cumulative_return'] = (1 + returns['return']).cumprod()
        
        return returns
        
    except Exception as e:
        print(f"计算收益率失败: {str(e)}")
        return None

def calculate_metrics(returns: pd.DataFrame) -> Optional[Dict]:
    """计算性能指标"""
    try:
        signal_returns = returns[returns['signal'] == 1]['return'].dropna()
        
        if len(signal_returns) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        metrics = {
            'total_trades': len(signal_returns),
            'win_rate': len(signal_returns[signal_returns > 0]) / len(signal_returns),
            'avg_return': signal_returns.mean(),
            'std_return': signal_returns.std(),
            'max_return': signal_returns.max(),
            'min_return': signal_returns.min(),
            'total_return': returns['cumulative_return'].iloc[-1] - 1
        }
        
        # 计算夏普比率
        if metrics['std_return'] != 0:
            metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['std_return'] * \
                                    np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # 计算最大回撤
        cumulative = returns['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdowns.min()
        
        return metrics
        
    except Exception as e:
        print(f"计算指标失败: {str(e)}")
        return None
