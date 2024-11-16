# Crypto Market Scanner

基于技术指标分析加密货币市场，寻找潜在的暴涨机会。

## 环境要求

- Python 3.7+
- pip 包管理器
- 网络连接（可能需要代理）

## 安装步骤

1. 安装依赖包：
```bash
pip install ccxt pandas numpy ta-lib
```

2. 配置代理（如果需要）：
   - 打开 config.json
   - 设置 "proxy.enabled" 为 true
   - 配置代理地址（默认为本地7890端口）

## 配置说明

配置文件 `config.json` 包含以下设置：

```json
{
    "proxy": {
        "enabled": true,          // 是否启用代理
        "http": "http://127.0.0.1:7890",  // HTTP代理地址
        "https": "http://127.0.0.1:7890"  // HTTPS代理地址
    },
    "api": {
        "timeout": 30000,         // API超时时间（毫秒）
        "rate_limit": 1000,       // 请求间隔（毫秒）
        "recv_window": 60000      // 接收窗口
    },
    "analysis": {
        "min_volume": 1000000,    // 最小24h交易量（USDT）
        "timeframe": "4h",        // K线时间周期
        "lookback_periods": 30,   // 回溯周期数
        "cache_expiry": 3600      // 缓存过期时间（秒）
    }
}
```

## 使用说明

1. 运行脚本：
```bash
python crypto_analyzer.py
```

2. 结果说明：
   - 分析结果保存在 analysis_results 目录
   - 每次运行生成新的时间戳文件
   - 包含以下信息：
     * 币种名称
     * 当前价格
     * 目标价格和涨幅
     * 成功概率
     * 预计达到时间
     * 触发信号
     * 24小时成交量

## 技术指标说明

分析使用以下技术指标：
- RSI（相对强弱指标）
  * 30-40区间视为超卖
  * 权重：0.6

- MACD（移动平均线趋同/背离）
  * MACD线上穿信号线且在0轴以下视为金叉即将形成
  * 权重：0.7

- 布林带
  * 价格触及下轨视为超卖
  * 权重：0.65

- 成交量分析
  * 成交量超过20周期均线1.5倍视为放量
  * 权重：0.55

## 信号系统

- 多重信号组合：
  * 每个技术指标独立评分
  * 最终概率为所有触发信号的平均值
  * 只有概率大于60%的交易对会被列出

- 目标价格计算：
  * 基于当前价格和概率值
  * 预期涨幅 = 概率 * 20%

- 时间预估：
  * 基于历史波动率和价格变化率
  * 考虑当前趋势方向
  * 自动转换为小时/天/周显示

## 缓存机制

- 市场数据缓存：5分钟
  * 减少API调用频率
  * 提高分析速度

- K线数据缓存：1小时
  * 保存在cache目录
  * 自动过期更新

## 常见问题

1. 连接问题：
   - 检查网络连接
   - 确认代理配置正确
   - 查看代理服务是否运行

2. 数据问题：
   - 检查缓存目录权限
   - 确认磁盘空间充足
   - 可以删除缓存目录重新运行

## 注意事项

- 分析结果仅供参考，不构成投资建议
- 建议在使用代理的情况下运行
- 定期清理缓存目录以节省空间
- 每次分析结果都会保存为独立文件，方便追踪和对比

## 结果文件说明

分析结果保存为JSON格式，包含以下字段：
```json
{
    "scan_time": "2024-01-01T12:00:00",  // 扫描时间
    "results": [
        {
            "symbol": "BTC/USDT",         // 交易对
            "current_price": 50000.0,      // 当前价格
            "target_price": 57500.0,       // 目标价格
            "probability": 0.7,            // 成功概率
            "signals": "RSI超卖, MACD金叉即将形成",  // 触发信号
            "24h_volume": 1000000.0,       // 24小时成交量
            "estimated_time": "2.5天"      // 预计达到时间
        }
    ],
    "total_opportunities": 1              // 机会总数
}
# qi
