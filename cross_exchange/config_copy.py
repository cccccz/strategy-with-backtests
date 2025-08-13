# config_copy.py
class Config:
    # 交易参数
    COMPUTE_INTERVAL = 1
    MINIMUM_PROFIT_PCT = 0.0025
    MAX_POSITION_SIZE = 1
    MAX_HOLDING_TIME = 300
    STOP_LOSS_PCT = -0.002
    # SLIPPAGE_RATE = 0.0005
    MAX_SLIPPAGE_RATE = 0.001
    
    # WebSocket重连参数
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_DELAY = 5
    CONNECTION_TIMEOUT = 10
    
        # 价差阈值设置
    # MIN_SPREAD_THRESHOLD = 0.001      # 最小开仓价差 (绝对值)
    MIN_SPREAD_PCT_THRESHOLD = 0.003 # 最小开仓价差百分比 自测0.005很难开仓，调低容易开仓

    
    MAGIC_THRESHOLD = 0.8        # 不确定它是否对应现实中的任何有意义的数据,测试调高容易平仓,0.5配合上面0.004
    STOP_LOSS_THRESHOLD = -0.002      # 止损阈值
    MAX_POSITION_TIME = 30           # 最大持仓时间 (秒)
    MAX_POSITION_SIZE = 1             # 最大持仓数量
    # 交易费率
    FUTURES_TRADING_FEES = {
        'Binance': {'maker': 0.0002, 'taker': 0.0005},
        'Bybit': {'maker': 0.0002, 'taker': 0.00055},
        'OKX': {'maker': 0.0002, 'taker': 0.0005},
        'Bitget': {'maker': 0.0002, 'taker': 0.0006}
    }
    # 资金管理配置
    INITIAL_CAPITAL = 10000.0
    # 各交易所资金分配比例
    EXCHANGE_CAPITAL_ALLOCATION = {
        'Binance': 0.25,
        'Bybit': 0.25,
        'OKX': 0.25,
        'Bitget': 0.25
    }

    # 单笔交易最大资金使用比例
    MAX_TRADE_CAPITAL_PCT = 0.1
    # 最小交易金额
    MIN_TRADE_AMOUNT = 10.0

    # 账本深度
    MAX_ORDERBOOK_DEPTH = 15

# 交易金额
# 交易数量
# 最终收益率 = 净收益 / 本金（固定）
