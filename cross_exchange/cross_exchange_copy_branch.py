from shared_imports import redis,logging,time,os,Config,asyncio,json,websockets,ConnectionClosedError,ConnectionClosedOK,copy
from trading_state import TradingState
from websocket_data_fetch import binance_ws,okx_ws,bybit_ws,bitget_ws
from display import display_terminal
from logging_setup import setup_loggers
from strategy_engine import compute_strategy
from trade_executor import execute_simulation
# 全局状态实例
state = TradingState()
decision_logger, output_logger = setup_loggers()

async def update_redis_data(state:TradingState):
    while True:
        await state.update_redis_data()
        await asyncio.sleep(1)


# 主程序
async def main():
    """主程序入口"""
    # 初始化
    state.init_symbols()
    state.init_exchange_balances()
    print(f"Monitoring {len(state.symbols)} currencies")
    
    # 启动所有任务
    tasks = [
        binance_ws(state),
        bitget_ws(state),
        okx_ws(state),
        bybit_ws(state),
        compute_strategy(state),
        execute_simulation(state),
        display_terminal(state),
        update_redis_data(state)
    ]
    
    await asyncio.gather(*tasks)

if __name__ == '__main__':  
    asyncio.run(main())