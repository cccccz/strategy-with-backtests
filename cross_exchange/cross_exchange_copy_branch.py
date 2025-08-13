from shared_imports import redis,logging,time,os,Config,asyncio,json,websockets,ConnectionClosedError,ConnectionClosedOK,copy
from trading_state import TradingState
from websocket_data_fetch import binance_ws,okx_ws,bybit_ws,bitget_ws,bybit_orderbook_ws,bitget_orderbook_ws,okx_orderbook_ws,binance_orderbook_ws
from display import display_terminal
from logging_setup import setup_loggers
from strategy_engine import compute_strategy
from trade_executor import execute_simulation

state = TradingState()
# Global trading state instance shared across all system components
# Manages market data, positions, balances, and trading history

decision_logger, output_logger = setup_loggers()  
# Configured loggers for decision tracking and output monitoring
async def update_redis_data(state: TradingState):
    """
    Continuously update Redis with current trading state data.
    
    Runs an infinite loop that syncs trading state to Redis every second
    for external monitoring and data persistence.
    
    Args:
        state: TradingState instance containing all trading data
        
    Loop Behavior:
        - Calls state.update_redis_data() to sync all trading metrics
        - Sleeps 1 second between updates
        - Runs indefinitely until program termination
        
    Redis Keys Updated:
        - trading:exchange_data (market data)
        - trading:current_position (active trades)
        - trading:formatted_history (trade history)
        - trading:balance (balance information)
        - trading:latest_opportunity (current opportunities)
        - trading:metrics (performance metrics hash)
    """
    while True:
        await state.update_redis_data()
        await asyncio.sleep(1)


async def main():
    """
    Main program entry point for cryptocurrency arbitrage trading system.
    
    Initializes the trading system and runs all concurrent components:
    1. Market data collection from multiple exchanges via WebSocket
    2. Arbitrage strategy computation and opportunity detection  
    3. Trade execution simulation with position management
    4. Terminal display for real-time monitoring
    5. Redis data synchronization for external access
    
    System Architecture:
        - WebSocket connections: Binance, Bitget, OKX, Bybit (price feeds)
        - Strategy engine: Continuous opportunity scanning and decision making
        - Trade executor: Position opening/closing simulation
        - Display system: Real-time terminal interface
        - Redis sync: External data access and monitoring
        
    Initialization:
        - Sets up trading symbols from common symbol list
        - Initializes exchange balance allocation per Config
        - Logs monitored currency count
        
    Concurrent Tasks:
        - binance_ws(state): Binance WebSocket price feed
        - bitget_ws(state): Bitget WebSocket price feed  
        - okx_ws(state): OKX WebSocket price feed
        - bybit_ws(state): Bybit WebSocket price feed
        - compute_strategy(state): Arbitrage opportunity detection loop
        - execute_simulation(state): Trade execution and position management
        - display_terminal(state): Real-time terminal display
        - update_redis_data(state): Redis synchronization loop
        
    Note: 
        Orderbook WebSocket connections are commented out (optional deeper data)
        
    Runs:
        Indefinitely until manual termination or system error
    """
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
        # binance_orderbook_ws(state),
        # bitget_orderbook_ws(state),
        # okx_orderbook_ws(state),
        # bybit_orderbook_ws(state),
        compute_strategy(state),
        execute_simulation(state),
        display_terminal(state),
        update_redis_data(state)
    ]
    
    await asyncio.gather(*tasks)

if __name__ == '__main__':  
    asyncio.run(main())