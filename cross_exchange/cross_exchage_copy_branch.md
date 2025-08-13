# Main Program Quick Reference

## Overview
Cryptocurrency arbitrage trading system orchestrator - initializes components and runs concurrent data collection, strategy execution, and monitoring.

## System Architecture

### 4-Layer Concurrent Design
```python
# Data Layer: WebSocket price feeds
binance_ws, bitget_ws, okx_ws, bybit_ws → state.shared_data

# Strategy Layer: Opportunity detection  
compute_strategy → state.latest_opportunity

# Execution Layer: Trade simulation
execute_simulation → state.active_trades, trade_history

# Interface Layer: Monitoring & persistence
display_terminal + update_redis_data → Real-time UI + Redis sync
```

## Main Function Flow
```python
async def main():
    # 1. Initialize
    state.init_symbols()           # Setup trading pairs
    state.init_exchange_balances() # Distribute capital
    
    # 2. Launch 8 concurrent tasks
    await asyncio.gather(
        # Data collection (4 exchanges)
        binance_ws, bitget_ws, okx_ws, bybit_ws,
        # Core logic
        compute_strategy,     # Find arbitrage opportunities  
        execute_simulation,   # Execute trades
        # Monitoring
        display_terminal,     # Real-time UI
        update_redis_data     # External data access
    )
```

## Data Flow
```
WebSockets → shared_data → Strategy → Opportunities → Executor → Trades
     ↓                                                              ↓
  Display ←←←←←←←←←←←←←←← Redis Sync ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

## Key Components

### Global State
- `state = TradingState()` - Central data hub shared across all components
- `decision_logger, output_logger` - Logging infrastructure

### Redis Sync Loop
```python
update_redis_data(state)
# Updates Redis every 1 second with:
# - Market data, positions, history, balances, opportunities, metrics
```

## Initialization
1. **Symbols**: Setup trading pairs from common symbol list
2. **Balances**: Distribute initial capital across exchanges per Config
3. **Monitoring**: Log currency count being monitored

## Execution Model
- **Concurrent**: All 8 tasks run simultaneously via `asyncio.gather()`
- **Persistent**: Runs indefinitely until manual termination
- **Real-time**: WebSocket feeds provide live market data
- **Simulated**: Trade execution in simulation mode (no real orders)

## Entry Point
```python
if __name__ == '__main__':
    asyncio.run(main())
```