# TradingState Quick Reference

## Overview
`TradingState` manages cryptocurrency arbitrage trading across multiple exchanges with thread-safe operations and Redis persistence.

## Core Data Structures

```python
# Market data: {symbol: {exchange: {"bid": price, "ask": price}}}
shared_data: Dict[str, Dict[str, Dict[str, Optional[float]]]]

# Active positions: List of open trade dictionaries
active_trades: List[Dict[str, Any]]

# All trades: List of open/close trade records  
trade_history: List[Dict[str, Any]]

# Exchange balances: {exchange: {"available": float, "used": float, "total": float}}
exchange_balances: Dict[str, Dict[str, float]]
```

## Key Methods

### Initialization
- `init_exchange_balances()` - Distribute capital across exchanges
- `init_symbols()` - Setup trading pairs and market data structure

### Capital Management
- `get_available_capital(exchange: str) -> float` - Get free capital
- `calculate_trade_amount(buy_ex, buy_price, sell_ex, sell_price) -> float` - Max tradeable quantity
- `update_balance(exchange: str, amount: float)` - Adjust balance

### Data Management
- `get_next_decision_id() -> int` - Generate unique trade ID
- `add_trade(trade_data: Dict)` - Add trade to history
- `update_redis_data()` - Sync state to Redis

### Analytics
- `calculate_metrics() -> Dict` - Performance stats (win_rate, total_profit, etc.)
- `_format_trade_history() -> Dict` - Paired trades with summary

## Trade Record Structure

### Open Trade
```python
{
    "action": "open",
    "symbol": "BTCUSDT", 
    "decision_id": 123,
    "best_buy_exchange": "Binance",
    "best_sell_exchange": "OKX",
    "trade_amount": 0.1,
    "trade_capital": 1000.0,
    # ... prices, fees, timestamps
}
```

### Close Trade  
```python
{
    "action": "close",
    "pnl": 15.50,
    "exit_reason": "profit_target",
    "close_time": 1234567890,
    # ... current prices, decision_id
}
```

## Thread Safety
All operations use asyncio locks (`lock`, `balance_lock`, `opportunity_lock`, `decision_id_lock`) for concurrent access.

## Redis Integration
Syncs to Redis keys: `trading:exchange_data`, `trading:current_position`, `trading:balance`, `trading:metrics`