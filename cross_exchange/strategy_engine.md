# Strategy Engine Quick Reference

## Overview
Cryptocurrency arbitrage strategy engine that detects price differences across exchanges, calculates costs/profits, and makes open/close decisions.

## Core Flow
1. **Detect** → Find price spreads across exchanges
2. **Calculate** → Compute costs, profits, trade sizing  
3. **Decide** → Determine open/close actions
4. **Evaluate** → Monitor active positions

## Key Functions

### Opportunity Detection
```python
calculate_opportunity(symbol, exchange_data) -> Optional[Dict]
# Finds best buy (lowest ask) and sell (highest bid) across exchanges
# Returns: {symbol, best_buy_exchange, best_sell_exchange, spread, spread_pct, ...}

compute_strategy(state) -> None  
# Main loop: scans all symbols, finds best opportunity, updates state
```

### Cost Calculation
```python
calculate_open_costs(buy_ex, sell_ex, prices, amount) -> Dict
# Returns: {total_cost, buy_fee, sell_fee, net_entry}

calculate_exit_costs(buy_ex, sell_ex, current_prices, amount) -> Dict  
# Returns: {total_cost, close_long_cost, close_short_cost, net_exit}
```

### Trade Enrichment
```python
enrich_with_costs_and_profits(opportunity, state) -> Dict
# Adds: trade_amount, trade_capital, estimated_net_profit, fees, stop_loss
```

### Decision Logic
```python
should_open_position(enriched_trade, state) -> bool
# Criteria: trade_amount > 0 AND spread_pct >= MIN_SPREAD_PCT_THRESHOLD

should_close_position(trade, current_status, state) -> bool  
# Criteria: Profit (spread shrunk) OR Stop loss (spread too negative)
```

### Position Monitoring
```python
evaluate_active_position(trade, market_snapshot, state) -> Dict
# Returns: {current_spread, unrealized_pnl, current_prices, position_age}
```

## Data Structures

### Opportunity
```python
{
    "symbol": "BTCUSDT",
    "best_buy_exchange": "Binance",     # Lowest ask
    "best_sell_exchange": "OKX",        # Highest bid  
    "open_spread": 15.5,                # Price difference
    "open_spread_pct": 0.004,           # Percentage spread
    "time_stamp_opportunity": 1234567890
}
```

### Enriched Trade
```python
{
    # ... all opportunity fields +
    "trade_amount": 0.1,
    "trade_capital": 1000.0,
    "estimated_net_profit": 25.5,
    "estimated_total_cost": 12.0,
    "decision_id": 123
}
```

## Strategy Parameters
- `MIN_SPREAD_PCT_THRESHOLD` - Minimum spread to open (0.004)
- `MAGIC_THRESHOLD` - Profit taking multiplier (0.5) 
- `STOP_LOSS_THRESHOLD` - Stop loss spread (-0.002)
- Trading fees from `Config.FUTURES_TRADING_FEES`

## Integration
Works with `TradingState` for market data, capital management, and trade tracking. Main entry point is `compute_strategy()` async loop.