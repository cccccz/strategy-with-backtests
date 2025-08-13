# Trade Executor Quick Reference

## Overview
Executes arbitrage trading decisions by managing position opening/closing and balance updates in simulation mode.

## Core Functions

### Position Management
```python
open_position(enriched_trade, state)
# Opens arbitrage position: deducts capital, records trade, logs action

close_position(trade, current_status, state)  
# Closes position: returns capital + PnL/2 to each exchange, updates totals
```

### Main Execution Loop
```python
execute_simulation(state)
# Processes opportunities → Opens positions → Monitors → Closes positions
# Runs indefinitely with 0.1s sleep cycles
```

## Key Operations

### Opening Position
- Deducts `trade_capital` from both exchanges' available balance
- Moves capital to `used` balance on both exchanges
- Adds trade to `active_trades` and `trade_history` 
- Increments `opening_positions` counter

### Closing Position
- Returns `trade_capital` from `used` to `available` on both exchanges
- Distributes `PnL/2` to each exchange's available balance
- Updates `total_pnl` and `total_balance`
- Removes from `active_trades`, adds close record to `trade_history`

## Balance Flow
```python
# Open: available → used (lock capital)
exchange_balances[ex]['available'] -= trade_capital
exchange_balances[ex]['used'] += trade_capital

# Close: used → available + profit (release capital + PnL)  
exchange_balances[ex]['used'] -= trade_capital
exchange_balances[ex]['available'] += trade_capital + pnl/2
```

## Execution Logic
```python
# Main loop checks:
1. New opportunities from strategy engine
2. Position limits (< MAX_POSITION_SIZE) 
3. Opening criteria (should_open_position)
4. Active position status (evaluate_active_position)
5. Closing criteria (should_close_position)
```

## Trade Records
- **Open**: `action="open"`, `trade_time`, all enriched trade data
- **Close**: `action="close"`, `close_time`, `pnl`, `exit_reason`, current prices

## Integration
- **Input**: Opportunities from `strategy_engine.compute_strategy()`
- **Output**: Trade executions recorded in `TradingState`
- **Dependencies**: Uses strategy functions for decision making and position evaluation