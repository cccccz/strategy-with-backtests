from typing import Any, Dict
from shared_imports import setup_loggers,time,asyncio
from strategy_engine import determine_exit_reason,enrich_with_costs_and_profits,should_close_position,should_open_position,Config,evaluate_active_position,copy
from trading_state import TradingState
decision_logger, output_logger = setup_loggers()

def open_position(enrich_trade: Dict[str, Any], state: TradingState):
    """
    Execute position opening by updating balances and recording trade.
    
    Simulates opening an arbitrage position by:
    1. Deducting trade capital from both exchange available balances
    2. Adding capital to used balances on both exchanges
    3. Recording trade in history and active trades
    4. Incrementing opening positions counter
    
    Args:
        enrich_trade: Enriched trade dictionary from enrich_with_costs_and_profits containing:
            - best_buy_exchange (str): Exchange for buy order
            - best_sell_exchange (str): Exchange for sell order  
            - trade_capital (float): Total capital required
            - decision_id (int): Unique decision identifier
            - symbol (str): Trading symbol
            - All cost and profit calculations
        state: TradingState instance for balance and trade management
        
    Side Effects:
        - Updates exchange_balances for both exchanges
        - Adds trade to trade_history with action='open' and trade_time
        - Adds trade to active_trades list
        - Increments opening_positions counter
        - Logs opening details to decision_logger and output_logger
    """
    buy_exchange = enrich_trade['best_buy_exchange']
    sell_exchange = enrich_trade['best_sell_exchange']
    trade_capital = enrich_trade['trade_capital']

    state.exchange_balances[buy_exchange]['available'] -= trade_capital
    state.exchange_balances[buy_exchange]['used'] += trade_capital
    state.exchange_balances[buy_exchange]['total'] = state.exchange_balances[buy_exchange]['used'] + state.exchange_balances[buy_exchange]['available']


    state.exchange_balances[sell_exchange]['available'] -= trade_capital
    state.exchange_balances[sell_exchange]['used'] += trade_capital
    state.exchange_balances[sell_exchange]['total'] = state.exchange_balances[sell_exchange]['used'] + state.exchange_balances[sell_exchange]['available']



    enrich_trade['trade_time'] = time.time()
    enrich_trade['action'] = 'open'
    state.trade_history.append(enrich_trade)
    state.active_trades.append(enrich_trade)
    state.opening_positions += 1
    decision_logger.info(f"决策id: {enrich_trade['decision_id']} 开仓成功, 使用资金: {trade_capital: .2f} USDT")
    output_logger.info(f"开仓：决策id: {enrich_trade['decision_id']}, 货币种类：{enrich_trade['symbol']}，购买交易所：{enrich_trade['best_buy_exchange']}，购买价：{enrich_trade['best_buy_price']}，出售交易所：{enrich_trade['best_sell_exchange']}，出售价：{enrich_trade['best_sell_price']}，价差：{enrich_trade['open_spread']}，价差比：{enrich_trade['open_spread_pct']}")


def close_position(trade: Dict[str, Any], current_status: Dict[str, Any], state: TradingState):
    """
    Execute position closing by updating balances and recording final PnL.
    
    Simulates closing an arbitrage position by:
    1. Returning trade capital from used to available balances
    2. Distributing realized PnL equally between both exchanges
    3. Updating total balance and PnL tracking
    4. Recording close trade in history
    5. Removing from active trades
    
    Args:
        trade: Active trade dictionary containing opening details:
            - best_buy_exchange (str): Original buy exchange
            - best_sell_exchange (str): Original sell exchange
            - trade_capital (float): Original capital used
            - symbol (str): Trading symbol
            - All opening prices and costs
        current_status: Current position status from evaluate_active_position:
            - unrealized_pnl (float): Current profit/loss
            - current_buy_price (float): Current market buy price
            - current_sell_price (float): Current market sell price
            - decision_id (int): Trade decision identifier
        state: TradingState instance for balance and trade management
        
    Side Effects:
        - Updates exchange_balances by returning capital + PnL/2 to each exchange
        - Updates state.total_pnl and state.total_balance
        - Adds close trade to trade_history with action='close'
        - Removes trade from active_trades list
        - Decrements opening_positions counter
        - Logs detailed closing information to decision_logger and output_logger
    """
    trade = trade.copy()

    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']
    trade_capital = trade['trade_capital']
    pnl = current_status['unrealized_pnl']

    state.exchange_balances[buy_exchange]['used'] -= trade_capital
    # 平分
    state.exchange_balances[buy_exchange]['available'] += trade_capital + pnl / 2
    state.exchange_balances[buy_exchange]['total'] = state.exchange_balances[buy_exchange]['available']+state.exchange_balances[buy_exchange]['used']
    # state.exchange_balances[buy_exchange]['available'] += trade_capital + pnl

    state.exchange_balances[sell_exchange]['used'] -= trade_capital
    state.exchange_balances[sell_exchange]['available'] += trade_capital + pnl / 2
    state.exchange_balances[sell_exchange]['total'] = state.exchange_balances[sell_exchange]['available']+state.exchange_balances[sell_exchange]['used']


    state.total_pnl += pnl
    state.total_balance  = state.initial_capital + state.total_pnl
    trade.update({
        'action': 'close',
        'close_time': time.time(),
        'pnl': current_status['unrealized_pnl'],
        'exit_reason': determine_exit_reason(trade, current_status),
        'current_buy_price':current_status['current_buy_price'],
        'current_sell_price':current_status['current_sell_price'],
        'decision_id': current_status['decision_id']
    })
    state.trade_history.append(trade)
    state.active_trades.pop()
    state.opening_positions -= 1

    decision_logger.info(
        f"平仓成功｜决策id: {current_status['decision_id']}｜币种: {trade['symbol']}｜"
        f"开仓: {trade['best_buy_price']}@{trade['best_buy_exchange']} → {trade['best_sell_price']}@{trade['best_sell_exchange']}｜"
        f"平仓: {current_status['current_buy_price']}@{trade['best_buy_exchange']} → {current_status['current_sell_price']}@{trade['best_sell_exchange']}｜"
        f"原始价差: {trade['open_spread']:.6f}｜当前价差: {current_status['current_sell_price'] - current_status['current_buy_price']:.6f}｜"
        f"交易本金: {trade['trade_capital']:.6f}｜净收益: {trade['pnl']:.6f}｜收益率: {(trade['pnl'] / state.initial_capital):.6%}"
    )

    current_spread = current_status['current_sell_price'] - current_status['current_buy_price']
    current_spread_pct = 2 * current_spread / (current_status['current_sell_price']+current_status['current_buy_price'])

    output_logger.info(
        f"平仓: 决策id: {current_status['decision_id']}, 货币种类: {trade['symbol']}，平仓（卖出）交易所: {trade['best_buy_exchange']}, 平仓（卖出）价: {current_status['current_buy_price']}, 出售交易所：{trade['best_sell_exchange']}，平仓价：{current_status['current_sell_price']}，原始价差：{trade['open_spread']}，原始价差比：{trade['open_spread_pct']}，当前价差：{current_spread:.6f}，当前价差比：{current_spread_pct:.6f}，最终收益：{trade['pnl']:.6f}"
    )

async def execute_simulation(state: TradingState):
    """
    Main trading simulation execution loop.
    
    Continuously processes trading opportunities and manages active positions:
    1. Monitors for new arbitrage opportunities from strategy engine
    2. Opens positions when criteria are met and capacity allows
    3. Evaluates and closes active positions based on market conditions
    4. Enforces position limits and validates state consistency
    
    Args:
        state: TradingState instance containing:
            - latest_opportunity: New opportunities from compute_strategy
            - active_trades: Currently open positions
            - shared_data: Real-time market data
            - opening_positions: Position count tracking
            
    Loop Logic:
        - Checks for new opportunities (clears after processing)
        - Opens position if: positions < MAX_POSITION_SIZE AND should_open_position
        - For active trades: evaluates current status and closes if should_close_position
        - Validates opening_positions >= 0 (safety check)
        - Sleeps 0.1s between opportunity checks and position evaluations
        
    Side Effects:
        - Calls open_position() and close_position() based on strategy decisions
        - Updates state.latest_opportunity (clears processed opportunities)
        - Logs simulation start message
        - Runs indefinitely until interrupted
        
    Raises:
        ValueError: If opening_positions becomes negative (state corruption)
    """
    output_logger.info("开始模拟")
    print("[SIMULATION] Simulation starts")

    while True:
        await asyncio.sleep(0.1)
        async with state.opportunity_lock:
            opportunity = state.latest_opportunity
            state.latest_opportunity = None  # 清空，避免重复处理

        if opportunity is None:
            continue
        enrich_trade = enrich_with_costs_and_profits(opportunity,state)
        async with state.lock:
            if  should_open_position(enrich_trade, state):
                open_position(enrich_trade, state)

        async with state.lock:
            if state.opening_positions < 0:
                raise ValueError('opening_positions cannot be negative')
            
            if len(state.active_trades) > 0:
                snapshot = copy.deepcopy(state.shared_data)
                trade = state.active_trades[0]
                current_status = evaluate_active_position(trade, snapshot, state)
                if current_status and should_close_position(trade, current_status, state):
                    close_position(trade, current_status, state)
                
        await asyncio.sleep(0.1)