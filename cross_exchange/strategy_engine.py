from typing import Any, Dict, List, Optional, Tuple
from shared_imports import time, setup_loggers,Config,copy
from trading_state import TradingState,asyncio
from liquidity_tools import evaluate_liquidity


def extract_valid_quotes(exchange_data: Dict[str, Dict[str, Optional[float]]]) -> List[Tuple[str, float, float]]:
    """
    Extract valid bid/ask quotes from exchange data.
    
    Filters out exchanges with missing bid or ask prices and returns
    valid quotes as tuples for arbitrage calculation.
    
    Args:
        exchange_data: Dictionary mapping exchange names to price data
        {exchange: {"bid": price, "ask": price, 'orderbook':{"bids":[(bp1,bs1),...],"asks":[(ap1,aq1),...]}}}
            
    Returns:
        List of tuples: [(exchange_name, best_bid_price, best_ask_price), ...]
        Only includes exchanges with both bid and ask prices available.
    """
    quotes = []
    for exchange, data in exchange_data.items():
        bid = data.get('bid')
        ask = data.get('ask')
        if bid and ask:
            best_bid_price = bid
            best_ask_price = ask

            quotes.append((exchange, best_bid_price, best_ask_price))
    return quotes

def create_opportunity_dict(symbol: str, best_buy: Tuple[str, float, float], 
                          best_sell: Tuple[str, float, float], 
                          quotes: List[Tuple[str, float, float]]) -> Dict[str, Any]:
    """
    Create arbitrage opportunity dictionary with spread calculations.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        best_buy: Tuple of (exchange, bid_price, ask_price) with lowest ask
        best_sell: Tuple of (exchange, bid_price, ask_price) with highest bid
        quotes: List of all valid exchange quotes
        
    Returns:
        Opportunity dictionary containing:
        - symbol (str): Trading symbol
        - best_buy_exchange (str): Exchange with lowest ask price
        - best_buy_price (float): Lowest ask price (buy price)
        - best_sell_exchange (str): Exchange with highest bid price  
        - best_sell_price (float): Highest bid price (sell price)
        - open_spread (float): Price difference (sell_price - buy_price)
        - open_spread_pct (float): Percentage spread (2 * spread / (buy + sell))
        - quotes (List[Tuple]): All valid exchange quotes
        - # orderbook : All valid orderbooks, added in the father function
        - time_stamp_opportunity (float): Unix timestamp of opportunity detection
    """
    open_spread = best_sell[1] - best_buy[2]
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    
    return {
        'symbol': symbol,
        'best_buy_exchange': best_buy[0],
        'best_buy_price': best_buy[2],
        'best_sell_exchange': best_sell[0],
        'best_sell_price': best_sell[1],
        'open_spread': open_spread,
        'open_spread_pct': open_spread_pct,
        'quotes': quotes,
        'time_stamp_opportunity': time.time()
    }

def calculate_opportunity(symbol: str, exchange_data: Dict[str, Dict[str, Optional[float]]]) -> Optional[Dict[str, Any]]:
    """
    Calculate arbitrage opportunity for a given symbol across exchanges.
    
    Identifies best buy (lowest ask) and sell (highest bid) prices across
    exchanges and creates opportunity record if sufficient exchanges available.
    
    Args:
        symbol: Trading symbol to analyze
        exchange_data: Price data from all exchanges
            {exchange: {"bid": price, "ask": price, 'orderbook':{"bids":[(bp1,bs1),...],"asks":[(ap1,aq1),...]}}}
            
    Returns:
        Opportunity dictionary (see create_opportunity_dict) or None if
        fewer than 2 exchanges have valid quotes.
    """
    quotes = extract_valid_quotes(exchange_data)
    if len(quotes) < 2:
        return None

    best_buy = min(quotes, key=lambda x: x[2])  # ask
    best_sell = max(quotes, key=lambda x: x[1])  # bid

    opp_dict =  create_opportunity_dict(symbol, best_buy, best_sell, quotes)
    #hotfix
    best_buy_exchange = opp_dict['best_buy_exchange']
    best_sell_exchange = opp_dict['best_sell_exchange']
    best_buy_orderbook = exchange_data[best_buy_exchange]['orderbook']
    best_sell_orderbook = exchange_data[best_sell_exchange]['orderbook']
    opp_dict['best_buy_orderbook'] = best_buy_orderbook
    opp_dict['best_sell_orderbook'] = best_sell_orderbook

    return opp_dict


async def compute_strategy(state: TradingState,decision_logger,output_logger):
    """
    Main strategy computation loop for arbitrage detection.
    
    Continuously scans all symbols across exchanges to find the best
    arbitrage opportunity based on spread percentage. Updates state
    with the highest spread opportunity found in each cycle.
    
    Args:
        state: TradingState instance for market data and opportunity storage
        
    Runs indefinitely with Config.COMPUTE_INTERVAL sleep between cycles.
    Logs decisions and updates state.latest_opportunity with best opportunity.
    """
    while True:
        async with state.lock: 
            snapshot = copy.deepcopy(state.shared_data)

        best_strategy = None
        max_spread = 0
        
        for symbol, exchange_data in snapshot.items():
            opportunity = calculate_opportunity(symbol, exchange_data)
            if opportunity is not None and opportunity['open_spread_pct'] > max_spread:
                max_spread = opportunity['open_spread_pct']
                best_strategy = opportunity
                
        if best_strategy is not None:
            decision_id = await state.get_next_decision_id()
            best_strategy['decision_id'] = decision_id
            decision_logger.info(f"决策id:{decision_id}, 检测到当前最大价差值{best_strategy['open_spread']}, 价差比{best_strategy['open_spread_pct']}, 货币种类: {best_strategy['symbol']}, 买价: {best_strategy['best_buy_price']}@{best_strategy['best_buy_exchange']}, 卖价{best_strategy['best_sell_price']}@{best_strategy['best_sell_exchange']}")
            async with state.opportunity_lock:
                state.latest_opportunity = best_strategy

            
        await asyncio.sleep(Config.COMPUTE_INTERVAL)

def calculate_open_costs(buy_exchange: str, sell_exchange: str, 
                        buy_price: float, sell_price: float, 
                        trade_amount: float) -> Dict[str, float]:
    """
    Calculate costs for opening arbitrage position.
    
    Computes trading fees and slippage costs for simultaneous
    buy and sell orders on different exchanges.
    
    Args:
        buy_exchange: Exchange for buy order
        sell_exchange: Exchange for sell order
        buy_price: Purchase price (ask price)
        sell_price: Sale price (bid price)
        trade_amount: Quantity to trade
        
    Returns:
        Cost breakdown dictionary:
        - total_cost (float): Sum of all opening costs
        - buy_fee (float): Trading fee for buy order
        - sell_fee (float): Trading fee for sell order
        - net_entry (float): Net price difference (sell_price - buy_price)
    """
    trade_amount = float(trade_amount)
    buy_price = float(buy_price)
    sell_price = float(sell_price)
    buy_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[buy_exchange]['taker']) * buy_price
    sell_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[sell_exchange]['taker']) * sell_price
    slippage_cost = 0
    
    return {
        'total_cost': buy_fee + sell_fee + slippage_cost,
        'buy_fee': buy_fee,
        'sell_fee': sell_fee,
        'net_entry': sell_price - buy_price
    }

def calculate_exit_costs(buy_exchange: str, sell_exchange: str,
                        current_buy_price: float, current_sell_price: float,
                        trade_amount: float) -> Dict[str, float]:
    """
    Calculate costs for closing arbitrage position.
    
    Computes trading fees for closing long and short positions
    (reverse of opening trades).
    
    Args:
        buy_exchange: Exchange where original buy occurred (close short here)
        sell_exchange: Exchange where original sell occurred (close long here)  
        current_buy_price: Current bid price at buy exchange
        current_sell_price: Current ask price at sell exchange
        trade_amount: Quantity to close
        
    Returns:
        Cost breakdown dictionary:
        - total_cost (float): Sum of all closing costs
        - close_long_cost (float): Fee for closing long position
        - close_short_cost (float): Fee for closing short position
        - net_exit (float): Net price difference (current_buy - current_sell)
    """
    trade_amount = float(trade_amount)
    current_buy_price = float(current_buy_price)
    current_sell_price = float(current_sell_price)
    close_long_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[buy_exchange]['taker']) * current_buy_price
    close_short_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[sell_exchange]['taker']) * current_sell_price
    slippage_cost = 0

    return {
        'total_cost': close_long_fee + close_short_fee + slippage_cost,
        'close_long_cost': close_long_fee,
        'close_short_cost': close_short_fee,
        'net_exit': current_buy_price - current_sell_price
    }

def calculate_required_margin(trade_amount: float) -> float:
    """
    Calculate margin requirement for trade amount.
    
    Args:
        trade_amount: Quantity to trade
        
    Returns:
        Required margin (currently returns 0 - not implemented)
    """
    return 0


def enrich_with_costs_and_profits(opportunity: Dict[str, Any], 
                                 state: TradingState) -> Optional[Dict[str, Any]]:
    """
    Enrich opportunity with detailed cost and profit calculations.
    
    Adds comprehensive financial analysis to basic opportunity data including
    trade sizing, cost calculations, profit estimates, and risk metrics.
    
    Args:
        opportunity: Basic opportunity dictionary from calculate_opportunity
        state: TradingState for capital calculations
        
    Returns:
        Enriched opportunity dictionary containing all original fields plus:
        - trade_amount (float): Calculated trade quantity
        - trade_capital (float): Required capital (trade_amount * buy_price)
        - capital_utilization_pct (float): Percentage of total capital used
        - margin_required (float): Required margin
        - estimated_total_cost (float): Total opening + closing costs
        - buy_fee (float): Buy order trading fee
        - sell_fee (float): Sell order trading fee  
        - net_entry (float): Net entry price difference
        - estimated_net_profit (float): Expected profit after all costs
        - estimated_net_profit_pct (float): Profit as percentage of buy price
        - close_spread_pct (float): Estimated closing spread percentage
        - stop_loss (float): Stop loss threshold
        - decision_id (int): Unique decision identifier
        # 新增流动性相关字段
        'liquidity_check': {
            'buy_liquidity': Dict[str, Any],      # calculate_available_liquidity 返回值
            'sell_liquidity': Dict[str, Any],     # calculate_available_liquidity 返回值
            'sufficient_liquidity': bool,         # 综合流动性判断
            'liquidity_score': float              # 流动性评分 0-1
        },
        
        # 新增滑点相关字段
        'slippage_costs': {
            'buy_slippage': Dict[str, float],     # estimate_slippage_cost 返回值
            'sell_slippage': Dict[str, float],    # estimate_slippage_cost 返回值
            'total_slippage_cost': float,         # 总滑点成本 USDT
            'slippage_impact_pct': float          # 滑点对盈利影响百分比
        },
        
        # 新增市场冲击字段
        'market_impact': {
            'buy_impact_ratio': float,            # 买单市场冲击比例
            'sell_impact_ratio': float,           # 卖单市场冲击比例
            'combined_impact': float,             # 综合市场冲击
            'risk_level': str                     # 'LOW'/'MEDIUM'/'HIGH'
        },
        
        # 调整后的盈利数据
        'adjusted_profit': {
            'net_profit_after_slippage': float,  # 考虑滑点后净利润
            'adjusted_profit_pct': float,        # 调整后利润率
            'profit_confidence': float           # 盈利置信度 0-1
        }
    }        
        Returns None if trade_amount <= 0
    """
    quotes = opportunity['quotes']
    symbol = opportunity['symbol']
    best_buy = min(quotes, key=lambda x: x[2])
    best_sell = max(quotes, key=lambda x: x[1])

    buy_orderbook = opportunity['best_buy_orderbook']
    # print(orderbook)
    sell_orderbook = opportunity['best_sell_orderbook']

    trade_amount = state.calculate_trade_amount(best_buy[0],best_buy[2],best_sell[0],best_sell[1])
    # moved this logic to should_open.... if performance issue then bring it back..
    # if trade_amount <= 0:
    #     return None

    buy_liquidity_result = evaluate_liquidity(buy_orderbook,trade_amount,'buy',Config.MAX_SLIPPAGE_RATE)
    sell_liquidity_result = evaluate_liquidity(sell_orderbook,trade_amount,'sell',Config.MAX_SLIPPAGE_RATE)

    # buy_slippage_cost_result = estimate_slippage_cost(buy_orderbook,trade_amount,'buy')
    # sell_slippage_cost_result = estimate_slippage_cost(sell_orderbook,trade_amount,'sell')

    # get the trade_amount_v2 from the min of available_quantity from buy and sell
    trade_amount_slp = min(buy_liquidity_result['available_quantity'],sell_liquidity_result['available_quantity'])

    real_buy_liquidity_result =evaluate_liquidity(buy_orderbook,trade_amount_slp,'buy',Config.MAX_SLIPPAGE_RATE)
    real_buy_slippage_cost = real_buy_liquidity_result['absolute_slippage']

    real_sell_liquidity_result = evaluate_liquidity(sell_orderbook,trade_amount_slp,'sell',Config.MAX_SLIPPAGE_RATE)
    real_sell_slippage_cost = real_sell_liquidity_result['absolute_slippage']

    total_slippage_cost = real_buy_slippage_cost + real_sell_slippage_cost

    effective_buy_price = real_buy_liquidity_result['effective_price']
    effective_sell_price =real_sell_liquidity_result['effective_price']
    # print(f'before: {trade_amount}, after: {trade_amount_slp}')

    # without slippage
    trade_capital = trade_amount * best_buy[2]
    costs = calculate_open_costs(best_buy[0], best_sell[0], best_buy[2], best_sell[1], trade_amount)
    estimated_close_costs = calculate_exit_costs(best_buy[0], best_sell[0], best_sell[1], best_buy[2], trade_amount)
    total_cost = costs['total_cost'] + estimated_close_costs['total_cost']
    
    open_spread = float(best_sell[1] - best_buy[2])
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

    estimated_net_profit = open_spread * trade_amount - total_cost
    # i think this should be estimated_net_profit / trade_capital
    estimated_net_profit_pct = estimated_net_profit / best_buy[2]

    margin_required = calculate_required_margin(trade_amount)

    # with slippage
    buy_exchange = best_buy[0]
    sell_exchange = best_sell[0]
    trade_capital_slp = trade_amount_slp * effective_buy_price
    buy_costs_slp = trade_amount_slp  * effective_buy_price * Config.FUTURES_TRADING_FEES[buy_exchange]['taker']
    sell_costs_slp = trade_amount_slp  * effective_sell_price * Config.FUTURES_TRADING_FEES[sell_exchange]['taker']

    open_cost_slp= buy_costs_slp + sell_costs_slp
    # 有点累了所以使用open_cost_slp 作为估算，如果效果不好再修改
    estimated_close_cost_slp = open_cost_slp
    total_cost_slp = open_cost_slp + estimated_close_cost_slp
    # 使用加权均价作为比较值
    open_spread_slp = effective_sell_price - effective_buy_price
    open_spread_pct_slp = 2 * open_spread_slp / (effective_sell_price + effective_buy_price) if open_spread_slp != 0 else 0
    # close spread pct or slp pct is just meaningless
    estimated_pnl_slp = open_spread_slp * trade_amount_slp - total_cost_slp
    estimated_pnl_pct_slp = estimated_pnl_slp / trade_capital_slp if trade_capital_slp != 0 else 0
    # margin_required 

    enriched = {
        'symbol': symbol,
        'best_buy_exchange': best_buy[0],
        'best_buy_price': best_buy[2],
        'best_sell_exchange': best_sell[0],
        'best_sell_price': best_sell[1],
        'open_spread': open_spread,
        'open_spread_pct': open_spread_pct,
        'close_spread_pct': close_spread_pct,

        'trade_amount': trade_amount,
        'trade_capital': trade_capital,
        'trade_capital_sell':trade_amount * best_sell[1],
        'capital_utilization_pct': trade_capital/ state.initial_capital,

        'margin_required': margin_required,
        'estimated_total_cost': total_cost,
        'buy_fee': costs['buy_fee'],
        'sell_fee': costs['sell_fee'],
        'net_entry': costs['net_entry'],
        'estimated_net_profit': estimated_net_profit,
        'estimated_net_profit_pct': estimated_net_profit_pct,
        'time_stamp_opportunity': opportunity['time_stamp_opportunity'],
        'stop_loss': -abs(estimated_net_profit * Config.STOP_LOSS_PCT),
        'decision_id': opportunity['decision_id'],

        'buy_liquidity_init': buy_liquidity_result,      # calculate_available_liquidity 返回值
        'sell_liquidity_init': sell_liquidity_result,     # calculate_available_liquidity 返回值

        'buy_liquidity_real': real_buy_liquidity_result,     # estimate_slippage_cost 返回值
        'sell_liqudity_real': real_sell_liquidity_result,    # estimate_slippage_cost 返回值
        'trade_capital_slp':trade_capital_slp,
        'trade_amount_slp':trade_amount_slp,
        'buy_costs_slp':buy_costs_slp,
        'close_costs_slp':estimated_close_cost_slp,
        'total_cost_slp':total_cost_slp,
        'open_spread_slp':open_spread_slp,
        'open_spread_pct_slp':open_spread_pct_slp,
        'estimated_pnl_slp':estimated_pnl_slp,
        'estimated_pnl_pct_slp':estimated_pnl_pct_slp
    }
    return enriched

def should_open_position(enrich_trade: Dict[str, Any], state: TradingState,decision_logger,output_logger) -> bool:
    """
    Determine if position should be opened based on strategy criteria.
    
    Evaluates enriched trade opportunity against opening thresholds
    and logs decision reasoning.
    
    Args:
        enrich_trade: Enriched opportunity from enrich_with_costs_and_profits
        state: TradingState instance
        
    Returns:
        True if position should be opened, False otherwise
        
    Criteria:
        - trade_amount > 0
        - open_spread_pct >= Config.MIN_SPREAD_PCT_THRESHOLD
    """
    if state.opening_positions >= Config.MAX_POSITION_SIZE:
        return False
    
    # if enrich_trade['trade_amount'] <= 0:
    #     return False
    
    if enrich_trade['trade_amount_slp'] <= 0:
        return False 
    
    # spread_pct = enrich_trade['open_spread_pct']
    
    spread_pct = enrich_trade['open_spread_pct_slp']
    
    if spread_pct >= Config.MIN_SPREAD_PCT_THRESHOLD:
    # if enrich_trade['estimated_net_profit'] > 0:
        decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
        return True
    
    decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 不满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
    return False

def should_close_position(trade: Dict[str, Any], current_status: Dict[str, Any], 
                         state: TradingState,decision_logger,output_logger) -> bool:
    """
    Determine if active position should be closed based on strategy criteria.
    
    Evaluates current market conditions against profit-taking and
    stop-loss thresholds.
    
    Args:
        trade: Original trade record from active_trades
        current_status: Current position evaluation from evaluate_active_position
        state: TradingState instance
        
    Returns:
        True if position should be closed, False otherwise
        
    Criteria:
        - Profit taking: current_spread_pct <= open_spread_pct * Config.MAGIC_THRESHOLD
        - Stop loss: current_spread_pct <= Config.STOP_LOSS_THRESHOLD
    """
    current_spread_pct =  2 * current_status['current_spread'] / (current_status['current_buy_price'] + current_status['current_sell_price'])

    spread_diff = trade['open_spread_pct']
    # if (trade['open_spread_pct'] - current_spread_pct) >= Config.PROFIT_THRESHOLD:
    if current_spread_pct <= trade['open_spread_pct'] * Config.MAGIC_THRESHOLD:

    # if  current_spread_pct <= 0:

        decision_logger.debug(
            f"Spread check: open={trade['open_spread_pct']:.6f}, current={current_spread_pct:.6f}, diff={trade['open_spread_pct'] - current_spread_pct:.6f}"
        )

        decision_logger.info(
            f"决策id:{current_status['decision_id']}，触发平仓条件：止盈（当前价差率: {current_spread_pct:.6f}）"
        )
        return True

    if current_spread_pct <= Config.STOP_LOSS_THRESHOLD:
        decision_logger.info(
            f"决策id:{current_status['decision_id']}，触发平仓条件：止损（当前价差率: {current_spread_pct:.6f}）"
        )
        return True

    return False

def evaluate_active_position(trade: Dict[str, Any], snapshot: Dict[str, Any], 
                           state: TradingState) -> Optional[Dict[str, Any]]:
    """
    Evaluate current status of active arbitrage position.
    
    Calculates unrealized PnL, current spreads, and position metrics
    for active trade monitoring and close decisions.
    
    Args:
        trade: Active trade record containing opening details
        snapshot: Current market data snapshot  
        state: TradingState instance
        
    Returns:
        Position status dictionary:
        - current_spread (float): Current price spread
        - unrealized_pnl (float): Unrealized profit/loss
        - exit_costs (Dict): Estimated closing costs
        - position_age (float): Time since position opened (seconds)
        - current_buy_price (float): Current bid at buy exchange
        - current_sell_price (float): Current ask at sell exchange
        - decision_id (int): Trade decision identifier
        
        Returns None if current prices unavailable
    """
    symbol = trade['symbol']
    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']

    current_buy_price = snapshot[symbol][buy_exchange]['bid']
    current_buy_book = snapshot[symbol][buy_exchange]['orderbook']['bids']
    current_sell_price = snapshot[symbol][sell_exchange]['ask']
    current_sell_book = snapshot[symbol][sell_exchange]['orderbook']['asks']
    
    if not (current_buy_price and current_sell_price):
        return None
    
    if not (current_buy_book and current_sell_book):
        return None
    
    exit_costs = calculate_exit_costs(
        buy_exchange,
        sell_exchange,
        current_buy_price,
        current_sell_price,
        trade['trade_amount']
    )

    # without slp
    position_age = time.time() - trade['trade_time']
    
    # trade['net_entry'] is something i forget, do not recommend using it
    entry_net = trade['net_entry']
    entry_costs = trade['buy_fee'] + trade['sell_fee'] 
    exit_net = exit_costs['total_cost']
    spread_diff = trade['best_sell_price'] - trade['best_buy_price'] + current_buy_price - current_sell_price
    total_fee_cost = entry_costs + exit_net
    unrealized_pnl = spread_diff * trade['trade_amount'] - total_fee_cost

    # with slp
    trade_amount_slp = trade['trade_amount_slp']
    # 模拟滑点平仓，计算实际能交易的量，或者初版保证全平？
    # current_spread_slp = 

    return {
        'current_spread': current_sell_price - current_buy_price,
        'unrealized_pnl': unrealized_pnl,
        'exit_costs': exit_costs,
        'position_age': position_age,
        'current_buy_price': current_buy_price,
        'current_sell_price': current_sell_price,
        'decision_id': trade['decision_id'],
        # 'current_spread_slp':0,
    }

def determine_exit_reason(trade: Dict[str, Any], current_status: Dict[str, Any]) -> str:
    """
    Determine reason for position exit.
    
    Args:
        trade: Trade record being closed
        current_status: Current position status
        
    Returns:
        Exit reason string (currently returns "unimplemented")
    """
    return "unimplemented"

