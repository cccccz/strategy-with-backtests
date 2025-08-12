from shared_imports import time, setup_loggers,Config,copy
from trading_state import TradingState,asyncio

decision_logger, output_logger = setup_loggers()

# 策略计算函数 (拆分成小函数)
def extract_valid_quotes(exchange_data):
    """提取有效报价"""
    quotes = []
    for exchange, data in exchange_data.items():
        bid, ask = data['bid'], data['ask']
        if bid and ask:
            quotes.append((exchange, float(bid), float(ask)))
    return quotes

def create_opportunity_dict(symbol, best_buy, best_sell, quotes):
    """创建机会字典"""
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

def calculate_opportunity(symbol, exchange_data):
    """计算套利机会"""
    quotes = extract_valid_quotes(exchange_data)
    if len(quotes) < 2:
        return None

    best_buy = min(quotes, key=lambda x: x[2])  # ask
    best_sell = max(quotes, key=lambda x: x[1])  # bid

    return create_opportunity_dict(symbol, best_buy, best_sell, quotes)

async def compute_strategy(state:TradingState):
    """计算策略"""
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

# 成本计算函数
def calculate_open_costs(buy_exchange, sell_exchange, buy_price, sell_price, trade_amount):
    """计算开仓成本"""
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

def calculate_exit_costs(buy_exchange, sell_exchange, current_buy_price, current_sell_price, trade_amount):
    """计算平仓成本"""
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

def calculate_required_margin(trade_amount):
    return 0

# 交易执行函数
def enrich_with_costs_and_profits(opportunity,state:TradingState):
    """丰富机会信息，添加成本和利润计算"""
    quotes = opportunity['quotes']
    symbol = opportunity['symbol']
    best_buy = min(quotes, key=lambda x: x[2])
    best_sell = max(quotes, key=lambda x: x[1])

    trade_amount = state.calculate_trade_amount(best_buy[0],best_buy[2],best_sell[0],best_sell[1])
    # moved this logic to should_open.... if performance issue then bring it back..
    # if trade_amount <= 0:
    #     return None
    trade_capital = trade_amount * best_buy[2]
    costs = calculate_open_costs(best_buy[0], best_sell[0], best_buy[2], best_sell[1], trade_amount)
    estimated_close_costs = calculate_exit_costs(best_buy[0], best_sell[0], best_sell[1], best_buy[2], trade_amount)
    total_cost = costs['total_cost'] + estimated_close_costs['total_cost']
    
    open_spread = float(best_sell[1] - best_buy[2])
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

    estimated_net_profit = open_spread * trade_amount - total_cost
    estimated_net_profit_pct = estimated_net_profit / best_buy[2]
    margin_required = calculate_required_margin(trade_amount)
    
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
        'decision_id': opportunity['decision_id']
    }
    return enriched

def should_open_position(enrich_trade, statestate:TradingState):
    if enrich_trade['trade_amount'] <= 0:
        return False
    spread_pct = enrich_trade['open_spread_pct']
    
    if spread_pct >= Config.MIN_SPREAD_PCT_THRESHOLD:
    # if enrich_trade['estimated_net_profit'] > 0:
        decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
        return True
    
    decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 不满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
    return False
def should_close_position(trade, current_status, state):
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

def evaluate_active_position(trade, snapshot, state):
    """评估活跃仓位"""
    symbol = trade['symbol']
    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']

    current_buy_price = snapshot[symbol][buy_exchange]['bid']
    current_sell_price = snapshot[symbol][sell_exchange]['ask']
    
    if not (current_buy_price and current_sell_price):
        return None
    
    exit_costs = calculate_exit_costs(
        buy_exchange,
        sell_exchange,
        current_buy_price,
        current_sell_price,
        trade['trade_amount']
    )
    position_age = time.time() - trade['trade_time']
    
    # trade['net_entry'] is something i forget, do not recommend using it
    entry_net = trade['net_entry']
    entry_costs = trade['buy_fee'] + trade['sell_fee'] 
    exit_net = exit_costs['total_cost']
    spread_diff = trade['best_sell_price'] - trade['best_buy_price'] + current_buy_price - current_sell_price
    total_fee_cost = entry_costs + exit_net
    unrealized_pnl = spread_diff * trade['trade_amount'] - total_fee_cost

    return {
        'current_spread': current_sell_price - current_buy_price,
        'unrealized_pnl': unrealized_pnl,
        'exit_costs': exit_costs,
        'position_age': position_age,
        'current_buy_price': current_buy_price,
        'current_sell_price': current_sell_price,
        'decision_id': trade['decision_id'],
    }


def determine_exit_reason(trade, current_status):
    return "unimplemented"
