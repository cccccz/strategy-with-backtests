import asyncio
import functools
import json
import copy
import numpy as np
import websockets
import os
import time
from rest_api import get_common_symbols
import pandas as pd
from decimal import Decimal, getcontext
from functools import wraps

def quick_timer(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[PERF] {func.__name__}: {(end-start)*1000:.2f}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[PERF] {func.__name__}: {(end-start)*1000:.2f}ms")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def display_exchange_data():
        clear_terminal()
        print("+------------+-----------+-------------+-------------+")
        print("|   Symbol   | Exchange  | Bid Price   |  Ask Price  |")
        print("+------------+-----------+-------------+-------------+")

        for symbols, orderbooks in shared_data.items():
            
            for exchange, book in orderbooks.items():
                # print(symbols, exchange, book)
                # print(orderbooks.items())
                bid = book['bid'] if book['bid'] else "..."
                ask = book['ask'] if book['ask'] else "..."
                print(f"| {symbols:<10} | {exchange:<9} | {str(bid):<11} | {str(ask):<11} |")

        print("+------------+-----------+-------------+-------------+")

def display_trade():
    if not active_trades.empty():
        strategy_result = active_trades[0]
        print(f"Symbol: {strategy_result['symbol']}")
        print(f"Best Buy on: {strategy_result['best_buy_exchange']}, {strategy_result['best_buy_price']}")
        print(f"Best Sell on: {strategy_result['best_sell_exchange']}, {strategy_result['best_sell_price']}")
        print(f"Open Spread: {strategy_result['open_spread']:.4f}")
        print(f"Open Spread Percentage: {strategy_result['open_spread_pct']*100:.2f}%")
        print(f"Close Spread Percentage: {strategy_result['close_spread_pct']*100:.2f}%")
        
        # 新增成本相关信息
        print(f"Buy Fee: {strategy_result['buy_fee']:.6f}")
        print(f"Sell Fee: {strategy_result['sell_fee']:.6f}")
        print(f"Slippage Cost: {strategy_result['slippage_cost']:.6f}")
        print(f"Total Cost: {strategy_result['total_cost']:.6f}")
        print(f"Estimated Net Profit: {strategy_result['estimated_net_profit']:.6f}")
        print(f"Estimated Net Profit Percentage: {strategy_result['estimated_net_profit_pct']*100:.2f}%")
        
        print(f"Time Stamp Opportunity: {strategy_result['time_stamp_opportunity']}")
        print(f"Trade Time: {strategy_result['trade_time']}")

        print("-" * 50)
    
async def display_terminal():
    while True:
        display_exchange_data()

        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(1)


# -------- Binance --------

async def binance_ws():
    params = [f"{s.lower()}@bookTicker" for s in symbols]
    uri = f"wss://fstream.binance.com/stream?streams=" + "/".join(params)

    subscribe_msg = {
        "method": "SUBSCRIBE",
        "params": params,
        "id": 1
    }
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            envelope = json.loads(msg)
            payload = envelope.get('data', {})
            symbol = payload.get('s')
            bid = payload.get('b')
            ask = payload.get('a')
            if symbol in shared_data:
                shared_data[symbol]["Binance"]["bid"] = Decimal(bid)
                shared_data[symbol]["Binance"]["ask"] = Decimal(bid)

            # print(f"[Binance] BTCUSDT: bid={bid} ask={ask}")

async def bybit_ws():
    uri = "wss://stream.bybit.com/v5/public/linear"
    async with websockets.connect(uri) as ws:
        params = [f"tickers.{symbol}" for symbol in symbols]
        subscribe_msg = {
            "op": "subscribe",
            "args": params
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            try:
                msg = await ws.recv()
                envelope = json.loads(msg)
                               
                # Check if this is a ticker data message
                if 'data' in envelope and envelope.get('topic', '').startswith('tickers.'):
                    payload = envelope['data']
                    
                    symbol = payload.get('symbol')
                    bid = payload.get('bid1Price') 
                    ask = payload.get('ask1Price')  
                    
                    if symbol and bid and ask and symbol in shared_data:
                        if 'bid1Price' in payload:
                            shared_data[symbol]["Bybit"]["bid"] = Decimal(bid)
                        if 'ask1Price' in payload:
                            shared_data[symbol]["Bybit"]["ask"] = Decimal(ask)
                        # print(f"[Bybit] {symbol}: bid={bid} ask={ask}")
                    # else:
                    #     # Debug missing data
                    #     print(f"[Bybit DEBUG] Symbol: {symbol}, bid1Price: {bid}, ask1Price: {ask}")
                    #     print(f"[Bybit DEBUG] Available fields: {list(payload.keys())}")
                        
            except Exception as e:
                print(f"[Bybit ERROR] {e}")
                await asyncio.sleep(1)

# -------- Bitget --------
async def bitget_ws():
    uri = "wss://ws.bitget.com/v2/ws/public"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "instType": "USDT-FUTURES",
                "channel": "ticker",
                "instId": symbol
            }for symbol in symbols]
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'data' in data and len(data['data']) > 0:
                item = data['data'][0]
                bid = item.get('bidPr')
                ask = item.get('askPr')
                symbol = item['instId']
                if symbol in shared_data:
                    shared_data[symbol]["Bitget"]["bid"] = Decimal(bid)
                    shared_data[symbol]["Bitget"]["ask"] = Decimal(ask)
                # print(f"[Bitget]  BTCUSDT: bid={bid} ask={ask}")

# -------- OKX --------
async def okx_ws():
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(uri) as ws:
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": symbol
            }for symbol in okx_symbols]
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'data' in data and len(data['data']) > 0:
                item = data['data'][0]
                bid = item.get('bidPx')
                ask = item.get('askPx')
                symbol = item['instId'].replace('-SWAP','').replace('-','')
                if symbol in shared_data:
                    shared_data[symbol]["OKX"]["bid"] = Decimal(bid)
                    shared_data[symbol]["OKX"]["ask"] = Decimal(ask)
                # print(f"[OKX]     BTC-USDT: bid={bid} ask={ask}")

# async def compute_strategy():
#     """
#     compute_strategy only runs every COMPUTE_INTERVAL seconds
#     bid 买价 ask 卖价
#     """
    
#     compute_interval = COMPUTE_INTERVAL
#     minimum_profit_pct = MINIMUM_PROFIT_PCT


#     while True:
#         loop_start = time.perf_counter()
#         async with lock: 
#             # snapshot = copy.deepcopy(shared_data)
#             snapshot = copy.deepcopy(shared_data)

#         best_strategy = None
#         print(f"total number of data {len(snapshot.items())}")

#         for symbol, exchange_data in snapshot.items():
#             quotes = []
#             for exchange, data in exchange_data.items():
#                 bid, ask = data['bid'], data['ask']
#                 if bid and ask:
#                     quotes.append((exchange, float(bid), float(ask)))

#             if len(quotes) < 2:
#                 continue

#             # 在最便宜的卖家 买入
#             # 在出价最高的买家 卖出
#             best_buy = min(quotes, key=lambda x: x[2])  # ask
#             best_sell = max(quotes, key=lambda x: x[1])  # bid
#             trade_amount = 1
#             costs =  calculate_total_cost(best_buy[0],best_sell[0],best_buy[2],best_sell[1],trade_amount)
#             total_cost = costs['total_cost']
#             buy_fee = costs['buy_fee'] 
#             sell_fee = costs['sell_fee']
#             slippage_cost = costs['slippage_cost']
#             open_spread = best_sell[1] - best_buy[2]
#             open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
#             close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])


#             estimated_net_profit = open_spread - total_cost
#             estimated_net_profit_pct = estimated_net_profit / best_buy[2]
#             strategy_result = None
#             if estimatednet_profit_pct > minimum_profit_pct:
#                 strategy_result = {
#                     'symbol': symbol,
#                     'best_buy': best_buy[0],
#                     'best_buy_price': best_buy[2],
#                     'best_sell': best_sell[0],
#                     'best_sell_price': best_sell[1],
#                     'open_spread': open_spread,
#                     'open_spread_pct': open_spread_pct,
#                     'close_spread_pct': close_spread_pct,

#                     'total_cost': total_cost,
#                     'buy_fee': buy_fee,
#                     'sell_fee': sell_fee,
#                     'slippage_cost': slippage_cost,
#                     'estimated_profit': estimated_net_profit,
#                     'estimated_net_profit_pct': estimated_net_profit_pct,
#                     # 'profit_pct': profit_pct,
#                     'time_stamp': time.time()}
            
#             if best_strategy is None or open_spread_pct > best_strategy['open_spread_pct']:
#                 best_strategy = strategy_result

#         if best_strategy:
#             await strategy_results_queue.put(best_strategy)
#                 # print(f"[ARBITRAGE] {symbol}: Buy on {best_buy[0]} @ {best_buy[2]}, "
#                 #       f"Sell on {best_sell[0]} @ {best_sell[1]} "
#                 #       f"=> Profit: {profit_pct:.4%} > Cost: {cost_pct:.4%}")
#                 # simulate_trade(...)
#         loop_time = (time.perf_counter() - loop_start) * 1000
#         print(f"[TIMING] Calculation Strategy Loop took: {loop_time:.2f}ms")
#         await asyncio.sleep(compute_interval)

async def compute_strategy():
    """
    compute_strategy only runs every COMPUTE_INTERVAL seconds
    bid 买价 ask 卖价
    """

    while True:
        async with lock: 
            # snapshot = copy.deepcopy(shared_data)
            snapshot = copy.deepcopy(shared_data)

        best_strategy = None
        print(f"total number of data {len(snapshot.items())}")
        max_spread = 0
        for symbol, exchange_data in snapshot.items():
            opportunity = calculate_opportunity(symbol,exchange_data)
            if opportunity is not None and opportunity['open_spread_pct'] >= MINIMUM_SPREAD_PCT and opportunity['open_spread_pct'] > max_spread:
                max_spread = opportunity['open_spread_pct']
                await strategy_results_queue.put(opportunity)
        await asyncio.sleep(COMPUTE_INTERVAL)


def calculate_opportunity(symbol, exchange_data):
    """
    return an opportunity or None
    """
    quotes = []
    for exchange, data in exchange_data.items():
        bid, ask = data['bid'], data['ask']
        if bid and ask:
            quotes.append((exchange, float(bid), float(ask)))

    if len(quotes) < 2:
        return None

    # 在最便宜的卖家 买入
    # 在出价最高的买家 卖出
    best_buy = min(quotes, key=lambda x: x[2])  # ask
    best_sell = max(quotes, key=lambda x: x[1])  # bid

    open_spread = best_sell[1] - best_buy[2]
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

    # in production: no need to calculate anything other than spread, all info in quotes
    return {
        'symbol': symbol,
        'best_buy_exchange': best_buy[0],
        'best_buy_price': best_buy[2],
        'best_sell_exchange': best_sell[0],
        'best_sell_price': best_sell[1],
        'open_spread': open_spread,
        'open_spread_pct': open_spread_pct,
        'quotes':quotes,
        # 'close_spread_pct': close_spread_pct,
        'time_stamp_opportunity': time.time()}



def enrich_with_costs_and_profits(opportunity):
    strategy_result = opportunity
    quotes = opportunity['quotes']
    symbol = opportunity['symbol']
    best_buy = min(quotes, key=lambda x: x[2])  # ask
    best_sell = max(quotes, key=lambda x: x[1])  # bid

    trade_amount = 1
    costs =  calculate_open_costs(best_buy[0],best_sell[0],best_buy[2],best_sell[1],trade_amount)
    total_cost = costs['total_cost']
    buy_fee = costs['buy_fee'] 
    sell_fee = costs['sell_fee']
    # slippage_cost = costs['slippage_cost']
    open_spread = best_sell[1] - best_buy[2]
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

# maybe add slippage here too
    estimated_net_profit = open_spread - total_cost
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
            'margin_required': margin_required,
            'total_cost': total_cost,
            'buy_fee': buy_fee,
            'sell_fee': sell_fee,
            'net_entry': costs['net_entry'],
            'estimated_profit': estimated_net_profit,
            'estimated_net_profit_pct': estimated_net_profit_pct,
            # 'profit_pct': profit_pct,
            'time_stamp_opportunity': opportunity['time_stamp_opportunity'],
             }
    return enriched

def open_position(enrich_trade):
    global opening_positions
    enrich_trade['trade_time'] = time.time()
    enrich_trade['action'] = 'open'
    trade_history.append(enrich_trade)
    active_trades.append(enrich_trade)
    opening_positions += 1

def calculate_open_costs(buy_exchange, sell_exchange, buy_price, sell_price, trade_amount):
    """Calculate costs to OPEN arbitrage position
    - BUY at buy_exchange's ASK price (pay ask)
    - Sell at sell_exchanges's BID price (receive bid)"""
    buy_fee = trade_amount * FUTURES_TRADING_FEES[buy_exchange]['taker'] * buy_price
    sell_fee = trade_amount * FUTURES_TRADING_FEES[sell_exchange]['taker'] * sell_price
    # slippage_cost = trade_amount * SLIPPAGE_RATE
    slippage_cost = 0

    
    return {
        'total_cost': buy_fee + sell_fee + slippage_cost,
        'buy_fee': buy_fee,
        'sell_fee': sell_fee,
        'net_entry' : sell_price - buy_price
    }

def calculate_required_margin(trade_amount):
    return 0

def calculate_exit_costs(buy_exchange,sell_exchange,current_buy_price,current_sell_price,trade_amount):
    """
    Calculate costs to CLOSE arbitrage position
    - Sell at buy_exchange's BID price (close long position)
    - BUY at sell_echange's ASK price (close short position)
    """
    close_long_fee = trade_amount * FUTURES_TRADING_FEES[buy_exchange]['taker'] * current_buy_price
    close_short_fee = trade_amount * FUTURES_TRADING_FEES[sell_exchange]['taker'] * current_sell_price
    # slippage_cost = trade_amount * SLIPPAGE_RATE
    slippage_cost = 0

    
    return {
        'total_cost': close_long_fee + close_short_fee + slippage_cost,
        'close_long_cost': close_long_fee,
        'close_short_cost': close_short_fee,
        'net_exit' : current_buy_price - current_sell_price
    }

# def calculate_unrealized_pnl(trade, current_buy_bid, current_sell_ask):
#     """profit/loss if we close now"""
#     entry_net = trade['net_entry']
#     entry_costs = trade['total_cost']


#     return unrealized_pnl

async def handle_active_positions():
    if opening_positions <= 0:
        raise ValueError('opening_positions cannot be negative')
    async with lock: 
        snapshot = copy.deepcopy(shared_data)

    # fetch trade info
    trade = active_trades[0]
    symbol = trade['symbol']
    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']

    # fetch current market info
    current_buy_price = snapshot[symbol][buy_exchange]['ask']
    current_sell_price = snapshot[symbol][sell_exchange]['bid']
    
    if not (current_buy_price and current_sell_price):
        return None
    
    # cost calculation
    exit_costs = calculate_exit_costs(
        buy_exchange,
        sell_exchange,
        current_buy_price,
        current_sell_price,
        trade['trade_amount']
    )
    position_age = time.time() - trade['trade_time']
    current_spread = current_sell_price - current_buy_price
    #unrealized_pnl

    # should close? = profit evaluation + risk management
    if position_age > MAX_HOLDING_TIME:
        return True
    entry_net = trade['net_entry']
    entry_costs = trade['total_cost']
    exit_net = exit_costs['exit_net']
    unrealized_pnl = entry_net + exit_net - entry_costs - exit_costs['total_cost']

    # TODO
    if unrealized_pnl >= trade['estimated_net_profit']:
        return True
    # execute close position or do nothing

    if unrealized_pnl <= trade['stop_loss']:
        return True

    return False

def should_open_position(enrich_trade):
    if opening_positions <= MAX_POSITION_SIZE:
        # logic about balance on those exchange...
        # TODO
        if enrich_trade['estimated_net_profit_pct'] >= MINIMUM_PROFIT_PCT:
            # and some logic about margin
            return True
    
    return False

def evaluate_active_position(trade,snapshot):
    # fetch trade info
    symbol = trade['symbol']
    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']

    # fetch current market info
    current_buy_price = snapshot[symbol][buy_exchange]['ask']
    current_sell_price = snapshot[symbol][sell_exchange]['bid']
    
    if not (current_buy_price and current_sell_price):
        return None
    
    # cost calculation
    exit_costs = calculate_exit_costs(
        buy_exchange,
        sell_exchange,
        current_buy_price,
        current_sell_price,
        trade['trade_amount']
    )
    position_age = time.time() - trade['trade_time']
    current_spread = current_sell_price - current_buy_price
    #unrealized_pnl

    # should close? = profit evaluation + risk management
    if position_age > MAX_HOLDING_TIME:
        return True
    entry_net = trade['net_entry']
    entry_costs = trade['total_cost']
    exit_net = exit_costs['exit_net']
    unrealized_pnl = entry_net + exit_net - entry_costs - exit_costs['total_cost']

    return {'current_spread': current_sell_price - current_buy_price,
            'unrealized_pnl': unrealized_pnl,
            'exit_costs': exit_costs,
            'position_age': position_age,
            'current_buy_price': current_buy_price,
            'current_sell_price': current_sell_price
            }

def should_close_position(trade,current_status):
    unrealized_pnl = current_status['unrealized_pnl']
    if unrealized_pnl >= trade['estimated_net_profit']* 0.5:
        return True
    # execute close position or do nothing

    if unrealized_pnl <= trade['stop_loss']:
        return True

    return False

def determine_exit_reason(trade,current_status):
    return "unimplemented"

def close_position(trade,current_status):
    global opening_positions
    trade = trade.copy()
    trade.update({
        'action': 'close',
        'close_time': time.time(),
        'pnl': trade['unrealized_pnl'],
        'exit_reason': determine_exit_reason(trade,current_status)})
    trade_history.append(trade)
    active_trades.pop()
    opening_positions -= 1

async def execute_simulation():
    print("[SIMULATION] Simulation starts")
    async with lock: 
        trades = copy.deepcopy(trade_history)
    #where to lock? # what to lock?

    while True:
    # await handle_new_opportunities()
        opportunity = await strategy_results_queue.get()
        enrich_trade = enrich_with_costs_and_profits(opportunity)
    
        if should_open_position(enrich_trade):
            open_position(enrich_trade)

        if opening_positions <= 0:
            raise ValueError('opening_positions cannot be negative')
        
        # handle active positions
        async with lock: 
            snapshot = copy.deepcopy(shared_data)

        trade = active_trades[0]
        current_status = evaluate_active_position(trade,snapshot)
        if current_status:
            unrealized_pnl = current_status['unrealized_pnl']
        # else handle current_status = None
        if should_close_position(trade,current_status):
            close_position(trade,current_status)
        
        # print('fake handling_active_position')
        await asyncio.sleep(0.1)


async def main():
    await asyncio.gather(
        binance_ws(),
        bitget_ws(),
        okx_ws(),
        bybit_ws(),
        compute_strategy(),
        execute_simulation(),
        display_terminal(),
    )


if __name__ == '__main__':  
    # 期货交易费率 2025-08-06 
    FUTURES_TRADING_FEES = {
        'Binance': {
            'maker': 0.0002,  # 0.02%
            'taker': 0.0005,  # 0.05%
        },
        'Bybit': {
            'maker': 0.0002,  # 0.02% 
            'taker': 0.00055   # 0.055%
        },
        'OKX': {
            'maker': 0.0002,  # 0.02%
            'taker': 0.0005   # 0.05%
        },
        'Bitget': {
            'maker': 0.0002,  # 0.02%
            'taker': 0.0006   # 0.06%
        }
    }
    SLIPPAGE_RATE = 0.0005
    COMPUTE_INTERVAL = 1
    MINIMUM_PROFIT_PCT = 0

    TRADE_AMOUNT_USDT = 1000    # 每次套利的金额
    MAX_POSITION_SIZE = 5000    # 最大持仓限制
    AVAILABLE_BALANCE = {       # 每个交易所的可用余额
        'Binance': 10000,
        'Bybit': 10000,
        'OKX': 10000,
        'Bitget': 10000
}
    MAX_HOLDING_TIME = 300      # 最大持仓时间(秒)
    STOP_LOSS_PCT = -0.5       # 止损百分比
    MAX_CONCURRENT_TRADES = 1   # 最大同时进行的套利数量
    MINIMUM_SPREAD_PCT = 0.0002
    active_trades = []
    trade_history = []

    # initialize shared_data
    exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
    symbols = get_common_symbols()
    okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in symbols]
    #print(f"monitoring {len(symbols)} currencies") 163
    shared_data = {
        symbol:{exchange:{"bid":None,"ask":None}for exchange in exchanges}for symbol in symbols
    }
    strategy_results_queue = asyncio.Queue()
    # initialize lock for thread safety
    lock = asyncio.Lock() 
    opening_positions = 0

    asyncio.run(main())
