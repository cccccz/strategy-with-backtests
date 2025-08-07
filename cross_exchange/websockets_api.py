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
from functools import wraps
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def display_exchange_data():
        # clear_terminal()
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
    global active_trades
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

def display_trade_history():
    """Display formatted trade history showing key information"""
    global trade_history
    
    if not trade_history:
        print("No trade history available.")
        return
    
    print("\n" + "="*120)
    print("TRADE HISTORY")
    print("="*120)
    
    # Group trades by pairs (open + close)
    trade_pairs = []
    open_trades = {}
    
    for trade in trade_history:
        if trade['action'] == 'open':
            # Store open trade with timestamp as key
            open_trades[trade['time_stamp_opportunity']] = trade
        elif trade['action'] == 'close':
            # Find matching open trade and pair them
            matching_open = None
            for timestamp, open_trade in open_trades.items():
                if (open_trade['symbol'] == trade['symbol'] and 
                    open_trade['best_buy_exchange'] == trade['best_buy_exchange'] and
                    open_trade['best_sell_exchange'] == trade['best_sell_exchange']):
                    matching_open = open_trade
                    break
            
            if matching_open:
                trade_pairs.append((matching_open, trade))
                del open_trades[timestamp]
    
    # Add any remaining open trades (not yet closed)
    for open_trade in open_trades.values():
        trade_pairs.append((open_trade, None))
    
    # Display header
    print(f"{'#':<3} {'Symbol':<12} {'Buy Exchange':<12} {'Buy Price':<12} {'Sell Exchange':<12} {'Sell Price':<12} {'Open Time':<20} {'Close Time':<20} {'PnL':<15} {'Status':<10}")
    print("-" * 120)
    
    # Display each trade pair
    for i, (open_trade, close_trade) in enumerate(trade_pairs, 1):
        symbol = open_trade['symbol']
        buy_exchange = open_trade['best_buy_exchange']
        buy_price = f"{open_trade['best_buy_price']:.6f}"
        sell_exchange = open_trade['best_sell_exchange'] 
        sell_price = f"{open_trade['best_sell_price']:.6f}"
        
        # Format timestamps
        open_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(open_trade['trade_time']))
        
        if close_trade:
            close_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(close_trade['close_time']))
            pnl = f"{close_trade['pnl']:.8f}"
            status = "CLOSED"
        else:
            close_time = "OPEN"
            pnl = f"{open_trade['estimated_net_profit']:.8f}"
            status = "OPEN"
        
        print(f"{i:<3} {symbol:<12} {buy_exchange:<12} {buy_price:<12} {sell_exchange:<12} {sell_price:<12} {open_time:<20} {close_time:<20} {pnl:<15} {status:<10}")
    
    # Summary statistics
    closed_trades = [pair for pair in trade_pairs if pair[1] is not None]
    if closed_trades:
        total_pnl = sum(close_trade['pnl'] for _, close_trade in closed_trades)
        avg_pnl = total_pnl / len(closed_trades)
        profitable_trades = len([pair for pair in closed_trades if pair[1]['pnl'] > 0])
        
        print("-" * 120)
        print(f"SUMMARY: Total Trades: {len(closed_trades)} | Profitable: {profitable_trades} | Total PnL: {total_pnl:.8f} | Avg PnL: {avg_pnl:.8f}")
    
    print("=" * 120)

# Update your display_terminal function to include this:
async def display_terminal():
    while True:
        display_exchange_data()
        display_trade_history()  # Add this line
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(1)

def display_trade_history_compact():
    """Compact version showing only recent trades"""
    global trade_history
    
    if not trade_history:
        return
    
    # Show only last 5 completed trade pairs
    print(f"\nRECENT TRADES (Last 5):")
    print(f"{'Symbol':<10} {'Buy@':<15} {'Sell@':<15} {'PnL':<12} {'Status':<8}")
    print("-" * 58)
    
    # Get recent trades (both open and close)
    recent_trades = trade_history[-10:]  # Get last 10 entries
    
    for trade in recent_trades:
        symbol = trade['symbol']
        buy_info = f"{trade['best_buy_exchange'][:3]}@{trade['best_buy_price']:.4f}"
        sell_info = f"{trade['best_sell_exchange'][:3]}@{trade['best_sell_price']:.4f}"
        
        if trade['action'] == 'close':
            pnl = f"{trade['pnl']:.6f}"
            status = "CLOSED"
        else:
            pnl = f"{trade['estimated_net_profit']:.6f}"
            status = "OPENED"
        
        print(f"{symbol:<10} {buy_info:<15} {sell_info:<15} {pnl:<12} {status:<8}")
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
                shared_data[symbol]["Binance"]["bid"] = float(bid)
                shared_data[symbol]["Binance"]["ask"] = float(bid)

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
                            shared_data[symbol]["Bybit"]["bid"] = float(bid)
                        if 'ask1Price' in payload:
                            shared_data[symbol]["Bybit"]["ask"] = float(ask)
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
                    shared_data[symbol]["Bitget"]["bid"] = float(bid)
                    shared_data[symbol]["Bitget"]["ask"] = float(ask)
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
                    shared_data[symbol]["OKX"]["bid"] = float(bid)
                    shared_data[symbol]["OKX"]["ask"] = float(ask)
                # print(f"[OKX]     BTC-USDT: bid={bid} ask={ask}")

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
        # print(f"total number of data {len(snapshot.items())}")
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
    estimated_close_costs = calculate_exit_costs(best_buy[0], best_sell[0], best_sell[1], best_buy[2], trade_amount)
    total_cost = costs['total_cost'] + estimated_close_costs['total_cost']
    buy_fee = costs['buy_fee'] 
    sell_fee = costs['sell_fee']
    # slippage_cost = costs['slippage_cost']
    open_spread = float(best_sell[1] - best_buy[2])
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
            'estimated_total_cost': total_cost,
            'buy_fee': buy_fee,
            'sell_fee': sell_fee,
            'net_entry': costs['net_entry'],
            'estimated_net_profit': estimated_net_profit,
            'estimated_net_profit_pct': estimated_net_profit_pct,
            # slippage ignored
            # 'profit_pct': profit_pct,
            'time_stamp_opportunity': opportunity['time_stamp_opportunity'],
            'stop_loss': -abs(estimated_net_profit * STOP_LOSS_PCT),
             }
    return enriched

def open_position(enrich_trade):
    global opening_positions,trade_history,active_trades
    enrich_trade['trade_time'] = time.time()
    enrich_trade['action'] = 'open'
    trade_history.append(enrich_trade)
    active_trades.append(enrich_trade)
    opening_positions += 1

def calculate_open_costs(buy_exchange, sell_exchange, buy_price, sell_price, trade_amount):
    """Calculate costs to OPEN arbitrage position
    - BUY at buy_exchange's ASK price (pay ask)
    - Sell at sell_exchanges's BID price (receive bid)"""
    trade_amount = float(trade_amount)
    buy_price = float(buy_price)
    sell_price = float(sell_price)
    buy_fee = trade_amount * float(FUTURES_TRADING_FEES[buy_exchange]['taker']) * buy_price
    sell_fee = trade_amount * float(FUTURES_TRADING_FEES[sell_exchange]['taker']) * sell_price
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
    trade_amount = float(trade_amount)
    current_buy_price = float(current_buy_price)
    current_sell_price = float(current_sell_price)
    close_long_fee = trade_amount * float(FUTURES_TRADING_FEES[buy_exchange]['taker']) * current_buy_price
    close_short_fee = trade_amount * float(FUTURES_TRADING_FEES[sell_exchange]['taker']) * current_sell_price
    # slippage_cost = trade_amount * SLIPPAGE_RATE
    slippage_cost = 0

    return {
        'total_cost': close_long_fee + close_short_fee + slippage_cost,
        'close_long_cost': close_long_fee,
        'close_short_cost': close_short_fee,
        'net_exit' : current_buy_price - current_sell_price
    }

def should_open_position(enrich_trade):
    global opening_positions
    if opening_positions <= MAX_POSITION_SIZE:
        # logic about balance on those exchange...
        # TODO
        if enrich_trade['estimated_net_profit_pct'] >= MINIMUM_PROFIT_PCT:
            # and some logic about margin
            # print("we should open position")
            return True
        
    # print(f"not an opportunity:{enrich_trade}")
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
    #slippage
    entry_costs = trade['buy_fee'] + trade['sell_fee'] 
    exit_net = exit_costs['total_cost']
    unrealized_pnl = entry_net + exit_net - entry_costs - exit_net

    return {'current_spread': current_sell_price - current_buy_price,
            'unrealized_pnl': unrealized_pnl,
            'exit_costs': exit_costs,
            'position_age': position_age,
            'current_buy_price': current_buy_price,
            'current_sell_price': current_sell_price
            }

def should_close_position(trade,current_status):
    unrealized_pnl_pct = current_status['unrealized_pnl']/trade['best_buy_price'] * trade['trade_amount']
    if unrealized_pnl_pct >= trade['estimated_net_profit_pct']*0.3:
        # print("close trade because earned enough")
        return True
    # execute close position or do nothing

    
    if unrealized_pnl_pct <= STOP_LOSS_PCT:
        print("close trade because of stop loss")
        return True

    print("holding position")
    return False

def determine_exit_reason(trade,current_status):
    return "unimplemented"

def close_position(trade,current_status):
    global opening_positions, trade_history, active_trades
    trade = trade.copy()
    trade.update({
        'action': 'close',
        'close_time': time.time(),
        'pnl': current_status['unrealized_pnl'],
        'exit_reason': determine_exit_reason(trade,current_status)})
    trade_history.append(trade)
    active_trades.pop()
    opening_positions -= 1

async def execute_simulation():
    global active_trades, trade_history, opening_positions
    print("[SIMULATION] Simulation starts")

    while True:
    # await handle_new_opportunities()
        opportunity = await strategy_results_queue.get()
        # print("Evaluating the strategy from strategy_results_queue")
        enrich_trade = enrich_with_costs_and_profits(opportunity)

        async with lock:
            if should_open_position(enrich_trade):
                open_position(enrich_trade)

            if opening_positions < 0:
                raise ValueError('opening_positions cannot be negative')
            
            if opening_positions > 1:
                print("lol race condition")
            # handle active positions
            if len(active_trades) > 0:
                # this should be equivalent to opening_positions == 0
                async with lock:
                    snapshot = copy.deepcopy(shared_data)
                trade = active_trades[0]
                current_status = evaluate_active_position(trade,snapshot)
                if current_status and should_close_position(trade,current_status):
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
    MAX_POSITION_SIZE = 1   # 最大持仓限制
    AVAILABLE_BALANCE = {       # 每个交易所的可用余额
        'Binance': 10000,
        'Bybit': 10000,
        'OKX': 10000,
        'Bitget': 10000
}
    MAX_HOLDING_TIME = 300      # 最大持仓时间(秒)
    STOP_LOSS_PCT = -0.01       # 止损百分比
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
