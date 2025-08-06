import asyncio
import functools
import json
from numpy import copy
import websockets
import os
import time
from rest_api import get_common_symbols
import pandas as pd
from utils import async_timeit, timeit, log_duration
from decimal import Decimal, getcontext

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
async def display_terminal():
    while True:
        # display_exchange_data()

        if not strategy_results_queue.empty():
            strategy_result = await strategy_results_queue.get()
            print(f"Symbol: {strategy_result['symbol']}")
            print(f"Best Buy on: {strategy_result['best_buy']}, {strategy_result['best_buy_price']}")
            print(f"Best Sell on: {strategy_result['best_sell']}, {strategy_result['best_sell_price']}")
            print(f"Open Spread: {strategy_result['open_spread']*1:.4f}")
            print(f"Open Spread Percentage: {strategy_result['open_spread_pct']*100:.2f}%")
            print(f"Close Spread Percentage: {strategy_result['close_spread_pct']*100:.2f}%")

            print(f"Time Stamp: {strategy_result['time_stamp']}")
            print("-" * 30)

        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(1)


# -------- Binance --------
@async_timeit
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

# # -------- Bybit --------
# @async_timeit
# async def bybit_ws():
#     uri = "wss://stream.bybit.com/v5/public/linear"
#     async with websockets.connect(uri) as ws:
#         params = [f"tickers.{symbol}" for symbol in symbols]
#         subscribe_msg = {
#             "op": "subscribe",
#             "args": params
#         }
#         await ws.send(json.dumps(subscribe_msg))

#         while True:
#             msg = await ws.recv()
#             envelope = json.loads(msg)
#             payload = envelope['data']
#             symbol = payload.get('symbol')
#             bid = payload.get('bid1Price')
#             ask = payload.get('ask1Price')
#             if symbol in shared_data:
#                 shared_data[symbol]["Bybit"]["bid"] = Decimal(bid)
#                 shared_data[symbol]["Bybit"]["ask"] = Decimal(ask)
#                 # print(f"[Bybit]   BTCUSDT: bid={bid} ask={ask}")
@async_timeit
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

async def compute_strategy():
    """
    compute_strategy only runs every COMPUTE_INTERVAL seconds
    bid 买价 ask 卖价
    """
    
    compute_interval = COMPUTE_INTERVAL
    minimum_profit_pct = MINIMUM_PROFIT_PCT


    while True:
        async with lock: 
            # snapshot = copy.deepcopy(shared_data)
            snapshot = shared_data.copy()

        best_strategy = None
        print(f"total number of data {len(snapshot.items())}")

        for symbol, exchange_data in snapshot.items():
            quotes = []
            for exchange, data in exchange_data.items():
                bid, ask = data['bid'], data['ask']
                if bid and ask:
                    quotes.append((exchange, float(bid), float(ask)))

            if len(quotes) < 2:
                continue

            # 在最便宜的卖家 买入
            # 在出价最高的买家 卖出
            best_buy = min(quotes, key=lambda x: x[2])  # ask
            best_sell = max(quotes, key=lambda x: x[1])  # bid

            open_spread = best_sell[1] - best_buy[2]
            open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
            close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

            # profit_pct = spread / best_buy[2]

            # cost_pct = estimate_cost(best_buy[0], best_sell[0], symbol)
            cost_pct = 0
            strategy_result = {
                'symbol': symbol,
                'best_buy': best_buy[0],
                'best_buy_price': best_buy[2],
                'best_sell': best_sell[0],
                'best_sell_price': best_sell[1],
                'open_spread': open_spread,
                'open_spread_pct': open_spread_pct,
                'close_spread_pct': close_spread_pct,
                # 'profit_pct': profit_pct,
                'time_stamp': time.time()}
            
            if best_strategy is None or open_spread_pct > best_strategy['open_spread_pct']:
                best_strategy = strategy_result

        if best_strategy:

            await strategy_results_queue.put(best_strategy)
                # print(f"[ARBITRAGE] {symbol}: Buy on {best_buy[0]} @ {best_buy[2]}, "
                #       f"Sell on {best_sell[0]} @ {best_sell[1]} "
                #       f"=> Profit: {profit_pct:.4%} > Cost: {cost_pct:.4%}")
                # simulate_trade(...)
        await asyncio.sleep(compute_interval)

def estimate_cost():
    return 0

async def execute_simulation():
    print('making money')

async def main():
    await asyncio.gather(
        binance_ws(),
        bitget_ws(),
        okx_ws(),
        bybit_ws(),
        compute_strategy(),
        # execute_simulation(),
        display_terminal(),
    )

if __name__ == '__main__':  
    COMPUTE_INTERVAL = 1
    MINIMUM_PROFIT_PCT = 0
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

    asyncio.run(main())
