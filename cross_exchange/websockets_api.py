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
# initialize shared_data
exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
symbols = get_common_symbols()
okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in symbols]
#print(f"monitoring {len(symbols)} currencies") 163
shared_data = {
    symbol:{exchange:{"bid":None,"ask":None}for exchange in exchanges}for symbol in symbols
}

# initialize lock for thread safety
lock = asyncio.Lock() 


# -------- Terminal 清屏函数 --------
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

async def display_terminal():
    while True:
        clear_terminal()
        print("+------------+-----------+-------------+-------------+")
        print("| Exchange   | Symbol    | Bid Price   | Ask Price   |")
        print("+------------+-----------+-------------+-------------+")

        for exchange, symbols in shared_data.items():
            
            for symbol, data in symbols.items():
                # print(symbol, exchange, data)

                bid = data['bid'] if data['bid'] else "..."
                ask = data['ask'] if data['ask'] else "..."
                print(f"| {exchange:<10} | {symbol:<9} | {str(bid):<11} | {str(ask):<11} |")

        print("+------------+-----------+-------------+-------------+")
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
                shared_data[symbol]["Binance"]["bid"] = bid
                shared_data[symbol]["Binance"]["ask"] = ask

            # print(f"[Binance] BTCUSDT: bid={bid} ask={ask}")

# -------- Bybit --------
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
            msg = await ws.recv()
            envelope = json.loads(msg)
            payload = envelope['data']
            symbol = payload.get('symbol')
            bid = payload.get('bid1Price')
            ask = payload.get('ask1Price')
            if symbol in shared_data:
                shared_data[symbol]["Bybit"]["bid"] = bid
                shared_data[symbol]["Bybit"]["ask"] = ask
                # print(f"[Bybit]   BTCUSDT: bid={bid} ask={ask}")

# -------- Corrected Bybit WebSocket --------
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
                            shared_data[symbol]["Bybit"]["bid"] = bid
                        if 'ask1Price' in payload:
                            shared_data[symbol]["Bybit"]["ask"] = ask
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
                bid = item['bidPr']
                ask = item['askPr']
                symbol = item['instId']
                if symbol in shared_data:
                    shared_data[symbol]["Bitget"]["bid"] = bid
                    shared_data[symbol]["Bitget"]["ask"] = ask
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
                bid = pd.to_numeric(item['bidPx'])
                ask = pd.to_numeric(item['askPx'])
                symbol = item['instId'].replace('-SWAP','').replace('-','')
                if symbol in shared_data:
                    shared_data[symbol]["OKX"]["bid"] = bid
                    shared_data[symbol]["OKX"]["ask"] = ask
                # print(f"[OKX]     BTC-USDT: bid={bid} ask={ask}")

async def compute_strategy():
    while True:
        async with lock:  # to safely copy shared_data
            snapshot = copy.deepcopy(shared_data)

        for symbol, exchange_data in snapshot.items():
            quotes = []
            for exchange, data in exchange_data.items():
                bid, ask = data['bid'], data['ask']
                if bid and ask:
                    quotes.append((exchange, float(bid), float(ask)))

            if len(quotes) < 2:
                continue

            best_buy = min(quotes, key=lambda x: x[2])  # ask
            best_sell = max(quotes, key=lambda x: x[1])  # bid

            spread = best_sell[1] - best_buy[2]
            profit_pct = spread / best_buy[2]

            cost_pct = estimate_cost(best_buy[0], best_sell[0], symbol)

            if profit_pct > cost_pct:
                print(f"[ARBITRAGE] {symbol}: Buy on {best_buy[0]} @ {best_buy[2]}, "
                      f"Sell on {best_sell[0]} @ {best_sell[1]} "
                      f"=> Profit: {profit_pct:.4%} > Cost: {cost_pct:.4%}")
                # simulate_trade(...)
        await asyncio.sleep(1)

def estimate_cost():
    return 0

async def execute_simulation():
    print('making money')

# -------- 主程序，聚合运行 --------
async def main():
    await asyncio.gather(
        binance_ws(),
        bitget_ws(),
        okx_ws(),
        bybit_ws(),
        display_terminal(),
        compute_strategy(),
        execute_simulation(),
    )

if __name__ == '__main__':  
# 启动异步主程序
    asyncio.run(main())
