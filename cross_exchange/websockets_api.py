import asyncio
import json
import websockets
import os
import time
from rest_api import get_common_symbols

# initialize shared_data
exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
symbols = get_common_symbols()
#print(f"monitoring {len(symbols)} currencies") 163
shared_data = {
    symbol:{exchange:{"bid":None,"ask":None}for exchange in exchanges}for symbol in symbols
}

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
            payload = envelope.get('data', {})
            symbol = payload.get('symbol')
            bid = payload.get('bid1Price')
            ask = payload.get('ask1Price')
            if symbol in shared_data:
                shared_data[symbol]["Bybit"]["bid"] = bid
                shared_data[symbol]["Bybit"]["ask"] = ask
                # print(f"[Bybit]   BTCUSDT: bid={bid} ask={ask}")

# -------- Bitget --------
async def bitget_ws():
    uri = "wss://ws.bitget.com/v2/ws/public"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "instType": "SPOT",
                "channel": "ticker",
                "instId": "BTCUSDT"
            }]
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'data' in data and len(data['data']) > 0:
                item = data['data'][0]
                bid = item['bidPr']
                ask = item['askPr']
                shared_data["Bitget"]["BTCUSDT"]["bid"] = bid
                shared_data["Bitget"]["BTCUSDT"]["ask"] = ask
                # print(f"[Bitget]  BTCUSDT: bid={bid} ask={ask}")

# -------- OKX --------
async def okx_ws():
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": "BTC-USDT-SWAP"
            }]
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'data' in data and len(data['data']) > 0:
                item = data['data'][0]
                bid = item['bidPx']
                ask = item['askPx']
                shared_data["OKX"]["BTC-USDT-SWAP"]["bid"] = bid
                shared_data["OKX"]["BTC-USDT-SWAP"]["ask"] = ask
                # print(f"[OKX]     BTC-USDT: bid={bid} ask={ask}")

# -------- 主程序，聚合运行 --------
async def main():
    await asyncio.gather(
        # binance_ws(),
        bybit_ws(),
        # bitget_ws(),
        # okx_ws(),
        display_terminal()
    )

if __name__ == '__main__':  
# 启动异步主程序
    asyncio.run(main())
