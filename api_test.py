import asyncio
import json
import websockets

# -------- Binance --------
async def binance_ws():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            bid = data['b']
            ask = data['a']
            print(f"[Binance] BTCUSDT: bid={bid} ask={ask}")

# -------- Bybit --------
async def bybit_ws():
    uri = "wss://stream.bybit.com/v5/public/spot"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "op": "subscribe",
            "args": ["orderbook.1.BTCUSDT"]
        }
        await ws.send(json.dumps(subscribe_msg))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if 'data' in data:
                bid = data['data']['b'][0][0]
                ask = data['data']['a'][0][0]
                print(f"[Bybit]   BTCUSDT: bid={bid} ask={ask}")

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
                print(f"[Bitget]  BTCUSDT: bid={bid} ask={ask}")

# -------- OKX --------
async def okx_ws():
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(uri) as ws:
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": "BTC-USDT"
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
                print(f"[OKX]     BTC-USDT: bid={bid} ask={ask}")

# -------- 主程序，聚合运行 --------
async def main():
    await asyncio.gather(
        binance_ws(),
        bybit_ws(),
        bitget_ws(),
        okx_ws()
    )

# 启动异步主程序
asyncio.run(main())
