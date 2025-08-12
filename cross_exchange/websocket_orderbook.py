import asyncio
import websockets
import json
from datetime import datetime
from shared_imports import get_common_symbols

def print_orderbook(symbol, bids, asks):
    """打印orderbook数据"""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n=== {symbol} Orderbook at {current_time} ===")
    
    # 显示前5档asks (卖单)
    print("ASKS (卖单):")
    for i, (price, qty) in enumerate(asks[:5]):
        print(f"  {price:>10.2f}  |  {qty:>10.6f}")
    
    # 价差
    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread = best_ask - best_bid
    
    print(f"  {'='*25}")
    print(f"  价差: {spread:.2f} USDT")
    print(f"  {'='*25}")
    
    # 显示前5档bids (买单)
    print("BIDS (买单):")
    for i, (price, qty) in enumerate(bids[:5]):
        print(f"  {price:>10.2f}  |  {qty:>10.6f}")
    print("-" * 40)

async def binance_orderbook():
    """连接Binance获取BTCUSDT的20档orderbook"""
    uri = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms"
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to Binance orderbook stream...")
            print("Listening for BTCUSDT orderbook data...\n")
            
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if 'data' in data:
                        payload = data['data']
                        symbol = "BTCUSDT"
                        
                        # 解析bids和asks
                        raw_bids = payload.get('b', [])
                        raw_asks = payload.get('a', [])
                        
                        if raw_bids and raw_asks:
                            bids = [(float(price), float(qty)) for price, qty in raw_bids]
                            asks = [(float(price), float(qty)) for price, qty in raw_asks]
                            
                            # 打印数据
                            print_orderbook(symbol, bids, asks)
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                except Exception as e:
                    print(f"处理数据错误: {e}")
                    
    except Exception as e:
        print(f"连接错误: {e}")


async def bybit_orderbook():
    """连接Bybit获取BTCUSDT的50档orderbook"""
    uri = "wss://stream.bybit.com/v5/public/linear"
    
    try:
        async with websockets.connect(uri) as ws:
            # 发送订阅消息 - 50档深度
            subscribe_msg = {
                "op": "subscribe",
                "args": ["orderbook.50.BTCUSDT"]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            print("Connected to Bybit orderbook stream...")
            print("Listening for BTCUSDT orderbook data (50 levels)...\n")
            
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if 'data' in data and data.get('topic') == 'orderbook.50.BTCUSDT':
                        payload = data['data']
                        symbol = payload.get('s', 'BTCUSDT')
                        
                        # 解析bids和asks
                        raw_bids = payload.get('b', [])
                        raw_asks = payload.get('a', [])
                        
                        if raw_bids and raw_asks:
                            bids = [(float(price), float(qty)) for price, qty in raw_bids]
                            asks = [(float(price), float(qty)) for price, qty in raw_asks]
                            
                            # 按价格排序
                            bids.sort(key=lambda x: x[0], reverse=True)  # 降序
                            asks.sort(key=lambda x: x[0])  # 升序
                            
                            # 打印数据
                            print_orderbook(symbol, bids, asks)
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                except Exception as e:
                    print(f"处理数据错误: {e}")
                    
    except Exception as e:
        print(f"连接错误: {e}")

async def bitget_orderbook():
    """连接Bitget获取BTCUSDT的orderbook（约15档）"""
    uri = "wss://ws.bitget.com/v2/ws/public"
    
    try:
        async with websockets.connect(uri) as ws:
            # 发送订阅消息 - 完整订单簿
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "instType": "USDT-FUTURES",
                    "channel": "books15",
                    "instId": "BTCUSDT"
                }]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            print("Connected to Bitget orderbook stream...")
            print("Listening for BTCUSDT orderbook data (full depth)...\n")
            
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        payload = data['data'][0]
                        symbol = payload.get('instId', 'BTCUSDT')
                        
                        # 解析bids和asks
                        raw_bids = payload.get('bids', [])
                        raw_asks = payload.get('asks', [])
                        
                        if raw_bids and raw_asks:
                            bids = [(float(price), float(qty)) for price, qty in raw_bids]
                            asks = [(float(price), float(qty)) for price, qty in raw_asks]
                            
                            # 按价格排序
                            bids.sort(key=lambda x: x[0], reverse=True)  # 降序
                            asks.sort(key=lambda x: x[0])  # 升序
                            
                            # 打印数据
                            print_orderbook(symbol, bids, asks)
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                except Exception as e:
                    print(f"处理数据错误: {e}")
                    
    except Exception as e:
        print(f"连接错误: {e}")

async def okx_orderbook():
    """连接OKX获取BTC-USDT-SWAP的15档orderbook
     An example of the array of asks and bids values: ["411.8", "10", "0", "4"]
    - "411.8" is the depth price
    - "10" is the quantity at the price (number of contracts for derivatives, quantity in base currency for Spot and Spot Margin)
    - "0" is part of a deprecated feature and it is always "0"
    - "4" is the number of orders at the price.
    """
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    
    try:
        async with websockets.connect(uri) as ws:
            # 发送订阅消息 - 15档深度
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "books",
                    "instId": "BTC-USDT-SWAP",
                    "sz": "15"
                }]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            print("Connected to OKX orderbook stream...")
            print("Listening for BTC-USDT-SWAP orderbook data (400 levels)...\n")
            
            async for message in ws:
                try:
                    data = json.loads(message)
                    
                    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        payload = data['data'][0]
                        symbol = payload.get('instId', 'BTC-USDT-SWAP').replace('-SWAP', '').replace('-', '')
                        
                        # 解析bids和asks
                        raw_bids = payload.get('bids', [])
                        raw_asks = payload.get('asks', [])
                        
                        if raw_bids and raw_asks:
                            bids = [(float(raw_bid[0]), float(raw_bid[1])) for raw_bid in raw_bids if len(raw_bids) >= 2]
                            asks = [(float(raw_ask[0]), float(raw_ask[1])) for raw_ask in raw_asks if len(raw_ask) >= 2]
                            
                            # 按价格排序
                            bids.sort(key=lambda x: x[0], reverse=True)  # 降序
                            asks.sort(key=lambda x: x[0])  # 升序
                            
                            # 打印数据
                            print_orderbook(symbol, bids, asks)
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                except Exception as e:
                    print(f"处理数据错误: {e}")
                    
    except Exception as e:
        print(f"连接错误: {e}")


if __name__ == "__main__":
    print("Binance BTCUSDT Orderbook 测试")
    print("按 Ctrl+C 停止")
    symbols = get_common_symbols()
    try:
        # asyncio.run(binance_orderbook())
        # asyncio.run(bybit_orderbook())
        # asyncio.run(bitget_orderbook())
        asyncio.run(okx_orderbook())

    except KeyboardInterrupt:
        print("\n程序已停止")