import asyncio
import websockets
import json
from datetime import datetime

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

# 各交易所的解析器
def parse_binance_orderbook(data):
    """解析Binance orderbook数据 - 20档"""
    if 'stream' in data and 'data' in data:
        stream_name = data['stream']
        if 'depth' in stream_name:
            payload = data['data']
            symbol = stream_name.split('@')[0].upper()
            
            raw_bids = payload.get('b', payload.get('bids', []))
            raw_asks = payload.get('a', payload.get('asks', []))
            
            if raw_bids and raw_asks:
                bids = [(float(price), float(qty)) for price, qty in raw_bids]
                asks = [(float(price), float(qty)) for price, qty in raw_asks]
                # bids.sort(key=lambda x: x[0], reverse=True)
                # asks.sort(key=lambda x: x[0])
                return symbol, bids, asks
    return None

def parse_bybit_orderbook(data):
    """解析Bybit orderbook数据 - 50档"""
    if 'data' in data and 'topic' in data and data['topic'].startswith('orderbook.50.'):
        payload = data['data']
        symbol = payload.get('s', 'UNKNOWN')
        
        raw_bids = payload.get('b', [])
        raw_asks = payload.get('a', [])
        
        if raw_bids and raw_asks:
            bids = [(float(price), float(qty)) for price, qty in raw_bids]
            asks = [(float(price), float(qty)) for price, qty in raw_asks]
            # bids.sort(key=lambda x: x[0], reverse=True)
            # asks.sort(key=lambda x: x[0])
            return symbol, bids, asks
    return None

def parse_bitget_orderbook(data):
    """解析Bitget orderbook数据 - 完整深度"""
    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
        payload = data['data'][0]
        symbol = data['arg'].get('instId', 'UNKNOWN')
        
        raw_bids = payload.get('bids', [])
        raw_asks = payload.get('asks', [])
        
        if raw_bids and raw_asks:
            bids = [(float(price), float(qty)) for price, qty in raw_bids]
            asks = [(float(price), float(qty)) for price, qty in raw_asks]
            # bids.sort(key=lambda x: x[0], reverse=True)
            # asks.sort(key=lambda x: x[0])
            return symbol, bids, asks
    return None

def parse_okx_orderbook(data):
    """解析OKX orderbook数据 - 15档，格式: ["price", "quantity", "0", "order_count"]"""
    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
        payload = data['data'][0]
        okx_symbol = data['arg'].get('instId', 'UNKNOWN')
        symbol = okx_symbol.replace('-SWAP', '').replace('-', '')
        
        raw_bids = payload.get('bids', [])
        raw_asks = payload.get('asks', [])
        
        if raw_bids and raw_asks:
            bids = [(float(item[0]), float(item[1])) for item in raw_bids if len(item) >= 2]
            asks = [(float(item[0]), float(item[1])) for item in raw_asks if len(item) >= 2]
            # bids.sort(key=lambda x: x[0], reverse=True)
            # asks.sort(key=lambda x: x[0])
            return symbol, bids, asks
    return None

# 通用WebSocket连接器 - 专门用于orderbook
async def generic_orderbook_connector(exchange_name, ws_config, symbols):
    """通用orderbook WebSocket连接器"""
    try:
        print(f"[{exchange_name}] Orderbook Connecting...")
        print(f"[{exchange_name}] Symbols: {', '.join(symbols)}")
        
        async with websockets.connect(
            ws_config['uri'], 
            ping_interval=20, 
            ping_timeout=10,
            close_timeout=10
        ) as ws:
            # 发送订阅消息
            if 'subscribe_msg' in ws_config:
                await ws.send(json.dumps(ws_config['subscribe_msg']))
            
            print(f"[{exchange_name}] Orderbook Connected successfully\n")
            
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    data = json.loads(msg)
                    
                    # 使用解析函数处理orderbook数据
                    parsed = ws_config['parser'](data)
                    if parsed:
                        symbol, bids, asks = parsed
                        print_orderbook(symbol, bids, asks)
                        
                except asyncio.TimeoutError:
                    print(f"[{exchange_name}] Orderbook: No data for 30s, checking connection...")
                    await ws.ping()
                    continue
                except json.JSONDecodeError as e:
                    print(f"[{exchange_name}] Orderbook JSON decode error: {e}")
                    continue
                    
    except Exception as e:
        print(f"[{exchange_name}] Orderbook error: {e}")

# 各交易所连接函数
async def binance_orderbook_ws(symbols):
    """连接Binance获取多个交易对的20档orderbook"""
    params = [f"{s.lower()}@depth20@100ms" for s in symbols]
    uri = f"wss://fstream.binance.com/stream?streams=" + "/".join(params)
    config = {
        'uri': uri,
        'parser': parse_binance_orderbook
    }
    await generic_orderbook_connector('Binance', config, symbols)

async def bybit_orderbook_ws(symbols):
    """连接Bybit获取多个交易对的50档orderbook"""
    config = {
        'uri': "wss://stream.bybit.com/v5/public/linear",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [f"orderbook.50.{symbol}" for symbol in symbols]
        },
        'parser': parse_bybit_orderbook
    }
    await generic_orderbook_connector('Bybit', config, symbols)

async def bitget_orderbook_ws(symbols):
    """连接Bitget获取多个交易对的15档orderbook（15档）"""
    config = {
        'uri': "wss://ws.bitget.com/v2/ws/public",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [{
                "instType": "USDT-FUTURES",
                "channel": "books15",
                "instId": symbol
            } for symbol in symbols]
        },
        'parser': parse_bitget_orderbook
    }
    await generic_orderbook_connector('Bitget', config, symbols)

async def okx_orderbook_ws(symbols):
    """连接OKX获取多个交易对的15档orderbook"""
    # 转换为OKX格式的交易对名称
    okx_symbols = [f"{symbol.replace('USDT', '')}-USDT-SWAP" for symbol in symbols]
    
    config = {
        'uri': "wss://ws.okx.com:8443/ws/v5/public",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [{
                "channel": "books",
                "instId": okx_symbol,
                "sz":"15"
            } for okx_symbol in okx_symbols]
        },
        'parser': parse_okx_orderbook
    }
    await generic_orderbook_connector('OKX', config, symbols)

def get_common_symbols():
    """获取常见的交易对列表"""
    return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'BNBUSDT']

if __name__ == "__main__":
    print("多交易对 Orderbook 测试")
    print("按 Ctrl+C 停止")
    
    # 获取要监控的交易对
    symbols = get_common_symbols()
    print(f"监控交易对: {', '.join(symbols)}\n")
    
    try:
        # 选择要测试的交易所，注释掉不需要的
        # asyncio.run(binance_orderbook_ws(symbols))       # Binance - 20档
        # asyncio.run(bybit_orderbook_ws(symbols))       # Bybit - 50档  
        # asyncio.run(bitget_orderbook_ws(symbols))      # Bitget - 15档
        asyncio.run(okx_orderbook_ws(symbols))         # OKX - 15档
    except KeyboardInterrupt:
        print("\n程序已停止")