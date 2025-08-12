from shared_imports import websockets,asyncio, json,ConnectionClosedError,ConnectionClosedOK,Config
from trading_state import TradingState
# 通用WebSocket连接器
async def generic_ws_connector(exchange_name, ws_config, state:TradingState):
    """通用WebSocket连接器，减少重复代码"""
    reconnect_count = 0
    
    while reconnect_count < Config.MAX_RECONNECT_ATTEMPTS:
        try:
            print(f"[{exchange_name}] Connecting... (attempt {reconnect_count + 1})")
            
            async with websockets.connect(
                ws_config['uri'], 
                ping_interval=20, 
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                # 发送订阅消息
                if 'subscribe_msg' in ws_config:
                    await ws.send(json.dumps(ws_config['subscribe_msg']))
                
                print(f"[{exchange_name}] Connected successfully")
                reconnect_count = 0
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        # 使用解析函数处理数据
                        parsed = ws_config['parser'](data)
                        if parsed:
                            symbol, bid, ask = parsed
                            async with state.lock:
                                if symbol in state.shared_data:
                                    state.shared_data[symbol][exchange_name]["bid"] = bid
                                    state.shared_data[symbol][exchange_name]["ask"] = ask
                                    
                    except asyncio.TimeoutError:
                        print(f"[{exchange_name}] No data received for 30s, checking connection...")
                        await ws.ping()
                        continue
                    except json.JSONDecodeError as e:
                        print(f"[{exchange_name}] JSON decode error: {e}")
                        continue
                        
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[{exchange_name}] Connection closed: {e}")
        except Exception as e:
            print(f"[{exchange_name}] Error: {e}")
        
        reconnect_count += 1
        if reconnect_count < Config.MAX_RECONNECT_ATTEMPTS:
            print(f"[{exchange_name}] Reconnecting in {Config.RECONNECT_DELAY}s...")
            await asyncio.sleep(Config.RECONNECT_DELAY)
        else:
            print(f"[{exchange_name}] Max reconnection attempts reached")
            break

# 各交易所的解析器
def parse_binance(data):
    if 'data' in data:
        payload = data['data']
        symbol = payload.get('s')
        bid = payload.get('b')
        ask = payload.get('a')
        if symbol and bid and ask:
            return symbol, float(bid), float(ask)
    return None

def parse_bybit(data):
    if 'data' in data and data.get('topic', '').startswith('tickers.'):
        payload = data['data']
        symbol = payload.get('symbol')
        bid = payload.get('bid1Price')
        ask = payload.get('ask1Price')
        if symbol and bid and ask:
            return symbol, float(bid), float(ask)
    return None

def parse_bitget(data):
    if 'data' in data and len(data['data']) > 0:
        item = data['data'][0]
        bid = item.get('bidPr')
        ask = item.get('askPr')
        symbol = item.get('instId')
        if bid and ask and symbol:
            return symbol, float(bid), float(ask)
    return None

def parse_okx(data):
    if 'data' in data and len(data['data']) > 0:
        item = data['data'][0]
        bid = item.get('bidPx')
        ask = item.get('askPx')
        symbol = item.get('instId', '').replace('-SWAP','').replace('-','')
        if bid and ask and symbol:
            return symbol, float(bid), float(ask)
    return None

# 各交易所连接函数
async def binance_ws(state:TradingState):
    params = [f"{s.lower()}@bookTicker" for s in state.symbols]
    uri = f"wss://fstream.binance.com/stream?streams=" + "/".join(params)
    config = {
        'uri': uri,
        'parser': parse_binance
    }
    await generic_ws_connector('Binance', config, state)

async def bybit_ws(state:TradingState):
    config = {
        'uri': "wss://stream.bybit.com/v5/public/linear",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [f"tickers.{symbol}" for symbol in state.symbols]
        },
        'parser': parse_bybit
    }
    await generic_ws_connector('Bybit', config, state)

async def bitget_ws(state:TradingState):
    config = {
        'uri': "wss://ws.bitget.com/v2/ws/public",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [{
                "instType": "USDT-FUTURES",
                "channel": "ticker",
                "instId": symbol
            } for symbol in state.symbols]
        },
        'parser': parse_bitget
    }
    await generic_ws_connector('Bitget', config, state)

async def okx_ws(state:TradingState):
    config = {
        'uri': "wss://ws.okx.com:8443/ws/v5/public",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": symbol
            } for symbol in state.okx_symbols]
        },
        'parser': parse_okx
    }
    await generic_ws_connector('OKX', config, state)

