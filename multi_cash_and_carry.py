from basic_cash_and_carry_cleaned import BinanceArbitrageBacktester
from enhanced_cash_and_carry import plot_futures_and_spot_data, plot_futures_and_spot_data_manual

def get_eth_data():
    eth_backtester = BinanceArbitrageBacktester('ETHUSDT','4h')

    # eth_backtester.get_spot_data(4)
    # eth_backtester.save_spot_data('4_year_eth_spot_data.pkl')
    # eth_backtester.get_all_futures_data('2022-01-01')
    # eth_backtester.save_futures_data('4_year_eth_futures_data.pkl')

    eth_backtester.load_spot_data('4_year_eth_spot_data.pkl')
    eth_backtester.load_futures_data('4_year_eth_futures_data.pkl')

    print(eth_backtester.spot_data.head(5))
    start_date = '2025-03-01 00:00:00'
    end_date = '2025-03-31 00:00:00'
    filtered_spot= eth_backtester.spot_data.loc[start_date:end_date]
    filtered_futures = eth_backtester.futures_data[start_date:end_date]   
    print(filtered_futures)
    print(filtered_spot)
    plot_futures_and_spot_data_manual(eth_backtester,filtered_spot,filtered_futures, 'eth_analysis_zoomed.png')

import asyncio
import json
import websockets

# 交易所 WebSocket 地址和订阅信息
EXCHANGES = {
    "binance": {
        "url": "wss://stream.binance.com:9443/ws/btcusdt@trade",
        "parser": lambda msg: float(json.loads(msg)['p']),
    },
    "okx": {
        "url": "wss://ws.okx.com:8443/ws/v5/public",
        "subscribe": {
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": "BTC-USDT-SWAP"}]
        },
        "parser": lambda msg: float(json.loads(msg)['data'][0]['last']),
    },
    "bitget": {
        "url": "wss://ws.bitget.com/mix/v1/stream",
        "subscribe": {
            "op": "subscribe",
            "args": [{"instType": "swap", "channel": "trade", "instId": "BTCUSDT_UMCBL"}]
        },
        "parser": lambda msg: float(json.loads(msg)['data'][0]['price']),
    },
    "bybit": {
        "url": "wss://stream.bybit.com/v5/public/linear",
        "subscribe": {
            "op": "subscribe",
            "args": ["publicTrade.BTCUSDT"]
        },
        "parser": lambda msg: float(json.loads(msg)['data'][0]['p']),
    },
}

async def connect_exchange(name, config):
    print(f"[{name}] Connecting to {config['url']}")
    async with websockets.connect(config["url"]) as ws:
        if "subscribe" in config:
            await ws.send(json.dumps(config["subscribe"]))

        while True:
            try:
                msg = await ws.recv()
                price = config["parser"](msg)
                print(f"[{name}] Price: {price}")
            except Exception as e:
                print(f"[{name}] Error: {e}")
                continue

async def main():
    tasks = [connect_exchange(name, cfg) for name, cfg in EXCHANGES.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
# 全量比较