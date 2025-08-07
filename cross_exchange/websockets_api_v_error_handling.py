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
import logging
from datetime import datetime

# 设置日志格式
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 1. 配置决策日志（decision_log）
decision_logger = logging.getLogger('decision_logger')
decision_logger.setLevel(logging.INFO)  # 可以根据需要设置日志级别，如DEBUG, INFO, ERROR等

decision_handler = logging.FileHandler('decision_log.txt')
decision_handler.setFormatter(logging.Formatter(log_format))

decision_logger.addHandler(decision_handler)

# 2. 配置输出日志（output_log）
output_logger = logging.getLogger('output_logger')
output_logger.setLevel(logging.INFO)

output_handler = logging.FileHandler('output_log.txt')
output_handler.setFormatter(logging.Formatter(log_format))

output_logger.addHandler(output_handler)
decision_id = 0
decision_id_lock = asyncio.Lock()

async def get_next_decision_id():
    global decision_id
    async with decision_id_lock:
        current_id = decision_id
        decision_id += 1
        return current_id


lock = asyncio.Lock() 

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
    """Display formatted trade history showing key information with current prices for open positions"""
    global trade_history, shared_data
    
    if not trade_history:
        print("No trade history available.")
        return
    
    print("\n" + "="*140)
    print("TRADE HISTORY")
    print("="*140)
    
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
    print(f"{'#':<3} {'Symbol':<12} {'Buy@':<20} {'Sell@':<20} {'Open Time':<20} {'Close Time':<20} {'PnL':<15} {'Status':<10}")
    print("-" * 140)
    
    # Display each trade pair
    for i, (open_trade, close_trade) in enumerate(trade_pairs, 1):
        symbol = open_trade['symbol']
        buy_exchange = open_trade['best_buy_exchange']
        sell_exchange = open_trade['best_sell_exchange']
        
        # Format timestamps
        open_time = time.strftime('%m-%d %H:%M:%S', time.localtime(open_trade['trade_time']))
        
        if close_trade:
            # Closed trade - show original prices
            buy_info = f"{buy_exchange[:3]}@{open_trade['best_buy_price']:.6f}"
            sell_info = f"{sell_exchange[:3]}@{open_trade['best_sell_price']:.6f}"
            close_time = time.strftime('%m-%d %H:%M:%S', time.localtime(close_trade['close_time']))
            pnl = f"{close_trade['pnl']:.8f}"
            status = "CLOSED"
        else:
            # Open trade - show original/current prices
            original_buy = open_trade['best_buy_price']
            original_sell = open_trade['best_sell_price']
            
            # Get current prices
            current_buy_price = "N/A"
            current_sell_price = "N/A"
            
            if symbol in shared_data:
                if shared_data[symbol][buy_exchange]['ask']:
                    current_buy_price = f"{shared_data[symbol][buy_exchange]['ask']:.6f}"
                if shared_data[symbol][sell_exchange]['bid']:
                    current_sell_price = f"{shared_data[symbol][sell_exchange]['bid']:.6f}"
            
            buy_info = f"{buy_exchange[:3]}@{original_buy:.6f}/{current_buy_price}"
            sell_info = f"{sell_exchange[:3]}@{original_sell:.6f}/{current_sell_price}"
            close_time = "OPEN"
            pnl = f"{open_trade['estimated_net_profit']:.8f}"
            status = "OPEN"
        
        print(f"{i:<3} {symbol:<12} {buy_info:<20} {sell_info:<20} {open_time:<20} {close_time:<20} {pnl:<15} {status:<10}")
    
    # Summary statistics
    closed_trades = [pair for pair in trade_pairs if pair[1] is not None]
    if closed_trades:
        total_pnl = sum(close_trade['pnl'] for _, close_trade in closed_trades)
        avg_pnl = total_pnl / len(closed_trades)
        profitable_trades = len([pair for pair in closed_trades if pair[1]['pnl'] > 0])
        
        print("-" * 140)
        print(f"SUMMARY: Total Trades: {len(closed_trades)} | Profitable: {profitable_trades} | Total PnL: {total_pnl:.8f} | Avg PnL: {avg_pnl:.8f}")
    
    print("=" * 140)

# 同样更新 compact 版本
def display_trade_history_compact():
    """Compact version showing only recent trades with current prices"""
    global trade_history, shared_data
    
    if not trade_history:
        return
    
    print(f"\nRECENT TRADES (Last 5):")
    print(f"{'Symbol':<10} {'Buy@':<20} {'Sell@':<20} {'PnL':<12} {'Status':<8}")
    print("-" * 75)
    
    # Get recent trades (both open and close)
    recent_trades = trade_history[-5:]
    
    for trade in recent_trades:
        symbol = trade['symbol']
        buy_exchange = trade['best_buy_exchange']
        sell_exchange = trade['best_sell_exchange']
        
        if trade['action'] == 'close':
            # Closed trade
            buy_info = f"{buy_exchange[:3]}@{trade['best_buy_price']:.4f}"
            sell_info = f"{sell_exchange[:3]}@{trade['best_sell_price']:.4f}"
            pnl = f"{trade['pnl']:.6f}"
            status = "CLOSED"
        else:
            # Open trade - show current prices
            original_buy = trade['best_buy_price']
            original_sell = trade['best_sell_price']
            
            current_buy = "N/A"
            current_sell = "N/A"
            if symbol in shared_data:
                if shared_data[symbol][buy_exchange]['ask']:
                    current_buy = f"{shared_data[symbol][buy_exchange]['ask']:.4f}"
                if shared_data[symbol][sell_exchange]['bid']:
                    current_sell = f"{shared_data[symbol][sell_exchange]['bid']:.4f}"
            
            buy_info = f"{buy_exchange[:3]}@{original_buy:.4f}/{current_buy}"
            sell_info = f"{sell_exchange[:3]}@{original_sell:.4f}/{current_sell}"
            pnl = f"{trade['estimated_net_profit']:.6f}"
            status = "OPENED"
        
        print(f"{symbol:<10} {buy_info:<20} {sell_info:<20} {pnl:<12} {status:<8}")

# Update your display_terminal function to include this:
async def display_terminal():
    while True:
        # display_exchange_data()
        display_trade_history()  # Add this line
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(1)


import asyncio
import json
import websockets
import time
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatusCode

# Add these constants at the top of your file
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 5  # seconds
CONNECTION_TIMEOUT = 10  # seconds

async def binance_ws():
    """Binance WebSocket with error handling and reconnection"""
    reconnect_count = 0
    
    while reconnect_count < MAX_RECONNECT_ATTEMPTS:
        try:
            params = [f"{s.lower()}@bookTicker" for s in symbols]
            uri = f"wss://fstream.binance.com/stream?streams=" + "/".join(params)
            
            print(f"[Binance] Connecting... (attempt {reconnect_count + 1})")
            
            async with websockets.connect(
                uri, 
                ping_interval=20, 
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                print(f"[Binance] Connected successfully")
                reconnect_count = 0  # Reset counter on successful connection
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        envelope = json.loads(msg)
                        payload = envelope.get('data', {})
                        symbol = payload.get('s')
                        bid = payload.get('b')
                        ask = payload.get('a')
                        
                        if symbol and bid and ask and symbol in shared_data:
                            shared_data[symbol]["Binance"]["bid"] = float(bid)
                            shared_data[symbol]["Binance"]["ask"] = float(ask)  # Fixed: was using bid instead of ask
                            
                    except asyncio.TimeoutError:
                        print(f"[Binance] No data received for 30s, checking connection...")
                        await ws.ping()
                        continue
                    except json.JSONDecodeError as e:
                        print(f"[Binance] JSON decode error: {e}")
                        continue
                        
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[Binance] Connection closed: {e}")
        except InvalidStatusCode as e:
            print(f"[Binance] Invalid status code: {e}")
        except OSError as e:
            print(f"[Binance] Network error: {e}")
        except Exception as e:
            print(f"[Binance] Unexpected error: {e}")
        
        reconnect_count += 1
        if reconnect_count < MAX_RECONNECT_ATTEMPTS:
            print(f"[Binance] Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
        else:
            print(f"[Binance] Max reconnection attempts reached, giving up")
            break

async def bybit_ws():
    """Bybit WebSocket with error handling and reconnection"""
    reconnect_count = 0
    
    while reconnect_count < MAX_RECONNECT_ATTEMPTS:
        try:
            uri = "wss://stream.bybit.com/v5/public/linear"
            print(f"[Bybit] Connecting... (attempt {reconnect_count + 1})")
            
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                params = [f"tickers.{symbol}" for symbol in symbols]
                subscribe_msg = {
                    "op": "subscribe",
                    "args": params
                }
                await ws.send(json.dumps(subscribe_msg))
                print(f"[Bybit] Connected and subscribed successfully")
                reconnect_count = 0
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        envelope = json.loads(msg)
                        
                        if 'data' in envelope and envelope.get('topic', '').startswith('tickers.'):
                            payload = envelope['data']
                            symbol = payload.get('symbol')
                            bid = payload.get('bid1Price') 
                            ask = payload.get('ask1Price')  
                            
                            if symbol and bid and ask and symbol in shared_data:
                                shared_data[symbol]["Bybit"]["bid"] = float(bid)
                                shared_data[symbol]["Bybit"]["ask"] = float(ask)
                                
                    except asyncio.TimeoutError:
                        print(f"[Bybit] No data received for 30s, checking connection...")
                        await ws.ping()
                        continue
                    except json.JSONDecodeError as e:
                        print(f"[Bybit] JSON decode error: {e}")
                        continue
                        
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[Bybit] Connection closed: {e}")
        except InvalidStatusCode as e:
            print(f"[Bybit] Invalid status code: {e}")
        except OSError as e:
            print(f"[Bybit] Network error: {e}")
        except Exception as e:
            print(f"[Bybit] Unexpected error: {e}")
        
        reconnect_count += 1
        if reconnect_count < MAX_RECONNECT_ATTEMPTS:
            print(f"[Bybit] Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
        else:
            print(f"[Bybit] Max reconnection attempts reached, giving up")
            break

async def bitget_ws():
    """Bitget WebSocket with error handling and reconnection"""
    reconnect_count = 0
    
    while reconnect_count < MAX_RECONNECT_ATTEMPTS:
        try:
            uri = "wss://ws.bitget.com/v2/ws/public"
            print(f"[Bitget] Connecting... (attempt {reconnect_count + 1})")
            
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [{
                        "instType": "USDT-FUTURES",
                        "channel": "ticker",
                        "instId": symbol
                    } for symbol in symbols]
                }
                await ws.send(json.dumps(subscribe_msg))
                print(f"[Bitget] Connected and subscribed successfully")
                reconnect_count = 0
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        if 'data' in data and len(data['data']) > 0:
                            item = data['data'][0]
                            bid = item.get('bidPr')
                            ask = item.get('askPr')
                            symbol = item['instId']
                            async with lock:
                                if bid and ask and symbol in shared_data:
                                    shared_data[symbol]["Bitget"]["bid"] = float(bid)
                                    shared_data[symbol]["Bitget"]["ask"] = float(ask)
                                
                    except asyncio.TimeoutError:
                        print(f"[Bitget] No data received for 30s, checking connection...")
                        await ws.ping()
                        continue
                    except json.JSONDecodeError as e:
                        print(f"[Bitget] JSON decode error: {e}")
                        continue
                        
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[Bitget] Connection closed: {e}")
        except InvalidStatusCode as e:
            print(f"[Bitget] Invalid status code: {e}")
        except OSError as e:
            print(f"[Bitget] Network error: {e}")
        except Exception as e:
            print(f"[Bitget] Unexpected error: {e}")
        
        reconnect_count += 1
        if reconnect_count < MAX_RECONNECT_ATTEMPTS:
            print(f"[Bitget] Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
        else:
            print(f"[Bitget] Max reconnection attempts reached, giving up")
            break

async def okx_ws():
    """OKX WebSocket with error handling and reconnection"""
    reconnect_count = 0
    
    while reconnect_count < MAX_RECONNECT_ATTEMPTS:
        try:
            uri = "wss://ws.okx.com:8443/ws/v5/public"
            print(f"[OKX] Connecting... (attempt {reconnect_count + 1})")
            
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [{
                        "channel": "tickers",
                        "instId": symbol
                    } for symbol in okx_symbols]
                }
                await ws.send(json.dumps(subscribe_msg))
                print(f"[OKX] Connected and subscribed successfully")
                reconnect_count = 0
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        if 'data' in data and len(data['data']) > 0:
                            item = data['data'][0]
                            bid = item.get('bidPx')
                            ask = item.get('askPx')
                            symbol = item['instId'].replace('-SWAP','').replace('-','')
                            
                            if bid and ask and symbol in shared_data:
                                shared_data[symbol]["OKX"]["bid"] = float(bid)
                                shared_data[symbol]["OKX"]["ask"] = float(ask)
                                
                    except asyncio.TimeoutError:
                        print(f"[OKX] No data received for 30s, checking connection...")
                        await ws.ping()
                        continue
                    except json.JSONDecodeError as e:
                        print(f"[OKX] JSON decode error: {e}")
                        continue
                        
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            print(f"[OKX] Connection closed: {e}")
        except InvalidStatusCode as e:
            print(f"[OKX] Invalid status code: {e}")
        except OSError as e:
            print(f"[OKX] Network error: {e}")
        except Exception as e:
            print(f"[OKX] Unexpected error: {e}")
        
        reconnect_count += 1
        if reconnect_count < MAX_RECONNECT_ATTEMPTS:
            print(f"[OKX] Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)
        else:
            print(f"[OKX] Max reconnection attempts reached, giving up")
            break

# Optional: Add a connection status monitoring function
def get_connection_status():
    """Get status of all exchange connections"""
    status = {}
    for exchange in ['Binance', 'OKX', 'Bitget', 'Bybit']:
        # Check if we have recent data from this exchange
        recent_data_count = 0
        for symbol in symbols[:10]: 
            if symbol in shared_data:
                if (shared_data[symbol][exchange]['bid'] is not None and 
                    shared_data[symbol][exchange]['ask'] is not None):
                    recent_data_count += 1
    
        status[exchange] = {
            'connected': recent_data_count > 5,  # Consider connected if >5 symbols have data
            'data_count': recent_data_count
        }
    return status

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
            if opportunity is not None and opportunity['open_spread_pct'] > max_spread:
                max_spread = opportunity['open_spread_pct']
                best_strategy = opportunity
        if best_strategy is not None:
            decision_id = await get_next_decision_id()
            best_strategy['decision_id'] = decision_id
            decision_logger.info(f"决策id:{decision_id}, 检测到当前最大价差值{best_strategy['open_spread']}, 价差比{best_strategy['open_spread_pct']}, 货币种类: {best_strategy['symbol']}, 买价: {best_strategy['best_buy_price']}@{best_strategy['best_buy_exchange']}, 卖价{best_strategy['best_sell_price']}@{best_strategy['best_sell_exchange']}, 订单数额详情：{best_strategy['quotes']}")
            decision_id +=1
            await strategy_results_queue.put(best_strategy)
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
            'decision_id' :opportunity['decision_id']
             }
    return enriched

def open_position(enrich_trade):
    global opening_positions,trade_history,active_trades
    enrich_trade['trade_time'] = time.time()
    enrich_trade['action'] = 'open'
    trade_history.append(enrich_trade)
    active_trades.append(enrich_trade)
    opening_positions += 1
    decision_logger.info(f"决策id: {enrich_trade['decision_id']} 开仓成功")
    output_logger.info(f"开仓：货币种类：{enrich_trade['symbol']}，购买交易所：{enrich_trade['best_buy_exchange']}，购买价：{enrich_trade['best_buy_price']}，出售交易所：{enrich_trade['best_sell_exchange']}，出售价：{enrich_trade['best_sell_price']}，价差：{enrich_trade['open_spread']}，价差比：{enrich_trade['open_spread_pct']}")

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
    decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 被接受，准备开仓")
        # logic about balance on those exchange...
        # TODO
        # if enrich_trade['estimated_net_profit_pct'] >= MINIMUM_PROFIT_PCT:
    if enrich_trade['estimated_net_profit_pct'] >= 0:

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
            'current_sell_price': current_sell_price,
            'decision_id': decision_id,
            }

def should_close_position(trade,current_status):
    unrealized_pnl_pct = current_status['unrealized_pnl']/(trade['best_buy_price'] * trade['trade_amount'])
    # if unrealized_pnl_pct >= trade['estimated_net_profit_pct']* 0.7:
    if unrealized_pnl_pct >= 0.001:
        decision_logger.info(f"决策id:{current_status['decision_id']}, 触发平仓条件：止盈, 相关数据：")
        return True
    # execute close position or do nothing

    if unrealized_pnl_pct <= STOP_LOSS_PCT:
        decision_logger.info(f"决策id:{current_status['decision_id']}, 触发平仓条件：止损, 相关数据：")

        return True

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
        'exit_reason': determine_exit_reason(trade,current_status),
        'decision_id': current_status['decision_id']})
    trade_history.append(trade)
    active_trades.pop()
    opening_positions -= 1
    decision_logger.info(f"决策id: {current_status['decision_id']}, 平仓成功， 相关数据")
    current_spread = current_status['current_sell_price'] - current_status['current_buy_price']
    current_spread_pct = current_spread / current_status['current_buy_price']

    output_logger.info(
        f"平仓: 决策id: {current_status['decision_id']}, 货币种类: {trade['symbol']}，平仓（卖出）交易所: {trade['best_buy_exchange']}, 平仓（卖出）价: {current_status['current_buy_price']}, 出售交易所：{trade['best_sell_exchange']}，平仓价：{current_status['current_sell_price']}，原始价差：{trade['open_spread']}，原始价差比：{trade['open_spread_pct']}，当前价差：{current_spread:.2f}，当前价差比：{current_spread_pct:.4f}，最终收益：{trade['pnl']:.2f}"
    )
async def execute_simulation():
    output_logger.info(f"开始模拟")
    global active_trades, trade_history, opening_positions
    print("[SIMULATION] Simulation starts")

    while True:
    # await handle_new_opportunities()
        opportunity = await strategy_results_queue.get()
        # print("Evaluating the strategy from strategy_results_queue")
        enrich_trade = enrich_with_costs_and_profits(opportunity)

        async with lock:
            if opening_positions < MAX_POSITION_SIZE and should_open_position(enrich_trade):
                open_position(enrich_trade)

        async with lock:
            if opening_positions < 0:
                raise ValueError('opening_positions cannot be negative')
            
            if opening_positions > 1:
                print("lol race condition")
            # handle active positions
            if len(active_trades) > 0:

                # this should be equivalent to opening_positions == 0
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
    MINIMUM_PROFIT_PCT = 0.0025

    TRADE_AMOUNT_USDT = 1000    # 每次套利的金额
    MAX_POSITION_SIZE = 1   # 最大持仓限制
    AVAILABLE_BALANCE = {       # 每个交易所的可用余额
        'Binance': 10000,
        'Bybit': 10000,
        'OKX': 10000,
        'Bitget': 10000
}
    MAX_HOLDING_TIME = 300      # 最大持仓时间(秒)
    STOP_LOSS_PCT = -0.002       # 止损百分比
    MAX_CONCURRENT_TRADES = 1   # 最大同时进行的套利数量
    MINIMUM_SPREAD_PCT = 0.0002
    active_trades = []
    trade_history = []

    exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
    symbols = get_common_symbols()
    okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in symbols]
    #print(f"monitoring {len(symbols)} currencies") 163
    shared_data = {
        symbol:{exchange:{"bid":None,"ask":None}for exchange in exchanges}for symbol in symbols
    }
    strategy_results_queue = asyncio.Queue()
    # initialize lock for thread safety
    opening_positions = 0

    asyncio.run(main())
