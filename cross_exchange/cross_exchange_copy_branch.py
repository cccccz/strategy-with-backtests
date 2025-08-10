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
import asyncio
import json
import websockets
import time
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatusCode
from config_copy import Config

# 日志设置
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

decision_logger = logging.getLogger('decision_logger')
decision_logger.setLevel(logging.INFO)
decision_handler = logging.FileHandler('./cross_exchange/decision_log.txt')
decision_handler.setFormatter(logging.Formatter(log_format))
decision_logger.addHandler(decision_handler)

output_logger = logging.getLogger('output_logger')
output_logger.setLevel(logging.INFO)
output_handler = logging.FileHandler('./cross_exchange/output_log.txt')
output_handler.setFormatter(logging.Formatter(log_format))
output_logger.addHandler(output_handler)

# 状态管理类
class TradingState:
    def __init__(self):
        self.shared_data = {}
        self.active_trades = []
        self.trade_history = []
        self.opening_positions = 0
        self.lock = asyncio.Lock()
        self.decision_id = 0
        self.decision_id_lock = asyncio.Lock()
        self.symbols = []
        self.okx_symbols = []
        self.latest_opportunity = None
        self.opportunity_lock = asyncio.Lock()

        # 资金管理
        self.initial_capital = Config.INITIAL_CAPITAL
        self.exchange_balances = {}
        self.total_balance = Config.INITIAL_CAPITAL
        self.total_pnl = 0.0
        self.balance_lock = asyncio.Lock()

    def init_exchange_balances(self):
        """初始化各交易所资金分配"""
        for exchange, allocation in Config.EXCHANGE_CAPITAL_ALLOCATION.items():
            self.exchange_balances[exchange] = {
                'available': self.initial_capital * allocation,
                'used': 0.0,
                'total': self.initial_capital * allocation
            }
    
    def get_available_capital(self, exchange):
        """获取指定交易所的可用资金"""
        return self.exchange_balances[exchange]['available']
    
    def calculate_trade_amount(self, buy_exchange, buy_price, sell_exchange, sell_price):
        buy_available = self.get_available_capital(buy_exchange)
        sell_available = self.get_available_capital(sell_exchange)

        # max_trade_capital = available_capital * Config.MAX_TRADE_CAPITAL_PCY

        # if max_trade_capital < Config.MIN_TRADE_AMOUNT:
        #     return 0.0
        
        buy_amount = buy_available / buy_price
        sell_amount = sell_available / sell_price
        trade_amount = min(buy_amount, sell_amount)
        return trade_amount

    
    async def get_next_decision_id(self):
        async with self.decision_id_lock:
            current_id = self.decision_id
            self.decision_id += 1
            return current_id

    def init_symbols(self):
        """初始化交易对"""
        self.symbols = get_common_symbols()
        self.okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in self.symbols]
        
        exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
        self.shared_data = {
            symbol: {exchange: {"bid": None, "ask": None} for exchange in exchanges}
            for symbol in self.symbols
        }

# 全局状态实例
state = TradingState()

# 工具函数
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

# 显示函数 (传入state参数)
def display_exchange_data(state):
    print("+------------+-----------+-------------+-------------+")
    print("|   Symbol   | Exchange  | Bid Price   |  Ask Price  |")
    print("+------------+-----------+-------------+-------------+")
    for symbols, orderbooks in state.shared_data.items():
        for exchange, book in orderbooks.items():
            bid = book['bid'] if book['bid'] else "..."
            ask = book['ask'] if book['ask'] else "..."
            print(f"| {symbols:<10} | {exchange:<9} | {str(bid):<11} | {str(ask):<11} |")
    print("+------------+-----------+-------------+-------------+")

def display_trade(state):
    if state.active_trades:
        strategy_result = state.active_trades[0]
        print(f"Symbol: {strategy_result['symbol']}")
        print(f"Best Buy on: {strategy_result['best_buy_exchange']}, {strategy_result['best_buy_price']}")
        print(f"Best Sell on: {strategy_result['best_sell_exchange']}, {strategy_result['best_sell_price']}")
        print(f"Open Spread: {strategy_result['open_spread']:.4f}")
        print(f"Open Spread Percentage: {strategy_result['open_spread_pct']*100:.2f}%")
        print(f"Time Stamp Opportunity: {strategy_result['time_stamp_opportunity']}")
        print(f"Trade Time: {strategy_result['trade_time']}")
        print("-" * 50)

def display_trade_history(state):
    """显示交易历史"""
    if not state.trade_history:
        print("No trade history available.")
        return

    print("\n" + "="*140)
    print("TRADE HISTORY")
    print("="*140)

    # 分组显示交易对
    trade_pairs = []
    open_trades = {}

    for trade in state.trade_history:
        if trade['action'] == 'open':
            open_trades[trade['time_stamp_opportunity']] = trade
        elif trade['action'] == 'close':
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

    # 添加未平仓的交易
    for open_trade in open_trades.values():
        trade_pairs.append((open_trade, None))

    # 显示表头
    print(f"{'#':<3} {'Symbol':<12} {'Buy@':<25} {'Sell@':<25} {'Open Time':<20} {'Close Time':<20} {'PnL':<15} {'Status':<10}")
    print("-" * 140)

    # 显示每个交易对
    for i, (open_trade, close_trade) in enumerate(trade_pairs, 1):
        symbol = open_trade['symbol']
        buy_ex = open_trade['best_buy_exchange']
        sell_ex = open_trade['best_sell_exchange']

        open_time = time.strftime('%m-%d %H:%M:%S', time.localtime(open_trade['trade_time']))

        # 初始开仓价格
        open_buy_price = open_trade['best_buy_price']
        open_sell_price = open_trade['best_sell_price']

        if close_trade:
            close_time = time.strftime('%m-%d %H:%M:%S', time.localtime(close_trade['close_time']))
            pnl = f"{close_trade['pnl']:.8f}"
            status = "CLOSED"

            # 如果记录了平仓价格，用于显示（否则设为 N/A）
            close_buy_price = close_trade.get('current_buy_price', 'N/A')
            close_sell_price = close_trade.get('current_sell_price', 'N/A')

            # 格式化价格信息：开仓价 / 平仓价
            if isinstance(close_buy_price, float):
                buy_info = f"{buy_ex[:3]}@{open_buy_price:.6f}/{close_buy_price:.6f}"
            else:
                buy_info = f"{buy_ex[:3]}@{open_buy_price:.6f}/N/A"

            if isinstance(close_sell_price, float):
                sell_info = f"{sell_ex[:3]}@{open_sell_price:.6f}/{close_sell_price:.6f}"
            else:
                sell_info = f"{sell_ex[:3]}@{open_sell_price:.6f}/N/A"

        else:
            close_time = "OPEN"
            pnl = f"{open_trade['estimated_net_profit']:.8f}"
            status = "OPEN"

            # 使用当前市场价格作为参考
            current_buy_price = "N/A"
            current_sell_price = "N/A"

            if symbol in state.shared_data:
                if state.shared_data[symbol][buy_ex]['ask']:
                    current_buy_price = f"{state.shared_data[symbol][buy_ex]['ask']:.6f}"
                if state.shared_data[symbol][sell_ex]['bid']:
                    current_sell_price = f"{state.shared_data[symbol][sell_ex]['bid']:.6f}"

            buy_info = f"{buy_ex[:3]}@{open_buy_price:.6f}/{current_buy_price}"
            sell_info = f"{sell_ex[:3]}@{open_sell_price:.6f}/{current_sell_price}"

        print(f"{i:<3} {symbol:<12} {buy_info:<25} {sell_info:<25} {open_time:<20} {close_time:<20} {pnl:<15} {status:<10}")

    # 统计信息
    closed_trades = [pair for pair in trade_pairs if pair[1] is not None]
    if closed_trades:
        total_pnl = sum(close_trade['pnl'] for _, close_trade in closed_trades)
        avg_pnl = total_pnl / len(closed_trades)
        profitable_trades = len([pair for pair in closed_trades if pair[1]['pnl'] > 0])

        print("-" * 140)
        print(f"SUMMARY: Total Trades: {len(closed_trades)} | Profitable: {profitable_trades} | Total PnL: {total_pnl:.8f} | Avg PnL: {avg_pnl:.8f}")

    print("=" * 140)

def display_balance_info(state):
    """显示资金信息"""
    print("\n" + "="*80)
    print("BALANCE INFORMATION")
    print("="*80)
    
    total_roi = (state.total_pnl / state.initial_capital) * 100
    
    print(f"Initial Capital: {state.initial_capital:,.2f} USDT")
    print(f"Current Balance: {state.total_balance:,.2f} USDT")
    print(f"Total PnL: {state.total_pnl:,.8f} USDT")
    print(f"ROI: {total_roi:.4f}%")
    print("-" * 80)
    print(f"{'Exchange':<10} {'Total':<12} {'Available':<12} {'Used':<12} {'Utilization':<12}")
    print("-" * 80)
    
    for exchange, balance in state.exchange_balances.items():
        utilization = (balance['used'] / balance['total']) * 100
        print(f"{exchange:<10} {balance['total']:<12.2f} {balance['available']:<12.2f} {balance['used']:<12.2f} {utilization:<12.2f}%")
    
    print("="*80)

async def display_terminal(state):
    while True:
        display_exchange_data(state)
        display_trade_history(state)
        display_balance_info(state)
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(1)

# 通用WebSocket连接器
async def generic_ws_connector(exchange_name, ws_config, state):
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
async def binance_ws(state):
    params = [f"{s.lower()}@bookTicker" for s in state.symbols]
    uri = f"wss://fstream.binance.com/stream?streams=" + "/".join(params)
    config = {
        'uri': uri,
        'parser': parse_binance
    }
    await generic_ws_connector('Binance', config, state)

async def bybit_ws(state):
    config = {
        'uri': "wss://stream.bybit.com/v5/public/linear",
        'subscribe_msg': {
            "op": "subscribe",
            "args": [f"tickers.{symbol}" for symbol in state.symbols]
        },
        'parser': parse_bybit
    }
    await generic_ws_connector('Bybit', config, state)

async def bitget_ws(state):
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

async def okx_ws(state):
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

# 策略计算函数 (拆分成小函数)
def extract_valid_quotes(exchange_data):
    """提取有效报价"""
    quotes = []
    for exchange, data in exchange_data.items():
        bid, ask = data['bid'], data['ask']
        if bid and ask:
            quotes.append((exchange, float(bid), float(ask)))
    return quotes

def create_opportunity_dict(symbol, best_buy, best_sell, quotes):
    """创建机会字典"""
    open_spread = best_sell[1] - best_buy[2]
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    
    return {
        'symbol': symbol,
        'best_buy_exchange': best_buy[0],
        'best_buy_price': best_buy[2],
        'best_sell_exchange': best_sell[0],
        'best_sell_price': best_sell[1],
        'open_spread': open_spread,
        'open_spread_pct': open_spread_pct,
        'quotes': quotes,
        'time_stamp_opportunity': time.time()
    }

def calculate_opportunity(symbol, exchange_data):
    """计算套利机会"""
    quotes = extract_valid_quotes(exchange_data)
    if len(quotes) < 2:
        return None

    best_buy = min(quotes, key=lambda x: x[2])  # ask
    best_sell = max(quotes, key=lambda x: x[1])  # bid

    return create_opportunity_dict(symbol, best_buy, best_sell, quotes)

async def compute_strategy(state):
    """计算策略"""
    while True:
        async with state.lock: 
            snapshot = copy.deepcopy(state.shared_data)

        best_strategy = None
        max_spread = 0
        
        for symbol, exchange_data in snapshot.items():
            opportunity = calculate_opportunity(symbol, exchange_data)
            if opportunity is not None and opportunity['open_spread_pct'] > max_spread:
                max_spread = opportunity['open_spread_pct']
                best_strategy = opportunity
                
        if best_strategy is not None:
            decision_id = await state.get_next_decision_id()
            best_strategy['decision_id'] = decision_id
            decision_logger.info(f"决策id:{decision_id}, 检测到当前最大价差值{best_strategy['open_spread']}, 价差比{best_strategy['open_spread_pct']}, 货币种类: {best_strategy['symbol']}, 买价: {best_strategy['best_buy_price']}@{best_strategy['best_buy_exchange']}, 卖价{best_strategy['best_sell_price']}@{best_strategy['best_sell_exchange']}")
            async with state.opportunity_lock:
                state.latest_opportunity = best_strategy

            
        await asyncio.sleep(Config.COMPUTE_INTERVAL)

# 成本计算函数
def calculate_open_costs(buy_exchange, sell_exchange, buy_price, sell_price, trade_amount):
    """计算开仓成本"""
    trade_amount = float(trade_amount)
    buy_price = float(buy_price)
    sell_price = float(sell_price)
    buy_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[buy_exchange]['taker']) * buy_price
    sell_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[sell_exchange]['taker']) * sell_price
    slippage_cost = 0
    
    return {
        'total_cost': buy_fee + sell_fee + slippage_cost,
        'buy_fee': buy_fee,
        'sell_fee': sell_fee,
        'net_entry': sell_price - buy_price
    }

def calculate_exit_costs(buy_exchange, sell_exchange, current_buy_price, current_sell_price, trade_amount):
    """计算平仓成本"""
    trade_amount = float(trade_amount)
    current_buy_price = float(current_buy_price)
    current_sell_price = float(current_sell_price)
    close_long_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[buy_exchange]['taker']) * current_buy_price
    close_short_fee = trade_amount * float(Config.FUTURES_TRADING_FEES[sell_exchange]['taker']) * current_sell_price
    slippage_cost = 0

    return {
        'total_cost': close_long_fee + close_short_fee + slippage_cost,
        'close_long_cost': close_long_fee,
        'close_short_cost': close_short_fee,
        'net_exit': current_buy_price - current_sell_price
    }

def calculate_required_margin(trade_amount):
    return 0

# 交易执行函数
def enrich_with_costs_and_profits(opportunity):
    """丰富机会信息，添加成本和利润计算"""
    quotes = opportunity['quotes']
    symbol = opportunity['symbol']
    best_buy = min(quotes, key=lambda x: x[2])
    best_sell = max(quotes, key=lambda x: x[1])

    trade_amount = state.calculate_trade_amount(best_buy[0],best_buy[2],best_sell[0],best_sell[1])
    # moved this logic to should_open.... if performance issue then bring it back..
    # if trade_amount <= 0:
    #     return None
    trade_capital = trade_amount * best_buy[2]
    costs = calculate_open_costs(best_buy[0], best_sell[0], best_buy[2], best_sell[1], trade_amount)
    estimated_close_costs = calculate_exit_costs(best_buy[0], best_sell[0], best_sell[1], best_buy[2], trade_amount)
    total_cost = costs['total_cost'] + estimated_close_costs['total_cost']
    
    open_spread = float(best_sell[1] - best_buy[2])
    open_spread_pct = 2 * open_spread / (best_buy[2] + best_sell[1])
    close_spread_pct = 2 * (best_sell[2] - best_buy[1]) / (best_buy[1] + best_sell[2])

    estimated_net_profit = open_spread * trade_amount - total_cost
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
        'trade_capital': trade_capital,
        'capital_utilization_pct': trade_capital/ state.initial_capital,

        'margin_required': margin_required,
        'estimated_total_cost': total_cost,
        'buy_fee': costs['buy_fee'],
        'sell_fee': costs['sell_fee'],
        'net_entry': costs['net_entry'],
        'estimated_net_profit': estimated_net_profit,
        'estimated_net_profit_pct': estimated_net_profit_pct,
        'time_stamp_opportunity': opportunity['time_stamp_opportunity'],
        'stop_loss': -abs(estimated_net_profit * Config.STOP_LOSS_PCT),
        'decision_id': opportunity['decision_id']
    }
    return enriched

# def should_open_position(enrich_trade, state):
#     """判断是否应该开仓"""
#     decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 被接受，准备开仓")
#     if enrich_trade['estimated_net_profit_pct'] >= 0:
#         return True
#     return False

def should_open_position(enrich_trade, state):
    if enrich_trade['trade_amount'] <= 0:
        return False
    spread_pct = enrich_trade['open_spread_pct']
    
    if spread_pct >= Config.MIN_SPREAD_PCT_THRESHOLD:
    # if enrich_trade['estimated_net_profit'] > 0:
        decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
        return True
    
    decision_logger.info(f"决策 id: {enrich_trade['decision_id']} 不满足开仓条件: 'estimated_net_profit': {enrich_trade['estimated_net_profit']}, spread_pct={spread_pct:.6f}")
    return False

def open_position(enrich_trade, state):
    """开仓"""
    buy_exchange = enrich_trade['best_buy_exchange']
    sell_exchange = enrich_trade['best_sell_exchange']
    trade_capital = enrich_trade['trade_capital']

    state.exchange_balances[buy_exchange]['available'] -= trade_capital
    state.exchange_balances[buy_exchange]['used'] += trade_capital
    state.exchange_balances[buy_exchange]['total'] = state.exchange_balances[buy_exchange]['used'] + state.exchange_balances[buy_exchange]['available']


    state.exchange_balances[sell_exchange]['available'] -= trade_capital
    state.exchange_balances[sell_exchange]['used'] += trade_capital
    state.exchange_balances[sell_exchange]['total'] = state.exchange_balances[sell_exchange]['used'] + state.exchange_balances[sell_exchange]['available']



    enrich_trade['trade_time'] = time.time()
    enrich_trade['action'] = 'open'
    state.trade_history.append(enrich_trade)
    state.active_trades.append(enrich_trade)
    state.opening_positions += 1
    decision_logger.info(f"决策id: {enrich_trade['decision_id']} 开仓成功, 使用资金: {trade_capital: .2f} USDT")
    output_logger.info(f"开仓：决策id: {enrich_trade['decision_id']}, 货币种类：{enrich_trade['symbol']}，购买交易所：{enrich_trade['best_buy_exchange']}，购买价：{enrich_trade['best_buy_price']}，出售交易所：{enrich_trade['best_sell_exchange']}，出售价：{enrich_trade['best_sell_price']}，价差：{enrich_trade['open_spread']}，价差比：{enrich_trade['open_spread_pct']}")

def evaluate_active_position(trade, snapshot, state):
    """评估活跃仓位"""
    symbol = trade['symbol']
    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']

    current_buy_price = snapshot[symbol][buy_exchange]['ask']
    current_sell_price = snapshot[symbol][sell_exchange]['bid']
    
    if not (current_buy_price and current_sell_price):
        return None
    
    exit_costs = calculate_exit_costs(
        buy_exchange,
        sell_exchange,
        current_buy_price,
        current_sell_price,
        trade['trade_amount']
    )
    position_age = time.time() - trade['trade_time']
    
    # trade['net_entry'] is something i forget, do not recommend using it
    entry_net = trade['net_entry']
    entry_costs = trade['buy_fee'] + trade['sell_fee'] 
    exit_net = exit_costs['total_cost']
    spread_diff = trade['best_sell_price'] - trade['best_buy_price'] + current_buy_price - current_sell_price
    total_fee_cost = entry_costs + exit_net
    unrealized_pnl = spread_diff * trade['trade_amount'] - total_fee_cost

    return {
        'current_spread': current_sell_price - current_buy_price,
        'unrealized_pnl': unrealized_pnl,
        'exit_costs': exit_costs,
        'position_age': position_age,
        'current_buy_price': current_buy_price,
        'current_sell_price': current_sell_price,
        'decision_id': trade['decision_id'],
    }

def should_close_position(trade, current_status, state):
    current_spread_pct =  2 * current_status['current_spread'] / (current_status['current_buy_price'] + current_status['current_sell_price'])

    spread_diff = trade['open_spread_pct']
    # if (trade['open_spread_pct'] - current_spread_pct) >= Config.PROFIT_THRESHOLD:
    if current_spread_pct <= trade['open_spread_pct'] * Config.MAGIC_THRESHOLD:

    # if  current_spread_pct <= 0:

        decision_logger.debug(
            f"Spread check: open={trade['open_spread_pct']:.6f}, current={current_spread_pct:.6f}, diff={trade['open_spread_pct'] - current_spread_pct:.6f}"
        )

        decision_logger.info(
            f"决策id:{current_status['decision_id']}，触发平仓条件：止盈（当前价差率: {current_spread_pct:.6f}）"
        )
        return True

    if current_spread_pct <= Config.STOP_LOSS_THRESHOLD:
        decision_logger.info(
            f"决策id:{current_status['decision_id']}，触发平仓条件：止损（当前价差率: {current_spread_pct:.6f}）"
        )
        return True

    return False


def determine_exit_reason(trade, current_status):
    return "unimplemented"

def close_position(trade, current_status, state):
    """平仓"""
    trade = trade.copy()

    buy_exchange = trade['best_buy_exchange']
    sell_exchange = trade['best_sell_exchange']
    trade_capital = trade['trade_capital']
    pnl = current_status['unrealized_pnl']

    state.exchange_balances[buy_exchange]['used'] -= trade_capital
    # 平分
    state.exchange_balances[buy_exchange]['available'] += trade_capital + pnl / 2
    state.exchange_balances[buy_exchange]['total'] = state.exchange_balances[buy_exchange]['available']+state.exchange_balances[buy_exchange]['used']
    # state.exchange_balances[buy_exchange]['available'] += trade_capital + pnl

    state.exchange_balances[sell_exchange]['used'] -= trade_capital
    state.exchange_balances[sell_exchange]['available'] += trade_capital + pnl / 2
    state.exchange_balances[sell_exchange]['total'] = state.exchange_balances[sell_exchange]['available']+state.exchange_balances[sell_exchange]['used']


    state.total_pnl += pnl
    state.total_balance  = state.initial_capital + state.total_pnl
    trade.update({
        'action': 'close',
        'close_time': time.time(),
        'pnl': current_status['unrealized_pnl'],
        'exit_reason': determine_exit_reason(trade, current_status),
        'current_buy_price':current_status['current_buy_price'],
        'current_sell_price':current_status['current_sell_price'],
        'decision_id': current_status['decision_id']
    })
    state.trade_history.append(trade)
    state.active_trades.pop()
    state.opening_positions -= 1

    decision_logger.info(
        f"平仓成功｜决策id: {current_status['decision_id']}｜币种: {trade['symbol']}｜"
        f"开仓: {trade['best_buy_price']}@{trade['best_buy_exchange']} → {trade['best_sell_price']}@{trade['best_sell_exchange']}｜"
        f"平仓: {current_status['current_buy_price']}@{trade['best_buy_exchange']} → {current_status['current_sell_price']}@{trade['best_sell_exchange']}｜"
        f"原始价差: {trade['open_spread']:.6f}｜当前价差: {current_status['current_sell_price'] - current_status['current_buy_price']:.6f}｜"
        f"交易本金: {trade['trade_capital']:.6f}｜净收益: {trade['pnl']:.6f}｜收益率: {(trade['pnl'] / state.initial_capital):.6%}"
    )

    current_spread = current_status['current_sell_price'] - current_status['current_buy_price']
    current_spread_pct = 2 * current_spread / (current_status['current_sell_price']+current_status['current_buy_price'])

    output_logger.info(
        f"平仓: 决策id: {current_status['decision_id']}, 货币种类: {trade['symbol']}，平仓（卖出）交易所: {trade['best_buy_exchange']}, 平仓（卖出）价: {current_status['current_buy_price']}, 出售交易所：{trade['best_sell_exchange']}，平仓价：{current_status['current_sell_price']}，原始价差：{trade['open_spread']}，原始价差比：{trade['open_spread_pct']}，当前价差：{current_spread:.6f}，当前价差比：{current_spread_pct:.6f}，最终收益：{trade['pnl']:.6f}"
    )

async def execute_simulation(state):
    """执行模拟交易"""
    output_logger.info("开始模拟")
    print("[SIMULATION] Simulation starts")

    while True:
        await asyncio.sleep(0.1)
        async with state.opportunity_lock:
            opportunity = state.latest_opportunity
            state.latest_opportunity = None  # 清空，避免重复处理

        if opportunity is None:
            continue
        enrich_trade = enrich_with_costs_and_profits(opportunity)
        async with state.lock:
            if state.opening_positions < Config.MAX_POSITION_SIZE and should_open_position(enrich_trade, state):
                open_position(enrich_trade, state)

        async with state.lock:
            if state.opening_positions < 0:
                raise ValueError('opening_positions cannot be negative')
            
            if len(state.active_trades) > 0:
                snapshot = copy.deepcopy(state.shared_data)
                trade = state.active_trades[0]
                current_status = evaluate_active_position(trade, snapshot, state)
                if current_status and should_close_position(trade, current_status, state):
                    close_position(trade, current_status, state)
                
        await asyncio.sleep(0.1)

# 主程序
async def main():
    """主程序入口"""
    # 初始化
    state.init_symbols()
    state.init_exchange_balances()
    print(f"Monitoring {len(state.symbols)} currencies")
    
    # 启动所有任务
    tasks = [
        binance_ws(state),
        bitget_ws(state),
        okx_ws(state),
        bybit_ws(state),
        compute_strategy(state),
        execute_simulation(state),
        display_terminal(state),
    ]
    
    await asyncio.gather(*tasks)

if __name__ == '__main__':  
    asyncio.run(main())