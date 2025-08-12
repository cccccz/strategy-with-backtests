from shared_imports import time, asyncio,os
from trading_state import TradingState

# 显示函数 (传入state参数)
def display_exchange_data(state: TradingState):
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

def display_trade_history(state: TradingState):
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

def display_balance_info(state: TradingState):
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

async def display_terminal(state: TradingState):
    while True:
        display_exchange_data(state)
        display_trade_history(state)
        display_balance_info(state)
        print(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        await asyncio.sleep(1)

# 工具函数
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

