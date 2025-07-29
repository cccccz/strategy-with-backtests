import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.trend import MACD  
import ta
from datetime import datetime, timedelta
import math

# 获取近 3 年比特币数据
# def get_btc_data(years=3, extra_days=26):
#     end_time = datetime.now()
#     start_time = end_time - timedelta(days=365 * years + extra_days)
    
#     url = "https://api.binance.com/api/v3/klines"
#     params = {
#         'symbol': 'BTCUSDT',
#         'interval': '4h',
#         'startTime': int(start_time.timestamp() * 1000),
#         'endTime': int(end_time.timestamp() * 1000),
#         'limit': 1000 + extra_days
#     }
    
#     response = requests.get(url, params=params)
#     data = response.json()
    
#     df = pd.DataFrame(data, columns=[
#         'timestamp', 'open', 'high', 'low', 'close', 'volume',
#         'close_time', 'quote_volume', 'trades', 
#         'taker_buy_base', 'taker_buy_quote', 'ignore'
#     ])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
#     return df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]

def get_btc_data(years=3, extra_days=26, interval='4h'):
    """
    获取Binance BTC/USDT数据（自动分页版）
    参数:
        years: 获取数据的年数
        extra_days: 额外天数
        interval: K线间隔（默认4小时）
    """
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365 * years + extra_days)
    
    # 计算总需要的数据量（按4小时K线估算）
    total_hours = (end_time - start_time).total_seconds() / 3600
    total_candles = math.ceil(total_hours / {'1h':1, '4h':4, '1d':24}[interval])
    
    print(f"需要获取的总K线数量: {total_candles}")
    
    # 计算需要分几页
    page_size = 1000  # Binance单次最大限制
    pages = math.ceil(total_candles / page_size)
    print(f"需要分 {pages} 页获取")
    
    all_data = []
    
    for page in range(pages):
        print(f"\n正在获取第 {page+1}/{pages} 页...")
        
        # 计算当前页的时间范围
        page_start = start_time + timedelta(
            hours=page * page_size * {'1h':1, '4h':4, '1d':24}[interval]
        )
        page_end = min(
            page_start + timedelta(hours=(page_size-1) * {'1h':1, '4h':4, '1d':24}[interval]),
            end_time
        )
        
        print(f"时间范围: {page_start} 到 {page_end}")
        
        # 构建请求参数
        params = {
            'symbol': 'BTCUSDT',
            'interval': interval,
            'startTime': int(page_start.timestamp() * 1000),
            'endTime': int(page_end.timestamp() * 1000),
            'limit': page_size
        }
        
        # 发送请求
        try:
            response = requests.get("https://api.binance.com/api/v3/klines", params=params)
            data = response.json()
            print(f"获取到 {len(data)} 条数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            
            # 添加到总数据
            all_data.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
            
        except Exception as e:
            print(f"获取第 {page+1} 页失败: {str(e)}")
            continue
    
    # 合并所有数据
    if all_data:
        final_df = pd.concat(all_data).drop_duplicates('timestamp')
        final_df = final_df.set_index('timestamp').sort_index()
        print(f"\n最终获取数据量: {len(final_df)} 条")
        print(f"时间范围: {final_df.index[0]} 到 {final_df.index[-1]}")
        return final_df
    else:
        print("未能获取任何数据")
        return pd.DataFrame()
    
# 计算 MACD 和金叉/死叉
def calculate_macd(df):
    # 使用 ta 库计算 MACD
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()          # DIF 线
    df['signal'] = macd.macd_signal() # DEA 线
    df['hist'] = macd.macd_diff()     # Histogram (MACD - Signal)
    
    # 计算金叉（DIF 上穿 DEA）和死叉（DIF 下穿 DEA）
    df['golden_cross'] = (df['macd'] > df['signal']) & (df['macd'].shift() <= df['signal'].shift())
    df['death_cross'] = (df['macd'] < df['signal']) & (df['macd'].shift() >= df['signal'].shift())
    
    return df.iloc[26:]


def print_crosses():
    # 获取数据并计算指标
    btc = get_btc_data(years=3)
    print("data length:", len(btc))
    print("first 5:", btc.head())
    btc = calculate_macd(btc)

    # 统计金叉/死叉次数
    golden_cross_count = btc['golden_cross'].sum()
    death_cross_count = btc['death_cross'].sum()
    print(f"金叉次数: {golden_cross_count}, 死叉次数: {death_cross_count}")

    # 标记每次金叉/死叉的日期和价格
    golden_dates = btc[btc['golden_cross']].index
    death_dates = btc[btc['death_cross']].index
    print("金叉日期:", golden_dates)
    print("死叉日期:", death_dates)

    # 打印前 5 行数据
    # print(btc[['close', 'macd', 'signal', 'hist', 'golden_cross', 'death_cross']].head())

    # 设置画布
    plt.figure(figsize=(14, 8))

    # 子图1：价格曲线 + 金叉/死叉标记
    ax1 = plt.subplot(211)
    btc['close'].plot(ax=ax1, color='black', label='BTC Price')
    ax1.scatter(golden_dates, btc.loc[golden_dates, 'close'], 
                color='green', marker='^', s=100, label='Golden Cross')
    ax1.scatter(death_dates, btc.loc[death_dates, 'close'], 
                color='red', marker='v', s=100, label='Death Cross')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()

    # 子图2：MACD指标
    ax2 = plt.subplot(212, sharex=ax1)
    btc[['macd', 'signal']].plot(ax=ax2)
    ax2.bar(btc.index, btc['hist'], 
            color=btc['hist'].apply(lambda x: 'green' if x > 0 else 'red'), 
            alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('MACD')

    plt.tight_layout()
    plt.show()

def macd_backtest(df):
    trades = []
    position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_time = None
    
    for i, row in df.iterrows():
        # 平仓条件检查
        if position == 'long':
            # 多单止损止盈检查
            if row['low'] <= entry_price * 0.99:  # 1%止损
                pnl = (entry_price * 0.99 - entry_price) / entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': entry_price * 0.99,
                    'exit_type': 'stop_loss',
                    'pnl': pnl
                })
                position = None
            elif row['high'] >= entry_price * 1.02:  # 2%止盈
                pnl = (entry_price * 1.02 - entry_price) / entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': entry_price * 1.02,
                    'exit_type': 'take_profit',
                    'pnl': pnl
                })
                position = None
                
        elif position == 'short':
            # 空单止损止盈检查
            if row['high'] >= entry_price * 1.01:  # 1%止损
                pnl = (entry_price - entry_price * 1.01) / entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': entry_price * 1.01,
                    'exit_type': 'stop_loss',
                    'pnl': pnl
                })
                position = None
            elif row['low'] <= entry_price * 0.98:  # 2%止盈
                pnl = (entry_price - entry_price * 0.98) / entry_price
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': entry_price * 0.98,
                    'exit_type': 'take_profit',
                    'pnl': pnl
                })
                position = None
        
        # 开仓条件检查（仅在无持仓时）
        if position is None:
            if row['golden_cross']:
                position = 'long'
                entry_price = row['close']
                entry_time = i
            elif row['death_cross']:
                position = 'short'
                entry_price = row['close']
                entry_time = i
    
    return pd.DataFrame(trades)

# ========================
# 分析结果
# ========================
def analyze_results(trades):
    if trades.empty:
        print("没有交易发生")
        return
    
    # 基础统计
    total_trades = len(trades)
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    win_rate = len(winning_trades) / total_trades
    
    print("\n" + "="*50)
    print("回测结果汇总")
    print("="*50)
    print(f"总交易次数: {total_trades}")
    print(f"盈利交易: {len(winning_trades)} ({win_rate:.1%})")
    print(f"亏损交易: {len(losing_trades)} ({(1-win_rate):.1%})")
    print(f"平均盈利: {winning_trades['pnl'].mean():.2%}")
    print(f"平均亏损: {losing_trades['pnl'].mean():.2%}")
    print(f"盈亏比: {abs(winning_trades['pnl'].mean()/losing_trades['pnl'].mean()):.2f}")
    print(f"累计收益率: {trades['pnl'].sum():.2%}")
    
    # 保存交易记录
    trades.to_csv('macd_crossbacktest_results.csv', index=False)
    print("\n交易记录已保存到 macd_crossbacktest_results.csv")

# ========================
# 可视化
# ========================
def visualize_results(df, trades):
    plt.figure(figsize=(16, 12))
    
    # ============= 价格图表（简化版） =============
    ax1 = plt.subplot(211)
    df['close'].plot(ax=ax1, color='black', alpha=0.8, label='BTC Price')
    
    # 仅标记入场点
    long_entries = df[df['golden_cross']]
    short_entries = df[df['death_cross']]
    ax1.scatter(long_entries.index, long_entries['close'], 
               color='green', marker='^', s=80, label='Long Entry')
    ax1.scatter(short_entries.index, short_entries['close'],
               color='red', marker='v', s=80, label='Short Entry')
    ax1.set_title('BTC Price with Entry Signals')
    ax1.legend()
    
    # ============= MACD图表（带交易结果标记） =============
    ax2 = plt.subplot(212, sharex=ax1)
    
    # 绘制MACD指标
    df['macd'].plot(ax=ax2, color='blue', label='MACD (DIF)')
    df['signal'].plot(ax=ax2, color='orange', label='Signal (DEA)')
    ax2.bar(df.index, df['hist'], 
           color=df['hist'].apply(lambda x: 'green' if x>0 else 'red'),
           alpha=0.3, label='Histogram')
    ax2.axhline(0, color='gray', linestyle='--')
    
    # 在MACD图表上标记交易结果
    for _, trade in trades.iterrows():
        exit_time = trade['exit_time']
        macd_val = df.loc[exit_time, 'macd']
        
        if trade['pnl'] > 0:
            if trade['exit_type'] == 'take_profit':
                # 盈利止盈 - 绿色五角星
                ax2.scatter(exit_time, macd_val, color='lime', marker='*', 
                           s=200, zorder=5, label='Win (TP)')
            else:
                # 其他盈利情况 - 绿色圆圈
                ax2.scatter(exit_time, macd_val, color='green', marker='o', 
                           s=100, zorder=5, label='Win (Other)')
        else:
            if trade['exit_type'] == 'stop_loss':
                # 亏损止损 - 红色X
                ax2.scatter(exit_time, macd_val, color='red', marker='x', 
                           s=150, linewidth=2, zorder=5, label='Loss (SL)')
            else:
                # 其他亏损情况 - 红色方块
                ax2.scatter(exit_time, macd_val, color='darkred', marker='s', 
                           s=80, zorder=5, label='Loss (Other)')
    
    # 去重图例
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    ax2.set_title('MACD (12,26,9) with Trade Results')
    
    # 设置X轴格式
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('macd_results_chart.png', dpi=300)
    print("图表已保存为 macd_results_chart.png")
    plt.figure(figsize=(16, 12))
    
    # 价格图表
    ax1 = plt.subplot(211)
    df['close'].plot(ax=ax1, color='black', alpha=0.8, label='BTC Price')
    
    # 标记交易信号
    long_entries = df[df['golden_cross']]
    short_entries = df[df['death_cross']]
    ax1.scatter(long_entries.index, long_entries['close'], 
               color='green', marker='^', s=100, label='Long Entry (Golden Cross)')
    ax1.scatter(short_entries.index, short_entries['close'],
               color='red', marker='v', s=100, label='Short Entry (Death Cross)')
    
    # 标记平仓点
    for _, trade in trades.iterrows():
        color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
        marker = '*' if trade['exit_type'] == 'take_profit' else 'x'
        ax1.scatter(trade['exit_time'], trade['exit_price'],
                   color=color, marker=marker, s=150, 
                   label=f"{'Win' if trade['pnl']>0 else 'Loss'} {trade['exit_type']}")
    
    ax1.set_title('BTC Price with MACD Cross Signals')
    ax1.legend()
    
    # MACD图表
    ax2 = plt.subplot(212, sharex=ax1)
    df['macd'].plot(ax=ax2, color='blue', label='MACD (DIF)')
    df['signal'].plot(ax=ax2, color='orange', label='Signal (DEA)')
    ax2.bar(df.index, df['hist'], 
           color=df['hist'].apply(lambda x: 'green' if x>0 else 'red'),
           alpha=0.3, label='Histogram')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('MACD (12,26,9)')
    ax2.legend()
    
    # 设置X轴格式
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=120))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('macd_crossbacktest.png', dpi=300)
    print("图表已保存为 macd_crossbacktest.png")

# ========================
# 主程序
# ========================

def macd():
    print("获取数据中...")
    btc_data = get_btc_data(years=5)
    
    print("计算MACD指标...")
    btc_data = calculate_macd(btc_data)
    
    print("运行回测...")
    trades_df = macd_backtest(btc_data)
    
    print("分析结果...")
    analyze_results(trades_df)
    
    print("生成可视化图表...")
    visualize_results(btc_data, trades_df)

# ============= RSI交易策略 =============
def rsi_strategy(data, overbought=70, oversold=30, risk_reward=2):
    signals = []
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None
    
    for i in range(len(data)):
        current_rsi = data['rsi'].iloc[i]
        current_price = data['close'].iloc[i]
        
        # 超卖信号 - 做多
        if current_rsi < oversold and position != 'long':
            if position == 'short':
                # 平空仓
                pnl = entry_price - current_price
                signals.append(('short_exit', data.index[i], current_price, pnl))
            
            # 开多仓
            position = 'long'
            entry_price = current_price
            stop_loss = entry_price * 0.99  # 1%止损
            take_profit = entry_price * 1.02  # 2%止盈（盈亏比2:1）
            signals.append(('long_entry', data.index[i], current_price, None))
        
        # 超买信号 - 做空
        elif current_rsi > overbought and position != 'short':
            if position == 'long':
                # 平多仓
                pnl = current_price - entry_price
                signals.append(('long_exit', data.index[i], current_price, pnl))
            
            # 开空仓
            position = 'short'
            entry_price = current_price
            stop_loss = entry_price * 1.01  # 1%止损
            take_profit = entry_price * 0.98  # 2%止盈（盈亏比2:1）
            signals.append(('short_entry', data.index[i], current_price, None))
        
        # 检查止损止盈
        if position == 'long':
            if current_price <= stop_loss or current_price >= take_profit:
                pnl = current_price - entry_price
                signals.append(('long_exit', data.index[i], current_price, pnl))
                position = None
        elif position == 'short':
            if current_price >= stop_loss or current_price <= take_profit:
                pnl = entry_price - current_price
                signals.append(('short_exit', data.index[i], current_price, pnl))
                position = None
    
    return pd.DataFrame(signals, columns=['action', 'time', 'price', 'pnl'])

# ============= 绩效分析 =============
def analyze_performance(trades):
    # 过滤出平仓交易
    closed_trades = trades[trades['action'].str.contains('exit')]
    
    # 基础统计
    total_trades = len(closed_trades)
    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100
    total_pnl = closed_trades['pnl'].sum()
    avg_pnl = closed_trades['pnl'].mean()
    
    print("\n========== 策略绩效 ==========")
    print(f"总交易次数: {total_trades}")
    print(f"盈利次数: {winning_trades} (胜率: {win_rate:.1f}%)")
    print(f"总盈亏: {total_pnl:.2f} USDT")
    print(f"平均每笔盈亏: {avg_pnl:.2f} USDT")
    trades.to_csv('rsibacktest_results.csv', index=False)
    print("\n交易记录已保存到 rsibacktest_results.csv")
    
    return closed_trades

def time_to_index(data, time_series):
    return [data.index.get_loc(t) for t in time_series]

# def plot_rsi_signals(data, trades):
#     plt.figure(figsize=(16, 14))
    
#     # ===== 1. 价格图表 =====
#     ax1 = plt.subplot(211)
#     ax1.plot(data.index, data['close'], color='black', linewidth=1.5, label='BTC Price')
    
#     # 标记买卖信号（价格图）
#     long_entries = trades[trades['action'] == 'long_entry']
#     short_entries = trades[trades['action'] == 'short_entry']
#     exits = trades[trades['action'].str.contains('exit')]
    
#     # 入场信号（价格图）
#     ax1.scatter(long_entries['time'], long_entries['price'], 
#                color='limegreen', marker='^', s=120, label='Long Entry (RSI<30)', zorder=5)
#     ax1.scatter(short_entries['time'], short_entries['price'],
#                color='red', marker='v', s=120, label='Short Entry (RSI>70)', zorder=5)
    
#     # 平仓信号（价格图，区分盈亏）
#     for _, trade in exits.iterrows():
#         color = 'darkgreen' if trade['pnl'] > 0 else 'maroon'
#         marker = '*' if trade['pnl'] > 0 else 'x'
#         label = 'Profit Exit' if trade['pnl'] > 0 else 'Loss Exit'
#         ax1.scatter(trade['time'], trade['price'], color=color, marker=marker,
#                    s=150, label=label, zorder=5)
    
#     ax1.set_title('BTC Price with RSI Trading Signals', fontsize=14, pad=20)
#     ax1.set_ylabel('Price (USDT)', fontsize=12)
#     ax1.legend(loc='upper left')
#     ax1.grid(True, linestyle='--', alpha=0.3)

#     ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#     plt.xticks(rotation=45)
    
#     # ===== 2. RSI图表 =====
#     ax2 = plt.subplot(212, sharex=ax1)
#     ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5, label='RSI (14)')
    
#     # 标记RSI信号时，使用索引位置而非时间戳
#     ax2.scatter(
#         time_to_index(data, long_entries['time']),  # x坐标：数据中的索引位置
#         data.loc[long_entries['time'], 'rsi'],     # y坐标：对应的RSI值
#         color='limegreen', marker='^', s=80, label='Long Signal', zorder=5
#     )
#     ax2.scatter(
#         time_to_index(data, short_entries['time']),
#         data.loc[short_entries['time'], 'rsi'],
#         color='red', marker='v', s=80, label='Short Signal', zorder=5
# )
    
#     # 超买/超卖区域
#     ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
#     ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
#     ax2.fill_between(data.index, 70, 100, color='red', alpha=0.1)
#     ax2.fill_between(data.index, 0, 30, color='green', alpha=0.1)
    
#     ax2.set_title('RSI (14) with Signal Markers', fontsize=14, pad=20)
#     ax2.set_ylabel('RSI', fontsize=12)
#     ax2.set_ylim(0, 100)
#     ax2.legend(loc='upper left')
#     ax2.grid(True, linestyle='--', alpha=0.3)
    
#     # ===== 3. 统一调整 =====
#     # X轴格式（间隔3天）
#     ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#     plt.xticks(rotation=45)
    
#     # 去重图例
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
#     ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig('rsi_signals_marked.png', dpi=300, bbox_inches='tight')
#     print("图表已保存为 rsi_signals_marked.png")
#     plt.show()
def plot_rsi_signals(data, trades):
    plt.figure(figsize=(16, 14))
    
    # ===== 1. 价格图表 =====
    ax1 = plt.subplot(211)
    ax1.plot(data.index, data['close'], color='black', linewidth=1.5, label='BTC Price')
    
    # 标记买卖信号（价格图）
    long_entries = trades[trades['action'] == 'long_entry']
    short_entries = trades[trades['action'] == 'short_entry']
    exits = trades[trades['action'].str.contains('exit')]

    # 入场信号（统一label）
    ax1.scatter(long_entries['time'], long_entries['price'], 
               color='limegreen', marker='^', s=120, label='Long Entry', zorder=5)
    ax1.scatter(short_entries['time'], short_entries['price'],
               color='red', marker='v', s=120, label='Short Entry', zorder=5)

    # 平仓信号（不添加label）
    for _, trade in exits.iterrows():
        color = 'darkgreen' if trade['pnl'] > 0 else 'maroon'
        marker = '*' if trade['pnl'] > 0 else 'x'
        ax1.scatter(trade['time'], trade['price'], color=color, marker=marker,
                   s=150, zorder=5, label=None)

    # ===== 2. RSI图表 =====
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5, label='RSI (14)')

    # 标记买卖信号（RSI图，统一label）
    ax2.scatter(
        long_entries['time'], data.loc[long_entries['time'], 'rsi'],
        color='limegreen', marker='^', s=80, label='Long Signal', zorder=5
    )
    ax2.scatter(
        short_entries['time'], data.loc[short_entries['time'], 'rsi'],
        color='red', marker='v', s=80, label='Short Signal', zorder=5
    )

    # 阈值线和填充区域（不添加label）
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label=None)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, label=None)
    ax2.fill_between(data.index, 70, 100, color='red', alpha=0.1, label=None)
    ax2.fill_between(data.index, 0, 30, color='green', alpha=0.1, label=None)

    # 手动控制图例（避免重复）
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))  # 去重
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()
    plt.savefig('rsi_signals_fixed.png', dpi=300)
    plt.show()
def rsi():
    btc_data = get_btc_data(years=5)
    btc_data['rsi'] = ta.momentum.rsi(btc_data['close'], window=14)
    
    # 运行策略
    trades = rsi_strategy(btc_data)
    performance = analyze_performance(trades)
    plot_rsi_signals(btc_data, trades)
if __name__ == "__main__":
    macd()
    # rsi()
    # btc = get_btc_data(years=3)
    # print(len(btc))
