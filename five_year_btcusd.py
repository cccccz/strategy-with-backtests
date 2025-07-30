import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.trend import MACD  
import ta
from datetime import datetime, timedelta
import math
from scipy.signal import argrelextrema
import numpy as np


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

#     # 仅标记入场点
#     long_entries = df[df['golden_cross']]
#     short_entries = df[df['death_cross']]
#     ax1.scatter(long_entries.index, long_entries['close'], 
#                color='green', marker='^', s=40, label='Long Entry')
#     ax1.scatter(short_entries.index, short_entries['close'],
#                color='red', marker='v', s=40, label='Short Entry')
#     ax1.set_title('BTC Price with Entry Signals')
#     ax1.legend()
    
#     # ============= MACD图表（带交易结果标记） =============
#     ax2 = plt.subplot(212, sharex=ax1)
    
#     # 绘制MACD指标
#     df['macd'].plot(ax=ax2, color='blue', label='MACD (DIF)')
#     df['signal'].plot(ax=ax2, color='orange', label='Signal (DEA)')
#     ax2.bar(df.index, df['hist'], 
#            color=df['hist'].apply(lambda x: 'green' if x>0 else 'red'),
#            alpha=0.3, label='Histogram')
#     ax2.axhline(0, color='gray', linestyle='--')
    
#     # 在MACD图表上标记交易结果
#     for _, trade in trades.iterrows():
#         exit_time = trade['exit_time']
#         macd_val = df.loc[exit_time, 'macd']
        
#         if trade['pnl'] > 0:
#             if trade['exit_type'] == 'take_profit':
#                 # 盈利止盈 - 绿色五角星
#                 ax2.scatter(exit_time, macd_val, color='lime', marker='*', 
#                            s=200, zorder=5, label='Win (TP)')
#            
#     handles, labels = ax2.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    
#     ax2.set_title('MACD (12,26,9) with Trade Results')
    
#     # 设置X轴格式
#     ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
#     plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.savefig('macd_results_chart.png', dpi=300)
#     print("图表已保存为 macd_results_chart.png")
#     plt.figure(figsize=(16, 12))
    
#     # 价格图表
#     ax1 = plt.subplot(211)
#     df['close'].plot(ax=ax1, color='black', alpha=0.8, label='BTC Price')
    
#     # 标记交易信号
#     long_entries = df[df['golden_cross']]
#     short_entries = df[df['death_cross']]
#     ax1.scatter(long_entries.index, long_entries['close'], 
#                color='green', marker='^', s=100, label='Long Entry (Golden Cross)')
#     ax1.scatter(short_entries.index, short_entries['close'],
#                color='red', marker='v', s=100, label='Short Entry (Death Cross)')
    
#     # 标记平仓点
#     for _, trade in trades.iterrows():
#         color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
#         marker = '*' if trade['exit_type'] == 'take_profit' else 'x'
#         ax1.scatter(trade['exit_time'], trade['exit_price'],
#                    color=color, marker=marker, s=150, 
#                    label=f"{'Win' if trade['pnl']>0 else 'Loss'} {trade['exit_type']}")
    
#     ax1.set_title('BTC Price with MACD Cross Signals')
#     ax1.legend()
    
#     # MACD图表
#     ax2 = plt.subplot(212, sharex=ax1)
#     df['macd'].plot(ax=ax2, color='blue', label='MACD (DIF)')
#     df['signal'].plot(ax=ax2, color='orange', label='Signal (DEA)')
#     ax2.bar(df.index, df['hist'], 
#            color=df['hist'].apply(lambda x: 'green' if x>0 else 'red'),
#            alpha=0.3, label='Histogram')
#     ax2.axhline(0, color='gray', linestyle='--')
#     ax2.set_title('MACD (12,26,9)')
#     ax2.legend()
    
#     # 设置X轴格式
#     ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
#     plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.savefig('macd_crossbacktest.png', dpi=300)
#     print("图表已保存为 macd_crossbacktest.png")

def plot_macd_signals(data, trades):
    plt.figure(figsize=(16, 14))
    
    # ===== 1. 价格图表 =====
    ax1 = plt.subplot(211)
    ax1.plot(data.index, data['close'], color='black', linewidth=1.5, label='BTC Price')
    
    # 标记买卖信号（价格图）
    long_entries = data[data['golden_cross']]
    short_entries = data[data['death_cross']]
    
    # 入场信号（统一label）
    ax1.scatter(long_entries.index, long_entries['close'], 
               color='limegreen', marker='^', s=120, label='Long Entry', zorder=5)
    ax1.scatter(short_entries.index, short_entries['close'],
               color='red', marker='v', s=120, label='Short Entry', zorder=5)

    # 平仓信号（修正：使用trades DataFrame的实际结构）
    if not trades.empty:
        # 分离盈利和亏损的交易
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        # 标记盈利的平仓点
        if not winning_trades.empty:
            ax1.scatter(winning_trades['exit_time'], winning_trades['exit_price'], 
                       color='darkgreen', marker='*', s=150, zorder=5, label='Exit (Win)')
        
        # 标记亏损的平仓点
        if not losing_trades.empty:
            ax1.scatter(losing_trades['exit_time'], losing_trades['exit_price'],
                       color='maroon', marker='x', s=150, zorder=5, label='Exit (Loss)')

    # ===== 2. MACD图表 =====
    ax2 = plt.subplot(212, sharex=ax1)
    
    # 绘制MACD指标
    ax2.plot(data.index, data['macd'], color='blue', linewidth=1.5, label='MACD')
    ax2.plot(data.index, data['signal'], color='orange', linewidth=1.5, label='Signal')
    
    # 柱状图（不添加label）
    ax2.bar(data.index, data['hist'], 
           color=data['hist'].apply(lambda x: 'green' if x>0 else 'red'),
           alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)

    # 标记买卖信号（MACD图，统一label）
    ax2.scatter(
        long_entries.index, data.loc[long_entries.index, 'macd'],
        color='limegreen', marker='^', s=80, label='Long Signal', zorder=5
    )
    ax2.scatter(
        short_entries.index, data.loc[short_entries.index, 'macd'],
        color='red', marker='v', s=80, label='Short Signal', zorder=5
    )

    # 手动控制图例（避免重复）
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # 自定义图例顺序和去重
    legend_order = ['BTC Price', 'Long Entry', 'Short Entry', 'Exit (Win)', 'Exit (Loss)',
                   'MACD', 'Signal', 'Long Signal', 'Short Signal']
    by_label = dict(zip(labels, handles))
    filtered_handles = [by_label[label] for label in legend_order if label in by_label]
    filtered_labels = [label for label in legend_order if label in by_label]
    
    ax1.legend(filtered_handles, filtered_labels, loc='upper left')
    
    # 设置X轴格式
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('macd_signals_fixed.png', dpi=300)
    plt.show()

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
    plot_macd_signals(btc_data, trades_df)

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


from scipy.signal import argrelextrema
import numpy as np

def detect_divergence(df, order=5, lookback=20,ref_idx='macd'):
    """
    检测 MACD 与价格之间的顶背离/底背离（带lookback）
    """
    df = df.copy()

    # 寻找局部极值
    price_max_idx = argrelextrema(df['close'].values, np.greater_equal, order=order)[0]
    price_min_idx = argrelextrema(df['close'].values, np.less_equal, order=order)[0]
    macd_max_idx = argrelextrema(df[ref_idx].values, np.greater_equal, order=order)[0]
    macd_min_idx = argrelextrema(df[ref_idx].values, np.less_equal, order=order)[0]

    df['price_local_max'] = np.nan
    df['price_local_min'] = np.nan
    df['macd_local_max'] = np.nan
    df['macd_local_min'] = np.nan

    df.loc[df.index[price_max_idx], 'price_local_max'] = df['close'].iloc[price_max_idx]
    df.loc[df.index[price_min_idx], 'price_local_min'] = df['close'].iloc[price_min_idx]
    df.loc[df.index[macd_max_idx], 'macd_local_max'] = df[ref_idx].iloc[macd_max_idx]
    df.loc[df.index[macd_min_idx], 'macd_local_min'] = df[ref_idx].iloc[macd_min_idx]

    df['top_divergence'] = False
    df['bottom_divergence'] = False

    # 顶背离：价格创新高 + MACD 走低
    for i in price_max_idx:
        for j in price_max_idx:
            if j < i and (i - j) <= lookback:
                if df['close'].iloc[i] > df['close'].iloc[j] and df['macd'].iloc[i] < df['macd'].iloc[j]:
                    df.at[df.index[i], 'top_divergence'] = True
                    break

    # 底背离：价格创新低 + MACD 走高
    for i in price_min_idx:
        for j in price_min_idx:
            if j < i and (i - j) <= lookback:
                if df['close'].iloc[i] < df['close'].iloc[j] and df['macd'].iloc[i] > df['macd'].iloc[j]:
                    df.at[df.index[i], 'bottom_divergence'] = True
                    break

    return df

import matplotlib.pyplot as plt

def extrema_divergence_check(df, order=5, lookback=20):
    df = detect_divergence(df, order=order, lookback=lookback)

    # 提取极值索引用于绘图
    price_max_idx = df[df['price_local_max'].notna()].index
    price_min_idx = df[df['price_local_min'].notna()].index
    macd_max_idx = df[df['macd_local_max'].notna()].index
    macd_min_idx = df[df['macd_local_min'].notna()].index

    # 绘图部分
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # ax1: 价格图 + 极值点
    ax1.plot(df.index, df['close'], label='BTC Price', color='black')
    ax1.scatter(price_max_idx, df.loc[price_max_idx, 'close'], color='red', label='Local Max', marker='^')
    ax1.scatter(price_min_idx, df.loc[price_min_idx, 'close'], color='green', label='Local Min', marker='v')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('Price with Local Extrema')

    # ax2: MACD 图 + 极值点
    ax2.plot(df.index, df['macd'], label='MACD', color='blue')
    ax2.plot(df.index, df['signal'], label='Signal', color='orange')
    ax2.bar(df.index, df['hist'], color=df['hist'].apply(lambda x: 'green' if x >= 0 else 'red'), alpha=0.3)
    ax2.scatter(macd_max_idx, df.loc[macd_max_idx, 'macd'], color='red', marker='^', label='MACD Max')
    ax2.scatter(macd_min_idx, df.loc[macd_min_idx, 'macd'], color='green', marker='v', label='MACD Min')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.set_title('MACD and Histogram with Extrema')

    # ax3: 叠加图 + 背离信号
    ax3.plot(df.index, df['close'], label='Price', color='gray')
    ax3.plot(df.index, df['macd'], label='MACD', color='blue', alpha=0.5)
    ax3.scatter(df[df['top_divergence']].index, df[df['top_divergence']]['close'],
                color='purple', marker='x', s=100, label='Top Divergence')
    ax3.scatter(df[df['bottom_divergence']].index, df[df['bottom_divergence']]['close'],
                color='orange', marker='o', s=100, label='Bottom Divergence')
    ax3.set_ylabel('Price / MACD')
    ax3.legend()
    ax3.set_title('Divergence Signals Overlay')

    plt.tight_layout()
    plt.savefig('divergence_check.png', dpi=300)
    plt.show()

    return df

def divergence_backtest(df, stop_loss=0.01, take_profit=0.02):
    """
    回测基于 divergence 信号的交易策略（盈亏比 2:1）- 带 DEBUG 输出
    """
    trades = []
    position = None
    entry_price = None
    entry_time = None

    for i, row in df.iterrows():
        timestamp = i.strftime('%Y-%m-%d %H:%M')
        close = row['close']
        high = row['high']
        low = row['low']

        # ======= 持仓检查 =======
        if position == 'long':
            print(f"[{timestamp}] Long 持仓中，当前价格：{close:.2f}")
            if low <= entry_price * (1 - stop_loss):  # 止损
                exit_price = entry_price * (1 - stop_loss)
                pnl = -stop_loss
                exit_type = 'stop_loss'
                close_pos = True
                print(f"→ LONG 止损触发！入场价 {entry_price:.2f}，止损价 {exit_price:.2f}")
            elif high >= entry_price * (1 + take_profit):  # 止盈
                exit_price = entry_price * (1 + take_profit)
                pnl = take_profit
                exit_type = 'take_profit'
                close_pos = True
                print(f"→ LONG 止盈触发！入场价 {entry_price:.2f}，止盈价 {exit_price:.2f}")
            else:
                close_pos = False

        elif position == 'short':
            print(f"[{timestamp}] Short 持仓中，当前价格：{close:.2f}")
            if high >= entry_price * (1 + stop_loss):  # 止损
                exit_price = entry_price * (1 + stop_loss)
                pnl = -stop_loss
                exit_type = 'stop_loss'
                close_pos = True
                print(f"→ SHORT 止损触发！入场价 {entry_price:.2f}，止损价 {exit_price:.2f}")
            elif low <= entry_price * (1 - take_profit):  # 止盈
                exit_price = entry_price * (1 - take_profit)
                pnl = take_profit
                exit_type = 'take_profit'
                close_pos = True
                print(f"→ SHORT 止盈触发！入场价 {entry_price:.2f}，止盈价 {exit_price:.2f}")
            else:
                close_pos = False
        else:
            close_pos = False

        # 平仓执行
        if close_pos:
            trades.append({
                'entry_time': entry_time,
                'exit_time': i,
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_type': exit_type,
                'pnl': pnl
            })
            print(f"📌 平仓完成：{position.upper()} @ {entry_price:.2f} → {exit_price:.2f} | PnL: {pnl:.2%}\n")
            position = None
            entry_price = None
            entry_time = None

        # ======= 入场判断 =======
        if position is None:
            if row.get('bottom_divergence'):
                position = 'long'
                entry_price = close
                entry_time = i
                print(f"[{timestamp}] 📈 检测到 BOTTOM 背离信号，开多单 @ {entry_price:.2f}")
            elif row.get('top_divergence'):
                position = 'short'
                entry_price = close
                entry_time = i
                print(f"[{timestamp}] 📉 检测到 TOP 背离信号，开空单 @ {entry_price:.2f}")

    print(f"\n✅ 回测完成，总交易数: {len(trades)}")
    return pd.DataFrame(trades)


def analyze_results_divergence(trades):
    if trades.empty:
        print("❌ 没有任何交易")
        return

    total = len(trades)
    win = trades[trades['pnl'] > 0]
    loss = trades[trades['pnl'] <= 0]
    win_rate = len(win) / total
    avg_win = win['pnl'].mean()
    avg_loss = loss['pnl'].mean()
    total_return = trades['pnl'].sum()

    print("="*40)
    print("📊 回测结果")
    print("="*40)
    print(f"总交易数      : {total}")
    print(f"盈利交易数    : {len(win)}")
    print(f"亏损交易数    : {len(loss)}")
    print(f"胜率          : {win_rate:.2%}")
    print(f"平均盈利      : {avg_win:.2%}")
    print(f"平均亏损      : {avg_loss:.2%}")
    print(f"盈亏比        : {abs(avg_win / avg_loss):.2f}")
    print(f"累计收益率    : {total_return:.2%}")

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_divergence_signals(df, trades=None, title="Divergence Signals & Trades",filename='divergence_and_trade_signal.png'):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df.index, df['close'], label='Price', color='black', linewidth=1)

    # --- 背离信号（非交易） ---
    top_signals = df[df['top_divergence']]
    bottom_signals = df[df['bottom_divergence']]
    ax.scatter(top_signals.index, top_signals['close'], marker='v', color='red', s=30, label='Top Divergence')
    ax.scatter(bottom_signals.index, bottom_signals['close'], marker='^', color='green', s=30, label='Bottom Divergence')

    if trades is not None and not trades.empty:
        # 分组：Long / Short 入场点
        long_trades = trades[trades['position'] == 'long']
        short_trades = trades[trades['position'] == 'short']

        # 平仓也分：盈利 / 亏损
        winning = trades[trades['pnl'] > 0]
        losing = trades[trades['pnl'] <= 0]

        # --- 入场点 ---
        ax.scatter(long_trades['entry_time'], long_trades['entry_price'], 
                   color='limegreen', marker='o', s=30, label='Long Entry', zorder=5, edgecolors='black')
        ax.scatter(short_trades['entry_time'], short_trades['entry_price'], 
                   color='crimson', marker='o', s=30, label='Short Entry', zorder=5, edgecolors='black')

        # --- 平仓点 ---
        ax.scatter(winning['exit_time'], winning['exit_price'], 
                   color='darkgreen', marker='*', s=30, label='Exit (Win)', zorder=5)
        ax.scatter(losing['exit_time'], losing['exit_price'], 
                   color='gray', marker='x', s=30, label='Exit (Loss)', zorder=5)

        # --- 连线交易路径 ---
        for _, row in trades.iterrows():
            ax.plot([row['entry_time'], row['exit_time']],
                    [row['entry_price'], row['exit_price']],
                    color='green' if row['pnl'] > 0 else 'red',
                    linestyle='--', alpha=0.6)

    # --- 图例去重（专业版） ---
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    # macd()
    # rsi()
    btc = get_btc_data(years=5)
    btc = calculate_macd(btc)
    btc = detect_divergence(btc)
    trades = divergence_backtest(btc)
    analyze_results_divergence(trades)
    plot_divergence_signals(btc,trades=trades)

    
