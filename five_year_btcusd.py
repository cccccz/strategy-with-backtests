import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.trend import MACD  # 使用 ta 库计算 MACD
from datetime import datetime, timedelta

# 获取近 3 年比特币数据
def get_btc_data(years=3, extra_days=26):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365 * years + extra_days)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '4h',
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': 6000 + extra_days
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    return df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]

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

# print_crosses()

# def backtest(df, risk_reward_ratio=2):
    df['position'] = 0          # 0=空仓, 1=多单, -1=空单
    df['entry_price'] = 0.0     # 入场价格
    df['stop_loss'] = 0.0       # 止损价
    df['take_profit'] = 0.0     # 止盈价
    df['pnl'] = 0.0             # 盈亏百分比
    df['exit_type'] = None      # 新增：标记平仓类型
    
    for i in range(26, len(df)):
        # 金叉做多
        if df['golden_cross'].iloc[i]:
            df['position'].iloc[i] = 1
            entry = df['close'].iloc[i]
            df['entry_price'].iloc[i] = entry
            df['stop_loss'].iloc[i] = entry * 0.99  # 止损1%
            df['take_profit'].iloc[i] = entry * 1.02  # 止盈2%
        
        # 死叉做空
        elif df['death_cross'].iloc[i]:
            df['position'].iloc[i] = -1
            entry = df['close'].iloc[i]
            df['entry_price'].iloc[i] = entry
            df['stop_loss'].iloc[i] = entry * 1.01  # 止损1%
            df['take_profit'].iloc[i] = entry * 0.98  # 止盈2%
        
        # 检查平仓条件
        if df['position'].iloc[i-1] != 0:
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            
            # 多单平仓逻辑
            if df['position'].iloc[i-1] == 1:
                if current_low <= df['stop_loss'].iloc[i-1]:
                    df['pnl'].iloc[i] = -0.01
                    df['exit_type'].iloc[i] = 'stop_loss'  # 标记止损
                elif current_high >= df['take_profit'].iloc[i-1]:
                    df['pnl'].iloc[i] = 0.02
                    df['exit_type'].iloc[i] = 'take_profit'  # 标记止盈
            
            # 空单平仓逻辑
            elif df['position'].iloc[i-1] == -1:
                if current_high >= df['stop_loss'].iloc[i-1]:
                    df['pnl'].iloc[i] = -0.01
                    df['exit_type'].iloc[i] = 'stop_loss'
                elif current_low <= df['take_profit'].iloc[i-1]:
                    df['pnl'].iloc[i] = 0.02
                    df['exit_type'].iloc[i] = 'take_profit'
    
    return df

# def backtest(df):
    df['position'] = 0
    df['entry_price'] = 0.0
    df['stop_loss'] = 0.0
    df['take_profit'] = 0.0
    df['pnl'] = 0.0
    df['exit_type'] = None

    for i in range(26, len(df)):
        # 修复：确保在信号触发时记录当前K线的收盘价
        if df['golden_cross'].iloc[i]:
            df['position'].iloc[i] = 1
            df['entry_price'].iloc[i] = df['close'].iloc[i]  # 用当前K线收盘价入场
            df['stop_loss'].iloc[i] = df['close'].iloc[i] * 0.99
            df['take_profit'].iloc[i] = df['close'].iloc[i] * 1.02

        elif df['death_cross'].iloc[i]:
            df['position'].iloc[i] = -1
            df['entry_price'].iloc[i] = df['close'].iloc[i]  # 用当前K线收盘价入场
            df['stop_loss'].iloc[i] = df['close'].iloc[i] * 1.01
            df['take_profit'].iloc[i] = df['close'].iloc[i] * 0.98

        # 平仓逻辑保持不变
        if df['position'].iloc[i-1] != 0:
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            entry = df['entry_price'].iloc[i-1]  # 使用前一根K线的入场价

            if df['position'].iloc[i-1] == 1:  # 多单
                if current_low <= df['stop_loss'].iloc[i-1]:
                    df['pnl'].iloc[i] = -0.01
                    df['exit_type'].iloc[i] = 'stop_loss'
                elif current_high >= df['take_profit'].iloc[i-1]:
                    df['pnl'].iloc[i] = 0.02
                    df['exit_type'].iloc[i] = 'take_profit'

            elif df['position'].iloc[i-1] == -1:  # 空单
                if current_high >= df['stop_loss'].iloc[i-1]:
                    df['pnl'].iloc[i] = -0.01
                    df['exit_type'].iloc[i] = 'stop_loss'
                elif current_low <= df['take_profit'].iloc[i-1]:
                    df['pnl'].iloc[i] = 0.02
                    df['exit_type'].iloc[i] = 'take_profit'
    return df

# def analyze_results(df):
    trades = df[df['pnl'] != 0].copy()
    
    # 分类统计
    stop_loss_trades = trades[trades['exit_type'] == 'stop_loss']
    take_profit_trades = trades[trades['exit_type'] == 'take_profit']
    
    # 基础统计
    win_rate = len(take_profit_trades) / len(trades) if len(trades) > 0 else 0
    avg_win = take_profit_trades['pnl'].mean()
    avg_loss = stop_loss_trades['pnl'].mean()
    
    # 输出详细结果
    print("\n=== 回测结果 ===")
    print(f"总交易次数: {len(trades)}")
    print(f"止损次数: {len(stop_loss_trades)}")
    print(f"止盈次数: {len(take_profit_trades)}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均盈利: {avg_win:.2%}, 平均亏损: {avg_loss:.2%}")
    print(f"盈亏比: {-avg_win/avg_loss:.2f}" if avg_loss !=0 else "N/A")
    
    # 打印最近5次交易的平仓详情
    print("\n=== 最近5次交易详情 ===")
    for idx, row in trades.tail(5).iterrows():
        direction = "多单" if row['position'] == 1 else "空单"
        print(
            f"{idx.strftime('%Y-%m-%d %H:%M')} | {direction} | "
            f"入场价: {row['entry_price']:.2f} | "
            f"结果: {'止盈' if row['exit_type'] == 'take_profit' else '止损'} | "
            f"盈亏: {row['pnl']*100:.2f}%"
        )

# def run_backtest(data):
    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for i, row in data.iterrows():
        # 开仓逻辑
        if position is None:
            if row['golden_cross']:
                position = 'long'
                entry_price = row['close']
                stop_loss = entry_price * 0.99  # 1%止损
                take_profit = entry_price * 1.02  # 2%止盈
                trades.append({
                    'entry_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
            elif row['death_cross']:
                position = 'short'
                entry_price = row['close']
                stop_loss = entry_price * 1.01  # 1%止损
                take_profit = entry_price * 0.98  # 2%止盈
                trades.append({
                    'entry_time': i,
                    'position': position,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
        
        # 平仓逻辑
        elif position == 'long':
            if row['low'] <= stop_loss:
                trades[-1].update({
                    'exit_time': i,
                    'exit_price': stop_loss,
                    'exit_type': 'stop_loss',
                    'pnl': (stop_loss - entry_price)/entry_price
                })
                position = None
            elif row['high'] >= take_profit:
                trades[-1].update({
                    'exit_time': i,
                    'exit_price': take_profit,
                    'exit_type': 'take_profit',
                    'pnl': (take_profit - entry_price)/entry_price
                })
                position = None
                
        elif position == 'short':
            if row['high'] >= stop_loss:
                trades[-1].update({
                    'exit_time': i,
                    'exit_price': stop_loss,
                    'exit_type': 'stop_loss',
                    'pnl': (entry_price - stop_loss)/entry_price
                })
                position = None
            elif row['low'] <= take_profit:
                trades[-1].update({
                    'exit_time': i,
                    'exit_price': take_profit,
                    'exit_type': 'take_profit',
                    'pnl': (entry_price - take_profit)/entry_price
                })
                position = None
    
    return pd.DataFrame(trades)

# ========================
# 分析结果
# ========================
# def analyze_results(trades):
    if trades.empty:
        print("没有交易发生")
        return
    
    # 基础统计
    total_trades = len(trades)
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    
    print("\n" + "="*50)
    print("回测结果汇总")
    print("="*50)
    print(f"总交易次数: {total_trades}")
    print(f"盈利交易: {len(winning_trades)} ({len(winning_trades)/total_trades:.1%})")
    print(f"亏损交易: {len(losing_trades)} ({len(losing_trades)/total_trades:.1%})")
    print(f"平均盈利: {winning_trades['pnl'].mean():.2%}")
    print(f"平均亏损: {losing_trades['pnl'].mean():.2%}")
    print(f"盈亏比: {-winning_trades['pnl'].mean()/losing_trades['pnl'].mean():.2f}")
    print(f"累计收益率: {trades['pnl'].sum():.2%}")
    
    # 最近5笔交易详情
    print("\n最近5笔交易详情:")
    print(trades[['entry_time', 'position', 'entry_price', 
                 'exit_price', 'exit_type', 'pnl']].tail(5).to_string())
    
    # 保存完整交易记录
    trades.to_csv('trade_log.csv', index=False)
    print("\n完整交易记录已保存到 trade_log.csv")

# ========================
# 可视化
# ========================
# def visualize(data, trades):
    plt.figure(figsize=(16, 12))
    
    # 价格图表
    ax1 = plt.subplot(211)
    data['close'].plot(ax=ax1, color='black', alpha=0.8, label='Price')
    
    # 标记交易信号
    long_entries = data[data['golden_cross']]
    short_entries = data[data['death_cross']]
    ax1.scatter(long_entries.index, long_entries['close'], 
               color='green', marker='^', s=100, label='Long Entry')
    ax1.scatter(short_entries.index, short_entries['close'],
               color='red', marker='v', s=100, label='Short Entry')
    
    # 标记平仓点
    for _, trade in trades.iterrows():
        color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
        marker = '*' if trade['exit_type'] == 'take_profit' else 'x'
        ax1.scatter(trade['exit_time'], trade['exit_price'],
                   color=color, marker=marker, s=150, 
                   label=f"{'Win' if trade['pnl']>0 else 'Loss'} {trade['exit_type']}")
    
    ax1.set_title('Price and Trades')
    ax1.legend()
    
    # MACD图表
    ax2 = plt.subplot(212, sharex=ax1)
    data['macd'].plot(ax=ax2, color='blue', label='MACD')
    data['signal'].plot(ax=ax2, color='orange', label='Signal')
    ax2.bar(data.index, data['hist'], 
           color=data['hist'].apply(lambda x: 'green' if x>0 else 'red'),
           alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title('MACD (12,26,9)')
    ax2.legend()
    
    # 设置X轴格式
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('backtest_result.png')
    print("\n图表已保存为 backtest_result.png")


# ========================
# 回测引擎
# ========================
def run_backtest(df):
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
if __name__ == "__main__":
    print("获取数据中...")
    btc_data = get_btc_data(years=5)
    
    print("计算MACD指标...")
    btc_data = calculate_macd(btc_data)
    
    print("运行回测...")
    trades_df = run_backtest(btc_data)
    
    print("分析结果...")
    analyze_results(trades_df)
    
    print("生成可视化图表...")
    visualize_results(btc_data, trades_df)