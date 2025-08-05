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
    # return format
    # trades: pd.DataFrame，每行代表一笔完整交易，列结构如下：
    # - entry_time:   datetime，入场时间
    # - exit_time:    datetime，出场时间
    # - position:     str，方向，'long' 或 'short'
    # - entry_price:  float，入场价格
    # - exit_price:   float，出场价格
    # - exit_type:    str，'take_profit' 或 'stop_loss'
    # - pnl:          float，收益率（相对，单位为比例，如 0.02 表示 +2%）
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

def analyze_trades(trades: pd.DataFrame, save_csv: bool = True, filename: str = "backtest_results.csv"):
    if trades.empty:
        print("⚠️ 没有交易数据")
        return

    # 基础统计
    total_trades = len(trades)
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    pnl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    total_return = trades['pnl'].sum()

    # 打印绩效
    print("\n" + "=" * 40)
    print("📊 策略绩效汇总")
    print("=" * 40)
    print(f"总交易次数      : {total_trades}")
    print(f"盈利交易笔数    : {len(winning_trades)}")
    print(f"亏损交易笔数    : {len(losing_trades)}")
    print(f"胜率            : {win_rate:.2%}")
    print(f"平均盈利        : {avg_win:.2%}")
    print(f"平均亏损        : {avg_loss:.2%}")
    print(f"盈亏比（R:R）   : {pnl_ratio:.2f}")
    print(f"累计收益率      : {total_return:.2%}")

    if save_csv:
        trades.to_csv(filename, index=False)
        print(f"\n💾 交易记录已保存至: {filename}")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pnl_ratio': pnl_ratio,
        'total_return': total_return
    }


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
    # analyze_results(trades_df)
    analyze_trades(trades_df)
    
    print("生成可视化图表...")
    plot_macd_signals(btc_data, trades_df)

def rsi_strategy(data, overbought=70, oversold=30, stop_loss_pct=0.01, take_profit_pct=0.02):
    # return format
    # trades: pd.DataFrame，每行代表一笔完整交易，列结构如下：
    # - entry_time:   datetime，入场时间
    # - exit_time:    datetime，出场时间
    # - position:     str，方向，'long' 或 'short'
    # - entry_price:  float，入场价格
    # - exit_price:   float，出场价格
    # - exit_type:    str，'take_profit' 或 'stop_loss'
    # - pnl:          float，收益率（相对，单位为比例，如 0.02 表示 +2%）

    trades = []
    position = None
    entry_price = None
    entry_time = None
    stop_loss = None
    take_profit = None

    for i in range(len(data)):
        row = data.iloc[i]
        time = data.index[i]
        rsi = row['rsi']
        price = row['close']

        # 平仓逻辑
        if position == 'long':
            if price <= stop_loss:
                exit_price = stop_loss
                pnl = (exit_price - entry_price) / entry_price
                exit_type = 'stop_loss'
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'position': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl
                })
                position = None
            elif price >= take_profit:
                exit_price = take_profit
                pnl = (exit_price - entry_price) / entry_price
                exit_type = 'take_profit'
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'position': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl
                })
                position = None

        elif position == 'short':
            if price >= stop_loss:
                exit_price = stop_loss
                pnl = (entry_price - exit_price) / entry_price
                exit_type = 'stop_loss'
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'position': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl
                })
                position = None
            elif price <= take_profit:
                exit_price = take_profit
                pnl = (entry_price - exit_price) / entry_price
                exit_type = 'take_profit'
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'position': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'pnl': pnl
                })
                position = None

        # 开仓逻辑
        if position is None:
            if rsi < oversold:
                position = 'long'
                entry_price = price
                entry_time = time
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)

            elif rsi > overbought:
                position = 'short'
                entry_price = price
                entry_time = time
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)

    return pd.DataFrame(trades)


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

    # 标记入场信号（价格图）
    long_entries = trades[trades['position'] == 'long']
    short_entries = trades[trades['position'] == 'short']

    ax1.scatter(long_entries['entry_time'], long_entries['entry_price'], 
                color='limegreen', marker='^', s=120, label='Long Entry', zorder=5)
    ax1.scatter(short_entries['entry_time'], short_entries['entry_price'],
                color='red', marker='v', s=120, label='Short Entry', zorder=5)

    # 平仓信号（盈利绿色星号，亏损红色叉）
    for _, row in trades.iterrows():
        color = 'darkgreen' if row['pnl'] > 0 else 'maroon'
        marker = '*' if row['pnl'] > 0 else 'x'
        ax1.scatter(row['exit_time'], row['exit_price'], color=color, marker=marker,
                    s=150, zorder=5, label=None)

    # ===== 2. RSI图表 =====
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(data.index, data['rsi'], color='purple', linewidth=1.5, label='RSI (14)')

    # RSI 入场信号位置（根据 entry_time）
    ax2.scatter(
        long_entries['entry_time'], data.loc[long_entries['entry_time'], 'rsi'],
        color='limegreen', marker='^', s=80, label='Long Signal', zorder=5
    )
    ax2.scatter(
        short_entries['entry_time'], data.loc[short_entries['entry_time'], 'rsi'],
        color='red', marker='v', s=80, label='Short Signal', zorder=5
    )

    # RSI 阈值线
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
    ax2.fill_between(data.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(data.index, 0, 30, color='green', alpha=0.1)

    # 图例去重
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()
    plt.savefig('rsi_signals_fixed.png', dpi=300)
    plt.show()


def rsi(years):
    btc_data = get_btc_data(years=years)
    btc_data['rsi'] = ta.momentum.rsi(btc_data['close'], window=14)
    
    # 运行策略
    trades = rsi_strategy(btc_data)
    # performance = analyze_performance(trades)
    performance = analyze_trades(trades)
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

def detect_divergence_no_lookahead(df, order=5, lookback=20, ref_idx='macd'):
    """
    return format
    trades: pd.DataFrame，每行代表一笔完整交易，列结构如下：
    - entry_time:   datetime，入场时间
    - exit_time:    datetime，出场时间
    - position:     str，方向，'long' 或 'short'
    - entry_price:  float，入场价格
    - exit_price:   float，出场价格
    - exit_type:    str，'take_profit' 或 'stop_loss'
    - pnl:          float，收益率（相对，单位为比例，如 0.02 表示 +2%）
    """
    df = df.copy()

    # 初始化列
    df['price_local_max'] = np.nan
    df['price_local_min'] = np.nan
    df['macd_local_max'] = np.nan
    df['macd_local_min'] = np.nan
    df['top_divergence'] = False
    df['bottom_divergence'] = False

    # ========== 1. 延迟确认极值 ==========
    price_max_idx = argrelextrema(df['close'].values, np.greater_equal, order=order)[0]
    price_min_idx = argrelextrema(df['close'].values, np.less_equal, order=order)[0]
    macd_max_idx = argrelextrema(df[ref_idx].values, np.greater_equal, order=order)[0]
    macd_min_idx = argrelextrema(df[ref_idx].values, np.less_equal, order=order)[0]

    # 回填信号到未来第 order 根K线
    for idx in price_max_idx:
        if idx + order < len(df):
            df.loc[df.index[idx + order], 'price_local_max'] = df['close'].iloc[idx]

    for idx in price_min_idx:
        if idx + order < len(df):
            df.loc[df.index[idx + order], 'price_local_min'] = df['close'].iloc[idx]

    for idx in macd_max_idx:
        if idx + order < len(df):
            df.loc[df.index[idx + order], 'macd_local_max'] = df[ref_idx].iloc[idx]

    for idx in macd_min_idx:
        if idx + order < len(df):
            df.loc[df.index[idx + order], 'macd_local_min'] = df[ref_idx].iloc[idx]

    # ========== 2. 检测背离 ==========
    for i in range(lookback + order, len(df)):
        # 顶背离：价格创新高 + MACD 走低
        recent_price_max = df['price_local_max'].iloc[i - lookback:i].dropna()
        recent_macd_max = df['macd_local_max'].iloc[i - lookback:i].dropna()
        if len(recent_price_max) >= 2 and len(recent_macd_max) >= 2:
            if recent_price_max.iloc[-1] > recent_price_max.iloc[-2] and \
               recent_macd_max.iloc[-1] < recent_macd_max.iloc[-2]:
                df.at[df.index[i], 'top_divergence'] = True

        # 底背离：价格创新低 + MACD 走高
        recent_price_min = df['price_local_min'].iloc[i - lookback:i].dropna()
        recent_macd_min = df['macd_local_min'].iloc[i - lookback:i].dropna()
        if len(recent_price_min) >= 2 and len(recent_macd_min) >= 2:
            if recent_price_min.iloc[-1] < recent_price_min.iloc[-2] and \
               recent_macd_min.iloc[-1] > recent_macd_min.iloc[-2]:
                df.at[df.index[i], 'bottom_divergence'] = True

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
    回测基于 divergence 信号的交易策略（盈亏比 2:1）- 简洁版（无调试输出）
    """
    trades = []
    position = None
    entry_price = None
    entry_time = None

    for i, row in df.iterrows():
        close = row['close']
        high = row['high']
        low = row['low']
        close_pos = False

        # ======= 持仓检查 =======
        if position == 'long':
            if low <= entry_price * (1 - stop_loss):  # 止损
                exit_price = entry_price * (1 - stop_loss)
                pnl = -stop_loss
                exit_type = 'stop_loss'
                close_pos = True
            elif high >= entry_price * (1 + take_profit):  # 止盈
                exit_price = entry_price * (1 + take_profit)
                pnl = take_profit
                exit_type = 'take_profit'
                close_pos = True

        elif position == 'short':
            if high >= entry_price * (1 + stop_loss):  # 止损
                exit_price = entry_price * (1 + stop_loss)
                pnl = -stop_loss
                exit_type = 'stop_loss'
                close_pos = True
            elif low <= entry_price * (1 - take_profit):  # 止盈
                exit_price = entry_price * (1 - take_profit)
                pnl = take_profit
                exit_type = 'take_profit'
                close_pos = True

        # ======= 平仓执行 =======
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
            position = None
            entry_price = None
            entry_time = None

        # ======= 入场判断 =======
        if position is None:
            if row.get('bottom_divergence'):
                position = 'long'
                entry_price = close
                entry_time = i
            elif row.get('top_divergence'):
                position = 'short'
                entry_price = close
                entry_time = i

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

# def plot_all_strategies_signals_marked(btc_data, trades_rsi, trades_macd, trades_div, trades_combined=None,
#                                        show_labels=True, show_annotations=True, show_pnl_subplot=True,
#                                        show_entry_markers=True, show_exit_markers=True, show_combined_circles=True):
#     """
#     Enhanced plotting function with cumulative PnL tracking and configurable display options.
    
#     Parameters:
#     -----------
#     btc_data : DataFrame
#         Bitcoin price data with OHLC columns
#     trades_rsi, trades_macd, trades_div : DataFrame
#         Individual strategy trade results
#     trades_combined : DataFrame, optional
#         Combined strategy trade results
#     show_labels : bool, default True
#         Show legend labels
#     show_annotations : bool, default True
#         Show text annotations for combined signals
#     show_pnl_subplot : bool, default True
#         Show cumulative PnL in separate subplot
#     show_entry_markers : bool, default True
#         Show entry point markers
#     show_exit_markers : bool, default True
#         Show exit point markers (green stars for profits)
#     show_combined_circles : bool, default True
#         Show circles around combined signals
#     """
#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
#     import pandas as pd
#     import numpy as np

#     # Calculate cumulative PnL for each strategy
#     def calculate_cumulative_pnl(trades):
#         if trades.empty:
#             return pd.Series(dtype=float), pd.Series(dtype='datetime64[ns]')
        
#         trades_sorted = trades.sort_values('entry_time')
#         cumulative_pnl = (1 + trades_sorted['pnl']).cumprod() - 1
#         return cumulative_pnl, trades_sorted['entry_time']

#     # Setup figure with or without subplot
#     if show_pnl_subplot:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 16), height_ratios=[3, 1])
#         plt.subplots_adjust(hspace=0.3)
#     else:
#         fig, ax1 = plt.subplots(figsize=(30, 10))

#     # === Main price plot ===
#     ax1.plot(btc_data.index, btc_data['close'], color='black', linewidth=1.2, 
#              label='BTC Price' if show_labels else '')

#     def plot_trades(trades, label_prefix, color, ax):
#         if trades.empty:
#             return
            
#         long_entries = trades[trades['position'] == 'long']
#         short_entries = trades[trades['position'] == 'short']

#         # Entry points
#         if show_entry_markers:
#             if not long_entries.empty:
#                 ax.scatter(long_entries['entry_time'], long_entries['entry_price'],
#                            marker='^', color=color, s=100, 
#                            label=f'{label_prefix} Long Entry' if show_labels else '', zorder=5)
#             if not short_entries.empty:
#                 ax.scatter(short_entries['entry_time'], short_entries['entry_price'],
#                            marker='v', color=color, s=100, 
#                            label=f'{label_prefix} Short Entry' if show_labels else '', zorder=5)

#         # Exit points (only profitable trades)
#         if show_exit_markers:
#             profitable_trades = trades[trades['pnl'] > 0]
#             if not profitable_trades.empty:
#                 ax.scatter(profitable_trades['exit_time'], profitable_trades['exit_price'],
#                            marker='*', color='green', s=130, 
#                            label=f'{label_prefix} Profit Exit' if show_labels and len(profitable_trades) > 0 else '',
#                            zorder=6)

#     # === Plot individual strategies ===
#     strategy_colors = {'RSI': 'blue', 'MACD': 'purple', 'Div': 'orange'}
#     all_trades = {'RSI': trades_rsi, 'MACD': trades_macd, 'Div': trades_div}
    
#     for strategy_name, trades in all_trades.items():
#         plot_trades(trades, strategy_name, strategy_colors[strategy_name], ax1)

#     # === Combined strategy circles and annotations ===
#     if trades_combined is not None and not trades_combined.empty and show_combined_circles:
#         print(f"Marking {len(trades_combined)} combined signals...")
        
#         long_combined = trades_combined[trades_combined['position'] == 'long']
#         short_combined = trades_combined[trades_combined['position'] == 'short']
        
#         if not long_combined.empty:
#             ax1.scatter(long_combined['entry_time'], long_combined['entry_price'],
#                        s=500, facecolors='none', edgecolors='limegreen',
#                        linewidths=3, marker='o', 
#                        label='Combined Long Signal' if show_labels else '', zorder=10)
        
#         if not short_combined.empty:
#             ax1.scatter(short_combined['entry_time'], short_combined['entry_price'],
#                        s=500, facecolors='none', edgecolors='red',
#                        linewidths=3, marker='o', 
#                        label='Combined Short Signal' if show_labels else '', zorder=10)
        
#         # Add text annotations
#         if show_annotations:
#             for i, (_, row) in enumerate(trades_combined.iterrows()):
#                 ax1.annotate(f'C{i+1}', 
#                            xy=(row['entry_time'], row['entry_price']),
#                            xytext=(10, 10), textcoords='offset points',
#                            fontsize=12, fontweight='bold', 
#                            color='darkgreen' if row['position'] == 'long' else 'darkred',
#                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
#                            zorder=11)

#     # === Main plot formatting ===
#     ax1.set_title("Trading Strategy Comparison" + (" with Combined Signals" if trades_combined is not None else ""), 
#                   fontsize=16)
#     ax1.set_ylabel("Price (USD)", fontsize=12)
#     ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax1.grid(True, alpha=0.3)

#     if not show_pnl_subplot:
#         ax1.set_xlabel("Time", fontsize=12)
#         plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

#     # === Cumulative PnL subplot ===
#     if show_pnl_subplot:
#         ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
#         # Plot cumulative PnL for each strategy
#         for strategy_name, trades in all_trades.items():
#             if not trades.empty:
#                 cum_pnl, times = calculate_cumulative_pnl(trades)
#                 ax2.plot(times, cum_pnl * 100, color=strategy_colors[strategy_name], 
#                         linewidth=2, marker='o', markersize=4,
#                         label=f'{strategy_name} Cumulative PnL' if show_labels else '')
        
#         # Plot combined strategy PnL
#         if trades_combined is not None and not trades_combined.empty:
#             cum_pnl_combined, times_combined = calculate_cumulative_pnl(trades_combined)
#             ax2.plot(times_combined, cum_pnl_combined * 100, color='red', 
#                     linewidth=3, marker='s', markersize=6,
#                     label='Combined Cumulative PnL' if show_labels else '')
        
#         ax2.set_xlabel("Time", fontsize=12)
#         ax2.set_ylabel("Cumulative PnL (%)", fontsize=12)
#         ax2.set_title("Cumulative Performance Comparison", fontsize=14)
#         ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#         ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#         ax2.grid(True, alpha=0.3)
#         plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
#         if show_labels:
#             ax2.legend(loc='upper left')

#     # === Legend handling ===
#     if show_labels:
#         # Remove duplicate labels in main plot legend
#         handles1, labels1 = ax1.get_legend_handles_labels()
#         by_label = dict(zip(labels1, handles1))
#         ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

#     plt.tight_layout()
    
#     # Save with descriptive filename
#     filename = "strategy_signals"
#     if trades_combined is not None:
#         filename += "_with_combined"
#     if show_pnl_subplot:
#         filename += "_pnl"
#     filename += ".png"
    
#     plt.savefig(filename, dpi=600, bbox_inches='tight')
#     plt.show()
    
#     # Print summary statistics
#     print("\n" + "="*80)
#     print("PERFORMANCE SUMMARY")
#     print("="*80)
    
#     def print_strategy_stats(name, trades, color_code=""):
#         if trades.empty:
#             print(f"{color_code}{name:<12} | No trades")
#             return
        
#         total_trades = len(trades)
#         win_trades = len(trades[trades['pnl'] > 0])
#         win_rate = (win_trades / total_trades) * 100
#         total_return = ((1 + trades['pnl']).prod() - 1) * 100
#         avg_pnl = trades['pnl'].mean() * 100
#         max_pnl = trades['pnl'].max() * 100
#         min_pnl = trades['pnl'].min() * 100
        
#         print(f"{color_code}{name:<12} | Trades: {total_trades:3d} | Win Rate: {win_rate:5.1f}% | "
#               f"Total Return: {total_return:7.2f}% | Avg PnL: {avg_pnl:6.2f}%")
    
#     print_strategy_stats("RSI", trades_rsi)
#     print_strategy_stats("MACD", trades_macd)
#     print_strategy_stats("Divergence", trades_div)
#     if trades_combined is not None and not trades_combined.empty:
#         print_strategy_stats("COMBINED", trades_combined)
    
#     print("="*80)


# def combined_strategy_no_lookahead(years=0.5, plot_kwargs=None):
#     """
#     Combined strategy with strict no-lookahead bias and enhanced plotting options.
    
#     Parameters:
#     -----------
#     years : float
#         Number of years of data to analyze
#     plot_kwargs : dict, optional
#         Additional parameters for the plotting function
#     """
#     import pandas as pd

#     # Default plotting parameters
#     default_plot_kwargs = {
#         'show_labels': True,
#         'show_annotations': True,
#         'show_pnl_subplot': True,
#         'show_entry_markers': True,
#         'show_exit_markers': True,
#         'show_combined_circles': True
#     }
    
#     if plot_kwargs:
#         default_plot_kwargs.update(plot_kwargs)

#     # === Get data and calculate indicators ===
#     btc = get_btc_data(years=years)
#     btc = calculate_macd(btc)
#     btc = detect_divergence_no_lookahead(btc)
#     btc['rsi'] = ta.momentum.rsi(btc['close'], window=14)

#     # === Run individual strategies ===
#     trades_rsi = rsi_strategy(btc)
#     trades_macd = macd_backtest(btc)
#     trades_div = divergence_backtest(btc)

#     print("Searching for combined signals with NO LOOKAHEAD BIAS...")
    
#     # === Combined strategy logic (unchanged) ===
#     combined_signals = []
#     tolerance = pd.Timedelta(hours=40)
    
#     trades_rsi_sorted = trades_rsi.sort_values('entry_time')
#     trades_macd_sorted = trades_macd.sort_values('entry_time')
#     trades_div_sorted = trades_div.sort_values('entry_time')
    
#     print(f"RSI signals: {len(trades_rsi_sorted)}")
#     print(f"MACD signals: {len(trades_macd_sorted)}")
#     print(f"Divergence signals: {len(trades_div_sorted)}")
    
#     for _, rsi_trade in trades_rsi_sorted.iterrows():
#         rsi_time = rsi_trade['entry_time']
#         rsi_pos = rsi_trade['position']
        
#         macd_before_or_at = trades_macd_sorted[
#             (trades_macd_sorted['entry_time'] <= rsi_time) &
#             (trades_macd_sorted['entry_time'] >= rsi_time - tolerance) &
#             (trades_macd_sorted['position'] == rsi_pos)
#         ]
        
#         div_before_or_at = trades_div_sorted[
#             (trades_div_sorted['entry_time'] <= rsi_time) &
#             (trades_div_sorted['entry_time'] >= rsi_time - tolerance) &
#             (trades_div_sorted['position'] == rsi_pos)
#         ]
        
#         if not macd_before_or_at.empty and not div_before_or_at.empty:
#             latest_macd = macd_before_or_at.iloc[-1]
#             latest_div = div_before_or_at.iloc[-1]
            
#             entry_time = rsi_time
#             entry_price = rsi_trade['entry_price']
            
#             combined_signals.append({
#                 'entry_time': entry_time,
#                 'position': rsi_pos,
#                 'entry_price': entry_price,
#                 'stop_loss': entry_price * (1 - 0.015) if rsi_pos == 'long' else entry_price * (1 + 0.015),
#                 'take_profit': entry_price * (1 + 0.025) if rsi_pos == 'long' else entry_price * (1 - 0.025),
#                 'rsi_time': rsi_time,
#                 'macd_time': latest_macd['entry_time'],
#                 'div_time': latest_div['entry_time'],
#                 'time_gap_macd': (rsi_time - latest_macd['entry_time']).total_seconds() / 3600,
#                 'time_gap_div': (rsi_time - latest_div['entry_time']).total_seconds() / 3600
#             })

#     print(f"\n总共找到 {len(combined_signals)} 个无前视偏差的组合信号")

#     # === Simulate combined trades ===
#     combined_trades = []
#     for signal in combined_signals:
#         entry_time = pd.to_datetime(signal['entry_time'])
#         idx = btc.index.get_indexer([entry_time], method='nearest')[0]
#         entry_price = signal['entry_price']
#         stop = signal['stop_loss']
#         tp = signal['take_profit']
#         position = signal['position']

#         for i in range(idx + 1, len(btc)):
#             row = btc.iloc[i]
#             t = btc.index[i]

#             if position == 'long':
#                 if row['low'] <= stop:
#                     exit_price = stop
#                     pnl = (exit_price - entry_price) / entry_price
#                     exit_type = 'stop_loss'
#                     break
#                 elif row['high'] >= tp:
#                     exit_price = tp
#                     pnl = (exit_price - entry_price) / entry_price
#                     exit_type = 'take_profit'
#                     break
#             elif position == 'short':
#                 if row['high'] >= stop:
#                     exit_price = stop
#                     pnl = (entry_price - exit_price) / entry_price
#                     exit_type = 'stop_loss'
#                     break
#                 elif row['low'] <= tp:
#                     exit_price = tp
#                     pnl = (entry_price - exit_price) / entry_price
#                     exit_type = 'take_profit'
#                     break
#         else:
#             continue

#         combined_trades.append({
#             'entry_time': entry_time,
#             'exit_time': t,
#             'position': position,
#             'entry_price': entry_price,
#             'exit_price': exit_price,
#             'exit_type': exit_type,
#             'pnl': pnl
#         })

#     df_combined = pd.DataFrame(combined_trades)

#     # === Enhanced plotting ===
#     plot_all_strategies_signals_marked(btc, trades_rsi, trades_macd, trades_div, df_combined, **default_plot_kwargs)
    
#     return df_combined, combined_signals


# # Example usage with different plotting configurations:

# # Full display (default)
# # df_combined, signals = combined_strategy_no_lookahead(years=0.5)

# # Minimal display - only price and combined signals
# # df_combined, signals = combined_strategy_no_lookahead(
# #     years=0.5, 
# #     plot_kwargs={
# #         'show_labels': False,
# #         'show_annotations': False,
# #         'show_entry_markers': False,
# #         'show_exit_markers': False,
# #         'show_combined_circles': True,
# #         'show_pnl_subplot': True
# #     }
# # )

# # Only PnL comparison without main chart markers
# # df_combined, signals = combined_strategy_no_lookahead(
# #     years=0.5,
# #     plot_kwargs={
# #         'show_entry_markers': False,
# #         'show_exit_markers': False,
# #         'show_combined_circles': False,
# #         'show_pnl_subplot': True
# #     }
# # )

def divergence():
    btc = get_btc_data(years=5)
    btc = calculate_macd(btc)
    btc = detect_divergence_no_lookahead(btc)
    trades = divergence_backtest(btc)
    # analyze_results_divergence(trades)
    analyze_trades(trades)
    plot_divergence_signals(btc,trades=trades)

def get_btc_data_period(start_date, end_date, extra_days=26, interval='4h'):
    """
    获取Binance BTC/USDT数据（自动分页版）
    参数:
        start_date: 开始日期 (str or datetime)
        end_date: 结束日期 (str or datetime)
        extra_days: 额外天数
        interval: K线间隔（默认4小时）
    """
    import pandas as pd
    import requests
    import math
    from datetime import timedelta
    
    # 转换为datetime对象
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 添加额外天数
    start_dt = start_dt - timedelta(days=extra_days)
    
    # 转换时间戳（毫秒）
    start_time_ms = int(start_dt.timestamp() * 1000)
    end_time_ms = int(end_dt.timestamp() * 1000)
    
    print(f"获取时间范围: {start_dt} 到 {end_dt}")
    
    # 计算时间间隔（毫秒）
    interval_ms_map = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '6h': 6 * 60 * 60 * 1000,
        '8h': 8 * 60 * 60 * 1000,
        '12h': 12 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '3d': 3 * 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
    }
    
    interval_ms = interval_ms_map.get(interval, 4 * 60 * 60 * 1000)  # 默认4小时
    
    # 计算总需要的数据量
    total_time_ms = end_time_ms - start_time_ms
    total_candles = math.ceil(total_time_ms / interval_ms)
    print(f"需要获取的总K线数量: {total_candles}")
    
    # 计算需要分几页
    page_size = 1000  # Binance单次最大限制
    pages = math.ceil(total_candles / page_size)
    print(f"需要分 {pages} 页获取")
    
    all_data = []
    
    current_start_ms = start_time_ms
    
    for page in range(pages):
        print(f"\n正在获取第 {page+1}/{pages} 页...")
        
        # 计算当前页的结束时间（毫秒）
        current_end_ms = min(
            current_start_ms + (page_size * interval_ms),
            end_time_ms
        )
        
        # 转换为可读时间用于显示
        current_start_dt = pd.to_datetime(current_start_ms, unit='ms')
        current_end_dt = pd.to_datetime(current_end_ms, unit='ms')
        print(f"时间范围: {current_start_dt} 到 {current_end_dt}")
        
        # 构建请求参数
        params = {
            'symbol': 'BTCUSDT',
            'interval': interval,
            'startTime': current_start_ms,
            'endTime': current_end_ms,
            'limit': page_size
        }
        
        # 发送请求
        try:
            response = requests.get("https://api.binance.com/api/v3/klines", params=params)
            response.raise_for_status()  # 检查HTTP错误
            data = response.json()
            
            if not data:
                print(f"第 {page+1} 页没有数据")
                break
                
            print(f"获取到 {len(data)} 条数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # 转换数据类型
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # 只保留需要的列
            df_clean = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            all_data.append(df_clean)
            
            # 更新下一页的开始时间（使用最后一条数据的时间戳 + 间隔）
            if len(data) > 0:
                last_timestamp = data[-1][0]  # 最后一条数据的时间戳
                current_start_ms = last_timestamp + interval_ms
            else:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {str(e)}")
            continue
        except Exception as e:
            print(f"获取第 {page+1} 页失败: {str(e)}")
            continue
        
        # 如果当前开始时间已经超过结束时间，停止
        if current_start_ms >= end_time_ms:
            break
    
    # 合并所有数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 去重并排序
        final_df = final_df.drop_duplicates('timestamp').sort_values('timestamp')
        
        # 设置索引
        final_df = final_df.set_index('timestamp')
        
        # 过滤到指定时间范围内
        final_df = final_df[(final_df.index >= start_dt) & (final_df.index <= end_dt)]
        
        print(f"\n最终获取数据量: {len(final_df)} 条")
        if len(final_df) > 0:
            print(f"时间范围: {final_df.index[0]} 到 {final_df.index[-1]}")
        
        return final_df
    else:
        print("未能获取任何数据")
        return pd.DataFrame()

def calculate_cumulative_pnl(trades):
    """Calculate cumulative PnL for a series of trades."""
    if trades.empty:
        return pd.Series(dtype=float), pd.Series(dtype='datetime64[ns]')
    
    trades_sorted = trades.sort_values('entry_time')
    cumulative_pnl = (1 + trades_sorted['pnl']).cumprod() - 1
    return cumulative_pnl, trades_sorted['entry_time']


def plot_trades_on_chart(trades, label_prefix, color, ax, show_entry_markers=False, 
                        show_exit_markers=False, show_labels=False):
    """Plot trade signals on the chart."""
    if trades.empty:
        return
        
    long_entries = trades[trades['position'] == 'long']
    short_entries = trades[trades['position'] == 'short']

    # Entry points
    if show_entry_markers:
        if not long_entries.empty:
            ax.scatter(long_entries['entry_time'], long_entries['entry_price'],
                       marker='^', color=color, s=100, 
                       label=f'{label_prefix} Long Entry' if show_labels else '', zorder=5)
        if not short_entries.empty:
            ax.scatter(short_entries['entry_time'], short_entries['entry_price'],
                       marker='v', color=color, s=100, 
                       label=f'{label_prefix} Short Entry' if show_labels else '', zorder=5)

    # Exit points (only profitable trades)
    if show_exit_markers:
        profitable_trades = trades[trades['pnl'] > 0]
        if not profitable_trades.empty:
            ax.scatter(profitable_trades['exit_time'], profitable_trades['exit_price'],
                       marker='*', color='green', s=130, 
                       label=f'{label_prefix} Profit Exit' if show_labels and len(profitable_trades) > 0 else '',
                       zorder=6)


def print_strategy_stats(name, trades, color_code=""):
    """Print strategy performance statistics."""
    if trades.empty:
        print(f"{color_code}{name:<12} | No trades")
        return
    
    total_trades = len(trades)
    win_trades = len(trades[trades['pnl'] > 0])
    win_rate = (win_trades / total_trades) * 100
    total_return = ((1 + trades['pnl']).prod() - 1) * 100
    avg_pnl = trades['pnl'].mean() * 100
    max_pnl = trades['pnl'].max() * 100
    min_pnl = trades['pnl'].min() * 100
    
    print(f"{color_code}{name:<12} | Trades: {total_trades:3d} | Win Rate: {win_rate:5.1f}% | "
          f"Total Return: {total_return:7.2f}% | Avg PnL: {avg_pnl:6.2f}%")


def plot_all_strategies_signals_marked(btc_data, trades_rsi, trades_macd, trades_div, trades_combined=None,
                                       show_labels=False, show_annotations=False, show_pnl_subplot=False,
                                       show_entry_markers=False, show_exit_markers=False, show_combined_circles=False):
    """
    Enhanced plotting function with cumulative PnL tracking and configurable display options.
    
    Parameters:
    -----------
    btc_data : DataFrame
        Bitcoin price data with OHLC columns
    trades_rsi, trades_macd, trades_div : DataFrame
        Individual strategy trade results
    trades_combined : DataFrame, optional
        Combined strategy trade results
    show_labels : bool, default False
        Show legend labels
    show_annotations : bool, default False
        Show text annotations for combined signals
    show_pnl_subplot : bool, default False
        Show cumulative PnL in separate subplot
    show_entry_markers : bool, default False
        Show entry point markers
    show_exit_markers : bool, default False
        Show exit point markers (green stars for profits)
    show_combined_circles : bool, default False
        Show circles around combined signals
    """
    # Setup figure with or without subplot
    if show_pnl_subplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 16), height_ratios=[3, 1])
        plt.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(figsize=(30, 10))

    # === Main price plot ===
    ax1.plot(btc_data.index, btc_data['close'], color='black', linewidth=1.2, 
             label='BTC Price' if show_labels else '')

    # === Plot individual strategies ===
    strategy_colors = {'RSI': 'blue', 'MACD': 'purple', 'Div': 'orange'}
    all_trades = {'RSI': trades_rsi, 'MACD': trades_macd, 'Div': trades_div}
    
    for strategy_name, trades in all_trades.items():
        plot_trades_on_chart(trades, strategy_name, strategy_colors[strategy_name], ax1,
                           show_entry_markers, show_exit_markers, show_labels)

    # === Combined strategy circles and annotations ===
    if trades_combined is not None and not trades_combined.empty and show_combined_circles:
        print(f"Marking {len(trades_combined)} combined signals...")
        
        long_combined = trades_combined[trades_combined['position'] == 'long']
        short_combined = trades_combined[trades_combined['position'] == 'short']
        
        if not long_combined.empty:
            ax1.scatter(long_combined['entry_time'], long_combined['entry_price'],
                       s=500, facecolors='none', edgecolors='limegreen',
                       linewidths=3, marker='o', 
                       label='Combined Long Signal' if show_labels else '', zorder=10)
        
        if not short_combined.empty:
            ax1.scatter(short_combined['entry_time'], short_combined['entry_price'],
                       s=500, facecolors='none', edgecolors='red',
                       linewidths=3, marker='o', 
                       label='Combined Short Signal' if show_labels else '', zorder=10)
        
        # Add text annotations
        if show_annotations:
            for i, (_, row) in enumerate(trades_combined.iterrows()):
                ax1.annotate(f'C{i+1}', 
                           xy=(row['entry_time'], row['entry_price']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold', 
                           color='darkgreen' if row['position'] == 'long' else 'darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           zorder=11)

    # === Main plot formatting ===
    ax1.set_title("Trading Strategy Comparison" + (" with Combined Signals" if trades_combined is not None else ""), 
                  fontsize=16)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    if not show_pnl_subplot:
        ax1.set_xlabel("Time", fontsize=12)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # === Cumulative PnL subplot ===
    if show_pnl_subplot:
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot cumulative PnL for each strategy
        for strategy_name, trades in all_trades.items():
            if not trades.empty:
                cum_pnl, times = calculate_cumulative_pnl(trades)
                ax2.plot(times, cum_pnl * 100, color=strategy_colors[strategy_name], 
                        linewidth=2, marker='o', markersize=4,
                        label=f'{strategy_name} Cumulative PnL' if show_labels else '')
        
        # Plot combined strategy PnL
        if trades_combined is not None and not trades_combined.empty:
            cum_pnl_combined, times_combined = calculate_cumulative_pnl(trades_combined)
            ax2.plot(times_combined, cum_pnl_combined * 100, color='red', 
                    linewidth=3, marker='s', markersize=6,
                    label='Combined Cumulative PnL' if show_labels else '')
        
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Cumulative PnL (%)", fontsize=12)
        ax2.set_title("Cumulative Performance Comparison", fontsize=14)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        if show_labels:
            ax2.legend(loc='upper left')

    # === Legend handling ===
    if show_labels:
        # Remove duplicate labels in main plot legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels1, handles1))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()
    
    # Save with descriptive filename
    filename = "strategy_signals"
    if trades_combined is not None:
        filename += "_with_combined"
    if show_pnl_subplot:
        filename += "_pnl"
    filename += ".png"
    
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print_strategy_stats("RSI", trades_rsi)
    print_strategy_stats("MACD", trades_macd)
    print_strategy_stats("Divergence", trades_div)
    if trades_combined is not None and not trades_combined.empty:
        print_strategy_stats("COMBINED", trades_combined)
    
    print("="*80)


def find_combined_signals(trades_rsi_sorted, trades_macd_sorted, trades_div_sorted):
    """Find combined signals with no lookahead bias."""
    combined_signals = []
    tolerance = pd.Timedelta(hours=40)
    
    # Traverse each RSI signal as "trigger point"
    for _, rsi_trade in trades_rsi_sorted.iterrows():
        rsi_time = rsi_trade['entry_time']
        rsi_pos = rsi_trade['position']
        
        # Find MACD signals before or at RSI time
        macd_before_or_at = trades_macd_sorted[
            (trades_macd_sorted['entry_time'] <= rsi_time) &
            (trades_macd_sorted['entry_time'] >= rsi_time - tolerance) &
            (trades_macd_sorted['position'] == rsi_pos)
        ]
        
        # Find divergence signals before or at RSI time
        div_before_or_at = trades_div_sorted[
            (trades_div_sorted['entry_time'] <= rsi_time) &
            (trades_div_sorted['entry_time'] >= rsi_time - tolerance) &
            (trades_div_sorted['position'] == rsi_pos)
        ]
        
        if not macd_before_or_at.empty and not div_before_or_at.empty:
            latest_macd = macd_before_or_at.iloc[-1]
            latest_div = div_before_or_at.iloc[-1]
            
            entry_time = rsi_time
            entry_price = rsi_trade['entry_price']
            
            combined_signals.append({
                'entry_time': entry_time,
                'position': rsi_pos,
                'entry_price': entry_price,
                'stop_loss': entry_price * (1 - 0.015) if rsi_pos == 'long' else entry_price * (1 + 0.015),
                'take_profit': entry_price * (1 + 0.025) if rsi_pos == 'long' else entry_price * (1 - 0.025),
                'rsi_time': rsi_time,
                'macd_time': latest_macd['entry_time'],
                'div_time': latest_div['entry_time'],
                'time_gap_macd': (rsi_time - latest_macd['entry_time']).total_seconds() / 3600,
                'time_gap_div': (rsi_time - latest_div['entry_time']).total_seconds() / 3600
            })
    
    return combined_signals


def simulate_combined_trades(combined_signals, btc_data):
    """Simulate trades based on combined signals."""
    combined_trades = []
    
    for signal in combined_signals:
        entry_time = pd.to_datetime(signal['entry_time'])
        idx = btc_data.index.get_indexer([entry_time], method='nearest')[0]
        entry_price = signal['entry_price']
        stop = signal['stop_loss']
        tp = signal['take_profit']
        position = signal['position']

        for i in range(idx + 1, len(btc_data)):
            row = btc_data.iloc[i]
            t = btc_data.index[i]

            if position == 'long':
                if row['low'] <= stop:
                    exit_price = stop
                    pnl = (exit_price - entry_price) / entry_price
                    exit_type = 'stop_loss'
                    break
                elif row['high'] >= tp:
                    exit_price = tp
                    pnl = (exit_price - entry_price) / entry_price
                    exit_type = 'take_profit'
                    break
            elif position == 'short':
                if row['high'] >= stop:
                    exit_price = stop
                    pnl = (entry_price - exit_price) / entry_price
                    exit_type = 'stop_loss'
                    break
                elif row['low'] <= tp:
                    exit_price = tp
                    pnl = (entry_price - exit_price) / entry_price
                    exit_type = 'take_profit'
                    break
        else:
            continue

        combined_trades.append({
            'entry_time': entry_time,
            'exit_time': t,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl': pnl
        })
    
    return combined_trades


def combined_strategy_no_lookahead(years=0.5, plot_kwargs=None):
    """
    Combined strategy with strict no-lookahead bias and enhanced plotting options.
    
    Parameters:
    -----------
    years : float
        Number of years of data to analyze
    plot_kwargs : dict, optional
        Additional parameters for the plotting function
    """
    # Default plotting parameters - all False by default
    default_plot_kwargs = {
        'show_labels': False,
        'show_annotations': False,
        'show_pnl_subplot': False,
        'show_entry_markers': False,
        'show_exit_markers': False,
        'show_combined_circles': False
    }
    
    if plot_kwargs:
        default_plot_kwargs.update(plot_kwargs)

    # === Get data and calculate indicators ===
    btc = get_btc_data(years=years)
    btc = calculate_macd(btc)
    btc = detect_divergence_no_lookahead(btc)
    btc['rsi'] = ta.momentum.rsi(btc['close'], window=14)

    # === Run individual strategies ===
    trades_rsi = rsi_strategy(btc)
    trades_macd = macd_backtest(btc)
    trades_div = divergence_backtest(btc)

    print("Searching for combined signals with NO LOOKAHEAD BIAS...")
    
    # Sort trades by time
    trades_rsi_sorted = trades_rsi.sort_values('entry_time')
    trades_macd_sorted = trades_macd.sort_values('entry_time')
    trades_div_sorted = trades_div.sort_values('entry_time')
    
    print(f"RSI signals: {len(trades_rsi_sorted)}")
    print(f"MACD signals: {len(trades_macd_sorted)}")
    print(f"Divergence signals: {len(trades_div_sorted)}")
    
    # Find combined signals
    combined_signals = find_combined_signals(trades_rsi_sorted, trades_macd_sorted, trades_div_sorted)
    
    print(f"\n总共找到 {len(combined_signals)} 个无前视偏差的组合信号")

    # Simulate combined trades
    combined_trades = simulate_combined_trades(combined_signals, btc)
    df_combined = pd.DataFrame(combined_trades)

    # === Enhanced plotting ===
    plot_all_strategies_signals_marked(btc, trades_rsi, trades_macd, trades_div, df_combined, **default_plot_kwargs)
    
    return df_combined, combined_signals


# Example usage with different plotting configurations:

# Minimal display (default - all False)
# df_combined, signals = combined_strategy_no_lookahead(years=0.5)

# Show only cumulative PnL
# df_combined, signals = combined_strategy_no_lookahead(
#     years=0.5,
#     plot_kwargs={'show_pnl_subplot': True, 'show_labels': True}
# )

# Show everything
# df_combined, signals = combined_strategy_no_lookahead(
#     years=0.5, 
#     plot_kwargs={
#         'show_labels': True,
#         'show_annotations': True,
#         'show_pnl_subplot': True,
#         'show_entry_markers': True,
#         'show_exit_markers': True,
#         'show_combined_circles': True
#     }
# )
if __name__ == "__main__":
    # macd()
    # rsi(0.5)
    # divergence()
    df_combined, signals = combined_strategy_no_lookahead(
        years=5,
        plot_kwargs={'show_pnl_subplot': True, 'show_labels': True}
    )
        # combined_strategy_period(start_date="2024-1-11 00:00", end_date="2024-05-09 00:00")


    