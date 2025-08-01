from calendar import monthrange
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
import math
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# 定义成本变量（后续可调整）
TRANSACTION_FEE = 0.001 # 单边手续费0.1%
FINANCING_RATE = 0  # 资金成本（每日）
HOLDING_DAYS = 30        # 预估持有天数
MIN_PROFIT_PCT = 0

class BinanceArbitrageBacktester:
    def __init__(self, symbol='BTCUSDT', interval='4h'):
        self.symbol = symbol
        self.interval = interval
        self.spot_data = None
        self.futures_data = None
        self.merged_data = None
        self.trades = []
        self.portfolio_value = []

        # Trading parameters
        self.initial_capital = 10000  # USDT
        self.transaction_fee = 0.001  # 0.1% per trade
        self.funding_rate = 0.0001   # Approximate funding rate per 8h
        self.min_profit_threshold = 0.002  # Minimum 0.2% profit to execute
        self.contract_dates = {
            "BTCUSD_220325": "2022-03-25",
            "BTCUSD_220624": "2022-06-24",
            "BTCUSD_220930": "2022-09-30",
            "BTCUSD_221230": "2022-12-30",
            "BTCUSD_230331": "2023-03-31",
            "BTCUSD_230630": "2023-06-30",
            "BTCUSD_230929": "2023-09-29",
            "BTCUSD_231229": "2023-12-29",
            "BTCUSD_240329": "2024-03-29",
            "BTCUSD_240628": "2024-06-28",
            "BTCUSD_240927": "2024-09-27",
            "BTCUSD_241227": "2024-12-27",
            "BTCUSD_250328": "2025-03-28",
            "BTCUSD_250627": "2025-06-27",
            # "BTCUSD_250926": "2025-09-26",
            # "BTCUSD_251226": "2025-12-26"
        }
        
    def get_spot_data(self, years=3, extra_days=26):
        """Enhanced version of your spot data fetching method"""
        print("Fetching spot market data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365 * years + extra_days)
        
        total_hours = (end_time - start_time).total_seconds() / 3600
        interval_hours = {'1h':1, '4h':4, '1d':24}[self.interval]
        total_candles = math.ceil(total_hours / interval_hours)
        
        print(f"Total candles needed: {total_candles}")
        
        page_size = 1000
        pages = math.ceil(total_candles / page_size)
        print(f"Pages needed: {pages}")
        
        all_data = []
        
        for page in range(pages):
            print(f"Fetching page {page+1}/{pages}...")
        
            page_start = start_time + timedelta(hours=page * page_size * interval_hours)
            page_end = min(
                page_start + timedelta(hours=(page_size-1) * interval_hours),
                end_time
            )
            
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'startTime': int(page_start.timestamp() * 1000),
                'endTime': int(page_end.timestamp() * 1000),
                'limit': page_size
            }
            
            try:
                response = requests.get("https://api.binance.com/api/v3/klines", params=params)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    
                    all_data.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
                
            except Exception as e:
                print(f"Error fetching page {page+1}: {str(e)}")
                continue
            
            # Rate limiting to avoid API limits
            time.sleep(0.1)
        
        if all_data:
            final_df = pd.concat(all_data).drop_duplicates('timestamp')
            final_df = final_df.set_index('timestamp').sort_index()
            final_df.columns = [f'spot_{col}' for col in final_df.columns]
            print(f"Spot data fetched: {len(final_df)} records")
            print(f"Date range: {final_df.index[0]} to {final_df.index[-1]}")
            self.spot_data = final_df
            return final_df
        else:
            print("Failed to fetch spot data")
            return pd.DataFrame()
    
    def get_historical_futures_data(self, contract_symbol):
        """
        获取指定合约符号的历史期货数据
        :param contract_symbol: 合约符号，如BTCUSD_220325
        :return: DataFrame格式的K线数据
        """
        print(f"📦 正在获取交割合约 {contract_symbol} 数据...")

        # 计算合约上线时间（大约在交割日前90天）
        delivery_date = datetime.strptime(self.contract_dates[contract_symbol], "%Y-%m-%d")
        start_time = delivery_date - timedelta(days=90)
        end_time = delivery_date  # 只获取到交割日的数据

        interval_hours = {'1h': 1, '4h': 4, '1d': 24}[self.interval]
        total_hours = (end_time - start_time).total_seconds() / 3600
        total_candles = math.ceil(total_hours / interval_hours)

        page_size = 1000
        pages = math.ceil(total_candles / page_size)

        all_data = []

        for page in range(pages):
            print(f"⏳ 正在获取第 {page + 1}/{pages} 页数据...")

            page_start = start_time + timedelta(hours=page * page_size * interval_hours)
            page_end = min(
                page_start + timedelta(hours=(page_size - 1) * interval_hours),
                end_time
            )

            params = {
                'symbol': contract_symbol,
                'interval': self.interval,
                'startTime': int(page_start.timestamp() * 1000),
                'endTime': int(page_end.timestamp() * 1000),
                'limit': page_size
            }

            try:
                response = requests.get("https://dapi.binance.com/dapi/v1/klines", params=params)
                response.raise_for_status()
                data = response.json()

                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                    df['contract'] = contract_symbol  # 添加合约标识列

                    all_data.append(df[['timestamp', 'contract', 'open', 'high', 'low', 'close', 'volume']])

            except Exception as e:
                print(f"❌ 获取第 {page + 1} 页数据时出错: {str(e)}")
                continue

            time.sleep(0.1)  # 避免请求过于频繁

        # === 合并结果 ===
        if all_data:
            final_df = pd.concat(all_data).drop_duplicates('timestamp')
            final_df = final_df.set_index('timestamp').sort_index()
            final_df.columns = [f'futures_{col}' for col in final_df.columns]
            print(f"✅ Delivery futures data fetched: {len(final_df)} rows")
            self.futures_data = final_df
            return final_df
        else:
            print("❌ No data fetched.")
            return pd.DataFrame()

    def get_all_futures_data(self, start_date='2022-01-01'):
        """
        获取所有季度合约的历史数据
        :param start_date: 开始日期
        :return: a list of contract dates
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        all_data = []
        contracts = []

        for contract, delivery_date in self.contract_dates.items():
            delivery_dt = datetime.strptime(delivery_date, "%Y-%m-%d")
            if delivery_dt >= start_dt:  # 只获取开始日期之后的合约
                contract_data = self.get_historical_futures_data(contract)
                if not contract_data.empty:
                    contract_data['contract'] = contract
                    all_data.append(contract_data)
                    contracts.append(contract)

        if all_data:
            self.futures_data = pd.concat(all_data).sort_index()
            
            print(f"✅ 成功获取所有期货数据，总计 {len(self.futures_data)} 行")
        else:
            print("❌ 未获取到任何期货数据")
            self.futures_data = pd.DataFrame()
        
        return contracts
    
    def get_current_delivery_futures_data(self,contract_symbol):
        """
        Fetch current-quarter delivery futures market data from Binance (COIN-M).
        自动获取当前季度BTCUSD合约，并抓取它全部的K线数据。
        """

        print("📦 Fetching delivery futures (current quarter) data...")

        # === 设置请求时间范围（只获取当前季度合约上线以来的数据） ===
        end_time = datetime.utcnow()
        # 交割合约一般上线于交割日前约90天；此处 conservatively 估算起始为90天前
        approx_launch = end_time - timedelta(days=90)

        interval_hours = {'1h': 1, '4h': 4, '1d': 24}[self.interval]
        total_hours = (end_time - approx_launch).total_seconds() / 3600
        total_candles = math.ceil(total_hours / interval_hours)

        page_size = 1000
        pages = math.ceil(total_candles / page_size)

        all_data = []

        for page in range(pages):
            print(f"⏳ Fetching page {page + 1}/{pages}...")

            page_start = approx_launch + timedelta(hours=page * page_size * interval_hours)
            page_end = min(
                page_start + timedelta(hours=(page_size - 1) * interval_hours),
                end_time
            )

            params = {
                # 'symbol': contract_symbol,
                'symbol': contract_symbol,
                'interval': self.interval,
                'startTime': int(page_start.timestamp() * 1000),
                'endTime': int(page_end.timestamp() * 1000),
                'limit': page_size
            }

            try:
                response = requests.get("https://dapi.binance.com/dapi/v1/klines", params=params)
                response.raise_for_status()
                data = response.json()

                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                    all_data.append(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])

            except Exception as e:
                print(f"❌ Error fetching page {page + 1}: {str(e)}")
                continue

            time.sleep(0.1)

        # === 合并结果 ===
        if all_data:
            final_df = pd.concat(all_data).drop_duplicates('timestamp')
            final_df = final_df.set_index('timestamp').sort_index()
            final_df.columns = [f'futures_{col}' for col in final_df.columns]
            print(f"✅ Delivery futures data fetched: {len(final_df)} rows")
            self.futures_data = final_df
            return final_df
        else:
            print("❌ No data fetched.")
            return pd.DataFrame()


    
    def merge_data(self):
        """Merge spot and futures data on timestamp"""
        if self.spot_data is None or self.futures_data is None:
            raise ValueError("Both spot and futures data must be fetched first")
        
        # Inner join to get only matching timestamps
        merged = pd.merge(self.spot_data, self.futures_data, 
                         left_index=True, right_index=True, how='inner')
        
        # Calculate spread metrics
        merged['spread'] = merged['futures_close'] - merged['spot_close']
        merged['spread_pct'] = (merged['spread'] / merged['spot_close']) * 100
        merged['basis'] = merged['spread']  # Basis is the same as spread
        
        # Calculate rolling statistics for dynamic thresholds
        merged['spread_mean'] = merged['spread_pct'].rolling(window=24).mean()
        merged['spread_std'] = merged['spread_pct'].rolling(window=24).std()
        merged['upper_threshold'] = merged['spread_mean'] + 2 * merged['spread_std']
        merged['lower_threshold'] = merged['spread_mean'] - 2 * merged['spread_std']
        
        print(f"Merged data: {len(merged)} records")
        self.merged_data = merged
        return merged
    

# Example usage
def main():
    """Example usage of the arbitrage backtester"""
    # Initialize backtester
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # Fetch data (reduced timeframe for testing)
    backtester.get_spot_data(years=1, extra_days=0)
    backtester.get_futures_data(years=1, extra_days=0)
    
    # Merge and analyze
    backtester.merge_data()
    backtester.identify_arbitrage_opportunities()
    
    # Run backtest
    backtester.simulate_trades(use_dynamic_thresholds=True)
    
    # Analyze results
    stats, trades_df = backtester.analyze_results()
    
    # Generate plots
    backtester.plot_results()
    
    return backtester

def stage_test():
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # Fetch data (reduced timeframe for testing)
    btc_spot = backtester.get_spot_data(years=1, extra_days=0)
    btc_future = backtester.get_futures_data(years=1, extra_days=0)

    # Merge and analyze
    btc_merged = backtester.merge_data()

    print(btc_spot.head(10))
    print(btc_future.tail(10))
    print(btc_merged.tail(10))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def plot_stage_data(backtester, stage='all', figsize=(15, 10), save_path=None):
    """
    Comprehensive plotting function for stage-by-stage data verification
    
    Parameters:
    - backtester: BinanceArbitrageBacktester instance
    - stage: 'spot', 'futures', 'merged', or 'all'
    - figsize: Figure size tuple
    - save_path: Optional path to save the plot
    """
    
    if stage == 'spot' or stage == 'all':
        if backtester.spot_data is not None:
            plot_spot_data(backtester.spot_data, figsize)
    
    if stage == 'futures' or stage == 'all':
        if backtester.futures_data is not None:
            plot_futures_data(backtester.futures_data, figsize)
    
    if stage == 'merged' or stage == 'all':
        if backtester.merged_data is not None:
            plot_merged_data(backtester.merged_data, figsize)
    
    if stage == 'all' and backtester.spot_data is not None and backtester.futures_data is not None:
        plot_comparison(backtester.spot_data, backtester.futures_data, figsize)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_spot_data(spot_data, figsize=(15, 8)):
    """Plot spot market data with OHLC and volume"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Price plot
    ax1.plot(spot_data.index, spot_data['spot_close'], label='Spot Close', color='blue', linewidth=1.5)
    ax1.plot(spot_data.index, spot_data['spot_high'], label='Spot High', color='green', alpha=0.5, linewidth=0.8)
    ax1.plot(spot_data.index, spot_data['spot_low'], label='Spot Low', color='red', alpha=0.5, linewidth=0.8)
    ax1.fill_between(spot_data.index, spot_data['spot_low'], spot_data['spot_high'], alpha=0.1, color='gray')
    
    ax1.set_title(f'Spot Market Data - BTCUSDT\nData Points: {len(spot_data)} | Period: {spot_data.index[0].strftime("%Y-%m-%d")} to {spot_data.index[-1].strftime("%Y-%m-%d")}', fontsize=12)
    ax1.set_ylabel('Price (USDT)', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add price statistics text
    stats_text = f"""Stats:
Min: ${spot_data['spot_low'].min():.2f}
Max: ${spot_data['spot_high'].max():.2f}
Mean: ${spot_data['spot_close'].mean():.2f}
Std: ${spot_data['spot_close'].std():.2f}"""
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    # Volume plot
    ax2.bar(spot_data.index, spot_data['spot_volume'], alpha=0.6, color='purple', width=0.001)
    ax2.set_title('Spot Volume', fontsize=10)
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Data quality checks
    print("=== SPOT DATA QUALITY CHECKS ===")
    print(f"Total records: {len(spot_data)}")
    print(f"Date range: {spot_data.index[0]} to {spot_data.index[-1]}")
    print(f"Missing values: {spot_data.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {spot_data.index.duplicated().sum()}")
    print(f"Zero volume records: {(spot_data['spot_volume'] == 0).sum()}")
    print(f"Price anomalies (Close > High or Close < Low): {((spot_data['spot_close'] > spot_data['spot_high']) | (spot_data['spot_close'] < spot_data['spot_low'])).sum()}")

def plot_futures_data(futures_data, figsize=(15, 8)):
    """Plot futures market data with OHLC and volume"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Price plot
    ax1.plot(futures_data.index, futures_data['futures_close'], label='Futures Close', color='orange', linewidth=1.5)
    ax1.plot(futures_data.index, futures_data['futures_high'], label='Futures High', color='green', alpha=0.5, linewidth=0.8)
    ax1.plot(futures_data.index, futures_data['futures_low'], label='Futures Low', color='red', alpha=0.5, linewidth=0.8)
    ax1.fill_between(futures_data.index, futures_data['futures_low'], futures_data['futures_high'], alpha=0.1, color='gray')
    
    ax1.set_title(f'Futures Market Data - BTCUSDT\nData Points: {len(futures_data)} | Period: {futures_data.index[0].strftime("%Y-%m-%d")} to {futures_data.index[-1].strftime("%Y-%m-%d")}', fontsize=12)
    ax1.set_ylabel('Price (USDT)', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add price statistics text
    stats_text = f"""Stats:
Min: ${futures_data['futures_low'].min():.2f}
Max: ${futures_data['futures_high'].max():.2f}
Mean: ${futures_data['futures_close'].mean():.2f}
Std: ${futures_data['futures_close'].std():.2f}"""
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    # Volume plot
    ax2.bar(futures_data.index, futures_data['futures_volume'], alpha=0.6, color='darkorange', width=0.001)
    ax2.set_title('Futures Volume', fontsize=10)
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('future.png')
    plt.show()
    
    # Data quality checks
    print("=== FUTURES DATA QUALITY CHECKS ===")
    print(f"Total records: {len(futures_data)}")
    print(f"Date range: {futures_data.index[0]} to {futures_data.index[-1]}")
    print(f"Missing values: {futures_data.isnull().sum().sum()}")
    print(f"Duplicate timestamps: {futures_data.index.duplicated().sum()}")
    print(f"Zero volume records: {(futures_data['futures_volume'] == 0).sum()}")
    print(f"Price anomalies (Close > High or Close < Low): {((futures_data['futures_close'] > futures_data['futures_high']) | (futures_data['futures_close'] < futures_data['futures_low'])).sum()}")

def plot_comparison(spot_data, futures_data, figsize=(15, 10)):
    """Plot spot vs futures comparison"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Spot vs Futures Data Comparison', fontsize=14)
    
    # Price comparison
    ax1 = axes[0, 0]
    ax1.plot(spot_data.index, spot_data['spot_close'], label='Spot', color='blue', alpha=0.8)
    ax1.plot(futures_data.index, futures_data['futures_close'], label='Futures', color='orange', alpha=0.8)
    ax1.set_title('Price Comparison')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume comparison
    ax2 = axes[0, 1]
    ax2.plot(spot_data.index, spot_data['spot_volume'], label='Spot Volume', color='purple', alpha=0.7)
    ax2.plot(futures_data.index, futures_data['futures_volume'], label='Futures Volume', color='darkorange', alpha=0.7)
    ax2.set_title('Volume Comparison')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Data availability comparison
    ax3 = axes[1, 0]
    spot_availability = spot_data.notna().sum(axis=1)
    futures_availability = futures_data.notna().sum(axis=1)
    ax3.plot(spot_data.index, spot_availability, label='Spot Data Points', color='blue')
    ax3.plot(futures_data.index, futures_availability, label='Futures Data Points', color='orange')
    ax3.set_title('Data Availability Over Time')
    ax3.set_ylabel('Available Data Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: Spot vs Futures prices
    ax4 = axes[1, 1]
    # Find common timestamps for fair comparison
    common_times = spot_data.index.intersection(futures_data.index)
    if len(common_times) > 0:
        spot_common = spot_data.loc[common_times, 'spot_close']
        futures_common = futures_data.loc[common_times, 'futures_close']
        ax4.scatter(spot_common, futures_common, alpha=0.5, s=1)
        
        # Add perfect correlation line
        min_price = min(spot_common.min(), futures_common.min())
        max_price = max(spot_common.max(), futures_common.max())
        ax4.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8, label='Perfect Correlation')
        
        # Calculate correlation
        corr = np.corrcoef(spot_common, futures_common)[0, 1]
        ax4.set_title(f'Spot vs Futures Correlation\nCorr: {corr:.4f}')
        ax4.set_xlabel('Spot Price')
        ax4.set_ylabel('Futures Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Common Timestamps\nFound for Comparison', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Spot vs Futures Correlation')
    
    plt.tight_layout()
    plt.show()
    
    # Comparison statistics
    print("=== SPOT VS FUTURES COMPARISON ===")
    print(f"Spot records: {len(spot_data)}")
    print(f"Futures records: {len(futures_data)}")
    print(f"Common timestamps: {len(common_times) if len(common_times) > 0 else 0}")
    if len(common_times) > 0:
        print(f"Price correlation: {corr:.4f}")
        avg_spread = (futures_common - spot_common).mean()
        print(f"Average spread (Futures - Spot): ${avg_spread:.2f}")
        print(f"Average spread %: {(avg_spread / spot_common.mean() * 100):.3f}%")

def plot_merged_data(merged_data, figsize=(15, 12)):
    """Plot merged data with spread analysis"""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Merged Data Analysis - Arbitrage Opportunities', fontsize=14)
    
    # Price comparison
    ax1 = axes[0, 0]
    ax1.plot(merged_data.index, merged_data['spot_close'], label='Spot', color='blue', alpha=0.8)
    ax1.plot(merged_data.index, merged_data['futures_close'], label='Futures', color='orange', alpha=0.8)
    ax1.set_title('Merged Price Data')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spread analysis
    ax2 = axes[0, 1]
    ax2.plot(merged_data.index, merged_data['spread'], label='Absolute Spread', color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Price Spread (Futures - Spot)')
    ax2.set_ylabel('Spread ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Spread percentage
    ax3 = axes[1, 0]
    ax3.plot(merged_data.index, merged_data['spread_pct'], label='Spread %', color='purple')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Spread Percentage')
    ax3.set_ylabel('Spread (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Dynamic thresholds (if available)
    ax4 = axes[1, 1]
    if 'upper_threshold' in merged_data.columns:
        ax4.plot(merged_data.index, merged_data['spread_pct'], label='Spread %', color='purple', alpha=0.7)
        ax4.plot(merged_data.index, merged_data['upper_threshold'], label='Upper Threshold', color='red', linestyle='--')
        ax4.plot(merged_data.index, merged_data['lower_threshold'], label='Lower Threshold', color='green', linestyle='--')
        ax4.fill_between(merged_data.index, merged_data['upper_threshold'], merged_data['lower_threshold'], 
                        alpha=0.2, color='gray', label='Neutral Zone')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Dynamic Arbitrage Thresholds')
        ax4.set_ylabel('Spread (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Dynamic Thresholds\nNot Calculated Yet', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Dynamic Thresholds')
    
    # Spread distribution
    ax5 = axes[2, 0]
    ax5.hist(merged_data['spread_pct'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_title('Spread Distribution')
    ax5.set_xlabel('Spread (%)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # Volume comparison
    ax6 = axes[2, 1]
    ax6.plot(merged_data.index, merged_data['spot_volume'], label='Spot Volume', alpha=0.7, color='blue')
    ax6.plot(merged_data.index, merged_data['futures_volume'], label='Futures Volume', alpha=0.7, color='orange')
    ax6.set_title('Volume Comparison')
    ax6.set_ylabel('Volume')
    ax6.set_xlabel('Date')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Merged data statistics
    print("=== MERGED DATA ANALYSIS ===")
    print(f"Total merged records: {len(merged_data)}")
    print(f"Date range: {merged_data.index[0]} to {merged_data.index[-1]}")
    print(f"Average spread: ${merged_data['spread'].mean():.2f}")
    print(f"Average spread %: {merged_data['spread_pct'].mean():.3f}%")
    print(f"Spread std dev: {merged_data['spread_pct'].std():.3f}%")
    print(f"Max positive spread: {merged_data['spread_pct'].max():.3f}% (Futures premium)")
    print(f"Max negative spread: {merged_data['spread_pct'].min():.3f}% (Spot premium)")
    
    # Arbitrage opportunity preview (if columns exist)
    if 'cash_carry_opportunity' in merged_data.columns:
        cash_carry_ops = merged_data['cash_carry_opportunity'].sum()
        reverse_carry_ops = merged_data['reverse_carry_opportunity'].sum()
        print(f"Cash-and-carry opportunities: {cash_carry_ops}")
        print(f"Reverse cash-and-carry opportunities: {reverse_carry_ops}")

def quick_data_summary(backtester):
    """Quick summary of all available data"""
    print("=== QUICK DATA SUMMARY ===")
    
    if backtester.spot_data is not None:
        print(f"✓ Spot data: {len(backtester.spot_data)} records")
        print(f"  Range: {backtester.spot_data.index[0]} to {backtester.spot_data.index[-1]}")
    else:
        print("✗ Spot data: Not loaded")
    
    if backtester.futures_data is not None:
        print(f"✓ Futures data: {len(backtester.futures_data)} records")
        print(f"  Range: {backtester.futures_data.index[0]} to {backtester.futures_data.index[-1]}")
    else:
        print("✗ Futures data: Not loaded")
    
    if backtester.merged_data is not None:
        print(f"✓ Merged data: {len(backtester.merged_data)} records")
        print(f"  Range: {backtester.merged_data.index[0]} to {backtester.merged_data.index[-1]}")
    else:
        print("✗ Merged data: Not created")
    
    print(f"Trades: {len(backtester.trades)} executed")
    print("-" * 40)

# Simple replacement for your current stage_test function
def stage_test_with_plots():
    """Your original stage_test but with plotting capabilities"""
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # Fetch data (reduced timeframe for testing)
    btc_spot = backtester.get_spot_data(years=0.5, extra_days=0)
    btc_future = backtester.get_current_delivery_futures_data()
    # print(btc_future['futures_high'].max())      152152.5

    # Merge and analyze
    btc_merged = backtester.merge_data()

    # print("=== DATA SAMPLES ===")
    # print("Spot data head:")
    # print(btc_spot.head(5))
    # print("\nFutures data tail:")
    # print(btc_future.tail(5))
    # print("\nMerged data tail:")
    # print(btc_merged.tail(5))
    
    # quick_data_summary(backtester)
    
    # print("\nGenerating verification plots...")
    # plot_stage_data(backtester, stage='all',save_path='./')
    backtester.identify_arbitrage_opportunities()
    
    return backtester

def check_dates():
        # 初始化
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. 获取数据并打印结构
    print("正在获取现货数据...")
    spot_data = backtester.get_spot_data(years=2, extra_days=0)
    # print("现货数据列名:", spot_data.columns.tolist())
    # print("现货数据示例:\n", spot_data.head(2))

    # 现货数据列名: ['spot_open', 'spot_high', 'spot_low', 'spot_close', 'spot_volume']
    # 现货数据示例:
    #                     spot_open  spot_high  spot_low  spot_close  spot_volume
    # timestamp                                                                   
    # 2023-08-02 00:00:00   29705.99   30047.50   29622.5    29632.96  12892.26158
    # 2023-08-02 04:00:00   29632.96   29719.77   29564.0    29577.26   5196.11558
    
    print("\n正在获取期货数据...")
    contracts = backtester.get_all_futures_data(start_date='2024-01-01')
    futures_data = backtester.futures_data
    # print("期货数据列名:", futures_data.columns.tolist())
    
    # 期货数据列名: ['futures_contract', 'futures_open', 'futures_high', 'futures_low', 'futures_close', 'futures_volume', 'contract']
    # 期货数据示例:
    #                     futures_contract  futures_open  futures_high  futures_low  futures_close  futures_volume       contract
    # timestamp                                                                                                                  
    # 2023-12-29 16:00:00    BTCUSD_240329       44031.4       44098.0      43422.6        43920.3        183725.0  BTCUSD_240329
    # 2023-12-29 20:00:00    BTCUSD_240329       43916.3       44033.3      43015.0        43882.5        175769.0  BTCUSD_240329
    print("期货数据示例:\n", futures_data.head(2))
    
    # 2. 提取合约周期
    print("\n合约周期信息:")
    contract_periods = []
    for contract in contracts:
        contract_df = futures_data[futures_data['contract'] == contract]
        if not contract_df.empty:
            start = contract_df.index.min()
            end = contract_df.index.max()
            contract_periods.append((start, end, contract))
            print(f"{contract}: {start} 至 {end} (共 {len(contract_df)} 条数据)")
        else:
            print(f"⚠️ {contract} 数据为空")

    # 可视化（更新版）
    plt.figure(figsize=(16, 8))

    # 1. 绘制价格曲线
    spot_col = 'spot_close'
    plt.plot(spot_data.index, spot_data[spot_col], label='BTC spot price', color='blue', alpha=0.7)

    # 为每个期货合约绘制价格曲线
    for contract in contracts:
        contract_df = futures_data[futures_data['contract'] == contract]
        plt.plot(contract_df.index, contract_df['futures_close'], 
                linestyle='--', alpha=0.5, label=f'{contract}')

    # 2. 标记合约周期
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
    for idx, (start, end, contract) in enumerate(contract_periods):
        color = colors[idx % len(colors)]
        plt.axvspan(start, end, alpha=0.05, color=color)
        plt.axvline(start, linestyle=':', color=color, alpha=0.5)
        plt.axvline(end, linestyle='-', color=color, alpha=0.8)
        plt.text(end, spot_data[spot_col].min()*0.98, 
                contract[-4:], rotation=90, color=color, va='top')

    # 3. 图表装饰
    plt.title("btc spot(usdt) vs btc delivery futures (for debug purpose)", pad=20)
    plt.xlabel("date", fontsize=12)
    plt.ylabel("price", fontsize=12)
    plt.legend(loc='upper left', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spot_vs_future.png')
    plt.show()

def main_history():
        # 初始化
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. 获取数据
    print("获取现货数据...")
    spot_data = backtester.get_spot_data(years=2, extra_days=0)
    print("获取期货数据...")
    contracts = backtester.get_all_futures_data(start_date='2023-12-01')  # 包含当前季度合约
    futures_data = backtester.futures_data

    # 2. 生成交易信号（季度展期逻辑）
    signals = pd.DataFrame(index=spot_data.index)
    signals['position'] = 0  # 0: 空仓, 1: 持有多头现货+空头期货
    
    # 找出每个季度的换仓时间点（前一个季度到期日前4小时）
    roll_dates = []
    for i in range(len(contracts)-1):
        current_contract = contracts[i]      # 当前季度合约（如240329）
        next_contract = contracts[i+1]       # 下一个季度合约（如240628）
        
        # 当前合约到期日
        current_end = futures_data[futures_data['contract'] == current_contract].index.max()
        # 换仓时间（提前4小时）
        roll_time = current_end - pd.Timedelta(hours=4)
        roll_dates.append((roll_time, current_contract, next_contract))
        print(f"换仓点 {i+1}: {roll_time} (平仓 {current_contract}, 开仓 {next_contract})")

    # 3. 执行回测
    results = []
    for i, (roll_time, current_contract, next_contract) in enumerate(roll_dates):
        if roll_time not in spot_data.index:
            print(f"⚠️ 换仓时间 {roll_time} 不在现货数据中")
            continue
            
        # 获取价格
        spot_price = spot_data.loc[roll_time, 'spot_close']
        print(f"\nDebug: 正在处理 {roll_time} 的合约 {next_contract}")
        print("可用时间范围:", futures_data[futures_data['contract'] == next_contract].index[[0, -1]])
        print("精确匹配结果:", futures_data[(futures_data.index == roll_time) & (futures_data['contract'] == next_contract)])
        next_future_price = futures_data.loc[
            (futures_data.index == roll_time) & 
            (futures_data['contract'] == next_contract),
            'futures_close'
        ].values[0]
        
        # 计算持有期收益（当前季度剩余时间 + 下个季度全周期）
        current_end = futures_data[futures_data['contract'] == current_contract].index.max()
        next_end = futures_data[futures_data['contract'] == next_contract].index.max()
        
        # 现货收益（roll_time买入，next_end卖出）
        spot_exit_price = spot_data.loc[next_end, 'spot_close']
        spot_return = (spot_exit_price - spot_price) / spot_price
        
        # 期货收益（roll_time做空，next_end平仓）
        future_exit_price = futures_data.loc[
            (futures_data.index == next_end) & 
            (futures_data['contract'] == next_contract),
            'futures_close'
        ].values[0]
        future_return = (next_future_price - future_exit_price) / next_future_price
        
        # 净收益（扣除手续费0.1% x 4次交易）
        net_return = spot_return + future_return - 0.004
        annualized_return = (1 + net_return)**(365/(next_end - roll_time).days) - 1
        
        results.append({
            'roll_date': roll_time,
            'current_contract': current_contract,
            'next_contract': next_contract,
            'spot_entry': spot_price,
            'future_entry': next_future_price,
            'spot_exit': spot_exit_price,
            'future_exit': future_exit_price,
            'holding_days': (next_end - roll_time).days,
            'spot_return': spot_return,
            'future_return': future_return,
            'net_return': net_return,
            'annualized_return': annualized_return
        })
    
    # 4. 结果分析
    results_df = pd.DataFrame(results)
    print("\n策略表现汇总:")
    print(results_df[['roll_date', 'current_contract', 'next_contract', 
                     'net_return', 'annualized_return']])
    
    # 计算累计收益
    results_df['cumulative_return'] = (1 + results_df['net_return']).cumprod() - 1
    total_return = results_df['cumulative_return'].iloc[-1]
    print(f"\n总收益率: {total_return*100:.2f}%")
    
    # 5. 可视化
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['roll_date'], results_df['net_return']*100, 
            width=20, label='单期收益(%)')
    plt.plot(results_df['roll_date'], results_df['cumulative_return']*100,
            'r-', marker='o', label='累计收益(%)')
    plt.title('BTC季度展期套利策略表现')
    plt.xlabel('换仓日期')
    plt.ylabel('收益率(%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Replace the problematic section in your main() function with this improved version:
# 在回测前添加验证
def validate_contract_dates(futures_data, contracts):
    for i, contract in enumerate(contracts):
        df = futures_data[futures_data['contract'] == contract]
        print(f"{contract}: {len(df)}条数据 | 开始:{df.index.min()} 到期:{df.index.max()}")
        
        # 检查合约间连续性
        if i > 0:
            prev_end = futures_data[
                futures_data['contract'] == contracts[i-1]
            ].index.max()
            gap = (df.index.min() - prev_end).total_seconds()/3600
            print(f"  与前一合约间隔: {gap:.1f}小时")

def main_fixed():
    # Initialize
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. Get data
    print("获取现货数据...")
    spot_data = backtester.get_spot_data(years=2, extra_days=0)
    print("获取期货数据...")
    contracts = backtester.get_all_futures_data(start_date='2023-12-01')
    futures_data = backtester.futures_data
    validate_contract_dates(futures_data,contracts)

    # 2. Generate trading signals with improved error handling
    signals = pd.DataFrame(index=spot_data.index)
    signals['position'] = 0
    
    # 修改roll_dates生成方式（匹配实际数据时间点）
    roll_dates = []
    for i in range(len(contracts)-1):
        current_contract = contracts[i]
        next_contract = contracts[i+1]
        
        # 使用期货数据的实际开始时间作为换仓点
        next_contract_start = futures_data[
            futures_data['contract'] == next_contract
        ].index.min()
        
        roll_dates.append((next_contract_start, current_contract, next_contract))
        print(f"修正后换仓点 {i+1}: {next_contract_start} (开仓 {next_contract})")

    # 3. Execute backtest with better error handling
    results = []
    for i, (roll_time, current_contract, next_contract) in enumerate(roll_dates):
        print(f"\n=== Processing rollover {i+1}: {roll_time} ===")
        
        # Check if roll_time exists in spot data
        if roll_time not in spot_data.index:
            print(f"⚠️ Roll time {roll_time} not in spot data, finding nearest...")
            # Find nearest available timestamp
            nearest_spot_time = spot_data.index[spot_data.index <= roll_time].max()
            if pd.isna(nearest_spot_time):
                print(f"❌ No spot data available before {roll_time}")
                continue
            roll_time = nearest_spot_time
            print(f"✓ Using nearest spot time: {roll_time}")
        
        # Get spot price
        spot_price = spot_data.loc[roll_time, 'spot_close']
        print(f"Spot price at {roll_time}: ${spot_price:.2f}")
        
        # Check futures data availability for next contract
        next_contract_data = futures_data[futures_data['contract'] == next_contract]
        if next_contract_data.empty:
            print(f"❌ No data for contract {next_contract}")
            continue
            
        print(f"Next contract {next_contract} data range: {next_contract_data.index.min()} to {next_contract_data.index.max()}")
        
        # Find futures price at roll time (or nearest available)
        available_times = next_contract_data.index
        matching_times = next_contract_data[next_contract_data.index == roll_time]
        
        if matching_times.empty:
            print(f"⚠️ Exact time {roll_time} not found for {next_contract}")
            # Find nearest available time
            nearest_future_time = available_times[available_times >= roll_time].min()
            if pd.isna(nearest_future_time):
                # If no future time, try past time
                nearest_future_time = available_times[available_times <= roll_time].max()
                
            if pd.isna(nearest_future_time):
                print(f"❌ No suitable time found for {next_contract}")
                continue
                
            print(f"✓ Using nearest future time: {nearest_future_time}")
            next_future_price = next_contract_data.loc[nearest_future_time, 'futures_close']
        else:
            next_future_price = matching_times['futures_close'].iloc[0]
            
        print(f"Future price for {next_contract}: ${next_future_price:.2f}")
        
        # Calculate holding period returns
        current_end = futures_data[futures_data['contract'] == current_contract].index.max()
        next_end = futures_data[futures_data['contract'] == next_contract].index.max()
        
        # Check if exit dates have data
        if next_end not in spot_data.index:
            print(f"⚠️ Exit date {next_end} not in spot data")
            nearest_exit_spot = spot_data.index[spot_data.index <= next_end].max()
            if pd.isna(nearest_exit_spot):
                print(f"❌ No spot exit data available")
                continue
            next_end = nearest_exit_spot
            
        # Get exit prices
        spot_exit_price = spot_data.loc[next_end, 'spot_close']
        
        # Future exit price
        future_exit_data = futures_data[
            (futures_data.index == next_end) & 
            (futures_data['contract'] == next_contract)
        ]
        
        if future_exit_data.empty:
            print(f"⚠️ No future exit data at {next_end} for {next_contract}")
            # Find nearest exit time
            exit_contract_data = futures_data[futures_data['contract'] == next_contract]
            nearest_exit_future = exit_contract_data.index[exit_contract_data.index <= next_end].max()
            if pd.isna(nearest_exit_future):
                print(f"❌ No suitable future exit time")
                continue
            future_exit_price = exit_contract_data.loc[nearest_exit_future, 'futures_close']
        else:
            future_exit_price = future_exit_data['futures_close'].iloc[0]
            
        # Calculate returns
        spot_return = (spot_exit_price - spot_price) / spot_price
        future_return = (next_future_price - future_exit_price) / next_future_price
        
        # Net return (minus 0.1% fee x 4 trades)
        net_return = spot_return + future_return - 4 * TRANSACTION_FEE
        holding_days = (next_end - roll_time).days
        annualized_return = (1 + net_return)**(365/holding_days) - 1 if holding_days > 0 else 0
        
        print(f"Spot return: {spot_return*100:.2f}%")
        print(f"Future return: {future_return*100:.2f}%")
        print(f"Net return: {net_return*100:.2f}%")
        
        results.append({
            'roll_date': roll_time,
            'current_contract': current_contract,
            'next_contract': next_contract,
            'spot_entry': spot_price,
            'future_entry': next_future_price,
            'spot_exit': spot_exit_price,
            'future_exit': future_exit_price,
            'holding_days': holding_days,
            'spot_return': spot_return,
            'future_return': future_return,
            'net_return': net_return,
            'annualized_return': annualized_return
        })
    
    # 4. Results analysis
    if not results:
        print("❌ No successful trades executed")
        return
        
    results_df = pd.DataFrame(results)
    print("\n策略表现汇总:")
    print(results_df[['roll_date', 'current_contract', 'next_contract', 
                     'net_return', 'annualized_return']])
    
    # Calculate cumulative returns
    results_df['cumulative_return'] = (1 + results_df['net_return']).cumprod() - 1
    total_return = results_df['cumulative_return'].iloc[-1]
    print(f"\n总收益率: {total_return*100:.2f}%")
    
    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['roll_date'], results_df['net_return']*100, 
            width=20, label='single season profit(%)')
    plt.plot(results_df['roll_date'], results_df['cumulative_return']*100,
            'r-', marker='o', label='acc profit(%)')
    plt.title('seasonal performance')
    plt.xlabel('exchange dates')
    plt.ylabel('profit ratio(%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results_df

# Additional helper function to debug data availability
def debug_data_availability(backtester, roll_time, contract):
    """Debug function to check data availability around a specific time"""
    futures_data = backtester.futures_data
    spot_data = backtester.spot_data
    
    print(f"\n=== Debugging data for {contract} around {roll_time} ===")
    
    # Check contract data
    contract_data = futures_data[futures_data['contract'] == contract]
    if contract_data.empty:
        print(f"❌ No data found for contract {contract}")
        return
        
    print(f"Contract {contract} data range: {contract_data.index.min()} to {contract_data.index.max()}")
    print(f"Total records: {len(contract_data)}")
    
    # Check around roll_time
    time_window = pd.Timedelta(hours=12)  # Check ±12 hours
    start_check = roll_time - time_window
    end_check = roll_time + time_window
    
    nearby_data = contract_data[
        (contract_data.index >= start_check) & 
        (contract_data.index <= end_check)
    ]
    
    print(f"\nData within ±12 hours of {roll_time}:")
    if nearby_data.empty:
        print("❌ No data found in time window")
    else:
        print(f"✓ Found {len(nearby_data)} records")
        print(nearby_data[['futures_close']].head())
    
    # Check spot data availability
    if roll_time in spot_data.index:
        print(f"✓ Spot data available at {roll_time}: ${spot_data.loc[roll_time, 'spot_close']:.2f}")
    else:
        print(f"⚠️ No spot data at exact time {roll_time}")

def generate_roll_dates(futures_data, contracts):
    """生成与数据严格对齐的换仓时间点"""
    roll_dates = []
    for i in range(len(contracts)-1):
        # 当前合约的到期日16:00（数据中最后一条）
        expiry_time = futures_data[futures_data['contract'] == contracts[i]].index.max()
        
        # 新合约的开始时间16:00（数据中第一条）
        new_contract_start = futures_data[futures_data['contract'] == contracts[i+1]].index.min()
        
        # 换仓时间 = 新合约开始时间（保持与数据严格一致）
        roll_dates.append((new_contract_start, contracts[i], contracts[i+1]))
        
        print(f"换仓点 {i+1}: {new_contract_start} | "
              f"平仓 {contracts[i]} (到期 {expiry_time}) | "
              f"开仓 {contracts[i+1]}")
    return roll_dates

def plot_results(results_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                         gridspec_kw={'height_ratios': [3, 1]},
                         facecolor='white')
    
    # Cumulative returns line plot
    results_df['cumulative'] = (1 + results_df['net_return']).cumprod()
    ax1.plot(results_df['expiry_date'], results_df['cumulative'], 
            'o-', color='#2c7bb6', linewidth=2, markersize=8,
            label='Strategy NAV')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(loc='upper left', framealpha=0.8)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Single period returns bar plot
    colors = ['#4daf4a' if x >=0 else '#e41a1c' for x in results_df['net_return']]
    ax2.bar(results_df['expiry_date'], results_df['net_return']*100, 
           width=15, color=colors, edgecolor='grey', alpha=0.8)
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Formatting
    fig.autofmt_xdate()
    plt.suptitle('BTC Quarterly Roll Arbitrage Strategy', y=0.98, fontsize=14)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('btc_roll_arbitrage.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_returns(spot_data, futures_data, roll_dates):
    """
    Calculate strategy returns with full trade records
    
    Returns:
    - DataFrame with complete trade details
    - Dictionary of cash flows for each trade
    """
    results = []
    cash_flows = {}
    spot_transaction_rate = 0.001
    future_transaction_rate = 0.0005
    
    for i, (roll_time, current_contract, next_contract) in enumerate(roll_dates):
        # Get entry prices
        spot_entry = spot_data.loc[roll_time, 'spot_close']
        future_entry = futures_data.loc[
            (futures_data.index == roll_time) & 
            (futures_data['contract'] == next_contract),
            'futures_close'
        ].iloc[0]
        
        # Get exit prices
        expiry_time = futures_data[futures_data['contract'] == current_contract].index.max()
        spot_exit = spot_data.loc[expiry_time, 'spot_close']
        future_exit = futures_data.loc[
            (futures_data.index == expiry_time) & 
            (futures_data['contract'] == current_contract),
            'futures_close'
        ].iloc[0]
        
        # Calculate basis
        entry_basis = future_entry - spot_entry
        exit_basis = future_exit - spot_exit
        basis_change = entry_basis - exit_basis

        spot_fee = spot_transaction_rate * (spot_entry + spot_exit)
        future_fee = future_transaction_rate * (future_entry + future_exit)
        total_fee = spot_fee + future_fee
        
        # Transaction details
        trade_details = {
            # Trade metadata
            'trade_id': i+1,
            'roll_date': roll_time,
            'expiry_date': expiry_time,
            'contract_in': current_contract,
            'contract_out': next_contract,
            
            # Entry positions
            'spot_entry_price': spot_entry,
            'future_entry_price': future_entry,
            'entry_basis': entry_basis,
            'entry_basis_pct': entry_basis/spot_entry,
            
            # Exit positions
            'spot_exit_price': spot_exit,
            'future_exit_price': future_exit,
            'exit_basis': exit_basis,
            'exit_basis_pct': exit_basis/spot_exit,
            
            # Performance metrics
            'basis_change': basis_change,
            'basis_change_pct': basis_change/spot_entry,
            'holding_days': (expiry_time - roll_time).days,
            
            # Returns calculation
            'spot_return': (spot_exit - spot_entry)/spot_entry,
            'future_return': (future_entry - future_exit)/future_entry,
            'gross_return': (basis_change)/spot_entry,
            'net_return': (basis_change-total_fee)/spot_entry,  
            'annualized_return': (1 + (basis_change-total_fee)/spot_entry)**(365/(expiry_time - roll_time).days) - 1,
            'total_fee':total_fee,
            'fee_breakdown': {  # 新增：手续费明细
                'spot_buy_fee': spot_transaction_rate * spot_entry,
                'spot_sell_fee': spot_transaction_rate * spot_exit,
                'future_open_fee': future_transaction_rate * future_entry,
                'future_close_fee': future_transaction_rate * future_exit
            }
        }
        
        # Cash flow records
        cash_flows[f"Trade_{i+1}"] = {
            # Timeline
            'date': {
                'open': roll_time.strftime('%Y-%m-%d'), 
                'close': expiry_time.strftime('%Y-%m-%d')
            },
            
            # Positions - record exact prices when actions happened
            'positions': {
                'open': {
                    'spot_price': spot_entry,
                    'future_price': future_entry,
                    'future_contract': next_contract
                },
                'close': {
                    'spot_price': spot_exit,
                    'future_price': future_exit,
                    'future_contract': current_contract
                }
            },
            
            # Money movement - simple cash accounting
            'cash': {
                'outflow': round(-spot_entry - future_entry*future_transaction_rate, 2),  # Money spent to open
                'inflow': round(spot_exit + (future_entry - future_exit) - future_exit*future_transaction_rate, 2),  # Money received when closing
                'net': round((spot_exit - spot_entry) + (future_entry - future_exit) 
                    - (future_entry + future_exit)*future_transaction_rate, 2)  # Total profit/loss
            },
            
            # Plain English summary
            'summary': [
                f"Opened on {roll_time.date()}: Bought 1 BTC at ${spot_entry} | Sold {next_contract} at ${future_entry}",
                f"Closed on {expiry_time.date()}: Sold BTC at ${spot_exit} | Bought back contract at ${future_exit}",
                f"Net result: ${round((spot_exit-spot_entry)+(future_entry-future_exit), 2)} before fees"
            ]
        }
        
        results.append(trade_details)
    
    return pd.DataFrame(results), cash_flows

def print_trade_summary(cash_flows):
    """Prints just the human-readable summary from cash_flows"""
    print("\n" + "="*80)
    print("TRADE SUMMARY".center(80))
    print("="*80)
    
    for trade_id, trade_data in cash_flows.items():
        print(f"\n{trade_id.replace('_', ' ').upper()}:")
        print("-"*80)
        for line in trade_data['summary']:
            print(f"• {line}")
        print(f"\n{'Final Net:':<15} ${trade_data['cash']['net']:,.2f}")
        print("-"*80)

def basic_strategy():
    # 1.在这个季度开始前（上一个季度末/上一个比特币季度合约交割日结束之前）：买比特币现货，做空这个季度的比特币季度合约。
    # 2.在这个季度末（这个比特币季度合约交割日结束之前）：卖出比特币现货，平仓这个季度的比特币合约（完成交割？）｜买比特币，做空下一个季度的比特币季度合约...

    # 参数， 交易手续费 基础用户： 现货 0.1% 合约  0.05%
    # 资金费率，暂不考虑
    # Initialize
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. Get data
    print("获取现货数据...")
    spot_data = backtester.get_spot_data(years=2, extra_days=0)
    print("获取期货数据...")
    contracts = backtester.get_all_futures_data(start_date='2023-12-01')
    futures_data = backtester.futures_data
    # validate_contract_dates(futures_data,contracts)

    # 2. Generate trading signals with improved error handling
    signals = pd.DataFrame(index=spot_data.index)
    signals['position'] = 0

    roll_dates = generate_roll_dates(futures_data,contracts)
    print("交易日：")
    print(roll_dates)
    return_data, cash_flows = calculate_returns(spot_data,futures_data,roll_dates)
    print("交易记录")
    print_trade_summary(cash_flows)
    print(return_data[['trade_id', 'roll_date', 'contract_out', 
                 'spot_entry_price', 'future_entry_price',
                 'net_return', 'annualized_return']])
    plot_results(return_data)
    #手续费
    # 季度中交易，第一阶段收益率最大值/平均值做参考，吃了跑
    #3 吃了还要吃，建仓条件？1.新价差=初始价差
if __name__ == "__main__":
    basic_strategy()