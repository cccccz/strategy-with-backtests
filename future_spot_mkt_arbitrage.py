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
TRANSACTION_FEE = 0.001  # 单边交易手续费0.1%
FINANCING_RATE = 0.0002  # 资金成本（每日）
HOLDING_DAYS = 30        # 预估持有天数
MIN_PROFIT_PCT = 0.003

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

if __name__ == "__main__":
    # Run the backtest
    # backtester = main()
    # stage_test_with_plots()
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    spot_data = backtester.get_spot_data(years=2, extra_days=0)
    contracts = backtester.get_all_futures_data(start_date='2024-01-01')
    btc_merged = backtester.merge_data()
    futures_data = backtester.futures_data
    #plot_stage_data(backtester, stage='all')
    # 从期货数据中提取合约到期日（Binance季度合约代码格式：BTCUSD_YYMMDD）
    print(contracts)
    # 为每个合约生成时间范围
    # 2. 生成合约时间范围（解决你的具体问题）
   # 2. 提取季度时间范围
    contract_periods = []
    for contract in contracts:
        contract_df = futures_data[futures_data['contract'] == contract]
        if not contract_df.empty:
            start = contract_df.index.min()  # 合约开始日
            end = contract_df.index.max()   # 合约到期日
            contract_periods.append((start, end, contract))
            print(f"合约 {contract} 周期: {start} 至 {end}")

    # 3. 计算每个季度的收益
    results = []
    for start, end, contract in contract_periods:
        # 获取现货价格（季度初买入，季度末卖出）
        spot_entry = spot_data.loc[start, 'close']  # 现货买入价
        spot_exit = spot_data.loc[end, 'close']     # 现货卖出价
        spot_return = (spot_exit - spot_entry) / spot_entry

        # 获取期货价格（季度初做空，季度末平仓）
        future_entry = futures_data.loc[(futures_data.index == start) & 
                                      (futures_data['contract'] == contract), 'close'].values[0]
        future_exit = futures_data.loc[(futures_data.index == end) & 
                                     (futures_data['contract'] == contract), 'close'].values[0]
        future_return = (future_entry - future_exit) / future_entry  # 做空收益

        # 净收益（现货+期货，扣除手续费）
        net_return = spot_return + future_return - 0.002  # 假设手续费0.2%
        results.append({
            'contract': contract,
            'start_date': start,
            'end_date': end,
            'spot_return': spot_return,
            'future_return': future_return,
            'net_return': net_return
        })

    # 4. 汇总结果
    results_df = pd.DataFrame(results)
    results_df['cumulative_return'] = (1 + results_df['net_return']).cumprod()
    print("\n季度收益明细:")
    print(results_df)

    # 5. 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['end_date'], results_df['cumulative_return'], marker='o')
    plt.title("BTC季度展期套利累计收益")
    plt.grid(True)
    plt.show()
