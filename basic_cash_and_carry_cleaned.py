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
import os
import json
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')


# 定义成本变量（后续可调整）
SPOT_TRANSACTION_RATE = 0.001
FUTURE_TRANSACTION_RATE = 0.0005
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
        if symbol == 'BTCUSDT':
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
        elif symbol == 'ETHUSDT':
            self.contract_dates = {
                "ETHUSD_200925": "2020-09-25",
                "ETHUSD_201225": "2020-12-25",
                "ETHUSD_210326": "2021-03-26",
                "ETHUSD_210625": "2021-06-25",
                "ETHUSD_210924": "2021-09-24",
                "ETHUSD_211231": "2021-12-31",
                "ETHUSD_220325": "2022-03-25",
                "ETHUSD_220624": "2022-06-24",
                "ETHUSD_220930": "2022-09-30",
                "ETHUSD_221230": "2022-12-30",
                "ETHUSD_230331": "2023-03-31",
                "ETHUSD_230630": "2023-06-30",
                "ETHUSD_230929": "2023-09-29",
                "ETHUSD_231229": "2023-12-29",
                "ETHUSD_240329": "2024-03-29",
                "ETHUSD_240628": "2024-06-28",
                "ETHUSD_240927": "2024-09-27",
                "ETHUSD_241227": "2024-12-27",
                "ETHUSD_250328": "2025-03-28",
                "ETHUSD_250627": "2025-06-27",
                # "ETHUSD_250926": "2025-09-26",
                # "ETHUSD_251226": "2025-12-26"
            }
            
        
    def save_data(self, df, filename, data_type="data"):
        """Save DataFrame to pickle file with metadata"""
        try:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            if df is None or df.empty:
                print(f"⚠️ Warning: Saving empty {data_type}")
            
            save_obj = {
                'data': df,
                'saved_at': datetime.now(),
                'data_type': data_type,
                'symbol': self.symbol,
                'interval': self.interval
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(save_obj, f)
            
            print(f"✅ {data_type.title()} saved: {df.shape[0]} rows")
            
        except Exception as e:
            print(f"❌ Save failed: {str(e)}")
            raise

    def load_data(self, filename, check_freshness=False, max_age_hours=24):
        """Load DataFrame from pickle file"""
        try:
            if not os.path.exists(filename):
                print(f"⚠️ File not found: {filename}")
                return None
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both new format (dict) and old format (direct DataFrame)
            if isinstance(data, dict) and 'data' in data:
                df = data['data']
                if check_freshness:
                    age_hours = (datetime.now() - data['saved_at']).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        print(f"⚠️ Data too old: {age_hours:.1f}h")
                        return None
            else:
                df = data  # Old format
            
            if df is None or df.empty:
                print("⚠️ Loaded data is empty")
                return None
            
            print(f"✅ Loaded: {df.shape[0]} rows, {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Load failed: {str(e)}")
            return None

    def save_spot_data(self, filename='spot_data.pkl'):
        """Save spot data"""
        if self.spot_data is not None:
            self.save_data(self.spot_data, filename, "spot")

    def save_futures_data(self, filename='futures_data.pkl'):
        """Save futures data"""
        if self.futures_data is not None:
            self.save_data(self.futures_data, filename, "futures")

    def load_spot_data(self, filename='spot_data.pkl'):
        """Load spot data with freshness check"""
        df = self.load_data(filename, check_freshness=True)
        if df is not None:
            self.spot_data = df
            return True
        return False

    def load_futures_data(self, filename='futures_data.pkl'):
        """Load futures data and return contract list"""
        df = self.load_data(filename)
        if df is not None:
            self.futures_data = df
            # Extract contracts
            contract_cols = [col for col in df.columns if 'contract' in col.lower()]
            if contract_cols:
                contracts = df[contract_cols[0]].dropna().unique().tolist()
                print(f"Found {len(contracts)} contracts")
                return contracts
        return []

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
    
def generate_roll_dates(futures_data, contracts):
    """
    Generates the roll schedule for futures contracts in a cash-and-carry arbitrage strategy.
    
    For each contract, determines when to open and close positions by finding:
    - The earliest available data point (open time)
    - The last available data point before expiration (close time)
    
    Args:
        futures_data: DataFrame containing futures prices with:
            - Index: DatetimeIndex
            - Columns: Must include 'contract' (str) and 'futures_close' (price)
        contracts: List of contract names (e.g. ['BTCUSD_2403', 'BTCUSD_2406'])
    
    Returns:
        List of tuples in format (open_time, close_time, contract_name):
        - open_time: pd.Timestamp - When to enter the trade
        - close_time: pd.Timestamp - When to exit the trade
        - contract_name: str - The futures contract being traded
    
    Example Output:
        [
            (pd.Timestamp('2023-12-01'), pd.Timestamp('2024-03-28'), 'BTCUSD_2403'),
            (pd.Timestamp('2024-03-29'), pd.Timestamp('2024-06-27'), 'BTCUSD_2406')
        ]
    """
    roll_schedule = []

    for i in range(len(contracts)):
        contract = contracts[i]

        # 开仓时间 = 当前合约的最早时间
        open_time = futures_data[futures_data['contract'] == contract].index.min()

        # 平仓时间 = 当前合约的最后时间（即交割前）
        close_time = futures_data[futures_data['contract'] == contract].index.max()

        roll_schedule.append((open_time, close_time, contract))

        print(f"Trade {i+1}: 开仓 {open_time}，平仓 {close_time}，合约 {contract}")

    return roll_schedule


def plot_results(results_df):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                         gridspec_kw={'height_ratios': [3, 1]},
                         facecolor='white')
    
    # 累计收益曲线
    results_df['cumulative'] = (1 + results_df['net_return']).cumprod()
    ax1.plot(results_df['close_time'], results_df['cumulative'], 
             'o-', color='#2c7bb6', linewidth=2, markersize=8,
             label='Strategy NAV')
    
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend(loc='upper left', framealpha=0.8)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # 单期收益柱状图
    colors = ['#4daf4a' if x >= 0 else '#e41a1c' for x in results_df['net_return']]
    ax2.bar(results_df['close_time'], results_df['net_return'] * 100, 
            width=15, color=colors, edgecolor='grey', alpha=0.8)
    
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # 格式化时间轴
    fig.autofmt_xdate()
    plt.suptitle('BTC Quarterly Roll Arbitrage Strategy', y=0.98, fontsize=14)
    plt.tight_layout()
    
    # 保存 + 展示图像
    plt.savefig('btc_roll_arbitrage.png', dpi=300, bbox_inches='tight')
    plt.show()


def calculate_returns(spot_data, futures_data, roll_schedule):
    """
    Calculates strategy returns by executing trades according to the roll schedule.
    
    For each trade window in roll_schedule:
    1. Records entry/exit prices for both spot and futures
    2. Calculates basis trade performance
    3. Tracks cash flows and fees
    
    Args:
        spot_data: DataFrame with:
            - Index: DatetimeIndex (aligned with futures_data)
            - Columns: Must include 'spot_close' (price)
        futures_data: DataFrame with:
            - Index: DatetimeIndex 
            - Columns: Must include 'contract' (str) and 'futures_close' (price)
        roll_schedule: Output from generate_roll_dates(), containing:
            - open_time: When to enter positions
            - close_time: When to exit positions
            - contract: Which futures contract to trade
    
    Returns:
        tuple: (results_df, cash_flows_dict)
        - results_df: DataFrame with detailed trade metrics including:
            * Trade metadata (timing, contract)
            * Entry/exit prices
            * Basis calculations
            * Return metrics (gross/net/annualized)
            * Fee breakdown
        - cash_flows_dict: Dictionary with:
            * Keys: 'Trade_1', 'Trade_2', etc.
            * Values: Dicts containing:
                - Timeline (open/close dates)
                - Position details (prices)
                - Cash movements (in/out/net)
                - Human-readable summary
    
    Data Flow Example:
        generate_roll_dates() → [
            (2023-12-01, 2024-03-28, 'BTCUSD_2403'),
            (2024-03-29, 2024-06-27, 'BTCUSD_2406')
        ]
        ↓
        calculate_returns() executes trades at these exact timestamps,
        fetching corresponding prices from spot_data/futures_data
    """
    results = []
    cash_flows = {}
    spot_transaction_rate = SPOT_TRANSACTION_RATE
    future_transaction_rate = FUTURE_TRANSACTION_RATE
    
    for i, (open_time, close_time, contract) in enumerate(roll_schedule):
        spot_entry = spot_data.loc[open_time, 'spot_close']
        future_entry = futures_data.loc[
            (futures_data.index == open_time) & 
            (futures_data['contract'] == contract),
            'futures_close'
        ].iloc[0]

        spot_exit = spot_data.loc[close_time, 'spot_close']
        future_exit = futures_data.loc[
            (futures_data.index == close_time) & 
            (futures_data['contract'] == contract),
            'futures_close'
        ].iloc[0]
        
        # Calculate basis
        entry_basis = future_entry - spot_entry
        exit_basis = future_exit - spot_exit
        basis_change = entry_basis - exit_basis

        spot_fee = spot_transaction_rate * (spot_entry + spot_exit)
        future_fee = future_transaction_rate * (future_entry + future_exit)
        total_fee = spot_fee + future_fee
        holding_days = (close_time-open_time).days
        # Transaction details
        trade_details = {
            # Trade metadata
            'trade_id': i+1,
            'open_time': open_time,
            'close_time': close_time,
            'contract': contract,
            
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
            'holding_days': holding_days,
            
            # Returns calculation
            'spot_return': (spot_exit - spot_entry)/spot_entry,
            'future_return': (future_entry - future_exit)/future_entry,
            'gross_return': (basis_change)/spot_entry,
            'net_return': (basis_change-total_fee)/spot_entry,  
            'annualized_return': (1 + (basis_change-total_fee)/spot_entry)**(365/holding_days) - 1,
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
                'open': open_time.strftime('%Y-%m-%d'), 
                'close': close_time.strftime('%Y-%m-%d')
            },
            
            # Positions - record exact prices when actions happened
            'positions': {
                'open': {
                    'spot_price': spot_entry,
                    'future_price': future_entry,
                    'future_contract': contract
                },
                'close': {
                    'spot_price': spot_exit,
                    'future_price': future_exit,
                    'future_contract': contract
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
                f"Opened on {open_time.date()}: Bought 1 BTC at ${spot_entry} | Sold {contract} at ${future_entry}",
                f"Closed on {close_time.date()}: Sold BTC at ${spot_exit} | Bought back {contract} at ${future_exit}",
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

def plot_results_enhanced(results_df, initial_nav=1.0):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 计算累计收益
    results_df['cumulative'] = (1 + results_df['net_return']).cumprod() * initial_nav
    results_df['return_pct'] = results_df['net_return'] * 100  # 单期收益%

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')

    # 累计收益曲线
    ax.plot(results_df['close_time'], results_df['cumulative'], 
            marker='o', color='#2c7bb6', label='Cumulative Return (NAV)', linewidth=2)

    # 标注每个点的收益%
    for idx, row in results_df.iterrows():
        label = f"{row['return_pct']:.2f}%"
        ax.annotate(label, 
                    (row['close_time'], row['cumulative']), 
                    textcoords="offset points", xytext=(0, 6), ha='center',
                    fontsize=8, color='gray')

    # 基础设置
    ax.set_title('BTC Quarterly Arbitrage Strategy – Cumulative Return', fontsize=14, pad=15)
    ax.set_ylabel('Cumulative NAV', fontsize=12)
    ax.set_xlabel('Quarter End', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    # 时间格式美化
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # 输出
    plt.tight_layout()
    plt.savefig('btc_arbitrage_enhanced.png', dpi=300)
    plt.show()

def save_results(return_data, cash_flows, output_dir="results"):
    """
    Saves trade results to files instead of printing to console.
    
    Creates:
    - results/trade_records.csv: Detailed trade records
    - results/cash_flows.json: Complete cash flow data
    - results/summary.txt: Human-readable summary
    """

    
    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Save trade records (return_data) to CSV
    trade_cols = ['trade_id', 'contract', 'open_time', 'close_time',
                 'spot_entry_price', 'future_entry_price',
                 'net_return', 'annualized_return']
    return_data[trade_cols].to_csv(f"{output_dir}/trade_records.csv", index=False)
    
    # 2. Save full cash flows to JSON (preserves structure)
    with open(f"{output_dir}/cash_flows.json", 'w') as f:
        json.dump(cash_flows, f, indent=2, default=str)
    
    # 3. Save human-readable summary to text file
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("TRADE SUMMARY\n")
        f.write("="*50 + "\n")
        for trade_id, data in cash_flows.items():
            f.write(f"\n{trade_id}:\n")
            f.write(f"Dates: {data['date']['open']} to {data['date']['close']}\n")
            f.write(f"Spot: {data['positions']['open']['spot_price']} → {data['positions']['close']['spot_price']}\n")
            f.write(f"Future: {data['positions']['open']['future_price']} → {data['positions']['close']['future_price']}\n")
            f.write(f"Net P&L: ${data['cash']['net']:,.2f}\n")
            f.write("-"*50 + "\n")

def save_all_results(return_data, cash_flows, output_dir="results"):
    """
    Saves all trade results to a single structured file.
    Creates:
    - results/full_results.json: Contains all data including:
        * trade_records (DataFrame contents)
        * cash_flows (original structure)
        * performance_metrics (analysis)
    """
    import os
    import json
    from pathlib import Path
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert DataFrame to dict
    trade_records = return_data.to_dict('records')
    
    # Generate performance metrics
    metrics = calculate_performance_metrics(return_data, cash_flows)
    
    # Combine everything
    full_results = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'strategy': 'BTC Basis Trade'
        },
        'trade_records': trade_records,
        'cash_flows': cash_flows,
        'performance_metrics': metrics
    }
    
    # Save to single file
    with open(f"{output_dir}/full_results.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"All results saved to {output_dir}/full_results.json")

def calculate_performance_metrics(return_data, cash_flows):
    """Calculates metrics without printing"""
    total_trades = len(return_data)
    winning_trades = len(return_data[return_data['net_return'] > 0])
    
    return {
        'total_trades': total_trades,
        'win_rate': winning_trades / total_trades,
        'total_net_profit': sum(trade['cash']['net'] for trade in cash_flows.values()),
        'avg_profit': return_data[return_data['net_return'] > 0]['net_return'].mean(),
        'max_profit': return_data[return_data['net_return'] > 0]['net_return'].max(),
        'min_profit': return_data[return_data['net_return'] > 0]['net_return'].min(),
        'max_loss': return_data[return_data['net_return'] <= 0]['net_return'].max(),
        'min_loss': return_data[return_data['net_return'] <= 0]['net_return'].min(),

        'avg_loss': return_data[return_data['net_return'] <= 0]['net_return'].mean(),
        # 'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
        'max_drawdown': (return_data['net_return'].cumsum().min()),
        # 'sharpe_ratio': (return_data['net_return'].mean() / return_data['net_return'].std()) * (252**0.5),
        'avg_holding_days': return_data['holding_days'].mean()
    }

def analyze_basic_strategy_results(return_data, cash_flows):
    """
    Analyzes strategy performance and prints key metrics
    """
    # Calculate basic metrics
    total_trades = len(return_data)
    winning_trades = len(return_data[return_data['net_return'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades
    
    # Calculate profit metrics
    total_net_profit = sum([trade['cash']['net'] for trade in cash_flows.values()])
    avg_net = return_data['net_return'].mean()

    avg_profit = return_data[return_data['net_return'] > 0]['net_return'].mean()
    avg_loss = return_data[return_data['net_return'] <= 0]['net_return'].mean()
    max_profit= return_data[return_data['net_return'] > 0]['net_return'].max()
    min_profit= return_data[return_data['net_return'] > 0]['net_return'].min()
    max_loss= return_data[return_data['net_return'] <= 0]['net_return'].max()
    min_loss= return_data[return_data['net_return'] <= 0]['net_return'].min()


# Calculate risk/reward
    profit_factor = -avg_profit / avg_loss if avg_loss != 0 else float('inf')
    
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE ANALYSIS".center(80))
    print("="*80)
    print(f"{'Total Trades:':<25}{total_trades:>10}")
    print(f"{'Winning Trades:':<25}{winning_trades:>10} ({win_rate:.1%})")
    print(f"{'Losing Trades:':<25}{losing_trades:>10}")
    print("-"*80)
    print(f"{'Total Net Profit:':<25}${total_net_profit:>10,.2f}")
    print(f"{'Average Net:':<25}{avg_net:>10.2%}")

    print(f"{'Average Profit:':<25}{avg_profit:>10.2%}")
    print(f"{'Average Loss:':<25}{avg_loss:>10.2%}")
    print(f"{'Max Profit:':<25}{max_profit:>10.2%}")
    print(f"{'Min Profit:':<25}{min_profit:>10.2%}")
    print(f"{'Max Loss:':<25}{max_loss:>10.2%}")
    print(f"{'Min Loss:':<25}{min_loss:>10.2%}")
    
    print(f"{'Profit Factor:':<25}{profit_factor:>10.2f}")
    print("="*80)
def save_performance_report(return_data, cash_flows, output_file="performance_report.txt"):
    """
    将策略分析结果保存为格式化的文本报告（仅包含最终显示的指标）
    """
    import pandas as pd
    from pathlib import Path
    
    # 计算所有指标
    total_trades = len(return_data)
    winning_trades = return_data[return_data['net_return'] > 0]
    losing_trades = return_data[return_data['net_return'] <= 0]
    win_rate = len(winning_trades) / total_trades
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'win_rate': win_rate,
        'losing_trades': len(losing_trades),
        'total_net_profit': sum(trade['cash']['net'] for trade in cash_flows.values()),
        'avg_profit': winning_trades['net_return'].mean(),
        'avg_loss': losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0,
        'max_profit': winning_trades['net_return'].max(),
        'min_profit': winning_trades['net_return'].min(),
        'max_loss': losing_trades['net_return'].max() if len(losing_trades) > 0 else 0,
        'min_loss': losing_trades['net_return'].min() if len(losing_trades) > 0 else 0,
        'profit_factor': -winning_trades['net_return'].mean() / losing_trades['net_return'].mean() \
                        if len(losing_trades) > 0 else float('inf')
    }
    
    # 生成格式化的文本
    report_text = f"""
{'='*80}
{'STRATEGY PERFORMANCE ANALYSIS'.center(80)}
{'='*80}
{'Total Trades:':<25}{metrics['total_trades']:>10}
{'Winning Trades:':<25}{metrics['winning_trades']:>10} ({metrics['win_rate']:.1%})
{'Losing Trades:':<25}{metrics['losing_trades']:>10}
{'-'*80}
{'Total Net Profit:':<25}${metrics['total_net_profit']:>10,.2f}
{'Average Profit:':<25}{metrics['avg_profit']:>10.2%}
{'Average Loss:':<25}{metrics['avg_loss']:>10.2%}
{'Max Profit:':<25}{metrics['max_profit']:>10.2%}
{'Min Profit:':<25}{metrics['min_profit']:>10.2%}
{'Max Loss:':<25}{metrics['max_loss']:>10.2%}
{'Min Loss:':<25}{metrics['min_loss']:>10.2%}
{'Profit Factor:':<25}{metrics['profit_factor']:>10.2f}
{'='*80}
"""
    
    # 写入文件
    Path(output_file).write_text(report_text)
    print(f"策略绩效报告已保存至 {output_file}")

# 调用示例

# 调用方式
    #手续费
    # 季度中交易，第一阶段收益率最大值/平均值做参考，吃了跑
    #3 吃了还要吃，建仓条件？1.新价差=初始价差
def basic_strategy():
    # 1.在这个季度开始前（上一个季度末/上一个比特币季度合约交割日结束之前）：买比特币现货，做空这个季度的比特币季度合约。
    # 2.在这个季度末（这个比特币季度合约交割日结束之前）：卖出比特币现货，平仓这个季度的比特币合约（完成交割？）｜买比特币，做空下一个季度的比特币季度合约...

    # 参数， 交易手续费 基础用户： 现货 0.1% 合约  0.05%
    # 资金费率，暂不考虑
    # Initialize
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. Get data
    print("获取现货数据...")
    # spot_data = backtester.get_spot_data(years=5, extra_days=0)
    # backtester.save_data(spot_data,'./data/5_year_spot_data.pkl')

    # load spot data of BTCUSTD from 5 years ago to 2025 Aug3 2 am
    backtester.load_spot_data('./data/5_year_spot_data.pkl')
    spot_data = backtester.spot_data
    print(spot_data.head(5))
    print("获取期货数据...")

    # load BTCUSD futures data since BTCUSD_20220325
    contracts = backtester.load_futures_data('./data/5_year_futures_data.pkl')
    print(backtester.futures_data.head(5))
    # contracts = backtester.get_all_futures_data(start_date='2021-12-01')
    # backtester.save_data(backtester.futures_data,'./data/5_year_futures_data.pkl')
    futures_data = backtester.futures_data
    # validate_contract_dates(futures_data,contracts)

    # 2. Generate trading signals with improved error handling
    roll_dates = generate_roll_dates(futures_data,contracts)
    # print("交易日：")
    # print(roll_dates)
    return_data, cash_flows = calculate_returns(spot_data,futures_data,roll_dates)
    # print("交易记录")
    # print_trade_summary(cash_flows)
    # print(return_data[['trade_id', 'contract', 'open_time', 'close_time',
    #                 'spot_entry_price', 'future_entry_price',
    #                 'net_return', 'annualized_return']])
    save_all_results(return_data, cash_flows)
    # plot_results(return_data)

    # plot_results_enhanced(return_data)
    save_performance_report(return_data, cash_flows)

def check_dates():
        # 初始化
    backtester = BinanceArbitrageBacktester(symbol='BTCUSDT', interval='4h')
    
    # 1. 获取数据并打印结构
    # print("正在获取现货数据...")
    # spot_data = backtester.get_spot_data(years=2, extra_days=0)

    # print("现货数据列名:", spot_data.columns.tolist())
    # print("现货数据示例:\n", spot_data.head(2))

    # 现货数据列名: ['spot_open', 'spot_high', 'spot_low', 'spot_close', 'spot_volume']
    # 现货数据示例:
    #                     spot_open  spot_high  spot_low  spot_close  spot_volume
    # timestamp                                                                   
    # 2023-08-02 00:00:00   29705.99   30047.50   29622.5    29632.96  12892.26158
    # 2023-08-02 04:00:00   29632.96   29719.77   29564.0    29577.26   5196.11558
    backtester.load_spot_data('./data/5_year_spot_data.pkl')
    spot_data = backtester.spot_data
    print("\n正在获取期货数据...")
    # contracts = backtester.get_all_futures_data(start_date='2024-01-01')
    contracts = backtester.load_futures_data('./data/5_year_futures_data.pkl')
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

if __name__ == "__main__":
    basic_strategy()
    # check_dates()