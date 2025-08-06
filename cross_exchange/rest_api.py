import requests
import pandas as pd
import functools
from functools import reduce
import time
import logging

def fetch_all_symbols(exchange, market_type="spot"):
    """获取交易所支持的所有交易对（现货或合约）"""
    suffix = None

    if exchange == "binance":
        if market_type == 'spot':
            url = "https://api.binance.com/api/v3/exchangeInfo"
            data = requests.get(url).json()
            symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
        elif market_type == "futures_p":
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            data = requests.get(url).json()
            symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
        elif market_type == "futures_d":
            url = "https://dapi.binance.com/dapi/v1/exchangeInfo"
            data = requests.get(url).json()
            symbols = [s["symbol"] for s in data["symbols"] if s["contractStatus"] == "TRADING"]

    
    elif exchange == "bybit":
        if market_type == 'spot':
            suffix = 'spot'
        elif market_type == 'futures_p':
            suffix = 'linear'
        elif market_type == 'futures_d':
            suffix = 'inverse'
            
        url = "https://api.bybit.com/v5/market/instruments-info?category=" + suffix
        data = requests.get(url).json()
        symbols = [s["symbol"] for s in data["result"]["list"] if s["status"] == "Trading"]
    
    elif exchange == "okx":
        if market_type == 'spot':
            suffix = 'SPOT'
        elif market_type == 'futures_p':
            suffix = 'SWAP'
        elif market_type == 'futures_d':
            suffix = 'FUTURES'
        url = "https://www.okx.com/api/v5/public/instruments?instType=" + suffix
        data = requests.get(url).json()
        symbols = [s["instId"] for s in data["data"]]

    
    elif exchange == "bitget":
        if market_type == 'spot':
            url = "https://api.bitget.com/api/spot/v1/public/products"
        
        elif market_type == 'futures_p':
            url = 'https://api.bitget.com/api/v2/mix/market/contracts?productType=usdt-futures'
        elif market_type == 'futures_d':
            url = "https://api.bitget.com/api/v2/mix/market/contracts?productType=coin-futures"
        data = requests.get(url).json()
        symbols = [s["symbol"] for s in data["data"]]
    
    else:
        raise ValueError("Unsupported exchange")
    
    return symbols

def get_all_symbols():
    platforms = ['binance', 'okx', 'bitget', 'bybit']
    market_types = ['spot', 'futures_p', 'futures_d']
    all_data = {}

    for platform in platforms:
        all_data[platform] = {}
        for market_type in market_types:
            try:
                symbols = fetch_all_symbols(platform, market_type)  # Now uses market_type
                all_data[platform][market_type] = symbols
                print(f"[SUCCESS] {platform} {market_type}: {len(symbols)} symbols")
            except Exception as e:
                print(f"[ERROR] {platform} {market_type}: {str(e)}")
                all_data[platform][market_type] = []  # Store empty list if failed

    return all_data

def save_symbols(all_data,file_path="./cross_exchange/exchange_symbols.csv"):
        # Convert all_data to a DataFrame
    rows = []
    for exchange in all_data:
        for market_type in all_data[exchange]:
            symbols = ",".join(all_data[exchange][market_type])  # Join symbols with commas
            rows.append({
                "Exchange": exchange,
                "Market Type": market_type,
                "Symbols": symbols
            })

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f'Saved to "{file_path}')

def get_all_symbols_and_save():
    all_data = get_all_symbols()
    save_symbols(all_data)

def clean_and_save():
    '''symbols get removed if does not contain USDT'''
    df = pd.read_csv('./cross_exchange/exchange_symbols.csv')
    df = df[df['Market Type'] != 'futures_d']
    df = df[df['Market Type'] != 'spot']


    filtered_symbols = []
    
    for symbols_str in df['Symbols']:
        symbols_list = symbols_str.split(',')
        usdt_symbols = []
        for symbol in symbols_list:
            if 'usdt' in symbol.lower():
                usdt_symbols.append(symbol)
        
        filtered_str = ','.join(usdt_symbols) if usdt_symbols else ''
        filtered_symbols.append(filtered_str)
    
    df['Symbols'] = filtered_symbols
    df = df[df['Symbols'] != ''].reset_index(drop=True)
    print(df.head())
    file_path = './cross_exchange/cleaned_symbols.csv'
    df.to_csv(file_path, index=False)
    print(f'Saved to "{file_path}')

def format_symbols():
    '''remove '-', '-SWAP' from symbols'''
    df = pd.read_csv('./cross_exchange/cleaned_symbols.csv')
    df['Symbols'] = df['Symbols'].str.replace('-SWAP','')
    df['Symbols'] = df['Symbols'].str.replace('-','')
    df = df[df['Symbols'] != ''].reset_index(drop=True)
    print(df.head())
    file_path = './cross_exchange/cleaner_symbols.csv'
    df.to_csv(file_path, index=False)
    print(f'Saved to "{file_path}')

def get_common_symbols():
    name_lists = []
    df = pd.read_csv('./cross_exchange/cleaner_symbols.csv')
    for symbols_str in df['Symbols']:
        symbols_list = symbols_str.split(',')
        name_lists.append(symbols_list)
    common_symbols = set(reduce(lambda x, y:set(x) & set(y), name_lists))
    return list(common_symbols)

if __name__ == '__main__':

    print("hello world!")
