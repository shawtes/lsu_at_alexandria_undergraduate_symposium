import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

def get_cached_symbols():
    """Get list of available cryptocurrency symbols"""
    try:
        url = "https://api.exchange.coinbase.com/products"
        response = requests.get(url)
        products = response.json()
        usd_pairs = [p['id'] for p in products if p['quote_currency'] == 'USD']
        return sorted(usd_pairs)
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def get_market_data(symbol, days=7, granularity=3600):
    """Fetch historical market data from Coinbase"""
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    headers = {'Accept': 'application/json'}
    df_list = []
    now = datetime.utcnow()
    step = timedelta(seconds=granularity * 300)
    start_time = now - timedelta(days=days)

    print(f"Fetching data for {symbol}...")
    while start_time < now:
        end_time = min(start_time + step, now)
        params = {
            'granularity': granularity,
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        }
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            start_time = end_time
            continue
        data = r.json()
        if data:
            df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df_list.append(df)
        start_time = end_time

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    return df.reset_index(drop=True)

def calculate_indicators(df):
    """Calculate technical indicators for analysis"""
    if len(df) < 50:
        return df
    
    # Trend Indicators
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Momentum Indicators
    df['RSI'] = 100 - (100 / (1 + df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() /
                               df['close'].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
    df['14-high'] = df['high'].rolling(window=14).max()
    df['14-low'] = df['low'].rolling(window=14).min()
    df['%K'] = 100 * ((df['close'] - df['14-low']) / (df['14-high'] - df['14-low']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Volatility Indicators
    df['rolling_std_10'] = df['close'].rolling(window=10).std()
    df['ATR'] = df['high'].sub(df['low']).rolling(window=14).mean()
    
    # Volume and Price Indicators
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['lag_1'] = df['close'].shift(1)
    df['lag_2'] = df['close'].shift(2)
    df['lag_3'] = df['close'].shift(3)
    
    return df

def calculate_momentum_score(df):
    """Calculate momentum score for a symbol"""
    if len(df) < 20:
        return 0
        
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate momentum indicators
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = 100 - (100 / (1 + df['returns'].apply(lambda x: max(x, 0)).rolling(14).mean() /
                               df['returns'].apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Calculate momentum score components
    trend_score = 100 * (latest['close'] / latest['ma20'] - 1)  # % above/below MA20
    ma_trend = 100 * (latest['ma20'] / latest['ma50'] - 1)  # MA20 vs MA50
    rsi_score = latest['rsi']  # RSI (0-100)
    vol_score = 100 * (df['volume'].tail(20).mean() / df['volume'].tail(50).mean() - 1)  # Volume trend
    
    # Combine into final score
    momentum_score = (
        0.4 * trend_score +
        0.3 * ma_trend +
        0.2 * (rsi_score - 50) +  # Center RSI around 0
        0.1 * vol_score
    )
    
    return momentum_score

def scan_market(symbols=None):
    """Scan market for trading opportunities"""
    if symbols is None:
        symbols = get_cached_symbols()
    
    results = []
    for symbol in symbols:
        try:
            df = get_market_data(symbol, days=7)
            if df.empty:
                continue
                
            score = calculate_momentum_score(df)
            if not np.isnan(score):
                results.append({
                    'symbol': symbol,
                    'momentum_score': score,
                    'last_price': df['close'].iloc[-1],
                    'volume_24h': df['volume'].tail(24).sum()
                })
                
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue
    
    results.sort(key=lambda x: x['momentum_score'], reverse=True)
    return results 