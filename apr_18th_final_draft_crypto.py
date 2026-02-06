import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import os
import joblib
import warnings
import queue
import requests
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from xgboost import XGBRegressor
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import warnings
from dash.exceptions import PreventUpdate
from simulation_tracker import SimulationTracker
import threading
import queue
import time
import sqlite3
import sys
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Import utility functions and ML backtest
from utils import get_cached_symbols, scan_market, get_market_data, calculate_indicators
from ml_backtest import run_ml_backtest

# Constants
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Add global variables for live trading
live_trading_thread = None
trading_queue = queue.Queue()
live_sim = None
is_trading_active = False

# Symbol cache configuration
SYMBOL_CACHE = {
    'data': None,
    'last_update': None,
    'cache_duration': timedelta(hours=1)  # Cache symbols for 1 hour
}

def get_cached_symbols():
    now = datetime.now()
    if (SYMBOL_CACHE['data'] is None or 
        SYMBOL_CACHE['last_update'] is None or 
        now - SYMBOL_CACHE['last_update'] > SYMBOL_CACHE['cache_duration']):
        try:
            url = "https://api.exchange.coinbase.com/products"
            response = requests.get(url)
            data = response.json()
            symbols = [product['id'] for product in data if product['quote_currency'] == 'USD']
            SYMBOL_CACHE['data'] = sorted(symbols)
            SYMBOL_CACHE['last_update'] = now
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            if SYMBOL_CACHE['data'] is None:
                SYMBOL_CACHE['data'] = []
    return SYMBOL_CACHE['data']

# === Data Fetching ===
def get_coinbase_data(symbol='BTC-USD', granularity=60, days=7):
    """
    Fetch historical data from Coinbase with improved rate limit handling.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC-USD')
        granularity (int): Time interval in seconds (60, 300, 900, 3600, etc.)
        days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: Historical price data with OHLCV columns
    """
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    headers = {'Accept': 'application/json'}
    df_list = []
    now = datetime.utcnow()
    
    # Calculate optimal step size based on granularity (max 300 candles per request)
    max_candles_per_request = 300
    step_seconds = min(granularity * max_candles_per_request, days * 24 * 3600)
    step = timedelta(seconds=step_seconds)
    start_time = now - timedelta(days=days)
    
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    print(f"\nFetching {days} days of {granularity}s data for {symbol}...")
    
    while start_time < now:
        end_time = min(start_time + step, now)
        params = {
            'granularity': granularity,
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        }
        
        for retry in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params)
                
                if r.status_code == 429:  # Rate limit hit
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    print(f"Rate limit hit, waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                    
                elif r.status_code != 200:
                    print(f"Error {r.status_code} fetching data for {symbol}: {r.text}")
                    break
                    
                data = r.json()
                if data:
                    temp_df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    df_list.append(temp_df)
                break  # Successful request, exit retry loop
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                if retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue
        
        start_time = end_time
        time.sleep(0.25)  # Small delay between successful requests
    
    if not df_list:
        print(f"No data retrieved for {symbol}")
        return pd.DataFrame()
    
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    df = df.drop_duplicates(subset=['timestamp'])
    
    print(f"Retrieved {len(df)} data points for {symbol}")
    return df.reset_index(drop=True)

def get_market_hours_data(symbol, period='7d', interval='1h', start_date=None, end_date=None):
    """
    Get market data for a given symbol with proper handling of market hours.
    
    Parameters:
    -----------
    symbol : str
        The trading symbol (e.g., 'BTC-USD')
    period : str
        The lookback period (e.g., '1d', '5d', '1mo')
    interval : str
        The data interval (e.g., '1m', '5m', '1h', '1d')
    start_date : datetime, optional
        Start date for data fetching
    end_date : datetime, optional
        End date for data fetching
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    try:
        # Convert interval to seconds for Coinbase API
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        granularity = interval_map.get(interval, 3600)
        
        # Calculate days based on period if dates not provided
        if not start_date and not end_date:
            period_days = {
                '1d': 7,  # Changed from 1 to 7 to ensure enough data
                '5d': 10, # Changed from 5 to 10
                '1mo': 35,
                '3mo': 95,
                '6mo': 185,
                '1y': 370
            }
            days = period_days.get(period, 30)
        else:
            days = (end_date - start_date).days if start_date and end_date else 30
            
        # Ensure we have enough days for indicator calculation
        days = max(days, 7)  # At least 7 days of data
        
        # Get data using existing function
        df = get_coinbase_data(symbol=symbol, granularity=granularity, days=days)
        
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame()
            
        # Apply date filters if provided
        if start_date:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]
            
        return df
        
    except Exception as e:
        print(f"Error fetching market data for {symbol}: {str(e)}")
        return pd.DataFrame()

# === Technical Indicators ===
def calculate_indicators(df):
    """
    Calculate technical indicators with proper error handling and data validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    try:
        # Verify required columns exist
        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return df
            
        # Convert price and volume columns to numeric, replacing errors with NaN
        for col in ['close', 'high', 'low', 'open', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['close', 'high', 'low'])
        
        if len(df) < 24:  # Minimum required for most indicators
            print("Insufficient data points for indicator calculation")
            return df
            
        # Calculate trend indicators
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # Calculate momentum indicators with proper handling of edge cases
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, float('inf'))  # Handle division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility indicators
        df['rolling_std_10'] = df['close'].rolling(window=10, min_periods=1).std()
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
        
        # Calculate volume indicators
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Calculate stochastic oscillator
        df['14-high'] = df['high'].rolling(window=14, min_periods=1).max()
        df['14-low'] = df['low'].rolling(window=14, min_periods=1).min()
        df['%K'] = 100 * ((df['close'] - df['14-low']) / (df['14-high'] - df['14-low']).replace(0, float('inf')))
        df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()
        
        # Calculate lagging indicators
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        df['lag_3'] = df['close'].shift(3)
        
        # Fill any remaining NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print("Successfully calculated all indicators")
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df

# === Model Training ===
def train_model_for_symbol(symbol, granularity=60):
    print(f"\n🔄 Training model for {symbol} at {granularity}s granularity...")
    
    # Determine appropriate training period based on granularity
    if granularity == 60:  # 1 minute
        training_days = 7  # 1 week of minute data
    elif granularity == 300:  # 5 minutes
        training_days = 14  # 2 weeks of 5-minute data
    elif granularity == 900:  # 15 minutes
        training_days = 30  # 1 month of 15-minute data
    else:  # 1 hour or higher
        training_days = 90  # 3 months of hourly data
    
    print(f"Fetching {training_days} days of {granularity}s data...")
    
    # Get and prepare data
    df = get_coinbase_data(symbol=symbol, granularity=granularity, days=training_days)
    if df.empty:
        print(f"❌ No data available for {symbol}")
        return None, None
    
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    if len(df) < 100:
        print(f"❌ Insufficient data for {symbol}: {len(df)} samples")
        return None, None
    
    feature_cols = [
        'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
        'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR', '%K', '%D'
    ]
    
    # Train price prediction model using a simplified ensemble
    base_models = [
        RandomForestRegressor(n_estimators=100, max_depth=10),
        XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=5),
        LinearRegression()
    ]
    
    # Create voting ensemble for regression
    reg_predictions = []
    for model in base_models:
        X_reg = df[feature_cols].iloc[:-1]
        y_reg = df['close'].shift(-1).dropna()
        model.fit(X_reg, y_reg)
        reg_predictions.append(model.predict(X_reg).reshape(-1, 1))
    
    # Average predictions from all models
    ensemble_predictions = np.mean(np.hstack(reg_predictions), axis=1)
    
    # Generate predictions for classifier
    df = df.iloc[:-1]
    df['predicted_close'] = ensemble_predictions
    df['direction'] = np.where(df['predicted_close'] > df['close'], 'BUY', 'SELL')
    
    # Train classifier using Random Forest
    X_cls = df[feature_cols + ['predicted_close']]
    y_cls = df['direction']
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_cls, y_cls)
    
    # Evaluate
    preds = clf.predict(X_cls)
    print("\n📊 Model Evaluation:")
    print(classification_report(y_cls, preds))
    
    # Save models
    model_prefix = f"{symbol.replace('-', '')}_{granularity}"
    reg_path = os.path.join(MODELS_DIR, f"{model_prefix}_reg.pkl")
    clf_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
    
    # Save the ensemble as a dictionary of models
    reg_models = {
        'models': base_models,
        'feature_cols': feature_cols
    }
    
    joblib.dump(reg_models, reg_path)
    joblib.dump(clf, clf_path)
    print(f"✅ Models saved: {model_prefix}")
    
    return reg_models, clf

# === Trading Simulation ===
def simulate_trading(df, reg_model, clf, initial_state=None):
    if initial_state is None:
        initial_state = {
            "cash": 100.0,
            "position": None,
            "quantity": 0,
            "entry": None,
            "total_profit": 0,
            "wins": 0,
            "losses": 0
        }
    
    sim_state = initial_state.copy()
    feature_cols = [
        'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
        'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR', '%K', '%D'
    ]
    
    trades = []
    for idx, row in df.iterrows():
        price = row['close']
        features = row[feature_cols].values.reshape(1, -1)
        
        # Get predictions
        price_pred = reg_model.predict(features)[0]
        features_with_pred = np.append(features, price_pred).reshape(1, -1)
        action_pred = clf.predict(features_with_pred)[0]
        
        action = "HOLD"
        profit = 0.0
        
        # Trading logic
        if action_pred == "BUY" and sim_state["cash"] > 0 and sim_state["position"] is None:
            # Enter long position
            buy_price = price * 1.001  # Include fee
            qty = (sim_state["cash"] * 0.95) / buy_price  # Use 95% of available cash
            sim_state["quantity"] = qty
            sim_state["position"] = "long"
            sim_state["entry"] = buy_price
            sim_state["cash"] -= qty * buy_price
            action = "BUY"
            
        elif action_pred == "SELL" and sim_state["position"] == "long":
            # Exit long position
            sell_price = price * 0.999  # Include fee
            qty = sim_state["quantity"]
            proceeds = qty * sell_price
            profit = proceeds - (qty * sim_state["entry"])
            
            sim_state["cash"] += proceeds
            sim_state["quantity"] = 0
            sim_state["position"] = None
            sim_state["total_profit"] += profit
            sim_state["wins"] += int(profit > 0)
            sim_state["losses"] += int(profit <= 0)
            action = "SELL"
        
        # Record trade
        crypto_value = sim_state["quantity"] * price
        total_value = sim_state["cash"] + crypto_value
        
        trades.append({
            "timestamp": row['timestamp'],
            "price": price,
            "predicted_price": price_pred,
            "action": action,
            "quantity": sim_state["quantity"],
            "cash": sim_state["cash"],
            "crypto_value": crypto_value,
            "total_value": total_value,
            "profit": profit
        })
    
    return pd.DataFrame(trades), sim_state

# === Scanner Functions ===
def analyze_momentum(df, symbol):
    """
    Analyze momentum indicators for a given symbol.
    
    Args:
        df (pd.DataFrame): DataFrame containing price data
        symbol (str): Symbol to analyze
    
    Returns:
        dict: Dictionary containing momentum analysis results
    """
    try:
        if df is None or df.empty or len(df) < 24:
            return None
            
        df = calculate_indicators(df)
        df.dropna(inplace=True)
        
        latest = df.iloc[-1]
        
        # Calculate price changes
        price_change_1h = ((latest['close'] / df['close'].iloc[-2]) - 1) * 100
        price_change_24h = ((latest['close'] / df['close'].iloc[-24]) - 1) * 100
        
        # Calculate momentum score
        momentum_score = (
            (latest['RSI'] / 100) * 0.3 +
            (1 if latest['MACD'] > latest['Signal_Line'] else 0) * 0.3 +
            (latest['%K'] / 100) * 0.2 +
            (1 if latest['close'] > latest['MA20'] else 0) * 0.2
        ) * 100
        
        return {
            'Symbol': symbol,
            'Price': latest['close'],
            'Momentum_Score': momentum_score,
            'Price_Change_1h': price_change_1h,
            'Price_Change_24h': price_change_24h,
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'Volume': latest['volume']
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def scan_market(symbols=None, batch_size=5):
    """
    Scan the market for trading opportunities.
    
    Args:
        symbols (list, optional): List of symbols to scan. If None, uses cached symbols.
        batch_size (int): Number of symbols to process in parallel.
    
    Returns:
        list: List of dictionaries containing analysis results for each symbol.
    """
    if symbols is None:
        symbols = get_cached_symbols()[:50]  # Limit to top 50 by default
    
    def process_symbol(symbol):
        try:
            df = get_coinbase_data(symbol=symbol, granularity=3600)  # 1-hour data
            if not df.empty:
                return analyze_momentum(df, symbol)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        return None
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
    
    return sorted(results, key=lambda x: x['Momentum_Score'], reverse=True)

def scan_for_crypto_runs(max_pairs=20):
    """
    Scan the crypto market for trading opportunities.
    Returns a list of dictionaries containing trading signals and metrics.
    """
    results = []
    try:
        symbols = get_cached_symbols()
        if not symbols:
            return []

        for symbol in symbols[:max_pairs]:  # Limit to max_pairs
            try:
                # Get recent data
                df = get_coinbase_data(symbol=symbol, granularity=3600, days=5)  # 5 days of hourly data
                if df.empty:
                    continue

                # Calculate indicators
                df = calculate_indicators(df)
                if df.empty:
                    continue

                # Get latest values
                latest = df.iloc[-1]
                
                # Calculate momentum score (0-100)
                momentum_score = 0
                if latest['RSI'] > 50:
                    momentum_score += 20
                if latest['MACD'] > latest['Signal_Line']:
                    momentum_score += 20
                if latest['close'] > latest['MA20']:
                    momentum_score += 20
                if latest['%K'] > latest['%D']:
                    momentum_score += 20
                if latest['OBV'] > df['OBV'].mean():
                    momentum_score += 20

                # Calculate volatility
                volatility = latest['ATR'] / latest['close'] * 100

                # Calculate trend strength
                trend_strength = abs(latest['close'] - latest['MA20']) / latest['MA20'] * 100

                results.append({
                    'symbol': symbol,
                    'current_price': float(latest['close']),
                    'momentum_score': float(momentum_score),
                    'rsi': float(latest['RSI']),
                    'volume_change_pct': float((latest['volume'] - df['volume'].mean()) / df['volume'].mean() * 100),
                    'price_change_pct': float((latest['close'] - df['close'].shift(1).iloc[-1]) / df['close'].shift(1).iloc[-1] * 100),
                    'timestamp': datetime.utcnow().isoformat()
                })

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue

        # Sort by momentum score
        results = sorted(results, key=lambda x: x['momentum_score'], reverse=True)
        
    except Exception as e:
        print(f"Error in scan_for_crypto_runs: {str(e)}")
    
    return results[:max_pairs] if results else []

# Define styles
styles = {
    'context_menu': {
        'display': 'none',
        'position': 'fixed',
        'backgroundColor': '#ffffff',
        'boxShadow': '2px 2px 5px rgba(0,0,0,0.2)',
        'zIndex': 1000,
        'borderRadius': '4px',
        'padding': '5px 0',
        'border': '1px solid #ddd'
    },
    'context_option': {
        'padding': '8px 20px',
        'cursor': 'pointer',
        'color': '#2c3e50',
        'hover': {'backgroundColor': '#f5f5f5'}
    },
    'main_container': {
        'fontFamily': 'Arial, sans-serif',
        'margin': '0',
        'padding': '20px',
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'minHeight': '100vh'
    },
    'table_cell': {
        'textAlign': 'center',
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'cursor': 'context-menu',
        'border': '1px solid #ddd'
    },
    'table_header': {
        'backgroundColor': '#f8f9fa',
        'color': '#2c3e50',
        'fontWeight': 'bold',
        'cursor': 'context-menu',
        'border': '1px solid #ddd'
    }
}

app.layout = html.Div([
    html.H1("🚀 Crypto Trading Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Add interval component for regular updates
    dcc.Interval(
        id='interval-component',
        interval=300000,  # 5 minutes in milliseconds
        n_intervals=0
    ),
    
    # Context Menu
    html.Div(
        id='context-menu',
        style=styles['context_menu'],
        children=[
            html.Div('Train Model', id='context-train', 
                    style=styles['context_option']),
            html.Div('Run Simulation', id='context-simulate', 
                    style=styles['context_option']),
            html.Div('View Analysis', id='context-analyze', 
                    style=styles['context_option'])
        ]
    ),
    
    # Store for selected symbol with default value
    dcc.Store(id='selected-symbol', data=None),
    
    # Tabs
    dcc.Tabs([
        # Scanner Tab
        dcc.Tab(label="Scanner", children=[
            html.Div([
                html.Div([
                    html.Button(
                        "🔄 Refresh Scanner", 
                        id="refresh-scanner", 
                        n_clicks=0,
                        style={
                            'marginBottom': '20px',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    ),
                    html.Div(
                        id="scanner-table",
                        children=html.Div("Click Refresh to load data", style={'color': '#2c3e50'})
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ])
        ], style={'backgroundColor': '#ffffff'}),
        
        # Training Tab
        dcc.Tab(label="Training", children=[
            html.Div([
                # Manual Input Section
                html.Div([
                    html.H3("Manual Training", style={'color': '#4CAF50'}),
                    dcc.Input(
                        id='manual-symbol-input',
                        type='text',
                        placeholder='Enter symbol (e.g., BTC-USD)',
                        style={
                            'width': '200px',
                            'marginRight': '10px',
                            'padding': '5px',
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'border': '1px solid #4CAF50'
                        }
                    ),
                    html.Button(
                        "🎯 Train Model",
                        id="manual-train-button",
                        n_clicks=0,
                        style={
                            'marginRight': '10px',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '5px 10px',
                            'borderRadius': '3px'
                        }
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Dropdown Selection Section
                html.Div([
                    html.H3("Selected Symbol Training", style={'color': '#2196F3'}),
                    dcc.Dropdown(
                        id='train-symbol-dropdown',
                        placeholder="Select Symbol",
                        style={'width': '200px', 'marginRight': '10px', 'backgroundColor': '#ffffff'}
                    ),
                    dcc.Dropdown(
                        id='train-granularity-dropdown',
                        options=[
                            {'label': '1 minute', 'value': 60},
                            {'label': '5 minutes', 'value': 300},
                            {'label': '15 minutes', 'value': 900},
                            {'label': '1 hour', 'value': 3600}
                        ],
                        style={'width': '200px', 'marginRight': '10px', 'backgroundColor': '#ffffff'}
                    ),
                    html.Button(
                        "🎯 Train Selected",
                        id="train-button",
                        n_clicks=0,
                        style={
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'border': 'none',
                            'padding': '5px 10px',
                            'borderRadius': '3px'
                        }
                    ),
                ]),
                
                # Training Status
                html.Div(id="training-status", style={'marginTop': '20px', 'color': '#2c3e50'}),
                
                # Training History
                html.Div(id="training-history", style={'marginTop': '20px', 'color': '#2c3e50'})
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Trading Tab
        dcc.Tab(label="Trading", children=[
            html.Div([
                dcc.Dropdown(
                    id='trade-symbol-dropdown',
                    placeholder="Select Symbol",
                    style={'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                ),
                dcc.Dropdown(
                    id='trade-granularity-dropdown',
                    options=[
                        {'label': '1 minute', 'value': 60},
                        {'label': '5 minutes', 'value': 300},
                        {'label': '15 minutes', 'value': 900},
                        {'label': '1 hour', 'value': 3600}
                    ],
                    value=3600,
                    style={'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                ),
                html.Div([
                    html.Label("Simulation Period:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                    dcc.DatePickerRange(
                        id='simulation-date-range',
                        min_date_allowed=datetime(2020, 1, 1),
                        max_date_allowed=datetime.now(),
                        initial_visible_month=datetime.now(),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        style={'backgroundColor': '#ffffff'}
                    ),
                ], style={'margin': '10px 0', 'display': 'flex', 'alignItems': 'center'}),
                html.Button(
                    "▶️ Start Simulation",
                    id="simulate-button",
                    n_clicks=0,
                    style={
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'borderRadius': '5px',
                        'cursor': 'pointer'
                    }
                ),
                dcc.Graph(id="price-chart"),
                dcc.Graph(id="portfolio-chart"),
                html.Div(id="simulation-stats", style={'color': '#2c3e50'})
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Analysis Tab
        dcc.Tab(label="Analysis", children=[
            html.Div([
                dcc.Dropdown(
                    id='analysis-symbol-dropdown',
                    placeholder="Select Symbol",
                    style={'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                ),
                dcc.Graph(id="profit-loss-chart"),
                dcc.Graph(id="drawdown-chart"),
                html.Div(id="analysis-stats", style={'color': '#2c3e50'})
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Portfolio Testing Tab
        dcc.Tab(label="Portfolio Testing", children=[
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label='Backtest', children=[
                        html.Div([
                            html.Label("Test Period:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.DatePickerRange(
                                id='portfolio-test-dates',
                                min_date_allowed=datetime(2020, 1, 1),
                                max_date_allowed=datetime.now(),
                                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                end_date=datetime.now().strftime('%Y-%m-%d'),
                                style={'backgroundColor': '#ffffff'}
                            ),
                            html.Label("Initial Portfolio Value:", style={'marginLeft': '20px', 'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.Input(
                                id='initial-portfolio-value',
                                type='number',
                                value=100000,
                                step=1000,
                                style={'width': '150px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                            ),
                            html.Label("Max Positions:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.Input(
                                id='max-positions',
                                type='number',
                                value=5,
                                min=1,
                                max=20,
                                step=1,
                                style={'width': '80px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                            ),
                            html.Button(
                                '▶️ Run Backtest',
                                id='run-portfolio-backtest-btn',
                                style={
                                    'backgroundColor': '#2196F3',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'borderRadius': '5px',
                                    'cursor': 'pointer'
                                }
                            )
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '20px'}),
                        html.Div(id='backtest-results', style={'color': '#2c3e50'})
                    ], style={'backgroundColor': '#ffffff'}),
                    dcc.Tab(label='Forward Test', children=[
                        html.Div([
                            html.Label("Days to Simulate:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.Input(
                                id='forward-test-days',
                                type='number',
                                value=30,
                                min=1,
                                max=90,
                                step=1,
                                style={'width': '80px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                            ),
                            html.Label("Initial Portfolio Value:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.Input(
                                id='forward-initial-portfolio',
                                type='number',
                                value=100000,
                                step=1000,
                                style={'width': '150px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                            ),
                            html.Label("Max Positions:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                            dcc.Input(
                                id='forward-max-positions',
                                type='number',
                                value=5,
                                min=1,
                                max=20,
                                step=1,
                                style={'width': '80px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                            ),
                            html.Button(
                                '▶️ Run Forward Test',
                                id='run-forward-test-btn',
                                style={
                                    'backgroundColor': '#4CAF50',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'borderRadius': '5px',
                                    'cursor': 'pointer'
                                }
                            )
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px', 'padding': '20px'}),
                        html.Div(id='forward-test-results', style={'color': '#2c3e50'})
                    ], style={'backgroundColor': '#ffffff'})
                ], style={'backgroundColor': '#ffffff'})
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'}),
        
        # Live Trading Tab
        dcc.Tab(label='Live Trading', children=[
            html.Div([
                # Trading Controls Section
                html.Div([
                    html.Label("Initial Portfolio Value ($):", style={'marginRight': '10px', 'color': '#2c3e50'}),
                    dcc.Input(
                        id='live-initial-portfolio',
                        type='number',
                        value=100000,
                        step=1000,
                        style={'width': '150px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                    ),
                    
                    html.Label("Max Positions:", style={'marginRight': '10px', 'color': '#2c3e50'}),
                    dcc.Input(
                        id='live-max-positions',
                        type='number',
                        value=5,
                        min=1,
                        max=20,
                        step=1,
                        style={'width': '80px', 'marginRight': '20px', 'backgroundColor': '#ffffff', 'color': '#2c3e50'}
                    ),
                    
                    html.Button(
                        '▶️ Start Live Trading',
                        id='start-live-trading-btn',
                        style={
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'marginRight': '10px'
                        }
                    ),
                    
                    html.Button(
                        '⏹️ Stop Trading',
                        id='stop-live-trading-btn',
                        style={
                            'backgroundColor': '#f44336',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '20px',
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Performance Metrics Section
                html.Div([
                    html.Div(id='live-trading-status', style={
                        'fontSize': '18px',
                        'marginBottom': '20px',
                        'color': '#2c3e50',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    }),
                    html.Div(id='live-performance-metrics', style={
                        'color': '#2c3e50',
                        'padding': '20px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '5px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'marginBottom': '20px'
                }),
                
                # Portfolio Chart
                html.Div([
                    dcc.Graph(
                        id='live-portfolio-chart',
                        style={'backgroundColor': '#ffffff'}
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Current Positions Table
                html.Div([
                    html.H4("Current Positions", style={
                        'color': '#2c3e50',
                        'marginBottom': '15px',
                        'paddingLeft': '10px'
                    }),
                    dash_table.DataTable(
                        id='live-positions-table',
                        columns=[
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Entry Price', 'id': 'entry_price'},
                            {'name': 'Current Price', 'id': 'current_price'},
                            {'name': 'P/L', 'id': 'pnl'},
                            {'name': 'P/L %', 'id': 'pnl_pct'}
                        ],
                        style_table={
                            'overflowX': 'auto',
                            'backgroundColor': '#ffffff'
                        },
                        style_cell={
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'padding': '10px',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'border': '1px solid #ddd'
                        },
                        style_data={
                            'border': '1px solid #ddd'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{pnl} contains "+"'},
                                'color': '#4CAF50'
                            },
                            {
                                'if': {'filter_query': '{pnl} contains "-"'},
                                'color': '#f44336'
                            }
                        ]
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginBottom': '20px'
                }),
                
                # Recent Trades Table
                html.Div([
                    html.H4("Recent Trades", style={
                        'color': '#2c3e50',
                        'marginBottom': '15px',
                        'paddingLeft': '10px'
                    }),
                    dash_table.DataTable(
                        id='live-trades-table',
                        columns=[
                            {'name': 'Time', 'id': 'timestamp'},
                            {'name': 'Symbol', 'id': 'symbol'},
                            {'name': 'Action', 'id': 'action'},
                            {'name': 'Price', 'id': 'price'},
                            {'name': 'Quantity', 'id': 'quantity'},
                            {'name': 'Value', 'id': 'value'}
                        ],
                        style_table={
                            'overflowX': 'auto',
                            'backgroundColor': '#ffffff'
                        },
                        style_cell={
                            'backgroundColor': '#ffffff',
                            'color': '#2c3e50',
                            'textAlign': 'center',
                            'padding': '10px',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'border': '1px solid #ddd'
                        },
                        style_data={
                            'border': '1px solid #ddd'
                        },
                        style_data_conditional=[
                            {
                                'if': {'filter_query': '{action} = "BUY"'},
                                'color': '#4CAF50'
                            },
                            {
                                'if': {'filter_query': '{action} = "SELL"'},
                                'color': '#f44336'
                            }
                        ]
                    )
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
                
                # Auto-refresh interval
                dcc.Interval(
                    id='live-trading-interval',
                    interval=30*1000,  # 30 seconds
                    n_intervals=0
                )
            ], style={
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={'backgroundColor': '#ffffff'})
    ], style={
        'backgroundColor': '#ffffff',
        'color': '#2c3e50',
        'borderBottom': '1px solid #ddd'
    })
], style={'backgroundColor': '#ffffff', 'color': '#2c3e50', 'minHeight': '100vh'})

# === Callbacks ===
@app.callback(
    [Output('scanner-table', 'children'),
     Output('refresh-scanner', 'children'),
     Output('refresh-scanner', 'disabled')],
    [Input('refresh-scanner', 'n_clicks')],
    prevent_initial_call=False
)
def update_scanner(n_clicks):
    if n_clicks is None:
        return (
            html.Div("Click Refresh to load data", style={'color': '#2c3e50'}),
            "🔄 Refresh Scanner",
            False
        )
    
    try:
        # Show loading state
        loading_div = html.Div([
            html.Div("Loading scanner data...", style={
                'marginBottom': '10px',
                'color': '#2c3e50',
                'textAlign': 'center'
            }),
            html.Div(className="loader")
        ])
        
        # Get symbols and limit to top 20
        symbols = get_cached_symbols()[:20]  # Limit to top 20 symbols
        
        # Run market scan with the symbols
        results = scan_market(symbols=symbols, batch_size=5)
        
        if not results:
            return (
                html.Div("No results found", style={'color': '#2c3e50'}),
                "🔄 Refresh Scanner",
                False
            )
        
        # Create DataFrame and table
        df = pd.DataFrame(results)
        
        # Create DataTable with consistent ID
        table = dash_table.DataTable(
            id='scanner-datatable',  # This ID must match the one used in other callbacks
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': '#ffffff',
                'color': '#2c3e50',
                'border': '1px solid #cccccc',
                'padding': '10px',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': '#f5f5f5',
                'fontWeight': 'bold',
                'border': '1px solid #cccccc'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'price_change_24h', 'filter_query': '{price_change_24h} > 0'},
                    'color': '#4CAF50'
                },
                {
                    'if': {'column_id': 'price_change_24h', 'filter_query': '{price_change_24h} < 0'},
                    'color': '#f44336'
                }
            ],
            sort_action='native',
            filter_action='native',
            row_selectable='single',
            selected_rows=[],
            page_size=10
        )
        
        return table, "🔄 Refresh Scanner", False
        
    except Exception as e:
        print(f"Error in update_scanner: {str(e)}")
        return (
            html.Div(f"Error refreshing data: {str(e)}", style={'color': '#2c3e50'}),
            "🔄 Refresh Scanner",
            False
        )

@app.callback(
    [Output('train-symbol-dropdown', 'options'),
     Output('trade-symbol-dropdown', 'options'),
     Output('analysis-symbol-dropdown', 'options')],
    Input('interval-component', 'n_intervals')
)
def update_symbol_dropdowns(n_intervals):
    symbols = get_cached_symbols()
    options = [{'label': s, 'value': s} for s in symbols]
    return options, options, options

@app.callback(
    Output('training-status', 'children'),
    [Input('manual-train-button', 'n_clicks'),
     Input('train-button', 'n_clicks')],
    [State('manual-symbol-input', 'value'),
     State('train-symbol-dropdown', 'value'),
     State('train-granularity-dropdown', 'value')]
)
def train_model(manual_clicks, dropdown_clicks, manual_symbol, dropdown_symbol, granularity):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine which symbol to use
    if trigger_id == 'manual-train-button':
        if not manual_symbol:
            return html.Div("❌ Please enter a symbol", style={'color': 'red'})
        symbol = manual_symbol.upper()
    else:  # train-button
        if not dropdown_symbol:
            return html.Div("❌ Please select a symbol", style={'color': 'red'})
        symbol = dropdown_symbol
    
    # Add loading message
    loading_div = html.Div([
        html.P("🔄 Training model...", style={'color': '#2196F3'}),
        html.Div(className="loader")
    ])
    
    # Train the model
    try:
        reg_model, clf = train_model_for_symbol(symbol, granularity)
        if reg_model is None or clf is None:
            return html.Div("❌ Training failed", style={'color': 'red'})
        
        # Return success message with details
        return html.Div([
            html.H4("✅ Training Complete", style={'color': 'green'}),
            html.P([
                html.Strong("Symbol: "), 
                html.Span(symbol)
            ]),
            html.P([
                html.Strong("Granularity: "), 
                html.Span(f"{granularity//60} minutes" if granularity < 3600 else "1 hour")
            ]),
            html.P([
                html.Strong("Models Saved: "), 
                html.Span(f"{symbol.replace('-', '')}_{granularity}")
            ])
        ])
        
    except Exception as e:
        return html.Div([
            html.H4("❌ Error", style={'color': 'red'}),
            html.P(str(e))
        ])

# Add callback to update training history
@app.callback(
    Output('training-history', 'children'),
    [Input('training-status', 'children')]
)
def update_training_history(status):
    try:
        # Get list of trained models
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if not model_files:
            return html.Div("No trained models found")
        
        # Group models by symbol
        models_by_symbol = {}
        for f in model_files:
            symbol = f.split('_')[0]
            if symbol not in models_by_symbol:
                models_by_symbol[symbol] = []
            models_by_symbol[symbol].append(f)
        
        # Create training history display
        return html.Div([
            html.H4("Trained Models", style={'color': '#2196F3'}),
            html.Div([
                html.Div([
                    html.H5(symbol),
                    html.Ul([
                        html.Li(model.replace('.pkl', '')) 
                        for model in sorted(models)
                    ])
                ]) 
                for symbol, models in models_by_symbol.items()
            ])
        ])
    except Exception as e:
        return html.Div(f"Error loading training history: {str(e)}")

def plot_prediction_errors(predictions_df):
    """
    Create a figure showing prediction errors and error distribution.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing actual and predicted prices
        
    Returns:
        go.Figure: A plotly figure with error analysis
    """
    if predictions_df.empty:
        return go.Figure()
    
    # Calculate errors
    predictions_df['error'] = predictions_df['actual_price'] - predictions_df['predicted_price']
    predictions_df['error_pct'] = (predictions_df['error'] / predictions_df['actual_price']) * 100
    predictions_df['abs_error'] = abs(predictions_df['error'])
    
    # Create subplots: error over time and error distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Prediction Error Over Time',
            'Error Distribution',
            'Error vs Price Level',
            'Cumulative Error'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Error over time
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['error'],
            mode='lines',
            name='Error',
            line=dict(color='#EF5350')
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", 
        line_color="gray",
        row=1, col=1
    )
    
    # 2. Error distribution histogram
    fig.add_trace(
        go.Histogram(
            x=predictions_df['error'],
            name='Error Distribution',
            nbinsx=30,
            marker_color='#42A5F5'
        ),
        row=1, col=2
    )
    
    # 3. Error vs Price Level scatter
    fig.add_trace(
        go.Scatter(
            x=predictions_df['actual_price'],
            y=predictions_df['error'],
            mode='markers',
            name='Error vs Price',
            marker=dict(
                color=predictions_df['abs_error'],
                colorscale='Viridis',
                showscale=True
            )
        ),
        row=2, col=1
    )
    
    # 4. Cumulative error
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['error'].cumsum(),
            mode='lines',
            name='Cumulative Error',
            line=dict(color='#66BB6A')
        ),
        row=2, col=2
    )
    
    # Calculate error metrics
    mae = predictions_df['abs_error'].mean()
    mse = (predictions_df['error'] ** 2).mean()
    rmse = np.sqrt(mse)
    mape = (predictions_df['error_pct'].abs()).mean()
    
    # Update layout with metrics
    fig.update_layout(
        title=dict(
            text=f'Prediction Error Analysis<br>MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | MAPE: {mape:.2f}%',
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        template='plotly_white',
        height=800,
        width=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Error ($)", row=1, col=2)
    fig.update_xaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    
    fig.update_yaxes(title_text="Error ($)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Error ($)", row=2, col=2)
    
    return fig

# Modify the run_simulation callback to include error plotting
@app.callback(
    [Output('price-chart', 'figure'),
     Output('portfolio-chart', 'figure'),
     Output('simulation-stats', 'children')],
    [Input('simulate-button', 'n_clicks'),
     Input('trade-symbol-dropdown', 'value'),
     Input('trade-granularity-dropdown', 'value'),
     Input('simulation-date-range', 'start_date'),
     Input('simulation-date-range', 'end_date')]
)
def run_simulation(n_clicks, symbol, granularity, start_date, end_date):
    if n_clicks is None or not n_clicks:
        return create_empty_figure(), create_empty_figure(), ""
    
    if not symbol:
        return create_empty_figure(), create_empty_figure(), html.Div("❌ Please select a symbol", style={'color': 'red'})
    
    if not start_date or not end_date:
        return create_empty_figure(), create_empty_figure(), html.Div("❌ Please select date range", style={'color': 'red'})
    
    try:
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Load models
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        try:
            reg_model, clf = train_model_for_symbol(symbol, granularity)
        except Exception as e:
            error_msg = html.Div([
                html.H4("❌ Error Loading Models", style={'color': 'red'}),
                html.P(f"Models not found for {symbol} at {granularity}s granularity."),
                html.P("Please train the model first using the Training tab.")
            ])
            return create_empty_figure(), create_empty_figure(), error_msg
        
        # Get data and run simulation
        try:
            print(f"Fetching data for {symbol}...")
            
            # Adjust data fetching period based on granularity
            if granularity == 60:  # 1 minute
                max_days = 7  # 1 week of minute data
            elif granularity == 300:  # 5 minutes
                max_days = 14  # 2 weeks of 5-minute data
            elif granularity == 900:  # 15 minutes
                max_days = 30  # 1 month of 15-minute data
            else:  # 1 hour or higher
                max_days = 90  # 3 months of hourly data
                
            # Calculate actual days needed based on date range and max_days
            requested_days = (end_date - start_date).days
            days_to_fetch = min(requested_days + 1, max_days)
            
            # Adjust start_date if needed
            if requested_days > max_days:
                print(f"⚠️ Warning: Requested period exceeds maximum of {max_days} days for {granularity}s granularity.")
                print(f"Fetching most recent {max_days} days of data.")
                start_date = end_date - timedelta(days=max_days-1)
            
            df = get_coinbase_data(symbol=symbol, granularity=granularity, days=days_to_fetch)
            if df.empty:
                return create_empty_figure(), create_empty_figure(), html.Div("❌ No data available", style={'color': 'red'})
            
            # Filter data for selected date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if len(df) < 50:
                return create_empty_figure(), create_empty_figure(), html.Div(
                    "❌ Insufficient data for selected date range. Please select a longer period.",
                    style={'color': 'red'}
                )
            
            print("Calculating indicators...")
            df = calculate_indicators(df)
            df.dropna(inplace=True)
            
            print("Running simulation...")
            initial_state = {
                "cash": 100.0,
                "position": None,
                "quantity": 0,
                "entry": None,
                "total_profit": 0,
                "wins": 0,
                "losses": 0
            }
            
            trades_df, final_state = simulate_trading(df, reg_model, clf, initial_state)
            
            # Get predictions for error analysis
            predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
            
            # Create error analysis plot
            error_fig = plot_prediction_errors(predictions)
            
            # Create price chart with candlesticks and trades
            price_fig = go.Figure()
            
            # Add candlestick chart
            price_fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Add predicted prices
            if not predictions.empty:
                price_fig.add_trace(go.Scatter(
                    x=predictions['timestamp'],
                    y=predictions['predicted_price'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='#42A5F5', dash='dash')
                ))
            
            # Add buy markers
            buy_points = trades_df[trades_df['action'] == 'BUY']
            if not buy_points.empty:
                price_fig.add_trace(go.Scatter(
                    x=buy_points['timestamp'],
                    y=buy_points['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Buy'
                ))
            
            # Add sell markers
            sell_points = trades_df[trades_df['action'] == 'SELL']
            if not sell_points.empty:
                price_fig.add_trace(go.Scatter(
                    x=sell_points['timestamp'],
                    y=sell_points['price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Sell'
                ))
            
            price_fig.update_layout(
                title=f"{symbol} Price and Trades",
                yaxis_title='Price',
                template='plotly',  # Changed from plotly_dark
                xaxis_rangeslider_visible=False
            )
            
            # Create portfolio value chart
            portfolio_fig = go.Figure()
            
            # Add total value line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['total_value'],
                name='Portfolio Value',
                line=dict(color='#2196F3')  # Blue color
            ))
            
            # Add cash line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['cash'],
                name='Cash',
                line=dict(color='#4CAF50', dash='dash')  # Green color
            ))
            
            # Add crypto value line
            portfolio_fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['crypto_value'],
                name='Crypto Value',
                line=dict(color='#FF9800', dash='dash')  # Orange color
            ))
            
            portfolio_fig.update_layout(
                title="Portfolio Value Over Time",
                yaxis_title='Value (USD)',
                template='plotly'  # Changed from plotly_dark
            )
            
            # Calculate additional statistics
            total_trades = final_state['wins'] + final_state['losses']
            win_rate = (final_state['wins'] / total_trades * 100) if total_trades > 0 else 0
            avg_profit = final_state['total_profit'] / total_trades if total_trades > 0 else 0
            
            # Create detailed stats
            stats = html.Div([
                html.H3("Simulation Results", style={'color': '#2196F3'}),
                html.Div([
                    html.Div([
                        html.H4("Portfolio Metrics", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Initial Value: "), 
                            html.Span(f"${100:,.2f}")
                        ]),
                        html.P([
                            html.Strong("Final Value: "), 
                            html.Span(f"${trades_df['total_value'].iloc[-1]:,.2f}")
                        ]),
                        html.P([
                            html.Strong("Total Return: "), 
                            html.Span(
                                f"{((trades_df['total_value'].iloc[-1] / 100 - 1) * 100):,.2f}%",
                                style={'color': 'green' if trades_df['total_value'].iloc[-1] > 10000 else 'red'}
                            )
                        ])
                    ], style={'flex': 1}),
                    
                    html.Div([
                        html.H4("Trading Metrics", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Total Trades: "), 
                            html.Span(f"{total_trades}")
                        ]),
                        html.P([
                            html.Strong("Win Rate: "), 
                            html.Span(f"{win_rate:.1f}%")
                        ]),
                        html.P([
                            html.Strong("Average Profit per Trade: "), 
                            html.Span(f"${avg_profit:.2f}")
                        ])
                    ], style={'flex': 1}),
                    
                    html.Div([
                        html.H4("Current Position", style={'color': '#4CAF50'}),
                        html.P([
                            html.Strong("Position: "), 
                            html.Span(final_state['position'] if final_state['position'] else 'None')
                        ]),
                        html.P([
                            html.Strong("Quantity: "), 
                            html.Span(f"{final_state['quantity']:.8f}")
                        ]),
                        html.P([
                            html.Strong("Entry Price: "), 
                            html.Span(f"${final_state['entry']:.2f}" if final_state['entry'] else 'N/A')
                        ])
                    ], style={'flex': 1})
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'padding': '20px', 'backgroundColor': '#1E1E1E', 'borderRadius': '5px'})
            
            # Save trade history
            trades_df.to_csv(os.path.join(LOGS_DIR, f"{symbol.replace('-', '')}_trades.csv"), index=False)
            
            # Update the layout to include both price and error charts
            price_fig.update_layout(
                title=f"{symbol} Price and Predictions",
                yaxis_title='Price',
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            
            # Return both figures
            return price_fig, error_fig, stats
            
        except Exception as e:
            error_msg = html.Div([
                html.H4("❌ Error During Simulation", style={'color': 'red'}),
                html.P(str(e))
            ])
            return create_empty_figure(), create_empty_figure(), error_msg
        
    except Exception as e:
        error_msg = html.Div([
            html.H4("❌ Error During Simulation", style={'color': 'red'}),
            html.P(str(e))
        ])
        return create_empty_figure(), create_empty_figure(), error_msg

def create_empty_figure():
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.update_layout(
        title="No Data Available",
        template='plotly',
        xaxis={'visible': False},
        yaxis={'visible': False},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[{
            'text': "Select a symbol and click 'Start Simulation'",
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 20, 'color': '#2c3e50'}
        }]
    )
    return fig

@app.callback(
    [Output('profit-loss-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('analysis-stats', 'children')],
    [Input('analysis-symbol-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_analysis(symbol, n_intervals):
    if not symbol:
        return create_empty_figure(), create_empty_figure(), ""
    
    # Load trade history
    try:
        trades_df = pd.read_csv(os.path.join(LOGS_DIR, f"{symbol.replace('-', '')}_trades.csv"))
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    except Exception as e:
        print(f"Error loading trade history: {e}")
        return create_empty_figure(), create_empty_figure(), "No trade history available"
    
    # Create P&L chart
    pnl_fig = go.Figure()
    pnl_fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['profit'].cumsum(),
        name='Cumulative P&L',
        line=dict(color='#4CAF50' if trades_df['profit'].sum() > 0 else '#f44336')  # Green if positive, red if negative
    ))
    pnl_fig.update_layout(
        title="Cumulative Profit/Loss",
        yaxis_title='Profit/Loss (USD)',
        template='plotly'  # Changed from plotly_dark
    )
    
    # Calculate and plot drawdown
    cummax = trades_df['total_value'].cummax()
    drawdown = (cummax - trades_df['total_value']) / cummax * 100
    
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=drawdown,
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#f44336')  # Red color
    ))
    dd_fig.update_layout(
        title="Drawdown Analysis",
        yaxis_title='Drawdown (%)',
        template='plotly'  # Changed from plotly_dark
    )
    
    # Calculate analysis statistics
    total_trades = len(trades_df[trades_df['action'].isin(['BUY', 'SELL'])])
    profitable_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    max_drawdown = drawdown.max()
    
    # Create analysis stats
    stats = html.Div([
        html.H3("Trading Analysis", style={'color': '#2196F3'}),
        html.Div([
            html.Div([
                html.H4("Performance Metrics", style={'color': '#4CAF50'}),
                html.P([
                    html.Strong("Total Trades: "), 
                    html.Span(f"{total_trades}")
                ]),
                html.P([
                    html.Strong("Win Rate: "), 
                    html.Span(f"{win_rate:.1f}%")
                ]),
                html.P([
                    html.Strong("Max Drawdown: "), 
                    html.Span(f"{max_drawdown:.1f}%")
                ])
            ], style={'flex': 1}),
            
            html.Div([
                html.H4("Risk Metrics", style={'color': '#4CAF50'}),
                html.P([
                    html.Strong("Avg Win: "), 
                    html.Span(f"${trades_df[trades_df['profit'] > 0]['profit'].mean():.2f}")
                ]),
                html.P([
                    html.Strong("Avg Loss: "), 
                    html.Span(f"${trades_df[trades_df['profit'] < 0]['profit'].mean():.2f}")
                ]),
                html.P([
                    html.Strong("Profit Factor: "), 
                    html.Span(f"{abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / trades_df[trades_df['profit'] < 0]['profit'].sum()):.2f}")
                ])
            ], style={'flex': 1})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'padding': '20px', 'backgroundColor': '#1E1E1E', 'borderRadius': '5px'})
    
    return pnl_fig, dd_fig, stats

# === New Callbacks for Context Menu and Row Selection ===
@app.callback(
    [Output('trade-symbol-dropdown', 'value'),
     Output('train-symbol-dropdown', 'value'),
     Output('analysis-symbol-dropdown', 'value')],
    [Input('context-train', 'n_clicks'),
     Input('context-simulate', 'n_clicks'),
     Input('context-analyze', 'n_clicks'),
     Input('scanner-datatable', 'selected_rows')],
    [State('selected-symbol', 'data'),
     State('scanner-datatable', 'data')],
    prevent_initial_call=True
)
def handle_symbol_selection(train_clicks, sim_clicks, analyze_clicks, selected_rows, selected_symbol, table_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle row selection from table
    if trigger_id == 'scanner-datatable':
        if not selected_rows or not table_data:
            raise PreventUpdate
        try:
            selected_row = table_data[selected_rows[0]]
            symbol = selected_row.get('symbol')
            if not symbol:
                raise PreventUpdate
            return symbol, symbol, symbol
        except Exception as e:
            print(f"Error in row selection: {str(e)}")
            raise PreventUpdate
    
    # Handle context menu clicks
    elif trigger_id in ['context-train', 'context-simulate', 'context-analyze']:
        if not selected_symbol:
            raise PreventUpdate
            
        if trigger_id == 'context-train':
            return dash.no_update, selected_symbol, dash.no_update
        elif trigger_id == 'context-simulate':
            return selected_symbol, dash.no_update, dash.no_update
        elif trigger_id == 'context-analyze':
            return dash.no_update, dash.no_update, selected_symbol
    
    raise PreventUpdate

@app.callback(
    [Output('context-menu', 'style'),
     Output('selected-symbol', 'data')],
    [Input('scanner-datatable', 'active_cell'),
     Input('scanner-datatable', 'data')],
    [State('context-menu', 'style')]
)
def show_context_menu(active_cell, data, current_style):
    if not active_cell or not data:
        style = dict(styles['context_menu'])
        style['display'] = 'none'
        return style, None
    
    try:
        row = data[active_cell['row']]
        symbol = row.get('symbol')
        if not symbol:
            style = dict(styles['context_menu'])
            style['display'] = 'none'
            return style, None
        
        style = dict(styles['context_menu'])
        style['display'] = 'block'
        style['left'] = f"{active_cell.get('column_id', 0)}px"
        style['top'] = f"{active_cell.get('row', 0)}px"
        
        return style, symbol
        
    except Exception as e:
        print(f"Error in show_context_menu: {str(e)}")
        style = dict(styles['context_menu'])
        style['display'] = 'none'
        return style, None

@app.callback(
    Output('forward-test-results', 'children'),
    [Input('run-forward-test-btn', 'n_clicks')],
    [State('forward-test-days', 'value'),
     State('forward-initial-portfolio', 'value'),
     State('forward-max-positions', 'value')]
)
def run_forward_test_callback(n_clicks, days, initial_value, max_positions):
    if not n_clicks:
        return ""
    
    # Initialize results structure
    results = {
        'summary': {
            'initial_value': initial_value,
            'final_value': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'win_rate': 0
        },
        'value_history': pd.DataFrame(),
        'history': pd.DataFrame()
    }
    
    try:
        # Get current market data for top opportunities
        symbols = get_cached_symbols()[:50]  # Get top 50 symbols
        market_data = scan_market(symbols=symbols)
        top_symbols = [item['Symbol'] for item in market_data[:max_positions]]
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_value,
            'positions': {},
            'value_history': []
        }
        
        # Simulate future dates
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        
        # Simple simulation using historical volatility and momentum
        for date in dates:
            daily_return = 0
            for symbol in top_symbols:
                # Get historical data for volatility calculation
                hist_data = get_coinbase_data(symbol=symbol, granularity=3600*24, days=30)
                if not hist_data.empty:
                    volatility = hist_data['close'].pct_change().std()
                    momentum = hist_data['close'].pct_change().mean()
                    
                    # Simulate daily return based on historical patterns
                    simulated_return = np.random.normal(momentum, volatility)
                    daily_return += simulated_return / len(top_symbols)
            
            # Update portfolio value
            portfolio_value = portfolio['cash'] * (1 + daily_return)
            portfolio['value_history'].append({
                'date': date,
                'total_value': portfolio_value
            })
            portfolio['cash'] = portfolio_value
        
        # Update results
        value_history = pd.DataFrame(portfolio['value_history'])
        returns = value_history['total_value'].pct_change().dropna()
        
        results['summary']['final_value'] = portfolio['cash']
        results['summary']['total_return'] = ((portfolio['cash'] / initial_value) - 1) * 100
        results['summary']['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 else 0
        results['summary']['max_drawdown'] = ((value_history['total_value'].cummax() - value_history['total_value']) / 
                                            value_history['total_value'].cummax()).max()
        results['summary']['total_trades'] = len(top_symbols)
        results['summary']['win_rate'] = 55.0  # Estimated win rate
        
        results['value_history'] = value_history
        
        # Get current market data for the selected symbols
        current_prices = {}
        for symbol in top_symbols[:5]:  # Get data for top 5 symbols
            try:
                current_data = get_coinbase_data(symbol=symbol, granularity=3600, days=1)  # Get latest day's data
                if not current_data.empty:
                    current_prices[symbol] = current_data['close'].iloc[-1]
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                continue
        
        # Create trade history with actual prices and current date
        current_date = pd.Timestamp.now()
        trade_records = []
        position_size = initial_value / max_positions
        
        for symbol, price in current_prices.items():
            if price > 0:  # Ensure we have a valid price
                quantity = position_size / price
                trade_records.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'value': position_size,
                    'profit': 0
                })
        
        results['history'] = pd.DataFrame(trade_records)
        
        # Add projected future trades
        future_trades = []
        for day in range(1, days + 1):
            future_date = current_date + pd.Timedelta(days=day)
            
            # For each active position, simulate potential exits
            for record in trade_records:
                symbol = record['symbol']
                entry_price = record['price']
                
                # Get historical volatility
                try:
                    hist_data = get_coinbase_data(symbol=symbol, granularity=86400, days=30)
                    if not hist_data.empty:
                        daily_returns = hist_data['close'].pct_change().dropna()
                        volatility = daily_returns.std()
                        trend = (hist_data['close'].iloc[-1] / hist_data['close'].iloc[0]) - 1
                        
                        # Project price movement based on volatility and trend
                        price_change = np.random.normal(trend/30, volatility)
                        projected_price = entry_price * (1 + price_change)
                        
                        # Simulate exit if projected return exceeds threshold
                        if abs(price_change) > 0.05:  # 5% threshold
                            action = 'SELL' if price_change < 0 else 'HOLD'
                            profit = record['quantity'] * (projected_price - entry_price)
                            
                            future_trades.append({
                                'date': future_date,
                                'symbol': symbol,
                                'action': action,
                                'price': projected_price,
                                'quantity': record['quantity'],
                                'value': record['quantity'] * projected_price,
                                'profit': profit
                            })
                except Exception as e:
                    print(f"Error projecting trades for {symbol}: {e}")
                    continue
        
        # Append projected trades to history
        if future_trades:
            future_df = pd.DataFrame(future_trades)
            results['history'] = pd.concat([results['history'], future_df])
        
        # Sort by date
        results['history'] = results['history'].sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"Error in forward test: {e}")
        return html.Div(f"Error running forward test: {str(e)}", style={'color': 'red'})
    
    # Create results display
    return create_results_display(results, is_forward_test=True)

@app.callback(
    Output('backtest-results', 'children'),
    [Input('run-portfolio-backtest-btn', 'n_clicks')],
    [State('portfolio-test-dates', 'start_date'),
     State('portfolio-test-dates', 'end_date'),
     State('initial-portfolio-value', 'value'),
     State('max-positions', 'value')]
)
def run_backtest_callback(n_clicks, start_date, end_date, initial_value, max_positions):
    if not n_clicks:
        return "Click 'Run Backtest' to start"

    try:
        # Convert string dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Run ML backtest
        results = run_ml_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_value=float(initial_value),
            max_positions=int(max_positions)
        )
        
        if 'error' in results and results['error']:
            return html.Div([
                html.H4("Backtest Error", style={'color': 'red'}),
                html.P(results['error'])
            ])
            
        # Save results to CSV files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if len(results['trades']) > 0:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(os.path.join(LOGS_DIR, f'backtest_trades_alpha_v1_{timestamp}.csv'), index=False)
            
        if not results['history'].empty:
            results['history'].to_csv(os.path.join(LOGS_DIR, f'backtest_value_history_{timestamp}.csv'), index=False)
            
        return create_results_display(results, is_forward_test=False)
        
    except Exception as e:
        import traceback
        print(f"Error running backtest: {str(e)}")
        print(traceback.format_exc())
        return html.Div([
            html.H4("Backtest Error", style={'color': 'red'}),
            html.P(str(e))
        ])

def create_results_display(results, is_forward_test=False):
    if isinstance(results, str):
        return html.Div(results)
    
    # Extract metrics from stats if available
    metrics = results.get('stats', {})
    history_df = results.get('history', pd.DataFrame())
    trades = results.get('trades', [])
    
    # Create portfolio value chart
    portfolio_chart = dcc.Graph(
        figure={
            'data': [
                go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['portfolio_value'],
                    name='Portfolio Value',
                    line={'color': '#2196F3'}
                )
            ],
            'layout': go.Layout(
                title='Portfolio Value Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Value ($)'},
                template='plotly_white',
                hovermode='x unified'
            )
        }
    )
    
    # Create the main metrics display
    metrics_div = html.Div([
        html.H4("Portfolio Performance", style={
            'color': '#2c3e50',
            'margin-bottom': '20px',
            'font-weight': 'bold'
        }),
        
        # Portfolio Value Chart
        html.Div([
            portfolio_chart
        ], style={
            'background-color': '#ffffff',
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
            'margin-bottom': '20px'
        }),
        
        # Metrics Sections
        html.Div([
            html.Div([
                # Portfolio metrics
                html.Div([
                    html.H5("Overall Metrics", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                    html.P([
                        html.Strong("Total Return: "),
                        html.Span(
                            f"{metrics.get('total_return', 0):.2f}%",
                            style={'color': '#4CAF50' if metrics.get('total_return', 0) > 0 else '#f44336'}
                        )
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Sharpe Ratio: "),
                        html.Span(f"{metrics.get('sharpe_ratio', 0):.2f}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Max Drawdown: "),
                        html.Span(f"{metrics.get('max_drawdown', 0):.2f}%")
                    ], style={'margin': '10px 0'})
                ], style={'flex': '1'}),
                
                # Trading metrics
                html.Div([
                    html.H5("Trading Statistics", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                    html.P([
                        html.Strong("Win Rate: "),
                        html.Span(f"{metrics.get('win_rate', 0):.2f}%")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Total Trades: "),
                        html.Span(f"{len(trades)}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Initial Value: "),
                        html.Span(f"${history_df['portfolio_value'].iloc[0]:,.2f}")
                    ], style={'margin': '10px 0'}),
                    html.P([
                        html.Strong("Final Value: "),
                        html.Span(f"${history_df['portfolio_value'].iloc[-1]:,.2f}")
                    ], style={'margin': '10px 0'})
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '8px',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin-bottom': '20px'
            })
        ]),
        
        # Recent Trades Table (if any trades exist)
        html.Div([
            html.H5("Recent Trades", style={'color': '#2c3e50', 'margin-bottom': '15px'}),
            html.Div(
                "No trades executed during this period." if not trades else
                dash_table.DataTable(
                    id='recent-trades-table',
                    columns=[
                        {'name': 'Time', 'id': 'timestamp'},
                        {'name': 'Symbol', 'id': 'symbol'},
                        {'name': 'Action', 'id': 'action'},
                        {'name': 'Price', 'id': 'price'},
                        {'name': 'Size', 'id': 'size'},
                        {'name': 'P/L', 'id': 'pnl'}
                    ],
                    data=trades[-10:],  # Show last 10 trades
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '10px',
                        'backgroundColor': '#ffffff'
                    },
                    style_header={
                        'backgroundColor': '#f8f9fa',
                        'fontWeight': 'bold',
                        'border': '1px solid #ddd'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'action', 'filter_query': '{action} eq "BUY"'},
                            'color': '#4CAF50'
                        },
                        {
                            'if': {'column_id': 'action', 'filter_query': '{action} eq "SELL"'},
                            'color': '#f44336'
                        },
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "+"'},
                            'color': '#4CAF50'
                        },
                        {
                            'if': {'column_id': 'pnl', 'filter_query': '{pnl} contains "-"'},
                            'color': '#f44336'
                        }
                    ]
                )
            )
        ], style={
            'background-color': '#ffffff',
            'padding': '20px',
            'border-radius': '8px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px'
    })
    
    return metrics_div

# Add live trading callbacks
@app.callback(
    [Output('live-trading-status', 'children'),
     Output('start-live-trading-btn', 'disabled'),
     Output('stop-live-trading-btn', 'disabled')],
    [Input('start-live-trading-btn', 'n_clicks'),
     Input('stop-live-trading-btn', 'n_clicks')],
    [State('live-initial-portfolio', 'value'),
     State('live-max-positions', 'value')]
)
def manage_live_trading(start_clicks, stop_clicks, initial_portfolio, max_positions):
    global live_trading_thread, is_trading_active, live_sim
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div([
            html.H4("Trading Status: Inactive", style={'color': '#888888'}),
            html.P("Click 'Start Live Trading' to begin")
        ]), False, True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Button clicked: {button_id}")
    
    if button_id == 'start-live-trading-btn' and start_clicks:
        if not live_trading_thread or not live_trading_thread.is_alive():
            is_trading_active = True
            live_sim = SimulationTracker(initial_portfolio=initial_portfolio, db_path='live_trading.db')
            live_sim.load_state()
            
            def trading_loop():
                while is_trading_active:
                    try:
                        print("\n=== Starting Trading Loop ===")
                        # Get top symbols by market cap and momentum
                        symbols = get_cached_symbols()[:50]  # Top 50 by market cap
                        market_data = scan_market(symbols=symbols)
                        
                        # Sort opportunities by momentum score
                        opportunities = sorted(market_data, key=lambda x: x['Momentum_Score'], reverse=True)
                        print(f"\nTop opportunities found: {[opp['Symbol'] for opp in opportunities[:5]]}")
                        
                        # Calculate available cash per position
                        total_positions = len(live_sim.positions)
                        available_slots = max_positions - total_positions
                        if available_slots > 0:
                            cash_per_position = live_sim.get_cash_balance() * 0.95 / available_slots  # Use 95% of available cash
                            print(f"\nAvailable slots: {available_slots}, Cash per position: ${cash_per_position:.2f}")
                        
                        # Process existing positions
                        print("\nProcessing existing positions...")
                        for symbol, position in list(live_sim.positions.items()):
                            try:
                                print(f"\nAnalyzing position: {symbol}")
                                # Get current market data
                                current_data = get_market_hours_data(symbol, period='1d', interval='1h')
                                if current_data.empty:
                                    print(f"No data available for {symbol}")
                                    continue
                                
                                # Calculate indicators
                                df = calculate_indicators(current_data)
                                if df.empty:
                                    print(f"No indicators available for {symbol}")
                                    continue
                                
                                # Get ML predictions
                                predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
                                if not predictions.empty:
                                    latest_pred = predictions.iloc[-1]
                                    latest_conf = confidence.iloc[-1]
                                    print(f"Predictions for {symbol}:")
                                    print(f"Direction: {latest_pred['direction']}")
                                    print(f"Buy confidence: {latest_conf['buy_confidence']:.3f}")
                                    print(f"Sell confidence: {latest_conf['sell_confidence']:.3f}")
                                
                                current_price = float(df['close'].iloc[-1])
                                entry_price = position['price']
                                current_return = (current_price / entry_price - 1) * 100
                                print(f"Current return: {current_return:.2f}%")
                                
                                # Exit conditions
                                should_exit = False
                                exit_reason = None
                                
                                # 1. Stop loss (-3%)
                                if current_return <= -3:
                                    should_exit = True
                                    exit_reason = "Stop Loss"
                                
                                # 2. Take profit (+5%)
                                elif current_return >= 5:
                                    should_exit = True
                                    exit_reason = "Take Profit"
                                
                                # 3. ML and Technical indicators combined
                                else:
                                    latest = df.iloc[-1]
                                    
                                    # ML-based exit signals (lowered threshold)
                                    ml_sell_signal = (
                                        not predictions.empty and
                                        latest_pred['direction'] == 'SELL' and
                                        latest_conf['sell_confidence'] > 0.6  # Lowered from 0.7
                                    )
                                    
                                    # Technical indicator signals
                                    tech_sell_signal = (
                                        latest['RSI'] > 70 or  # Overbought
                                        (latest['MACD'] < latest['Signal_Line'] and  # MACD bearish crossover
                                         latest['close'] < latest['MA20'])  # Price below MA20
                                    )
                                    
                                    # Combined decision with more lenient conditions
                                    if ml_sell_signal or (tech_sell_signal and current_return > 0):
                                        should_exit = True
                                        exit_reason = "ML and Technical Exit"
                                
                                if should_exit:
                                    print(f"Exiting {symbol} position - {exit_reason}")
                                    trading_queue.put({
                                        'action': 'SELL',
                                        'symbol': symbol,
                                        'price': current_price,
                                        'quantity': position['quantity'],
                                        'timestamp': datetime.now()
                                    })
                            
                            except Exception as e:
                                print(f"Error processing position {symbol}: {str(e)}")
                        
                        # Enter new positions
                        print("\nLooking for new positions...")
                        if available_slots > 0:
                            for opp in opportunities[:available_slots]:
                                symbol = opp['Symbol']
                                if symbol not in live_sim.positions and cash_per_position > 0:
                                    try:
                                        print(f"\nAnalyzing opportunity: {symbol}")
                                        # Get current market data
                                        current_data = get_market_hours_data(symbol, period='1d', interval='1h')
                                        if current_data.empty:
                                            continue
                                        
                                        # Calculate indicators
                                        df = calculate_indicators(current_data)
                                        if df.empty:
                                            continue
                                        
                                        # Get ML predictions
                                        predictions, confidence = predict_with_pretrained_model(df, symbol, interval='1h')
                                        if not predictions.empty:
                                            latest_pred = predictions.iloc[-1]
                                            latest_conf = confidence.iloc[-1]
                                            print(f"Predictions for {symbol}:")
                                            print(f"Direction: {latest_pred['direction']}")
                                            print(f"Buy confidence: {latest_conf['buy_confidence']:.3f}")
                                            print(f"Sell confidence: {latest_conf['sell_confidence']:.3f}")
                                        
                                        latest = df.iloc[-1]
                                        current_price = float(latest['close'])
                                        
                                        # Entry conditions with more lenient thresholds
                                        ml_buy_signal = (
                                            not predictions.empty and
                                            (
                                                latest_conf['buy_confidence'] > 0.55 or  # High confidence override regardless of direction
                                                (latest_pred['direction'] == 'BUY' and
                                                 latest_conf['buy_confidence'] > 0.6 and  # Regular threshold with technical confirmation
                                                 latest['RSI'] < 40 and  
                                                 latest['MACD'] > latest['Signal_Line'] and
                                                 opp['Momentum_Score'] > 60)
                                            )
                                        )
                                        
                                        tech_buy_signal = (
                                            latest['RSI'] < 40 and  
                                            latest['MACD'] > latest['Signal_Line'] and
                                            opp['Momentum_Score'] > 60
                                        )
                                        
                                        should_enter = ml_buy_signal or tech_buy_signal
                                        
                                        if should_enter:
                                            # Calculate position size based on volatility
                                            volatility = latest['ATR'] / current_price
                                            position_size = min(
                                                cash_per_position,
                                                live_sim.get_cash_balance() * 0.25  # Max 25% of portfolio per position
                                            )
                                            
                                            # Adjust position size based on volatility
                                            if volatility > 0.03:  # High volatility
                                                position_size *= 0.7  # Reduce position size
                                            
                                            quantity = (position_size * 0.995) / current_price  # Account for fees
                                            
                                            if quantity * current_price >= 10:  # Lowered minimum position size to $10
                                                print(f"Entering {symbol} position - ML Score: {latest_conf['buy_confidence']:.2f}")
                                                trading_queue.put({
                                                    'action': 'BUY',
                                                    'symbol': symbol,
                                                    'price': current_price,
                                                    'quantity': quantity,
                                                    'timestamp': datetime.now()
                                                })
                                        else:
                                            print(f"No entry signal for {symbol}")
                                    
                                    except Exception as e:
                                        print(f"Error processing opportunity {symbol}: {str(e)}")
                            
                            # Save state and wait
                            live_sim.save_state()
                            print("\n=== Trading Loop Complete ===")
                            time.sleep(60)  # Check every minute instead of 5 minutes
                        
                    except Exception as e:
                        print(f"Error in trading loop: {str(e)}")
                        time.sleep(60)
            
            live_trading_thread = threading.Thread(target=trading_loop)
            live_trading_thread.daemon = True
            live_trading_thread.start()
            
            return html.Div([
                html.H4("Trading Status: Active", style={'color': '#4CAF50'}),
                html.P("Live trading is running with backtesting strategy")
            ]), True, False
    
    elif button_id == 'stop-live-trading-btn' and stop_clicks:
        is_trading_active = False
        if live_trading_thread:
            live_trading_thread.join(timeout=1)
        return html.Div([
            html.H4("Trading Status: Stopped", style={'color': '#f44336'}),
            html.P("Trading has been stopped")
        ]), False, True
    
    return dash.no_update

@app.callback(
    [Output('live-performance-metrics', 'children'),
     Output('live-portfolio-chart', 'figure'),
     Output('live-positions-table', 'data'),
     Output('live-trades-table', 'data')],
    [Input('live-trading-interval', 'n_intervals')]
)
def update_live_trading_display(n_intervals):
    if not live_sim:
        return html.Div("No trading data available"), {}, [], []
    
    # Process any pending trades
    while not trading_queue.empty():
        trade = trading_queue.get()
        live_sim.execute_trade(
            trade['symbol'],
            trade['action'],
            trade['price'],
            trade['quantity']
        )
    
    # Get performance metrics
    metrics = live_sim.get_performance_metrics()
    
    # Create metrics display
    metrics_html = html.Div([
        html.Div([
            html.H4("Portfolio Overview", style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.P("Current Portfolio Value", style={'color': '#888888'}),
                    html.H3(f"${live_sim.current_portfolio:,.2f}")
                ], style={'flex': '1'}),
                html.Div([
                    html.P("Total P/L", style={'color': '#888888'}),
                    html.H3(
                        f"${metrics['total_profit']:,.2f}",
                        style={'color': '#4CAF50' if metrics['total_profit'] > 0 else '#f44336'}
                    )
                ], style={'flex': '1'}),
                html.Div([
                    html.P("Win Rate", style={'color': '#888888'}),
                    html.H3(f"{metrics['win_rate']:.1f}%")
                ], style={'flex': '1'}),
                html.Div([
                    html.P("Max Drawdown", style={'color': '#888888'}),
                    html.H3(f"{metrics['max_drawdown']:.1f}%")
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'padding': '20px', 'backgroundColor': '#1c1c1c', 'borderRadius': '5px'})
    ])
    
    # Create portfolio chart
    chart_fig = live_sim.plot_performance()
    if chart_fig:
        chart_fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    # Create positions table data
    positions_data = []
    for symbol, pos in live_sim.positions.items():
        try:
            current_data = get_market_hours_data(symbol, period='1d', interval='1h')
            if not current_data.empty:
                current_price = current_data['close'].iloc[-1]
                pnl = (current_price - pos['price']) * pos['quantity']
                pnl_pct = (current_price / pos['price'] - 1) * 100
                
                positions_data.append({
                    'symbol': symbol,
                    'quantity': f"{pos['quantity']:.6f}",
                    'entry_price': f"${pos['price']:.2f}",
                    'current_price': f"${current_price:.2f}",
                    'pnl': f"${pnl:+,.2f}",
                    'pnl_pct': f"{pnl_pct:+.1f}%"
                })
        except Exception as e:
            print(f"Error updating position data for {symbol}: {str(e)}")
    
    # Get recent trades
    conn = sqlite3.connect('live_trading.db')
    trades_df = pd.read_sql_query(
        'SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10',
        conn
    )
    conn.close()
    
    trades_data = []
    for _, trade in trades_df.iterrows():
        trades_data.append({
            'timestamp': pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': trade['symbol'],
            'action': trade['action'],
            'price': f"${trade['price']:.2f}",
            'quantity': f"{trade['quantity']:.6f}",
            'value': f"${trade['value']:,.2f}"
        })
    
    return metrics_html, chart_fig, positions_data, trades_data

# Add CSS for loading animation and context menu
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Crypto Trading Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .loader {
                border: 4px solid #f3f3f3;
                border-radius: 50%;
                border-top: 4px solid #2196F3;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            #context-menu {
                position: fixed;
                background-color: #ffffff;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                border-radius: 4px;
                padding: 5px 0;
                z-index: 1000;
                border: 1px solid #ddd;
            }
            
            #context-menu div {
                padding: 8px 20px;
                cursor: pointer;
                color: #000000;
                transition: background-color 0.2s;
            }
            
            #context-menu div:hover {
                background-color: #f5f5f5;
                color: #2196F3;
            }
            
            .dash-table-container {
                cursor: context-menu;
            }
            
            .dash-cell {
                cursor: context-menu;
                color: #000000 !important;
            }
            
            /* Update table styles */
            .dash-spreadsheet-container .dash-spreadsheet-inner td {
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            
            .dash-spreadsheet-container .dash-spreadsheet-inner th {
                color: #000000 !important;
                background-color: #f5f5f5 !important;
            }
            
            /* Style for positive/negative values */
            .positive-value {
                color: #4CAF50 !important;
                font-weight: bold;
            }
            
            .negative-value {
                color: #f44336 !important;
                font-weight: bold;
            }
            
            /* Update dropdown styles */
            .Select-control {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
            }
            
            .Select-menu-outer {
                background-color: #ffffff !important;
                border: 1px solid #ddd !important;
            }
            
            .Select-option {
                color: #000000 !important;
            }
            
            .Select-option:hover {
                background-color: #f5f5f5 !important;
                color: #2196F3 !important;
            }
            
            /* Button styles */
            button {
                background-color: #2196F3 !important;
                color: #ffffff !important;
                border: none !important;
                padding: 8px 16px !important;
                border-radius: 4px !important;
                cursor: pointer !important;
                transition: background-color 0.2s !important;
            }
            
            button:hover {
                background-color: #1976D2 !important;
            }
            
            /* Input styles */
            input {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
                padding: 6px !important;
                border-radius: 4px !important;
            }
            
            /* Tab styles */
            .dash-tab {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ddd !important;
            }
            
            .dash-tab--selected {
                background-color: #2196F3 !important;
                color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def predict_with_pretrained_model(df, symbol, interval='1h'):
    """
    Make predictions using pretrained models for a given symbol.
    Uses an ensemble approach similar to the backtesting strategy.
    """
    try:
        # First ensure all required indicators are calculated
        df = calculate_indicators(df)
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
            
        # Convert interval to seconds
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        granularity = interval_map.get(interval, 3600)
        
        # Check if models exist, if not, train them
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        reg_model_path = os.path.join(MODELS_DIR, f"{model_prefix}_reg.pkl")
        clf_model_path = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
        
        if not os.path.exists(reg_model_path) or not os.path.exists(clf_model_path):
            print(f"Training new models for {symbol}...")
            reg_models, clf = train_model_for_symbol(symbol, granularity)
            if reg_models is None or clf is None:
                print(f"Failed to train models for {symbol}")
                return pd.DataFrame(), pd.DataFrame()
        else:
            try:
                reg_models = joblib.load(reg_model_path)
                clf = joblib.load(clf_model_path)
            except Exception as e:
                print(f"Error loading models for {symbol}: {str(e)}")
                return pd.DataFrame(), pd.DataFrame()
        
        # Use the same feature set as in training
        feature_cols = [
            'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
            'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR', '%K', '%D'
        ]
        
        # Verify all features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            print(f"Missing features for {symbol}: {missing_features}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Reset index to avoid indexing issues
        df = df.reset_index(drop=True)
        X = df[feature_cols].copy()
        
        # Drop any NaN values
        X = X.dropna()
        if X.empty:
            print(f"No valid data points for {symbol} after dropping NaN values")
            return pd.DataFrame(), pd.DataFrame()
        
        # Make predictions using ensemble
        reg_predictions = []
        reg_confidences = []
        
        # Get the models from the dictionary
        if isinstance(reg_models, dict) and 'models' in reg_models:
            models_list = reg_models['models']
        else:
            models_list = [reg_models]  # If it's a single model
            
        for model in models_list:
            try:
                # Get predictions
                pred = model.predict(X)
                reg_predictions.append(pred.reshape(-1, 1))
                
                # Calculate prediction confidence based on model type
                if hasattr(model, 'predict_proba'):
                    conf = np.max(model.predict_proba(X), axis=1)
                elif hasattr(model, 'feature_importances_'):
                    conf = np.ones_like(pred) * np.mean(model.feature_importances_)
                else:
                    conf = np.ones_like(pred) * 0.5
                    
                reg_confidences.append(conf.reshape(-1, 1))
                
            except Exception as e:
                print(f"Error with model prediction for {symbol}: {str(e)}")
                continue
        
        if not reg_predictions:
            print(f"No valid predictions for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine predictions with confidence weighting
        reg_predictions = np.hstack(reg_predictions)
        reg_confidences = np.hstack(reg_confidences)
        
        # Weighted average of predictions based on confidence
        weights = reg_confidences / reg_confidences.sum(axis=1, keepdims=True)
        price_predictions = np.sum(reg_predictions * weights, axis=1)
        
        # Calculate prediction confidence
        prediction_std = np.std(reg_predictions, axis=1)
        prediction_confidence = 1 / (1 + prediction_std)
        
        # Prepare features for classification
        X_with_pred = X.copy()
        X_with_pred['predicted_close'] = price_predictions
        
        try:
            # Get classification predictions and probabilities
            direction_pred = clf.predict(X_with_pred)
            confidence_scores = clf.predict_proba(X_with_pred)
            
            # Convert numeric predictions to BUY/SELL strings
            direction_pred = np.where(direction_pred == 1, 'BUY', 'SELL')
            
            # Combine regression and classification confidence
            combined_confidence = prediction_confidence.reshape(-1, 1) * confidence_scores
            
            # Create results DataFrames with proper indexing
            predictions = pd.DataFrame({
                'timestamp': df.loc[X.index, 'timestamp'],
                'actual_price': df.loc[X.index, 'close'],
                'predicted_price': price_predictions,
                'direction': direction_pred,
                'prediction_std': prediction_std
            })
            
            confidence = pd.DataFrame({
                'timestamp': df.loc[X.index, 'timestamp'],
                'buy_confidence': combined_confidence[:, 1] if combined_confidence.shape[1] > 1 else np.zeros(len(combined_confidence)),
                'sell_confidence': combined_confidence[:, 0] if combined_confidence.shape[1] > 0 else np.zeros(len(combined_confidence)),
                'prediction_confidence': prediction_confidence
            })
            
            return predictions, confidence
            
        except Exception as e:
            print(f"Error in classification step for {symbol}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        print(f"Error making predictions for {symbol}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_crypto_scanner_table(results):
    """Create a formatted table of crypto scanner results"""
    if not results:
        return html.Div("No results found", style={'color': '#FFFFFF', 'padding': '20px'})
        
    # Create header row
    header = html.Tr([
        html.Th('Symbol', style={'padding': '10px', 'textAlign': 'left'}),
        html.Th('Price', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('Momentum', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('RSI', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('24h Change', style={'padding': '10px', 'textAlign': 'right'}),
        html.Th('Volume Change', style={'padding': '10px', 'textAlign': 'right'})
    ], style={'backgroundColor': '#1e1e1e'})
    
    # Create rows for each result
    rows = [header]
    for result in results:
        rows.append(html.Tr([
            html.Td(result['symbol'], 
                   style={'padding': '8px', 'color': '#FFFFFF'}),
            html.Td(f"${result['current_price']:.2f}", 
                   style={'padding': '8px', 'color': '#FFFFFF', 'textAlign': 'right'}),
            html.Td(f"{result['momentum_score']:.1f}", 
                   style={'padding': '8px', 'textAlign': 'right', 
                         'color': '#FFFFFF', 
                         'backgroundColor': get_score_color(result['momentum_score'])}),
            html.Td(f"{result['rsi']:.1f}", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': get_rsi_color(result['rsi'])}),
            html.Td(f"{result['price_change_pct']:.1f}%", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': '#26a69a' if result['price_change_pct'] > 0 else '#ef5350'}),
            html.Td(f"{result['volume_change_pct']:.1f}%", 
                   style={'padding': '8px', 'textAlign': 'right',
                         'color': '#26a69a' if result['volume_change_pct'] > 0 else '#ef5350'})
        ], style={'backgroundColor': '#2d2d2d'}))
    
    return html.Table(rows, style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': '#1e1e1e',
        'color': '#FFFFFF',
        'border': '1px solid #333333'
    })

if __name__ == '__main__':
    app.run_server(debug=True, port=8054) 