import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import get_cached_symbols, scan_market, get_market_data, calculate_indicators

# Constants
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Define feature columns used for ML models
feature_cols = [
    'EMA12', 'EMA26', 'MACD', 'Signal_Line', 'RSI', 'MA20',
    'rolling_std_10', 'lag_1', 'lag_2', 'lag_3', 'OBV', 'ATR',
    '%K', '%D'
]

def get_coinbase_data(symbol='BTC-USD', granularity=60, days=7):
    """Wrapper for get_market_data with default parameters"""
    return get_market_data(symbol, days=days, granularity=granularity)

def calculate_indicators(df):
    """Calculate technical indicators for feature generation"""
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

def train_model_for_symbol(symbol, start_date, end_date):
    """Train ML models for a specific symbol using data from before the backtest period"""
    try:
        # Get training data (30 days before backtest start)
        training_start = start_date - timedelta(days=30)
        training_data = get_coinbase_data(symbol=symbol, granularity=3600, days=30)
        
        if len(training_data) < 20:  # Require minimum amount of training data
            return None, None, f"Insufficient training data for {symbol}"
            
        # Calculate indicators
        training_data = calculate_indicators(training_data)
        training_data = training_data.dropna()
        
        if len(training_data) < 20:
            return None, None, f"Insufficient clean data for {symbol} after calculating indicators"
            
        # Prepare features
        X = training_data[feature_cols].values
        
        # Prepare targets
        y_reg = training_data['close'].shift(-1).fillna(method='ffill').values
        y_clf = (y_reg > training_data['close'].values).astype(int)
        
        # Train regression model
        from sklearn.ensemble import GradientBoostingRegressor
        reg_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        reg_model.fit(X, y_reg)
        
        # Train classifier with predicted prices as additional feature
        from sklearn.ensemble import GradientBoostingClassifier
        price_preds = reg_model.predict(X).reshape(-1, 1)
        X_clf = np.hstack([X, price_preds])
        clf_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        clf_model.fit(X_clf, y_clf)
        
        return reg_model, clf_model, None
        
    except Exception as e:
        return None, None, str(e)

def run_ml_backtest(start_date, end_date, initial_value, max_positions, symbols=None):
    """
    Run ML-based backtest using trained models with live trading conditions
    
    Parameters:
    - start_date: datetime, start of backtest period
    - end_date: datetime, end of backtest period
    - initial_value: float, initial portfolio value
    - max_positions: int, maximum number of concurrent positions
    - symbols: list, optional list of symbols to backtest (if None, will use scanner)
    
    Returns:
    - dict containing backtest results
    """
    try:
        # Initialize portfolio
        portfolio = {
            'cash': initial_value,
            'positions': {},
            'value_history': pd.DataFrame(columns=['timestamp', 'portfolio_value']),
            'trades': []
        }

        # Get symbols from scanner if not provided
        if symbols is None:
            try:
                # Get all available symbols
                all_symbols = get_cached_symbols()
                
                # Run market scanner to get top opportunities
                scan_results = scan_market(symbols=all_symbols)
                
                # Sort by momentum score and take top symbols up to max_positions
                sorted_results = sorted(scan_results, key=lambda x: x.get('momentum_score', 0), reverse=True)
                symbols = [result['symbol'] for result in sorted_results[:max_positions]]
                
                print(f"Selected top {len(symbols)} symbols from scanner based on momentum score:")
                for result in sorted_results[:max_positions]:
                    print(f"- {result['symbol']}: Momentum Score = {result.get('momentum_score', 0):.2f}")
                    
            except Exception as e:
                print(f"Error getting symbols from scanner: {e}")
                print("Falling back to default symbols...")
                symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD'][:max_positions]

        # Train or load ML models for each symbol
        models = {}
        available_symbols = []
        training_errors = []
        
        print("Training/loading models for selected symbols...")
        for symbol in symbols:
            try:
                model_prefix = f"{symbol.replace('-', '')}_{3600}"
                model_path_reg = os.path.join(MODELS_DIR, f"{model_prefix}_reg.pkl")
                model_path_clf = os.path.join(MODELS_DIR, f"{model_prefix}_clf.pkl")
                
                # Try to load existing models first
                try:
                    reg_model = joblib.load(model_path_reg)
                    clf_model = joblib.load(model_path_clf)
                    print(f"✅ Loaded existing models for {symbol}")
                except:
                    # Train new models if loading fails
                    print(f"Training new models for {symbol}...")
                    reg_model, clf_model, error = train_model_for_symbol(symbol, start_date, end_date)
                    
                    if error:
                        training_errors.append(f"{symbol}: {error}")
                        continue
                        
                    # Save newly trained models
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    joblib.dump(reg_model, model_path_reg)
                    joblib.dump(clf_model, model_path_clf)
                    print(f"✅ Trained and saved new models for {symbol}")
                
                models[symbol] = {'reg': reg_model, 'clf': clf_model}
                available_symbols.append(symbol)
                
            except Exception as e:
                training_errors.append(f"{symbol}: {str(e)}")
                print(f"❌ Error processing {symbol}: {e}")
                continue

        if not available_symbols:
            error_msg = "No models available. Training errors:\n" + "\n".join(training_errors)
            return {
                'error': error_msg,
                'history': pd.DataFrame({'timestamp': [start_date], 'portfolio_value': [initial_value]}),
                'trades': [],
                'stats': {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0
                }
            }

        print(f"\nProceeding with backtest using {len(available_symbols)} symbols: {', '.join(available_symbols)}")
        if training_errors:
            print("\nTraining errors occurred for:")
            for error in training_errors:
                print(f"- {error}")

        # Create timeline for backtest
        timeline = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Store historical data for each symbol
        symbol_data = {}
        
        # Process the backtest hour by hour
        for current_time in timeline:
            total_value = portfolio['cash']
            
            # Update data and get predictions for each symbol
            for symbol in available_symbols:
                try:
                    # Get or update historical data
                    if symbol not in symbol_data:
                        hist_data = get_coinbase_data(symbol=symbol, granularity=3600, days=30)
                        if hist_data.empty:
                            continue
                        symbol_data[symbol] = hist_data
                    
                    # Get current data point
                    current_data = symbol_data[symbol]
                    current_data = current_data[current_data['timestamp'] <= current_time]
                    
                    if len(current_data) < 50:  # Need enough data for indicators
                        continue
                    
                    # Calculate indicators
                    current_data = calculate_indicators(current_data)
                    current_row = current_data.iloc[-1]
                    
                    # Calculate position value if symbol is in portfolio
                    if symbol in portfolio['positions']:
                        position = portfolio['positions'][symbol]
                        position_value = position['quantity'] * current_row['close']
                        total_value += position_value
                        
                        # Check exit conditions (matching live trading)
                        current_price = current_row['close']
                        entry_price = position['entry_price']
                        current_return = (current_price / entry_price - 1) * 100
                        
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
                            # Get ML predictions
                            features = current_row[feature_cols].values.reshape(1, -1)
                            price_pred = models[symbol]['reg'].predict(features)[0]
                            features_with_pred = np.append(features, price_pred).reshape(1, -1)
                            action_probs = models[symbol]['clf'].predict_proba(features_with_pred)[0]
                            sell_confidence = action_probs[0]  # Assuming 0 is SELL class
                            
                            # ML-based exit signals
                            ml_sell_signal = (
                                sell_confidence > 0.6 and  # Matching live trading threshold
                                models[symbol]['clf'].predict(features_with_pred)[0] == 0  # 0 for SELL
                            )
                            
                            # Technical indicator signals
                            tech_sell_signal = (
                                current_row['RSI'] > 70 or  # Overbought
                                (current_row['MACD'] < current_row['Signal_Line'] and  # MACD bearish crossover
                                 current_row['close'] < current_row['MA20'])  # Price below MA20
                            )
                            
                            if ml_sell_signal or (tech_sell_signal and current_return > 0):
                                should_exit = True
                                exit_reason = "ML and Technical Exit"
                        
                        if should_exit:
                            # Close position
                            sell_price = current_price * 0.999  # Include fee
                            profit = (sell_price - entry_price) * position['quantity']
                            portfolio['cash'] += position['quantity'] * sell_price
                            
                            # Record trade
                            portfolio['trades'].append({
                                'timestamp': current_time,
                                'symbol': symbol,
                                'action': 'SELL',
                                'price': sell_price,
                                'quantity': position['quantity'],
                                'profit': profit,
                                'exit_reason': exit_reason
                            })
                            del portfolio['positions'][symbol]
                            continue
                    
                    # Entry conditions (matching live trading)
                    if symbol not in portfolio['positions']:
                        # Get ML predictions
                        features = current_row[feature_cols].values.reshape(1, -1)
                        price_pred = models[symbol]['reg'].predict(features)[0]
                        features_with_pred = np.append(features, price_pred).reshape(1, -1)
                        action_probs = models[symbol]['clf'].predict_proba(features_with_pred)[0]
                        buy_confidence = action_probs[1]  # Assuming 1 is BUY class
                        
                        # ML buy signal
                        ml_buy_signal = (
                            buy_confidence > 0.6 and  # Matching live trading threshold
                            models[symbol]['clf'].predict(features_with_pred)[0] == 1  # 1 for BUY
                        )
                        
                        # Technical buy signal
                        tech_buy_signal = (
                            current_row['RSI'] < 40 and  # Changed from 30 to match live trading
                            current_row['MACD'] > current_row['Signal_Line'] and  # MACD bullish crossover
                            current_row.get('momentum_score', 0) > 60  # Changed from 70 to match live trading
                        )
                        
                        should_enter = ml_buy_signal or tech_buy_signal
                        
                        if should_enter:
                            # Calculate position size based on volatility (matching live trading)
                            available_slots = max_positions - len(portfolio['positions'])
                            if available_slots > 0:
                                cash_per_position = portfolio['cash'] * 0.95 / available_slots
                                
                                # Volatility-based position sizing
                                volatility = current_row['ATR'] / current_row['close']
                                position_size = min(
                                    cash_per_position,
                                    portfolio['cash'] * 0.25  # Max 25% of portfolio per position
                                )
                                
                                # Adjust for high volatility
                                if volatility > 0.03:
                                    position_size *= 0.7
                                
                                buy_price = current_row['close'] * 1.001  # Include fee
                                quantity = (position_size * 0.995) / buy_price  # Account for fees
                                
                                # Check minimum position size ($10)
                                if quantity * buy_price >= 10:
                                    portfolio['positions'][symbol] = {
                                        'quantity': quantity,
                                        'entry_price': buy_price
                                    }
                                    portfolio['cash'] -= quantity * buy_price
                                    
                                    # Record trade
                                    portfolio['trades'].append({
                                        'timestamp': current_time,
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'price': buy_price,
                                        'quantity': quantity,
                                        'value': quantity * buy_price,
                                        'profit': 0,
                                        'buy_confidence': buy_confidence
                                    })
                
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            # Record portfolio value history
            portfolio['value_history'] = pd.concat([
                portfolio['value_history'],
                pd.DataFrame({
                    'timestamp': [current_time],
                    'portfolio_value': [total_value]
                })
            ], ignore_index=True)

        # Calculate performance metrics
        history_df = portfolio['value_history']
        returns = history_df['portfolio_value'].pct_change().dropna()
        
        total_return = (history_df['portfolio_value'].iloc[-1] / initial_value - 1) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() != 0 else 0
        
        # Calculate max drawdown
        peak = history_df['portfolio_value'].expanding(min_periods=1).max()
        drawdown = (history_df['portfolio_value'] - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
        
        # Calculate win rate
        if len(portfolio['trades']) > 0:
            profitable_trades = sum(1 for trade in portfolio['trades'] if trade.get('profit', 0) > 0)
            win_rate = (profitable_trades / len(portfolio['trades'])) * 100
        else:
            win_rate = 0

        return {
            'history': portfolio['value_history'],
            'trades': portfolio['trades'],
            'stats': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
        }

    except Exception as e:
        import traceback
        print(f"Error in backtest: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'history': pd.DataFrame({'timestamp': [start_date], 'portfolio_value': [initial_value]}),
            'trades': [],
            'stats': {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        }

if __name__ == '__main__':
    # Example usage
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    initial_value = 100000
    max_positions = 5
    
    results = run_ml_backtest(start_date, end_date, initial_value, max_positions)
    
    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print("\nBacktest Results:")
        print(f"Initial Value: ${initial_value:,.2f}")
        print(f"Final Value: ${results['history']['portfolio_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['stats']['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['stats']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['stats']['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['stats']['win_rate']:.2f}%")
        