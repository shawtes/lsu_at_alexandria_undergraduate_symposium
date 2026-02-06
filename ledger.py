import pandas as pd
import os
from datetime import datetime
import json

class Ledger:
    def __init__(self, tab_name, save_dir='ledgers'):
        """
        Initialize a ledger for a specific tab.
        
        Args:
            tab_name (str): Name of the tab (e.g., 'scanner', 'backtest', 'live')
            save_dir (str): Directory to save ledger files
        """
        self.tab_name = tab_name
        self.save_dir = save_dir
        self.filename = f"{tab_name}_ledger.csv"
        self.filepath = os.path.join(save_dir, self.filename)
        self.metadata_file = os.path.join(save_dir, f"{tab_name}_metadata.json")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize or load existing ledger
        if os.path.exists(self.filepath):
            self.transactions = pd.read_csv(self.filepath)
            self.transactions['timestamp'] = pd.to_datetime(self.transactions['timestamp'])
        else:
            self.transactions = pd.DataFrame(columns=[
                'timestamp', 'symbol', 'action', 'price', 'quantity', 
                'value', 'fees', 'pnl', 'balance', 'note'
            ])
        
        # Load or initialize metadata
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from file or initialize with defaults."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'initial_balance': 0,
                'current_balance': 0,
                'total_pnl': 0,
                'win_count': 0,
                'loss_count': 0,
                'positions': {}
            }

    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def add_transaction(self, symbol, action, price, quantity, fees=0, note=""):
        """
        Add a new transaction to the ledger.
        
        Args:
            symbol (str): Trading symbol
            action (str): 'BUY' or 'SELL'
            price (float): Transaction price
            quantity (float): Transaction quantity
            fees (float): Transaction fees
            note (str): Additional notes
        """
        value = price * quantity
        pnl = 0
        
        if action == 'BUY':
            self.metadata['current_balance'] -= (value + fees)
            if symbol not in self.metadata['positions']:
                self.metadata['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': price
                }
            else:
                # Update average price
                current_pos = self.metadata['positions'][symbol]
                total_quantity = current_pos['quantity'] + quantity
                total_value = (current_pos['quantity'] * current_pos['avg_price']) + (quantity * price)
                current_pos['avg_price'] = total_value / total_quantity
                current_pos['quantity'] = total_quantity
        
        elif action == 'SELL':
            if symbol in self.metadata['positions']:
                entry_price = self.metadata['positions'][symbol]['avg_price']
                pnl = (price - entry_price) * quantity
                self.metadata['total_pnl'] += pnl
                self.metadata['current_balance'] += (value - fees)
                
                # Update win/loss count
                if pnl > 0:
                    self.metadata['win_count'] += 1
                else:
                    self.metadata['loss_count'] += 1
                
                # Update position
                current_pos = self.metadata['positions'][symbol]
                current_pos['quantity'] -= quantity
                if current_pos['quantity'] <= 0:
                    del self.metadata['positions'][symbol]

        # Add transaction to DataFrame
        new_transaction = pd.DataFrame([{
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'value': value,
            'fees': fees,
            'pnl': pnl,
            'balance': self.metadata['current_balance'],
            'note': note
        }])
        
        self.transactions = pd.concat([self.transactions, new_transaction], ignore_index=True)
        self.save()

    def get_position(self, symbol):
        """Get current position for a symbol."""
        return self.metadata['positions'].get(symbol)

    def get_all_positions(self):
        """Get all current positions."""
        return self.metadata['positions']

    def get_performance_metrics(self):
        """Get performance metrics for the ledger."""
        total_trades = self.metadata['win_count'] + self.metadata['loss_count']
        win_rate = (self.metadata['win_count'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_pnl': self.metadata['total_pnl'],
            'current_balance': self.metadata['current_balance'],
            'win_rate': win_rate,
            'total_trades': total_trades,
            'win_count': self.metadata['win_count'],
            'loss_count': self.metadata['loss_count']
        }

    def save(self):
        """Save the ledger and metadata to files."""
        self.transactions.to_csv(self.filepath, index=False)
        self.save_metadata()

    def reset(self, initial_balance=0):
        """Reset the ledger with a new initial balance."""
        self.transactions = pd.DataFrame(columns=[
            'timestamp', 'symbol', 'action', 'price', 'quantity', 
            'value', 'fees', 'pnl', 'balance', 'note'
        ])
        
        self.metadata = {
            'initial_balance': initial_balance,
            'current_balance': initial_balance,
            'total_pnl': 0,
            'win_count': 0,
            'loss_count': 0,
            'positions': {}
        }
        
        self.save()

    def get_transaction_history(self, start_date=None, end_date=None):
        """
        Get transaction history within a date range.
        
        Args:
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            
        Returns:
            pd.DataFrame: Filtered transaction history
        """
        if start_date is None and end_date is None:
            return self.transactions
            
        mask = pd.Series(True, index=self.transactions.index)
        if start_date:
            mask &= self.transactions['timestamp'] >= pd.to_datetime(start_date)
        if end_date:
            mask &= self.transactions['timestamp'] <= pd.to_datetime(end_date)
            
        return self.transactions[mask]

    def get_daily_pnl(self):
        """Get daily PnL summary."""
        return self.transactions.groupby(self.transactions['timestamp'].dt.date)['pnl'].sum()

    def get_symbol_pnl(self):
        """Get PnL summary by symbol."""
        return self.transactions.groupby('symbol')['pnl'].sum() 