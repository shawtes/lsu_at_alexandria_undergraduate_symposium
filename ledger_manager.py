from ledger import Ledger

class LedgerManager:
    def __init__(self):
        """Initialize ledger manager with separate ledgers for each tab."""
        self.ledgers = {
            'scanner': Ledger('scanner'),
            'backtest': Ledger('backtest'),
            'forward_test': Ledger('forward_test'),
            'live': Ledger('live')
        }
    
    def get_ledger(self, tab_name):
        """Get ledger for a specific tab."""
        return self.ledgers.get(tab_name)
    
    def reset_ledger(self, tab_name, initial_balance=0):
        """Reset ledger for a specific tab."""
        if tab_name in self.ledgers:
            self.ledgers[tab_name].reset(initial_balance)
    
    def reset_all_ledgers(self, initial_balance=0):
        """Reset all ledgers."""
        for ledger in self.ledgers.values():
            ledger.reset(initial_balance)
    
    def add_transaction(self, tab_name, symbol, action, price, quantity, fees=0, note=""):
        """Add transaction to a specific tab's ledger."""
        if tab_name in self.ledgers:
            self.ledgers[tab_name].add_transaction(
                symbol=symbol,
                action=action,
                price=price,
                quantity=quantity,
                fees=fees,
                note=note
            )
    
    def get_performance_metrics(self, tab_name):
        """Get performance metrics for a specific tab."""
        if tab_name in self.ledgers:
            return self.ledgers[tab_name].get_performance_metrics()
        return None
    
    def get_positions(self, tab_name):
        """Get current positions for a specific tab."""
        if tab_name in self.ledgers:
            return self.ledgers[tab_name].get_all_positions()
        return {}
    
    def get_transaction_history(self, tab_name, start_date=None, end_date=None):
        """Get transaction history for a specific tab."""
        if tab_name in self.ledgers:
            return self.ledgers[tab_name].get_transaction_history(start_date, end_date)
        return None
    
    def get_daily_pnl(self, tab_name):
        """Get daily PnL for a specific tab."""
        if tab_name in self.ledgers:
            return self.ledgers[tab_name].get_daily_pnl()
        return None
    
    def get_symbol_pnl(self, tab_name):
        """Get symbol-wise PnL for a specific tab."""
        if tab_name in self.ledgers:
            return self.ledgers[tab_name].get_symbol_pnl()
        return None 