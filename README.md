# lsu_at_alexandria_undergraduate_symposium
Professor-sponsored undergraduate research project presented at the LSU Alexandria Research Symposium (1st Place) — mathematics + computer science methods with reproducible experiments and analysis.
# Undergraduate Research Symposium Project — ML-Driven Crypto Trading Dashboard (Coinbase)

A research prototype that combines **market scanning**, **technical indicator feature engineering**, and an **ML-based portfolio backtest** inside an interactive **Dash** dashboard.

This project was built as part of my undergraduate research work and demonstrates an end-to-end workflow:
data ingestion → feature engineering → model/backtest logic → portfolio + trade logging → dashboard visualization.

> **Note:** This is a research/prototype system intended for experimentation and learning. It is not financial advice.

---

## What’s Included

### 1) Interactive Dash Dashboard
- Dash web app that visualizes price action, indicators, and strategy outputs.  
- Includes UI components (tables, charts, and callbacks) for running analysis/backtests.  
- Main entry: `apr_18th_final_draft_crypto.py`.  
  - It imports utility functions and the ML backtest runner.  
  - It also defines symbol caching and Coinbase candle fetching logic.  
  :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

### 2) Coinbase Market Data (OHLCV)
- Pulls historical candles from Coinbase Exchange API endpoints.
- Utilities support fetching USD-quoted pairs and time-window candle retrieval.  
  :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

### 3) Feature Engineering (Indicators)
Calculates common technical features used for both scanning and modeling:
- EMA12 / EMA26, MACD + signal, MA20  
- RSI, stochastic (%K/%D)  
- Rolling volatility (std), ATR  
- OBV + lag features  
  :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

### 4) ML + Portfolio Backtest
- Portfolio-style ML backtest that tracks:
  - portfolio value history
  - trades
  - Sharpe ratio, max drawdown, total return, win rate  
- Script exposes `run_ml_backtest(...)` and includes example usage in `__main__`.  
  :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

### 5) Trade Ledger + Performance Tracking
- A `Ledger` class for saving trades to CSV and storing metadata (PnL, balance, positions, win/loss counts).  
- A `LedgerManager` that keeps separate ledgers per dashboard tab (scanner/backtest/forward/live).  
  :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}

---

## Repository Structure (Suggested)

> Rename files if you want a cleaner repo layout, but this matches what’s here now.

- `apr_18th_final_draft_crypto.py` — Dash dashboard (main app entry)
- `utils.py` — symbol discovery, Coinbase OHLCV fetch, indicators, scanning
- `ml_backtest.py` — ML backtest runner + performance metrics
- `ledger.py` — persistent CSV ledger + positions + PnL logic
- `ledger_manager.py` — multiple ledgers (scanner/backtest/forward/live)

---

## Setup

### Requirements
You’ll need Python 3.9+ (recommended) and common ML/data packages:
- `dash`, `plotly`
- `pandas`, `numpy`, `scikit-learn`
- `xgboost`
- `scipy`, `statsmodels`
- `requests`, `joblib`

> Easiest path: create a `requirements.txt` from your environment once it runs.

### Install
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
