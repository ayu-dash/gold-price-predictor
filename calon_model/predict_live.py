import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_PATH = os.path.join(SCRIPT_DIR, "candidate_model.pkl")
REG_PATH = os.path.join(SCRIPT_DIR, "candidate_regressor.pkl")
METRICS_PATH = os.path.join(SCRIPT_DIR, "candidate_metrics.json")

from core.data.loader import fetch_market_data

def get_live_data():
    """Fetch the most recent live data via the professional loader."""
    logger.info("📡 Fetching LIVE market data via Professional Loader...")
    
    # We need enough history for indicators (at least 100 days)
    df = fetch_market_data(period="1y")
    
    if df.empty or 'Gold' not in df.columns:
        logger.error("Failed to fetch live data via loader.")
        sys.exit(1)
            
    return df

def engineer_live_features(df):
    """Apply exact same engineering as training for the LATEST row."""
    # Returns
    df['Returns'] = df['Gold'].pct_change()
    
    # Trend
    df['SMA_7'] = ta.trend.sma_indicator(df['Gold'], window=7)
    df['SMA_14'] = ta.trend.sma_indicator(df['Gold'], window=14)
    
    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Gold'], window=14)
    df['RSI_7'] = ta.momentum.rsi(df['Gold'], window=7)
    df['ROC_10'] = ta.momentum.roc(df['Gold'], window=10)
    df['Stoch'] = ta.momentum.stoch(df['Gold'], df['Gold'], df['Gold'], window=14)
    df['WilliamsR'] = ta.momentum.williams_r(df['Gold'], df['Gold'], df['Gold'], lbp=14)
    df['CCI'] = ta.trend.cci(df['Gold'], df['Gold'], df['Gold'], window=20)
    
    # Volatility
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Gold'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['Gold'], df['Gold'], df['Gold'], window=14)
    
    # Custom Lags
    for lag in [1, 2, 3]:
        df[f'Return_Lag{lag}'] = df['Returns'].shift(lag)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    
    # Rolling
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Momentum_5'] = df['Gold'] / df['Gold'].shift(5) - 1
    
    # Inter-market
    df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']
    df['VIX_Return'] = df['VIX'].pct_change()
    df['VIX_Lag1'] = df['VIX_Return'].shift(1)
    df['US10Y_Diff'] = df['US10Y'].diff()
    df['US10Y_Lag1'] = df['US10Y_Diff'].shift(1)
    df['DXY_Ret_Lag1'] = df['DXY'].pct_change().shift(1)
    df['SP500_Ret_Lag1'] = df['SP500'].pct_change().shift(1)
    
    return df.dropna().tail(1)

def run_prediction():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(REG_PATH):
        logger.error("Sandbox models missing. Please run train_candidate.py first.")
        return

    # Load Models
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(REG_PATH, 'rb') as f:
        reg = pickle.load(f)
        
    # Load Threshold
    threshold = 0.67
    if os.path.exists(METRICS_PATH):
        import json
        with open(METRICS_PATH, 'r') as f:
            threshold = json.load(f).get('threshold', 0.67)

    # Get Data
    raw_df = get_live_data()
    latest_feat = engineer_live_features(raw_df)
    
    if latest_feat.empty:
        logger.error("Failed to engineer features for latest data.")
        return

    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width',
        'Stoch', 'WilliamsR', 'CCI', 'ATR',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'RSI_Lag1',
        'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]
    
    X = latest_feat[features]
    prob = model.predict_proba(X)[0, 1]
    price_pred = reg.predict(X)[0]
    
    current_price = latest_feat['Gold'].iloc[0]
    date_now = latest_feat.index[0].strftime('%Y-%m-%d')
    
    print("\n" + "="*40)
    print(f" LIVE PREDICTION FOR NEXT TRADING DAY")
    print("="*40)
    print(f"Current Date   : {date_now}")
    print(f"Current Price  : ${current_price:.2f}")
    print(f"Est. Next Price: ${price_pred:.2f}")
    print("-"*40)
    
    confidence = prob if prob >= 0.5 else 1 - prob
    direction = "UP 📈" if prob >= 0.5 else "DOWN 📉"
    
    print(f"Predicted Dir  : {direction}")
    print(f"Confidence     : {confidence:.2%}")
    print(f"Threshold      : {threshold:.2f}")
    
    status = "🎯 SNIPER SIGNAL" if confidence >= threshold else "⏳ NO SIGNAL (Wait for better setup)"
    print(f"Action Status  : {status}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_prediction()
