"""
Centralized application configuration.

Provides all paths and constants used throughout the application,
eliminating hardcoded values in individual modules.
"""

import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
CSV_PATH = os.path.join(BASE_DIR, "data", "gold_history.csv")
SIGNAL_LOG_PATH = os.path.join(BASE_DIR, "data", "signals.csv")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Model directory and paths
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
MODEL_MED_PATH = os.path.join(MODELS_DIR, "gold_model_med.pkl")
MODEL_LOW_PATH = os.path.join(MODELS_DIR, "gold_model_low.pkl")
MODEL_HIGH_PATH = os.path.join(MODELS_DIR, "gold_model_high.pkl")
MODEL_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "gold_classifier.pkl")
MODEL_NN_PATH = os.path.join(MODELS_DIR, "gold_model_nn.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
SENTIMENT_CACHE_PATH = os.path.join(MODELS_DIR, "last_sentiment.pkl")

# Constants
GRAMS_PER_OZ = 31.1035
UPDATE_INTERVAL_SECONDS = 43200  # 12 hours
HOLD_THRESHOLD = 0.005  # 0.5% threshold for hold signal

# Flask Server
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000
DEBUG = False
# AI Model Features
MODEL_FEATURES = [
    'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
    'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width', 
    'Stoch', 'WilliamsR', 'CCI', 'ATR', 'Return_Lag1', 
    'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5',
    'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
]
