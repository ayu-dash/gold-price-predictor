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
UPDATE_INTERVAL_SECONDS = 3600  # 1 hour
HOLD_THRESHOLD = 0.005  # 0.5% threshold for hold signal

# Flask Server
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000
