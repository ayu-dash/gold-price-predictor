
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
from src.data import loader
from src.features import engineering
from src.models import predictor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def fetch_and_prepare_data() -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Fetches market data, adds indicators, and retrieves sentiment."""
    logger.info("Fetching market data...")
    market_data = loader.update_local_database()

    if market_data.empty:
        logger.error("Failed to fetch market data.")
        return None, None

    # Technical Indicators
    df = engineering.add_technical_indicators(market_data)

    # Sentiment Analysis
    logger.info("Fetching news sentiment...")
    sentiment, _, sentiment_breakdown = loader.fetch_news_sentiment()
    
    # Log Sentiment Stats
    total = sum(sentiment_breakdown.values())
    if total > 0:
        bull_ratio = (sentiment_breakdown.get('positive', 0) / total) * 100
        bear_ratio = (sentiment_breakdown.get('negative', 0) / total) * 100
        logger.info(f"Market Tendency: {bull_ratio:.1f}% Bullish vs {bear_ratio:.1f}% Bearish")

    df['Sentiment'] = sentiment
    return df, sentiment


def train_pipeline(df: pd.DataFrame) -> Tuple[Any, float]:
    """Prepares data, trains the model, and returns artifacts."""
    logger.info("Starting training pipeline...")
    
    # Feature Engineering
    df_clean = df.dropna().copy()
    
    # Sanitize Infs
    import numpy as np
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    
    df_clean['Target_Return'] = df_clean['Gold_Returns'].shift(-1)
    df_train = df_clean.dropna()

    features = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'US10Y', 'Nikkei', 'DAX',
        'SMA_14', 'RSI', 'MACD', 'Sentiment'
    ]
    
    # Filter available features
    valid_features = [f for f in features if f in df_train.columns]
    X = df_train[valid_features]
    y = df_train['Target_Return']
    
    # Double check for NaNs in X
    if X.isnull().values.any():
        logger.warning(f"NaNs found in X after initial cleaning. Dropping rows...")
        inds = X.isnull().any(axis=1)
        X = X[~inds]
        y = y[~inds]
        
    if len(X) == 0:
        raise ValueError("No valid data left for training after cleaning.")

    # Train/Test Split (Time Series: No Shuffle)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training Quantile Ensemble on {len(X_train)} samples.")
    
    # Train 3 models for confidence intervals
    model_med, _, _ = predictor.train_model(X_train, y_train, quantile=0.5)
    model_low, _, _ = predictor.train_model(X_train, y_train, quantile=0.05)
    model_high, _, _ = predictor.train_model(X_train, y_train, quantile=0.95)
    
    # Persistence
    predictor.save_model(model_med, "models/gold_model_med.pkl")
    predictor.save_model(model_low, "models/gold_model_low.pkl")
    predictor.save_model(model_high, "models/gold_model_high.pkl")
    
    # Legacy support (copy med to main path)
    predictor.save_model(model_med, "models/gold_model.pkl")

    # Evaluation (on Median)
    rmse, mae, _ = predictor.evaluate_model(model_med, X_test, y_test)
    logger.info(f"Median Model Results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Save Metrics Metadata
    import json
    metrics = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_used": valid_features
    }
    
    metrics_path = "models/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Performance metrics saved to {metrics_path}")
    
    return model_med, mae


def run_prediction(
    model: Any, 
    df: pd.DataFrame, 
    mae: float
) -> Dict[str, Any]:
    """Generates next-day prediction and returns context."""
    features = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'US10Y', 'Nikkei', 'DAX',
        'SMA_14', 'RSI', 'MACD', 'Sentiment'
    ]
    valid_features = [f for f in features if f in df.columns]
    
    latest_features = df[valid_features].iloc[[-1]].copy()
    predicted_return = model.predict(latest_features)[0]

    current_price_usd = df['Gold'].iloc[-1]
    current_rate_idr = df['USD_IDR'].iloc[-1]
    predicted_price_usd = current_price_usd * (1 + predicted_return)

    # Unit Conversion
    GRAMS_PER_OZ = 31.1035
    current_price_gram = (current_price_usd * current_rate_idr) / GRAMS_PER_OZ
    predicted_price_gram = (predicted_price_usd * current_rate_idr) / GRAMS_PER_OZ
    
    rec_usd, change_pct = predictor.make_recommendation(
        current_price_usd, predicted_price_usd
    )

    return {
        "current_price_idr": current_price_gram,
        "predicted_price_idr": predicted_price_gram,
        "change_pct": change_pct,
        "recommendation": rec_usd,
        "current_usd": current_price_usd,
        "current_rate_idr": current_rate_idr,
        "latest_features": latest_features
    }


def interactive_forecast(
    model: Any, 
    latest_features: pd.DataFrame, 
    current_usd: float, 
    current_idr: float, 
    days: int,
    historical_df: Optional[pd.DataFrame] = None
) -> None:
    """Runs recursive forecast for N days."""
    logger.info(f"Generating recursive forecast for {days} days...")
    
    # Use tail of history for simulation buffer
    history_buffer = None
    if historical_df is not None:
        history_buffer = historical_df.tail(100).copy()
        
    forecasts = predictor.recursive_forecast(
        model, latest_features, current_usd, current_idr, days=days,
        historical_df=history_buffer
    )
    
    print(f"\n{'Day':<5} | {'Date':<12} | {'Price (IDR/g)':<18} | {'Change'}")
    print("-" * 55)
    
    base_price = (current_usd * current_idr) / 31.1035
    for f in forecasts:
        price_gram = f['Price_IDR'] / 31.1035
        change = ((price_gram - base_price) / base_price) * 100
        print(f"{f['Day']:<5} | {f['Date']:<12} | "
              f"Rp {price_gram:,.0f}        | {change:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Gold Price Predictor Engine")
    parser.add_argument("--days", type=int, default=0, help="Interactive forecast days")
    args = parser.parse_args()

    # Pipeline Execution
    df, _ = fetch_and_prepare_data()
    if df is None:
        return

    model, mae = train_pipeline(df)
    result = run_prediction(model, df, mae)

    # Output Report
    action_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print("\n" + "="*42)
    print(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"ACTION WINDOW: {action_date}")
    print("-"*42)
    print(f"Current:   Rp {result['current_price_idr']:,.0f}")
    print(f"Predicted: Rp {result['predicted_price_idr']:,.0f}")
    print(f"Change:    {result['change_pct']*100:+.2f}%")
    print(f"Signal:    {result['recommendation']}")
    print("="*42 + "\n")

    # Interactive Mode
    days = args.days
    if days == 0:
        # Check if running interactively (TTY)
        if sys.stdin.isatty():
            try:
                inp = input("Forecast days (Default 1): ")
                days = int(inp) if inp.strip() else 1
            except ValueError:
                days = 1
        else:
            days = 1

    if days > 1:
        interactive_forecast(
            model, 
            result['latest_features'], 
            result['current_usd'], 
            result['current_rate_idr'], 
            days,
            historical_df=df
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
