"""
Model training and prediction CLI.

Handles data fetching, model training, and prediction generation.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from core.data import loader, signal_logger
from core.features import engineering
from core.prediction import predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def fetch_and_prepare_data() -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Fetch market data, add indicators, and retrieve sentiment.

    Returns:
        Tuple of (processed dataframe, sentiment score).
    """
    logger.info("Fetching market data...")
    market_data = loader.update_local_database()

    if market_data.empty:
        logger.error("Failed to fetch market data.")
        return None, None

    df = engineering.add_technical_indicators(market_data)

    logger.info("Fetching news sentiment...")
    sentiment, _, sentiment_breakdown = loader.fetch_news_sentiment()

    total = sum(sentiment_breakdown.values())
    if total > 0:
        bull_ratio = (sentiment_breakdown.get('positive', 0) / total) * 100
        bear_ratio = (sentiment_breakdown.get('negative', 0) / total) * 100
        logger.info(f"Market Tendency: {bull_ratio:.1f}% Bullish vs {bear_ratio:.1f}% Bearish")

    df['Sentiment'] = sentiment
    return df, sentiment


def train_pipeline(df: pd.DataFrame) -> Tuple[Any, float]:
    """
    Prepare data, train models, and return artifacts.

    Args:
        df: Prepared dataframe with features.

    Returns:
        Tuple of (trained model, MAE score).
    """
    logger.info("Starting training pipeline...")

    df_clean = df.dropna().copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)

    df_clean['Target_Return'] = df_clean['Gold_Returns'].shift(-1)
    df_train = df_clean.dropna()

    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width', 
        'Stoch', 'WilliamsR', 'CCI', 'ATR', 'Return_Lag1', 
        'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]

    valid_features = [f for f in features if f in df_train.columns]
    print(f"DEBUG: Training features ({len(valid_features)}): {valid_features}")
    X = df_train[valid_features]
    y = df_train['Target_Return']
    y_class = (y > 0).astype(int)  # 1 for Up, 0 for Down

    if X.isnull().values.any():
        logger.warning("NaNs found after cleaning. Dropping affected rows...")
        mask = X.isnull().any(axis=1)
        X = X[~mask]
        y = y[~mask]

    if len(X) == 0:
        raise ValueError("No valid data left for training after cleaning.")

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train quantile regression ensemble
    print("\n--- Training Model Ensemble (Low/Med/High) ---")

    med_model, X_test_out, y_test_out = predictor.train_model(X, y, quantile=0.5)
    predictor.save_model(med_model, config.MODEL_MED_PATH)

    low_model, _, _ = predictor.train_model(X, y, quantile=0.05)
    high_model, _, _ = predictor.train_model(X, y, quantile=0.95)

    predictor.save_model(low_model, config.MODEL_LOW_PATH)
    predictor.save_model(high_model, config.MODEL_HIGH_PATH)

    # --- CLASSIFIER TRAINING (Deep Sniper Sniper Logic) ---
    # Filter for Significant Moves (>0.15%) to reduce noise
    df_clf = df_train[abs(df_train['Target_Return']) > 0.0015].copy()
    X_clf = df_clf[valid_features]
    y_clf = (df_clf['Target_Return'] > 0).astype(int)
    
    print(f"\n--- Training Direction Classifier (Sniper Filter: {len(X_clf)} samples) ---")
    clf_model, clf_acc, clf_prec, clf_rec, clf_f1 = predictor.train_classifier(X_clf, y_clf)
    predictor.save_model(clf_model, config.MODEL_CLASSIFIER_PATH)
    print(f"Classifier Metrics -> Acc: {clf_acc:.2%}, Prec: {clf_prec:.2%}, Rec: {clf_rec:.2%}, F1: {clf_f1:.2%}")

    # Train neural network (Experimental)
    print("\n--- Training Neural Network (MLP) ---")
    nn_model, nn_rmse, nn_mae = predictor.train_neural_network(X, y)
    predictor.save_model(nn_model, config.MODEL_NN_PATH)
    print(f"Neural Network MAE: {nn_mae:.4f}")

    # Evaluate models
    rmse_med, mae_med, _ = predictor.evaluate_model(med_model, X_test_out, y_test_out)
    rmse_low, mae_low, _ = predictor.evaluate_model(low_model, X_test_out, y_test_out)
    rmse_high, mae_high, _ = predictor.evaluate_model(high_model, X_test_out, y_test_out)

    logger.info(f"Median Model -> MAE: {mae_med:.4f}")

    # Save metrics
    metrics = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "models": {
            "median": {"mae": round(float(mae_med), 6), "rmse": round(float(rmse_med), 6)},
            "low": {"mae": round(float(mae_low), 6), "rmse": round(float(rmse_low), 6)},
            "high": {"mae": round(float(mae_high), 6), "rmse": round(float(rmse_high), 6)},
            "classifier": {
                "accuracy": round(clf_acc, 4),
                "precision": round(clf_prec, 4),
                "recall": round(clf_rec, 4),
                "f1": round(clf_f1, 4)
            },
            "neural_network": {"mae": round(float(nn_mae), 6), "rmse": round(float(nn_rmse), 6)}
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test_out),
        "features_used": valid_features
    }

    temp_metrics = config.METRICS_PATH + ".tmp"
    try:
        with open(temp_metrics, 'w') as f:
            json.dump(metrics, f, indent=4)
        os.replace(temp_metrics, config.METRICS_PATH)
        logger.info(f"Metrics saved to {config.METRICS_PATH}")
    except Exception as e:
        if os.path.exists(temp_metrics):
            os.remove(temp_metrics)
        logger.error(f"Failed to save metrics: {e}")

    return med_model, mae_med


def run_prediction(model: Any, df: pd.DataFrame, mae: float) -> Dict[str, Any]:
    """
    Generate next-day prediction and return context.

    Args:
        model: Trained model.
        df: Feature dataframe.
        mae: Model MAE score.

    Returns:
        Dictionary with prediction results.
    """
    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width', 
        'Stoch', 'WilliamsR', 'CCI', 'ATR', 'Return_Lag1', 
        'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]
    # Standardize feature vector (Force 27 features, fill missing with 0.0)
    latest_features = df.reindex(columns=features).iloc[[-1]].copy().fillna(0.0)
    predicted_return = model.predict(latest_features)[0]

    current_price_usd = df['Gold'].iloc[-1]
    current_rate_idr = df['USD_IDR'].iloc[-1]
    predicted_price_usd = current_price_usd * (1 + predicted_return)

    current_price_gram = (current_price_usd * current_rate_idr) / config.GRAMS_PER_OZ
    predicted_price_gram = (predicted_price_usd * current_rate_idr) / config.GRAMS_PER_OZ

    rec_usd, change_pct = predictor.make_recommendation(
        current_price_usd, predicted_price_usd
    )

    conf_direction = "N/A"
    conf_score = 0.0
    if isinstance(model, dict) and 'clf' in model:
        conf_direction, conf_score = predictor.get_classification_confidence(
            model['clf'], latest_features
        )
        conf_score = round(conf_score * 100, 1)
        rec_usd, change_pct = predictor.make_recommendation(
            current_price_usd, predicted_price_usd,
            conf_direction=conf_direction,
            conf_score=conf_score
        )

    return {
        "current_price_idr": current_price_gram,
        "predicted_price_idr": predicted_price_gram,
        "change_pct": change_pct,
        "recommendation": rec_usd,
        "current_usd": current_price_usd,
        "current_rate_idr": current_rate_idr,
        "latest_features": latest_features,
        "confidence_score": conf_score,
        "confidence_direction": conf_direction
    }


def interactive_forecast(
    model: Any,
    latest_features: pd.DataFrame,
    current_usd: float,
    current_idr: float,
    days: int,
    historical_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Run recursive forecast for N days.

    Args:
        model: Trained model.
        latest_features: Latest feature row.
        current_usd: Current USD price.
        current_idr: Current IDR rate.
        days: Number of days to forecast.
        historical_df: Historical dataframe for indicator calculation.
    """
    logger.info(f"Generating {days}-day recursive forecast...")

    history_buffer = None
    if historical_df is not None:
        history_buffer = historical_df.tail(100).copy()

    forecasts = predictor.recursive_forecast(
        model, latest_features, current_usd, current_idr,
        days=days, historical_df=history_buffer
    )

    print(f"\n{'Day':<5} | {'Date':<12} | {'Price (IDR/g)':<18} | {'Change'}")
    print("-" * 55)

    base_price = (current_usd * current_idr) / config.GRAMS_PER_OZ
    for f in forecasts:
        price_gram = f['Price_IDR'] / config.GRAMS_PER_OZ
        change = ((price_gram - base_price) / base_price) * 100
        print(f"{f['Day']:<5} | {f['Date']:<12} | Rp {price_gram:,.0f}        | {change:+.2f}%")


def main():
    """Main entry point for training and prediction."""
    parser = argparse.ArgumentParser(description="Gold Price Predictor Engine")
    parser.add_argument("--days", type=int, default=0, help="Interactive forecast days")
    args = parser.parse_args()

    df, _ = fetch_and_prepare_data()
    if df is None:
        return

    model, mae = train_pipeline(df)
    result = run_prediction(model, df, mae)

    action_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print("\n" + "=" * 42)
    print(f"DATE: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"ACTION WINDOW: {action_date}")
    print("-" * 42)
    print(f"Current:   Rp {result['current_price_idr']:,.0f}")
    print(f"Predicted: Rp {result['predicted_price_idr']:,.0f}")
    print(f"Change:    {result['change_pct']*100:+.2f}%")
    print(f"Signal:    {result['recommendation']}")
    print("=" * 42 + "\n")

    signal_logger.log_daily_signal(
        date=action_date,
        price_usd=result['current_usd'],
        predicted_usd=result['current_usd'] * (1 + result['change_pct']),
        signal=result['recommendation'],
        confidence_score=result['confidence_score'],
        confidence_direction=result['confidence_direction']
    )

    days = args.days
    if days == 0:
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
        logger.info("Operation cancelled.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
