
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
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

    # --- 4. Quantile Regression Training ---
    print("\n--- Training Model Ensemble (Low/Med/High) ---")
    
    # Train Median (Main Forecast)
    med_model, X_test, y_test = predictor.train_model(X, y, quantile=0.5)
    predictor.save_model(med_model, "models/gold_model_med.pkl")
    
    # Train Low/High (Confidence Intervals)
    low_model, _, _ = predictor.train_model(X, y, quantile=0.05)
    high_model, _, _ = predictor.train_model(X, y, quantile=0.95)
    
    predictor.save_model(low_model, "models/gold_model_low.pkl")
    predictor.save_model(high_model, "models/gold_model_high.pkl")
    
    # --- 5. Classification Training (Up/Down) ---
    print("\n--- Training Direction Classifier (Implicit) ---")
    # We now derive direction from the Median Regression Model
    # This ensures consistency: If Forecast > 0, Signal is UP.
    
    # Calculate metrics on Test Set using REGRESSION model
    # FILTER: Only evaluate accuracy when the model predicts a specific move > 0.1% (Noise Filter)
    y_pred_reg = med_model.predict(X_test)
    y_true_reg = y_test.values
    
    # Create mask for "High Confidence" (Significant Move) predictions
    # We ignore days where model predicts ~0.0% change
    significant_move_mask = np.abs(y_pred_reg) > 0.001
    
    if np.sum(significant_move_mask) > 10:
        # Evaluate on the subset of data where model had conviction
        y_pred_filtered = (y_pred_reg[significant_move_mask] > 0).astype(int)
        y_true_filtered = (y_true_reg[significant_move_mask] > 0).astype(int)
        
        clf_acc = accuracy_score(y_true_filtered, y_pred_filtered)
        clf_prec = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        clf_rec = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
        print(f"Filtered (High Conf) Direction -> Acc: {clf_acc:.2%}, Samples: {np.sum(significant_move_mask)}")
    else:
        # Fallback to full set if not enough samples
        print("Not enough significant moves for filtered eval. Using full set.")
        y_pred_class = (y_pred_reg > 0).astype(int)
        y_true_class = (y_test > 0).astype(int)
        clf_acc = accuracy_score(y_true_class, y_pred_class)
        clf_prec = precision_score(y_true_class, y_pred_class, zero_division=0)
        clf_rec = recall_score(y_true_class, y_pred_class, zero_division=0)

    print(f"Regression-Derived Direction -> Acc: {clf_acc:.2%}, Prec: {clf_prec:.2%}, Rec: {clf_rec:.2%}")
    
    # We still keep the NN classifier in the dict for legacy compatibility
    clf_model = None 
    
    # --- 5b. Neural Network Experiment (Deep Learning Lite) ---
    print("\n--- Training Neural Network (MLP) ---")
    nn_model, nn_rmse, nn_mae = predictor.train_neural_network(X, y)
    predictor.save_model(nn_model, "models/gold_model_nn.pkl")
    print(f"Neural Network MAE: {nn_mae:.4f}")
    # ---------------------------------------------
    
    # --- 6. Evaluation ---
    rmse_med, mae_med, _ = predictor.evaluate_model(med_model, X_test, y_test)
    rmse_low, mae_low, _ = predictor.evaluate_model(low_model, X_test, y_test)
    rmse_high, mae_high, _ = predictor.evaluate_model(high_model, X_test, y_test)
    
    logger.info(f"Median Model -> MAE: {mae_med:.4f}")
    
    # Save Metrics Metadata
    import json
    metrics = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "models": {
            "median": {"mae": round(float(mae_med), 6), "rmse": round(float(rmse_med), 6)},
            "low": {"mae": round(float(mae_low), 6), "rmse": round(float(rmse_low), 6)},
            "high": {"mae": round(float(mae_high), 6), "rmse": round(float(rmse_high), 6)},
            "classifier": {
                "accuracy": round(clf_acc, 4),
                "precision": round(clf_prec, 4),
                "recall": round(clf_rec, 4)
            },
            "neural_network": {"mae": round(float(nn_mae), 6), "rmse": round(float(nn_rmse), 6)}
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_used": valid_features
    }
    
    metrics_path = "models/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Performance metrics saved to {metrics_path}")
    
    return med_model, mae_med


def run_prediction(
    model: Any, 
    df: pd.DataFrame, 
    mae: float
) -> Dict[str, Any]:
    """Generates next-day prediction and returns context."""
    features = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
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
    
    # Get Confidence (if classifier exists)
    conf_direction = "N/A"
    conf_score = 0.0
    if isinstance(model, dict) and 'clf' in model:
        conf_direction, conf_score = predictor.get_classification_confidence(
            model['clf'], latest_features
        )
        conf_score = round(conf_score * 100, 1)
        
        # Re-run recommendation with confidence
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

    # LOGGING
    from src.data import signal_logger
    
    # Log the signal
    signal_logger.log_daily_signal(
        date=action_date,
        price_usd=result['current_usd'],
        predicted_usd=result['current_usd'] * (1 + result['change_pct']), 
        signal=result['recommendation'],
        confidence_score=result['confidence_score'],
        confidence_direction=result['confidence_direction']
    )

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
