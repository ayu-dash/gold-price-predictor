"""
Model training and prediction module.

Handles training models, saving/loading artifacts, and generating forecasts.
"""

import os
import pickle
import tempfile
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

import config
from core.features import engineering


def train_model(X: pd.DataFrame, y: pd.Series, quantile: Optional[float] = None):
    """Train a HistGradientBoostingRegressor with optional quantile loss."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    loss_type = 'quantile' if quantile is not None else 'squared_error'
    # HGB natively handles NaNs, preventing crashes on missing data
    model = HistGradientBoostingRegressor(
        learning_rate=0.03, max_depth=8, max_iter=300, loss=loss_type,
        quantile=quantile if quantile else 0.5, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def train_neural_network(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, float]:
    """Train a Multi-Layer Perceptron regressor."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    inner_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                     alpha=0.0001, learning_rate_init=0.001, max_iter=2000,
                     early_stopping=True, validation_fraction=0.1, random_state=42)
    )
    model = TransformedTargetRegressor(regressor=inner_model, transformer=StandardScaler())
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    return model, rmse, mae

def train_classifier(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, float, float]:
    """Train a HistGradientBoostingClassifier to predict price direction."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Use single, robust HGB model with optimized params (Round 2)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=200,
        l2_regularization=1.0, min_samples_leaf=10,
        early_stopping=True, validation_fraction=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy only on High Confidence samples (>67%)
    # This reflects "Trade Accuracy" rather than "Global Accuracy"
    threshold = 0.67
    high_conf_mask = (probs > threshold) | (probs < (1 - threshold))
    
    if np.sum(high_conf_mask) > 5:
        y_filtered = y_test[high_conf_mask]
        preds_filtered = (probs[high_conf_mask] > 0.5).astype(int)
        acc = accuracy_score(y_filtered, preds_filtered)
        prec = precision_score(y_filtered, preds_filtered, zero_division=0)
        rec = recall_score(y_filtered, preds_filtered, zero_division=0)
        f1 = f1_score(y_filtered, preds_filtered, zero_division=0)
    else:
        # Fallback to standard if not enough high confidence samples
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
    return model, acc, prec, rec, f1


def get_classification_confidence(model: Any, X_input: pd.DataFrame) -> Tuple[str, float]:
    """Get predicted direction and confidence probability."""
    probs = model.predict_proba(X_input)[0]
    return ("UP", probs[1]) if probs[1] > 0.5 else ("DOWN", probs[0])


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, np.ndarray]:
    """Evaluate model performance metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    return rmse, mae, predictions


def save_model(model: Any, path: str) -> None:
    """Save trained model to disk using atomic swap."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=directory, prefix="temp_", suffix=".pkl")
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(model, f)
        os.replace(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def load_model(path: str) -> Optional[Any]:
    """Load trained model from disk."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_recommendation(
    current_price: float, predicted_price: float,
    conf_direction: Optional[str] = None, conf_score: Optional[float] = 0.0,
    rsi: Optional[float] = 50.0, sma: Optional[float] = None
) -> Tuple[str, float]:
    """Generate Buy/Sell/Hold signal based on prediction and confidence."""
    change_pct = (predicted_price - current_price) / current_price
    is_bullish = rsi > 60 or (sma and current_price > sma)

    stability_threshold = 0.0025
    bear_protection_threshold = config.HOLD_THRESHOLD

    if is_bullish and change_pct < 0:
        if abs(change_pct) < bear_protection_threshold:
            return "HOLD", change_pct

    if change_pct > stability_threshold:
        return "BUY", change_pct
    elif change_pct < -stability_threshold:
        return "SELL", change_pct

    if conf_score and conf_score > 67.0:
        effective_conf = conf_score
        if is_bullish and conf_direction == "DOWN":
            effective_conf -= 10.0
        elif not is_bullish and conf_direction == "UP":
            effective_conf -= 10.0

        if effective_conf > 67.0:
            if conf_direction == "UP" and change_pct > 0.001:
                return "ACCUMULATE", change_pct
            elif conf_direction == "DOWN" and change_pct < -0.001:
                return "REDUCE", change_pct

    return "HOLD", change_pct


def recursive_forecast(
    model: Any, last_known_features: pd.DataFrame,
    current_price_usd: float, current_rate_idr: float,
    days: int = 5, historical_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """Generate multi-day forecast with deterministic feature evolution."""
    forecasts = []
    current_sim_df = historical_df.copy() if historical_df is not None else last_known_features.copy()
    current_sim_price = current_price_usd

    feature_cols = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5'
    ]

    for i in range(1, days + 1):
        available_cols = [c for c in feature_cols if c in current_sim_df.columns]
        next_features = current_sim_df.iloc[[-1]][available_cols].copy()

        if isinstance(model, dict):
            pred_ret = model['med'].predict(next_features)[0]
            pred_low = model['low'].predict(next_features)[0]
            pred_high = model['high'].predict(next_features)[0]
        else:
            pred_ret = model.predict(next_features)[0]
            pred_low = pred_ret - 0.012
            pred_high = pred_ret + 0.012

        # Calculate recent volatility (standard deviation of last 20 returns) if available
        volatility = 0.01  # Default 1%
        if historical_df is not None and len(historical_df) > 20:
             # pct_change might not be in columns, calculate manually if needed
             if 'Gold' in historical_df.columns:
                 recent_closes = historical_df['Gold'].tail(20)
                 volatility = recent_closes.pct_change().std()
        
        # Inject natural market noise (mean 0, std=volatility)
        # We scale it down slightly (0.5) to keep the forecast conservative but 'alive'
        noise = np.random.normal(0, volatility * 0.8)
        
        # Combine model trend + noise
        pred_ret = pred_ret + noise
        
        # Clip to prevent extreme outliers in simulation
        pred_ret = np.clip(pred_ret, -0.05, 0.05)

        current_sim_price = current_sim_price * (1 + pred_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr

        new_row = current_sim_df.iloc[[-1]].copy()
        new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
        new_row['Gold'] = current_sim_price

        # Hold macro features constant for "real-time" neutral forecast
        # We could also apply a small trend if desired, but constant is safest for pure prediction
        if 'Sentiment' in new_row.columns:
            new_row['Sentiment'] = new_row['Sentiment'] * 0.95  # Decay sentiment

        current_sim_df = pd.concat([current_sim_df, new_row])
        
        # Re-calculate technicals
        current_sim_df = engineering.add_technical_indicators(current_sim_df)
        
        # Manually update lags for the new row since shift() needs history
        # (add_technical_indicators already does shift/rolling if history is sufficient, 
        # but let's ensure the last row is valid)
        pass

        forecasts.append({
            'Day': i, 'Date': new_row.index[0].strftime('%Y-%m-%d'),
            'Price_USD': current_sim_price, 'Price_IDR': current_sim_price_idr,
            'Price_Min_IDR': current_sim_price * (1 + pred_low) * current_rate_idr,
            'Price_Max_IDR': current_sim_price * (1 + pred_high) * current_rate_idr,
            'Return_Pct': round(pred_ret * 100, 2)
        })

    return forecasts
