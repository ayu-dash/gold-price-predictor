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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

import config
from core.features import engineering


def train_model(X: pd.DataFrame, y: pd.Series, quantile: Optional[float] = None):
    """Train a Gradient Boosting Regressor with optional quantile loss."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    loss_type = 'quantile' if quantile is not None else 'squared_error'
    model = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=4, loss=loss_type,
        alpha=quantile if quantile else 0.9, subsample=0.8,
        validation_fraction=0.1, n_iter_no_change=20, random_state=42
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
    """Train an MLP classifier to predict price direction."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                      alpha=0.0001, learning_rate_init=0.001, max_iter=1000,
                      early_stopping=True, validation_fraction=0.1, random_state=42)
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    return model, acc, prec, rec


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

    if conf_score and conf_score > 65.0:
        effective_conf = conf_score
        if is_bullish and conf_direction == "DOWN":
            effective_conf -= 10.0
        elif not is_bullish and conf_direction == "UP":
            effective_conf -= 10.0

        if effective_conf > 65.0:
            if conf_direction == "UP" and change_pct > 0.001:
                return "ACCUMULATE", change_pct
            elif conf_direction == "DOWN" and change_pct < -0.001:
                return "REDUCE", change_pct

    return "HOLD", change_pct


def recursive_forecast(
    model: Any, last_known_features: pd.DataFrame,
    current_price_usd: float, current_rate_idr: float,
    days: int = 5, historical_df: Optional[pd.DataFrame] = None,
    shifts: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """Generate multi-day forecast with dynamic feature simulation."""
    forecasts = []
    current_sim_df = historical_df.copy() if historical_df is not None else last_known_features.copy()
    current_sim_price = current_price_usd

    feature_cols = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
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

        pred_ret = pred_ret * 0.5 * (0.85 ** (i / 5))
        pred_ret += np.random.normal(0, 0.003)
        pred_ret = np.clip(pred_ret, -0.03, 0.03)

        current_sim_price = current_sim_price * (1 + pred_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr

        new_row = current_sim_df.iloc[[-1]].copy()
        new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
        new_row['Gold'] = current_sim_price

        macro_features = ['Oil', 'DXY', 'SP500', 'Silver', 'Copper', 'US10Y', 'USD_IDR']
        for feat in macro_features:
            if feat in new_row.columns:
                if shifts and feat in shifts and shifts[feat] != 0:
                    new_row[feat] = new_row[feat] * (1 + shifts[feat] / days)
                else:
                    new_row[feat] = new_row[feat] * (1 + np.random.normal(0, 0.005))

        if 'Sentiment' in new_row.columns:
            new_row['Sentiment'] = new_row['Sentiment'] * 0.95

        current_sim_df = pd.concat([current_sim_df, new_row])
        current_sim_df = engineering.add_technical_indicators(current_sim_df)

        forecasts.append({
            'Day': i, 'Date': new_row.index[0].strftime('%Y-%m-%d'),
            'Price_USD': current_sim_price, 'Price_IDR': current_sim_price_idr,
            'Price_Min_IDR': current_sim_price * (1 + pred_low) * current_rate_idr,
            'Price_Max_IDR': current_sim_price * (1 + pred_high) * current_rate_idr,
            'Return_Pct': round(pred_ret * 100, 2)
        })

    return forecasts
