"""
Model training and prediction module.

Handles training models, saving/loading artifacts, and generating forecasts.
"""

import os
import pickle
import tempfile
import logging
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier, 
    RandomForestClassifier, HistGradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier, HistGradientBoostingRegressor,
    VotingRegressor, ExtraTreesRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

import config
from core.features import engineering

logger = logging.getLogger(__name__)


def train_model(X: pd.DataFrame, y: pd.Series, quantile: Optional[float] = None):
    """Train a HistGradientBoostingRegressor or Ensemble with optimized v5 Sandbox parameters."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    if quantile is not None:
        # Quantile regression for Low/High bounds
        model = HistGradientBoostingRegressor(
            loss='quantile', 
            quantile=quantile,
            learning_rate=0.04,
            max_iter=400,
            max_depth=6,
            l2_regularization=0.1,
            random_state=42
        )
    else:
        # High-performance Ensemble for Median (Target MAE < 0.5%)
        hgb = HistGradientBoostingRegressor(
            loss='absolute_error',
            learning_rate=0.03,
            max_iter=500,
            max_depth=8,
            l2_regularization=0.5,
            random_state=42
        )
        et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            bootstrap=True,
            random_state=42
        )
        model = VotingRegressor(estimators=[('hgb', hgb), ('et', et)])

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

def train_classifier(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, float, float, float]:
    """Train optimized classifier with class balancing and tuned hyperparameters."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # Check class distribution
    n_up = y_train.sum()
    n_down = len(y_train) - n_up
    logger.info(f"Class distribution - UP: {n_up}, DOWN: {n_down}")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_depth=8,
        l2_regularization=0.2,
        class_weight='balanced',
        random_state=42
    )
    
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model = VotingClassifier(
        estimators=[('rf', rf), ('hgb', hgb), ('et', et)],
        voting='soft'
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    
    threshold = 0.55
    preds = (probs > threshold).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    logger.info(f"Classifier threshold: {threshold}, Acc: {acc:.2%}, F1: {f1:.2%}")
        
    return model, acc, prec, rec, f1


def get_classification_confidence(model: Any, X_input: pd.DataFrame, predicted_return: float = None) -> Tuple[str, float]:
    """Get predicted direction and confidence using hybrid approach.
    
    Hybrid Logic:
    - If classifier confidence > 60%: Use classifier direction
    - If classifier confidence <= 60%: Fall back to regressor direction
    """
    CONFIDENCE_THRESHOLD = 0.60
    
    probs = model.predict_proba(X_input)[0]
    clf_prob = max(probs[0], probs[1])
    clf_direction = "UP" if probs[1] > probs[0] else "DOWN"
    
    if clf_prob <= CONFIDENCE_THRESHOLD and predicted_return is not None:
        reg_direction = "UP" if predicted_return > 0.001 else ("DOWN" if predicted_return < -0.001 else clf_direction)
        logger.debug(f"Hybrid fallback: clf={clf_direction}({clf_prob:.2%}) -> reg={reg_direction}")
        return (reg_direction, clf_prob)
    
    return (clf_direction, clf_prob)


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

    reduce_threshold = 0.004
    sell_threshold = 0.0075
    bear_protection_threshold = config.HOLD_THRESHOLD

    if is_bullish and change_pct < 0:
        if abs(change_pct) < bear_protection_threshold:
            return "HOLD", change_pct
        elif abs(change_pct) > sell_threshold:
            return "REDUCE", change_pct

    if change_pct > sell_threshold:
        return "BUY", change_pct
    elif change_pct > reduce_threshold:
        return "ACCUMULATE", change_pct
    elif change_pct < -sell_threshold:
        return "SELL", change_pct
    elif change_pct < -reduce_threshold:
        return "REDUCE", change_pct

    if conf_score and conf_score > 75.0:
        if conf_direction == "UP" and change_pct > 0.0015:
            return "ACCUMULATE", change_pct
        elif conf_direction == "DOWN" and change_pct < -0.0015:
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

    feature_cols = last_known_features.columns.tolist()

    for i in range(1, days + 1):
        next_features = current_sim_df.reindex(columns=feature_cols).iloc[[-1]].fillna(0.0)
        
        target_model = model['med'] if isinstance(model, dict) else model
        if hasattr(target_model, 'feature_names_in_'):
            expected = target_model.feature_names_in_
            next_features = next_features.reindex(columns=expected).fillna(0.0)

        if i == 1:
            logger.debug(f"Forecast step 1 features: {next_features.columns.tolist()}")

        if isinstance(model, dict):
            pred_ret = model['med'].predict(next_features)[0]
            pred_low = model['low'].predict(next_features)[0]
            pred_high = model['high'].predict(next_features)[0]
        else:
            pred_ret = model.predict(next_features)[0]
            pred_low = pred_ret - 0.012
            pred_high = pred_ret + 0.012

        volatility = 0.01
        if historical_df is not None and len(historical_df) > 20:
             if 'Gold' in historical_df.columns:
                 recent_closes = historical_df['Gold'].tail(20)
                 volatility = recent_closes.pct_change().std()
        
        noise = np.random.normal(0, volatility * 0.8)
        
        pred_ret = pred_ret + noise
        
        pred_ret = np.clip(pred_ret, -0.05, 0.05)

        current_sim_price = current_sim_price * (1 + pred_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr

        new_row = current_sim_df.iloc[[-1]].copy()
        new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
        new_row['Gold'] = current_sim_price

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
