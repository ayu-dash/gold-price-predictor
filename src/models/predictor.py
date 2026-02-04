"""
Model Training and Prediction Module.

Handles training Gradient Boosting models, saving/loading artifacts,
and generating recursive forecasts with dynamic feature simulation.
"""
import os
import pickle
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Local import for dynamic feature engineering
from src.features import engineering

HOLD_THRESHOLD = 0.005  # 0.5%


def train_model(
    X: pd.DataFrame, 
    y: pd.Series,
    quantile: Optional[float] = None
) -> Tuple[GradientBoostingRegressor, pd.DataFrame, pd.Series]:
    """
    Trains a Gradient Boosting Regressor (Support for Quantile Loss).

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector (returns).
        quantile (Optional[float]): If provided, uses 'quantile' loss with specified alpha.

    Returns:
        Tuple[GradientBoostingRegressor, pd.DataFrame, pd.Series]: 
            Trained model, X_test, and y_test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    loss_type = 'quantile' if quantile is not None else 'squared_error'
    
    model = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        loss=loss_type,
        alpha=quantile if quantile is not None else 0.9,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model, X_test, y_test


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_classifier(
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[GradientBoostingClassifier, float]:
    """
    Trains a Gradient Boosting Classifier to predict Price Direction (Up/Down).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector (1 for Up, 0 for Down/Flat).

    Returns:
        Tuple[GradientBoostingClassifier, float]: Trained model and accuracy score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    clf = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Calculate accuracy
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return clf, acc


def get_classification_confidence(
    model: GradientBoostingClassifier,
    X_input: pd.DataFrame
) -> Tuple[str, float]:
    """
    Returns the predicted direction and the confidence (probability).
    
    Args:
        model (GradientBoostingClassifier): Trained classifier.
        X_input (pd.DataFrame): Single row of features.
        
    Returns:
        Tuple[str, float]: Direction ("UP"/"DOWN") and Confidence (0.0 - 1.0).
    """
    # [P(Down), P(Up)]
    probs = model.predict_proba(X_input)[0] 
    
    if probs[1] > 0.5:
        return "UP", probs[1]
    else:
        return "DOWN", probs[0]


def evaluate_model(
    model: GradientBoostingRegressor, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluates the model performance.

    Args:
        model (GradientBoostingRegressor): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.

    Returns:
        Tuple[float, float, np.ndarray]: RMSE, MAE, and Predictions.
    """
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    return rmse, mae, predictions


def save_model(model: Any, path: str = "models/gold_model.pkl") -> None:
    """
    Saves the trained model to disk.

    Args:
        model (Any): The model object to save.
        path (str): Destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str = "models/gold_model.pkl") -> Optional[Any]:
    """
    Loads a trained model from disk.

    Args:
        path (str): Path to the model file.

    Returns:
        Optional[Any]: The loaded model or None if not found.
    """
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_recommendation(
    current_price: float, 
    predicted_price: float,
    conf_direction: Optional[str] = None,
    conf_score: Optional[float] = 0.0,
    rsi: Optional[float] = 50.0,
    sma: Optional[float] = None
) -> Tuple[str, float]:
    """
    Generates a Buy/Sell/Hold signal based on predicted change & confidence,
    with momentum-sensing filters to avoid "Permabear" bias.
    
    Logic:
    - Stability Threshold: 0.25% (ignore noise)
    - Bear Protection: In bullish trends (RSI > 60 or Price > SMA), 
      only SELL if predicted drop > 0.5%.
    - Confidence Calibration: Penalize contrarian signals.
    """
    change_pct = (predicted_price - current_price) / current_price
    
    # 1. Trend Sensing
    is_bullish = False
    if rsi > 60:
        is_bullish = True
    if sma and current_price > sma:
        is_bullish = True
        
    STABILITY_THRESHOLD = 0.0025 # 0.25%
    BEAR_PROTECTION_THRESHOLD = 0.005 # 0.5%
    
    # 2. Bullish Market Protection (Fixes Permabear Bias)
    if is_bullish and change_pct < 0:
        # Market is bullish, but AI predicts a dip
        if abs(change_pct) < BEAR_PROTECTION_THRESHOLD:
            # Drop is too small to justify a SELL/REDUCE in a strong uptrend
            return "HOLD", change_pct
            
    # 3. Strong Price-Driven Signals
    if change_pct > STABILITY_THRESHOLD:
        return "BUY", change_pct
    elif change_pct < -STABILITY_THRESHOLD:
        return "SELL", change_pct
    
    # 4. Nuanced Confidence-Driven Signals
    if conf_score and conf_score > 65.0:
        # Calibration: Penalize confidence if signal goes against the trend
        effective_conf = conf_score
        if is_bullish and conf_direction == "DOWN":
            effective_conf -= 10.0 # Caution: Contrarian Bearish
        elif not is_bullish and conf_direction == "UP":
            effective_conf -= 10.0 # Caution: Contrarian Bullish
            
        if effective_conf > 65.0:
            if conf_direction == "UP" and change_pct > 0.001:
                return "ACCUMULATE", change_pct
            elif conf_direction == "DOWN" and change_pct < -0.001:
                return "REDUCE", change_pct
                
    return "HOLD", change_pct


def recursive_forecast(
    model: Any, 
    last_known_features: pd.DataFrame, 
    current_price_usd: float, 
    current_rate_idr: float, 
    days: int = 5, 
    historical_df: Optional[pd.DataFrame] = None,
    shifts: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Generates multi-day forecast with dynamic feature simulation.
    
    Performs a 'Gaussian Random Walk' on macro features and recalculates
    technical indicators (RSI, MACD) after every simulated step to ensure
    features remain consistent with the simulated price.

    Args:
        model (Any): Trained regressor.
        last_known_features (pd.DataFrame): The latest feature row.
        current_price_usd (float): Latest Gold price in USD.
        current_rate_idr (float): Latest USD/IDR rate.
        days (int): Number of days to forecast.
        historical_df (Optional[pd.DataFrame]): Buffer of historical data 
                                                for indicator calculation.

    Returns:
        List[Dict[str, Any]]: List of daily forecast objects.
    """
    forecasts = []
    
    # Initialize simulation dataframe
    if historical_df is None:
        # Fallback if no history provided (indicators won't be dynamic)
        current_sim_df = last_known_features.copy()
    else:
        current_sim_df = historical_df.copy()

    current_sim_price = current_price_usd
    
    # Feature columns expected by the model
    feature_cols = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
    ]

    for i in range(1, days + 1):
        # 1. Prepare Input Features
        available_cols = [c for c in feature_cols if c in current_sim_df.columns]
        next_features = current_sim_df.iloc[[-1]][available_cols].copy()

        # 2. Predict Return
        # Support for multi-model quantile ensemble
        if isinstance(model, dict):
            # Quantile Ensemble: {'low': model_0.05, 'med': model_0.5, 'high': model_0.95}
            pred_ret = model['med'].predict(next_features)[0]
            pred_low = model['low'].predict(next_features)[0]
            pred_high = model['high'].predict(next_features)[0]
        else:
            # Traditional single model
            pred_ret = model.predict(next_features)[0]
            pred_low = pred_ret - 0.012 # Fallback fixed interval
            pred_high = pred_ret + 0.012
        
        # --- REALISM LOGIC ---
        # A. Dampening
        pred_ret = pred_ret * 0.5 
        
        # B. Decay
        decay_factor = 0.85 ** (i / 5) 
        pred_ret = pred_ret * decay_factor
        
        # C. Jitter
        jitter = np.random.normal(0, 0.003) 
        pred_ret += jitter
        
        # D. Clamp
        pred_ret = np.clip(pred_ret, -0.03, 0.03)
        # ---------------------
        
        # 3. Update Simulated Price
        current_sim_price = current_sim_price * (1 + pred_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr
        
        # 4. Create New Row for Next Iteration
        new_row = current_sim_df.iloc[[-1]].copy()
        
        # Shift date
        last_date = new_row.index[0]
        new_row.index = [last_date + pd.Timedelta(days=1)]
        
        # Update Gold Price
        new_row['Gold'] = current_sim_price
        
        # Update Macro Features (Gaussian Random Walk or Manual Shift)
        macro_features = ['Oil', 'DXY', 'SP500', 'Silver', 'Copper', 'US10Y', 'USD_IDR']
        for feat in macro_features:
            if feat in new_row.columns:
                if shifts and feat in shifts and shifts[feat] != 0:
                    # Apply manual linear shift divided by days to make it progressive
                    step_shift = shifts[feat] / days
                    new_row[feat] = new_row[feat] * (1 + step_shift)
                else:
                    # Random walk: +/- 0.5% volatility
                    noise = np.random.normal(0, 0.005)
                    new_row[feat] = new_row[feat] * (1 + noise)
                
        # Decay Sentiment slowly to neutral
        if 'Sentiment' in new_row.columns:
            new_row['Sentiment'] = new_row['Sentiment'] * 0.95

        # Append new row to simulation buffer
        current_sim_df = pd.concat([current_sim_df, new_row])
        
        # 5. Dynamic Indicator Recalculation
        # This updates RSI, MACD, SMAs based on the NEW price and NEW history
        current_sim_df = engineering.add_technical_indicators(current_sim_df)
        
        # Store Forecast
        forecasts.append({
            'Day': i,
            'Date': new_row.index[0].strftime('%Y-%m-%d'),
            'Price_USD': current_sim_price,
            'Price_IDR': current_sim_price_idr,
            'Price_Min_IDR': current_sim_price * (1 + pred_low) * current_rate_idr,
            'Price_Max_IDR': current_sim_price * (1 + pred_high) * current_rate_idr,
            'Return_Pct': round(pred_ret * 100, 2)
        })
            
    return forecasts
