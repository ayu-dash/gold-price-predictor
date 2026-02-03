"""
Model Training and Prediction Module.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

HOLD_THRESHOLD = 0.005  # 0.5%

def train_model(X, y):
    """Trains a Gradient Boosting Regressor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Using previous params
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the model."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    return rmse, mae, predictions

def save_model(model, path="models/gold_model.pkl"):
    """Saves model to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path="models/gold_model.pkl"):
    """Loads model from file."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_recommendation(current_price, predicted_price):
    """Generates Buy/Sell/Hold signal."""
    change_pct = (predicted_price - current_price) / current_price
    
    if change_pct > HOLD_THRESHOLD:
        return "BUY", change_pct
    elif change_pct < -HOLD_THRESHOLD:
        return "SELL", change_pct
    else:
        return "HOLD", change_pct

from src.features import engineering

def recursive_forecast(model, last_known_features, current_price_usd, 
                      current_rate_idr, days=5, historical_df=None):
    """
    Generates multi-day forecast with dynamic feature updating.
    Recalculates technical indicators (RSI/SMA/MACD) after each simulated day.
    """
    from src.features import engineering
    
    forecasts = []
    
    # Use historical data for indicator calculation buffer
    if historical_df is None:
        current_sim_df = last_known_features.copy()
    else:
        current_sim_df = historical_df.copy()

    current_sim_price = current_price_usd

    for i in range(1, days + 1):
        # 1. Ensure features are consistent with training
        feature_cols = [
            'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'VIX_Norm', 'GVZ_Norm',
            'Silver', 'Copper', 'US10Y', 'Nikkei', 'DAX',
            'SMA_14', 'RSI', 'MACD', 'Sentiment'
        ]
        
        # Prepare input features from the last row of simulation df
        next_features = current_sim_df.iloc[[-1]][feature_cols].copy()

        # 2. Predict Return for the next day
        pred_ret = model.predict(next_features)[0]
        
        # 3. Update Simulated Price
        current_sim_price = current_sim_price * (1 + pred_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr
        
        # 4. Append new state to history
        new_row = current_sim_df.iloc[[-1]].copy()
        new_row.index = [pd.Timestamp.now() + pd.Timedelta(days=i)]
        new_row['Gold'] = current_sim_price
        
        current_sim_df = pd.concat([current_sim_df, new_row])
        
        # 5. Recalculate Indicators (The Critical Fix)
        # This updates RSI, MACD, etc. based on the new price
        current_sim_df = engineering.add_technical_indicators(current_sim_df)
        
        forecasts.append({
            'Day': i,
            'Date': new_row.index[0].strftime('%Y-%m-%d'),
            'Price_USD': current_sim_price,
            'Price_IDR': current_sim_price_idr,
            'Return_Pct': round(pred_ret * 100, 2)
        })
            
    return forecasts
