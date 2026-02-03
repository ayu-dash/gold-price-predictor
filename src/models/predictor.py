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

def recursive_forecast(model, last_known_features, current_price_usd, 
                      current_rate_idr, days=5):
    """
    Generates multi-day forecast using recursive prediction strategy.
    """
    forecasts = []
    current_sim_price = current_price_usd
    # Ensure features is a DataFrame to avoid warning
    if isinstance(last_known_features, pd.Series):
        next_features = last_known_features.to_frame().T
    else:
        next_features = last_known_features.copy()

    for i in range(1, days + 1):
        # Predict return
        pred_ret = model.predict(next_features)[0]
        
        # Confidence Decay
        decay_factor = max(0, 1 - (i * 0.05))
        stable_ret = pred_ret * decay_factor
        
        # Add Noise (User Request: 0.5% volatility)
        uncertainty = 0.005 * (1 + (i * 0.05))
        noise = np.random.normal(0, uncertainty)
        
        active_ret = stable_ret + noise
        
        # Update Price
        current_sim_price = current_sim_price * (1 + active_ret)
        current_sim_price_idr = current_sim_price * current_rate_idr
        
        future_date = pd.Timestamp.now() + pd.Timedelta(days=i)
        
        forecasts.append({
            'Day': i,
            'Date': future_date.strftime('%Y-%m-%d'),
            'Price_USD': current_sim_price,
            'Price_IDR': current_sim_price_idr,
            'Return_Pct': round(active_ret * 100, 2)
        })
        
        # Update features (Gold price only)
        # Assuming 'Gold' column exists
        if 'Gold' in next_features.columns:
            next_features['Gold'] = current_sim_price
            
    return forecasts
