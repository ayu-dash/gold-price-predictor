"""
Signal Logger Module.

Handles persistent storage of daily trading signals to a CSV file.
This allows for historical tracking of model performance and signal consistency.
"""
import os
import pandas as pd
from datetime import datetime

SIGNAL_LOG_PATH = "data/signals.csv"

def log_daily_signal(
    date: str,
    price_usd: float,
    predicted_usd: float,
    signal: str,
    confidence_score: float,
    confidence_direction: str
) -> None:
    """
    Appends a daily signal record to the CSV log.
    
    Args:
        date (str): Date string (YYYY-MM-DD).
        price_usd (float): Current Price at time of signal.
        predicted_usd (float): Predicted Price.
        signal (str): The signal (e.g. BUY, SELL, REDUCE).
        confidence_score (float): Confidence score (0-100).
        confidence_direction (str): 'UP' or 'DOWN'.
    """
    os.makedirs(os.path.dirname(SIGNAL_LOG_PATH), exist_ok=True)
    
    new_record = {
        'Date': date,
        'Price_USD': round(price_usd, 2),
        'Predicted_USD': round(predicted_usd, 2),
        'Signal': signal,
        'Confidence': f"{confidence_direction} ({confidence_score}%)",
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check if file exists to determine if we need header
    file_exists = os.path.exists(SIGNAL_LOG_PATH)
    
    df = pd.DataFrame([new_record])
    
    if file_exists:
        # Check for duplicates (don't log same date twice)
        existing_df = pd.read_csv(SIGNAL_LOG_PATH)
        if date in existing_df['Date'].values:
            # Update existing row logic could go here, but for now we skip or overwrite?
            # Let's just append for now, or maybe overwrite if same day?
            # Simple approach: Overwrite if date exists
            existing_df = existing_df[existing_df['Date'] != date]
            df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(SIGNAL_LOG_PATH, index=False)
        else:
            df.to_csv(SIGNAL_LOG_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(SIGNAL_LOG_PATH, index=False)

def get_signal_history() -> pd.DataFrame:
    """
    Retrieves the full signal history.
    
    Returns:
        pd.DataFrame: History dataframe or empty if none.
    """
    if not os.path.exists(SIGNAL_LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(SIGNAL_LOG_PATH).sort_values('Date', ascending=False)
