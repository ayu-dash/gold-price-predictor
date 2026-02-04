"""
Signal logger module.

Handles persistent storage of daily trading signals to a CSV file.
"""

import os
from datetime import datetime

import pandas as pd

import config


def log_daily_signal(
    date: str,
    price_usd: float,
    predicted_usd: float,
    signal: str,
    confidence_score: float,
    confidence_direction: str
) -> None:
    """Append a daily signal record to the CSV log."""
    log_path = config.SIGNAL_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    new_record = {
        'Date': date,
        'Price_USD': round(price_usd, 2),
        'Predicted_USD': round(predicted_usd, 2),
        'Signal': signal,
        'Confidence': f"{confidence_direction} ({confidence_score}%)",
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    file_exists = os.path.exists(log_path)
    df = pd.DataFrame([new_record])

    if file_exists:
        existing_df = pd.read_csv(log_path)
        if date in existing_df['Date'].values:
            existing_df = existing_df[existing_df['Date'] != date]
            df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(log_path, index=False)
        else:
            df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def get_signal_history() -> pd.DataFrame:
    """Retrieve the full signal history."""
    log_path = config.SIGNAL_LOG_PATH
    if not os.path.exists(log_path):
        return pd.DataFrame()
    return pd.read_csv(log_path).sort_values('Date', ascending=False)
