"""
Debug utility for signal outcome calculation.

Simulates the outcome logic in the API to verify signal accuracy.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def debug_logic():
    """Test the signal outcome calculation logic."""
    signals_df = pd.read_csv(config.SIGNAL_LOG_PATH)
    history_df = pd.read_csv(config.CSV_PATH, parse_dates=['Date'])
    history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d')
    actual_prices = history_df.set_index('Date')['Gold'].to_dict()

    print(f"Debug: Jan 30 Price in history: {actual_prices.get('2026-01-30')}")

    row = signals_df[signals_df['Date'] == '2026-01-30'].iloc[0]

    target_date = row['Date']
    signal_price = row['Price_USD']
    predicted_price = row['Predicted_USD']
    actual_price = actual_prices.get(target_date)

    if actual_price:
        predicted_diff = predicted_price - signal_price
        actual_diff = actual_price - signal_price

        print(f"Date: {target_date}")
        print(f"Signal Price: {signal_price}")
        print(f"Predicted Price: {predicted_price}")
        print(f"Actual Price: {actual_price}")
        print(f"Pred Diff: {predicted_diff}")
        print(f"Actual Diff: {actual_diff}")
        print(f"Product: {predicted_diff * actual_diff}")

        if predicted_diff * actual_diff > 0:
            outcome = "Correct"
        elif abs(predicted_diff) < 0.1 and abs(actual_diff) < 0.1:
            outcome = "Correct"
        else:
            outcome = "Wrong"

        print(f"Calculated Outcome: {outcome}")


if __name__ == "__main__":
    debug_logic()
