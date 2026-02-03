"""
Feature Engineering Module for Gold Price Predictor.
Handles technical indicators and data preparation for training.
"""

import numpy as np
import pandas as pd
import ta


def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe.
    """
    if df.empty:
        return df

    if 'Gold' not in df.columns:
        raise ValueError("Gold price column missing in dataframe")

    # Simple Moving Averages
    df['SMA_14'] = ta.trend.sma_indicator(df['Gold'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Gold'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Gold'], window=200)

    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Gold'], window=14)

    # MACD
    df['MACD'] = ta.trend.macd(df['Gold'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Gold'])

    # Bollinger Bands
    df['BB_High'] = ta.volatility.bollinger_hband(df['Gold'],
                                                  window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Gold'],
                                                 window=20, window_dev=2)

    # Daily Returns
    df['Gold_Returns'] = df['Gold'].pct_change()

    # Normalize Volatility Indices
    if 'VIX' in df.columns:
        df['VIX_Norm'] = df['VIX'] / 100.0

    if 'GVZ' in df.columns:
        df['GVZ_Norm'] = df['GVZ'] / 100.0

    return df




