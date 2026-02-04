"""
Feature engineering module.

Handles calculation of technical indicators.
"""

import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe using the 'ta' library."""
    if df.empty:
        return df

    if 'Gold' not in df.columns:
        raise ValueError("Gold price column missing in dataframe")

    df['SMA_7'] = ta.trend.sma_indicator(df['Gold'], window=7)
    df['SMA_14'] = ta.trend.sma_indicator(df['Gold'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Gold'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Gold'], window=200)

    df['RSI'] = ta.momentum.rsi(df['Gold'], window=14)
    df['RSI_7'] = ta.momentum.rsi(df['Gold'], window=7)

    df['MACD'] = ta.trend.macd(df['Gold'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Gold'])

    df['BB_High'] = ta.volatility.bollinger_hband(df['Gold'], window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Gold'], window=20, window_dev=2)
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Gold'], window=20, window_dev=2)

    df['Gold_Returns'] = df['Gold'].pct_change()

    if 'VIX' in df.columns:
        df['VIX_Norm'] = df['VIX'] / 100.0

    if 'GVZ' in df.columns:
        df['GVZ_Norm'] = df['GVZ'] / 100.0

    return df
