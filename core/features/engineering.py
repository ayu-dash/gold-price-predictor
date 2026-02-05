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
    df['EMA_14'] = ta.trend.ema_indicator(df['Gold'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Gold'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Gold'], window=200)

    df['RSI'] = ta.momentum.rsi(df['Gold'], window=14)
    df['RSI_7'] = ta.momentum.rsi(df['Gold'], window=7)
    df['ROC_10'] = ta.momentum.roc(df['Gold'], window=10)

    df['MACD'] = ta.trend.macd(df['Gold'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Gold'])

    high = df['Gold']
    low = df['Gold']
    close = df['Gold']
    df['Stoch'] = ta.momentum.stoch(high, low, close, window=14)
    df['WilliamsR'] = ta.momentum.williams_r(high, low, close, lbp=14)
    df['CCI'] = ta.trend.cci(high, low, close, window=20)

    df['BB_High'] = ta.volatility.bollinger_hband(df['Gold'], window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Gold'], window=20, window_dev=2)
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Gold'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)

    df['Gold_Returns'] = df['Gold'].pct_change()
    
    df['Returns'] = df['Gold_Returns']

    if 'VIX' in df.columns:
        df['VIX_Norm'] = df['VIX'] / 100.0
        df['VIX_Return'] = df['VIX'].pct_change()
        df['VIX_Lag1'] = df['VIX_Return'].shift(1)

    if 'GVZ' in df.columns:
        df['GVZ_Norm'] = df['GVZ'] / 100.0
        
    if 'US10Y' in df.columns:
        df['US10Y_Diff'] = df['US10Y'].diff()
        df['US10Y_Lag1'] = df['US10Y_Diff'].shift(1)

    if 'Silver' in df.columns and (df['Silver'] > 0).all():
        df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']
    
    if 'DXY' in df.columns:
        df['DXY_Ret_Lag1'] = df['DXY'].pct_change().shift(1)
        
    if 'SP500' in df.columns:
        df['SP500_Ret_Lag1'] = df['SP500'].pct_change().shift(1)

    df['Return_Lag1'] = df['Gold_Returns'].shift(1)
    df['Return_Lag2'] = df['Gold_Returns'].shift(2)
    df['Return_Lag3'] = df['Gold_Returns'].shift(3)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    
    df['Volatility_5'] = df['Gold_Returns'].rolling(window=5).std()
    df['Momentum_5'] = df['Gold'].pct_change(periods=5)

    return df
