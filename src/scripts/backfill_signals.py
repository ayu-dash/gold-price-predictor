"""
Backfill Signals Script.

Loads historical data and the trained model to generate daily signals
for the past N days. This populates the signals.csv file for the dashboard.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data import loader, signal_logger
from src.features import engineering
from src.models import predictor
import app  # To reuse get_model logic if needed, or we just load manually

def backfill(days=30):
    print(f"Starting backfill for last {days} days...")
    
    # 1. Load Data
    df = loader.update_local_database()
    df = engineering.add_technical_indicators(df)
    
    # 2. Load Model
    # we can borrow logic from app.py or just load directly
    model_obj = app.get_model()
    if model_obj is None:
        print("Error: Models not found. Train them first with main.py")
        return

    # 3. Iterate
    # We need to simulate "being in the past"
    # For each day in the last 30 days:
    #   subset data up to that day
    #   predict for day+1
    #   log signal
    
    end_date = df.index.max()
    start_date = end_date - timedelta(days=days)
    
    dates_to_process = df[df.index >= start_date].index
    
    for current_date in dates_to_process:
        # Define "Past" context
        # We need enough history for features
        past_df = df[df.index <= current_date].copy()
        
        # Mock/Fill Sentiment since we don't have historical daily sentiment easily
        # We can use the current known sentiment or a random walk if we wanted to be fancy.
        # But Model expects the column.
        if 'Sentiment' not in past_df.columns:
            # Load current sentiment as baseline
            # In a real backtest we'd load historical sentiment, but for "filling the UI" this is acceptable.
            # We'll just use a neutral-ish value or the current value.
            # Let's check if loader has it cached or we fetch it once.
            pass 
        
        # We need to make sure 'Sentiment' is in latest_row.
        # The 'df' from loader doesn't have sentiment column by default unless we merge it.
        # We will inject it.
        # Let's assume neutral 0.0 for deep history, or variable.
        # Getting fancy: Use VIX as a proxy for sentiment? High VIX = Negative Sentiment?
        # VIX is in df['VIX'].
        # Correlation: VIX up -> Sentiment down.
        # Let's synthesize it: Sentiment ~ - (VIX_Norm - 0.5) * 2
        
        vix_val = past_df['VIX_Norm'].iloc[-1] if 'VIX_Norm' in past_df else 0.5
        synthetic_sent = - (vix_val - 0.5) * 1.5 # Rough proxy
        past_df['Sentiment'] = synthetic_sent # Broadcast to all rows? No need, just latest.
        
        features = [
            'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
            'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
            'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
        ]
        
        # Ensure sentiment is available for the row slicing
        past_df['Sentiment'] = past_df['Sentiment'].fillna(synthetic_sent)
        
        available_features = [f for f in features if f in past_df.columns]
        latest_row = past_df[available_features].iloc[[-1]]
        
        # Predict
        if isinstance(model_obj, dict):
            pred_ret = model_obj['med'].predict(latest_row)[0]
        else:
            pred_ret = model_obj.predict(latest_row)[0]
            
        current_usd = past_df['Gold'].iloc[-1]
        predicted_usd = current_usd * (1 + pred_ret)
        
        # Classification
        conf_dir = "N/A"
        conf_score = 0.0
        if isinstance(model_obj, dict) and 'clf' in model_obj:
            conf_dir, conf_score = predictor.get_classification_confidence(
                model_obj['clf'], latest_row
            )
            conf_score = round(conf_score * 100, 1)

        # The predictive target date is tomorrow
        target_date_obj = current_date + timedelta(days=1)
        target_date_str = target_date_obj.strftime('%Y-%m-%d')
        
        # Get Technicals for the recommendation logic
        rsi_val = past_df['RSI'].iloc[-1] if 'RSI' in past_df.columns else 50.0
        sma_val = past_df['SMA_14'].iloc[-1] if 'SMA_14' in past_df.columns else None

        rec, _ = predictor.make_recommendation(
            current_usd, 
            predicted_usd, 
            conf_direction=conf_dir, 
            conf_score=conf_score,
            rsi=rsi_val,
            sma=sma_val
        )
        
        print(f"[{target_date_str}] Price: {current_usd:.2f} -> Sig: {rec} (Observed: {current_date.strftime('%Y-%m-%d')})")
        
        # Log it (Overwriting if exists because we want clean history)
        signal_logger.log_daily_signal(
            date=target_date_str,
            price_usd=current_usd,
            predicted_usd=predicted_usd,
            signal=rec,
            confidence_score=conf_score,
            confidence_direction=conf_dir
        )

if __name__ == "__main__":
    backfill(days=45) # 45 days backfill
