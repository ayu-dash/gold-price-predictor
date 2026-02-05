"""
Sentiment logging utility.

Handles saving and retrieving historical sentiment data.
"""

import os
import pandas as pd
from datetime import datetime
import logging
import config

logger = logging.getLogger(__name__)

SENTIMENT_CSV_PATH = os.path.join(os.path.dirname(config.CSV_PATH), "sentiment_history.csv")

def log_sentiment(avg_sentiment: float, sentiment_counts: dict):
    """
    Log daily sentiment data to CSV.
    
    Args:
        avg_sentiment: Average sentiment score.
        sentiment_counts: Dictionary with 'positive', 'negative', 'neutral' counts.
    """
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    total = sum(sentiment_counts.values())
    pos_ratio = sentiment_counts.get('positive', 0) / total if total > 0 else 0
    neg_ratio = sentiment_counts.get('negative', 0) / total if total > 0 else 0
    neu_ratio = sentiment_counts.get('neutral', 0) / total if total > 0 else 0
    
    new_entry = {
        'Date': date_str,
        'Avg_Sentiment': avg_sentiment,
        'Pos_Ratio': pos_ratio,
        'Neg_Ratio': neg_ratio,
        'Neu_Ratio': neu_ratio,
        'Count': total,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    df_new = pd.DataFrame([new_entry])
    
    if os.path.exists(SENTIMENT_CSV_PATH):
        try:
            df_old = pd.read_csv(SENTIMENT_CSV_PATH)
            # Avoid duplicate entries for the same day
            if date_str in df_old['Date'].values:
                df_old = df_old[df_old['Date'] != date_str]
            
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final.to_csv(SENTIMENT_CSV_PATH, index=False)
            logger.info(f"Sentiment logged for {date_str}")
        except Exception as e:
            logger.error(f"Failed to update sentiment log: {e}")
    else:
        try:
            os.makedirs(os.path.dirname(SENTIMENT_CSV_PATH), exist_ok=True)
            df_new.to_csv(SENTIMENT_CSV_PATH, index=False)
            logger.info(f"Sentiment log created and entry added for {date_str}")
        except Exception as e:
            logger.error(f"Failed to create sentiment log: {e}")

def get_sentiment_history() -> pd.DataFrame:
    """Retrieve historical sentiment data."""
    if os.path.exists(SENTIMENT_CSV_PATH):
        try:
            df = pd.read_csv(SENTIMENT_CSV_PATH, index_col='Date', parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read sentiment history: {e}")
            return pd.DataFrame()
    return pd.DataFrame()
