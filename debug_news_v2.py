
import sys
import os
import random

# Add project root to path
sys.path.append(os.getcwd())

from core.data import loader

print("Testing fetch_news_sentiment() with EXPANDED sources...")
try:
    # Force refresh to ignore cache (though we deleted it)
    sentiment, headlines, counts = loader.fetch_news_sentiment(force_refresh=True)
    
    print(f"\nSentiment Score: {sentiment}")
    print(f"Total Headlines Returned: {len(headlines)}")
    print("-" * 30)
    for i, h in enumerate(headlines):
        print(f"{i+1}. {h}")
    print("-" * 30)
    
    # Check for ID keywords
    id_keywords = ["Indonesia", "Antam", "Rupiah", "IHSG", "Emas"]
    has_id = any(any(k.lower() in h.lower() for k in id_keywords) for h in headlines)
    
    if has_id:
        print("SUCCESS: Found Indonesian content.")
    else:
        print("WARNING: No explicit Indonesian keywords found in top 10 (might be chance).")

except Exception as e:
    print(f"ERROR: {e}")
