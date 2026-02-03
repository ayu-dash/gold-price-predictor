"""
Data Collection Module for Gold Price Predictor.
Handles fetching market data from yfinance and news sentiment from RSS/FinBERT.
"""

import os
import re
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from textblob import TextBlob
import pickle
import time





# -----------------------------
# Physical Price Scraper
# -----------------------------
def fetch_antam_price():
    """
    Scrapes daily Antam gold price from emasantam.id.
    Returns integer price (e.g., 3027000) or None if failed.
    """
    url = "https://emasantam.id/harga-emas-antam-harian/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()
        content = re.sub(r'\s+', ' ', content)
        
        # Pattern match specifically for "Harga Emas 1 gram" followed by price
        match = re.search(r'Harga Emas 1 gram.*?Rp\.?\s*([\d\.]+)', content, re.IGNORECASE)
        
        if match:
            price_str = match.group(1).replace('.', '')
            return int(price_str)
            
    except Exception as e:
        print(f"Error fetching Antam price: {e}")
    
    return None

def fetch_market_data(period="max"):
    """
    Fetches historical market data using yfinance.
    """
    print("[1/5] Fetching Market Data...")
    print(f"Fetching market data for period: {period}...")

    tickers = {
        'Gold': 'GC=F',
        'USD_IDR': 'IDR=X',
        'DXY': 'DX-Y.NYB',
        'Oil': 'CL=F',
        'SP500': '^GSPC',
        'VIX': '^VIX',
        'GVZ': '^GVZ',
        'Silver': 'SI=F',
        'Copper': 'HG=F',
        'US10Y': '^TNX',
        'Nikkei': '^N225',
        'DAX': '^GDAXI'
    }

    data_frames = []

    for name, ticker_symbol in tickers.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            p = 'max' if (period == 'max' and name == 'Gold') else period
            hist = ticker.history(period=p)

            if hist.empty:
                print(f"Warning: No data found for {name} ({ticker_symbol})")
                continue

            df_ticker = hist[['Close']].rename(columns={'Close': name})

            if df_ticker.index.tz is not None:
                df_ticker.index = df_ticker.index.tz_localize(None)

            data_frames.append(df_ticker)

        except Exception as e:
            print(f"Error fetching {name}: {e}")

    if not data_frames:
        raise ValueError("No market data could be fetched.")

    market_data = pd.concat(data_frames, axis=1)
    market_data = market_data.ffill().dropna()

    return market_data


def update_local_database(csv_path="gold_history.csv"):
    """
    Updates the local CSV database with new daily data.
    """
    print("\n[Database] Checking for local history...")

    full_data = pd.DataFrame()

    if os.path.exists(csv_path):
        try:
            full_data = pd.read_csv(csv_path, index_col='Date',
                                    parse_dates=True)
            last_date = full_data.index.max()
            print(f"      Found existing database. Last date: {last_date.date()}")

            start_date = last_date + pd.Timedelta(days=1)

            if start_date > pd.Timestamp.now():
                print("      Database is up to date.")
                return full_data

            print(f"      Fetching updates since {start_date.date()}...")
            new_data = _fetch_incremental_data(start_date)

            if not new_data.empty:
                print(f"      Append {len(new_data)} new rows.")
                full_data = pd.concat([full_data, new_data])
                full_data = full_data[~full_data.index.duplicated(keep='last')]
                full_data.to_csv(csv_path)
                print("      Database updated and saved.")
            else:
                print("      No new data available from markets.")

        except Exception as e:
            print(f"      Error reading local DB: {e}. Re-fetching all.")
            full_data = pd.DataFrame()

    if full_data.empty:
        print("      No local database found. Initializing...")
        full_data = fetch_market_data(period="max")
        if not full_data.empty:
            full_data.to_csv(csv_path)
            print("      Database created.")

    return full_data


def _fetch_incremental_data(start_date):
    """
    Helper to fetch data from a specific start date.
    """
    tickers = {
        'Gold': 'GC=F', 'USD_IDR': 'IDR=X', 'DXY': 'DX-Y.NYB', 'Oil': 'CL=F',
        'SP500': '^GSPC', 'VIX': '^VIX', 'GVZ': '^GVZ',
        'Silver': 'SI=F', 'Copper': 'HG=F',
        'US10Y': '^TNX', 'Nikkei': '^N225', 'DAX': '^GDAXI'
    }

    data_frames = []
    start_str = start_date.strftime('%Y-%m-%d')

    for name, ticker_symbol in tickers.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start_str)

            if not hist.empty:
                df = hist[['Close']].rename(columns={'Close': name})
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                data_frames.append(df)
        except Exception:
            pass

    if not data_frames:
        return pd.DataFrame()

    new_data = pd.concat(data_frames, axis=1)
    new_data = new_data.ffill().dropna()
    return new_data




def fetch_google_news_rss(query, days=7):
    """
    Fetch news from Google News RSS feed.
    """
    encoded_query = requests.utils.quote(query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )

    print(f"Fetching news for query: '{query}'...")

    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"Error fetching RSS: {e}")
        return []

    articles = []

    for entry in feed.entries:
        dt = None
        if hasattr(entry, 'published_parsed'):
            dt = datetime(*entry.published_parsed[:6])

        if dt:
            articles.append({
                'date': dt.date(),
                'title': entry.title,
                'summary': entry.summary if hasattr(entry, 'summary') else ''
            })

    return articles


def fetch_news_sentiment(lookback_days=30, force_refresh=False):
    """
    Fetches news sentiment with 1-hour caching.
    """
    cache_path = "models/last_sentiment.pkl"
    cache_expiry = 3600  # 1 hour

    # 1. Check Cache
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                cache_time = cache.get('timestamp', 0)
                if (time.time() - cache_time) < cache_expiry:
                    print(f"      Using cached sentiment (Age: {int(time.time() - cache_time)}s)")
                    return cache['avg_sentiment'], cache['headlines'], cache['sentiment_counts']
        except Exception as e:
            print(f"      Cache read failed: {e}")

    # 2. Real Fetch
    print("\n[Sentiment] Fetching fresh news insights...")
    
    # Expanded Queries (Global & Local)
    queries = [
        # GLOBAL (English)
        "Gold Price Forecast", "US Inflation Data", "Federal Reserve Rate Decisions",
        "Geopolitical Conflict Middle East", "China Gold Demand", "US Dollar Index Analysis",
        "Global Recession Risks", "Central Bank Gold Buying",
        
        # INDONESIA (Bahasa)
        "Harga Emas Antam Hari Ini", "Prediksi Harga Emas Indonesia",
        "Kurs Rupiah terhadap Dollar", "Kebijakan Suku Bunga Bank Indonesia",
        "Investasi Emas di Indonesia", "Inflasi Indonesia Terkini"
    ]

    all_articles = []
    
    # ID Keywords check
    id_keywords = ["Indonesia", "Antam", "Rupiah", "Bank Indonesia"]

    for q in queries:
        try:
            # Determine region
            is_indo = any(k in q for k in id_keywords)
            lang = 'id-ID' if is_indo else 'en-US'
            region = 'ID' if is_indo else 'US'
            ceid = 'ID:id' if is_indo else 'US:en'
            
            # Custom fetch for localization
            encoded = requests.utils.quote(q)
            rss_url = f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={region}&ceid={ceid}"
            
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:10]: # Increased limit to 10 per query (User Request)
                dt = None
                if hasattr(entry, 'published_parsed'):
                    dt = datetime(*entry.published_parsed[:6])
                
                if dt:
                    all_articles.append({
                        'date': dt.date(),
                        'title': entry.title,
                        'summary': entry.summary if hasattr(entry, 'summary') else ''
                    })
                    
        except Exception as e:
            print(f"Failed to fetch {q}: {e}")

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for a in all_articles:
        if a['title'] not in seen_titles:
            unique_articles.append(a)
            seen_titles.add(a['title'])
            
    # Process Sentiment
    sentiment_score = 0
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    print(f"      Analyzing {len(unique_articles)} news articles...")

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    try:
        from transformers import pipeline
        classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

        scores = []
        for article in all_articles:
            result = classifier(article['title'][:512])[0]
            label = result['label']
            score = result['score']

            sentiment_counts[label] += 1

            val = 0
            if label == 'positive':
                val = score
            elif label == 'negative':
                val = -score

            scores.append(val)

        avg_sentiment = sum(scores) / len(scores) if scores else 0

    except Exception as e:
        print(f"Warning: FinBERT failed ({e}), falling back to TextBlob...")
        df = pd.DataFrame(all_articles)

        def get_weighted_sentiment(text):
            blob = TextBlob(text)
            score = blob.sentiment.polarity
            text_lower = text.lower()

            keywords_en = ['recession', 'crash', 'soar', 'record',
                           'plunge', 'crisis']
            if any(k in text_lower for k in keywords_en):
                score *= 1.5

            keywords_id_neg = ['anjlok', 'resesi', 'krisis', 'turun',
                               'melemah', 'rugi']
            keywords_id_pos = ['naik', 'menguat', 'untung', 'rekor', 'bullish']

            if any(k in text_lower for k in keywords_id_neg):
                score -= 0.5
            if any(k in text_lower for k in keywords_id_pos):
                score += 0.5

            return score

        df['sentiment'] = df['title'].apply(get_weighted_sentiment)
        avg_sentiment = df['sentiment'].mean()

        sentiment_counts['positive'] = len(df[df['sentiment'] > 0.1])
        sentiment_counts['negative'] = len(df[df['sentiment'] < -0.1])
        sentiment_counts['neutral'] = (
            len(df) - sentiment_counts['positive'] -
            sentiment_counts['negative']
        )

    print(f"Analyzed {len(all_articles)} news items. "
          f"Average Sentiment: {avg_sentiment:.4f}")

    headlines = [a['title'] for a in all_articles[:5]]

    # 3. Save Cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'timestamp': time.time(),
                'avg_sentiment': avg_sentiment,
                'headlines': headlines,
                'sentiment_counts': sentiment_counts
            }, f)
    except Exception as e:
        print(f"      Cache save failed: {e}")

    return avg_sentiment, headlines, sentiment_counts
