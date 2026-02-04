"""
Data Collection Module for Gold Price Predictor.

Handles fetching market data from yfinance, scraping physical gold prices,
and retrieving news sentiment from Google News RSS.
"""

import os
import re
import pickle
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union, Any

import feedparser
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from textblob import TextBlob
from requests.exceptions import RequestException


# -----------------------------
# Physical Price Scraper
# -----------------------------
def fetch_antam_price(force_refresh: bool = False, current_spot_price: float = None) -> Optional[int]:
    """
    Scrapes daily Antam gold price from emasantam.id with caching and fallback.
    
    Args:
        force_refresh (bool): Skip cache.
        current_spot_price (float): Current Spot price in IDR/g to use as fallback.
    
    Returns:
        Optional[int]: The price per gram in IDR.
    """
    cache_path = "models/last_antam_price.pkl"
    cache_expiry = 21600  # 6 hours

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                if (time.time() - cache.get('timestamp', 0)) < cache_expiry:
                    return cache.get('price')
        except Exception:
            pass

    url = "https://emasantam.id/harga-emas-antam-harian/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # Short timeout to avoid hanging the dashboard
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text()
        content = re.sub(r'\s+', ' ', content)
        
        match = re.search(r'Harga Emas 1 gram.*?Rp\.?\s*([\d\.]+)', content, re.IGNORECASE)
        
        if match:
            price_str = match.group(1).replace('.', '')
            price = int(price_str)
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'timestamp': time.time(), 'price': price}, f)
            
            return price
            
    except Exception as e:
        print(f"Scraper failed: {e}. Using fallback.")
        
    # FALLBACK LOGIC: If scraper fails, use last cache OR Spot + 11% spread
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f).get('price')
        except Exception:
            pass
            
    if current_spot_price:
        # Typical Antam spread over spot is 10-12%
        return int(current_spot_price * 1.11)
    
    return None


def fetch_market_data(period: str = "max") -> pd.DataFrame:
    """
    Fetches historical market data using yfinance.

    Args:
        period (str): Valid yfinance period (e.g., '1y', 'max').

    Returns:
        pd.DataFrame: Combined dataframe of all tickers. Or empty if failed.
    """
    print("[1/5] Fetching Market Data...")
    print(f"Fetching market data for period: {period}...")

    tickers = {
        'Gold': 'GC=F',
        'USD_IDR': 'IDR=X',
        'DXY': 'DX-Y.NYB',
        'Oil': 'CL=F',
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'VIX': '^VIX',
        'GVZ': '^GVZ',
        'Silver': 'SI=F',
        'Copper': 'HG=F',
        'Platinum': 'PL=F',
        'Palladium': 'PA=F',
        'US10Y': '^TNX',
        'USD_CNY': 'CNY=X',
        'Nikkei': '^N225',
        'DAX': '^GDAXI'
    }

    data_frames = []

    for name, ticker_symbol in tickers.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            p = 'max' if (period == 'max' and name == 'Gold') else period
            
            # Auto-adjust period if specific ticker fails? yfinance handles this well usually.
            hist = ticker.history(period=p)

            if hist.empty:
                print(f"Warning: No data found for {name} ({ticker_symbol})")
                continue

            df_ticker = hist[['Close']].rename(columns={'Close': name})

            if df_ticker.index.tz is not None:
                df_ticker.index = df_ticker.index.tz_localize(None)

            data_frames.append(df_ticker)

        except Exception as e:
            print(f"Error fetching {name} ({ticker_symbol}): {e}")

    if not data_frames:
        print("Error: No market data could be fetched from any source.")
        return pd.DataFrame()

    try:
        market_data = pd.concat(data_frames, axis=1)
        market_data = market_data.ffill().dropna()
        return market_data
    except Exception as e:
        print(f"Error combining market data: {e}")
        return pd.DataFrame()


def update_local_database(csv_path: str = "data/gold_history.csv") -> pd.DataFrame:
    """
    Updates the local CSV database with new daily data.
    
    Checks last date in CSV and fetches incremental updates if valid.
    
    Args:
        csv_path (str): Path to local CSV file.
        
    Returns:
        pd.DataFrame: Full updated dataframe.
    """
    print("\n[Database] Checking for local history...")

    full_data = pd.DataFrame()

    if os.path.exists(csv_path):
        try:
            full_data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            last_date = full_data.index.max()
            print(f"      Found existing database. Last date: {last_date.date()}")
            
            # Check if up to date (last date is yesterday or today)
            today = pd.Timestamp.now().normalize()
            if last_date >= (today - pd.Timedelta(days=1)):
                print("      Database is up to date (Last date is yesterday/today).")
                return full_data
            
            start_date = last_date + pd.Timedelta(days=1)
            
            # Prevent fetching future or incomplete 'today'
            if start_date >= today:
                 print("      Next start date is today/future. Skipping until market close.")
                 return full_data

            print(f"      Fetching updates since {start_date.date()}...")
            new_data = _fetch_incremental_data(start_date)

            if not new_data.empty:
                # Validation
                if 'Gold' not in new_data.columns:
                     print("      Warning: New data missing 'Gold' column. Aborting update.")
                elif new_data['Gold'].isnull().all():
                     print("      Warning: New data has 'Gold' column but all values are NaN. Aborting.")
                else:
                    print(f"      Append {len(new_data)} new rows.")
                    full_data = pd.concat([full_data, new_data])
                    # Remove duplicates just in case
                    full_data = full_data[~full_data.index.duplicated(keep='last')]
                    full_data.to_csv(csv_path)
                    print("      Database updated and saved.")
            else:
                print("      No new data available from markets.")

        except Exception as e:
            print(f"      Error reading/updating local DB: {e}. Re-fetching all.")
            full_data = pd.DataFrame()

    if full_data.empty:
        print("      No local database found or corrupted. Initializing...")
        full_data = fetch_market_data(period="max")
        if not full_data.empty and 'Gold' in full_data.columns:
            full_data.to_csv(csv_path)
            print("      Database created.")
        elif full_data.empty:
             print("      Failed to fetch initial data.")

    return full_data


def _fetch_incremental_data(start_date: pd.Timestamp) -> pd.DataFrame:
    """
    Helper to fetch data from a specific start date.
    
    Args:
        start_date (pd.Timestamp): Start date for yfinance query.
        
    Returns:
        pd.DataFrame: Incremental data.
    """
    tickers = {
        'Gold': 'GC=F', 'USD_IDR': 'IDR=X', 'DXY': 'DX-Y.NYB', 'Oil': 'CL=F',
        'SP500': '^GSPC', 'NASDAQ': '^IXIC', 'VIX': '^VIX', 'GVZ': '^GVZ',
        'Silver': 'SI=F', 'Copper': 'HG=F', 'Platinum': 'PL=F', 'Palladium': 'PA=F',
        'US10Y': '^TNX', 'USD_CNY': 'CNY=X', 'Nikkei': '^N225', 'DAX': '^GDAXI'
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
        except Exception as e:
            # Silent fail for individual tickers in incremental update is okay
            pass

    if not data_frames:
        return pd.DataFrame()

    try:
        new_data = pd.concat(data_frames, axis=1)
        new_data = new_data.ffill().dropna()
        return new_data
    except Exception:
        return pd.DataFrame()


def fetch_google_news_rss(query: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch news from Google News RSS feed.
    
    Args:
        query (str): Search query.
        days (int): Lookback window in days.
        
    Returns:
        List[Dict[str, Any]]: List of article dictionaries.
    """
    encoded_query = requests.utils.quote(query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )

    # print(f"Fetching news for query: '{query}'...")

    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"Error fetching RSS: {e}")
        return []

    articles = []

    for entry in feed.entries:
        dt = None
        if hasattr(entry, 'published_parsed'):
            try:
                dt = datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        if dt:
            articles.append({
                'date': dt.date(),
                'title': entry.title,
                'summary': entry.summary if hasattr(entry, 'summary') else ''
            })

    return articles


def fetch_news_sentiment(
    lookback_days: int = 30, 
    force_refresh: bool = False
) -> Tuple[float, List[str], Dict[str, int]]:
    """
    Fetches news sentiment with 1-hour caching.
    
    Args:
        lookback_days (int): Ignored currently (uses fixed query params).
        force_refresh (bool): Ignore cache if True.
        
    Returns:
        Tuple[float, List[str], Dict[str, int]]: 
            Average sentiment score, Top headlines, Sentiment breakdown count.
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
                    return (
                        cache['avg_sentiment'], 
                        cache['headlines'], 
                        cache['sentiment_counts']
                    )
        except Exception as e:
            print(f"      Cache read failed: {e}")

    # 2. Real Fetch
    print("\n[Sentiment] Fetching fresh news insights (this may take a moment)...")
    
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
            
            encoded = requests.utils.quote(q)
            rss_url = f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={region}&ceid={ceid}"
            
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:10]:
                dt = None
                if hasattr(entry, 'published_parsed'):
                     dt = datetime(*entry.published_parsed[:6])
                
                if dt:
                    all_articles.append({
                        'date': dt.date(),
                        'title': entry.title,
                        'summary': entry.summary if hasattr(entry, 'summary') else ''
                    })
                    
        except Exception:
            # Skip individual query failures
            continue

    # --- Sentiment 2.0: Indonesian News Scraper ---
    try:
        print("      Fetching local news (CNBC Indonesia)...")
        id_news_url = "https://www.cnbcindonesia.com/market/emas"
        res = requests.get(id_news_url, timeout=5)
        if res.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(res.text, 'html.parser')
            # Look for article titles in the list
            articles = soup.select('.list article h2')
            for art in articles[:10]:
                title = art.get_text(strip=True)
                all_articles.append({'title': title, 'source': 'CNBC ID'})
    except Exception as e:
        print(f"      Warning: Local news fetch failed ({e})")
    # ---------------------------------------------

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for a in all_articles:
        if a['title'] not in seen_titles:
            unique_articles.append(a)
            seen_titles.add(a['title'])
            
    # Process Sentiment
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    avg_sentiment = 0.0
    headlines = []

    if unique_articles:
        print(f"      Analyzing {len(unique_articles)} news articles...")
        try:
            from transformers import pipeline
            # Suppress logs from transformers?
            classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

            scores = []
            for article in unique_articles:
                try:
                    # Truncate to 512 tokens approx
                    result = classifier(article['title'][:512])[0]
                    label = result['label']
                    score = result['score']

                    sentiment_counts[label] += 1

                    val = 0.0
                    if label == 'positive':
                        val = score
                    elif label == 'negative':
                        val = -score
                    
                    scores.append(val)
                except Exception:
                    pass

            if scores:
                avg_sentiment = sum(scores) / len(scores)

        except Exception as e:
            print(f"Warning: FinBERT failed ({e}), falling back to TextBlob...")
            # Fallback Logic
            scores = []
            for article in unique_articles:
                text = article['title']
                blob = TextBlob(text)
                score = blob.sentiment.polarity
                
                # Simple keyword weighting
                text_lower = text.lower()
                if any(k in text_lower for k in ['recession', 'crash', 'plunge', 'anjlok', 'jatuh', 'lemah']):
                     score -= 0.5
                if any(k in text_lower for k in ['soar', 'record', 'rally', 'melejit', 'rekor', 'terbang']):
                     score += 0.5
                     
                scores.append(score)
                
                if score > 0.1: sentiment_counts['positive'] += 1
                elif score < -0.1: sentiment_counts['negative'] += 1
                else: sentiment_counts['neutral'] += 1

            if scores:
                avg_sentiment = sum(scores) / len(scores)

        print(f"Analyzed {len(unique_articles)} news items. Avg Sentiment: {avg_sentiment:.4f}")
        headlines = [a['title'] for a in unique_articles[:5]]

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


def fetch_live_data(ticker: str = 'GC=F') -> Optional[Dict[str, float]]:
    """
    Fetches the latest available intraday price (1-minute interval) 
    to simulate real-time updates.

    Args:
        ticker (str): Ticker symbol (default Gold Futures 'GC=F').

    Returns:
        Optional[Dict[str, float]]: Dictionary with 'price' and 'change_pct',
                                    or None if fetch fails.
    """
    try:
        # Fetch 1-day history with 1-minute interval to get latest candle
        # 'period=1d' might be empty if market is closed, so we use '5d' to be safe
        # but '1m' interval is usually limited to 7 days max.
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1d", interval="1m")
        
        if df.empty:
            # Fallback to daily if minute data is unavailable (market closed/weekend)
            df = ticker_obj.history(period="5d")
            
        if df.empty:
            return None
            
        latest_price = df['Close'].iloc[-1]
        
        # Calculate change (vs previous close, or previous minute?)
        # For "Realtime" feel, we want change vs yesterday's close usually.
        # fast_info usually has 'regularMarketPreviousClose'
        prev_close = ticker_obj.info.get('regularMarketPreviousClose')
        
        # Fallback if info is missing (common with yfinance sometimes)
        if not prev_close and len(df) > 1:
             prev_close = df['Close'].iloc[0] # Open of the day roughly
             
        if prev_close:
            change_pct = ((latest_price - prev_close) / prev_close) * 100
        else:
            change_pct = 0.0
            
        return {
            'price': latest_price,
            'change_pct': change_pct
        }
        
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None
