"""
Data collection module.

Handles fetching market data from yfinance, scraping physical gold prices,
and retrieving news sentiment from Google News RSS.
"""

import os
import re
import pickle
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

import feedparser
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from textblob import TextBlob

import config


def fetch_antam_price(
    force_refresh: bool = False,
    current_spot_price: Optional[float] = None
) -> Optional[int]:
    """
    Scrape daily Antam gold price from emasantam.id with caching and fallback.

    Args:
        force_refresh: Skip cache and fetch fresh data.
        current_spot_price: Current spot price in IDR/g for fallback calculation.

    Returns:
        The price per gram in IDR, or None if unavailable.
    """
    cache_path = os.path.join(config.MODELS_DIR, "last_antam_price.pkl")
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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        content = re.sub(r'\s+', ' ', soup.get_text())

        match = re.search(
            r'Harga Emas 1 gram.*?Rp\.?\s*([\d\.]+)',
            content,
            re.IGNORECASE
        )

        if match:
            price_str = match.group(1).replace('.', '')
            price = int(price_str)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temp_cache = cache_path + ".tmp"
            try:
                with open(temp_cache, 'wb') as f:
                    pickle.dump({'timestamp': time.time(), 'price': price}, f)
                os.replace(temp_cache, cache_path)
            except Exception:
                if os.path.exists(temp_cache):
                    os.remove(temp_cache)

            return price

    except Exception as e:
        print(f"Scraper failed: {e}. Using fallback.")

    # Fallback: use cached value or calculate from spot price
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f).get('price')
        except Exception:
            pass

    if current_spot_price:
        return int(current_spot_price * 1.11)  # Typical 11% spread

    return None


def fetch_market_data(period: str = "max") -> pd.DataFrame:
    """
    Fetch historical market data using yfinance.

    Args:
        period: Valid yfinance period (e.g., '1y', 'max').

    Returns:
        Combined dataframe of all tickers, or empty DataFrame if failed.
    """
    print("Fetching market data...")

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
            hist = ticker.history(period=p)

            if hist.empty:
                print(f"Warning: No data for {name} ({ticker_symbol})")
                continue

            df_ticker = hist[['Close']].rename(columns={'Close': name})

            if df_ticker.index.tz is not None:
                df_ticker.index = df_ticker.index.tz_localize(None)

            data_frames.append(df_ticker)

        except Exception as e:
            print(f"Error fetching {name}: {e}")

    if not data_frames:
        print("Error: No market data could be fetched.")
        return pd.DataFrame()

    try:
        market_data = pd.concat(data_frames, axis=1)
        market_data = market_data.ffill().dropna()
        return market_data
    except Exception as e:
        print(f"Error combining market data: {e}")
        return pd.DataFrame()


def update_local_database(
    csv_path: Optional[str] = None,
    force: bool = False
) -> pd.DataFrame:
    """
    Update the local CSV database with new daily data.

    Args:
        csv_path: Path to local CSV file. Defaults to config.CSV_PATH.
        force: If True, bypass the up-to-date check.

    Returns:
        Full updated dataframe.
    """
    if csv_path is None:
        csv_path = config.CSV_PATH

    print("\n[Database] Checking local history...")

    full_data = pd.DataFrame()

    if os.path.exists(csv_path):
        try:
            full_data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            last_date = full_data.index.max()
            print(f"  Found database. Last date: {last_date.date()}")

            today = pd.Timestamp.now().normalize()
            if not force and last_date >= (today - pd.Timedelta(days=1)):
                print("  Database is up to date.")
                return full_data

            start_date = last_date + pd.Timedelta(days=1)

            if start_date >= today:
                print("  Next date is today/future. Skipping.")
                return full_data

            print(f"  Fetching updates since {start_date.date()}...")
            new_data = _fetch_incremental_data(start_date)

            if not new_data.empty:
                if 'Gold' not in new_data.columns:
                    print("  Warning: New data missing 'Gold' column.")
                elif new_data['Gold'].isnull().all():
                    print("  Warning: 'Gold' column is all NaN.")
                else:
                    print(f"  Appending {len(new_data)} new rows.")
                    full_data = pd.concat([full_data, new_data])
                    full_data = full_data[~full_data.index.duplicated(keep='last')]
                    full_data.to_csv(csv_path)
                    print("  Database updated.")
            else:
                print("  No new data available.")

        except Exception as e:
            print(f"  Error reading database: {e}. Re-fetching all.")
            full_data = pd.DataFrame()

    if full_data.empty:
        print("  No local database found. Initializing...")
        full_data = fetch_market_data(period="max")
        if not full_data.empty and 'Gold' in full_data.columns:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            full_data.to_csv(csv_path)
            print("  Database created.")
        else:
            print("  Failed to fetch initial data.")

    return full_data


def _fetch_incremental_data(start_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch market data from a specific start date."""
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
        except Exception:
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
    """Fetch news from Google News RSS feed."""
    encoded_query = requests.utils.quote(query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}"
        f"+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    )

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
                'summary': getattr(entry, 'summary', '')
            })

    return articles


def fetch_news_sentiment(
    lookback_days: int = 30,
    force_refresh: bool = False
) -> Tuple[float, List[str], Dict[str, int]]:
    """
    Fetch news sentiment with 1-hour caching.

    Returns:
        Tuple of (average sentiment score, top headlines, sentiment counts).
    """
    cache_path = config.SENTIMENT_CACHE_PATH
    cache_expiry = 3600

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                cache_time = cache.get('timestamp', 0)
                if (time.time() - cache_time) < cache_expiry:
                    age = int(time.time() - cache_time)
                    print(f"  Using cached sentiment (Age: {age}s)")
                    return (
                        cache['avg_sentiment'],
                        cache['headlines'],
                        cache['sentiment_counts']
                    )
        except Exception as e:
            print(f"  Cache read failed: {e}")

    print("\n[Sentiment] Fetching news insights...")

    queries = [
        "Gold Price Forecast", "US Inflation Data", "Federal Reserve Rate Decisions",
        "Geopolitical Conflict Middle East", "China Gold Demand",
        "US Dollar Index Analysis", "Global Recession Risks", "Central Bank Gold Buying",
        "Harga Emas Antam Hari Ini", "Prediksi Harga Emas Indonesia",
        "Kurs Rupiah terhadap Dollar", "Kebijakan Suku Bunga Bank Indonesia",
        "Investasi Emas di Indonesia", "Inflasi Indonesia Terkini"
    ]

    all_articles = []
    id_keywords = ["Indonesia", "Antam", "Rupiah", "Bank Indonesia"]

    for q in queries:
        try:
            is_indo = any(k in q for k in id_keywords)
            lang = 'id-ID' if is_indo else 'en-US'
            region = 'ID' if is_indo else 'US'
            ceid = 'ID:id' if is_indo else 'US:en'

            encoded = requests.utils.quote(q)
            rss_url = (
                f"https://news.google.com/rss/search?q={encoded}"
                f"&hl={lang}&gl={region}&ceid={ceid}"
            )

            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:10]:
                dt = None
                if hasattr(entry, 'published_parsed'):
                    dt = datetime(*entry.published_parsed[:6])
                if dt:
                    all_articles.append({
                        'date': dt.date(),
                        'title': entry.title,
                        'summary': getattr(entry, 'summary', '')
                    })
        except Exception:
            continue

    # Fetch local Indonesian news
    try:
        print("  Fetching local news (CNBC Indonesia)...")
        id_news_url = "https://www.cnbcindonesia.com/market/emas"
        res = requests.get(id_news_url, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            articles = soup.select('.list article h2')
            for art in articles[:10]:
                title = art.get_text(strip=True)
                all_articles.append({'title': title, 'source': 'CNBC ID'})
    except Exception as e:
        print(f"  Local news fetch failed: {e}")

    # Deduplicate
    seen_titles = set()
    unique_articles = []
    for a in all_articles:
        if a['title'] not in seen_titles:
            unique_articles.append(a)
            seen_titles.add(a['title'])

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    avg_sentiment = 0.0
    headlines = []

    if unique_articles:
        print(f"  Analyzing {len(unique_articles)} articles...")
        try:
            if os.environ.get('USE_ADVANCED_NLP', 'true').lower() == 'true':
                from transformers import pipeline
                classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
            else:
                raise ImportError("Advanced NLP disabled.")

            scores = []
            for article in unique_articles:
                try:
                    result = classifier(article['title'][:512])[0]
                    label = result['label']
                    score = result['score']
                    sentiment_counts[label] += 1
                    val = score if label == 'positive' else (-score if label == 'negative' else 0)
                    scores.append(val)
                except Exception:
                    pass

            if scores:
                avg_sentiment = sum(scores) / len(scores)

        except Exception as e:
            print(f"FinBERT unavailable ({e}), using TextBlob...")
            scores = []
            for article in unique_articles:
                text = article['title']
                blob = TextBlob(text)
                score = blob.sentiment.polarity

                text_lower = text.lower()
                if any(k in text_lower for k in ['recession', 'crash', 'plunge', 'anjlok', 'jatuh']):
                    score -= 0.5
                if any(k in text_lower for k in ['soar', 'record', 'rally', 'melejit', 'rekor']):
                    score += 0.5

                scores.append(score)
                if score > 0.1:
                    sentiment_counts['positive'] += 1
                elif score < -0.1:
                    sentiment_counts['negative'] += 1
                else:
                    sentiment_counts['neutral'] += 1

            if scores:
                avg_sentiment = sum(scores) / len(scores)

        print(f"Analyzed {len(unique_articles)} items. Avg: {avg_sentiment:.4f}")
        headlines = [a['title'] for a in unique_articles[:5]]

    # Save cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        temp_cache = cache_path + ".tmp"
        with open(temp_cache, 'wb') as f:
            pickle.dump({
                'timestamp': time.time(),
                'avg_sentiment': avg_sentiment,
                'headlines': headlines,
                'sentiment_counts': sentiment_counts
            }, f)
        os.replace(temp_cache, cache_path)
    except Exception as e:
        print(f"  Cache save failed: {e}")

    return avg_sentiment, headlines, sentiment_counts


def fetch_live_data(ticker: str = 'GC=F') -> Optional[Dict[str, float]]:
    """Fetch the latest available intraday price."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1d", interval="1m")

        if df.empty:
            df = ticker_obj.history(period="5d")

        if df.empty:
            return None

        latest_price = df['Close'].iloc[-1]
        prev_close = ticker_obj.info.get('regularMarketPreviousClose')

        if not prev_close and len(df) > 1:
            prev_close = df['Close'].iloc[0]

        change_pct = ((latest_price - prev_close) / prev_close) * 100 if prev_close else 0.0

        return {'price': float(latest_price), 'change_pct': float(change_pct)}

    except Exception as e:
        print(f"Error fetching live data for {ticker}: {e}")
        return None


def fetch_market_snapshot() -> Dict[str, Any]:
    """Fetch a snapshot of major markets for the dashboard."""
    tickers = {
        'GOLD': 'GC=F', 'SILVER': 'SI=F', 'OIL': 'CL=F', 'DXY': 'DX-Y.NYB',
        'S&P500': '^GSPC', 'NASDAQ': '^IXIC', 'VIX': '^VIX', 'BITCOIN': 'BTC-USD',
        'USD/IDR': 'IDR=X', 'US10Y': '^TNX'
    }

    snapshot = {}
    for name, symbol in tickers.items():
        data = fetch_live_data(symbol)
        if data:
            snapshot[name] = data

    return snapshot
