"""
Web API and Dashboard for Gold Price Predictor.
Serves prediction and historical data via Flask.
"""

import os
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.data import loader
from src.features import engineering
from src.models import predictor

app = Flask(__name__)
CORS(app)

# Path to history and model
CSV_PATH = "gold_history.csv"
MODEL_PATH = "models/gold_model.pkl"

# Global model variable
trained_model = None


import sys
import subprocess

def get_model():
    """Lazy loads or returns the global trained model. Auto-retrains on failure."""
    global trained_model
    if trained_model is None:
        print(f"Loading model from {MODEL_PATH}...")
        try:
            trained_model = predictor.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model ({e}). Version mismatch likely. Retraining...")
            try:
                # Remove corrupt model
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                
                # Retrain
                subprocess.run([sys.executable, "main.py", "--days", "1"], check=True)
                trained_model = predictor.load_model(MODEL_PATH)
            except Exception as e2:
                print(f"Critical Error: Failed to retrain model: {e2}")

        if trained_model is None:
            print("Warning: No model found even after checks. Please run main.py peridocially.")
    return trained_model


@app.route("/")
def index():
    """Serves the dashboard home page."""
    return render_template("index.html")


@app.route("/api/history")
def get_history():
    """Returns historical price data for charting."""
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
    recent_df = df.tail(100)

    data = {
        "dates": recent_df.index.strftime('%Y-%m-%d').tolist(),
        "prices": recent_df['Gold'].tolist(),
        "idr_rate": recent_df['USD_IDR'].tolist()
    }
    return jsonify(data)


@app.route("/api/prediction")
def get_prediction():
    """Runs prediction logic using pre-trained model."""
    model_obj = get_model()
    if model_obj is None:
        return jsonify({"error": "Model not trained yet"}), 503

    # 1. Fetch latest data (Market only, sentiment is real-time)
    market_data = loader.update_local_database()

    # 2. Features
    df = engineering.add_technical_indicators(market_data)
    sentiment, headlines, sentiment_breakdown = loader.fetch_news_sentiment()

    df['Sentiment'] = sentiment
    
    features = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'US10Y', 'Nikkei', 'DAX',
        'SMA_14', 'RSI', 'MACD', 'Sentiment'
    ]
    available_features = [f for f in features if f in df.columns]

    # Predict
    latest_row = df[available_features].iloc[[-1]]
    predicted_return = model_obj.predict(latest_row)[0]

    current_usd = market_data['Gold'].iloc[-1]
    current_idr_rate = market_data['USD_IDR'].iloc[-1]
    predicted_usd = current_usd * (1 + predicted_return)

    # Signal
    rec, _ = predictor.make_recommendation(current_usd, predicted_usd)

    # Local Specs
    grams_per_oz = 31.1035
    price_gram_idr = (current_usd * current_idr_rate) / grams_per_oz
    pred_gram_idr = (predicted_usd * current_idr_rate) / grams_per_oz

    # Calculate daily change (historical)
    prev_usd = market_data['Gold'].iloc[-2] if len(market_data) > 1 else current_usd
    daily_change_pct = ((current_usd - prev_usd) / prev_usd) * 100

    # Fetch Physical Price (Antam) - Real Data (User Request: No Mock)
    physical_price = loader.fetch_antam_price()
    # physical_price = None

    # Estimated Retail Price REMOVED (Was mock/calculation)
    # est_retail_price = price_gram_idr * 1.08
    
    result = {
        "current_price_usd": round(current_usd, 2),
        "predicted_price_usd": round(predicted_usd, 2),
        "current_price_idr_gram": round(price_gram_idr, 0),
        "physical_price_idr": physical_price, 
        # "est_retail_price_idr": round(est_retail_price, 0), # Removed
        "predicted_price_idr_gram": round(pred_gram_idr, 0),
        "change_pct": round(predicted_return * 100, 2),
        "daily_change_pct": round(daily_change_pct, 2), # Actual Today's Change
        "recommendation": rec,
        "sentiment_score": round(sentiment, 4),
        "sentiment_breakdown": sentiment_breakdown,
        "action_date": (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime(
            '%Y-%m-%d'),
        "top_headlines": headlines
    }

    return jsonify(result)

    return jsonify(result)


@app.route("/api/live_price")
def get_live_price():
    """Returns realtime price update (1-min delay)."""
    # 1. Fetch Gold Price
    live_gold = loader.fetch_live_data('GC=F')
    if not live_gold:
        return jsonify({"error": "Failed to fetch live data"}), 503

    # 2. Fetch USD/IDR Rate (also live if possible, or fallback)
    live_idr = loader.fetch_live_data('IDR=X')
    rate = live_idr['price'] if live_idr else 15500.0 # Logical fallback
    
    current_usd = live_gold['price']
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Conversion
    grams_per_oz = 31.1035
    price_gram_idr = (current_usd * rate) / grams_per_oz
    
    return jsonify({
        "timestamp": current_time,
        "price_usd": round(current_usd, 2),
        "price_idr_gram": round(price_gram_idr, 0),
        "change_pct": round(live_gold['change_pct'], 2),
        "rate": round(rate, 2)
    })
@app.route("/api/forecast")
def get_forecast():
    """Generates a recursive multi-day forecast using pre-trained model."""
    model_obj = get_model()
    if model_obj is None:
        return jsonify({"error": "Model not trained yet"}), 503

    days = request.args.get('days', default=7, type=int)

    market_data = loader.update_local_database()
    df = engineering.add_technical_indicators(market_data)
    sentiment, _, _ = loader.fetch_news_sentiment()
    df['Sentiment'] = sentiment

    features = [
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'US10Y', 'Nikkei', 'DAX',
        'SMA_14', 'RSI', 'MACD', 'Sentiment'
    ]
    available_features = [f for f in features if f in df.columns]

    latest_features = df[available_features].iloc[[-1]]
    current_usd = market_data['Gold'].iloc[-1]
    current_idr_rate = market_data['USD_IDR'].iloc[-1]

    # Ambil data history (misal 100 hari terakhir) untuk buffer perhitungan indikator
    history_buffer = df.tail(100).copy()

    forecast_data = predictor.recursive_forecast(
        model_obj, latest_features, current_usd, current_idr_rate, days=days,
        historical_df=history_buffer
    )

    grams_per_oz = 31.1035
    formatted = []
    for f in forecast_data:
        formatted.append({
            "date": f['Date'],
            "price_idr": round(f['Price_IDR'] / grams_per_oz, 0),
            "change_pct": f['Return_Pct']
        })

    return jsonify(formatted)


@app.route("/analytics")
def analytics():
    """Serves the yearly analytics page."""
    return render_template("analytics.html")


@app.route("/charts")
def charts():
    """Serves the detailed charts page."""
    return render_template("charts.html")


@app.route("/signals")
def signals():
    """Serves the signals analysis page."""
    return render_template("signals.html")


@app.route("/api/yearly_stats")
def get_yearly_stats():
    """Aggregates gold data by year for performance comparison."""
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
    
    # Calculate IDR price per gram if not present
    grams_per_oz = 31.1035
    df['Price_IDR_g'] = (df['Gold'] * df['USD_IDR']) / grams_per_oz
    
    # Group by year
    yearly = df.resample('YE').agg({
        'Price_IDR_g': ['first', 'last', 'max', 'min']
    })
    
    yearly.columns = ['open', 'close', 'high', 'low']
    yearly = yearly.dropna()
    
    result = []
    for year, row in yearly.iterrows():
        roi = ((row['close'] - row['open']) / row['open']) * 100
        result.append({
            "year": year.year,
            "open": round(row['open'], 0),
            "close": round(row['close'], 0),
            "high": round(row['high'], 0),
            "low": round(row['low'], 0),
            "roi_pct": round(roi, 2)
        })
        
    return jsonify(result)


@app.route("/api/technical")
def get_technical():
    """Returns detailed technical indicators for charts."""
    if not os.path.exists(CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
    df = engineering.add_technical_indicators(df)
    
    recent_df = df.tail(60) # Last 60 trading days
    
    data = {
        "dates": recent_df.index.strftime('%Y-%m-%d').tolist(),
        "prices": recent_df['Gold'].tolist(),
        "rsi": recent_df['RSI'].tolist(),
        "macd": recent_df['MACD'].tolist(),
        "macd_signal": recent_df['MACD_Signal'].tolist(),
        "sma_50": recent_df['SMA_50'].tolist(),
        "sma_200": recent_df['SMA_200'].tolist()
    }
    return jsonify(data)



# ----------------------------------------------------
# Background Scheduler
# ----------------------------------------------------
import threading
import subprocess
import time
import sys

UPDATE_INTERVAL = 43200  # 12 Hours

def run_periodic_update():
    """Reads main.py periodically to update model and data."""
    while True:
        print(f"\n[Scheduler] Usage monitoring... Next update in {UPDATE_INTERVAL/3600} hours.")
        time.sleep(UPDATE_INTERVAL)
        
        print("\n[Scheduler] Starting periodic model update (main.py)...")
        try:
            # Run main.py in a separate process
            # Pass --days 1 to ensure it runs non-interactively
            result = subprocess.run(
                [sys.executable, "main.py", "--days", "1"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[Scheduler] Update completed successfully.")
                print(f"[Scheduler] Output Summary: {result.stdout[-200:]}") # Log last bit
                
                # Invalidate global model cache to force reload on next request
                global trained_model
                trained_model = None
                print("[Scheduler] Global model cache cleared. New model will be loaded on next request.")
            else:
                print("[Scheduler] Update FAILED.")
                print(f"[Scheduler] Error: {result.stderr}")
                
        except Exception as e:
            print(f"[Scheduler] System Error: {e}")

def start_scheduler():
    """Starts the background thread."""
    thread = threading.Thread(target=run_periodic_update, daemon=True)
    thread.start()
    print("[Scheduler] Background service started.")

# Helper function for history data
def fetch_and_prepare_data():
    """Fetches and prepares historical data from CSV."""
    if not os.path.exists(CSV_PATH):
        return None, jsonify({"error": "No history found"}), 404

    df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
    df = engineering.add_technical_indicators(df)
    return df, None

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/api/history/full')
def get_full_history():
    """Returns complete historical data for the table view."""
    df, error_response = fetch_and_prepare_data()
    if df is None:
        return error_response
    
    # Calculate daily change properly on ascending data first
    if 'Gold_Returns' in df.columns:
        df['Daily_Return'] = df['Gold_Returns'] * 100
    else:
        df['Daily_Return'] = df['Gold'].pct_change() * 100

    # Sort by date descending
    df_sorted = df.sort_index(ascending=False).copy()

    dates = df_sorted.index.strftime('%Y-%m-%d').tolist()
    prices_usd = df_sorted['Gold'].tolist()
    rates = df_sorted['USD_IDR'].tolist()
    daily_returns = df_sorted['Daily_Return'].fillna(0).tolist()
    
    # Calculate IDR prices per gram
    grams_per_oz = 31.1035
    prices_idr = [(p * r) / grams_per_oz for p, r in zip(prices_usd, rates)]
    
    # We can pass simple records
    # The original snippet had a commented-out loop and then a new loop.
    # I'll use the second, correct loop.
    data = []
    for date, usd, rate, idr, change in zip(dates, prices_usd, rates, prices_idr, daily_returns):
        data.append({
            'date': date,
            'price_usd': round(usd, 2),
            'price_idr': round(idr, 0),
            'rate': round(rate, 2),
            'change_pct': round(change, 2)
        })

    return jsonify(data)

if __name__ == "__main__":
    # Ensure scheduler only runs in the reloader process (not the main watcher)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_scheduler()
    
    app.run(debug=True, port=5000)
