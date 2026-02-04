"""
Web API and Dashboard for Gold Price Predictor.
Serves prediction and historical data via Flask.
"""

import os
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.data import loader, signal_logger
from src.features import engineering
from src.models import predictor

app = Flask(__name__)
CORS(app)

# Path to history and model
CSV_PATH = "data/gold_history.csv"
MODEL_PATH = "models/gold_model.pkl"

# Global model variable
trained_model = None

import threading
get_model_lock = threading.Lock()

import sys
import subprocess

def get_model():
    """Lazy loads or returns the global trained model ensemble. Auto-retrains on failure."""
    global trained_model
    
    # Quick check without lock
    if trained_model is not None:
        return trained_model

    with get_model_lock:
        # Double-check inside lock
        if trained_model is not None:
            return trained_model
            
        paths = {
            'med': "models/gold_model_med.pkl",
            'low': "models/gold_model_low.pkl",
            'high': "models/gold_model_high.pkl"
        }
        
        try:
            ensemble = {}
            for q, p in paths.items():
                if not os.path.exists(p):
                    # Fallback to legacy path if specific quantile not found
                    ensemble[q] = predictor.load_model("models/gold_model.pkl")
                else:
                    ensemble[q] = predictor.load_model(p)
            
            if all(m is not None for m in ensemble.values()):
                # Load Classifier
                clf = predictor.load_model("models/gold_classifier.pkl")
                if clf:
                    ensemble['clf'] = clf
                    trained_model = ensemble
                    print("Model ensemble + Classifier loaded successfully.")
                else:
                    print("Warning: Classifier not found. Continuing with Regressors only.")
                    trained_model = ensemble
            else:
                raise ValueError("Some models failed to load.")
                
        except Exception as e:
            print(f"Error loading ensemble ({e}). Retraining...")
            try:
                subprocess.run([sys.executable, "main.py", "--days", "1"], check=True)
                # Retry loading
                return get_model() 
            except Exception as e2:
                print(f"Critical Error: Failed to retrain model: {e2}")
    
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
    recent_df = df.tail(100).copy()

    # --- REALTIME CHART UPDATE ---
    # Fetch live price to append as the final point
    live_gold = loader.fetch_live_data('GC=F')
    if live_gold:
        # Create a new index for "Now"
        now_idx = pd.Timestamp.now().normalize()
        
        # If the last date in history is NOT today (or if it is, we overwrite/update it?)
        # For chart continuity, if today exists, we replace it. If not, we append.
        if recent_df.index[-1].normalize() == now_idx:
            # Update last row
            recent_df.iloc[-1, recent_df.columns.get_loc('Gold')] = live_gold['price']
        else:
            # Append new row
            new_row = pd.DataFrame(
                {'Gold': [live_gold['price']], 'USD_IDR': [recent_df['USD_IDR'].iloc[-1]]}, 
                index=[now_idx]
            )
            # Fill missing columns with ffill/bfill or ignore if only Gold is needed for chart
            recent_df = pd.concat([recent_df, new_row])
            
    # -----------------------------

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
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
    ]
    available_features = [f for f in features if f in df.columns]

    # Fetch LIVE data for actual current price
    live_gold = loader.fetch_live_data('GC=F')
    live_idr = loader.fetch_live_data('IDR=X')
    
    current_usd = live_gold['price'] if live_gold else market_data['Gold'].iloc[-1]
    current_idr_rate = live_idr['price'] if live_idr else market_data['USD_IDR'].iloc[-1]

    # --- DATA LAG FIX: Inject live data as Today's row ---
    if live_gold and live_idr:
        new_row = df.iloc[[-1]].copy()
        new_row.index = [pd.Timestamp.now().normalize()]
        new_row['Gold'] = current_usd
        new_row['USD_IDR'] = current_idr_rate
        
        # Add slight jitter for macro context
        for feat in ['DXY', 'Oil', 'SP500', 'Silver']:
            if feat in new_row.columns:
                new_row[feat] = new_row[feat] * (1 + (live_gold.get('change_pct', 0)/200))
        
        df = pd.concat([df, new_row])
        df = engineering.add_technical_indicators(df)
        latest_row = df[available_features].iloc[[-1]]
    else:
        latest_row = df[available_features].iloc[[-1]]

    # Predict
    # model_obj is now a dict {'low':..., 'med':..., 'high':...}
    if isinstance(model_obj, dict):
        predicted_return = model_obj['med'].predict(latest_row)[0]
    else:
        predicted_return = model_obj.predict(latest_row)[0]
        
    predicted_usd = current_usd * (1 + predicted_return)

    predicted_usd = current_usd * (1 + predicted_return)

    # Classification (Confidence via Quantile Regression)
    # We use the spread between Low (5%) and High (95%) to estimate probability of > 0
    import scipy.stats as stats
    
    pred_med = predicted_return
    pred_low = 0.0
    pred_high = 0.0
    
    if isinstance(model_obj, dict):
        pred_low = model_obj['low'].predict(latest_row)[0]
        pred_high = model_obj['high'].predict(latest_row)[0]
    else:
        # Fallback if no ensemble
        pred_low = pred_med - 0.01 
        pred_high = pred_med + 0.01

    conf_direction = "UP" if pred_med > 0 else "DOWN"
    
    # Calculate Z-score for 0
    # spread = (High - Low) covers 90% confidence (approx 3.29 sigmas)
    spread = pred_high - pred_low
    if spread <= 0: spread = 0.0001 # Avoid div by zero
    
    sigma = spread / 3.29
    z_score = abs(pred_med) / sigma
    
    # Probability that the true value is on the same side of 0 as the median
    # CDF(z_score) gives 0.5 to 1.0
    probability = stats.norm.cdf(z_score)
    
    conf_score = round(probability * 100, 1)
    
    # Log debug
    print(f"DEBUG: Med={pred_med*100:.2f}%, Low={pred_low*100:.2f}%, High={pred_high*100:.2f}%, Z={z_score:.2f}, Conf={conf_score}%")

    # Signal
    rsi_val = latest_row['RSI'].iloc[0] if 'RSI' in latest_row.columns else 50.0
    sma_val = latest_row['SMA_14'].iloc[0] if 'SMA_14' in latest_row.columns else None
    
    rec, _ = predictor.make_recommendation(
        current_usd, 
        predicted_usd,
        conf_direction=conf_direction,
        conf_score=conf_score,
        rsi=rsi_val,
        sma=sma_val
    )

    # Log the signal persistently
    signal_logger.log_daily_signal(
        date=(pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        price_usd=current_usd,
        predicted_usd=predicted_usd,
        signal=rec,
        confidence_score=conf_score,
        confidence_direction=conf_direction
    )

    # Local Specs
    grams_per_oz = 31.1035
    price_gram_idr = (current_usd * current_idr_rate) / grams_per_oz
    pred_gram_idr = (predicted_usd * current_idr_rate) / grams_per_oz

    # Calculate daily change (historical)
    prev_usd = market_data['Gold'].iloc[-2] if len(market_data) > 1 else current_usd
    daily_change_pct = ((current_usd - prev_usd) / prev_usd) * 100

    # Fetch Physical Price (Antam) - Real Data with Spot Fallback
    physical_price = loader.fetch_antam_price(current_spot_price=price_gram_idr)
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
        "confidence_direction": conf_direction,
        "confidence_score": conf_score,
        "is_bullish": bool(rsi_val > 60 or (sma_val and current_usd > sma_val)),
        "rsi": round(rsi_val, 2),
        "action_date": (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime(
            '%Y-%m-%d'),
        "top_headlines": headlines,
        "dynamic_risks": [] # Initialized below
    }

    # Dynamic Risk Engine
    dynamic_risks = []
    # 1. Trend/RSI Risk
    rsi_val = latest_row['RSI'].iloc[0] if 'RSI' in latest_row.columns else 50
    if rsi_val > 70:
        dynamic_risks.append({
            "title": "Profit Taking",
            "desc": "High overbought levels detected (RSI > 70). Risk of short-term correction."
        })
    elif rsi_val < 30:
        dynamic_risks.append({
            "title": "Oversold Bounce",
            "desc": "Market is extremely oversold. Watch for sharp relief rallies."
        })
    else:
        dynamic_risks.append({
            "title": "Trend Consolidation",
            "desc": "Neutral RSI range. Market is searching for clear direction."
        })

    # 2. Volatility/Momentum Risk
    abs_daily_change = abs(daily_change_pct)
    if abs_daily_change > 2.0:
        dynamic_risks.append({
            "title": "High Volatility",
            "desc": "Significant price movement detected. Increased risk of swing reversals."
        })
    elif predicted_usd > current_usd * 1.03:
        dynamic_risks.append({
            "title": "Upside Breakout",
            "desc": "Strong AI bullish bias. Watch for volume confirmation at resistance."
        })
    else:
        dynamic_risks.append({
            "title": "The Fed Policy",
            "desc": "Ongoing interest rate expectations remain a primary gold price anchor."
        })

    # 3. Macro/External Risk
    dxy_val = latest_row['DXY'].iloc[0] if 'DXY' in latest_row.columns else 100
    if dxy_val > 105:
        dynamic_risks.append({
            "title": "USD Strength",
            "desc": "Strong Dollar (DXY > 105) acting as a heavy resistance for Gold."
        })
    elif latest_row['Sentiment'].iloc[0] < -0.1 if 'Sentiment' in latest_row.columns else False:
        dynamic_risks.append({
            "title": "Negative Sentiment",
            "desc": "Prevailing news bias is bearish. Caution on weak support levels."
        })
    else:
        dynamic_risks.append({
            "title": "Geopolitics",
            "desc": "Safe-haven demand remains elevated amid global trade uncertainties."
        })
    
    result["dynamic_risks"] = dynamic_risks

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
        'Gold', 'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'VIX_Norm', 'GVZ_Norm',
        'Silver', 'Copper', 'Platinum', 'Palladium', 'USD_CNY', 'US10Y', 'Nikkei', 'DAX',
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'MACD', 'BB_Width', 'Sentiment'
    ]
    available_features = [f for f in features if f in df.columns]
    # Fetch LIVE data for t0 (Start of recursive forecast)
    live_gold = loader.fetch_live_data('GC=F')
    live_idr = loader.fetch_live_data('IDR=X')
    
    current_usd = live_gold['price'] if live_gold else market_data['Gold'].iloc[-1]
    current_idr_rate = live_idr['price'] if live_idr else market_data['USD_IDR'].iloc[-1]

    # --- DATA LAG FIX ---
    # Append the live data as a new row to the history buffer so technical indicators (RSI/MACD) 
    # are recalculated for "Today" before forecasting "Tomorrow".
    history_buffer = df.tail(100).copy()
    
    if live_gold and live_idr:
        new_row_data = history_buffer.iloc[[-1]].copy()
        new_row_data.index = [pd.Timestamp.now().normalize()]
        new_row_data['Gold'] = current_usd
        new_row_data['USD_IDR'] = current_idr_rate
        # Add slight jitter to macro features to avoid static flatlines if market is open
        for feat in ['DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 'Platinum', 'Palladium']:
            if feat in new_row_data.columns:
                new_row_data[feat] = new_row_data[feat] * (1 + (live_gold['change_pct']/200))
        
        history_buffer = pd.concat([history_buffer, new_row_data])
        # Recalculate indicators with the new live row
        history_buffer = engineering.add_technical_indicators(history_buffer)
        latest_features = history_buffer[available_features].iloc[[-1]]
    else:
        latest_features = df[available_features].iloc[[-1]]
    # -------------------

    # Handle USD/IDR shift properly (it's a percentage shift from the live rate)
    idr_shift_val = request.args.get('idr_shift', default=0, type=float) / 100
    manual_idr_rate = current_idr_rate * (1 + idr_shift_val)

    # Collect all shifts
    shifts = {
        'DXY': request.args.get('dxy_shift', default=0, type=float) / 100,
        'Oil': request.args.get('oil_shift', default=0, type=float) / 100,
        'SP500': request.args.get('sp500_shift', default=0, type=float) / 100,
        'Silver': request.args.get('silver_shift', default=0, type=float) / 100,
        'US10Y': request.args.get('us10y_shift', default=0, type=float) / 100,
        'USD_IDR': idr_shift_val # Pass to recursive_forecast as well for intra-step consistency
    }

    forecast_data = predictor.recursive_forecast(
        model_obj, latest_features, current_usd, manual_idr_rate, days=days,
        historical_df=history_buffer,
        shifts=shifts
    )

    grams_per_oz = 31.1035
    formatted = []
    for f in forecast_data:
        formatted.append({
            "date": f['Date'],
            "price_idr": round(f['Price_IDR'] / grams_per_oz, 0),
            "price_min": round(f['Price_Min_IDR'] / grams_per_oz, 0),
            "price_max": round(f['Price_Max_IDR'] / grams_per_oz, 0),
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


@app.route("/api/signals_history")
def get_signals_history():
    """Returns the CSV log of signals with calculated outcomes."""
    from src.data import signal_logger
    signals_df = signal_logger.get_signal_history()
    
    if signals_df.empty:
        return jsonify([])

    # Load actual history for comparison
    if os.path.exists(CSV_PATH):
        history_df = pd.read_csv(CSV_PATH)
        history_df['Date'] = pd.to_datetime(history_df['Date']).dt.strftime('%Y-%m-%d')
        actual_prices = history_df.set_index('Date')['Gold'].to_dict()
    else:
        actual_prices = {}

    # Define outcome logic
    now_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    outcomes = []
    for _, row in signals_df.iterrows():
        target_date = row['Date']
        signal_price = row['Price_USD']
        predicted_price = row['Predicted_USD']
        
        actual_price = actual_prices.get(target_date)
        
        if actual_price:
            predicted_diff = predicted_price - signal_price
            actual_diff = actual_price - signal_price
            
            # Outcome based on direction matching
            if predicted_diff * actual_diff > 0:
                outcome = "Correct"
            elif abs(predicted_diff) < 0.1 and abs(actual_diff) < 0.1: # Neutral match
                outcome = "Correct"
            else:
                outcome = "Wrong"
        elif target_date >= now_str:
            outcome = "Pending"
        else:
            outcome = "Expired" # Past date but no history found (e.g. weekend)
            
        outcomes.append(outcome)
        
    signals_df['Outcome'] = outcomes
    return jsonify(signals_df.to_dict(orient='records'))


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



@app.route("/api/model_metrics")
def get_model_metrics():
    """Returns model performance metrics from training metadata."""
    path = "models/metrics.json"
    if not os.path.exists(path):
        return jsonify({"error": "Metrics not found"}), 404
    
    import json
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------
# Background Scheduler
# ----------------------------------------------------
import threading
import subprocess
import time
import sys

UPDATE_INTERVAL = 3600  # 1 Hour (Continuous Learning Mode)

def run_periodic_update():
    """Reads main.py periodically to update model and data."""
    while True:
        print(f"\n[Scheduler] Usage monitoring... Next update in {UPDATE_INTERVAL/60} minutes.")
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

# Global Training State
TRAINING_STATE = {
    "status": "idle", # or 'running'
    "message": "",
    "timestamp": 0
}

@app.route('/api/training_status')
def get_training_status():
    return jsonify(TRAINING_STATE)

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Manually triggers model retraining."""
    def run_training():
        global TRAINING_STATE
        TRAINING_STATE['status'] = 'running'
        TRAINING_STATE['message'] = 'Starting training process...'
        TRAINING_STATE['timestamp'] = int(time.time())
        
        print("[Manual] Starting manual model training...")
        try:
             TRAINING_STATE['message'] = 'Running main.py...'
             result = subprocess.run(
                [sys.executable, "main.py", "--days", "1"],
                capture_output=True,
                text=True
            )
             if result.returncode == 0:
                 print("\n[Manual] Training Success!")
                 print(result.stdout[-200:])
                 
                 # INVALIDATE CACHE
                 global trained_model
                 trained_model = None
                 print("[Manual] Cache cleared. New model ready.")
                 
                 TRAINING_STATE['status'] = 'done'
                 TRAINING_STATE['message'] = 'Training completed successfully.'
             else:
                 print("\n[Manual] Training Failed!")
                 print(result.stderr)
                 TRAINING_STATE['status'] = 'error'
                 TRAINING_STATE['message'] = f"Training failed: {result.stderr[-50:]}"
                 
        except Exception as e:
            print(f"[Manual] Error: {e}")
            TRAINING_STATE['status'] = 'error'
            TRAINING_STATE['message'] = str(e)

    if TRAINING_STATE['status'] == 'running':
         return jsonify({"status": "error", "message": "Training already in progress."})

    # Run in background to not block response
    threading.Thread(target=run_training).start()
    return jsonify({"status": "started", "message": "Training started."})

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

    return jsonify(data)

if __name__ == "__main__":
    # Ensure scheduler only runs in the reloader process (not the main watcher)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_scheduler()
    
    # --- COLD START CHECK ---
    # If models are missing (fresh install or wiped), train immediately.
    # We check independently of reloader to ensure it exists before serving.
    required_model = "models/gold_model_med.pkl"
    if not os.path.exists(required_model):
        print("\n[Cold Start] Model not found. Initializing training... (This may take 1-2 mins)")
        try:
             subprocess.run([sys.executable, "main.py", "--days", "1"], check=True)
             print("[Cold Start] Training Complete. Starting Server...")
        except Exception as e:
            print(f"[Cold Start] Failed to train initial model: {e}")

    app.run(debug=True, port=5000)
