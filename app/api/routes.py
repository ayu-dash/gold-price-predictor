"""API routes for data endpoints."""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime

import pandas as pd
import scipy.stats as stats
from flask import current_app, jsonify, request

import config
from app.api import api_bp
from core.data import loader, signal_logger
from core.features import engineering
from core.prediction import predictor


@api_bp.route("/history")
def get_history():
    """Return historical price data for charting."""
    if not os.path.exists(config.CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(config.CSV_PATH, index_col='Date', parse_dates=True)
    recent_df = df.tail(100).copy()

    # Append live price as final point
    live_gold = loader.fetch_live_data('GC=F')
    if live_gold:
        now_idx = pd.Timestamp.now().normalize()
        if recent_df.index[-1].normalize() == now_idx:
            recent_df.iloc[-1, recent_df.columns.get_loc('Gold')] = live_gold['price']
        else:
            new_row = pd.DataFrame(
                {'Gold': [live_gold['price']], 'USD_IDR': [recent_df['USD_IDR'].iloc[-1]]},
                index=[now_idx]
            )
            recent_df = pd.concat([recent_df, new_row])

    data = {
        "dates": recent_df.index.strftime('%Y-%m-%d').tolist(),
        "prices": recent_df['Gold'].tolist(),
        "idr_rate": recent_df['USD_IDR'].tolist()
    }
    return jsonify(data)


@api_bp.route("/market_monitor")
def market_monitor():
    """Return a snapshot of global market health."""
    snapshot = loader.fetch_market_snapshot()
    return jsonify(snapshot)


@api_bp.route("/force_db_update")
def force_db_update():
    """Manually trigger database update."""
    try:
        loader.update_local_database(force=True)
        return jsonify({"status": "success", "message": "Database sync triggered"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/prediction")
def get_prediction():
    """Run prediction using pre-trained model."""
    get_model = current_app.config['get_model']
    model_obj = get_model()

    if model_obj is None:
        return jsonify({"error": "Model not trained yet"}), 503

    market_data = loader.update_local_database()
    df = engineering.add_technical_indicators(market_data)
    sentiment, headlines, sentiment_breakdown = loader.fetch_news_sentiment()
    df['Sentiment'] = sentiment

    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width', 
        'Stoch', 'WilliamsR', 'CCI', 'ATR', 'Return_Lag1', 
        'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]
    available_features = [f for f in features if f in df.columns]

    live_gold = loader.fetch_live_data('GC=F')
    live_idr = loader.fetch_live_data('IDR=X')

    current_usd = live_gold['price'] if live_gold else market_data['Gold'].iloc[-1]
    current_idr_rate = live_idr['price'] if live_idr else market_data['USD_IDR'].iloc[-1]

    # Inject live data as today's row
    if live_gold and live_idr:
        new_row = df.iloc[[-1]].copy()
        new_row.index = [pd.Timestamp.now().normalize()]
        new_row['Gold'] = current_usd
        new_row['USD_IDR'] = current_idr_rate

        for feat in ['DXY', 'Oil', 'SP500', 'Silver']:
            if feat in new_row.columns:
                new_row[feat] = new_row[feat] * (1 + (live_gold.get('change_pct', 0) / 200))

        df = pd.concat([df, new_row])
        df = engineering.add_technical_indicators(df)
        
    # Standardize feature vector (Force 27 features, fill missing with 0.0)
    latest_row = df.reindex(columns=features).iloc[[-1]].fillna(0.0)

    # Predict
    # v5 Deep Sniper: Using Classifier for high-confidence direction
    # and Regressor (med) for nominal target
    if isinstance(model_obj, dict):
        predicted_return = model_obj['med'].predict(latest_row)[0]
        # Get high-conf classifier if available
        clf_model = model_obj.get('clf')
        if clf_model:
            conf_direction, raw_prob = predictor.get_classification_confidence(clf_model, latest_row)
            conf_score = round(raw_prob * 100, 1)
        else:
            # Fallback to Z-score if clf is missing
            predicted_return = model_obj['med'].predict(latest_row)[0]
            pred_low = model_obj['low'].predict(latest_row)[0]
            pred_high = model_obj['high'].predict(latest_row)[0]
            conf_direction = "UP" if predicted_return > 0 else "DOWN"
            spread = max(0.0001, pred_high - pred_low)
            sigma = spread / 3.29
            z_score = abs(predicted_return) / sigma
            conf_score = round(stats.norm.cdf(z_score) * 100, 1)
    else:
        # Legacy support
        predicted_return = model_obj.predict(latest_row)[0]
        conf_direction = "UP" if predicted_return > 0 else "DOWN"
        conf_score = 50.0

    predicted_usd = current_usd * (1 + predicted_return)

    print(f"DEBUG v5: Ret={predicted_return*100:.2f}%, Dir={conf_direction}, Conf={conf_score}%")

    # Generate signal
    rsi_val = latest_row['RSI'].iloc[0] if 'RSI' in latest_row.columns else 50.0
    sma_val = latest_row['SMA_14'].iloc[0] if 'SMA_14' in latest_row.columns else None

    rec, _ = predictor.make_recommendation(
        current_usd, predicted_usd,
        conf_direction=conf_direction,
        conf_score=conf_score,
        rsi=rsi_val,
        sma=sma_val
    )

    # Log signal
    signal_logger.log_daily_signal(
        date=(pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        price_usd=current_usd,
        predicted_usd=predicted_usd,
        signal=rec,
        confidence_score=conf_score,
        confidence_direction=conf_direction
    )

    # Convert to local specs
    price_gram_idr = (current_usd * current_idr_rate) / config.GRAMS_PER_OZ
    pred_gram_idr = (predicted_usd * current_idr_rate) / config.GRAMS_PER_OZ

    prev_usd = market_data['Gold'].iloc[-2] if len(market_data) > 1 else current_usd
    daily_change_pct = ((current_usd - prev_usd) / prev_usd) * 100

    physical_price = loader.fetch_antam_price(current_spot_price=price_gram_idr)

    result = {
        "current_price_usd": round(current_usd, 2),
        "predicted_price_usd": round(predicted_usd, 2),
        "current_price_idr_gram": round(price_gram_idr, 0),
        "physical_price_idr": physical_price,
        "predicted_price_idr_gram": round(pred_gram_idr, 0),
        "change_pct": round(predicted_return * 100, 2),
        "daily_change_pct": round(daily_change_pct, 2),
        "recommendation": rec,
        "sentiment_score": round(sentiment, 4),
        "sentiment_breakdown": sentiment_breakdown,
        "confidence_direction": conf_direction,
        "confidence_score": conf_score,
        "last_db_date": market_data.index.max().strftime('%Y-%m-%d'),
        "is_bullish": bool(rsi_val > 60 or (sma_val and current_usd > sma_val)),
        "rsi": round(rsi_val, 2),
        "action_date": (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        "top_headlines": headlines,
        "dynamic_risks": []
    }

    # Dynamic risk engine
    dynamic_risks = []
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

    dxy_val = latest_row['DXY'].iloc[0] if 'DXY' in latest_row.columns else 100
    if dxy_val > 105:
        dynamic_risks.append({
            "title": "USD Strength",
            "desc": "Strong Dollar (DXY > 105) acting as heavy resistance for Gold."
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


@api_bp.route("/live_price")
def get_live_price():
    """Return realtime price update."""
    live_gold = loader.fetch_live_data('GC=F')
    if not live_gold:
        return jsonify({"error": "Failed to fetch live data"}), 503

    live_idr = loader.fetch_live_data('IDR=X')
    rate = live_idr['price'] if live_idr else 15500.0

    current_usd = live_gold['price']
    current_time = datetime.now().strftime('%H:%M:%S')

    price_gram_idr = (current_usd * rate) / config.GRAMS_PER_OZ

    return jsonify({
        "timestamp": current_time,
        "price_usd": round(current_usd, 2),
        "price_idr_gram": round(price_gram_idr, 0),
        "change_pct": round(live_gold['change_pct'], 2),
        "rate": round(rate, 2)
    })


@api_bp.route("/forecast")
def get_forecast():
    """Generate multi-day recursive forecast."""
    get_model = current_app.config['get_model']
    model_obj = get_model()

    if model_obj is None:
        return jsonify({"error": "Model not trained yet"}), 503

    days = request.args.get('days', default=7, type=int)

    market_data = loader.update_local_database()
    df = engineering.add_technical_indicators(market_data)
    sentiment, _, _ = loader.fetch_news_sentiment()
    df['Sentiment'] = sentiment

    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width', 
        'Stoch', 'WilliamsR', 'CCI', 'ATR', 'Return_Lag1', 
        'Return_Lag2', 'Return_Lag3', 'RSI_Lag1', 'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]
    available_features = [f for f in features if f in df.columns]

    live_gold = loader.fetch_live_data('GC=F')
    live_idr = loader.fetch_live_data('IDR=X')

    current_usd = live_gold['price'] if live_gold else market_data['Gold'].iloc[-1]
    current_idr_rate = live_idr['price'] if live_idr else market_data['USD_IDR'].iloc[-1]

    history_buffer = df.tail(100).copy()

    if live_gold and live_idr:
        new_row_data = history_buffer.iloc[[-1]].copy()
        new_row_data.index = [pd.Timestamp.now().normalize()]
        new_row_data['Gold'] = current_usd
        new_row_data['USD_IDR'] = current_idr_rate

        for feat in ['DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 'Platinum', 'Palladium']:
            if feat in new_row_data.columns:
                new_row_data[feat] = new_row_data[feat] * (1 + (live_gold['change_pct'] / 200))

        history_buffer = pd.concat([history_buffer, new_row_data])
        history_buffer = engineering.add_technical_indicators(history_buffer)
        
    # Standardize feature vector (Force 27 features, fill missing with 0.0)
    latest_features = history_buffer.reindex(columns=features).iloc[[-1]].fillna(0.0)

    # Predictor with no manual shifts
    forecast_data = predictor.recursive_forecast(
        model_obj, latest_features, current_usd, current_idr_rate,
        days=days, historical_df=history_buffer
    )

    formatted = []
    for f in forecast_data:
        formatted.append({
            "date": f['Date'],
            "price_idr": round(f['Price_IDR'] / config.GRAMS_PER_OZ, 0),
            "price_min": round(f['Price_Min_IDR'] / config.GRAMS_PER_OZ, 0),
            "price_max": round(f['Price_Max_IDR'] / config.GRAMS_PER_OZ, 0),
            "change_pct": f['Return_Pct']
        })

    return jsonify(formatted)


@api_bp.route("/signals_history")
def get_signals_history():
    """Return the CSV log of signals with calculated outcomes."""
    signals_df = signal_logger.get_signal_history()

    # Ensure we have the latest historical data for outcomes
    loader.update_local_database()

    if os.path.exists(config.CSV_PATH):
        history_df = pd.read_csv(config.CSV_PATH)
        history_df['Date'] = pd.to_datetime(history_df['Date']).dt.strftime('%Y-%m-%d')
        actual_prices = history_df.set_index('Date')['Gold'].to_dict()
    else:
        actual_prices = {}

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    live_price_data = loader.fetch_live_data('GC=F')
    current_live_price = live_price_data['price'] if live_price_data else None

    outcomes = []
    for _, row in signals_df.iterrows():
        target_date = row['Date']
        signal_price = row['Price_USD']
        predicted_price = row['Predicted_USD']
        actual_price = actual_prices.get(target_date)

        # Use live price as fallback for TODAY
        is_live = False
        if not actual_price and target_date == now_str and current_live_price:
            actual_price = current_live_price
            is_live = True

        if actual_price:
            predicted_diff = predicted_price - signal_price
            actual_diff = actual_price - signal_price

            if predicted_diff * actual_diff > 0:
                base_outcome = "Correct"
            elif abs(predicted_diff) < 0.1 and abs(actual_diff) < 0.1:
                base_outcome = "Correct"
            else:
                base_outcome = "Wrong"
            
            # Status Logic
            if is_live:
                # Market is still open for this signal's target date
                status_prefix = "↑" if actual_diff > 0 else "↓"
                outcome = f"Live Tracking ({status_prefix}{abs(actual_diff):.2f})"
            else:
                # Target date has passed, result is LOCKED from database
                outcome = f"{base_outcome} (Final)"
        elif target_date > now_str:
            outcome = "Upcoming"
        else:
            outcome = "Waiting for Daily Close"

        outcomes.append(outcome)

    signals_df['Outcome'] = outcomes
    return jsonify(signals_df.to_dict(orient='records'))


@api_bp.route("/yearly_stats")
def get_yearly_stats():
    """Aggregate gold data by year for performance comparison."""
    if not os.path.exists(config.CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(config.CSV_PATH, index_col='Date', parse_dates=True)
    df['Price_IDR_g'] = (df['Gold'] * df['USD_IDR']) / config.GRAMS_PER_OZ

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


@api_bp.route("/technical")
def get_technical():
    """Return detailed technical indicators for charts."""
    if not os.path.exists(config.CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(config.CSV_PATH, index_col='Date', parse_dates=True)
    df = engineering.add_technical_indicators(df)

    recent_df = df.tail(60)

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


@api_bp.route("/model_metrics")
def get_model_metrics():
    """Return model performance metrics from training metadata."""
    if not os.path.exists(config.METRICS_PATH):
        return jsonify({"error": "Metrics not found"}), 404

    import json
    try:
        with open(config.METRICS_PATH, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/training_status")
def get_training_status():
    """Return current training status."""
    return jsonify(current_app.config['TRAINING_STATE'])


@api_bp.route("/next_update")
def get_next_update():
    """Return seconds remaining until next scheduled refresh."""
    next_time = current_app.config['get_next_update_time']()
    remaining = max(0, int(next_time - time.time()))
    return jsonify({
        "seconds_remaining": remaining,
        "next_update_timestamp": int(next_time)
    })


@api_bp.route("/retrain", methods=['POST'])
def retrain_model():
    """Manually trigger model retraining."""
    training_state = current_app.config['TRAINING_STATE']
    invalidate_cache = current_app.config['invalidate_model_cache']

    def run_training():
        training_state['status'] = 'running'
        training_state['message'] = 'Starting training process...'
        training_state['timestamp'] = int(time.time())

        print("[Manual] Starting model training...")
        try:
            training_state['message'] = 'Running training script...'
            script_path = os.path.join(config.BASE_DIR, "bin", "run_training.py")
            result = subprocess.run(
                [sys.executable, script_path, "--days", "1"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print("[Manual] Training success!")
                invalidate_cache()
                training_state['status'] = 'done'
                training_state['message'] = 'Training completed successfully.'
            else:
                print(f"[Manual] Training failed: {result.stderr[-200:]}")
                training_state['status'] = 'error'
                training_state['message'] = f"Training failed: {result.stderr[-50:]}"

        except Exception as e:
            print(f"[Manual] Error: {e}")
            training_state['status'] = 'error'
            training_state['message'] = str(e)

    if training_state['status'] == 'running':
        return jsonify({"status": "error", "message": "Training already in progress."})

    threading.Thread(target=run_training).start()
    return jsonify({"status": "started", "message": "Training started."})


@api_bp.route("/history/full")
def get_full_history():
    """Return complete historical data for table view."""
    if not os.path.exists(config.CSV_PATH):
        return jsonify({"error": "No history found"}), 404

    df = pd.read_csv(config.CSV_PATH, index_col='Date', parse_dates=True)
    df = engineering.add_technical_indicators(df)

    if 'Gold_Returns' in df.columns:
        df['Daily_Return'] = df['Gold_Returns'] * 100
    else:
        df['Daily_Return'] = df['Gold'].pct_change() * 100

    df_sorted = df.sort_index(ascending=False).copy()

    dates = df_sorted.index.strftime('%Y-%m-%d').tolist()
    prices_usd = df_sorted['Gold'].tolist()
    rates = df_sorted['USD_IDR'].tolist()
    daily_returns = df_sorted['Daily_Return'].fillna(0).tolist()

    prices_idr = [
        (p * r) / config.GRAMS_PER_OZ
        for p, r in zip(prices_usd, rates)
    ]

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
