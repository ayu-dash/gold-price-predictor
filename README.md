# Gold Price Predictor Pro

A professional Gold price prediction and market monitoring dashboard built with Flask and Scikit-Learn.

## Features

- **Live Market Monitoring**: Real-time tracking of Gold Spot (IDR/g) and global market indices (DXY, Oil, S&P 500, etc.).
- **AI-Powered Forecasts**: Multi-model ensemble (Gradient Boosting, MLP) for daily price direction and regression.
- **Sentiment Analysis**: Integration of global and local Indonesian news sentiment using FinBERT/TextBlob.
- **Interactive Scenarios**: "What-if" simulator to see how macro-economic shifts (e.g., USD/IDR, DXY) affect gold forecasts.
- **Signal Logic**: Momentum-aware Buy/Sell/Hold recommendations with confidence scores.
- **Asset Portfolio**: Track your physical gold holdings with real-time P/L calculation.

## Project Structure

- `app/`: Flask application with modular Blueprints (API & Views).
- `core/`: Core business logic for data collection, feature engineering, and prediction.
- `bin/`: Executable scripts for running the server and training the model.
- `trained_models/`: Serialized model artifacts and metrics.
- `config.py`: Centralized configuration for paths and constants.

## Quick Start

### 1. Installation
```bash
# Recommendation: Use a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Edit `config.py` to change the server `HOST`, `PORT`, or update intervals.

### 3. Running the Application
```bash
# Start the web server
python bin/run_server.py

# Run model training and manual prediction
python bin/run_training.py --days 1
```

## Dashboard
Once running, access the dashboard at `http://localhost:5000` (or your configured IP).

## License
MIT
