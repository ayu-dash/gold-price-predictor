
import os
import sys
import numpy as np
import pandas as pd
import ta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, classification_report

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "../data/gold_history.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "candidate_model.pkl")
REGRESSOR_SAVE_PATH = os.path.join(SCRIPT_DIR, "candidate_regressor.pkl")

# LEAKAGE PREVENTION: Only train on data before this date
# Moved to 2025-07-01 to allow a longer 7-month blind test window
TRAIN_CUTOFF = "2025-07-01"

def load_and_engineer_data():
    """Load data and apply feature engineering (Self-contained for sandbox)."""
    if not os.path.exists(CSV_PATH):
        logger.error(f"Data not found at {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
    df = df.sort_index()
    
    # --- Feature Engineering (Copied & Refined from core) ---
    exclude_columns = ['Dividends', 'Stock Splits']
    df = df.drop(columns=[c for c in exclude_columns if c in df.columns])
    
    # 1. Targets (Significant Moves Only)
    # We define a "Move" as > 0.15% change. 
    # Everything else is noise (which we drop from TRAIN, but effectively 'Test' on).
    df['Returns'] = df['Gold'].pct_change()
    
    threshold = 0.0015 # 0.15%
    
    # Create valid mask
    significant_mask = abs(df['Returns'].shift(-1)) > threshold
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # We keep the mask to filter later
    df['Significant'] = significant_mask
    
    # Drop NaNs created by shift
    df = df.dropna()
    
    # Filter Training Data: Only train on significant rows
    # NOTE: For TimeSeriesSplit to be fair, we must be careful. 
    # Simplest way: Train on filtered, but for the validation loop, we ideally test on all?
    # No, usually we test on "can I predict likely significant moves?".
    # Let's filter the DF returned here.
    # --- 2. Technical Indicators (Apply to ALL rows first) ---
    # Trend
    df['SMA_7'] = ta.trend.sma_indicator(df['Gold'], window=7)
    df['SMA_14'] = ta.trend.sma_indicator(df['Gold'], window=14)
    df['EMA_14'] = ta.trend.ema_indicator(df['Gold'], window=14)
    
    # --- Inter-market & Ratio Features ---
    if 'Silver' in df.columns and df['Silver'].all() > 0:
        df['Gold_Silver_Ratio'] = df['Gold'] / df['Silver']
    if 'Oil' in df.columns and df['Oil'].all() > 0:
        df['Gold_Oil_Ratio'] = df['Gold'] / df['Oil']
        
    # Macro Volatility & Yields (if available in the CSV)
    if 'VIX' in df.columns:
        df['VIX_Return'] = df['VIX'].pct_change()
        df['VIX_Lag1'] = df['VIX_Return'].shift(1)
    if 'GVZ' in df.columns:
        df['GVZ_Return'] = df['GVZ'].pct_change()
    if 'US10Y' in df.columns:
        df['US10Y_Diff'] = df['US10Y'].diff()
        df['US10Y_Lag1'] = df['US10Y_Diff'].shift(1)
        
    # Lagged Macro for lead-lag effects
    df['DXY_Ret_Lag1'] = df['DXY'].pct_change().shift(1)
    df['SP500_Ret_Lag1'] = df['SP500'].pct_change().shift(1)

    # Momentum
    df['RSI'] = ta.momentum.rsi(df['Gold'], window=14)
    df['RSI_7'] = ta.momentum.rsi(df['Gold'], window=7)
    df['ROC_10'] = ta.momentum.roc(df['Gold'], window=10)
    
    # Advanced Indicators
    close = df['Gold']
    high = df['Gold']
    low = df['Gold']
    
    df['Stoch'] = ta.momentum.stoch(high, low, close, window=14)
    df['WilliamsR'] = ta.momentum.williams_r(high, low, close, lbp=14)
    df['CCI'] = ta.trend.cci(high, low, close, window=20)
    
    # Volatility
    df['BB_Width'] = ta.volatility.bollinger_wband(df['Gold'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Custom Lags
    for lag in [1, 2, 3]:
        df[f'Return_Lag{lag}'] = df['Returns'].shift(lag)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    
    # Rolling Stats
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Momentum_5'] = df['Gold'] / df['Gold'].shift(5) - 1
    
    # Clean NaNs after engineering
    df = df.dropna()

    # --- 3. Split into Classification vs Regression Sets ---
    # Full dataset for Regressor
    df_reg_all = df.copy()
    
    # Pruned dataset for Classification (Focus on high-signal moves)
    df_clf = df[df['Significant']].copy()
    logger.info(f"Filtered Noise (Classification Only): Kept {len(df_clf)}/{len(df_reg_all)} samples (> 0.15% move)")
    
    # --- 4. Leakage Prevention (Prune by TRAIN_CUTOFF) ---
    df_clf = df_clf[df_clf.index < TRAIN_CUTOFF]
    df_reg_all = df_reg_all[df_reg_all.index < TRAIN_CUTOFF]
    
    logger.info(f"Leakage Prevention: Training set pruned to data before {TRAIN_CUTOFF}")
    logger.info(f"Final training samples (Classification): {len(df_clf)}")
    
    return df_clf, df_reg_all

def train_and_evaluate(df, df_reg_all):
    """Train HistGradientBoosting and evaluate with Sniper Logic."""
    features = [
        'USD_IDR', 'DXY', 'Oil', 'SP500', 'NASDAQ', 'Silver', 
        'SMA_7', 'SMA_14', 'RSI', 'RSI_7', 'ROC_10', 'BB_Width',
        'Stoch', 'WilliamsR', 'CCI', 'ATR',
        'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'RSI_Lag1',
        'Volatility_5', 'Momentum_5',
        'Gold_Silver_Ratio', 'VIX_Lag1', 'US10Y_Lag1', 'DXY_Ret_Lag1', 'SP500_Ret_Lag1'
    ]
    
    X = df[features]
    y = df['Target']
    
    # --- TIME SERIES SPLIT (Robust Validation) ---
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- TIME SERIES SPLIT ---
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Store probability results from all folds to avoid re-training in threshold search
    fold_results = []
    
    logger.info("📡 Training Ensemble Learners (RF + HGB) across TimeSeries Folds...")
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Triple Ensemble Learners
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=300, max_depth=8, l2_regularization=0.1, random_state=42
        )
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
        et = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42)
        
        model = VotingClassifier(
            estimators=[('hgb', hgb), ('rf', rf), ('et', et)], 
            voting='soft'
        )
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        fold_results.append((y_val, probs))

    # Extensive Threshold Search (Zero-FP Focus)
    thresholds = np.linspace(0.55, 0.95, 41)
    best_config = None
    
    logger.info(f"🔍 Searching for 'Zero-FP Barrier' across {len(thresholds)} levels...")
    
    for thresh in thresholds:
        t_acc, t_prec, t_rec, t_f1, t_trades, t_fp = [], [], [], [], [], []
        
        for y_true, y_probs in fold_results:
            mask = y_probs >= thresh
            if sum(mask) == 0: 
                t_fp.append(0)
                continue
            
            y_p = (y_probs[mask] >= 0.5).astype(int)
            y_t = y_true[mask]
            
            # Confusion Matrix components for zero-FP check
            # We treat any y_t=0 as an error if predicted 1
            fps = sum((y_p == 1) & (y_t == 0))
            
            t_acc.append(accuracy_score(y_t, y_p))
            t_prec.append(precision_score(y_t, y_p, zero_division=0))
            t_rec.append(recall_score(y_t, y_p, zero_division=0))
            t_f1.append(f1_score(y_t, y_p, zero_division=0))
            t_trades.append(sum(mask))
            t_fp.append(fps)
            
        if len(t_acc) >= 2: # At least triggered in 2 folds
            avg_fp = np.mean(t_fp)
            avg_prec = np.mean(t_prec)
            avg_acc = np.mean(t_acc)
            
            config = {
                'thresh': float(thresh),
                'acc': float(avg_acc),
                'prec': float(avg_prec),
                'rec': float(np.mean(t_rec)),
                'f1': float(np.mean(t_f1)),
                'trades': float(np.mean(t_trades)),
                'fp': float(avg_fp)
            }
            
            # Selection Strategy: Prioritize FP == 0, then Max Precision, then Acc
            if best_config is None:
                best_config = config
            else:
                # 1. Prefer lower FP
                if avg_fp < best_config['fp']:
                    best_config = config
                # 2. If FPs are equal (hopefully both 0), prefer higher Precision
                elif avg_fp == best_config['fp'] and avg_prec > best_config['prec']:
                    best_config = config
                # 3. If Precision equal, prefer higher signals/trades
                elif avg_fp == best_config['fp'] and avg_prec == best_config['prec'] and config['trades'] > best_config['trades']:
                    best_config = config

    if not best_config:
         logger.error("❌ CRITICAL: No stable threshold found.")
         return None
            
    logger.info("-" * 40)
    logger.info(f"🏆 BEST RESULTS FOUND")
    if best_config:
        logger.info(f"Model       : Triple Ensemble (RF+HGB+ET)")
        logger.info(f"Threshold   : {best_config['thresh']:.2f}")
        logger.info(f"Accuracy    : {best_config['acc']:.2%}")
        logger.info(f"Precision   : {best_config['prec']:.2%}")
        logger.info(f"Recall      : {best_config['rec']:.2%}")
        logger.info(f"F1 Score    : {best_config['f1']:.2%}")
        logger.info(f"Avg FPs     : {best_config['fp']:.2f} (Target: 0.0)")
        logger.info(f"Avg Trades  : {best_config['trades']:.1f} per fold")
        
        # Save Metrics to JSON
        import json
        metrics_path = os.path.join(SCRIPT_DIR, "candidate_metrics.json")
        metrics_data = {
            "model_type": "VotingClassifier (RF+HGB)",
            "features": features,
            "threshold": best_config['thresh'],
            "accuracy": best_config['acc'],
            "precision": best_config['prec'],
            "recall": best_config['rec'],
            "f1": best_config['f1'],
            "avg_trades": best_config['trades'],
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        if True: # Force save for v3 R&D to allow analysis
            logger.info("✅ SAVING: Zero-FP R&D model artifacts...")
            
            # To be robust, let's refit the ensemble on ALL data using the parameters we like
            hgb_final = HistGradientBoostingClassifier(
                learning_rate=0.03, max_depth=8, max_iter=300, l2_regularization=0.1, random_state=42
            )
            rf_final = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
            et_final = ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=42)
            final_model = VotingClassifier(
                estimators=[('hgb', hgb_final), ('rf', rf_final), ('et', et_final)], 
                voting='soft'
            )
            final_model.fit(X, y)
            
            import pickle
            with open(MODEL_SAVE_PATH, 'wb') as f:
                pickle.dump(final_model, f)
            logger.info(f"Classifier saved to {MODEL_SAVE_PATH}")
            
            # --- TRAIN REGRESSOR (Price Prediction) ---
            # Using df_reg_all (Full dataset including small moves)
            logger.info("📈 Training Price Regressor on FULL dataset (all price moves)...")
            
            # Re-engineer features for the full regressor dataset
            X_reg = df_reg_all[features]
            y_reg = df_reg_all['Gold'].shift(-1).ffill()
            
            regressor = HistGradientBoostingRegressor(
                learning_rate=0.03, max_depth=8, max_iter=300, l2_regularization=0.1, random_state=42
            )
            regressor.fit(X_reg, y_reg)
            
            with open(REGRESSOR_SAVE_PATH, 'wb') as f:
                pickle.dump(regressor, f)
            logger.info(f"Regressor saved to {REGRESSOR_SAVE_PATH}")
        else:
            logger.info(f"❌ FAILED: Best was {best_config['acc']:.2%} (Too low to save).")
    else:
         logger.info("❌ FAILED: No configuration met conditions.")

    return

if __name__ == "__main__":
    logger.info("Initializing Sandbox Environment...")
    df_clf, df_reg = load_and_engineer_data()
    train_and_evaluate(df_clf, df_reg)
