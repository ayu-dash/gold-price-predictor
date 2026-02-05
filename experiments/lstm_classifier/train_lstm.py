"""
LSTM Classifier Experiment for Gold Price Direction Prediction
Isolated test - does not affect main codebase.

Usage:
    cd /home/wtf/Documents/Projects/gold-price-predictor
    source .venv/bin/activate
    python experiments/lstm_classifier/train_lstm.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config
from core.data import loader
from core.features import engineering

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
except ImportError:
    print("TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# Configuration
SEQUENCE_LENGTH = 20  # Look back 20 days
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

def load_and_prepare_data():
    """Load and prepare data for LSTM training."""
    print("Loading market data...")
    market_data = loader.update_local_database()
    df = engineering.add_technical_indicators(market_data)
    
    # Clean data
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Create target: 1 if price goes up tomorrow, 0 if down
    df['Target'] = (df['Gold'].shift(-1) > df['Gold']).astype(int)
    df = df.dropna()
    
    # Select features
    feature_cols = config.MODEL_FEATURES
    valid_features = [f for f in feature_cols if f in df.columns]
    
    print(f"Using {len(valid_features)} features: {valid_features}")
    
    return df, valid_features


def create_sequences(X, y, seq_length):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape, num_classes=2):
    """Build LSTM model for classification."""
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate():
    """Main training and evaluation function."""
    print("=" * 60)
    print("LSTM CLASSIFIER EXPERIMENT")
    print("=" * 60)
    
    # Load data
    df, feature_cols = load_and_prepare_data()
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Class distribution - UP: {y.sum()}, DOWN: {len(y) - y.sum()}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    print(f"\nCreating sequences with lookback={SEQUENCE_LENGTH}...")
    X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
    print(f"Sequence shape: {X_seq.shape}")
    
    # Split data (time-series aware)
    total_samples = len(X_seq)
    test_size = int(total_samples * TEST_SPLIT)
    val_size = int(total_samples * VALIDATION_SPLIT)
    train_size = total_samples - test_size - val_size
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, len(feature_cols)))
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING LSTM...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nAccuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Compare with baseline
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"\nBaseline (majority class): {baseline_acc:.2%}")
    print(f"LSTM Improvement: {(acc - baseline_acc):.2%}")
    
    # Save model
    model_path = Path(__file__).parent / "lstm_classifier_model.keras"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = Path(__file__).parent / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    return acc, prec, rec, f1


if __name__ == "__main__":
    # Set GPU memory growth if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    else:
        print("No GPU found, using CPU")
    
    # Run training
    acc, prec, rec, f1 = train_and_evaluate()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final Accuracy: {acc:.2%}")
    print(f"Final F1 Score: {f1:.2%}")
