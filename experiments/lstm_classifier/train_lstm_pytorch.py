"""
PyTorch LSTM Classifier Experiment (v2.0) - Tuning & Deep Sentiment
Isolated test - does not affect main codebase.

Features:
- Hyperparameter Tuning Loop
- Support for Sentiment Features (if logs available)
- Flexible Architecture (LSTM, GRU, Attention)

Usage:
    cd /home/wtf/Documents/Projects/gold-price-predictor
    source .venv/bin/activate
    python experiments/lstm_classifier/train_lstm_pytorch.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import itertools
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config
from core.data import loader, sentiment_logger
from core.features import engineering

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Global Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- ARCHITECTURES ---

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        # attn_weights shape: (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        # context shape: (batch, hidden_size)
        return context, attn_weights

class FlexibleClassifier(nn.Module):
    """Flexible model supporting LSTM/GRU and Attention."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, model_type='LSTM', use_attention=False):
        super(FlexibleClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.use_attention = use_attention
        
        # RNN layer
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        if use_attention:
            self.attention = Attention(hidden_size)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # Sigmoid removed for BCEWithLogitsLoss
        )
    
    def forward(self, x):
        # Forward pass through RNN
        outputs, _ = self.rnn(x)
        
        if self.use_attention:
            x, _ = self.attention(outputs)
        else:
            x = outputs[:, -1, :]  # Last hidden state
            
        x = self.batch_norm(x)
        x = self.fc(x)
        return x

# --- DATA PREPARATION ---

def load_and_prepare_data(include_sentiment=True):
    """Load market data and optionally join with sentiment history."""
    print("Loading market data...")
    market_data = loader.update_local_database()
    df = engineering.add_technical_indicators(market_data)
    
    # Clean and create target
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['Target'] = (df['Gold'].shift(-1) > df['Gold']).astype(int)
    df = df.dropna()
    
    if include_sentiment:
        sent_df = sentiment_logger.get_sentiment_history()
        if not sent_df.empty:
            print(f"Joining with sentiment history ({len(sent_df)} entries)...")
            df = df.join(sent_df[['Avg_Sentiment', 'Pos_Ratio', 'Neg_Ratio', 'Neu_Ratio']], how='left')
            df = df.fillna(0.0) # Assume 0/neutral if no sentiment logged
            sentiment_cols = ['Avg_Sentiment', 'Pos_Ratio', 'Neg_Ratio', 'Neu_Ratio']
        else:
            print("No sentiment history found. Skipping sentiment features.")
            sentiment_cols = []
    else:
        sentiment_cols = []
        
    # Features
    feature_cols = config.MODEL_FEATURES + sentiment_cols
    valid_features = [f for f in feature_cols if f in df.columns]
    
    print(f"Final feature set: {len(valid_features)} columns")
    return df, valid_features

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

# --- TRAINING & EVALUATION ---

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, pos_weight=1.0):
    # Use BCEWithLogitsLoss for better numerical stability and pos_weight support
    weight = torch.tensor([pos_weight]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 12
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            # Predictions are raw logits
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                val_loss += criterion(model(X_batch), y_batch).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss

def optimize_threshold(model, dataloader):
    """Search for optimal threshold to maximize precision-weighted score."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch.to(DEVICE))
            all_logits.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    best_score = -1
    best_thresh = 0.5
    
    # Test thresholds from 0.4 to 0.75
    for thresh in np.linspace(0.4, 0.75, 36):
        preds = (all_logits > thresh).astype(float)
        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        
        # Weighted score: Prioritize precision (0.7) over recall (0.3)
        score = 0.7 * p + 0.3 * r
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            
    return best_thresh, best_score

def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(float)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1

# --- TUNING LOOP ---

def run_tuning():
    print("\n" + "="*50)
    print("STARTING PRECISION-FOCUSED TUNING (v3.0)")
    print("="*50)
    
    df, features = load_and_prepare_data()
    X = df[features].values
    y = df['Target'].values
    
    # TUNE PARAMS
    param_grid = {
        'seq_length': [10, 20],
        'hidden_size': [32, 64],
        'model_type': ['LSTM', 'GRU'],
        'use_attention': [True, False],
        'pos_weight': [0.5, 0.8, 1.0] # Weights < 1.0 punish False Positives more
    }
    
    # Generate combinations
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combos)}")
    
    # Pre-scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_p_score = 0
    best_params = None
    
    results = []
    
    for i, p in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] Testing: {p}")
        
        # Prepare sequences
        X_seq, y_seq = create_sequences(X_scaled, y, p['seq_length'])
        
        # Split (Simplified for tuning)
        split = int(len(X_seq) * 0.8)
        train_X, test_X = X_seq[:split], X_seq[split:]
        train_y, test_y = y_seq[:split], y_seq[split:]
        
        # Val split from train
        vsplit = int(len(train_X) * 0.85)
        X_t, X_v = train_X[:vsplit], train_X[vsplit:]
        y_t, y_v = train_y[:vsplit], train_y[vsplit:]
        
        # Dataloaders
        train_ld = DataLoader(TensorDataset(torch.FloatTensor(X_t), torch.FloatTensor(y_t).unsqueeze(1)), batch_size=32, shuffle=True)
        val_ld = DataLoader(TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v).unsqueeze(1)), batch_size=32)
        test_ld = DataLoader(TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_y).unsqueeze(1)), batch_size=32)
        
        # Build & Train
        model = FlexibleClassifier(
            input_size=len(features),
            hidden_size=p['hidden_size'],
            num_layers=1, # Fixed to 1 for speed during tuning
            dropout=0.2,
            model_type=p['model_type'],
            use_attention=p['use_attention']
        ).to(DEVICE)
        
        _ = train_model(model, train_ld, val_ld, epochs=25, pos_weight=p['pos_weight'])
        
        # Find optimal threshold on validation set
        opt_thresh, p_score = optimize_threshold(model, val_ld)
        
        # Evaluate on test set with opt_thresh
        acc, prec, rec, f1 = evaluate_model(model, test_ld, threshold=opt_thresh)
        
        print(f"Results -> Thresh: {opt_thresh:.2f}, Prec: {prec:.2%}, Rec: {rec:.2%}, P-Score: {p_score:.2f}")
        
        results.append({'params': p, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'opt_thresh': opt_thresh})
        
        if p_score > best_p_score:
            best_p_score = p_score
            best_params = {**p, 'opt_thresh': opt_thresh}
            print(f"NEW BEST P-SCORE: {p_score:.2f}")
    
    print("\n" + "="*50)
    print(f"TUNING COMPLETE. Best P-Score: {best_p_score:.2f}")
    print(f"Best Params: {best_params}")
    
    # Save results
    results_path = Path(__file__).parent / "tuning_results_v3.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    return best_params

if __name__ == "__main__":
    best = run_tuning()

