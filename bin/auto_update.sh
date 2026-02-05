#!/bin/bash

# Gold Predictor Daily Auto-Update Script
# Recommended cron: 0 5 * * * (Every day at 05:00 AM)

PROJECT_DIR="/home/wtf/Documents/Projects/gold-price-predictor"
LOG_FILE="$PROJECT_DIR/logs/auto_update.log"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"

mkdir -p "$PROJECT_DIR/logs"

echo "[$(date)] Starting Daily Market Update..." >> "$LOG_FILE"

cd "$PROJECT_DIR"

# Run Training and Prediction Pipeline
# --days 1 ensures we generate a fresh daily signal
"$PYTHON_BIN" bin/run_training.py --days 1 >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] Success: Market database and models updated." >> "$LOG_FILE"
    # Optional: Trigger API reload if running under supervisor/systemd
    # curl -X POST http://localhost:5000/api/retrain (Using internal endpoint)
else
    echo "[$(date)] ERROR: Update pipeline failed. Check logs/training.log" >> "$LOG_FILE"
fi

echo "-----------------------------------------------" >> "$LOG_FILE"
