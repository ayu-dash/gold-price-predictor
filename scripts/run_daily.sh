#!/bin/bash

# Navigate to project directory
cd /home/wtf/Documents/Projects/gold-price-predictor

# Activate Virtual Environment
source .venv/bin/activate

# Print Divider to Log
echo "==================================================" >> logs/predictions.log
echo "RUN DATE: $(date)" >> logs/predictions.log
echo "==================================================" >> logs/predictions.log

# Run the prediction and log output
python3 bin/run_training.py --days 1 >> logs/predictions.log 2>&1

# Deactivate
deactivate
