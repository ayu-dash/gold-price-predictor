"""
Flask application entry point.

Run this script to start the Gold Price Predictor web server.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import run_app

if __name__ == "__main__":
    run_app()
