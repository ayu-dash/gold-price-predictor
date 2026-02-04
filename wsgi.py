"""
WSGI entry point for Gunicorn.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import create_app, _start_scheduler

app = create_app()

_start_scheduler()

if __name__ == "__main__":
    app.run()
