"""
Flask application factory.

Creates and configures the Flask application with blueprints
for API and view routes.
"""

import os
import sys
import time
import threading
import subprocess
from typing import Optional

from flask import Flask
from flask_cors import CORS

import config
from core.prediction import predictor


# Global model cache
_trained_model: Optional[dict] = None
_model_lock = threading.Lock()

# Training state
TRAINING_STATE = {
    "status": "idle",
    "message": "",
    "timestamp": 0
}

# Scheduler timing
UPDATE_INTERVAL = config.UPDATE_INTERVAL_SECONDS
_scheduler_started = False


def _get_last_train_time() -> float:
    """Read the last training timestamp from metrics.json."""
    if os.path.exists(config.METRICS_PATH):
        try:
            import json
            from datetime import datetime
            with open(config.METRICS_PATH, 'r') as f:
                data = json.load(f)
                ts_str = data.get('timestamp')
                if ts_str:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    return dt.timestamp()
        except Exception:
            pass
    return 0.0


# Initialize NEXT_UPDATE_TIME based on actual file state
_last_train = _get_last_train_time()
NEXT_UPDATE_TIME = max(time.time(), _last_train + UPDATE_INTERVAL)


def get_model(retry: bool = True) -> Optional[dict]:
    """
    Load or return the cached trained model ensemble.

    Args:
        retry: If True, attempt emergency retrain on failure.

    Returns:
        Model ensemble dict or None if unavailable.
    """
    global _trained_model

    if _trained_model is not None:
        return _trained_model

    with _model_lock:
        if _trained_model is not None:
            return _trained_model

        paths = {
            'med': config.MODEL_MED_PATH,
            'low': config.MODEL_LOW_PATH,
            'high': config.MODEL_HIGH_PATH
        }

        try:
            ensemble = {}
            for q, p in paths.items():
                if not os.path.exists(p):
                    # Fallback to legacy path
                    legacy = os.path.join(config.MODELS_DIR, "gold_model.pkl")
                    ensemble[q] = predictor.load_model(legacy)
                else:
                    ensemble[q] = predictor.load_model(p)

            if all(m is not None for m in ensemble.values()):
                clf = predictor.load_model(config.MODEL_CLASSIFIER_PATH)
                if clf:
                    ensemble['clf'] = clf
                _trained_model = ensemble
                print("Model ensemble loaded successfully.")
            else:
                raise ValueError("Could not load required models.")

        except Exception as e:
            print(f"Model load error: {e}")
            if retry:
                print("Attempting emergency retrain...")
                try:
                    script_path = os.path.join(config.BASE_DIR, "bin", "run_training.py")
                    subprocess.run(
                        [sys.executable, script_path, "--days", "1"],
                        check=True,
                        timeout=300
                    )
                    return get_model(retry=False)
                except Exception as e2:
                    print(f"Emergency retrain failed: {e2}")

    return _trained_model


def invalidate_model_cache() -> None:
    """Clear the cached model to force reload."""
    global _trained_model
    _trained_model = None


def _run_periodic_update() -> None:
    """Background task for periodic model updates."""
    global NEXT_UPDATE_TIME

    while True:
        # Calculate exactly how long to wait based on NEXT_UPDATE_TIME
        wait_time = max(0, NEXT_UPDATE_TIME - time.time())
        if wait_time > 0:
            time.sleep(wait_time)

        print("\n[Scheduler] Starting periodic update...")
        try:
            script_path = os.path.join(config.BASE_DIR, "bin", "run_training.py")
            result = subprocess.run(
                [sys.executable, script_path, "--days", "1"],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                print("[Scheduler] Update completed.")
                invalidate_model_cache()
            else:
                print(f"[Scheduler] Update failed: {result.stderr[-200:]}")

        except Exception as e:
            print(f"[Scheduler] Error: {e}")

        # Schedule next update
        NEXT_UPDATE_TIME = time.time() + UPDATE_INTERVAL


def _start_scheduler() -> None:
    """Start the background scheduler thread."""
    global _scheduler_started
    if _scheduler_started:
        return

    thread = threading.Thread(target=_run_periodic_update, daemon=True)
    thread.start()
    _scheduler_started = True
    print("[Scheduler] Background service started.")


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )
    CORS(app)

    # Store shared state in app config
    app.config['get_model'] = get_model
    app.config['invalidate_model_cache'] = invalidate_model_cache
    app.config['TRAINING_STATE'] = TRAINING_STATE
    app.config['get_next_update_time'] = lambda: NEXT_UPDATE_TIME

    # Register blueprints
    from app.api import api_bp
    from app.views import views_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)

    return app


def run_app() -> None:
    """Run the Flask application with scheduler."""
    # Start scheduler:
    # 1. If not in debug mode, start immediately.
    # 2. If in debug mode, start only in the child (reloader) process to avoid duplicates.
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    is_debug = config.DEBUG

    if not is_debug or is_reloader:
        _start_scheduler()

    # Cold start check
    if not os.path.exists(config.MODEL_MED_PATH):
        print("\n[Cold Start] Model not found. Initializing training...")
        try:
            script_path = os.path.join(config.BASE_DIR, "bin", "run_training.py")
            subprocess.run(
                [sys.executable, script_path, "--days", "1"],
                check=True
            )
        except Exception as e:
            print(f"Cold start training failed: {e}")

    app = create_app()
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
