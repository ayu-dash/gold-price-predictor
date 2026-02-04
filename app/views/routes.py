"""Template-rendering view routes."""

from flask import render_template

from app.views import views_bp


@views_bp.route("/")
def index():
    """Serve the dashboard home page."""
    return render_template("index.html")


@views_bp.route("/analytics")
def analytics():
    """Serve the yearly analytics page."""
    return render_template("analytics.html")


@views_bp.route("/charts")
def charts():
    """Serve the detailed charts page."""
    return render_template("charts.html")


@views_bp.route("/signals")
def signals():
    """Serve the signals analysis page."""
    return render_template("signals.html")


@views_bp.route("/history")
def history():
    """Serve the history table page."""
    return render_template("history.html")
