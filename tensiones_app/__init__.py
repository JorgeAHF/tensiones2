"""Application factory for the tension monitoring Dash project."""
from __future__ import annotations

from dash import Dash

from .callbacks import register_callbacks
from .layout import build_layout


def create_app() -> Dash:
    """Instantiate and configure the Dash application."""

    app = Dash(__name__)
    app.title = "Monitor de tensión"
    app.layout = build_layout()
    register_callbacks(app)
    return app
