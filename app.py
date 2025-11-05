"""Entry point for running the Dash application."""
from __future__ import annotations

from tensiones_app import create_app

app = create_app()
server = app.server


if __name__ == "__main__":
    app.run_server(debug=True)
