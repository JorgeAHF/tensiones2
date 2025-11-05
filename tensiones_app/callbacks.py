"""Dash callbacks for the tension monitoring app."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from .analysis import analyse_signal, get_directory_files, load_and_prepare_data_from_file


EMPTY_FIGURE = go.Figure()


def register_callbacks(app: Dash) -> None:
    """Register all Dash callbacks."""

    @app.callback(Output("polling-interval", "interval"), Input("poll-seconds", "value"))
    def update_interval(seconds: Optional[int]) -> int:
        if seconds is None or seconds < 1:
            return 30000
        return int(seconds * 1000)

    @app.callback(
        Output("files-store", "data"),
        Output("file-dropdown", "options"),
        Output("file-dropdown", "value"),
        Input("polling-interval", "n_intervals"),
        Input("directory-input", "value"),
        State("file-dropdown", "value"),
    )
    def refresh_file_list(_: int, directory: str, current_value: Optional[str]):
        files = get_directory_files(directory or "")
        options = [
            {
                "label": f"{os.path.basename(path)} ({datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')})",
                "value": path,
            }
            for path in files
        ]
        value = current_value if current_value in files else (files[0] if files else None)
        return files, options, value

    @app.callback(
        Output("data-store", "data"),
        Output("sensor-dropdown", "options"),
        Output("sensor-dropdown", "value"),
        Output("file-info", "children"),
        Output("error-message", "children", allow_duplicate=True),
        Input("file-dropdown", "value"),
        Input("map-textarea", "value"),
    )
    def load_file(path: Optional[str], mapping_text: str):
        if not path:
            raise PreventUpdate

        try:
            sensor_map: Dict[str, str] = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            return None, [], None, "", f"Error en mapeo JSON: {exc}"

        try:
            df = load_and_prepare_data_from_file(path, sensor_map)
        except Exception as exc:  # pylint: disable=broad-except
            return None, [], None, "", f"Error al leer archivo: {exc}"

        store_data = {"df": df.to_json(orient="split")}
        sensors = [col for col in df.columns if col != df.columns[0]]
        sensor_options = [{"label": col, "value": col} for col in sensors]
        sensor_value = sensors[0] if sensors else None
        info = f"Total de muestras: {len(df)}"
        return store_data, sensor_options, sensor_value, info, ""

    @app.callback(
        Output("accelerogram-full", "figure"),
        Output("accelerogram-segment", "figure"),
        Output("psd-graph", "figure"),
        Output("stft-graph", "figure"),
        Output("results-table", "columns"),
        Output("results-table", "data"),
        Output("pct-label", "children"),
        Output("error-message", "children", allow_duplicate=True),
        Input("data-store", "data"),
        Input("sensor-dropdown", "value"),
        Input("fs-input", "value"),
        Input("pct-range", "value"),
        Input("nperseg-input", "value"),
        Input("noverlap-input", "value"),
        Input("sigma-input", "value"),
        Input("threshold-input", "value"),
        Input("min-distance-input", "value"),
        Input("harmonics-input", "value"),
        Input("use-f0-hint", "value"),
        Input("f0-hint-input", "value"),
        Input("tol-input", "value"),
        Input("length-input", "value"),
        Input("density-input", "value"),
        prevent_initial_call=True,
    )
    def update_analysis(
        store_data: Optional[Dict[str, Any]],
        sensor: Optional[str],
        fs: Optional[float],
        pct_range,
        nperseg,
        noverlap,
        sigma,
        threshold,
        min_distance,
        n_harmonics,
        use_hint_values,
        f0_hint,
        tol_hz,
        length_m,
        linear_density,
    ):
        if not store_data or not sensor:
            return EMPTY_FIGURE, EMPTY_FIGURE, EMPTY_FIGURE, EMPTY_FIGURE, [], [], "", ""

        df = pd.read_json(store_data["df"], orient="split")
        use_hint = "use" in (use_hint_values or [])

        try:
            (
                accel_full_fig,
                accel_segment_fig,
                psd_fig,
                stft_fig,
                results,
                (start_s, end_s, duration_s),
            ) = analyse_signal(
                df=df,
                sensor=sensor,
                fs=fs,
                pct_range=pct_range,
                nperseg=nperseg,
                noverlap=noverlap,
                smooth_sigma=sigma,
                threshold=threshold,
                min_distance_hz=min_distance,
                n_harmonics=n_harmonics,
                use_hint=use_hint,
                f0_hint=f0_hint,
                tol_hz=tol_hz,
                length_m=length_m,
                linear_density=linear_density,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return (
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                [],
                [],
                "",
                str(exc),
            )

        pct_label = f"Rango temporal: {start_s:.2f} s → {end_s:.2f} s (duración: {duration_s:.2f} s)"

        if use_hint and results.refined_from_hint:
            fundamental_display = results.refined_from_hint
            harmonics_display = results.harmonics_from_hint
        else:
            fundamental_display = results.refined_fundamental
            harmonics_display = results.harmonics

        table_data = {
            "Frecuencia Fundamental [Hz]": (
                f"{fundamental_display:.2f}" if fundamental_display is not None else "—"
            ),
            "Armónicos detectados [Hz]": ", ".join(
                f"{harmonic:.2f}" for harmonic in harmonics_display
            )
            if harmonics_display
            else "—",
            "Tensión estimada [N]": (
                f"{results.tension:.2f}" if results.tension is not None else "—"
            ),
        }

        columns = [{"name": name, "id": name} for name in table_data.keys()]
        data = [{key: value for key, value in table_data.items()}]

        return (
            accel_full_fig,
            accel_segment_fig,
            psd_fig,
            stft_fig,
            columns,
            data,
            pct_label,
            "",
        )
