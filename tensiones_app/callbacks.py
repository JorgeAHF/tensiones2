"""Dash callbacks for the tension monitoring app."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, no_update
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
        Output("directory-input", "value"),
        Output("error-message", "children", allow_duplicate=True),
        Input("select-directory-button", "n_clicks"),
        State("directory-input", "value"),
        prevent_initial_call=True,
    )
    def select_directory(n_clicks: Optional[int], current_value: Optional[str]):
        if not n_clicks:
            raise PreventUpdate

        selected_dir: Optional[str] = None
        root = None

        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initial_dir = current_value if current_value else os.getcwd()
            selected_dir = filedialog.askdirectory(initialdir=initial_dir)
        except Exception as exc:  # pylint: disable=broad-except
            if root is not None:
                root.destroy()
            return current_value, f"No se pudo abrir el selector de directorio: {exc}"
        finally:
            if root is not None:
                root.destroy()

        if not selected_dir:
            raise PreventUpdate

        return selected_dir, ""

    @app.callback(
        Output("sensor-config-table", "data"),
        Output("mapping-status", "children"),
        Output("sensor-config-store", "data"),
        Output("error-message", "children", allow_duplicate=True),
        Input("apply-map-button", "n_clicks"),
        State("map-textarea", "value"),
        prevent_initial_call=True,
    )
    def apply_mapping(n_clicks: Optional[int], mapping_text: Optional[str]):
        if not n_clicks:
            raise PreventUpdate

        try:
            sensor_map: Dict[str, str] = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            return no_update, "", None, f"Error en mapeo JSON: {exc}"

        if not sensor_map:
            return [], "Defina al menos un sensor en el mapeo.", None, ""

        table_rows = []
        for original, alias in sensor_map.items():
            if isinstance(alias, dict):
                tirante_value = alias.get("tirante") or original
                f0_value = alias.get("f0")
                ke_value = alias.get("ke")
            else:
                tirante_value = alias or original
                f0_value = None
                ke_value = None

            table_rows.append(
                {
                    "column": original,
                    "tirante": tirante_value,
                    "f0": f0_value,
                    "ke": ke_value,
                }
            )

        mapping_message = (
            f"Mapeo aplicado para {len(table_rows)} tirantes. Complete f₀ y Ke para continuar."
        )
        store_payload = {"rows": table_rows, "complete": False, "by_sensor": {}}
        return table_rows, mapping_message, store_payload, ""

    @app.callback(
        Output("sensor-config-store", "data", allow_duplicate=True),
        Output("sensor-config-status", "children"),
        Input("sensor-config-table", "data"),
        prevent_initial_call=True,
    )
    def sync_sensor_config(rows: Optional[list[dict[str, Any]]]):
        if not rows:
            return None, "Configure los sensores para habilitar la lectura de datos."

        complete = True
        issues: list[str] = []
        by_sensor: Dict[str, Dict[str, Optional[float]]] = {}
        seen: set[str] = set()

        for idx, row in enumerate(rows, start=1):
            column = row.get("column")
            tirante = (row.get("tirante") or "").strip()

            def _to_float(value: Any) -> Optional[float]:
                if value in (None, ""):
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            f0 = _to_float(row.get("f0"))
            ke_value = _to_float(row.get("ke"))

            if not tirante:
                complete = False
                issues.append(f"Fila {idx}: faltó definir el nombre del tirante.")
            elif tirante in seen:
                complete = False
                issues.append(f"Fila {idx}: el tirante '{tirante}' está duplicado.")
            else:
                seen.add(tirante)

            if f0 is None or f0 <= 0:
                complete = False
            if ke_value is None or ke_value <= 0:
                complete = False

            if tirante:
                by_sensor[tirante] = {
                    "column": column,
                    "f0": f0,
                    "ke": ke_value,
                }

        status = (
            "Parámetros completos. Los archivos nuevos se analizarán automáticamente."
            if complete
            else "Complete la frecuencia fundamental y el valor de Ke para cada tirante."
        )
        if issues:
            status = " | ".join([status] + issues)

        store_payload = {"rows": rows, "complete": complete, "by_sensor": by_sensor}
        return store_payload, status

    @app.callback(
        Output("files-store", "data"),
        Output("active-file-store", "data"),
        Input("polling-interval", "n_intervals"),
        Input("directory-input", "value"),
        Input("sensor-config-store", "data"),
        State("active-file-store", "data"),
    )
    def refresh_file_list(
        _: int,
        directory: str,
        config_data: Optional[Dict[str, Any]],
        active_file: Optional[str],
    ):
        config_complete = bool(config_data and config_data.get("complete"))
        if not directory or not config_complete:
            return [], None

        files = get_directory_files(directory or "")
        next_file = active_file if active_file in files else (files[0] if files else None)
        return files, next_file

    @app.callback(
        Output("data-store", "data"),
        Output("sensor-dropdown", "options"),
        Output("sensor-dropdown", "value"),
        Output("file-info", "children"),
        Output("error-message", "children", allow_duplicate=True),
        Output("active-file-store", "data", allow_duplicate=True),
        Input("active-file-store", "data"),
        State("map-textarea", "value"),
        State("sensor-config-store", "data"),
        prevent_initial_call=True,
    )
    def load_file(
        path: Optional[str],
        mapping_text: Optional[str],
        config_data: Optional[Dict[str, Any]],
    ):
        if not path:
            raise PreventUpdate

        if not config_data or not config_data.get("complete"):
            return (
                None,
                [],
                None,
                "En espera de completar la configuración de los tirantes.",
                "Complete la configuración de los tirantes antes de continuar.",
                None,
            )

        try:
            sensor_map: Dict[str, str] = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            return None, [], None, "", f"Error en mapeo JSON: {exc}", None

        try:
            df = load_and_prepare_data_from_file(path, sensor_map)
        except Exception as exc:  # pylint: disable=broad-except
            return None, [], None, "", f"Error al leer archivo: {exc}", None

        by_sensor: Dict[str, Dict[str, Any]] = config_data.get("by_sensor", {})
        sensors = [col for col in df.columns if col in by_sensor]
        if not sensors:
            return None, [], None, "", "El archivo no contiene tirantes configurados.", None

        store_data = {"df": df.to_json(orient="split")}
        sensor_options = [{"label": col, "value": col} for col in sensors]
        sensor_value = sensors[0]

        processed_dir = os.path.join(os.path.dirname(path), "procesados")
        os.makedirs(processed_dir, exist_ok=True)
        destination = os.path.join(processed_dir, os.path.basename(path))
        if os.path.exists(destination):
            base, ext = os.path.splitext(os.path.basename(path))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = os.path.join(processed_dir, f"{base}_{timestamp}{ext}")

        try:
            shutil.move(path, destination)
        except Exception as exc:  # pylint: disable=broad-except
            info = f"Archivo: {os.path.basename(path)}. Total de muestras: {len(df)}. No se pudo mover el archivo: {exc}"
        else:
            info = (
                f"Archivo: {os.path.basename(path)}. Total de muestras: {len(df)}. "
                f"Movido a 'procesados/{os.path.basename(destination)}'."
            )

        return store_data, sensor_options, sensor_value, info, "", None

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
        Input("tol-input", "value"),
        Input("sensor-config-store", "data"),
        prevent_initial_call="initial_duplicate",
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
        tol_hz,
        sensor_config_data: Optional[Dict[str, Any]],
    ):
        if not store_data or not sensor:
            return EMPTY_FIGURE, EMPTY_FIGURE, EMPTY_FIGURE, EMPTY_FIGURE, [], [], "", ""

        sensor_params = (sensor_config_data or {}).get("by_sensor", {}).get(sensor)
        if not sensor_params:
            return (
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                EMPTY_FIGURE,
                [],
                [],
                "",
                "No se encontraron parámetros configurados para el tirante seleccionado.",
            )

        df = pd.read_json(store_data["df"], orient="split")
        f0_hint = sensor_params.get("f0")
        ke_value = sensor_params.get("ke")
        use_hint = f0_hint is not None and f0_hint > 0

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
                length_m=None,
                linear_density=None,
                ke_ton_s=ke_value,
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

        tension_units = "Ton" if (ke_value is not None and ke_value > 0) else "N"
        table_data = {
            "Frecuencia Fundamental [Hz]": (
                f"{fundamental_display:.2f}" if fundamental_display is not None else "—"
            ),
            "Armónicos detectados [Hz]": ", ".join(
                f"{harmonic:.2f}" for harmonic in harmonics_display
            )
            if harmonics_display
            else "—",
            f"Tensión estimada [{tension_units}]": (
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

    @app.callback(
        Output("sensor-dropdown", "disabled"),
        Input("data-store", "data"),
    )
    def toggle_sensor_dropdown(store_data: Optional[Dict[str, Any]]) -> bool:
        return not bool(store_data)

    @app.callback(
        Output("selected-sensor-summary", "children"),
        Input("sensor-dropdown", "value"),
        Input("sensor-config-store", "data"),
    )
    def update_selected_sensor_summary(
        sensor: Optional[str], config_data: Optional[Dict[str, Any]]
    ) -> str:
        if not sensor or not config_data:
            return ""

        params = (config_data.get("by_sensor") or {}).get(sensor)
        if not params:
            return "No hay parámetros guardados para el tirante seleccionado."

        f0 = params.get("f0")
        ke_value = params.get("ke")
        details = []
        if f0:
            details.append(f"f₀ inicial: {f0:.2f} Hz")
        if ke_value:
            details.append(f"Ke: {ke_value:.3f} Ton·s")
        return " · ".join(details) if details else ""
