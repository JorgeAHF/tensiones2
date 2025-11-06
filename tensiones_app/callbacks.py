"""Dash callbacks for the tension monitoring app."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from .analysis import analyse_signal, get_directory_files, load_and_prepare_data_from_file
from .storage import save_mapping_text, save_sensor_config_store


EMPTY_FIGURE = go.Figure()
DEFAULT_SENSOR_STATUS = "Configure los sensores para habilitar la lectura de datos."


def build_sensor_config_payload(
    rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    """Validate and summarise the sensor configuration rows."""

    complete = True
    issues: list[str] = []
    by_sensor: Dict[str, Dict[str, Optional[float]]] = {}
    seen: set[str] = set()

    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for idx, row in enumerate(rows, start=1):
        column = row.get("column")
        tirante = (row.get("tirante") or "").strip()
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


def register_callbacks(app: Dash) -> None:
    """Register all Dash callbacks."""

    @app.callback(Output("polling-interval", "interval"), Input("poll-seconds", "value"))
    def update_interval(seconds: Optional[int]) -> int:
        if seconds is None or seconds < 1:
            return 30000
        return int(seconds * 1000)

    def _normalize_directory(path: str) -> str:
        """Return a validated absolute directory path."""

        if not path:
            raise ValueError("ruta vacía.")

        expanded = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(expanded):
            raise ValueError("la carpeta no existe.")
        if not os.path.isdir(expanded):
            raise ValueError("la ruta no es una carpeta.")
        if not os.access(expanded, os.R_OK):
            raise ValueError("no hay permisos de lectura.")

        return expanded

    def _list_subdirectories(path: str) -> list[dict[str, str]]:
        """Return dropdown options for the subdirectories of ``path``."""

        try:
            entries = [
                (entry.name, entry.path)
                for entry in os.scandir(path)
                if entry.is_dir()
            ]
        except OSError as exc:
            raise ValueError(str(exc)) from exc

        entries.sort(key=lambda item: item[0].lower())
        return [{"label": name, "value": full_path} for name, full_path in entries]

    def _breadcrumb_label(path: str) -> str:
        if not path:
            return "Ninguna carpeta seleccionada."
        return f"Carpeta actual: {path}"

    @app.callback(
        Output("directory-browser-store", "data"),
        Output("error-message", "children", allow_duplicate=True),
        Input("directory-input", "value"),
        Input("directory-browser-dropdown", "value"),
        Input("directory-browser-up", "n_clicks"),
        State("directory-browser-store", "data"),
        prevent_initial_call=True,
    )
    def update_directory_browser(
        manual_path: Optional[str],
        selected_subdir: Optional[str],
        up_clicks: Optional[int],
        store_data: Optional[Dict[str, Any]],
    ):
        triggered = ctx.triggered_id
        store_data = store_data or {}
        current_path = store_data.get("path") or ""

        if triggered == "directory-browser-up":
            if not current_path:
                raise PreventUpdate
            parent = os.path.abspath(os.path.join(current_path, os.pardir))
            if parent == current_path:
                raise PreventUpdate
            target = parent
        elif triggered == "directory-browser-dropdown":
            if not selected_subdir:
                raise PreventUpdate
            target = selected_subdir
        elif triggered == "directory-input":
            if manual_path is None or manual_path == current_path:
                raise PreventUpdate
            target = manual_path
        else:
            raise PreventUpdate

        try:
            normalized = _normalize_directory(target)
        except ValueError as exc:
            return store_data, f"No se puede acceder a '{target}': {exc}"

        return {"path": normalized}, ""

    @app.callback(
        Output("directory-input", "value"),
        Output("directory-browser-dropdown", "options"),
        Output("directory-browser-dropdown", "value"),
        Output("directory-browser-breadcrumb", "children"),
        Output("directory-browser-up", "disabled"),
        Output("error-message", "children", allow_duplicate=True),
        Input("directory-browser-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_directory_browser(store_data: Optional[Dict[str, Any]]):
        path = ""
        if isinstance(store_data, dict):
            path = store_data.get("path") or ""

        if not path:
            breadcrumb = _breadcrumb_label(path)
            return "", [], None, breadcrumb, True, ""

        try:
            options = _list_subdirectories(path)
        except ValueError as exc:
            return "", [], None, _breadcrumb_label(""), True, f"Error al explorar la carpeta: {exc}"

        breadcrumb = _breadcrumb_label(path)
        parent = os.path.abspath(os.path.join(path, os.pardir))
        disable_up = parent == path

        return path, options, None, breadcrumb, disable_up, ""

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

        save_mapping_text(sensor_map)

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
            save_sensor_config_store(None)
            return None, DEFAULT_SENSOR_STATUS

        store_payload, status = build_sensor_config_payload(rows)
        save_sensor_config_store(store_payload)
        return store_payload, status

    @app.callback(
        Output("files-store", "data"),
        Output("active-file-store", "data"),
        Output("processing-status", "children", allow_duplicate=True),
        Output("processing-metadata-store", "data", allow_duplicate=True),
        Input("polling-interval", "n_intervals"),
        Input("directory-input", "value"),
        Input("sensor-config-store", "data"),
        Input("processing-state", "data"),
        State("active-file-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_file_list(
        _: int,
        directory: str,
        config_data: Optional[Dict[str, Any]],
        processing_state: Optional[str],
        active_file,
    ):
        config_complete = bool(config_data and config_data.get("complete"))
        if not directory or not config_complete:
            status_message = (
                "Defina una carpeta de trabajo y complete la configuración de los tirantes"
                " para comenzar a monitorear."
            )
            metadata_payload = {
                "pending": 0,
                "active": None,
                "last_poll": None,
                "last_poll_display": None,
            }
            return [], None, status_message, metadata_payload

        triggered = ctx.triggered_id
        if triggered == "polling-interval" and processing_state != "running":
            raise PreventUpdate

        raw_files = get_directory_files(directory or "")
        files = []
        for path in raw_files:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            files.append({"path": path, "name": os.path.basename(path), "mtime": mtime})

        if isinstance(active_file, str):
            active_metadata = next(
                (item for item in files if item["path"] == active_file), None
            )
        elif isinstance(active_file, dict):
            active_metadata = active_file
        else:
            active_metadata = None

        candidate_file = (
            active_metadata if active_metadata in files else (files[0] if files else None)
        )
        next_file = candidate_file if processing_state == "running" else None

        now = datetime.now()
        poll_display = now.strftime("%H:%M:%S")
        pending_count = len(files)
        metadata_payload = {
            "pending": pending_count,
            "active": (candidate_file or {}).get("name"),
            "last_poll": now.isoformat(),
            "last_poll_display": poll_display,
        }

        base_status = {
            "running": "Lectura en ejecución.",
            "paused": "Lectura pausada.",
            "stopped": "Lectura detenida.",
        }.get(processing_state or "stopped", "Lectura detenida.")

        status_parts = []
        if pending_count:
            status_parts.append(f"{pending_count} archivo(s) pendientes")
        else:
            status_parts.append("Sin archivos pendientes")

        candidate_name = (candidate_file or {}).get("name")
        if candidate_name:
            status_parts.append(f"Próximo archivo: {candidate_name}")

        status_parts.append(f"Último sondeo: {poll_display}")

        status_message = f"{base_status} {' · '.join(status_parts)}"

        return files, next_file, status_message, metadata_payload

    @app.callback(
        Output("processing-state", "data"),
        Output("polling-interval", "disabled"),
        Output("processing-status", "children"),
        Output("active-file-store", "data", allow_duplicate=True),
        Output("data-store", "data", allow_duplicate=True),
        Output("file-info", "children", allow_duplicate=True),
        Output("processing-metadata-store", "data", allow_duplicate=True),
        Input("start-processing", "n_clicks"),
        Input("pause-processing", "n_clicks"),
        Input("stop-processing", "n_clicks"),
        State("processing-state", "data"),
        State("sensor-config-store", "data"),
        State("directory-input", "value"),
        State("processing-metadata-store", "data"),
        prevent_initial_call=True,
    )
    def control_processing(
        start_clicks: Optional[int],
        pause_clicks: Optional[int],
        stop_clicks: Optional[int],
        current_state: Optional[str],
        config_data: Optional[Dict[str, Any]],
        directory: Optional[str],
        metadata_state: Optional[Dict[str, Any]],
    ):
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        current_state = current_state or "stopped"
        config_complete = bool(config_data and config_data.get("complete"))

        metadata_state = metadata_state or {}

        def _metadata_suffix() -> str:
            pending = metadata_state.get("pending")
            last_poll_display = metadata_state.get("last_poll_display")
            pieces = []
            if pending is not None:
                pieces.append(f"Pendientes: {pending}")
            if metadata_state.get("active"):
                pieces.append(f"Próximo archivo: {metadata_state['active']}")
            if last_poll_display:
                pieces.append(f"Último sondeo: {last_poll_display}")
            return " · ".join(pieces)

        if trigger == "start-processing":
            if not directory:
                message = "Seleccione una carpeta válida en el explorador antes de iniciar."
                return (
                    current_state,
                    True,
                    message,
                    no_update,
                    no_update,
                    message,
                    no_update,
                )
            if not config_complete:
                message = "Complete la configuración de los tirantes antes de iniciar."
                return (
                    current_state,
                    True,
                    message,
                    no_update,
                    no_update,
                    message,
                    no_update,
                )

            suffix = _metadata_suffix()
            status_message = "Lectura en ejecución."
            if suffix:
                status_message = f"{status_message} {suffix}"
            info_message = "Buscando archivos nuevos en la carpeta seleccionada."
            return (
                "running",
                False,
                status_message,
                no_update,
                no_update,
                info_message,
                no_update,
            )

        if trigger == "pause-processing":
            suffix = _metadata_suffix()
            status_message = "Lectura pausada. Presione Iniciar para reanudar."
            if suffix:
                status_message = f"{status_message} {suffix}"
            return (
                "paused",
                True,
                status_message,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        if trigger == "stop-processing":
            suffix = _metadata_suffix()
            status_message = "Lectura detenida. Presione Iniciar para comenzar."
            if suffix:
                status_message = f"{status_message} {suffix}"
            return (
                "stopped",
                True,
                status_message,
                None,
                None,
                "Lectura detenida. Seleccione una carpeta cuando desee reanudar.",
                None,
            )

        raise PreventUpdate

    @app.callback(
        Output("data-store", "data", allow_duplicate=True),
        Output("sensor-dropdown", "options"),
        Output("sensor-dropdown", "value"),
        Output("file-info", "children", allow_duplicate=True),
        Output("error-message", "children", allow_duplicate=True),
        Output("processed-history-store", "data", allow_duplicate=True),
        Input("active-file-store", "data"),
        State("map-textarea", "value"),
        State("sensor-config-store", "data"),
        State("processed-history-store", "data"),
        prevent_initial_call=True,
    )
    def load_file(
        path_info,
        mapping_text: Optional[str],
        config_data: Optional[Dict[str, Any]],
        history_state,
    ):
        if isinstance(path_info, str):
            path = path_info
        elif isinstance(path_info, dict):
            path = path_info.get("path")
        else:
            raise PreventUpdate

        if not isinstance(path, str) or not path:
            raise PreventUpdate

        if not config_data or not config_data.get("complete"):
            return (
                None,
                [],
                None,
                "En espera de completar la configuración de los tirantes.",
                "Complete la configuración de los tirantes antes de continuar.",
                no_update,
            )

        try:
            sensor_map: Dict[str, str] = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            return None, [], None, "", f"Error en mapeo JSON: {exc}", no_update

        try:
            df = load_and_prepare_data_from_file(path, sensor_map)
        except Exception as exc:  # pylint: disable=broad-except
            return None, [], None, "", f"Error al leer archivo: {exc}", no_update

        by_sensor: Dict[str, Dict[str, Any]] = config_data.get("by_sensor", {})
        sensors = [col for col in df.columns if col in by_sensor]
        if not sensors:
            return (
                None,
                [],
                None,
                "",
                "El archivo no contiene tirantes configurados.",
                no_update,
            )
        timestamp = datetime.now()
        timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        store_data = {"df": df.to_json(orient="split")}
        sensor_options = [{"label": col, "value": col} for col in sensors]
        sensor_value = sensors[0]

        processed_dir = os.path.join(os.path.dirname(path), "procesados")
        os.makedirs(processed_dir, exist_ok=True)
        destination = os.path.join(processed_dir, os.path.basename(path))
        if os.path.exists(destination):
            base, ext = os.path.splitext(os.path.basename(path))
            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = os.path.join(processed_dir, f"{base}_{timestamp_suffix}{ext}")

        try:
            shutil.move(path, destination)
        except Exception as exc:  # pylint: disable=broad-except
            info = f"Archivo: {os.path.basename(path)}. Total de muestras: {len(df)}. No se pudo mover el archivo: {exc}"
            status_note = f"Procesado con alertas · {timestamp_display}"
        else:
            info = (
                f"Archivo: {os.path.basename(path)}. Total de muestras: {len(df)}. "
                f"Movido a 'procesados/{os.path.basename(destination)}'."
            )
            status_note = f"Procesado correctamente · {timestamp_display}"

        history_list: list[dict[str, Any]] = []
        if isinstance(history_state, list):
            history_list = [entry for entry in history_state if isinstance(entry, dict)]

        history_entry = {
            "name": os.path.basename(path),
            "processed_at": timestamp.isoformat(),
            "processed_at_display": timestamp_display,
            "note": info,
            "status": status_note,
        }
        history_list.insert(0, history_entry)
        history_list = history_list[:10]

        return store_data, sensor_options, sensor_value, info, "", history_list

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

    @app.callback(
        Output("processed-history-list", "children"),
        Input("processed-history-store", "data"),
    )
    def render_processed_history(history_data):
        if not history_data:
            return [
                html.Li(
                    "Aún no se han procesado archivos.",
                    className="history-item history-empty",
                )
            ]

        items = []
        for entry in history_data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "Archivo desconocido")
            processed_at = entry.get("processed_at_display") or entry.get(
                "processed_at"
            )
            status = entry.get("status")
            note = entry.get("note")
            items.append(
                html.Li(
                    [
                        html.Span(name, className="history-name"),
                        html.Span(processed_at or "", className="history-timestamp"),
                        html.Span(status or "", className="history-status"),
                        html.Span(note or "", className="history-note"),
                    ],
                    className="history-item",
                )
            )

        return items or [
            html.Li(
                "No hay registros disponibles.",
                className="history-item history-empty",
            )
        ]

    @app.callback(
        Output("processing-progress", "value"),
        Output("processing-progress", "max"),
        Output("processing-progress", "title"),
        Input("processing-state", "data"),
        State("processing-metadata-store", "data"),
    )
    def update_processing_indicator(
        processing_state: Optional[str], metadata: Optional[Dict[str, Any]]
    ):
        metadata = metadata or {}
        state = processing_state or "stopped"

        if state == "running":
            value = 100
        elif state == "paused":
            value = 50
        else:
            value = 0

        pending = metadata.get("pending")
        next_name = metadata.get("active")
        last_poll_display = metadata.get("last_poll_display")

        title_parts = []
        if pending is not None:
            title_parts.append(f"Pendientes: {pending}")
        if next_name:
            title_parts.append(f"Próximo: {next_name}")
        if last_poll_display:
            title_parts.append(f"Último sondeo: {last_poll_display}")

        if not title_parts:
            title_parts.append("Sin actividad registrada")

        return value, 100, " · ".join(title_parts)
