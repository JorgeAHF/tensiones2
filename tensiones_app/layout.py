"""Dash layout factory."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from dash import dcc, html
from dash.dash_table import DataTable

from .callbacks import DEFAULT_SENSOR_STATUS, build_sensor_config_payload
from .storage import (
    load_last_mapping_text,
    load_sensor_config_store,
    save_sensor_config_store,
)


def build_layout() -> html.Div:
    """Return the root layout for the application."""

    mapping_text = load_last_mapping_text()
    try:
        mapping_definition = json.loads(mapping_text)
    except json.JSONDecodeError:
        mapping_definition = {}
    if not isinstance(mapping_definition, dict):
        mapping_definition = {}

    stored_config = load_sensor_config_store()
    stored_rows: list[dict[str, Any]] = []
    if isinstance(stored_config, dict):
        rows = stored_config.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("column"), str):
                    stored_rows.append(row)

    stored_by_column: Dict[str, dict[str, Any]] = {
        row["column"]: row for row in stored_rows if isinstance(row.get("column"), str)
    }

    def _default_row(column: str, alias: Any) -> dict[str, Any]:
        tirante = column
        f0_value = None
        ke_value = None
        if isinstance(alias, dict):
            tirante = alias.get("tirante") or tirante
            f0_value = alias.get("f0")
            ke_value = alias.get("ke")
        elif isinstance(alias, str):
            tirante = alias or tirante
        return {
            "column": column,
            "tirante": tirante,
            "f0": f0_value,
            "ke": ke_value,
        }

    initial_rows: list[dict[str, Any]] = []
    for column, alias in mapping_definition.items():
        if not isinstance(column, str):
            continue
        base_row = _default_row(column, alias)
        saved_row = stored_by_column.get(column)
        if saved_row:
            raw_tirante = saved_row.get("tirante") or base_row["tirante"] or column
            tirante_value = raw_tirante.strip() if isinstance(raw_tirante, str) else str(raw_tirante)
            row = {
                "column": column,
                "tirante": tirante_value,
                "f0": saved_row.get("f0") if saved_row.get("f0") not in ("", None) else base_row["f0"],
                "ke": saved_row.get("ke") if saved_row.get("ke") not in ("", None) else base_row["ke"],
            }
        else:
            row = base_row
        initial_rows.append(row)

    if initial_rows:
        sensor_store_data, sensor_status = build_sensor_config_payload(initial_rows)
        save_sensor_config_store(sensor_store_data)
    else:
        sensor_store_data = None
        sensor_status = DEFAULT_SENSOR_STATUS
        if stored_config is not None:
            save_sensor_config_store(None)

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Monitor automático de tensión", className="app-title"),
                    html.P(
                        "Configure y supervise los parámetros clave de cada tirante en tiempo real.",
                        className="app-subtitle",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Paso 1", className="step-badge"),
                                    html.Div(
                                        [
                                            html.H3(
                                                "Directorio de trabajo",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Seleccione la carpeta que contiene los archivos CSV generados por el sistema y defina cada cuánto se actualizará la lectura.",
                                                className="section-description",
                                            ),
                                        ],
                                        className="panel-heading-text",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Div(
                                [
                                    html.Label("Directorio de datos"),
                                    dcc.Input(
                                        id="directory-input",
                                        type="text",
                                        value="",
                                        debounce=True,
                                        placeholder="Ej: C:/monitoreo/tirantes",
                                        style={"width": "100%"},
                                    ),
                                ],
                                className="input-stack",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Explorar subcarpetas"),
                                            html.Div(
                                                "Ninguna carpeta seleccionada.",
                                                id="directory-browser-breadcrumb",
                                                className="info",
                                            ),
                                            dcc.Dropdown(
                                                id="directory-browser-dropdown",
                                                options=[],
                                                placeholder="Seleccione una subcarpeta",
                                                clearable=False,
                                            ),
                                            html.Button(
                                                "Subir un nivel",
                                                id="directory-browser-up",
                                                className="directory-button",
                                                n_clicks=0,
                                                disabled=True,
                                            ),
                                        ],
                                        className="input-stack",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Intervalo de actualización (s)"),
                                            dcc.Input(
                                                id="poll-seconds",
                                                type="number",
                                                min=5,
                                                max=300,
                                                step=5,
                                                value=30,
                                            ),
                                        ],
                                        className="control-inline",
                                    ),
                                ],
                                className="panel-actions",
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Paso 2", className="step-badge"),
                                    html.Div(
                                        [
                                            html.H3(
                                                "Mapeo y parámetros",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Asigne un nombre a cada tirante, indique su frecuencia fundamental propuesta y el valor de Ke (Ton·s).",
                                                className="section-description",
                                            ),
                                        ],
                                        className="panel-heading-text",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Label("Mapeo de sensores (JSON)"),
                            dcc.Textarea(
                                id="map-textarea",
                                value=load_last_mapping_text(),
                                placeholder='{"canal_raw": {"tirante": "Tirante 1", "f0": 1.2, "ke": 0.45}}',
                                style={"width": "100%", "height": "140px"},
                            ),
                            html.Details(
                                [
                                    html.Summary("Ver ejemplo completo de mapeo"),
                                    html.Div(
                                        [
                                            html.P(
                                                "Puedes copiar y adaptar el siguiente formato para mapear múltiples sensores:",
                                                className="section-description",
                                            ),
                                            html.Pre(
                                                '{\n    "canal_x": {"tirante": "Tirante Norte", "f0": 1.35, "ke": 0.42},\n    "canal_y": {"tirante": "Tirante Sur", "f0": 1.18, "ke": 0.38},\n    "canal_z": {"tirante": "Tirante Central", "f0": 1.42, "ke": 0.47}\n}',
                                                className="code-example",
                                            ),
                                            html.P(
                                                "Cada clave representa el nombre de la columna en el archivo CSV y el valor define el tirante asociado junto a sus parámetros iniciales.",
                                                className="info-note",
                                            ),
                                        ],
                                        className="example-card",
                                    ),
                                ],
                                className="example-details",
                            ),
                            html.Button(
                                "Aplicar mapeo",
                                id="apply-map-button",
                                className="directory-button",
                                n_clicks=0,
                            ),
                            html.Div(id="mapping-status", className="info"),
                            DataTable(
                                id="sensor-config-table",
                                columns=[
                                    {
                                        "name": "Columna original",
                                        "id": "column",
                                        "editable": False,
                                    },
                                    {
                                        "name": "Tirante",
                                        "id": "tirante",
                                        "editable": True,
                                    },
                                    {
                                        "name": "f₀ propuesta (Hz)",
                                        "id": "f0",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                    {
                                        "name": "Ke (Ton·s)",
                                        "id": "ke",
                                        "type": "numeric",
                                        "editable": True,
                                    },
                                ],
                                data=initial_rows,
                                editable=True,
                                style_header={
                                    "backgroundColor": "#eef2ff",
                                    "fontWeight": "600",
                                    "color": "#1e1b4b",
                                    "border": "0px",
                                },
                                style_cell={
                                    "backgroundColor": "#ffffff",
                                    "border": "0px",
                                    "color": "#1f2937",
                                    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "column"},
                                        "fontWeight": "600",
                                        "color": "#1e1b4b",
                                    }
                                ],
                            ),
                            html.Div(sensor_status, id="sensor-config-status", className="info"),
                        ],
                        className="panel",
                    ),
                ],
                className="layout",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Paso 3", className="step-badge"),
                                    html.Div(
                                        [
                                            html.H3(
                                                "Análisis y visualización",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Cuando se detecte un archivo nuevo se cargará automáticamente. Seleccione el tirante a visualizar.",
                                                className="section-description",
                                            ),
                                        ],
                                        className="panel-heading-text",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            html.Label("Archivo en análisis"),
                            html.Div(
                                "Seleccione una carpeta de datos y espere la llegada de nuevos archivos.",
                                id="file-info",
                                className="info",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Iniciar",
                                        id="start-processing",
                                        className="directory-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Pausar",
                                        id="pause-processing",
                                        className="directory-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Detener",
                                        id="stop-processing",
                                        className="directory-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="panel-actions",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Lectura detenida. Presione Iniciar para comenzar.",
                                        id="processing-status",
                                        className="info",
                                    ),
                                    html.Progress(
                                        id="processing-progress",
                                        max=100,
                                        value=0,
                                        className="processing-progress",
                                        title="Sin actividad registrada",
                                    ),
                                ],
                                className="processing-status-container",
                            ),
                            html.Br(),
                            html.Label("Tirante"),
                            dcc.Dropdown(
                                id="sensor-dropdown",
                                placeholder="Seleccione un tirante",
                                disabled=True,
                            ),
                            html.Div(id="selected-sensor-summary", className="info"),
                            html.Div(
                                [
                                    html.H4(
                                        "Historial de archivos procesados",
                                        className="subsection-title",
                                    ),
                                    html.Ul(
                                        id="processed-history-list",
                                        className="history-list",
                                    ),
                                ],
                                className="history-panel",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="layout",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Parámetros de análisis", className="section-title"),
                            html.P(
                                "Ajusta finamente la forma en que se procesa la señal para detectar la frecuencia fundamental con mayor confianza.",
                                className="section-description section-intro",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H4("Frecuencia y resolución", className="subsection-title"),
                                            html.P(
                                                "Controla la frecuencia de muestreo y la resolución espectral.",
                                                className="section-description",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Fs (Hz)"),
                                                            dcc.Input(
                                                                id="fs-input",
                                                                type="number",
                                                                min=10,
                                                                max=2000,
                                                                step=1,
                                                                value=128,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("nperseg"),
                                                            dcc.Input(
                                                                id="nperseg-input",
                                                                type="number",
                                                                min=256,
                                                                max=16384,
                                                                step=256,
                                                                value=4096,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("noverlap"),
                                                            dcc.Input(
                                                                id="noverlap-input",
                                                                type="number",
                                                                min=0,
                                                                max=16384,
                                                                step=256,
                                                                value=2048,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                ],
                                                className="control-grid",
                                            ),
                                        ],
                                        className="parameter-section",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Detección de picos", className="subsection-title"),
                                            html.P(
                                                "Configura los criterios para identificar la frecuencia fundamental y armónicos.",
                                                className="section-description",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Suavizado σ"),
                                                            dcc.Input(
                                                                id="sigma-input",
                                                                type="number",
                                                                min=0.1,
                                                                max=10.0,
                                                                step=0.1,
                                                                value=0.6,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Threshold"),
                                                            dcc.Input(
                                                                id="threshold-input",
                                                                type="number",
                                                                min=1e-12,
                                                                max=1e-3,
                                                                step=1e-7,
                                                                value=2.5e-7,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Min distancia (Hz)"),
                                                            dcc.Input(
                                                                id="min-distance-input",
                                                                type="number",
                                                                min=0.1,
                                                                max=10.0,
                                                                step=0.1,
                                                                value=0.3,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("N° Armónicos"),
                                                            dcc.Input(
                                                                id="harmonics-input",
                                                                type="number",
                                                                min=1,
                                                                max=5,
                                                                step=1,
                                                                value=2,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                ],
                                                className="control-grid",
                                            ),
                                        ],
                                        className="parameter-section",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Ventana temporal", className="subsection-title"),
                                            html.P(
                                                "Limita el tramo del archivo que se utiliza en el cálculo.",
                                                className="section-description",
                                            ),
                                            dcc.RangeSlider(
                                                id="pct-range",
                                                min=0,
                                                max=100,
                                                step=0.1,
                                                value=[0, 100],
                                                className="range-slider",
                                            ),
                                            html.Div(id="pct-label", className="info"),
                                        ],
                                        className="parameter-section",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Búsqueda guiada", className="subsection-title"),
                                            html.P(
                                                "El análisis utilizará la frecuencia fundamental inicial definida para el tirante seleccionado.",
                                                className="section-description",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Tol ± (Hz)"),
                                                            dcc.Input(
                                                                id="tol-input",
                                                                type="number",
                                                                min=0.01,
                                                                max=5.0,
                                                                step=0.01,
                                                                value=0.15,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                ],
                                                className="control-grid",
                                            ),
                                        ],
                                        className="parameter-section",
                                    ),
                                ],
                                className="parameter-stack",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="layout",
            ),
            html.Hr(className="divider"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Acelerograma completo", className="graph-title"),
                            dcc.Graph(id="accelerogram-full", config={"displaylogo": False}),
                        ],
                        className="graph-card",
                    ),
                    html.Div(
                        [
                            html.H3("Segmento seleccionado", className="graph-title"),
                            dcc.Graph(id="accelerogram-segment", config={"displaylogo": False}),
                        ],
                        className="graph-card",
                    ),
                ],
                className="graph-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Espectro de potencia", className="graph-title"),
                            dcc.Graph(id="psd-graph", config={"displaylogo": False}),
                        ],
                        className="graph-card",
                    ),
                    html.Div(
                        [
                            html.H3("Espectrograma (STFT)", className="graph-title"),
                            dcc.Graph(id="stft-graph", config={"displaylogo": False}),
                        ],
                        className="graph-card",
                    ),
                ],
                className="graph-row",
            ),
            html.H2("Resultados", className="section-title results-title"),
            DataTable(
                id="results-table",
                columns=[],
                data=[],
                style_header={
                    "backgroundColor": "transparent",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "color": "#0f172a",
                },
                style_cell={
                    "backgroundColor": "#ffffff",
                    "border": "0px",
                    "color": "#1f2937",
                    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
                    "padding": "12px",
                },
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "rgba(14, 165, 233, 0.2)",
                        "border": "0px",
                    }
                ],
            ),
            dcc.Interval(
                id="polling-interval", interval=30000, n_intervals=0, disabled=True
            ),
            dcc.Store(id="data-store"),
            dcc.Store(id="files-store"),
            dcc.Store(id="sensor-config-store", data=sensor_store_data),
            dcc.Store(id="active-file-store"),
            dcc.Store(id="processing-state", data="stopped"),
            dcc.Store(id="processing-metadata-store"),
            dcc.Store(id="processed-history-store", data=[]),
            dcc.Store(
                id="directory-browser-store",
                data={"path": os.getcwd()},
            ),
            html.Div(id="error-message", className="error"),
        ],
        className="app-container",
    )
