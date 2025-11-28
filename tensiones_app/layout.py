"""Dash layout factory."""
from __future__ import annotations

import os
from typing import Any, Dict

from dash import dcc, html
from dash.dash_table import DataTable

from .callbacks import DEFAULT_SENSOR_STATUS, build_sensor_config_payload
from .storage import load_sensor_config_store, save_sensor_config_store


DEFAULT_SENSOR_IDS = ["3808", "3810", "10589", "10598", "10603", "14030", "14031"]


def build_layout() -> html.Div:
    """Return the root layout for the application."""

    stored_config = load_sensor_config_store()
    stored_rows: list[dict[str, Any]] = []
    if isinstance(stored_config, dict):
        rows = stored_config.get("rows")
        if isinstance(rows, list):
            stored_rows = [row for row in rows if isinstance(row, dict)]

    stored_by_sensor: Dict[str, dict[str, Any]] = {}
    for row in stored_rows:
        sensor_id = str(row.get("sensor_id") or row.get("column") or "").strip()
        if sensor_id:
            stored_by_sensor[sensor_id] = row

    def _base_row(sensor_id: str) -> dict[str, Any]:
        return {
            "sensor_id": sensor_id,
            "tirante": f"Tirante {sensor_id}",
            "f0": None,
            "ke": None,
            "active": True,
        }

    def _merge_row(sensor_id: str) -> dict[str, Any]:
        base = _base_row(sensor_id)
        saved = stored_by_sensor.get(sensor_id) or {}
        tirante_raw = saved.get("tirante") or base["tirante"]
        tirante_value = tirante_raw.strip() if isinstance(tirante_raw, str) else str(tirante_raw)
        f0_value = saved.get("f0") if saved.get("f0") not in ("", None) else base["f0"]
        ke_value = saved.get("ke") if saved.get("ke") not in ("", None) else base["ke"]
        active_value = saved.get("active")
        if isinstance(active_value, str):
            active_value = active_value.lower() == "true"
        elif active_value is None:
            active_value = True
        else:
            active_value = bool(active_value)
        return {
            "sensor_id": sensor_id,
            "tirante": tirante_value,
            "f0": f0_value,
            "ke": ke_value,
            "active": active_value,
        }

    initial_rows: list[dict[str, Any]] = [_merge_row(sensor_id) for sensor_id in DEFAULT_SENSOR_IDS]

    # Keep any stored sensors that are not in the default list.
    for sensor_id, row in stored_by_sensor.items():
        if sensor_id not in DEFAULT_SENSOR_IDS:
            initial_rows.append(_merge_row(sensor_id))

    sensor_store_data, sensor_status = build_sensor_config_payload(initial_rows)
    save_sensor_config_store(sensor_store_data)

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Monitor automático de tensión", className="app-title"),
                            html.P(
                                "Configura y supervisa los parámetros clave de cada tirante en tiempo real.",
                                className="app-subtitle",
                            ),
                            html.Div(
                                [
                                    html.Span("Monitoreo continuo", className="hero-chip"),
                                    html.Span("Alertas configurables", className="hero-chip"),
                                    html.Span("Analítica visual", className="hero-chip"),
                                ],
                                className="hero-chip-group",
                            ),
                        ],
                        className="hero-content",
                    ),
                    html.Div(className="hero-visual"),
                ],
                className="hero",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Configuración de sensores", className="status-label"),
                            html.Div(
                                sensor_status,
                                id="sensor-config-status",
                                className="status-value",
                            ),
                        ],
                        className="status-card status-card--primary",
                    ),
                    html.Div(
                        [
                            html.Span("Archivo en análisis", className="status-label"),
                            html.Div(
                                "Seleccione una carpeta de datos y espere la llegada de nuevos archivos.",
                                id="file-info",
                                className="status-value",
                            ),
                        ],
                        className="status-card",
                    ),
                    html.Div(
                        [
                            html.Span("Estado del procesamiento", className="status-label"),
                            html.Div(
                                "Lectura detenida. Presione Iniciar para comenzar.",
                                id="processing-status",
                                className="status-value",
                            ),
                            html.Progress(
                                id="processing-progress",
                                max="100",
                                value="0",
                                className="processing-progress status-progress",
                                title="Sin actividad registrada",
                            ),
                        ],
                        className="status-card status-card--progress",
                    ),
                ],
                className="overview-grid",
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
                                                "Activa los sensores disponibles y define el tirante, la frecuencia fundamental f₀ y el valor de Ke (Ton·s).",
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
                                    html.P(
                                        "Los siguientes sensores están preconfigurados. Marca cuáles estarán activos y ajusta sus parámetros.",
                                        className="info",
                                    ),
                                    html.Div(
                                        f"{len(initial_rows)} sensor(es) preconfigurados. Activa solo los necesarios y completa sus parámetros.",
                                        id="mapping-status",
                                        className="info info--subtle",
                                    ),
                                ],
                                className="input-stack",
                            ),
                            html.Div(
                                DataTable(
                                    id="sensor-config-table",
                                    columns=[
                                        {
                                            "name": "Activo",
                                            "id": "active",
                                            "presentation": "dropdown",
                                            "editable": True,
                                        },
                                        {
                                            "name": "Sensor ID",
                                            "id": "sensor_id",
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
                                    dropdown={
                                        "active": {
                                            "options": [
                                                {"label": "Sí", "value": True},
                                                {"label": "No", "value": False},
                                            ],
                                            "clearable": False,
                                        }
                                    },
                                    editable=True,
                                    row_deletable=False,
                                    fill_width=True,
                                    style_table={
                                        "overflowX": "auto",
                                        "minWidth": "100%",
                                    },
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
                                        "padding": "10px",
                                    },
                                    style_cell_conditional=[
                                        {"if": {"column_id": "active"}, "width": "80px"},
                                        {"if": {"column_id": "sensor_id"}, "width": "120px"},
                                        {"if": {"column_id": "tirante"}, "width": "200px"},
                                    ],
                                    style_data_conditional=[
                                        {
                                            "if": {"column_id": "sensor_id"},
                                            "fontWeight": "600",
                                            "color": "#1e1b4b",
                                        }
                                    ],
                                ),
                                className="table-card",
                            ),
                        ],
                        className="panel",
                    ),
                ],
                className="layout panel-layout",
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
                            html.P(
                                "Controla el procesamiento y gestiona los tirantes disponibles.",
                                className="panel-intro",
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
                className="layout layout--single",
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
                                                            html.Label("Modo de frecuencia base"),
                                                            html.P(
                                                                "El análisis usa la f₀ configurada para el tirante. Activa el modo manual si necesitas probar otra frecuencia mientras retensas.",
                                                                className="info-note",
                                                            ),
                                                            dcc.Checklist(
                                                                id="manual-mode-toggle",
                                                                options=[
                                                                    {
                                                                        "label": "Ingresar f₀ manual",
                                                                        "value": "manual",
                                                                    }
                                                                ],
                                                                value=[],
                                                                className="compact-checklist",
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("f₀ manual (Hz)"),
                                                            dcc.Input(
                                                                id="manual-frequency-input",
                                                                type="number",
                                                                min=0.01,
                                                                step=0.01,
                                                                placeholder="Ej: 3.25",
                                                            ),
                                                            html.P(
                                                                "Esta frecuencia guía la búsqueda de armónicos y la estimación de tensión mientras el modo manual está activo.",
                                                                className="info-note",
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                ],
                                                className="control-grid",
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
                className="layout layout--stacked",
            ),
            html.Hr(className="divider"),
            html.Div(
                [
                    html.Label("Mostrar gráficas"),
                    dcc.Checklist(
                        id="graphs-toggle",
                        options=[
                            {"label": "Acelerograma completo", "value": "full"},
                            {"label": "Segmento seleccionado", "value": "segment"},
                            {"label": "Espectro de potencia", "value": "psd"},
                            {"label": "Espectrograma (STFT)", "value": "stft"},
                        ],
                        value=["full", "segment", "psd", "stft"],
                        labelStyle={"display": "inline-block", "marginRight": "16px"},
                    ),
                ],
                className="graph-controls",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Acelerograma completo", className="graph-title"),
                            dcc.Loading(
                                dcc.Graph(id="accelerogram-full", config={"displaylogo": False}),
                                type="circle",
                                color="#0ea5e9",
                                className="graph-loading",
                            ),
                        ],
                        className="graph-card",
                        id="accelerogram-full-card",
                    ),
                    html.Div(
                        [
                            html.H3("Segmento seleccionado", className="graph-title"),
                            dcc.Loading(
                                dcc.Graph(id="accelerogram-segment", config={"displaylogo": False}),
                                type="circle",
                                color="#0ea5e9",
                                className="graph-loading",
                            ),
                        ],
                        className="graph-card",
                        id="accelerogram-segment-card",
                    ),
                ],
                className="graph-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Espectro de potencia", className="graph-title"),
                            dcc.Loading(
                                dcc.Graph(id="psd-graph", config={"displaylogo": False}),
                                type="circle",
                                color="#0ea5e9",
                                className="graph-loading",
                            ),
                        ],
                        className="graph-card",
                        id="psd-graph-card",
                    ),
                    html.Div(
                        [
                            html.H3("Espectrograma (STFT)", className="graph-title"),
                            dcc.Loading(
                                dcc.Graph(id="stft-graph", config={"displaylogo": False}),
                                type="circle",
                                color="#0ea5e9",
                                className="graph-loading",
                            ),
                        ],
                        className="graph-card",
                        id="stft-graph-card",
                    ),
                ],
                className="graph-row",
            ),
            html.H2("Resultados", className="section-title results-title"),
            html.Div(
                [
                    html.H3("Tensión por tirante", className="graph-title"),
                    dcc.Graph(
                        id="results-history-graph",
                        config={"displaylogo": False},
                        className="results-history-figure",
                    ),
                ],
                className="graph-card results-history-card",
            ),
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
            dcc.Store(id="results-history-store", data=[]),
            dcc.Store(
                id="directory-browser-store",
                data={"path": os.getcwd()},
            ),
            html.Div(id="error-message", className="error"),
        ],
        className="app-shell app-container",
    )
