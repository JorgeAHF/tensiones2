"""Dash layout factory."""
from __future__ import annotations

from dash import dcc, html
from dash.dash_table import DataTable


def build_layout() -> html.Div:
    """Return the root layout for the application."""

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
                            html.H3("Configuración", className="section-title"),
                            html.Label("Directorio de datos"),
                            dcc.Input(
                                id="directory-input",
                                type="text",
                                value="./data",
                                debounce=True,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Intervalo de actualización (s)"),
                            dcc.Input(
                                id="poll-seconds",
                                type="number",
                                min=5,
                                max=300,
                                step=5,
                                value=30,
                            ),
                            html.Br(),
                            html.Label("Mapeo de sensores (JSON)"),
                            dcc.Textarea(
                                id="map-textarea",
                                value="{}",
                                style={"width": "100%", "height": "120px"},
                            ),
                            html.Br(),
                            html.Label("Archivo seleccionado"),
                            dcc.Dropdown(id="file-dropdown", placeholder="Seleccione un archivo"),
                            html.Div(id="file-info", className="info"),
                            html.Br(),
                            html.Label("Tirante"),
                            dcc.Dropdown(id="sensor-dropdown", placeholder="Seleccione un tirante"),
                        ],
                        className="panel",
                    ),
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
                                            html.H4("Ayuda basada en conocimiento", className="subsection-title"),
                                            html.P(
                                                "Utiliza una estimación previa de f₀ para guiar la detección automática y restringir el rango de búsqueda.",
                                                className="section-description",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        options=[{"label": "Usar f₀ propuesta", "value": "use"}],
                                                        value=[],
                                                        id="use-f0-hint",
                                                        className="toggle-input",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("f₀ propuesta (Hz)"),
                                                            dcc.Input(
                                                                id="f0-hint-input",
                                                                type="number",
                                                                min=0.0,
                                                                step=0.01,
                                                                value=2.0,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
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
                                    html.Div(
                                        [
                                            html.H4("Cálculo de tensión", className="subsection-title"),
                                            html.P(
                                                "Define las propiedades físicas del tirante para estimar la tensión correspondiente a la frecuencia fundamental.",
                                                className="section-description",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label("Longitud (m)"),
                                                            dcc.Input(
                                                                id="length-input",
                                                                type="number",
                                                                min=0.0,
                                                                step=0.1,
                                                                value=1.0,
                                                            ),
                                                        ],
                                                        className="control-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label("Masa lineal (kg/m)"),
                                                            dcc.Input(
                                                                id="density-input",
                                                                type="number",
                                                                min=0.0,
                                                                step=0.01,
                                                                value=0.5,
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
            html.H2("Resultados", className="section-title"),
            DataTable(
                id="results-table",
                columns=[],
                data=[],
                style_header={
                    "backgroundColor": "transparent",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                },
                style_cell={
                    "backgroundColor": "rgba(255, 255, 255, 0.05)",
                    "border": "0px",
                    "color": "#f1f5f9",
                    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
                    "padding": "12px",
                },
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "rgba(148, 163, 184, 0.25)",
                        "border": "0px",
                    }
                ],
            ),
            dcc.Interval(id="polling-interval", interval=30000, n_intervals=0),
            dcc.Store(id="data-store"),
            dcc.Store(id="files-store"),
            html.Div(id="error-message", className="error"),
        ],
        className="app-container",
    )
