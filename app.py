import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from dash.dash_table import DataTable
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, spectrogram, welch


@dataclass
class AnalysisResults:
    fundamental: Optional[float]
    harmonics: List[float]
    refined_fundamental: Optional[float]
    harmonics_from_hint: List[float]
    refined_from_hint: Optional[float]
    tension: Optional[float]


def load_and_prepare_data_from_file(path: str, sensor_map: Dict[str, str]) -> pd.DataFrame:
    """Load CSV files that contain a DATA_START marker and rename sensors."""
    with open(path, "r", encoding="utf-8") as file_handle:
        lines = file_handle.read().splitlines()

    data_start_index = None
    for idx, line in enumerate(lines):
        if "DATA_START" in line:
            data_start_index = idx + 1
            break

    if data_start_index is None:
        raise ValueError("No se encontró la etiqueta 'DATA_START' en el archivo.")

    headers = lines[data_start_index].strip().split(",")
    data_lines = lines[data_start_index + 1 :]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), names=headers)
    if sensor_map:
        df.rename(columns=sensor_map, inplace=True)

    df = df.dropna()
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna()
    return df


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    smooth_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, psd = welch(signal, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    psd_smooth = gaussian_filter1d(psd, sigma=smooth_sigma)
    return freqs, psd, psd_smooth


def detect_peaks(
    freqs: np.ndarray,
    psd_smooth: np.ndarray,
    threshold: float,
    min_distance_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.where(freqs >= 0.1)
    freqs_v = freqs[valid_idx]
    psd_s_v = psd_smooth[valid_idx]
    if len(freqs_v) < 2:
        return freqs_v, np.array([], dtype=int), valid_idx[0]

    freq_resolution = freqs_v[1] - freqs_v[0]
    min_distance_idx = max(1, int(min_distance_hz / freq_resolution))
    peaks, _ = find_peaks(psd_s_v, height=threshold, distance=min_distance_idx)
    return freqs_v, peaks, valid_idx[0]


def find_real_peak_by_index(psd: np.ndarray, peak_idx: int, window: int = 10) -> Tuple[int, float]:
    start = max(0, peak_idx - window)
    end = min(len(psd), peak_idx + window + 1)
    local_max_idx = np.argmax(psd[start:end])
    return start + local_max_idx, psd[start + local_max_idx]


def identify_fundamental_and_harmonics_auto(
    freqs: np.ndarray,
    psd: np.ndarray,
    candidate_freqs: List[float],
    n_harmonics: int,
    tolerance: float = 0.5,
) -> Tuple[Optional[float], List[float], Optional[float]]:
    best_f0 = None
    best_harmonics: List[float] = []
    best_refined_f0 = None
    best_score = 0

    for f0_candidate in candidate_freqs:
        expected = [f0_candidate * i for i in range(2, 2 + n_harmonics)]
        matches: List[float] = []
        harmonic_numbers: List[int] = []
        for harmonic_number, expected_freq in enumerate(expected, start=2):
            close = [f for f in candidate_freqs if abs(f - expected_freq) <= tolerance]
            if close:
                matches.append(close[0])
                harmonic_numbers.append(harmonic_number)
        if len(matches) > best_score:
            best_score = len(matches)
            best_f0 = f0_candidate
            best_harmonics = matches
            if harmonic_numbers:
                best_refined_f0 = np.mean([freq / num for freq, num in zip(matches, harmonic_numbers)])

    if best_f0 is None and candidate_freqs:
        best_f0 = candidate_freqs[0]
        best_refined_f0 = best_f0

    return best_f0, best_harmonics, best_refined_f0


def identify_from_hint(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0_hint: Optional[float],
    n_harmonics: int,
    tol_hz: float,
) -> Tuple[Optional[float], List[float], Optional[float]]:
    if f0_hint is None or f0_hint <= 0:
        return None, [], None

    def peak_in_window(f_center: float, tol: float) -> Optional[float]:
        fmin, fmax = max(0.0, f_center - tol), f_center + tol
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return None
        idx_local = np.argmax(psd[mask])
        idx_global = np.where(mask)[0][0] + idx_local
        return freqs[idx_global]

    harmonics_found: List[float] = []
    ks: List[int] = []
    for k in range(2, 2 + n_harmonics):
        f_expected = k * f0_hint
        f_peak = peak_in_window(f_expected, tol_hz)
        if f_peak is not None and f_peak > 0:
            harmonics_found.append(f_peak)
            ks.append(k)

    if harmonics_found:
        ks_arr = np.array(ks, dtype=float)
        fs_arr = np.array(harmonics_found, dtype=float)
        denom = np.sum(ks_arr ** 2)
        f0_refined = np.sum(ks_arr * fs_arr) / denom if denom > 0 else f0_hint
    else:
        f0_refined = f0_hint

    return f0_refined, harmonics_found, f0_hint


def plot_accelerogram(signal: np.ndarray, fs: float, name: str, start_s: float, end_s: float) -> go.Figure:
    t_full = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_full, y=signal, mode="lines", name="Aceleración"))
    fig.add_vline(x=start_s, line=dict(color="red", dash="dash"), name="Inicio")
    fig.add_vline(x=end_s, line=dict(color="green", dash="dash"), name="Fin")
    fig.update_layout(
        title=f"Acelerograma - {name}",
        xaxis_title="Tiempo [s]",
        yaxis_title="Aceleración [g]",
        template="plotly_white",
    )
    return fig


def plot_segment(signal: np.ndarray, fs: float, name: str) -> go.Figure:
    t_seg = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_seg, y=signal, mode="lines", name="Aceleración segmento"))
    fig.update_layout(
        title=f"Segmento seleccionado - {name}",
        xaxis_title="Tiempo [s]",
        yaxis_title="Aceleración [g]",
        template="plotly_white",
    )
    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    fundamental: Optional[float],
    harmonics: List[float],
    threshold: Optional[float],
    f0_hint: Optional[float],
    harmonics_from_hint: List[float],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name="PSD", line=dict(color="blue")))

    if fundamental is not None:
        peaks_x = [fundamental] + harmonics
        peaks_y = [psd[np.argmin(np.abs(freqs - fx))] for fx in peaks_x]
        fig.add_trace(
            go.Scatter(
                x=peaks_x,
                y=peaks_y,
                mode="markers+text",
                name="Picos (auto)",
                marker=dict(color="red", size=10, symbol="circle"),
                text=[f"{fx:.2f} Hz" for fx in peaks_x],
                textposition="top center",
            )
        )

    if f0_hint is not None:
        fig.add_trace(
            go.Scatter(
                x=[f0_hint, f0_hint],
                y=[float(np.min(psd)), float(np.max(psd))],
                mode="lines",
                name="f₀ propuesta",
                line=dict(color="green", dash="dash"),
            )
        )

    if harmonics_from_hint:
        peaks_y2 = [psd[np.argmin(np.abs(freqs - fx))] for fx in harmonics_from_hint]
        fig.add_trace(
            go.Scatter(
                x=harmonics_from_hint,
                y=peaks_y2,
                mode="markers+text",
                name="Armónicos (f₀ propuesta)",
                marker=dict(size=10, symbol="x"),
                text=[f"{fx:.2f} Hz" for fx in harmonics_from_hint],
                textposition="bottom center",
            )
        )

    if threshold:
        fig.add_trace(
            go.Scatter(
                x=[freqs[0], freqs[-1]],
                y=[threshold, threshold],
                mode="lines",
                name="Threshold",
                line=dict(color="orange", dash="dash"),
            )
        )

    fig.update_layout(
        title="PSD",
        xaxis_title="Frecuencia [Hz]",
        yaxis_title="PSD [g²/Hz]",
        yaxis=dict(type="log", tickformat=".2e"),
        template="plotly_white",
    )
    return fig


def plot_stft(signal: np.ndarray, fs: float) -> go.Figure:
    f, t, sxx = spectrogram(signal, fs, nperseg=1024, noverlap=512)
    z_db = 10 * np.log10(sxx + 1e-12)
    cmap = cm.get_cmap("viridis")
    colorscale = [[i / 255, f"rgb{tuple(int(255 * c) for c in cmap(i / 255)[:3])}"] for i in range(256)]
    heatmap = go.Heatmap(
        z=z_db,
        x=t,
        y=f,
        colorscale=colorscale,
        zmin=-120,
        zmax=-40,
        colorbar=dict(title="dB"),
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title="STFT",
        xaxis_title="Tiempo [s]",
        yaxis_title="Frecuencia [Hz]",
        yaxis_range=[0, 50],
        template="plotly_white",
    )
    return fig


def compute_tension(frequency: Optional[float], length_m: float, linear_density: float) -> Optional[float]:
    if frequency is None or frequency <= 0:
        return None
    if length_m <= 0 or linear_density <= 0:
        return None
    return 4 * (length_m ** 2) * (frequency ** 2) * linear_density


def analyse_signal(
    df: pd.DataFrame,
    sensor: str,
    fs: float,
    pct_range: Tuple[float, float],
    nperseg: int,
    noverlap: int,
    smooth_sigma: float,
    threshold: float,
    min_distance_hz: float,
    n_harmonics: int,
    use_hint: bool,
    f0_hint: float,
    tol_hz: float,
    length_m: float,
    linear_density: float,
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, AnalysisResults, Tuple[float, float, float]]:
    signal_full = df[sensor].values.astype(float)
    signal_full = signal_full - np.mean(signal_full)

    n = len(signal_full)
    if n < 2:
        raise ValueError("La señal es demasiado corta para el análisis.")

    p0, p1 = pct_range
    p0 = np.clip(p0, 0.0, 100.0)
    p1 = np.clip(p1, 0.0, 100.0)
    if p1 < p0:
        p0, p1 = p1, p0

    i0 = int(np.floor(p0 / 100.0 * (n - 1)))
    i1 = int(np.floor(p1 / 100.0 * (n - 1)))
    if i1 <= i0:
        i1 = min(n - 1, i0 + 1)

    start_s = i0 / fs
    end_s = i1 / fs

    segment = slice(i0, i1 + 1)
    signal_segment = signal_full[segment]

    freqs, psd, psd_smooth = compute_psd(signal_segment, fs, nperseg, noverlap, smooth_sigma)
    _freqs_filt, peaks_idx, valid_idx_offset = detect_peaks(freqs, psd_smooth, threshold, min_distance_hz)

    real_peaks: List[float] = []
    for idx in peaks_idx:
        real_idx, _ = find_real_peak_by_index(psd, valid_idx_offset[idx])
        real_peaks.append(freqs[real_idx])

    f0_auto: Optional[float] = None
    harmonics_auto: List[float] = []
    f0_refined_auto: Optional[float] = None
    harmonics_from_hint: List[float] = []
    f0_refined_hint: Optional[float] = None

    if use_hint and f0_hint > 0:
        f0_refined_hint, harmonics_from_hint, _ = identify_from_hint(
            freqs=freqs,
            psd=psd,
            f0_hint=f0_hint,
            n_harmonics=n_harmonics,
            tol_hz=tol_hz,
        )
    else:
        candidates_sorted = sorted(real_peaks, key=lambda x: -psd[np.argmin(np.abs(freqs - x))])
        f0_auto, harmonics_auto, f0_refined_auto = identify_fundamental_and_harmonics_auto(
            freqs=freqs,
            psd=psd,
            candidate_freqs=candidates_sorted,
            n_harmonics=n_harmonics,
            tolerance=0.5,
        )

    psd_fig = plot_psd(
        freqs,
        psd,
        fundamental=None if use_hint else f0_auto,
        harmonics=[] if use_hint else harmonics_auto,
        threshold=threshold,
        f0_hint=f0_refined_hint if use_hint else None,
        harmonics_from_hint=harmonics_from_hint if use_hint else [],
    )

    accel_full_fig = plot_accelerogram(signal_full, fs, sensor, start_s, end_s)
    accel_segment_fig = plot_segment(signal_segment, fs, sensor)
    stft_fig = plot_stft(signal_segment, fs)

    fundamental_used = f0_refined_hint if use_hint else f0_refined_auto
    tension = compute_tension(fundamental_used, length_m, linear_density)

    results = AnalysisResults(
        fundamental=f0_auto,
        harmonics=harmonics_auto,
        refined_fundamental=f0_refined_auto,
        harmonics_from_hint=harmonics_from_hint,
        refined_from_hint=f0_refined_hint,
        tension=tension,
    )

    return (
        accel_full_fig,
        accel_segment_fig,
        psd_fig,
        stft_fig,
        results,
        (start_s, end_s, end_s - start_s),
    )


def get_directory_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(".csv")
    ]
    files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return files


app = Dash(__name__)
app.title = "Monitor de tensión"
server = app.server

app.layout = html.Div(
    [
        html.H1("Monitor automático de tensión"),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Configuración"),
                        html.Label("Directorio de datos"),
                        dcc.Input(id="directory-input", type="text", value="./data", debounce=True, style={"width": "100%"}),
                        html.Br(),
                        html.Label("Intervalo de actualización (s)"),
                        dcc.Input(id="poll-seconds", type="number", min=5, max=300, step=5, value=30),
                        html.Br(),
                        html.Label("Mapeo de sensores (JSON)"),
                        dcc.Textarea(id="map-textarea", value="{}", style={"width": "100%", "height": "120px"}),
                        html.Br(),
                        html.Label("Archivo seleccionado"),
                        dcc.Dropdown(id="file-dropdown"),
                        html.Div(id="file-info", className="info"),
                        html.Br(),
                        html.Label("Tirante"),
                        dcc.Dropdown(id="sensor-dropdown"),
                    ],
                    className="panel",
                ),
                html.Div(
                    [
                        html.H3("Parámetros de análisis"),
                        html.Div(
                            [
                                html.Label("Fs (Hz)"),
                                dcc.Input(id="fs-input", type="number", min=10, max=2000, step=1, value=128),
                                html.Label("nperseg"),
                                dcc.Input(id="nperseg-input", type="number", min=256, max=16384, step=256, value=4096),
                                html.Label("noverlap"),
                                dcc.Input(id="noverlap-input", type="number", min=0, max=16384, step=256, value=2048),
                                html.Label("Suavizado σ"),
                                dcc.Input(id="sigma-input", type="number", min=0.1, max=10.0, step=0.1, value=0.6),
                                html.Label("Threshold"),
                                dcc.Input(id="threshold-input", type="number", min=1e-12, max=1e-3, step=1e-7, value=2.5e-7),
                                html.Label("Min distancia (Hz)"),
                                dcc.Input(id="min-distance-input", type="number", min=0.1, max=10.0, step=0.1, value=0.3),
                                html.Label("N° Armónicos"),
                                dcc.Input(id="harmonics-input", type="number", min=1, max=5, step=1, value=2),
                                html.Label("Rango del archivo (%)"),
                                dcc.RangeSlider(id="pct-range", min=0, max=100, step=0.1, value=[0, 100]),
                                html.Div(id="pct-label", className="info"),
                            ],
                            className="grid",
                        ),
                        html.H3("f₀ propuesta"),
                        dcc.Checklist(
                            options=[{"label": "Usar f₀ propuesta", "value": "use"}],
                            value=[],
                            id="use-f0-hint",
                        ),
                        html.Label("f₀ propuesta (Hz)"),
                        dcc.Input(id="f0-hint-input", type="number", min=0.0, step=0.01, value=2.0),
                        html.Label("Tol ± (Hz)"),
                        dcc.Input(id="tol-input", type="number", min=0.01, max=5.0, step=0.01, value=0.15),
                        html.H3("Cálculo de tensión"),
                        html.Label("Longitud (m)"),
                        dcc.Input(id="length-input", type="number", min=0.0, step=0.1, value=1.0),
                        html.Label("Masa lineal (kg/m)"),
                        dcc.Input(id="density-input", type="number", min=0.0, step=0.01, value=0.5),
                    ],
                    className="panel",
                ),
            ],
            className="layout",
        ),
        html.Hr(),
        html.Div(
            [
                dcc.Graph(id="accelerogram-full"),
                dcc.Graph(id="accelerogram-segment"),
            ],
            className="graph-row",
        ),
        html.Div(
            [
                dcc.Graph(id="psd-graph"),
                dcc.Graph(id="stft-graph"),
            ],
            className="graph-row",
        ),
        html.H2("Resultados"),
        DataTable(id="results-table", columns=[], data=[]),
        dcc.Interval(id="polling-interval", interval=30000, n_intervals=0),
        dcc.Store(id="data-store"),
        dcc.Store(id="files-store"),
        html.Div(id="error-message", className="error"),
    ]
)


@app.callback(
    Output("polling-interval", "interval"),
    Input("poll-seconds", "value"),
)
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
        {"label": f"{os.path.basename(path)} ({datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')})", "value": path}
        for path in files
    ]
    value = current_value if current_value in files else (files[0] if files else None)
    return files, options, value


@app.callback(
    Output("data-store", "data"),
    Output("sensor-dropdown", "options"),
    Output("sensor-dropdown", "value"),
    Output("file-info", "children"),
    Output("error-message", "children"),
    Input("file-dropdown", "value"),
    Input("map-textarea", "value"),
)
def load_file(path: Optional[str], mapping_text: str):
    if not path:
        raise PreventUpdate

    try:
        sensor_map = json.loads(mapping_text or "{}")
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
    store_data,
    sensor,
    fs,
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
    empty_fig = go.Figure()
    if not store_data or not sensor:
        return empty_fig, empty_fig, empty_fig, empty_fig, [], [], "", ""

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
            pct_range=tuple(pct_range),
            nperseg=int(nperseg),
            noverlap=int(noverlap),
            smooth_sigma=float(sigma),
            threshold=float(threshold),
            min_distance_hz=float(min_distance),
            n_harmonics=int(n_harmonics),
            use_hint=use_hint,
            f0_hint=float(f0_hint),
            tol_hz=float(tol_hz),
            length_m=float(length_m),
            linear_density=float(linear_density),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return empty_fig, empty_fig, empty_fig, empty_fig, [], [], "", str(exc)

    pct_label = f"Rango temporal: {start_s:.2f} s → {end_s:.2f} s (duración: {duration_s:.2f} s)"

    if "use" in (use_hint_values or []) and results.refined_from_hint:
        fundamental_display = results.refined_from_hint
        harmonics_display = results.harmonics_from_hint
    else:
        fundamental_display = results.refined_fundamental
        harmonics_display = results.harmonics

    table_data = {
        "Frecuencia Fundamental [Hz]": fundamental_display,
        "Armónicos detectados [Hz]": ", ".join(f"{h:.2f}" for h in harmonics_display) if harmonics_display else "—",
        "Tensión estimada [N]": f"{results.tension:.2f}" if results.tension is not None else "—",
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


if __name__ == "__main__":
    app.run_server(debug=True)
