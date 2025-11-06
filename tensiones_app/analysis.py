"""Signal analysis utilities for the tension monitoring dashboard."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import cm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, spectrogram, welch


@dataclass
class AnalysisResults:
    """Container with the relevant outcomes of the spectral analysis."""

    fundamental: Optional[float]
    harmonics: List[float]
    refined_fundamental: Optional[float]
    harmonics_from_hint: List[float]
    refined_from_hint: Optional[float]
    tension: Optional[float]


def load_and_prepare_data_from_file(path: str, sensor_map: Dict[str, Any]) -> pd.DataFrame:
    """Load CSV data that contains a ``DATA_START`` marker and prepare it for analysis."""

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
        rename_map: Dict[str, str] = {}
        for original, alias in sensor_map.items():
            if isinstance(alias, dict):
                tirante_name = alias.get("tirante")
                if tirante_name:
                    rename_map[original] = tirante_name
            elif isinstance(alias, str) and alias:
                rename_map[original] = alias

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

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
    """Compute the power spectral density of the segment and smooth it."""

    freqs, psd = welch(signal, fs, nperseg=nperseg, noverlap=noverlap, window="hann")
    psd_smooth = gaussian_filter1d(psd, sigma=smooth_sigma)
    return freqs, psd, psd_smooth


def detect_peaks(
    freqs: np.ndarray,
    psd_smooth: np.ndarray,
    threshold: Optional[float],
    min_distance_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify candidate peaks on the PSD curve above the desired threshold."""

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
    """Return the precise index and value of the peak around the given estimate."""

    start = max(0, peak_idx - window)
    end = min(len(psd), peak_idx + window + 1)
    local_max_idx = np.argmax(psd[start:end])
    return start + local_max_idx, psd[start + local_max_idx]


def identify_fundamental_and_harmonics_auto(
    freqs: np.ndarray,
    psd: np.ndarray,
    candidate_freqs: Sequence[float],
    n_harmonics: int,
    tolerance: float = 0.5,
) -> Tuple[Optional[float], List[float], Optional[float]]:
    """Attempt to find the fundamental frequency and harmonics automatically."""

    best_f0: Optional[float] = None
    best_harmonics: List[float] = []
    best_refined_f0: Optional[float] = None
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
    """Refine the fundamental frequency around a user-provided hint."""

    if f0_hint is None or f0_hint <= 0:
        return None, [], None

    def peak_in_window(f_center: float, tol: float) -> Optional[float]:
        fmin, fmax = max(0.0, f_center - tol), f_center + tol
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return None
        idx_local = np.argmax(psd[mask])
        idx_global = np.where(mask)[0][0] + idx_local
        return float(freqs[idx_global])

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
        f0_refined = float(np.sum(ks_arr * fs_arr) / denom) if denom > 0 else float(f0_hint)
    else:
        f0_refined = float(f0_hint)

    return f0_refined, harmonics_found, f0_hint


def plot_accelerogram(signal: np.ndarray, fs: float, name: str, start_s: float, end_s: float) -> go.Figure:
    """Return the figure with the full accelerogram and selection cursors."""

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
    """Return the figure for the selected accelerogram segment."""

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
    harmonics: Iterable[float],
    threshold: Optional[float],
    f0_hint: Optional[float],
    harmonics_from_hint: Iterable[float],
    *,
    sensor_name: Optional[str] = None,
) -> go.Figure:
    """Plot the PSD along with the identified peaks and thresholds."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name="PSD", line=dict(color="blue")))

    if fundamental is not None:
        harmonics_list = list(harmonics)
        peaks_x = [fundamental] + harmonics_list
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

    harmonics_hint_list = list(harmonics_from_hint)
    if harmonics_hint_list:
        peaks_y2 = [psd[np.argmin(np.abs(freqs - fx))] for fx in harmonics_hint_list]
        fig.add_trace(
            go.Scatter(
                x=harmonics_hint_list,
                y=peaks_y2,
                mode="markers+text",
                name="Armónicos (f₀ propuesta)",
                marker=dict(size=10, symbol="x"),
                text=[f"{fx:.2f} Hz" for fx in harmonics_hint_list],
                textposition="bottom center",
            )
        )

    if threshold is not None and threshold > 0:
        fig.add_trace(
            go.Scatter(
                x=[freqs[0], freqs[-1]],
                y=[threshold, threshold],
                mode="lines",
                name="Threshold",
                line=dict(color="orange", dash="dash"),
            )
        )

    title = "Espectro de potencia"
    if sensor_name:
        title = f"{title} – {sensor_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Frecuencia [Hz]",
        yaxis_title="PSD [g²/Hz]",
        yaxis=dict(type="log", tickformat=".2e"),
        template="plotly_white",
    )
    return fig


def plot_stft(signal: np.ndarray, fs: float, *, sensor_name: Optional[str] = None) -> go.Figure:
    """Return an STFT heatmap of the selected signal segment."""

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
    title = "Espectrograma (STFT)"
    if sensor_name:
        title = f"{title} – {sensor_name}"

    fig.update_layout(
        title=title,
        xaxis_title="Tiempo [s]",
        yaxis_title="Frecuencia [Hz]",
        yaxis_range=[0, 50],
        template="plotly_white",
    )
    return fig


def compute_tension(
    frequency: Optional[float],
    length_m: Optional[float] = None,
    linear_density: Optional[float] = None,
    ke_ton_s: Optional[float] = None,
) -> Optional[float]:
    """Estimate the cable tension given a fundamental frequency."""

    if frequency is None or frequency <= 0:
        return None

    if ke_ton_s is not None and ke_ton_s > 0:
        return ke_ton_s * frequency

    if (
        length_m is None
        or linear_density is None
        or length_m <= 0
        or linear_density <= 0
    ):
        return None

    return 4 * (length_m ** 2) * (frequency ** 2) * linear_density


def _validate_range(range_values: Optional[Sequence[float]]) -> Tuple[float, float]:
    if not range_values:
        return 0.0, 100.0
    p0, p1 = range_values
    p0 = float(np.clip(p0, 0.0, 100.0))
    p1 = float(np.clip(p1, 0.0, 100.0))
    if p1 < p0:
        p0, p1 = p1, p0
    return p0, p1


def _normalise_numeric(value: Optional[float], default: float, *, minimum: Optional[float] = None) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = default
    result = float(value)
    if minimum is not None:
        result = max(minimum, result)
    return result


def analyse_signal(
    df: pd.DataFrame,
    sensor: str,
    fs: Optional[float],
    pct_range: Optional[Sequence[float]],
    nperseg: Optional[int],
    noverlap: Optional[int],
    smooth_sigma: Optional[float],
    threshold: Optional[float],
    min_distance_hz: Optional[float],
    n_harmonics: Optional[int],
    use_hint: bool,
    f0_hint: Optional[float],
    tol_hz: Optional[float],
    length_m: Optional[float] = None,
    linear_density: Optional[float] = None,
    ke_ton_s: Optional[float] = None,
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, AnalysisResults, Tuple[float, float, float]]:
    """Run the full analysis pipeline for a selected sensor."""

    if sensor not in df.columns:
        raise ValueError(f"El sensor '{sensor}' no existe en el archivo.")

    fs_value = _normalise_numeric(fs, 1.0, minimum=1.0)
    pct0, pct1 = _validate_range(pct_range)

    signal_full = df[sensor].values.astype(float)
    signal_full = signal_full - np.mean(signal_full)
    n = len(signal_full)
    if n < 2:
        raise ValueError("La señal es demasiado corta para el análisis.")

    i0 = int(np.floor(pct0 / 100.0 * (n - 1)))
    i1 = int(np.floor(pct1 / 100.0 * (n - 1)))
    if i1 <= i0:
        i1 = min(n - 1, i0 + 1)

    start_s = i0 / fs_value
    end_s = i1 / fs_value

    segment = slice(i0, i1 + 1)
    signal_segment = signal_full[segment]

    nperseg_value = int(_normalise_numeric(nperseg, min(len(signal_segment), 4096), minimum=8))
    nperseg_value = min(nperseg_value, len(signal_segment)) or 1
    noverlap_value = int(_normalise_numeric(noverlap, nperseg_value // 2, minimum=0))
    if noverlap_value >= nperseg_value:
        noverlap_value = nperseg_value - 1
    smooth_sigma_value = _normalise_numeric(smooth_sigma, 0.6, minimum=0.1)
    min_distance_value = _normalise_numeric(min_distance_hz, 0.3, minimum=0.01)
    n_harmonics_value = max(1, int(_normalise_numeric(n_harmonics, 2, minimum=1)))
    tol_hz_value = _normalise_numeric(tol_hz, 0.15, minimum=0.01)
    threshold_value = None if threshold is None or threshold <= 0 else float(threshold)

    freqs, psd, psd_smooth = compute_psd(
        signal_segment,
        fs_value,
        nperseg_value,
        noverlap_value,
        smooth_sigma_value,
    )
    _freqs_filt, peaks_idx, valid_idx_offset = detect_peaks(
        freqs,
        psd_smooth,
        threshold_value,
        min_distance_value,
    )

    real_peaks: List[float] = []
    for idx in peaks_idx:
        real_idx, _ = find_real_peak_by_index(psd, int(valid_idx_offset[idx]))
        real_peaks.append(float(freqs[real_idx]))

    f0_auto: Optional[float] = None
    harmonics_auto: List[float] = []
    f0_refined_auto: Optional[float] = None
    harmonics_from_hint: List[float] = []
    f0_refined_hint: Optional[float] = None

    if use_hint and f0_hint and f0_hint > 0:
        f0_refined_hint, harmonics_from_hint, _ = identify_from_hint(
            freqs=freqs,
            psd=psd,
            f0_hint=float(f0_hint),
            n_harmonics=n_harmonics_value,
            tol_hz=tol_hz_value,
        )
    else:
        candidates_sorted = sorted(real_peaks, key=lambda x: -psd[np.argmin(np.abs(freqs - x))])
        f0_auto, harmonics_auto, f0_refined_auto = identify_fundamental_and_harmonics_auto(
            freqs=freqs,
            psd=psd,
            candidate_freqs=candidates_sorted,
            n_harmonics=n_harmonics_value,
            tolerance=0.5,
        )

    psd_fig = plot_psd(
        freqs,
        psd,
        fundamental=None if use_hint else f0_auto,
        harmonics=[] if use_hint else harmonics_auto,
        threshold=threshold_value,
        f0_hint=f0_refined_hint if use_hint else None,
        harmonics_from_hint=harmonics_from_hint if use_hint else [],
        sensor_name=sensor,
    )

    accel_full_fig = plot_accelerogram(signal_full, fs_value, sensor, start_s, end_s)
    accel_segment_fig = plot_segment(signal_segment, fs_value, sensor)
    stft_fig = plot_stft(signal_segment, fs_value, sensor_name=sensor)

    fundamental_used = f0_refined_hint if use_hint else f0_refined_auto
    tension = compute_tension(
        fundamental_used,
        length_m=float(length_m) if length_m is not None else None,
        linear_density=float(linear_density) if linear_density is not None else None,
        ke_ton_s=float(ke_ton_s) if ke_ton_s is not None else None,
    )

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
    """Return the list of CSV files in ``directory`` ordered by modification time."""

    if not directory or not os.path.isdir(directory):
        return []
    files = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(".csv")
    ]
    # Process the oldest files first so the historical data is handled before
    # newer captures. Ordering ascending by modification time makes the
    # refresh callback pick the earliest pending file.
    files.sort(key=lambda path: os.path.getmtime(path))
    return files
