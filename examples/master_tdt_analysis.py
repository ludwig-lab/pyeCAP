# master_tdt_analysis.py
from __future__ import annotations
import numpy as np
import pandas as pd
import dask.array as da


# -----------------------------
# Stage 1 — filtering (raw)
# -----------------------------
def apply_filters(ephys, *, median_low=False, powerline=False, median_high=False):
    """
    Apply filters to the raw time series BEFORE windowing/averaging.
    This calls your existing ephys-level filters (not the mean-trace filters).
    """
    X = ephys
    if median_low:
        X = X.filter_median(btype="lowpass", kernel_size=11)
    if powerline:
        X = X.filter_powerline()
    if median_high:
        X = X.filter_median(btype="highpass")
    return X


# -----------------------------
# Stage 2 — window & average
# -----------------------------
def stack_mean_waveforms(ecap, *, time_chunk_ms=None,
                         k_window=6, target_block_mb=16, max_span_s=0.5):
    """
    Build a lazy 3D array of mean waveforms stacked over parameters.
    Returns (params, channels, samples), and the parameter index list.

    By default, the time chunk size is auto-tuned from epoch length, fs,
    channel count, and a small memory budget.
    """

    def autotune_time_chunk(ecap, *, k_window=6, target_block_mb=16, max_span_s=0.5):
        """
        Choose a time_chunk (in samples) for ecap.persist_ts(...)
        - k_window: chunk ≈ k_window × epoch window length
        - target_block_mb: aim per-block memory across all channels
        - max_span_s: hard cap on time span per chunk
        """
        fs = float(ecap.ts_data.sample_rate)
        # epoch window length in seconds
        if ecap.epoch_window in (None, "auto"):
            win_s = 0.010
        else:
            win_s = float(ecap.epoch_window[1] - ecap.epoch_window[0])
        W = max(1, int(round(win_s * fs)))  # samples per epoch window

        C = len(ecap.ts_data.ch_names)
        try:
            dtype = np.dtype(ecap.ts_data.array.dtype)
        except Exception:
            dtype = np.float64
        bytes_per_sample_all_ch = C * dtype.itemsize

        min_chunk = max(W * int(k_window), 1)
        budget_samples = max(1, int((target_block_mb * 1024 ** 2) / bytes_per_sample_all_ch))
        cap_samples = max(1, int(max_span_s * fs))

        # choose within [min_chunk, cap_samples], preferring budget_samples
        tc = max(min_chunk, min(budget_samples, cap_samples))
        return int(tc)

    fs = float(ecap.ts_data.sample_rate)

    if time_chunk_ms is None:
        # auto-tune in samples
        time_chunk = autotune_time_chunk(
            ecap, k_window=k_window,
            target_block_mb=target_block_mb,
            max_span_s=max_span_s
        )
    else:
        time_chunk = max(1, int(round((time_chunk_ms / 1000.0) * fs)))

    # Persist raw ts array for fast repeated slicing
    ecap.persist_ts(time_chunk=time_chunk)

    param_keys = list(ecap.parameters.parameters.index)
    mean_list = [ecap.mean_waveform(p) for p in param_keys]  # each (ch, samples) dask
    mean_3d = da.stack(mean_list, axis=0)  # (params, ch, samples)
    return mean_3d, param_keys


# -----------------------------
# Stage 3 — ECAP features
# -----------------------------
def _windows_samples_to_times(win_idx, fs, x0):
    """Convert {fiber: (i0,i1)} -> {fiber: (t0,t1)} in seconds."""
    return {f: (i0 / fs + x0, i1 / fs + x0) for f, (i0, i1) in win_idx.items()}


def _rms(y, axis=-1):
    return da.sqrt(da.mean(y ** 2, axis=axis))


def _peak_to_trough(y, axis=-1):
    # Peak method: signed peak-to-trough within the window
    # Choose max absolute excursion (handles pos/neg deflection)
    peak = y.max(axis=axis)
    trough = y.min(axis=axis)
    # if negative-going dominates, return (trough - baseline) in magnitude
    # but a simple |peak - trough| is robust for magnitude
    return da.maximum(da.abs(peak - trough), 0)


def compute_ecap_features(ecap, mean_3d, *, baseline_s=None):
    """
    Compute ECAP metrics per fiber window for every (param, channel).
    Returns a Pandas DataFrame with MultiIndex (param, channel) and columns like:
      RMS_Aα, RMS_Aβ, ..., PEAKS_Aα, PEAKS_Aβ, ...
    """
    fs = float(ecap.ts_data.sample_rate)

    # Resolve epoch start for time conversion
    if ecap.epoch_window in (None, "auto"):
        x0 = 0.0
    else:
        x0, _ = ecap.epoch_window

    # Ensure neural windows exist and are non-zero
    # (Call your window calculator if needed)
    if not hasattr(ecap, "neural_window_indices") or not ecap.neural_window_indices:
        # If you have a method to compute these, call it here
        # ecap.calculate_neural_window_lengths()
        raise ValueError("neural_window_indices are missing; compute them before features.")

    fiber_names = list(ecap.neural_window_indices.keys())  # e.g., ["Aα","Aβ","Aδ","B"]

    # mean_3d is (P, C, T), still dask
    P, C, T = mean_3d.shape

    # Optional: baseline subtraction (per channel & parameter) before windows
    # Provide a pre-stim window like (-3 ms, -1 ms) and make sure x_lim includes it
    if baseline_s is not None:
        b0 = int(np.floor((baseline_s[0] - x0) * fs))
        b1 = int(np.ceil((baseline_s[1] - x0) * fs))
        # guard
        b0 = max(0, min(int(T) - 1, b0))
        b1 = max(b0 + 1, min(int(T), b1))
        base = mean_3d[..., b0:b1].mean(axis=-1, keepdims=True)
        mean_3d = mean_3d - base

    # For each fiber window, slice, then compute RMS and PEAKS
    rms_cols = {}
    peaks_cols = {}

    for fiber in fiber_names:
        i0, i1 = ecap.neural_window_indices[fiber]  # sample indices in epoch
        i0 = int(max(0, min(int(T) - 1, i0)))
        i1 = int(max(i0 + 1, min(int(T), i1)))

        seg = mean_3d[..., i0:i1]  # (P, C, W)
        rms_cols[f"RMS_{fiber}"] = _rms(seg, axis=-1)  # (P, C)
        peaks_cols[f"PEAKS_{fiber}"] = _peak_to_trough(seg, axis=-1)  # (P, C)

    # Stack columns and compute once
    all_cols = list(rms_cols.keys()) + list(peaks_cols.keys())
    mats = [rms_cols[k] for k in rms_cols] + [peaks_cols[k] for k in peaks_cols]  # each (P,C) dask
    feat = da.stack(mats, axis=-1).compute()  # (P, C, n_cols) numpy

    # Build a tidy DataFrame
    params = list(ecap.parameters.parameters.index)
    chs = list(ecap.ts_data.ch_names)
    idx = pd.MultiIndex.from_product([params, chs], names=["parameter", "channel"])
    df = pd.DataFrame(
        feat.reshape(len(params) * len(chs), len(all_cols)),
        index=idx,
        columns=all_cols
    )
    return df


# -----------------------------
# Orchestrator
# -----------------------------
def master_tdt_analysis(ephys, stim, *,
                        x_lim=(-0.004, 0.010),
                        baseline_s=(-0.003, -0.001),
                        median_low=False, powerline=False, median_high=False,
                        time_chunk_ms=60):
    """
    End-to-end:
      1) filter -> 2) window & average -> 3) ECAP features (RMS and PEAKS) per fiber
    Returns (mean_waveforms, features_df).
    """
    # 1) Filtering
    ephys_f = apply_filters(ephys,
                            median_low=median_low,
                            powerline=powerline,
                            median_high=median_high)

    # 2) ECAP object with the filtered data
    from pyeCAP import ECAP
    ecap = ECAP(ephys_f, stim)
    ecap.epoch_window = x_lim

    # If your neural windows depend on distance, ensure distance_log is valid and windows computed
    # ecap.calculate_neural_window_lengths()  # if needed

    mean_3d, param_keys = stack_mean_waveforms(ecap, time_chunk_ms=time_chunk_ms)

    # 3) ECAP features per fiber (RMS + PEAKS)
    features = compute_ecap_features(ecap, mean_3d, baseline_s=baseline_s)

    # Optionally materialize mean waveforms now (if downstream code expects numpy)
    mean_np = mean_3d.compute()  # (params, channels, samples)

    return mean_np, features, ecap
