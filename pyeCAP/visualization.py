from __future__ import annotations

from numbers import Real
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# =============================================================================
# Shared helpers
# =============================================================================

def _parameter_table(epoch) -> pd.DataFrame:
    """Return the stimulation-parameter table used by EpochData/ECAP."""
    parameters = getattr(epoch, "parameters", None)
    table = getattr(parameters, "parameters", None)

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "epoch.parameters.parameters must be a pandas DataFrame."
        )
    if not table.index.is_unique:
        raise ValueError(
            "epoch.parameters.parameters must have a unique index so each "
            "row can be used directly as an epoch parameter key."
        )

    return table


def _channel_names(epoch) -> list[str]:
    """Return recording-channel names in EpochData channel order."""
    source = getattr(epoch, "ts_data", None)
    if source is None:
        source = getattr(epoch, "ephys", None)

    if source is None:
        raise AttributeError("Could not find epoch.ts_data or epoch.ephys.")

    names = getattr(source, "ch_names", None)
    if names is None:
        return [f"Channel {i}" for i in range(int(source.shape[0]))]

    return [str(name) for name in names]


def _find_amp_col(df: pd.DataFrame) -> str:
    """Find the stimulation-amplitude column used by the parameter table."""
    candidates = [
        "pulse amplitude (μA)",
        "pulse amplitude (uA)",
        "pulse amplitude A (μA)",
        "pulse amplitude A (uA)",
        "pulse amplitude",
        "pulse_amplitude_uA",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    # Last-resort case-insensitive search for common variants.
    normalized = {
        str(column).lower().replace("µ", "μ"): column
        for column in df.columns
    }
    for normalized_name, original_name in normalized.items():
        if "pulse" in normalized_name and "amp" in normalized_name:
            return original_name

    raise KeyError(
        "Could not find a pulse-amplitude column in "
        "epoch.parameters.parameters."
    )


def _resolve_channel_index(epoch, channel) -> int:
    """Convert one recording-channel name/index to a global channel index."""
    names = _channel_names(epoch)

    if isinstance(channel, str):
        try:
            return names.index(channel)
        except ValueError as exc:
            raise KeyError(
                f"Recording channel {channel!r} was not found. "
                f"Available channels: {names}"
            ) from exc

    if isinstance(channel, (int, np.integer)):
        channel_index = int(channel)
        if channel_index < 0 or channel_index >= len(names):
            raise IndexError(
                f"Recording-channel index {channel_index} is out of range "
                f"for {len(names)} channels."
            )
        return channel_index

    raise TypeError("channel must be a recording-channel name or integer index.")


def _resolve_neural_channels(epoch, rec_channels=None):
    """
    Resolve selected neural channels.

    Returns
    -------
    window_positions : np.ndarray
        Positions along neural_window_indices axis 0.
    recording_indices : np.ndarray
        Global recording-channel indices passed to epoch.epoch(...).
    channel_names : np.ndarray
        Display labels corresponding to recording_indices.

    Notes
    -----
    In the current ECAP design, ``neural_window_indices`` is ordered by the
    neural-channel list, while ``epoch(..., recording_channels=...)`` selects
    from the full recording-channel axis. This helper keeps those two index
    spaces explicit.
    """
    window_indices = np.asarray(epoch.neural_window_indices)
    if window_indices.ndim != 3 or window_indices.shape[2] != 2:
        raise ValueError(
            "neural_window_indices must have shape "
            "[neural_channel, fiber_window, 2]."
        )

    n_window_channels = int(window_indices.shape[0])
    names = np.asarray(_channel_names(epoch), dtype=object)

    neural_channels = getattr(epoch, "neural_channels", None)
    if neural_channels is None:
        # Compatibility fallback for data sets where neural channels occupy
        # the first N recording channels.
        neural_channels = np.arange(n_window_channels, dtype=int)
    else:
        neural_channels = np.asarray(neural_channels, dtype=int).reshape(-1)

    if len(neural_channels) != n_window_channels:
        raise ValueError(
            "epoch.neural_channels and epoch.neural_window_indices disagree "
            "about the number of neural recording channels."
        )

    if np.any(neural_channels < 0) or np.any(neural_channels >= len(names)):
        raise IndexError("epoch.neural_channels contains an invalid channel index.")

    if rec_channels is None:
        recording_indices = neural_channels.copy()

    elif isinstance(rec_channels, str):
        recording_indices = np.array(
            [_resolve_channel_index(epoch, rec_channels)],
            dtype=int,
        )

    elif isinstance(rec_channels, (int, np.integer)):
        recording_indices = np.array([int(rec_channels)], dtype=int)

    else:
        values = list(rec_channels)
        if not values:
            raise ValueError("No recording channels were selected.")

        resolved = []
        for value in values:
            if isinstance(value, str):
                resolved.append(_resolve_channel_index(epoch, value))
            elif isinstance(value, (int, np.integer)):
                resolved.append(int(value))
            else:
                raise TypeError(
                    "rec_channels entries must be recording-channel names "
                    "or integer indices."
                )
        recording_indices = np.asarray(resolved, dtype=int)

    if recording_indices.size == 0:
        raise ValueError("No recording channels were selected.")

    if np.any(recording_indices < 0) or np.any(recording_indices >= len(names)):
        raise IndexError("A selected recording-channel index is out of range.")

    window_positions = []
    for recording_index in recording_indices:
        matches = np.flatnonzero(neural_channels == recording_index)
        if matches.size == 0:
            raise ValueError(
                f"Recording channel {names[recording_index]!r} is not an ENG "
                "channel represented in neural_window_indices."
            )
        window_positions.append(int(matches[0]))

    return (
        np.asarray(window_positions, dtype=int),
        recording_indices,
        names[recording_indices],
    )


def _highest_amplitude_parameter(epoch):
    """Return (parameter_key, amplitude, amplitude_column) for max amplitude."""
    par_df = _parameter_table(epoch)
    amp_col = _find_amp_col(par_df)
    values = pd.to_numeric(par_df[amp_col], errors="coerce")

    if not np.isfinite(values.to_numpy(dtype=float)).any():
        raise ValueError(f"No valid amplitudes were found in {amp_col!r}.")

    parameter_key = values.idxmax()
    amplitude = float(values.loc[parameter_key])
    return parameter_key, amplitude, amp_col


def _nearest_amplitude_parameter(epoch, requested_amplitude: float):
    """Return nearest parameter key and actual amplitude to a request."""
    par_df = _parameter_table(epoch)
    amp_col = _find_amp_col(par_df)

    values = pd.to_numeric(par_df[amp_col], errors="coerce").to_numpy(dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        raise ValueError(f"No valid amplitudes were found in {amp_col!r}.")

    distances = np.abs(values[valid_positions] - requested_amplitude)
    nearest_position = int(valid_positions[np.argmin(distances)])

    parameter_key = par_df.index[nearest_position]
    selected_amplitude = float(values[nearest_position])
    exact_match = bool(
        np.isclose(
            selected_amplitude,
            requested_amplitude,
            rtol=0.0,
            atol=1e-9,
        )
    )

    return parameter_key, selected_amplitude, exact_match, amp_col


def _load_epoch_numpy(epoch, parameter_key, recording_channels=None) -> np.ndarray:
    """Load one parameter as (pulses, recording_channels, samples)."""
    data = epoch.epoch(
        parameter_key,
        recording_channels=recording_channels,
    )

    if hasattr(data, "compute"):
        data = data.compute()

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(
            "epoch.epoch(parameter, ...) must return data with shape "
            "[pulses, recording_channels, samples]."
        )

    return data


def _resolve_time_axis(epoch, parameter_key, n_samples: int, time=None) -> np.ndarray:
    """Return a validated time axis, preferring the current EpochData API."""
    if time is None:
        time = np.asarray(epoch.time_axis(parameter_key), dtype=float)
    else:
        time = np.asarray(time, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if len(time) != n_samples:
        raise ValueError(
            "time must have the same length as the epoch signal axis. "
            f"Received {len(time)} values for {n_samples} samples."
        )

    return time


def _time_window_mask(time: np.ndarray, window_s) -> np.ndarray:
    """Return a half-open boolean mask for a relative-time window."""
    try:
        start_s, stop_s = map(float, window_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("A time window must contain exactly two values.") from exc

    if stop_s <= start_s:
        raise ValueError("A time window must satisfy stop > start.")

    mask = (time >= start_s) & (time < stop_s)
    if not np.any(mask):
        raise ValueError(
            f"Requested time window {window_s!r} contains no epoch samples."
        )

    return mask


def _mean_trace_for_parameter(ecap, parameter_key, channel_index: int) -> np.ndarray:
    """
    Return one channel's pulse-mean waveform using compute_mean_traces().

    ``compute_mean_traces(parameter)`` is the current public API for a
    computed mean shaped (recording_channels, samples). It also reuses the
    all-parameter mean cache when that cache already exists.
    """
    mean_trace = np.asarray(ecap.compute_mean_traces(parameter_key))

    if mean_trace.ndim != 2:
        raise ValueError(
            "compute_mean_traces(parameter) must return "
            "[recording_channels, samples]."
        )
    if channel_index < 0 or channel_index >= mean_trace.shape[0]:
        raise IndexError("Selected recording channel is not present in mean trace.")

    return np.asarray(mean_trace[channel_index], dtype=float)


def _surface_data(ecap, channel, par_df: pd.DataFrame):
    """Build amplitude-sorted mean waveforms and their shared time axis."""
    amp_col = _find_amp_col(par_df)
    channel_index = _resolve_channel_index(ecap, channel)

    groups = []
    for amplitude, df_amp in par_df.groupby(amp_col, sort=False):
        try:
            amplitude_float = float(amplitude)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(amplitude_float) or df_amp.empty:
            continue
        groups.append((amplitude_float, df_amp.index[0]))

    if not groups:
        raise RuntimeError(
            "No valid parameter rows remained after applying constraints."
        )

    groups.sort(key=lambda item: item[0])
    amplitudes = np.asarray([item[0] for item in groups], dtype=float)
    parameter_keys = [item[1] for item in groups]

    traces = []
    reference_time = None

    for parameter_key in parameter_keys:
        trace = _mean_trace_for_parameter(ecap, parameter_key, channel_index)
        time = np.asarray(ecap.time_axis(parameter_key), dtype=float)

        if trace.ndim != 1:
            raise ValueError("A selected mean waveform must be one-dimensional.")
        if len(time) != len(trace):
            raise ValueError(
                f"Time-axis length for parameter {parameter_key!r} does not "
                "match its mean waveform."
            )

        if reference_time is None:
            reference_time = time
        elif len(time) != len(reference_time) or not np.allclose(
            time,
            reference_time,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Selected parameters do not share one epoch time axis. "
                "Set x_lim to a fixed (start, stop) epoch window before "
                "creating a surface plot."
            )

        traces.append(trace)

    return amplitudes, parameter_keys, np.stack(traces, axis=0), reference_time


def _finite_min_max(values):
    values = np.asarray(values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(finite_values.min()), float(finite_values.max())


def _padded_limits(values, padding_fraction: float):
    result = _finite_min_max(values)
    if result is None:
        return None

    lower, upper = result
    data_range = upper - lower
    if data_range == 0:
        data_range = max(abs(lower), 1.0)

    padding = data_range * padding_fraction
    return lower - padding, upper + padding


def _limited_shoulder_limits(
    window_values,
    displayed_values,
    *,
    ylim_padding: float,
    shoulder_ylim_factor: float,
):
    window_limits = _padded_limits(window_values, ylim_padding)
    if window_limits is None:
        return _padded_limits(displayed_values, ylim_padding)

    main_lower, main_upper = window_limits
    displayed_extrema = _finite_min_max(displayed_values)
    if displayed_extrema is None:
        return window_limits

    displayed_lower, displayed_upper = displayed_extrema
    main_center = (main_lower + main_upper) / 2.0
    main_half_range = (main_upper - main_lower) / 2.0
    if main_half_range <= 0:
        main_half_range = 1.0

    allowed_lower = main_center - main_half_range * shoulder_ylim_factor
    allowed_upper = main_center + main_half_range * shoulder_ylim_factor

    shoulder_padding = (main_upper - main_lower) * ylim_padding
    requested_lower = min(main_lower, displayed_lower - shoulder_padding)
    requested_upper = max(main_upper, displayed_upper + shoulder_padding)

    final_lower = max(requested_lower, allowed_lower)
    final_upper = min(requested_upper, allowed_upper)

    # Always retain the true neural-window range.
    final_lower = min(final_lower, main_lower)
    final_upper = max(final_upper, main_upper)
    return final_lower, final_upper


def _validate_ylim_settings(ylim, ylim_padding, shoulder_ylim_factor):
    if not isinstance(ylim_padding, Real):
        raise TypeError("ylim_padding must be numeric.")
    ylim_padding = float(ylim_padding)
    if not np.isfinite(ylim_padding) or ylim_padding < 0:
        raise ValueError("ylim_padding must be finite and nonnegative.")

    if not isinstance(shoulder_ylim_factor, Real):
        raise TypeError("shoulder_ylim_factor must be numeric.")
    shoulder_ylim_factor = float(shoulder_ylim_factor)
    if not np.isfinite(shoulder_ylim_factor) or shoulder_ylim_factor < 1:
        raise ValueError(
            "shoulder_ylim_factor must be finite and greater than or equal to 1."
        )

    if ylim is True or ylim is False or ylim is None:
        fixed_ylim = None
    else:
        if not isinstance(ylim, (tuple, list, np.ndarray)) or len(ylim) != 2:
            raise ValueError(
                "ylim must be True, False, None, or a two-value sequence."
            )
        fixed_ylim = (float(ylim[0]), float(ylim[1]))

    return fixed_ylim, ylim_padding, shoulder_ylim_factor


def _shoulder_percent(show_shoulders) -> float:
    if isinstance(show_shoulders, (bool, np.bool_)):
        return 100.0 if show_shoulders else 0.0

    if isinstance(show_shoulders, Real):
        value = float(show_shoulders)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "show_shoulders must be a finite, nonnegative percentage."
            )
        return value

    raise TypeError(
        "show_shoulders must be False, True, or a nonnegative percentage."
    )


def _fiber_selection(epoch, plot_idv_fibers, n_available_fibers: int):
    if plot_idv_fibers is False or plot_idv_fibers is None:
        plot_fibers = []
    elif plot_idv_fibers is True:
        plot_fibers = list(range(n_available_fibers))
    elif isinstance(plot_idv_fibers, (int, np.integer)):
        plot_fibers = [int(plot_idv_fibers)]
    else:
        plot_fibers = [int(index) for index in plot_idv_fibers]

    if any(index < 0 or index >= n_available_fibers for index in plot_fibers):
        raise IndexError("Fiber-window index out of range.")

    names = np.asarray(epoch.neural_fiber_names, dtype=object)
    if len(names) < n_available_fibers:
        raise ValueError(
            "neural_fiber_names does not contain enough labels for "
            "neural_window_indices."
        )

    return plot_fibers, names[plot_fibers] if plot_fibers else np.array([])


# =============================================================================
# Surface plots
# =============================================================================

def plot_ecap_surface(
    ecap,
    channel="RawE 1",
    x_lim=None,
    constraints=None,
    wireframe=False,
):
    """Plot mean ECAP waveforms across stimulation amplitudes in Matplotlib."""
    if x_lim is not None:
        ecap.epoch_window = x_lim

    par_df = _parameter_table(ecap).copy()

    if constraints:
        for column, value in constraints.items():
            if column in par_df.columns:
                par_df = par_df.loc[par_df[column] == value]

    amplitudes, _, Z, time = _surface_data(ecap, channel, par_df)
    T, A = np.meshgrid(time, amplitudes)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    if wireframe:
        ax.plot_wireframe(T, A, Z)
    else:
        ax.plot_surface(
            T,
            A,
            Z,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stim amplitude (µA)")
    ax.set_zlabel("ECAP (a.u.)")
    ax.set_title(f"Average ECAP vs time across amplitudes - channel: {channel}")
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_ecap_surface_plotly(
    ecap,
    channel,
    x_lim=None,
    y_lim=None,
    baseline_s=None,
    absolute=False,
    downsample=1,
    renderer=None,
):
    """Create an interactive Plotly ECAP surface across amplitudes."""
    if x_lim is not None:
        ecap.epoch_window = x_lim

    if not isinstance(downsample, (int, np.integer)) or int(downsample) < 1:
        raise ValueError("downsample must be an integer >= 1.")
    downsample = int(downsample)

    par_df = _parameter_table(ecap).copy()
    amplitudes, _, Z, time = _surface_data(ecap, channel, par_df)

    if baseline_s is not None:
        baseline_mask = _time_window_mask(time, baseline_s)
        baseline = np.nanmean(Z[:, baseline_mask], axis=1, keepdims=True)
        Z = Z - baseline

    if absolute:
        Z = np.abs(Z)

    if downsample > 1:
        Z = Z[:, ::downsample]
        time = time[::downsample]

    T = np.tile(time, (len(amplitudes), 1))
    A = np.tile(amplitudes[:, None], (1, len(time)))

    fig = go.Figure(
        data=[
            go.Surface(
                x=T,
                y=A,
                z=Z,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )

    scene = {
        "xaxis_title": "Time (s)",
        "yaxis_title": "Stim amplitude (µA)",
        "zaxis_title": "ECAP (a.u.)",
        "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}},
    }
    if y_lim is not None:
        if len(y_lim) != 2:
            raise ValueError("y_lim must contain exactly two values.")
        scene["yaxis"] = {
            "title": "Stim amplitude (µA)",
            "range": [float(y_lim[0]), float(y_lim[1])],
        }

    fig.update_layout(
        title=f"Interactive ECAP surface - ch: {channel}",
        scene=scene,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    if renderer is not None:
        fig.show(renderer=renderer)

    return fig


# =============================================================================
# Matplotlib pulse-level traces
# =============================================================================

def plot_highest_amp_traces(
    epoch,
    rec_channels=None,
    time=None,
    plot_v_lines=False,
    plot_idv_fibers=False,
    show_shoulders=False,
    ylim=None,
    ylim_padding=0.10,
    shoulder_ylim_factor=2.0,
):
    """
    Plot pulse-level traces and their mean at the highest stimulation amplitude.

    The current EpochData API is parameter driven: the highest-amplitude
    parameter key is selected from ``epoch.parameters.parameters`` and passed
    directly to ``epoch.epoch(parameter, recording_channels=...)``.
    """
    fixed_ylim, ylim_padding, shoulder_ylim_factor = _validate_ylim_settings(
        ylim,
        ylim_padding,
        shoulder_ylim_factor,
    )
    shoulder_percent = _shoulder_percent(show_shoulders)

    parameter_key, max_amp, _ = _highest_amplitude_parameter(epoch)

    window_indices = np.asarray(epoch.neural_window_indices)
    if window_indices.ndim != 3 or window_indices.shape[2] != 2:
        raise ValueError(
            "neural_window_indices must have shape "
            "[neural_channel, fiber_window, 2]."
        )

    window_positions, recording_indices, channel_names = _resolve_neural_channels(
        epoch,
        rec_channels,
    )

    data = _load_epoch_numpy(
        epoch,
        parameter_key,
        recording_channels=recording_indices.tolist(),
    )

    n_pulses, n_plot_channels, n_samples = data.shape
    time = _resolve_time_axis(epoch, parameter_key, n_samples, time=time)

    n_available_fibers = int(window_indices.shape[1])
    plot_fibers, fiber_labels = _fiber_selection(
        epoch,
        plot_idv_fibers,
        n_available_fibers,
    )

    n_rows = 1 + len(plot_fibers)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_plot_channels,
        figsize=(4 * n_plot_channels, 2.8 * n_rows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # -----------------------------------------------------------------
    # Full-trace row
    # -----------------------------------------------------------------
    for col in range(n_plot_channels):
        ax = axes[0, col]
        traces = data[:, col, :]
        mean_trace = np.nanmean(traces, axis=0)

        ax.plot(time, traces.T, color="gray", alpha=0.20, linewidth=0.5)
        ax.plot(time, mean_trace, color="blue", linewidth=2)

        if plot_v_lines:
            boundaries = window_indices[window_positions[col]].reshape(-1)
            boundaries = boundaries[np.isfinite(boundaries)]
            boundaries = np.unique(np.rint(boundaries).astype(int))
            boundaries = boundaries[
                (boundaries >= 0) & (boundaries < n_samples)
            ]

            for boundary_index in boundaries:
                ax.axvline(
                    x=time[boundary_index],
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

        ax.set_title(str(channel_names[col]))
        if col == 0:
            ax.set_ylabel("Full trace")

        if ylim is True:
            limits = _padded_limits(traces, ylim_padding)
            if limits is not None:
                ax.set_ylim(limits)
        elif fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)

    # -----------------------------------------------------------------
    # Fiber-window rows
    # -----------------------------------------------------------------
    for row, fiber_index in enumerate(plot_fibers, start=1):
        fiber_label = str(fiber_labels[row - 1])

        for col in range(n_plot_channels):
            ax = axes[row, col]
            bounds = window_indices[window_positions[col], fiber_index]

            if not np.all(np.isfinite(bounds)):
                ax.text(
                    0.5,
                    0.5,
                    "Invalid window",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            raw_start, raw_stop = np.rint(bounds).astype(int)
            window_length = raw_stop - raw_start
            if window_length <= 0:
                ax.text(
                    0.5,
                    0.5,
                    "Invalid window",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            shoulder_samples = int(
                round(window_length * shoulder_percent / 100.0)
            )

            start = max(0, raw_start)
            stop = min(n_samples, raw_stop)
            display_start = max(0, raw_start - shoulder_samples)
            display_stop = min(n_samples, raw_stop + shoulder_samples)

            if stop <= start or display_stop <= display_start:
                ax.text(
                    0.5,
                    0.5,
                    "Window outside signal",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            traces = data[:, col, :]
            window_traces = traces[:, start:stop]
            displayed_traces = traces[:, display_start:display_stop]
            displayed_time = time[display_start:display_stop]
            displayed_mean = np.nanmean(displayed_traces, axis=0)

            ax.plot(
                displayed_time,
                displayed_traces.T,
                color="gray",
                alpha=0.20,
                linewidth=0.5,
            )
            ax.plot(displayed_time, displayed_mean, color="blue", linewidth=2)

            if plot_v_lines:
                if display_start <= start < display_stop:
                    ax.axvline(
                        x=time[start],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                    )

                if display_start < stop < display_stop and stop < n_samples:
                    ax.axvline(
                        x=time[stop],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                    )

            if col == 0:
                ax.set_ylabel(fiber_label)

            if ylim is True:
                limits = _limited_shoulder_limits(
                    window_traces,
                    displayed_traces,
                    ylim_padding=ylim_padding,
                    shoulder_ylim_factor=shoulder_ylim_factor,
                )
                if limits is not None:
                    ax.set_ylim(limits)
            elif fixed_ylim is not None:
                ax.set_ylim(fixed_ylim)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Highest Stimulation Amplitude: {max_amp:g} μA",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

    return fig, axes


# =============================================================================
# Plotly pulse-level traces
# =============================================================================

def _add_window_line(fig, x_value, row, col):
    fig.add_vline(
        x=x_value,
        line_width=1,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        row=row,
        col=col,
    )


def plot_highest_amp_traces_plotly(
    epoch,
    rec_channels=None,
    time=None,
    plot_v_lines=False,
    ylim=None,
    show_shoulders=True,
):
    """
    Plot only the full mean trace at the highest stimulation amplitude.

    This preserves the original one-row Plotly behavior while using the
    current parameter-driven EpochData API.
    """
    del show_shoulders  # retained for call compatibility; no fiber rows here

    parameter_key, max_amp, _ = _highest_amplitude_parameter(epoch)

    window_positions, recording_indices, channel_names = _resolve_neural_channels(
        epoch,
        rec_channels,
    )

    data = _load_epoch_numpy(
        epoch,
        parameter_key,
        recording_channels=recording_indices.tolist(),
    )

    _, n_plot_channels, n_samples = data.shape
    time = _resolve_time_axis(epoch, parameter_key, n_samples, time=time)

    if ylim is True:
        max_value = np.nanpercentile(np.abs(data), 95) * 1.25
        computed_ylim = (-max_value, max_value)
    elif ylim is None or ylim is False:
        computed_ylim = None
    else:
        if len(ylim) != 2:
            raise ValueError("A fixed ylim must contain two values.")
        computed_ylim = (float(ylim[0]), float(ylim[1]))

    fig = make_subplots(
        rows=1,
        cols=n_plot_channels,
        subplot_titles=list(channel_names),
        shared_yaxes=True,
        shared_xaxes=True,
    )

    window_indices = np.asarray(epoch.neural_window_indices)

    for col in range(1, n_plot_channels + 1):
        local_index = col - 1
        mean_trace = np.nanmean(data[:, local_index, :], axis=0)

        fig.add_trace(
            go.Scatter(
                x=time,
                y=mean_trace,
                mode="lines",
                line={"width": 2},
                name=str(channel_names[local_index]),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        if plot_v_lines:
            boundaries = window_indices[window_positions[local_index]].reshape(-1)
            boundaries = boundaries[np.isfinite(boundaries)]
            boundaries = np.unique(np.rint(boundaries).astype(int))
            boundaries = boundaries[
                (boundaries >= 0) & (boundaries < n_samples)
            ]

            for boundary_index in boundaries:
                _add_window_line(
                    fig,
                    time[boundary_index],
                    row=1,
                    col=col,
                )

        if computed_ylim is not None:
            fig.update_yaxes(range=list(computed_ylim), row=1, col=col)

    fig.update_layout(
        title=f"Highest Stimulation Amplitude: {max_amp:g} μA",
        height=400,
        width=max(350 * n_plot_channels, 600),
        template="plotly_white",
        autosize=True,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Signal", row=1, col=1)

    output = Path("plot_highest_amp_traces.html").resolve()
    pio.write_html(
        fig,
        file=str(output),
        auto_open=True,
        config={"responsive": True},
    )

    return fig, max_amp, parameter_key


def a_alpha_range_max_right_bound(data, margin=0.1):
    data = np.asarray(data)
    max_val = data.max()
    right_bound = data[-1]
    ymax = max_val + (max_val - right_bound) * margin
    ymin = right_bound - (max_val - right_bound) * margin
    return ymin, ymax


def plot_stim_amp_traces_plotly(
    epoch,
    stim_amplitude,
    rec_channels=None,
    time=None,
    plot_v_lines=True,
    plot_idv_fibers=True,
    show_shoulders=False,
    ylim=True,
    ylim_padding=0.10,
    shoulder_ylim_factor=2.0,
    show=True,
):
    """
    Plot pulse-level ECAP traces at a requested stimulation amplitude.

    If the exact amplitude is absent, the closest available amplitude is
    selected and a warning is emitted. The selected parameter key is passed
    directly to ``epoch.epoch(...)``; there is no amplitude-index lookup.
    """
    if not isinstance(stim_amplitude, Real):
        raise TypeError("stim_amplitude must be numeric.")

    requested_amplitude = float(stim_amplitude)
    if not np.isfinite(requested_amplitude):
        raise ValueError("stim_amplitude must be finite.")

    fixed_ylim, ylim_padding, shoulder_ylim_factor = _validate_ylim_settings(
        ylim,
        ylim_padding,
        shoulder_ylim_factor,
    )
    shoulder_percent = _shoulder_percent(show_shoulders)

    parameter_key, selected_amplitude, exact_match, _ = (
        _nearest_amplitude_parameter(epoch, requested_amplitude)
    )

    if not exact_match:
        warnings.warn(
            f"Requested amplitude {requested_amplitude:g} μA was not found. "
            f"Plotting the closest available amplitude: "
            f"{selected_amplitude:g} μA.",
            UserWarning,
            stacklevel=2,
        )

    window_indices = np.asarray(epoch.neural_window_indices)
    if window_indices.ndim != 3 or window_indices.shape[2] != 2:
        raise ValueError(
            "neural_window_indices must have shape "
            "[neural_channel, fiber_window, 2]."
        )

    window_positions, recording_indices, channel_names = _resolve_neural_channels(
        epoch,
        rec_channels,
    )

    data = _load_epoch_numpy(
        epoch,
        parameter_key,
        recording_channels=recording_indices.tolist(),
    )

    n_pulses, n_plot_channels, n_samples = data.shape
    time = _resolve_time_axis(epoch, parameter_key, n_samples, time=time)

    n_available_fibers = int(window_indices.shape[1])
    plot_fibers, fiber_labels = _fiber_selection(
        epoch,
        plot_idv_fibers,
        n_available_fibers,
    )

    n_rows = 1 + len(plot_fibers)

    subplot_titles = []
    for row_index in range(n_rows):
        for channel_name in channel_names:
            subplot_titles.append(str(channel_name) if row_index == 0 else "")

    fig = make_subplots(
        rows=n_rows,
        cols=n_plot_channels,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04 if n_plot_channels > 1 else 0.08,
        vertical_spacing=min(0.08, 0.25 / n_rows),
    )

    # -----------------------------------------------------------------
    # Full-trace row
    # -----------------------------------------------------------------
    for col in range(1, n_plot_channels + 1):
        local_index = col - 1
        traces = data[:, local_index, :]
        mean_trace = np.nanmean(traces, axis=0)

        for pulse_index in range(n_pulses):
            fig.add_trace(
                go.Scattergl(
                    x=time,
                    y=traces[pulse_index],
                    mode="lines",
                    line={"color": "rgba(120,120,120,0.25)", "width": 0.7},
                    name="Individual trace",
                    legendgroup="individual",
                    showlegend=False,
                    customdata=np.full(len(time), pulse_index),
                    hovertemplate=(
                        f"Channel: {channel_names[local_index]}"
                        "<br>Pulse: %{customdata}"
                        "<br>Time: %{x:.6g} s"
                        "<br>Signal: %{y:.5g}"
                        f"<br>Amplitude: {selected_amplitude:g} μA"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

        fig.add_trace(
            go.Scattergl(
                x=time,
                y=mean_trace,
                mode="lines",
                line={"color": "blue", "width": 2.5},
                name="Mean trace",
                legendgroup="mean",
                showlegend=(col == 1),
                hovertemplate=(
                    f"Channel: {channel_names[local_index]}"
                    "<br>Mean trace"
                    "<br>Time: %{x:.6g} s"
                    "<br>Signal: %{y:.5g}"
                    f"<br>Amplitude: {selected_amplitude:g} μA"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )

        if plot_v_lines:
            boundaries = window_indices[window_positions[local_index]].reshape(-1)
            boundaries = boundaries[np.isfinite(boundaries)]
            boundaries = np.unique(np.rint(boundaries).astype(int))
            boundaries = boundaries[
                (boundaries >= 0) & (boundaries < n_samples)
            ]

            for boundary_index in boundaries:
                _add_window_line(
                    fig,
                    time[boundary_index],
                    row=1,
                    col=col,
                )

        if ylim is True:
            limits = _padded_limits(traces, ylim_padding)
            if limits is not None:
                fig.update_yaxes(range=list(limits), row=1, col=col)
        elif fixed_ylim is not None:
            fig.update_yaxes(range=list(fixed_ylim), row=1, col=col)

        if col == 1:
            fig.update_yaxes(title_text="Full trace", row=1, col=col)

    # -----------------------------------------------------------------
    # Fiber-window rows
    # -----------------------------------------------------------------
    for fiber_row, fiber_index in enumerate(plot_fibers, start=2):
        fiber_label = str(fiber_labels[fiber_row - 2])

        for col in range(1, n_plot_channels + 1):
            local_index = col - 1
            bounds = window_indices[window_positions[local_index], fiber_index]

            if not np.all(np.isfinite(bounds)):
                continue

            raw_start, raw_stop = np.rint(bounds).astype(int)
            window_length = raw_stop - raw_start
            if window_length <= 0:
                continue

            shoulder_samples = int(
                round(window_length * shoulder_percent / 100.0)
            )

            start = max(0, raw_start)
            stop = min(n_samples, raw_stop)
            display_start = max(0, raw_start - shoulder_samples)
            display_stop = min(n_samples, raw_stop + shoulder_samples)

            if stop <= start or display_stop <= display_start:
                continue

            traces = data[:, local_index, :]
            window_traces = traces[:, start:stop]
            displayed_traces = traces[:, display_start:display_stop]
            displayed_time = time[display_start:display_stop]
            displayed_mean = np.nanmean(displayed_traces, axis=0)

            for pulse_index in range(n_pulses):
                fig.add_trace(
                    go.Scattergl(
                        x=displayed_time,
                        y=displayed_traces[pulse_index],
                        mode="lines",
                        line={
                            "color": "rgba(120,120,120,0.25)",
                            "width": 0.7,
                        },
                        name="Individual trace",
                        legendgroup="individual",
                        showlegend=False,
                        customdata=np.full(len(displayed_time), pulse_index),
                        hovertemplate=(
                            f"Channel: {channel_names[local_index]}"
                            f"<br>Window: {fiber_label}"
                            "<br>Pulse: %{customdata}"
                            "<br>Time: %{x:.6g} s"
                            "<br>Signal: %{y:.5g}"
                            f"<br>Amplitude: {selected_amplitude:g} μA"
                            "<extra></extra>"
                        ),
                    ),
                    row=fiber_row,
                    col=col,
                )

            fig.add_trace(
                go.Scattergl(
                    x=displayed_time,
                    y=displayed_mean,
                    mode="lines",
                    line={"color": "blue", "width": 2.5},
                    name="Mean trace",
                    legendgroup="mean",
                    showlegend=False,
                    hovertemplate=(
                        f"Channel: {channel_names[local_index]}"
                        f"<br>Window: {fiber_label}"
                        "<br>Mean trace"
                        "<br>Time: %{x:.6g} s"
                        "<br>Signal: %{y:.5g}"
                        f"<br>Amplitude: {selected_amplitude:g} μA"
                        "<extra></extra>"
                    ),
                ),
                row=fiber_row,
                col=col,
            )

            if plot_v_lines:
                if display_start <= start < display_stop:
                    _add_window_line(
                        fig,
                        time[start],
                        row=fiber_row,
                        col=col,
                    )

                if display_start < stop < display_stop and stop < n_samples:
                    _add_window_line(
                        fig,
                        time[stop],
                        row=fiber_row,
                        col=col,
                    )

            if ylim is True:
                limits = _limited_shoulder_limits(
                    window_traces,
                    displayed_traces,
                    ylim_padding=ylim_padding,
                    shoulder_ylim_factor=shoulder_ylim_factor,
                )
                if limits is not None:
                    fig.update_yaxes(
                        range=list(limits),
                        row=fiber_row,
                        col=col,
                    )
            elif fixed_ylim is not None:
                fig.update_yaxes(
                    range=list(fixed_ylim),
                    row=fiber_row,
                    col=col,
                )

            if col == 1:
                fig.update_yaxes(
                    title_text=fiber_label,
                    row=fiber_row,
                    col=col,
                )

    # -----------------------------------------------------------------
    # Layout and export
    # -----------------------------------------------------------------
    for col in range(1, n_plot_channels + 1):
        fig.update_xaxes(title_text="Time (s)", row=n_rows, col=col)

    if exact_match:
        title = f"Stimulation amplitude: {selected_amplitude:g} μA"
    else:
        title = (
            f"Requested {requested_amplitude:g} μA - showing closest "
            f"available amplitude: {selected_amplitude:g} μA"
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        hovermode="closest",
        height=max(500, 280 * n_rows),
        width=max(750, 400 * n_plot_channels),
        margin={"l": 80, "r": 30, "t": 100, "b": 70},
    )

    fig.write_html(
        r"D:\ImThera\data_processed\test.html",
        include_plotlyjs=True,
        full_html=True,
        auto_open=True,
    )

    if show:
        fig.show()

    return fig, selected_amplitude, parameter_key
