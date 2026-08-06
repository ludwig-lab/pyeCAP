import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numbers import Real
import warnings

def _find_amp_col(df):
    cand = [
        "pulse amplitude (μA)", "pulse amplitude (uA)",
        "pulse amplitude A (μA)", "pulse amplitude A (uA)",
        "pulse amplitude", "pulse_amplitude_uA"
    ]
    for c in cand:
        if c in df.columns:
            return c
    raise KeyError("Could not find a pulse-amplitude column in stim.parameters")

def plot_ecap_surface(ecap, channel="RawE 1", x_lim=None, constraints=None, wireframe=False):
    """
    ecap         : your ECAP object (Ephys+Stim already loaded)
    channel      : recording channel name or index to plot (e.g., 'RawE 1')
    x_lim        : tuple (t0, t1) seconds for the averaging window; overrides ecap.x_lim if given
    constraints  : dict to filter a consistent subset of parameters (e.g., {'frequency (Hz)': 25, 'pulse duration (ms)': 0.2})
    wireframe    : True for ax.plot_wireframe; False for ax.plot_surface
    """
    if x_lim is not None:
        ecap.epoch_window = x_lim  # set a fixed epoch window for all parameters

    # Make epoching snappy if you’re iterating a lot
    if not hasattr(ecap, "_persisted_array"):
        fs = ecap.ts_data.sample_rate
        win = (ecap.epoch_window[1] - ecap.epoch_window[0]) if ecap.epoch_window else 0.010
        ecap.persist_ts(time_chunk=int(max(1, win*fs*6)))

    par_df = ecap.parameters.parameters  # DataFrame of stim parameter rows, indexed by the ECAP 'parameter' key
    amp_col = _find_amp_col(par_df)

    # Optional constraints (keep frequency/pulse width constant, pick a voice/store, etc.)
    if constraints:
        for k, v in constraints.items():
            if k in par_df.columns:
                par_df = par_df.loc[par_df[k] == v]
            else:
                # ignore silently if constraint column not present
                pass

    # Group by amplitude
    groups = []
    for amp, df_amp in par_df.groupby(amp_col):
        # Pick one representative parameter key per amplitude (or average later if you have many)
        key = df_amp.index[0]
        groups.append((float(amp), key))

    if not groups:
        raise RuntimeError("No parameters found after applying constraints; cannot plot.")

    # Sort by amplitude
    groups.sort(key=lambda t: t[0])
    amps, keys = zip(*groups)

    # Build Z: averaged waveform for the chosen channel at each amplitude
    # Lazily compute mean waveforms, then pull to numpy once.
    ch = [channel] if isinstance(channel, str) else channel
    wf_list = []
    for p in keys:
        wf = ecap.mean_waveform(p, channels=ch)   # (ch, samples) as Dask
        wf_list.append(wf)

    # Compute all at once to minimize scheduler overhead
    wf_stack = np.stack([w.compute() for w in wf_list], axis=0)  # (n_amp, ch, samples)
    # Select first (and only) channel axis
    Z = wf_stack[:, 0, :]  # (n_amp, samples)

    # Time axis (same length for all keys if x_lim is fixed)
    # If you added a time_axis(param) helper, you can use it; otherwise build from x_lim/fs.
    n_samples = Z.shape[1]
    fs = ecap.ts_data.sample_rate
    if ecap.epoch_window is None or ecap.epoch_window == 'auto':
        t = np.arange(n_samples) / fs
    else:
        t0, _ = ecap.epoch_window
        t = (np.arange(n_samples) / fs) + t0

    # Meshgrid for surface
    T, A = np.meshgrid(t, np.array(amps))

    # 3D plot
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    if wireframe:
        ax.plot_wireframe(T, A, Z)
    else:
        ax.plot_surface(T, A, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stim amplitude (µA)")
    ax.set_zlabel("ECAP (a.u.)")
    ax.set_title(f"Average ECAP vs time across amplitudes — channel: {channel}")
    plt.tight_layout()
    plt.show()

def plot_ecap_surface_plotly(
    ecap,
    channel,
    x_lim=None,
    y_lim=None,
    baseline_s=None,          # e.g. (-0.003, -0.001) if x_lim starts negative
    absolute=False,
    downsample=1,             # e.g. 2 to halve the time resolution for snappier plots
    renderer=None             # e.g. "browser", "vscode", "notebook_connected"
):
    # Set shared epoch window (time axis)
    if x_lim is not None:
        ecap.epoch_window = x_lim

    fs = ecap.ts_data.sample_rate
    win = (ecap.epoch_window[1] - ecap.epoch_window[0]) if ecap.epoch_window else 0.010
    if not hasattr(ecap, "_persisted_array"):
        ecap.persist_ts(time_chunk=int(max(1, win*fs*6)))  # auto-persist for speed

    par_df = ecap.parameters.parameters.copy()

    try:
        amp_col = next(c for c in par_df.columns if c.startswith("pulse amp"))
    except StopIteration:
        raise KeyError("Couldn't find pulse amplitude column in stim.parameters")

    groups = []
    for amp, df_amp in par_df.groupby(amp_col):
        if len(df_amp) == 0: continue
        groups.append((float(amp), df_amp.index[0]))  # one key per amplitude
    if not groups:
        raise RuntimeError("No parameter rows")

    groups.sort(key=lambda t: t[0])
    amps, keys = zip(*groups)

    ch = [channel] if isinstance(channel, str) else channel
    wf_list = []
    for p in keys:
        wf = ecap.mean_waveform(p, channels=ch)  # (ch, samples) dask
        if baseline_s is not None:
            b0 = int(np.floor(baseline_s[0] * fs)); b1 = int(np.ceil(baseline_s[1] * fs))
            wf = wf - wf[:, b0:b1].mean(axis=1, keepdims=True)
        if absolute:
            wf = wf.map_blocks(np.abs)
        wf_list.append(wf)

    # Compute in one go
    wf_stack = np.stack([w.compute() for w in wf_list], axis=0)  # (n_amp, ch, samples)
    Z = wf_stack[:, 0, :]  # first (or only) channel

    # Time axis
    n_samples = Z.shape[1]
    if ecap.epoch_window is None or ecap.epoch_window == "auto":
        t = np.arange(n_samples) / fs
    else:
        t0, _ = ecap.epoch_window
        t = (np.arange(n_samples) / fs) + t0

    # Optional downsample for interactivity
    if downsample > 1:
        Z = Z[:, ::downsample]
        t = t[::downsample]

    T = np.tile(t, (len(amps), 1))
    A = np.tile(np.array(amps)[:, None], (1, len(t)))

    fig = go.Figure(data=[go.Surface(x=T, y=A, z=Z, colorscale="Viridis", showscale=True)])
    fig.update_layout(
        title=f"Interactive ECAP surface — ch: {channel}",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Stim amplitude (µA)",
            zaxis_title="ECAP (a.u.)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if renderer is not None:
        fig.show(renderer=renderer)
    return fig

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
    Plot individual traces and their average at the highest stimulation
    amplitude.

    Expected array shapes
    ---------------------
    epoch.epoch_array:
        [stimulation_amplitude, pulse_train, channel, signal]

    epoch.neural_window_indices:
        [neural_channel, fiber_window, start/end]

    Parameters
    ----------
    epoch
        Epoch object containing the data and neural-window information.

    rec_channels : int, sequence of int, or None
        Neural recording-channel indices to plot.

        If None, all channels represented in neural_window_indices are
        plotted.

    time : array-like or None
        Time axis corresponding to the signal dimension. If None, sample
        indices are used.

    plot_v_lines : bool
        Mark the true neural-window boundaries.

    plot_idv_fibers : bool, int, sequence of int, or None
        False or None:
            Plot only full traces.
        True:
            Plot every neural/fiber window.
        int:
            Plot one fiber window.
        sequence of int:
            Plot selected fiber windows.

    show_shoulders : bool or number
        Amount of additional signal displayed on each side of each window.

        False:
            No shoulders.
        True:
            Add 100% of the original window length to each side.
        number:
            Percentage of the original window length added to each side.

        Example:
            show_shoulders=50 adds half a window length on each side.

    ylim : bool, tuple, or None
        None or False:
            Use ordinary Matplotlib autoscaling.
        True:
            Calculate subplot-specific y-limits.
        (lower, upper):
            Apply fixed limits to every subplot.

    ylim_padding : float
        Fractional padding added above and below automatically calculated
        limits.

    shoulder_ylim_factor : float
        Maximum amount that shoulder values may expand the y-axis relative
        to the main neural-window limits.

        1.0:
            Ignore shoulder values when setting y-limits.
        2.0:
            Allow shoulders to expand each side of the y-axis up to twice
            the main-window scale.
        3.0:
            Allow more shoulder variation.

        Extreme shoulder artifacts beyond this limit will be clipped.

    Returns
    -------
    fig, axes
        Matplotlib figure and two-dimensional axes array.
    """

    # ============================================================
    # Y-limit helpers
    # ============================================================
    def finite_min_max(values):
        values = np.asarray(values)
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            return None

        return finite_values.min(), finite_values.max()

    def padded_limits(values):
        """
        Calculate limits containing every finite value plus padding.
        """
        result = finite_min_max(values)

        if result is None:
            return None

        y_min, y_max = result
        y_range = y_max - y_min

        if y_range == 0:
            y_range = max(abs(y_min), 1.0)

        padding = y_range * ylim_padding

        return y_min - padding, y_max + padding

    def limited_shoulder_limits(window_values, displayed_values):
        """
        Base limits on the true neural window.

        Shoulder values may expand the limits, but only up to
        shoulder_ylim_factor times the main-window scale.
        """
        window_limits = padded_limits(window_values)

        if window_limits is None:
            return padded_limits(displayed_values)

        main_lower, main_upper = window_limits

        displayed_result = finite_min_max(displayed_values)

        if displayed_result is None:
            return window_limits

        displayed_min, displayed_max = displayed_result

        main_center = (main_lower + main_upper) / 2

        main_lower_span = main_center - main_lower
        main_upper_span = main_upper - main_center

        # Maximum permitted expansion caused by shoulder values.
        allowed_lower = (
            main_center
            - main_lower_span * shoulder_ylim_factor
        )

        allowed_upper = (
            main_center
            + main_upper_span * shoulder_ylim_factor
        )

        # Add some padding around shoulder values when they are within
        # the permitted range.
        main_range = main_upper - main_lower
        shoulder_padding = main_range * ylim_padding

        requested_lower = min(
            main_lower,
            displayed_min - shoulder_padding,
        )

        requested_upper = max(
            main_upper,
            displayed_max + shoulder_padding,
        )

        final_lower = max(
            requested_lower,
            allowed_lower,
        )

        final_upper = min(
            requested_upper,
            allowed_upper,
        )

        # The true neural window must always remain fully visible.
        final_lower = min(final_lower, main_lower)
        final_upper = max(final_upper, main_upper)

        return final_lower, final_upper

    def apply_full_trace_ylim(ax, traces):
        if ylim is True:
            limits = padded_limits(traces)

            if limits is not None:
                ax.set_ylim(limits)

        elif fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)

    def apply_fiber_ylim(
        ax,
        window_traces,
        displayed_traces,
    ):
        if ylim is True:
            limits = limited_shoulder_limits(
                window_values=window_traces,
                displayed_values=displayed_traces,
            )

            if limits is not None:
                ax.set_ylim(limits)

        elif fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)

    # ============================================================
    # Validate y-limit parameters
    # ============================================================
    if not isinstance(ylim_padding, Real):
        raise TypeError("ylim_padding must be numeric")

    ylim_padding = float(ylim_padding)

    if not np.isfinite(ylim_padding) or ylim_padding < 0:
        raise ValueError(
            "ylim_padding must be a finite, nonnegative number"
        )

    if not isinstance(shoulder_ylim_factor, Real):
        raise TypeError("shoulder_ylim_factor must be numeric")

    shoulder_ylim_factor = float(shoulder_ylim_factor)

    if (
        not np.isfinite(shoulder_ylim_factor)
        or shoulder_ylim_factor < 1
    ):
        raise ValueError(
            "shoulder_ylim_factor must be a finite number "
            "greater than or equal to 1"
        )

    if ylim is None or ylim is False or ylim is True:
        fixed_ylim = None

    else:
        if not isinstance(
            ylim,
            (tuple, list, np.ndarray),
        ):
            raise TypeError(
                "ylim must be None, False, True, or a "
                "two-value sequence"
            )

        if len(ylim) != 2:
            raise ValueError(
                "A fixed ylim must contain exactly two values"
            )

        fixed_ylim = tuple(ylim)

    # ============================================================
    # Interpret show_shoulders
    # ============================================================
    if isinstance(show_shoulders, (bool, np.bool_)):
        shoulder_percent = 100.0 if show_shoulders else 0.0

    elif isinstance(show_shoulders, Real):
        shoulder_percent = float(show_shoulders)

        if not np.isfinite(shoulder_percent):
            raise ValueError("show_shoulders must be finite")

        if shoulder_percent < 0:
            raise ValueError(
                "show_shoulders cannot be negative"
            )

    else:
        raise TypeError(
            "show_shoulders must be False, True, or a "
            "nonnegative percentage"
        )

    # ============================================================
    # Highest stimulation amplitude
    # ============================================================
    par_df = epoch.parameters.parameters
    amp_col = _find_amp_col(par_df)

    amp_label = par_df[amp_col].idxmax()
    max_amp = par_df.loc[amp_label, amp_col]
    amp_idx = epoch.parameters_dictionary[amp_label]

    # ============================================================
    # Neural-window indices
    # ============================================================
    window_indices = epoch.neural_window_indices

    if hasattr(window_indices, "compute"):
        window_indices = window_indices.compute()

    window_indices = np.asarray(window_indices)

    if (
        window_indices.ndim != 3
        or window_indices.shape[2] != 2
    ):
        raise ValueError(
            "neural_window_indices must have shape "
            "[channel, fiber_window, 2]"
        )

    n_window_channels = window_indices.shape[0]
    n_available_fibers = window_indices.shape[1]

    # ============================================================
    # Channel selection
    # ============================================================
    n_data_channels = epoch.epoch_array.shape[2]

    if n_window_channels > n_data_channels:
        raise ValueError(
            "neural_window_indices contains more channels "
            "than epoch_array"
        )

    if rec_channels is None:
        plot_channels = np.arange(
            n_window_channels,
            dtype=int,
        )

    elif isinstance(rec_channels, (int, np.integer)):
        plot_channels = np.array(
            [int(rec_channels)],
            dtype=int,
        )

    else:
        plot_channels = np.asarray(
            rec_channels,
            dtype=int,
        )

        if plot_channels.ndim != 1:
            raise ValueError(
                "rec_channels must be an integer or a "
                "one-dimensional sequence"
            )

    if plot_channels.size == 0:
        raise ValueError(
            "No recording channels were selected"
        )

    if np.any(plot_channels < 0):
        raise IndexError(
            "Channel indices cannot be negative"
        )

    if np.any(plot_channels >= n_window_channels):
        raise IndexError(
            "Every selected channel must have a corresponding "
            "entry in neural_window_indices"
        )

    n_plot_channels = len(plot_channels)

    channel_names = np.asarray(
        epoch.ephys.ch_names
    )[plot_channels]

    # ============================================================
    # Highest-amplitude data
    # ============================================================
    data = epoch.epoch_array[amp_idx, :, :, :]
    data = data[:, plot_channels, :]

    if hasattr(data, "compute"):
        data = data.compute()

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            "Selected data must have shape "
            "[pulse_train, channel, signal]"
        )

    _, selected_channel_count, n_samples = data.shape

    if selected_channel_count != n_plot_channels:
        raise ValueError(
            "Selected data does not contain the expected "
            "number of channels"
        )

    # ============================================================
    # Time axis
    # ============================================================
    if time is None:
        time = np.arange(n_samples)

    else:
        time = np.asarray(time)

        if time.ndim != 1:
            raise ValueError(
                "time must be one-dimensional"
            )

        if len(time) != n_samples:
            raise ValueError(
                "time must have the same length as the signal axis"
            )

    # ============================================================
    # Fiber-window selection
    # ============================================================
    if plot_idv_fibers is False or plot_idv_fibers is None:
        plot_fibers = []

    elif plot_idv_fibers is True:
        plot_fibers = list(
            range(n_available_fibers)
        )

    elif isinstance(
        plot_idv_fibers,
        (int, np.integer),
    ):
        plot_fibers = [
            int(plot_idv_fibers)
        ]

    else:
        plot_fibers = [
            int(fiber_idx)
            for fiber_idx in plot_idv_fibers
        ]

    if any(
        fiber_idx < 0
        or fiber_idx >= n_available_fibers
        for fiber_idx in plot_fibers
    ):
        raise IndexError(
            "Fiber-window index out of range"
        )

    n_fibers = len(plot_fibers)

    if n_fibers > 0:
        fiber_names = np.asarray(
            epoch.neural_fiber_names
        )

        if len(fiber_names) < n_available_fibers:
            raise ValueError(
                "neural_fiber_names does not contain "
                "enough labels"
            )

        fiber_labels = fiber_names[plot_fibers]

    else:
        fiber_labels = []

    # ============================================================
    # Create figure
    # ============================================================
    n_rows = 1 + n_fibers

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_plot_channels,
        figsize=(
            4 * n_plot_channels,
            2.8 * n_rows,
        ),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # ============================================================
    # Full-trace row
    # ============================================================
    for col, channel_idx in enumerate(plot_channels):
        ax = axes[0, col]

        traces = data[:, col, :]
        mean_trace = np.nanmean(
            traces,
            axis=0,
        )

        ax.plot(
            time,
            traces.T,
            color="gray",
            alpha=0.20,
            linewidth=0.5,
        )

        ax.plot(
            time,
            mean_trace,
            color="blue",
            linewidth=2,
        )

        if plot_v_lines:
            boundaries = np.unique(
                window_indices[
                    channel_idx,
                    :,
                    :,
                ].astype(int).ravel()
            )

            boundaries = boundaries[
                (boundaries >= 0)
                & (boundaries < n_samples)
            ]

            for boundary_idx in boundaries:
                ax.axvline(
                    x=time[boundary_idx],
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

        ax.set_title(
            str(channel_names[col])
        )

        if col == 0:
            ax.set_ylabel("Full trace")

        apply_full_trace_ylim(
            ax,
            traces,
        )

    # ============================================================
    # Fiber-window rows
    # ============================================================
    for row, fiber_idx in enumerate(
        plot_fibers,
        start=1,
    ):
        for col, channel_idx in enumerate(plot_channels):
            ax = axes[row, col]

            raw_start_idx, raw_end_idx = window_indices[
                channel_idx,
                fiber_idx,
            ].astype(int)

            window_length = (
                raw_end_idx - raw_start_idx
            )

            if window_length <= 0:
                ax.text(
                    0.5,
                    0.5,
                    "Invalid window",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                if col == 0:
                    ax.set_ylabel(
                        str(fiber_labels[row - 1])
                    )

                continue

            shoulder_samples = int(
                round(
                    window_length
                    * shoulder_percent
                    / 100.0
                )
            )

            # True neural window.
            start_idx = max(
                0,
                raw_start_idx,
            )

            end_idx = min(
                n_samples,
                raw_end_idx,
            )

            # Expanded display window.
            display_start_idx = max(
                0,
                raw_start_idx - shoulder_samples,
            )

            display_end_idx = min(
                n_samples,
                raw_end_idx + shoulder_samples,
            )

            if (
                end_idx <= start_idx
                or display_end_idx <= display_start_idx
            ):
                ax.text(
                    0.5,
                    0.5,
                    "Window outside signal",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

                if col == 0:
                    ax.set_ylabel(
                        str(fiber_labels[row - 1])
                    )

                continue

            traces = data[:, col, :]

            # Data from the true analysis window.
            window_traces = traces[
                :,
                start_idx:end_idx,
            ]

            # Data shown, including shoulders.
            displayed_traces = traces[
                :,
                display_start_idx:display_end_idx,
            ]

            displayed_mean = np.nanmean(
                displayed_traces,
                axis=0,
            )

            displayed_time = time[
                display_start_idx:display_end_idx
            ]

            ax.plot(
                displayed_time,
                displayed_traces.T,
                color="gray",
                alpha=0.20,
                linewidth=0.5,
            )

            ax.plot(
                displayed_time,
                displayed_mean,
                color="blue",
                linewidth=2,
            )

            if plot_v_lines:
                if (
                    display_start_idx
                    <= start_idx
                    < display_end_idx
                ):
                    ax.axvline(
                        x=time[start_idx],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                    )

                if (
                    display_start_idx
                    < end_idx
                    < display_end_idx
                    and end_idx < n_samples
                ):
                    ax.axvline(
                        x=time[end_idx],
                        color="red",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                    )

            if col == 0:
                ax.set_ylabel(
                    str(fiber_labels[row - 1])
                )

            apply_fiber_ylim(
                ax=ax,
                window_traces=window_traces,
                displayed_traces=displayed_traces,
            )

    # ============================================================
    # Labels and layout
    # ============================================================
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")

    fig.suptitle(
        f"Highest Stimulation Amplitude: {max_amp} μA",
        y=0.995,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    plt.show()

    return fig, axes

def plot_highest_amp_traces_plotly(
    epoch,
    rec_channels=None,
    time=None,
    plot_v_lines=False,
    ylim=None,
    show_shoulders=True
):
    from plotly.subplots import make_subplots
    from pathlib import Path
    import plotly.io as pio

    """
    Plot only the first row in Plotly:
    mean trace for each selected channel + optional vertical lines.

    epoch.epoch_array shape: [amp, pulse_train, channel, signal]
    """

    # --- highest amplitude ---
    amp_label = epoch.parameters.parameters["pulse amplitude (μA)"].idxmax()
    max_amp = epoch.parameters.parameters.loc[amp_label, "pulse amplitude (μA)"]

    amp_idx = epoch.parameters_dictionary[amp_label]
    data = epoch.epoch_array[amp_idx]   # [pulse_train, channel, signal]

    if hasattr(data, "compute"):
        data = data.compute()

    n_pulses, n_channels_total, n_samples = data.shape

    # --- channel selection ---
    if rec_channels is None:
        plot_channels = np.arange(n_channels_total)
    else:
        plot_channels = np.asarray(rec_channels)

    n_plot_channels = len(plot_channels)
    channel_names = np.array(epoch.ephys.ch_names)[plot_channels]

    # --- time axis ---
    if time is None:
        time = np.arange(n_samples)
    else:
        time = np.asarray(time)
        if len(time) != n_samples:
            raise ValueError("time must have same length as the signal axis")

    # --- ylim handling ---
    if ylim is True:
        max_val = np.percentile(np.abs(data[:, plot_channels, :]), 95) * 1.25
        computed_ylim = (-max_val, max_val)
    elif ylim is None or ylim is False:
        computed_ylim = None
    else:
        computed_ylim = ylim

    # --- make subplots ---
    fig = make_subplots(
        rows=1,
        cols=n_plot_channels,
        subplot_titles=list(channel_names),
        shared_yaxes=True,
        shared_xaxes=True
    )

    for col, ch in enumerate(plot_channels, start=1):
        mean_trace = data[:, ch, :].mean(axis=0)

        fig.add_trace(
            go.Scatter(
                x=time,
                y=mean_trace,
                mode="lines",
                line=dict(width=2),
                name=str(channel_names[col - 1]),
                showlegend=False,
            ),
            row=1,
            col=col
        )

        if plot_v_lines:
            window_ends = epoch.neural_window_indices[ch, :, 1].astype(int)
            a_alpha_start = int(epoch.neural_window_indices[ch, 0, 0])
            vlines = np.concatenate(([a_alpha_start], window_ends))
            vlines = vlines[(vlines >= 0) & (vlines < len(time))]

            for v in vlines:
                fig.add_vline(
                    x=time[v],
                    line_width=1,
                    line_dash="dash",
                    line_color="red",
                    row=1,
                    col=col
                )

        if computed_ylim is not None:
            fig.update_yaxes(range=list(computed_ylim), row=1, col=col)

    fig.update_layout(
        title=f"Highest Stimulation Amplitude: {max_amp} μA",
        height=400,
        width=max(350 * n_plot_channels, 600),
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Signal", row=1, col=1)

    out = Path("plot_highest_amp_traces.html").resolve()
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
    pio.write_html(fig, file=str(out), auto_open=True, config={"responsive": True})

def a_alpha_range_max_right_bound(data, margin=.1):
    max_val = data.max()
    r_bound = data[-1]
    ymax = max_val + (max_val - r_bound) * margin
    ymin = r_bound - (max_val - r_bound) * margin
    return (ymin, ymax)


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
    Plot traces at a requested stimulation amplitude using Plotly.

    If the requested amplitude does not exist, the closest available
    amplitude is selected.

    Expected array shapes
    ---------------------
    epoch.epoch_array:
        [stimulation_amplitude, pulse_train, channel, signal]

    epoch.neural_window_indices:
        [channel, fiber_window, start/end]

    Parameters
    ----------
    epoch
        Epoch object containing data, parameters, channel names, and
        neural-window information.

    stim_amplitude : float
        Requested stimulation amplitude in microamps.

    rec_channels : int, sequence of int, or None
        Recording channels to plot.

        If None, all channels represented in neural_window_indices
        are plotted.

    time : array-like or None
        Time axis corresponding to the signal dimension.

        If None, sample indices are used.

    plot_v_lines : bool
        If True, show the true neural-window boundaries.

    plot_idv_fibers : bool, int, sequence of int, or None
        False or None:
            Show only the full traces.
        True:
            Show all neural/fiber windows.
        int:
            Show one fiber window.
        sequence of int:
            Show selected fiber windows.

    show_shoulders : bool or float
        Additional data shown on each side of the neural window.

        False:
            No shoulders.
        True:
            Add 100% of the window length to each side.
        number:
            Percentage of the window length added to each side.

    ylim : bool, tuple, or None
        True:
            Calculate subplot-specific y-limits. Fiber limits are based
            primarily on the true neural window, with limited expansion
            for shoulder values.
        False or None:
            Let Plotly autoscale using all displayed data.
        (lower, upper):
            Apply fixed y-limits to all subplots.

    ylim_padding : float
        Fractional padding around automatically calculated y-limits.

    shoulder_ylim_factor : float
        Maximum amount by which shoulder values can expand the fiber
        subplot limits relative to the true-window limits.

        1.0:
            Ignore shoulders when calculating y-limits.
        2.0:
            Allow up to twice the true-window scale.
        3.0:
            Allow more shoulder expansion.

    show : bool
        If True, display the figure immediately.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure.

    selected_amplitude : float
        Actual amplitude selected.

    amp_label
        Parameter-table index label associated with the selected amplitude.
    """

    # ============================================================
    # Helper functions
    # ============================================================
    def finite_min_max(values):
        values = np.asarray(values)

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            return None

        return (
            float(finite_values.min()),
            float(finite_values.max()),
        )

    def padded_limits(values):
        result = finite_min_max(values)

        if result is None:
            return None

        lower, upper = result
        data_range = upper - lower

        if data_range == 0:
            data_range = max(abs(lower), 1.0)

        padding = data_range * ylim_padding

        return (
            lower - padding,
            upper + padding,
        )

    def limited_shoulder_limits(
        window_values,
        displayed_values,
    ):
        """
        Base the y-limits on the true neural window.

        Shoulder values can expand the limits only up to
        shoulder_ylim_factor times the true-window scale.
        """
        window_limits = padded_limits(window_values)

        if window_limits is None:
            return padded_limits(displayed_values)

        main_lower, main_upper = window_limits

        displayed_extrema = finite_min_max(displayed_values)

        if displayed_extrema is None:
            return window_limits

        displayed_lower, displayed_upper = displayed_extrema

        main_center = (
            main_lower + main_upper
        ) / 2

        main_half_range = (
            main_upper - main_lower
        ) / 2

        if main_half_range <= 0:
            main_half_range = 1.0

        allowed_lower = (
            main_center
            - main_half_range * shoulder_ylim_factor
        )

        allowed_upper = (
            main_center
            + main_half_range * shoulder_ylim_factor
        )

        shoulder_padding = (
            main_upper - main_lower
        ) * ylim_padding

        requested_lower = min(
            main_lower,
            displayed_lower - shoulder_padding,
        )

        requested_upper = max(
            main_upper,
            displayed_upper + shoulder_padding,
        )

        final_lower = max(
            requested_lower,
            allowed_lower,
        )

        final_upper = min(
            requested_upper,
            allowed_upper,
        )

        # The true neural window must always remain visible.
        final_lower = min(
            final_lower,
            main_lower,
        )

        final_upper = max(
            final_upper,
            main_upper,
        )

        return final_lower, final_upper

    def add_window_line(
        figure,
        x_value,
        row,
        col,
    ):
        figure.add_vline(
            x=x_value,
            line_width=1,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            row=row,
            col=col,
        )

    # ============================================================
    # Validate requested amplitude
    # ============================================================
    if not isinstance(stim_amplitude, Real):
        raise TypeError(
            "stim_amplitude must be numeric"
        )

    requested_amplitude = float(stim_amplitude)

    if not np.isfinite(requested_amplitude):
        raise ValueError(
            "stim_amplitude must be finite"
        )

    # ============================================================
    # Validate shoulder settings
    # ============================================================
    if isinstance(show_shoulders, (bool, np.bool_)):
        shoulder_percent = (
            100.0 if show_shoulders else 0.0
        )

    elif isinstance(show_shoulders, Real):
        shoulder_percent = float(show_shoulders)

        if (
            not np.isfinite(shoulder_percent)
            or shoulder_percent < 0
        ):
            raise ValueError(
                "show_shoulders must be a finite, "
                "nonnegative percentage"
            )

    else:
        raise TypeError(
            "show_shoulders must be False, True, "
            "or a nonnegative percentage"
        )

    # ============================================================
    # Validate y-limit settings
    # ============================================================
    if not isinstance(ylim_padding, Real):
        raise TypeError(
            "ylim_padding must be numeric"
        )

    ylim_padding = float(ylim_padding)

    if (
        not np.isfinite(ylim_padding)
        or ylim_padding < 0
    ):
        raise ValueError(
            "ylim_padding must be finite and nonnegative"
        )

    if not isinstance(shoulder_ylim_factor, Real):
        raise TypeError(
            "shoulder_ylim_factor must be numeric"
        )

    shoulder_ylim_factor = float(
        shoulder_ylim_factor
    )

    if (
        not np.isfinite(shoulder_ylim_factor)
        or shoulder_ylim_factor < 1
    ):
        raise ValueError(
            "shoulder_ylim_factor must be finite "
            "and greater than or equal to 1"
        )

    if ylim is True or ylim is False or ylim is None:
        fixed_ylim = None

    else:
        if len(ylim) != 2:
            raise ValueError(
                "A fixed ylim must contain two values"
            )

        fixed_ylim = [
            float(ylim[0]),
            float(ylim[1]),
        ]

    # ============================================================
    # Find the closest available stimulation amplitude
    # ============================================================
    par_df = epoch.parameters.parameters
    amp_col = _find_amp_col(par_df)

    amplitude_values = np.asarray(
        par_df[amp_col],
        dtype=float,
    )

    valid_positions = np.flatnonzero(
        np.isfinite(amplitude_values)
    )

    if valid_positions.size == 0:
        raise ValueError(
            f"No valid amplitudes were found in {amp_col!r}"
        )

    amplitude_distances = np.abs(
        amplitude_values[valid_positions]
        - requested_amplitude
    )

    nearest_position = valid_positions[
        np.argmin(amplitude_distances)
    ]

    selected_amplitude = float(
        amplitude_values[nearest_position]
    )

    amp_label = par_df.index[nearest_position]
    amp_idx = epoch.parameters_dictionary[amp_label]

    exact_match = np.isclose(
        selected_amplitude,
        requested_amplitude,
        rtol=0,
        atol=1e-9,
    )

    if not exact_match:
        warnings.warn(
            f"Requested amplitude {requested_amplitude:g} μA "
            f"was not found. Plotting the closest available "
            f"amplitude: {selected_amplitude:g} μA.",
            UserWarning,
            stacklevel=2,
        )

    # ============================================================
    # Neural-window information
    # ============================================================
    window_indices = epoch.neural_window_indices

    if hasattr(window_indices, "compute"):
        window_indices = window_indices.compute()

    window_indices = np.asarray(
        window_indices
    )

    if (
        window_indices.ndim != 3
        or window_indices.shape[2] != 2
    ):
        raise ValueError(
            "neural_window_indices must have shape "
            "[channel, fiber_window, 2]"
        )

    n_window_channels = window_indices.shape[0]
    n_available_fibers = window_indices.shape[1]

    # ============================================================
    # Channel selection
    # ============================================================
    if rec_channels is None:
        plot_channels = np.arange(
            n_window_channels,
            dtype=int,
        )

    elif isinstance(rec_channels, (int, np.integer)):
        plot_channels = np.array(
            [int(rec_channels)],
            dtype=int,
        )

    else:
        plot_channels = np.asarray(
            rec_channels,
            dtype=int,
        )

        if plot_channels.ndim != 1:
            raise ValueError(
                "rec_channels must be an integer or "
                "a one-dimensional sequence"
            )

    if plot_channels.size == 0:
        raise ValueError(
            "No recording channels were selected"
        )

    if np.any(plot_channels < 0):
        raise IndexError(
            "Channel indices cannot be negative"
        )

    if np.any(
        plot_channels >= n_window_channels
    ):
        raise IndexError(
            "Every selected channel must have an entry "
            "in neural_window_indices"
        )

    n_plot_channels = len(plot_channels)

    channel_names = np.asarray(
        epoch.ephys.ch_names
    )[plot_channels]

    # ============================================================
    # Load data at the selected amplitude
    # ============================================================
    data = epoch.epoch_array[
        amp_idx,
        :,
        :,
        :,
    ]

    data = data[
        :,
        plot_channels,
        :,
    ]

    if hasattr(data, "compute"):
        data = data.compute()

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            "Selected data must have shape "
            "[pulse_train, channel, signal]"
        )

    n_pulses, selected_channel_count, n_samples = (
        data.shape
    )

    if selected_channel_count != n_plot_channels:
        raise ValueError(
            "Selected data does not contain the expected "
            "number of channels"
        )

    # ============================================================
    # Time axis
    # ============================================================
    if time is None:
        time = np.arange(
            n_samples
        )

    else:
        time = np.asarray(time)

        if time.ndim != 1:
            raise ValueError(
                "time must be one-dimensional"
            )

        if len(time) != n_samples:
            raise ValueError(
                "time must have the same length as "
                "the signal axis"
            )

    # ============================================================
    # Fiber-window selection
    # ============================================================
    if (
        plot_idv_fibers is False
        or plot_idv_fibers is None
    ):
        plot_fibers = []

    elif plot_idv_fibers is True:
        plot_fibers = list(
            range(n_available_fibers)
        )

    elif isinstance(
        plot_idv_fibers,
        (int, np.integer),
    ):
        plot_fibers = [
            int(plot_idv_fibers)
        ]

    else:
        plot_fibers = [
            int(fiber_idx)
            for fiber_idx in plot_idv_fibers
        ]

    if any(
        fiber_idx < 0
        or fiber_idx >= n_available_fibers
        for fiber_idx in plot_fibers
    ):
        raise IndexError(
            "Fiber-window index out of range"
        )

    n_fibers = len(plot_fibers)

    fiber_names = np.asarray(
        epoch.neural_fiber_names
    )

    if n_fibers > 0:
        fiber_labels = fiber_names[
            plot_fibers
        ]
    else:
        fiber_labels = []

    # ============================================================
    # Create Plotly subplot grid
    # ============================================================
    n_rows = 1 + n_fibers

    subplot_titles = []

    for row_idx in range(n_rows):
        for channel_name in channel_names:
            if row_idx == 0:
                subplot_titles.append(
                    str(channel_name)
                )
            else:
                subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows,
        cols=n_plot_channels,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=subplot_titles,
        horizontal_spacing=(
            0.04 if n_plot_channels > 1 else 0.08
        ),
        vertical_spacing=min(
            0.08,
            0.25 / n_rows,
        ),
    )

    # ============================================================
    # Full-trace row
    # ============================================================
    for col, channel_idx in enumerate(
        plot_channels,
        start=1,
    ):
        local_channel_idx = col - 1

        traces = data[
            :,
            local_channel_idx,
            :,
        ]

        mean_trace = np.nanmean(
            traces,
            axis=0,
        )

        for pulse_idx in range(n_pulses):
            fig.add_trace(
                go.Scattergl(
                    x=time,
                    y=traces[pulse_idx],
                    mode="lines",
                    line={
                        "color": "rgba(120,120,120,0.25)",
                        "width": 0.7,
                    },
                    name="Individual trace",
                    legendgroup="individual",
                    showlegend=False,
                    customdata=np.full(
                        len(time),
                        pulse_idx,
                    ),
                    hovertemplate=(
                        f"Channel: {channel_names[local_channel_idx]}"
                        "<br>Pulse: %{customdata}"
                        "<br>Time: %{x}"
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
                line={
                    "color": "blue",
                    "width": 2.5,
                },
                name="Mean trace",
                legendgroup="mean",
                showlegend=(col == 1),
                hovertemplate=(
                    f"Channel: {channel_names[local_channel_idx]}"
                    "<br>Mean trace"
                    "<br>Time: %{x}"
                    "<br>Signal: %{y:.5g}"
                    f"<br>Amplitude: {selected_amplitude:g} μA"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )

        if plot_v_lines:
            channel_boundaries = window_indices[
                channel_idx
            ].reshape(-1)

            channel_boundaries = channel_boundaries[
                np.isfinite(channel_boundaries)
            ]

            channel_boundaries = np.unique(
                np.rint(
                    channel_boundaries
                ).astype(int)
            )

            channel_boundaries = channel_boundaries[
                (channel_boundaries >= 0)
                & (channel_boundaries < n_samples)
            ]

            for boundary_idx in channel_boundaries:
                add_window_line(
                    figure=fig,
                    x_value=time[boundary_idx],
                    row=1,
                    col=col,
                )

        if ylim is True:
            full_limits = padded_limits(
                traces
            )

            if full_limits is not None:
                fig.update_yaxes(
                    range=list(full_limits),
                    row=1,
                    col=col,
                )

        elif fixed_ylim is not None:
            fig.update_yaxes(
                range=fixed_ylim,
                row=1,
                col=col,
            )

        if col == 1:
            fig.update_yaxes(
                title_text="Full trace",
                row=1,
                col=col,
            )

    # ============================================================
    # Fiber-window rows
    # ============================================================
    for fiber_row, fiber_idx in enumerate(
        plot_fibers,
        start=2,
    ):
        fiber_label = str(
            fiber_labels[fiber_row - 2]
        )

        for col, channel_idx in enumerate(
            plot_channels,
            start=1,
        ):
            local_channel_idx = col - 1

            bounds = window_indices[
                channel_idx,
                fiber_idx,
            ]

            if not np.all(
                np.isfinite(bounds)
            ):
                continue

            raw_start_idx, raw_end_idx = (
                np.rint(bounds).astype(int)
            )

            window_length = (
                raw_end_idx - raw_start_idx
            )

            if window_length <= 0:
                continue

            shoulder_samples = int(
                round(
                    window_length
                    * shoulder_percent
                    / 100.0
                )
            )

            start_idx = max(
                0,
                raw_start_idx,
            )

            end_idx = min(
                n_samples,
                raw_end_idx,
            )

            display_start_idx = max(
                0,
                raw_start_idx - shoulder_samples,
            )

            display_end_idx = min(
                n_samples,
                raw_end_idx + shoulder_samples,
            )

            if (
                end_idx <= start_idx
                or display_end_idx <= display_start_idx
            ):
                continue

            traces = data[
                :,
                local_channel_idx,
                :,
            ]

            window_traces = traces[
                :,
                start_idx:end_idx,
            ]

            displayed_traces = traces[
                :,
                display_start_idx:display_end_idx,
            ]

            displayed_time = time[
                display_start_idx:display_end_idx
            ]

            displayed_mean = np.nanmean(
                displayed_traces,
                axis=0,
            )

            for pulse_idx in range(n_pulses):
                fig.add_trace(
                    go.Scattergl(
                        x=displayed_time,
                        y=displayed_traces[pulse_idx],
                        mode="lines",
                        line={
                            "color": "rgba(120,120,120,0.25)",
                            "width": 0.7,
                        },
                        name="Individual trace",
                        legendgroup="individual",
                        showlegend=False,
                        customdata=np.full(
                            len(displayed_time),
                            pulse_idx,
                        ),
                        hovertemplate=(
                            f"Channel: {channel_names[local_channel_idx]}"
                            f"<br>Window: {fiber_label}"
                            "<br>Pulse: %{customdata}"
                            "<br>Time: %{x}"
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
                    line={
                        "color": "blue",
                        "width": 2.5,
                    },
                    name="Mean trace",
                    legendgroup="mean",
                    showlegend=False,
                    hovertemplate=(
                        f"Channel: {channel_names[local_channel_idx]}"
                        f"<br>Window: {fiber_label}"
                        "<br>Mean trace"
                        "<br>Time: %{x}"
                        "<br>Signal: %{y:.5g}"
                        f"<br>Amplitude: {selected_amplitude:g} μA"
                        "<extra></extra>"
                    ),
                ),
                row=fiber_row,
                col=col,
            )

            if plot_v_lines:
                if (
                    display_start_idx
                    <= start_idx
                    < display_end_idx
                ):
                    add_window_line(
                        figure=fig,
                        x_value=time[start_idx],
                        row=fiber_row,
                        col=col,
                    )

                if (
                    display_start_idx
                    < end_idx
                    < display_end_idx
                    and end_idx < n_samples
                ):
                    add_window_line(
                        figure=fig,
                        x_value=time[end_idx],
                        row=fiber_row,
                        col=col,
                    )

            if ylim is True:
                fiber_limits = limited_shoulder_limits(
                    window_values=window_traces,
                    displayed_values=displayed_traces,
                )

                if fiber_limits is not None:
                    fig.update_yaxes(
                        range=list(fiber_limits),
                        row=fiber_row,
                        col=col,
                    )

            elif fixed_ylim is not None:
                fig.update_yaxes(
                    range=fixed_ylim,
                    row=fiber_row,
                    col=col,
                )

            if col == 1:
                fig.update_yaxes(
                    title_text=fiber_label,
                    row=fiber_row,
                    col=col,
                )

    # ============================================================
    # Axis labels and figure layout
    # ============================================================
    for col in range(
        1,
        n_plot_channels + 1,
    ):
        fig.update_xaxes(
            title_text="Time",
            row=n_rows,
            col=col,
        )

    if exact_match:
        title = (
            f"Stimulation amplitude: "
            f"{selected_amplitude:g} μA"
        )
    else:
        title = (
            f"Requested {requested_amplitude:g} μA — "
            f"showing closest available amplitude: "
            f"{selected_amplitude:g} μA"
        )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        hovermode="closest",
        height=max(
            500,
            280 * n_rows,
        ),
        width=max(
            750,
            400 * n_plot_channels,
        ),
        margin={
            "l": 80,
            "r": 30,
            "t": 100,
            "b": 70,
        },
    )

    fig.write_html(
        r"D:\ImThera\data_processed\test.html",
        include_plotlyjs=True,
        full_html=True,
        auto_open=True,
    )

    if show:
        fig.show()

    return fig, selected_amplitude, amp_label