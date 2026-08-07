from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import sys
import warnings

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks, medfilt, savgol_filter

from .base.epoch_data import _EpochData
from .base.ts_data import _TsData
from .base.utils.numeric import _to_numeric_array
from .utilities.ancillary_functions import check_make_dir


ECAP_VERSION = "2026-08-06-compute-mean-traces-v5-channel-types"


def _normalize_channel_type(value: object) -> str:
    """Return a normalized recording-channel type label."""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return str(value).strip().upper()


def _resolve_recording_channel_types(
    ephys_data: _TsData,
    n_recording_channels: int,
) -> list[str]:
    """Resolve exactly one channel-type label per recording channel.

    The legacy ``_TsData.types`` property contains only unique type names and
    therefore cannot be used to map types back to channel indices. Prefer the
    established ``_ch_num_mask_by_type`` mapping, then fall back to the
    per-channel ``ch_types`` property and finally the underlying metadata.
    """
    diagnostics: list[str] = []

    # This is the same channel mapping used by the original ECAP constructor.
    try:
        masks_by_type = ephys_data._ch_num_mask_by_type
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        diagnostics.append(f"_ch_num_mask_by_type failed: {exc}")
    else:
        if isinstance(masks_by_type, Mapping):
            resolved = np.full(n_recording_channels, "", dtype=object)
            masks_valid = True

            for channel_type, mask in masks_by_type.items():
                mask_array = np.asarray(mask, dtype=bool).reshape(-1)
                if mask_array.size != n_recording_channels:
                    diagnostics.append(
                        f"mask for {channel_type!r} contained "
                        f"{mask_array.size} entries"
                    )
                    masks_valid = False
                    break
                resolved[mask_array] = _normalize_channel_type(channel_type)

            if masks_valid and np.all(resolved != ""):
                return resolved.tolist()

            if masks_valid:
                missing = np.flatnonzero(resolved == "").tolist()
                diagnostics.append(
                    f"channel masks did not classify channel indices {missing}"
                )
        else:
            diagnostics.append("_ch_num_mask_by_type was not a mapping")

    # Normal modern path: one label for every recording channel.
    for source_name, source in (
        ("ephys_data", ephys_data),
        ("ephys_data.ts_data", getattr(ephys_data, "ts_data", None)),
    ):
        if source is None:
            continue
        try:
            values = list(source.ch_types)
        except (AttributeError, TypeError, ValueError) as exc:
            diagnostics.append(f"{source_name}.ch_types failed: {exc}")
            continue

        if len(values) == n_recording_channels:
            return [_normalize_channel_type(value) for value in values]

        diagnostics.append(
            f"{source_name}.ch_types contained {len(values)} entries"
        )

    # Last-resort metadata path for compatible _TsData objects.
    metadata = getattr(ephys_data, "metadata", None)
    if isinstance(metadata, Mapping):
        metadata_items = [metadata]
    elif isinstance(metadata, (list, tuple)):
        metadata_items = list(metadata)
    else:
        metadata_items = []

    for index, item in enumerate(metadata_items):
        if not isinstance(item, Mapping):
            continue
        for key in ("types", "ch_types"):
            if key not in item:
                continue
            values = list(item[key])
            if len(values) == n_recording_channels:
                return [_normalize_channel_type(value) for value in values]
            diagnostics.append(
                f"metadata[{index}][{key!r}] contained {len(values)} entries"
            )

    details = "; ".join(diagnostics) if diagnostics else "no type metadata found"
    raise ValueError(
        "Could not determine one recording-channel type per channel. "
        f"Expected {n_recording_channels} labels. {details}. "
        "Confirm that ephys_data.ch_types returns one value per channel."
    )


class ECAP(_EpochData):
    """ECAP-specific analysis built on the generic :class:`_EpochData` API.

    Generic epoch selection and pulse reductions are inherited from
    ``_EpochData``. In particular:

    - ``epoch(parameter, ...)`` returns ``(pulses, channels, samples)``.
    - ``compute_mean_traces(parameter)`` returns a computed NumPy array
      shaped ``(channels, samples)``.
    - ``compute_mean_traces()`` computes all parameter means, stores them in
      ``mean_traces``, and returns ``(parameters, channels, samples)``.
    """

    def __init__(
        self,
        ephys_data: _TsData,
        stim_data,
        distance_log=None,
        preload: bool = False,
        *,
        epoch_window: tuple[float, float] | str | None = "auto",
        epoch_cluster_gap: int = 500_000,
    ) -> None:
        super().__init__(
            ephys_data,
            stim_data,
            stim_data,
            epoch_window=epoch_window,
            epoch_cluster_gap=epoch_cluster_gap,
        )

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = float(ephys_data.sample_rate)
        self.log_path = distance_log

        self.neural_fiber_names = [
            "A-alpha",
            "A-beta",
            "A-gamma",
            "A-delta",
            "B",
        ]
        self.emg_window = ["Total EMG"]
        self.fiber_windows = None
        self.master_df = pd.DataFrame()

        n_recording_channels = int(ephys_data.shape[0])
        channel_types = _resolve_recording_channel_types(
            ephys_data,
            n_recording_channels,
        )
        self.recording_channel_types = tuple(channel_types)

        normalized_types = np.asarray(channel_types, dtype=str)
        self.neural_channels = np.flatnonzero(normalized_types == "ENG")
        self.emg_channels = np.flatnonzero(normalized_types == "EMG")

        if self.neural_channels.size == 0:
            raise ValueError(
                "No ENG recording channels were identified. "
                f"Resolved channel types were {channel_types!r}. "
                "Assign per-channel types with ephys_data.set_ch_types(...) "
                "before constructing ECAP."
            )

        distances = (
            np.asarray([0.0], dtype=float)
            if distance_log is None
            else np.asarray(_to_numeric_array(distance_log), dtype=float).reshape(-1)
        )
        if distances.size == 1 and self.neural_channels.size > 1:
            distances = np.repeat(distances, self.neural_channels.size)
        if distances.size != self.neural_channels.size:
            raise ValueError(
                "distance_log must contain one distance or one distance per "
                f"neural recording channel ({self.neural_channels.size})."
            )
        self.distance_log = distances

        self.neural_window_indices = self.calculate_neural_window_lengths()
        if self.emg_channels.size:
            self.emg_window_indices = np.asarray(
                self.calculate_emg_window_lengths(),
                dtype=int,
            )
            # Temporary spelling compatibility for existing plotting code.
            self.EMG_window_indicies = self.emg_window_indices

        if preload:
            self.compute_mean_traces()

    def preload_epoch_graphs(self, parameters=None) -> ECAP:
        """Construct and cache each selected parameter's lazy epoch graph.

        This does not compute the numerical arrays or create a second
        pulse-mean cache. ``dask_array()`` remains the graph cache.
        """
        for parameter_key in self._normalize_parameter_keys(parameters):
            self.dask_array(parameter_key)
        return self

    def _default_windows_for_channel(
        self,
        channel: int,
        parameter: object,
    ) -> dict[str, tuple[float, float]]:
        """Return ECAP or EMG measurement windows for one recording channel."""
        channel = int(channel)
        sample_rate = float(self.ts_data.sample_rate)

        if channel in self.neural_channels:
            local_index = int(np.flatnonzero(self.neural_channels == channel)[0])
            windows = np.asarray(self.neural_window_indices[local_index])
            return {
                name: tuple(np.asarray(window, dtype=float) / sample_rate)
                for name, window in zip(self.neural_fiber_names, windows)
            }

        if channel in self.emg_channels:
            local_index = int(np.flatnonzero(self.emg_channels == channel)[0])
            windows = np.asarray(
                self.calculate_emg_window_lengths(parameter=parameter)[local_index]
            )
            return {
                name: tuple(np.asarray(window, dtype=float) / sample_rate)
                for name, window in zip(self.emg_window, windows)
            }

        raise ValueError(
            f"No default ECAP/EMG window is defined for channel {channel} "
            f"({self.ephys.ch_names[channel]})."
        )

    def _windows_dict_for_channel(
        self,
        neural_windows: np.ndarray,
        channel_index: int,
    ) -> dict[str, tuple[float, float]]:
        """Convert one neural channel's window array to seconds."""
        windows = np.asarray(neural_windows)[int(channel_index)]
        if windows.ndim != 2 or windows.shape[1] != 2:
            raise ValueError(
                "neural_windows must have shape (channels, windows, 2)."
            )
        if len(self.neural_fiber_names) != windows.shape[0]:
            raise ValueError(
                "The number of neural windows does not match "
                "neural_fiber_names."
            )

        if np.issubdtype(windows.dtype, np.integer):
            windows = windows.astype(float) / float(self.ts_data.sample_rate)
        else:
            windows = windows.astype(float)

        return {
            name: (float(window[0]), float(window[1]))
            for name, window in zip(self.neural_fiber_names, windows)
        }

    def _calculate_AUC_single_param(
        self,
        parameter,
        windows,
        channels=None,
        method="RMS",
        baseline=None,
        absolute=False,
        artifact_subtraction=False,
    ):
        """Calculate one windowed measurement for one stimulation parameter."""
        if artifact_subtraction:
            raise NotImplementedError(
                "artifact_subtraction is not yet implemented for windowed AUC."
            )

        output = {}
        method = str(method).upper()
        for name, window in windows.items():
            if method == "RMS":
                output[name] = self.rms_in_window(
                    parameter,
                    channels=channels,
                    window_s=window,
                    baseline_s=baseline,
                )
            elif method == "AUC":
                output[name] = self.auc_in_window(
                    parameter,
                    channels=channels,
                    window_s=window,
                    baseline_s=baseline,
                    absolute=absolute,
                )
            else:
                raise ValueError("method must be 'RMS' or 'AUC'.")
        return output

    def gather_num_conditions(self):
        all_indices = self.stim.parameters.index
        if len(all_indices[0]) == 1:
            return 1
        elif len(all_indices[0]) == 2:
            count = 1
            previous_index = all_indices[0]
            for i in all_indices[1:]:
                if i[0] != previous_index[0]:
                    count += 1
                previous_index = i
            return count
        else:
            sys.exit("Improper dimensions.")

    def gather_num_amplitudes(self):
        max_num_amps = 0
        all_indices = self.stim.parameters.index
        if len(all_indices[0]) == 1:
            return len(all_indices)
        elif len(all_indices[0]) == 2:
            for i in all_indices:
                if i[1] + 1 > max_num_amps:
                    max_num_amps = i[1] + 1
            return max_num_amps
        else:
            sys.exit("Improper Dimensions.")

    def calculate_neural_window_lengths(self):
        """
        :return: time_windows[recording_electrode][fiber_type][start/stop]
        """
        # min and max conduction velocities of various fiber types
        a_alpha = [120,
                   60]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_beta = [60,
                  30]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_gamma = [30,
                   15]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_delta = [30,
                   5]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        B = [15, 3]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en

        velocity_matrix = np.array([a_alpha, a_beta, a_gamma, a_delta, B])
        # create time_windows based on fiber type activation for every recording
        # time_window[rec_electrode][fiber_type][start/stop]
        num_neural_channels = len(self.neural_channels)
        num_fiber_types = len(self.neural_fiber_names)

        time_windows = np.zeros((num_neural_channels, num_fiber_types, 2))
        for i, vel in enumerate(velocity_matrix):
            for j, length_cm in enumerate(self.distance_log):
                time_windows[j][i] = ([length_cm / 100 * (1 / ii) for ii in vel])

        time_windows = np.array(np.round(time_windows * self.fs).astype(int))

        return time_windows

    def calculate_neural_window_times(self):
        """
        :return: time_windows[recording_electrode][fiber_type][start/stop]
        """
        # min and max conduction velocities of various fiber types
        a_alpha = [120, 70]
        a_beta = [70, 30]
        a_gamma = [30, 15]
        a_delta = [30, 5]
        B = [15, 3]

        velocity_matrix = np.array([a_alpha, a_beta, a_gamma, a_delta, B])
        num_neural_channels = len(self.neural_channels)
        num_fiber_types = len(self.neural_fiber_names)

        time_windows = np.zeros((num_neural_channels, num_fiber_types, 2))
        for i, vel in enumerate(velocity_matrix):
            for j, length_cm in enumerate(self.distance_log):
                time_windows[j][i] = ([length_cm / 100 * (1 / ii) for ii in vel])

        return time_windows

    def calculate_emg_window_lengths(self, parameter=None):
        """Return parameter-aware EMG window indices.

        The window begins six pulse widths after time zero and ends at the
        shorter of one stimulation period, the next pulse, or the epoch end.
        """
        if parameter is None:
            parameter = self._normalize_parameter_keys(None)[0]
        parameter = self._normalize_parameter_key(parameter)

        parameter_row = self.parameters.parameters.loc[parameter]
        pulse_width_s = float(parameter_row["pulse duration (ms)"]) / 1_000.0
        frequency_hz = float(parameter_row["frequency (Hz)"])
        if frequency_hz <= 0:
            raise ValueError("frequency (Hz) must be positive.")

        time = self.time_axis(parameter)
        if time.size == 0:
            raise ValueError(f"Parameter {parameter!r} has an empty epoch.")

        start_s = 6.0 * pulse_width_s
        stop_s = 1.0 / frequency_hz

        event_times = self._event_times(parameter)
        if event_times.size > 1:
            positive_intervals = np.diff(event_times)
            positive_intervals = positive_intervals[positive_intervals > 0]
            if positive_intervals.size:
                stop_s = min(stop_s, float(positive_intervals.min()))

        onset = int(np.searchsorted(time, start_s, side="left"))
        offset = int(np.searchsorted(time, stop_s, side="left"))
        onset = min(max(onset, 0), max(time.size - 1, 0))
        offset = min(max(offset, onset + 1), time.size)

        return np.repeat(
            np.asarray([[[onset, offset]]], dtype=int),
            self.emg_channels.size,
            axis=0,
        )

    def calc_AUC_method(self, signal, recording_idx, window_type, calculation_type, metadata, plot_AUCs=False,
                        save_path=None):
        if calculation_type == "RMS":
            if window_type.endswith("EMG"):
                window_onset_idx = self.EMG_window_indicies
            else: # recordings are neural or undefined
                window_onset_idx = self.neural_window_indices

            current_list = []
            for specific_window_idx, specific_window in enumerate(window_onset_idx[recording_idx]):
                specific_onset = specific_window[0]
                specific_offset = specific_window[1]
                current_list.append(np.sqrt(np.mean(signal[specific_onset:specific_offset] ** 2)))

                if plot_AUCs:
                    fig, ax = plt.subplots(1, dpi=300)
                    ax.plot(signal)
                    ax.vlines([specific_onset, specific_offset], np.max(signal[specific_onset:specific_offset]),
                              np.min(signal[specific_onset:specific_offset]))
                    plt.show()
            return current_list

        elif calculation_type == "Peaks":

            def find_max(signal, start, stop, jitter_percentage=0.01):
                """
                Returns the index of the maximum peak in signal[start:stop].
                Returns None if no peaks found
                """
                n = len(signal)
                stop = min(stop, n)

                # Ensure non-degenerate window
                if stop <= start:
                    return None

                # Estimate minimum distance between peaks
                peak_distance = max(int((stop - start) / 10), 1)

                # Find candidate peaks in the window
                rel_peaks, _ = find_peaks(signal[start:stop], distance=peak_distance)
                if rel_peaks.size == 0:
                    return jitter_percentage

                # Map relative → absolute indices
                peaks = rel_peaks + start

                # Apply jitter exclusion near window start (optional)
                jitter_start = start + jitter_percentage * (stop - start)
                peaks = peaks[peaks > jitter_start]
                if peaks.size == 0:
                    return None

                # Choose the highest-valued peak
                max_idx = peaks[np.argmax(signal[peaks])]
                return int(max_idx)

            def find_minima(signal, start, stop, max_idx, overlap=True):
                try:
                    smoothed_curve = savgol_filter(signal, 5, 1)
                except:
                    warnings.warn("Curve couldn't be smoothed")
                    smoothed_curve = signal

                max_min = np.diff(np.sign(np.diff(smoothed_curve)))
                # max_min loses a data point for every derivative it takes. Therefore, it is necessary to pad the end.
                max_min = np.append(max_min, [0, 0])
                # go from max forward in time until you reach boundary, or you find a min (max_min = 2)

                # There is a case for A delta fibers that due to their conduction velocities, their onset occurs after the sampling window.
                # This if statement is meant to address this case. It will return the last points in the data set
                if len(signal) - max_idx <= 2:
                    min1 = len(signal) - 3
                    min2 = len(signal) - 1
                    return min1, min2

                min1 = []
                min2 = []

                for ii in range(max_idx + 1, len(max_min)):
                    if not overlap and ii >= stop:
                        min2 = stop
                        break
                    elif max_min[ii] == 2 and signal[ii] < signal[max_idx]:
                        min2 = ii + 1
                        break
                    elif ii == len(signal) - 1 or ii - stop > .25 * (stop - start):
                        if stop > len(signal):
                            min2 = len(signal) - 1
                        else:
                            min2 = stop
                        break

                if not min2:
                    if stop < len(max_min):
                        min2 = stop
                    else:
                        min2 = len(max_min) - 1

                for ii in range(max_idx - 1, 0, -1):
                    if not overlap and ii <= start:
                        min1 = start
                        break
                    elif max_min[ii] == 2 and signal[ii] < signal[max_idx]:
                        # adding +1 to make up for the filter removing one data point
                        min1 = ii + 1
                        break
                    elif ii == 0 or start - ii > .25 * (stop - start):
                        min1 = start
                        break

                if not min1:
                    min1 = start

                while signal[min1] > signal[max_idx]:
                    max_min = np.diff(np.sign(np.diff(signal)))
                    # max_min loses a data point for every derivative it takes. Therefore, it is necessary to pad the end.
                    max_min = np.append(max_min, [0, 0])
                    # go from max forward in time until you reach boundary, or you find a min (max_min = 2)

                    # There is a case for A delta fibers that due to their conduction velocities, their onset occurs after the sampling window.
                    # This if statement is meant to address this case. It will return the last points in the data set

                    min1 = []
                    min2 = []

                    for ii in range(max_idx - 1, 0, -1):
                        if not overlap and ii <= start:
                            min1 = start
                            break
                        elif max_min[ii] == 2 and signal[ii] < signal[max_idx]:
                            # adding +1 to make up for the filter removing one data point
                            min1 = ii + 1
                            break
                        elif ii == 0 or start - ii > .25 * (stop - start):
                            min1 = start
                            break

                    # worst case: subtract one datapoint from max_idx
                    min1 = max_idx - 1
                    break

                if not min2:
                    min2 = stop

                return [min1, min2]

            def determine_plotting_boundaries(signal, fiber_minima, fiber_maxima):
                min_y = signal[fiber_minima[0][0]]
                max_y = signal[fiber_maxima[0]]
                for i in fiber_maxima:
                    if signal[i] > max_y:
                        max_y = signal[i]

                for i in fiber_minima:
                    for j in i:
                        if signal[j] < min_y:
                            min_y = signal[j]

                my_range = max_y - min_y

                return min_y - my_range / 2, max_y + my_range / 2

            def relevant_AUC(self, signal, recording_idx, fs, *,
                             baseline_mode="linear",  # "linear" | "mean" | "median" | "none"
                             polarity="auto",  # "auto" | "positive" | "negative" | "absolute"
                             clip_negative=False):
                """
                Compute lobe AUCs per fiber window for one channel's averaged ECAP.

                signal        : 1D array of samples (this should be the averaged waveform for a channel)
                recording_idx : which recording's windows to use (index into self.neural_window_indices)
                fs            : sampling rate (Hz)
                Returns: np.ndarray of AUCs (µV·s if signal in µV)
                """
                dt = 1.0 / float(fs)
                aucs = []

                # 1) windows: list of (i0, i1) sample indices per fiber
                try:
                    windows = list(self.neural_window_indices[recording_idx])
                except Exception:
                    raise ValueError(
                        "neural_window_indices missing or invalid for recording_idx={}".format(recording_idx))

                # 2) optional polarity detection (for 'auto')
                sign = 1.0
                if polarity == "auto":
                    # take the largest magnitude lobe across all windows
                    mags = []
                    for (i0, i1) in windows:
                        seg = signal[i0:i1]
                        if seg.size:
                            mags.append(seg.max() if abs(seg.max()) >= abs(seg.min()) else seg.min())
                    if mags:
                        sign = 1.0 if abs(max(mags)) >= abs(min(mags)) else -1.0
                    else:
                        sign = 1.0
                elif polarity == "positive":
                    sign = 1.0
                elif polarity == "negative":
                    sign = -1.0
                elif polarity == "absolute":
                    sign = None  # handled below
                else:
                    raise ValueError("polarity must be 'auto'|'positive'|'negative'|'absolute'")

                for (i0, i1) in windows:
                    i0 = int(i0)
                    i1 = int(i1)
                    if i1 - i0 < 2:
                        aucs.append(0.0)
                        continue

                    seg = signal[i0:i1].astype(float)

                    # 3) build baseline
                    if baseline_mode == "linear":
                        base = np.linspace(seg[0], seg[-1], num=seg.size)
                    elif baseline_mode == "mean":
                        base = np.full(seg.size, seg.mean())
                    elif baseline_mode == "median":
                        base = np.full(seg.size, np.median(seg))
                    elif baseline_mode == "none":
                        base = 0.0
                    else:
                        raise ValueError("baseline_mode must be 'linear'|'mean'|'median'|'none'")

                    y = seg - base if isinstance(base, np.ndarray) else seg - float(base)

                    # 4) apply polarity
                    if sign is None:  # absolute
                        y = np.abs(y)
                    else:
                        y = sign * y

                    # 5) integrate with correct units
                    auc = np.trapz(y, dx=dt)

                    if clip_negative and auc < 0:
                        auc = 0.0

                    aucs.append(float(auc))

                return np.asarray(aucs, dtype=float)




            current_list = relevant_AUC(signal, recording_idx, plot_AUCs, save_path)
            return current_list

    def calculate_AUC(self, parameter=None, windows=None, channels=None, method="RMS", baseline=None, absolute=False, artifact_subtraction=False, window_names=None):
        """
        Calculate AUC/RMS values across stimulation parameters and recording channels.

        Default behavior:
            - Uses all ephys channels.
            - Neural channels use self.neural_window_indices.
            - EMG channels use self.EMG_window_indicies.

        Parameters
        ----------
        parameter : tuple, list, or None
            Parameter index or list of parameter indices. If None, all parameters are used.

        windows : None, dict, or ndarray
            None:
                Use default channel-aware windows.
            dict:
                Same window dictionary applied to all selected channels.
            ndarray:
                Shape (n_channels, n_windows, 2). Values can be sample indices or seconds.

        channels : None, int, str, list, or boolean mask
            Recording channels to calculate. If None, all ephys channels are used.

        method : str
            "RMS" or "AUC".

        baseline : tuple or None
            Baseline window in seconds.

        absolute : bool
            Whether to integrate absolute value for AUC.

        artifact_subtraction : bool
            Passed to _calculate_AUC_single_param.

        window_names : list or None
            Optional names for ndarray windows.
        """

        # -----------------------------
        # Resolve parameters
        # -----------------------------
        if parameter is None:
            parameter = list(self.parameters.parameters.index)

        # Check for legacy condition of only 1 parameter
        elif isinstance(parameter, tuple):
            parameter = [parameter]

        else:
            parameter = list(parameter)

        # -----------------------------
        # Resolve channels
        # Default: all channels, not just neural
        # -----------------------------
        if channels is None:
            channels = np.arange(self.ephys.shape[0], dtype=int)

        else:
            ch_idx = self.ts_data._ch_to_index(channels)

            if isinstance(ch_idx, slice):
                channels = np.arange(self.ephys.shape[0], dtype=int)[ch_idx]

            else:
                ch_idx = np.asarray(ch_idx)

                if ch_idx.dtype == bool:
                    channels = np.flatnonzero(ch_idx)
                else:
                    channels = ch_idx.astype(int)

        channels = np.asarray(channels, dtype=int)

        # -----------------------------
        # Helper: convert windows to seconds
        # -----------------------------
        def _windows_to_seconds(wins):
            wins = np.asarray(wins)

            if np.issubdtype(wins.dtype, np.integer):
                return wins.astype(float) / self.ts_data.sample_rate

            return wins.astype(float)

        # -----------------------------
        # Build per-channel windows
        # Each channel gets its own dict:
        # {'A-alpha': (t0, t1), ...}
        # or
        # {'Total EMG': (t0, t1)}
        # -----------------------------

        per_channel_windows = {}
        per_channel_window_source = {}

        if windows is None:
            # Default: channel-aware windows

            neural_channels = np.asarray(getattr(self, "neural_channels", []), dtype=int)
            emg_channels = np.asarray(getattr(self, "emg_channels", []), dtype=int)

            for ch in channels:
                ch = int(ch)

                if ch in neural_channels:
                    neural_idx = np.where(neural_channels == ch)[0][0]

                    wins = _windows_to_seconds(self.neural_window_indices[neural_idx])
                    names = list(self.neural_fiber_names)

                    per_channel_windows[ch] = {
                        names[i]: tuple(wins[i])
                        for i in range(len(names))
                    }
                    per_channel_window_source[ch] = "neural"

                elif ch in emg_channels:
                    emg_idx = np.where(emg_channels == ch)[0][0]

                    wins = _windows_to_seconds(self.EMG_window_indicies[emg_idx])

                    # Usually this is ["Total EMG"]
                    names = list(getattr(self, "emg_window", ["Total EMG"]))

                    per_channel_windows[ch] = {
                        names[i]: tuple(wins[i])
                        for i in range(len(names))
                    }
                    per_channel_window_source[ch] = "EMG"

                else:
                    raise ValueError(
                        f"No default window is defined for channel index {ch} "
                        f"({self.ephys.ch_names[ch]}). Pass a custom windows dict or ndarray."
                    )

        elif isinstance(windows, dict):
            # Same custom windows applied to all selected channels
            for ch in channels:
                ch = int(ch)
                per_channel_windows[ch] = windows
                per_channel_window_source[ch] = "custom"

        elif isinstance(windows, np.ndarray):
            wins_all = _windows_to_seconds(windows)

            if wins_all.ndim != 3 or wins_all.shape[2] != 2:
                raise ValueError(
                    "windows ndarray must have shape (n_channels, n_windows, 2)"
                )

            n_windows = wins_all.shape[1]

            if window_names is None:
                if n_windows == len(self.neural_fiber_names):
                    window_names = list(self.neural_fiber_names)
                elif n_windows == 1:
                    window_names = ["custom_window"]
                else:
                    window_names = [f"window_{i}" for i in range(n_windows)]

            if len(window_names) != n_windows:
                raise ValueError(
                    f"window_names has length {len(window_names)}, "
                    f"but windows has {n_windows} windows"
                )

            # windows can match selected channels or all ephys channels
            if wins_all.shape[0] == len(channels):
                selected_wins = wins_all

            elif wins_all.shape[0] == self.ephys.shape[0]:
                selected_wins = wins_all[channels]

            else:
                raise ValueError(
                    "windows must have one row per selected channel "
                    "or one row per all ephys channels"
                )

            for ch, wins in zip(channels, selected_wins):
                ch = int(ch)
                per_channel_windows[ch] = {
                    window_names[i]: tuple(wins[i])
                    for i in range(len(window_names))
                }
                per_channel_window_source[ch] = "custom"

        else:
            raise TypeError(
                "windows must be None, a dict, or an ndarray of shape "
                "(n_channels, n_windows, 2)"
            )

        # -----------------------------
        # Calculate values row-by-row
        # This supports neural + EMG channels with different window names
        # -----------------------------
        records = []
        values = []

        for param in parameter:
            stimulation_amplitude = self.parameters.parameters.at[
                param, "pulse amplitude (μA)"
            ]

            for ch in channels:
                ch = int(ch)

                if windows is None:
                    # This is now parameter-aware
                    wins = self._default_windows_for_channel(ch, param)

                elif isinstance(windows, dict):
                    wins = windows

                else:
                    raise NotImplementedError(
                        "Custom ndarray windows can be added back in here if needed."
                    )

                res = self._calculate_AUC_single_param(
                    param,
                    wins,
                    channels=[ch],
                    method=method,
                    baseline=baseline,
                    absolute=absolute,
                    artifact_subtraction=artifact_subtraction,
                )

                for window_type, val in res.items():
                    records.append({
                        "parameter": param,
                        "stimulation_contact": self.parameters.parameters.at[param, "contact_name"],
                        "stimulation_amplitude": stimulation_amplitude,
                        "recording_channel_index": ch,
                        "recording_channel": self.ephys.ch_names[ch],
                        "window_type": window_type,
                        "auc_calculation_method": method,
                        "artifact_subtraction": artifact_subtraction,
                        "subject": self.ephys.metadata[param[0]].get("Subject")
                    })

                    values.append(da.asarray(val).reshape(-1)[0])

        # -----------------------------
        # Compute once and build dataframe
        # -----------------------------
        computed_values = da.stack(values, axis=0).compute()

        df = pd.DataFrame.from_records(records)
        df["AUC"] = np.asarray(computed_values).ravel()

        if self.master_df.empty:
            self.master_df = df
        else:
            self.master_df = pd.concat(
                [self.master_df, df],
                ignore_index=True
            )

        return df

    def filter_mean_waveforms(
        self,
        parameters=None,
        *,
        filter_channels=None,
        filter_median_highpass=False,
        filter_median_lowpass=False,
        filter_gaussian_highpass=False,
        filter_powerline=False,
    ) -> np.ndarray:
        """Compute and filter pulse-mean waveforms without mutating ECAP state.

        Returns
        -------
        np.ndarray
            Array shaped ``(parameters, channels, samples)``.
        """
        parameter_keys = self._normalize_parameter_keys(parameters)
        if parameters is None:
            waveforms = self.compute_mean_traces().copy()
        elif parameter_keys:
            waveforms = np.stack(
                [
                    self.compute_mean_traces(parameter_key)
                    for parameter_key in parameter_keys
                ],
                axis=0,
            ).copy()
        else:
            waveforms = np.empty(
                (0, int(self.ts_data.shape[0]), 0),
                dtype=self.ts_data.dtype,
            )
        n_channels = waveforms.shape[1]

        if filter_channels is None:
            channel_indices = np.arange(n_channels, dtype=int)
        else:
            selected = self._channel_index(filter_channels)
            channel_indices = np.arange(n_channels, dtype=int)[selected]
            channel_indices = np.asarray(channel_indices, dtype=int).reshape(-1)

        target = waveforms[:, channel_indices, :]

        if filter_median_highpass:
            target = np.apply_along_axis(
                lambda trace: trace - medfilt(trace, 201),
                2,
                target,
            )
        if filter_median_lowpass:
            target = np.apply_along_axis(
                lambda trace: medfilt(trace, 11),
                2,
                target,
            )
        if filter_gaussian_highpass:
            cutoff_hz = 4_000.0
            sigma = (2 * np.pi * (cutoff_hz / self.fs)) / np.sqrt(2 * np.log(2))
            target = np.apply_along_axis(
                lambda trace: trace - ndimage.gaussian_filter1d(trace, sigma),
                2,
                target,
            )
        if filter_powerline:
            warnings.warn(
                "filter_powerline is not implemented for mean waveforms.",
                stacklevel=2,
            )

        waveforms[:, channel_indices, :] = target
        return waveforms

    def plot_average_emg(
        self,
        condition,
        amplitude,
        recording_channels=None,
        *,
        display=True,
    ):
        """Plot pulse-mean EMG waveforms matching condition and amplitude."""
        table = self.parameters.parameters
        mask = (
            (table["condition"] == condition)
            & (table["pulse amplitude (μA)"] == amplitude)
        )
        parameter_keys = list(table.index[mask])
        if not parameter_keys:
            raise ValueError("No matching stimulation parameters were found.")

        channels = self.emg_channels if recording_channels is None else recording_channels
        figure, axis = plt.subplots()
        for parameter_key in parameter_keys:
            waveform = self.compute_mean_traces(parameter_key)
            selected = self._channel_index(channels)
            channel_indices = np.arange(
                int(self.ts_data.shape[0]),
                dtype=int,
            )[selected]
            channel_indices = np.asarray(channel_indices, dtype=int).reshape(-1)
            waveform = waveform[channel_indices]
            time = self.time_axis(parameter_key)
            for trace in np.asarray(waveform):
                axis.plot(time, trace)

        axis.set_title(f"Condition: {condition}; amplitude: {amplitude}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        if display:
            plt.show()
        return figure, axis

    def plot_recording_channels(
        self,
        parameter,
        *,
        relative_time_frame=None,
        display=True,
    ):
        """Plot neural and EMG pulse-mean traces for one parameter."""
        waveform = self.compute_mean_traces(parameter)
        time = self.time_axis(parameter)

        if relative_time_frame is not None:
            start, stop = map(float, relative_time_frame)
            mask = (time >= start) & (time < stop)
            time = time[mask]
            waveform = waveform[:, mask]

        n_rows = max(
            int(self.neural_channels.size),
            int(self.emg_channels.size),
            1,
        )
        figure, axes = plt.subplots(
            n_rows,
            2,
            squeeze=False,
            sharex=True,
            figsize=(10, 3 * n_rows),
        )

        for row, channel in enumerate(self.neural_channels):
            axes[row, 0].plot(time, waveform[int(channel)])
            axes[row, 0].set_title(self.ephys.ch_names[int(channel)])
            local_channel = int(np.flatnonzero(self.neural_channels == channel)[0])
            for start, _ in self.neural_window_indices[local_channel]:
                axes[row, 0].axvline(float(start) / self.fs)

        for row, channel in enumerate(self.emg_channels):
            axes[row, 1].plot(time, waveform[int(channel)])
            axes[row, 1].set_title(self.ephys.ch_names[int(channel)])
            local_channel = int(np.flatnonzero(self.emg_channels == channel)[0])
            for start, stop in self.calculate_emg_window_lengths(parameter)[local_channel]:
                axes[row, 1].axvline(float(start) / self.fs)
                axes[row, 1].axvline(float(stop) / self.fs)

        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        figure.tight_layout()
        if display:
            plt.show()
        return figure, axes

    def features_per_fiber(
        self,
        parameters=None,
        *,
        baseline_s=None,
        methods=("RMS", "PEAKS"),
    ) -> pd.DataFrame:
        """Compute pulse-mean ECAP features for each neural fiber window."""
        parameter_keys = self._normalize_parameter_keys(parameters)
        methods = tuple(str(method).upper() for method in methods)
        unknown = set(methods) - {"RMS", "PEAKS"}
        if unknown:
            raise ValueError(f"Unknown feature methods: {sorted(unknown)}")

        records = []
        arrays = []
        for parameter_key in parameter_keys:
            waveform = da.asarray(
                self.compute_mean_traces(parameter_key)[self.neural_channels]
            )

            if baseline_s is not None:
                start, stop = self._time_window_to_indices(
                    parameter_key,
                    baseline_s,
                )
                waveform = waveform - waveform[:, start:stop].mean(
                    axis=1,
                    keepdims=True,
                )

            parameter_features = []
            columns = []
            for method in methods:
                per_fiber = []
                for fiber_index, fiber_name in enumerate(self.neural_fiber_names):
                    channel_values = []
                    for local_channel in range(self.neural_channels.size):
                        start, stop = self.neural_window_indices[
                            local_channel,
                            fiber_index,
                        ]
                        segment = waveform[local_channel, int(start):int(stop)]
                        if method == "RMS":
                            value = da.sqrt(da.mean(segment**2))
                        else:
                            value = segment.max() - segment.min()
                        channel_values.append(value)
                    per_fiber.append(da.stack(channel_values))
                    columns.append(f"{method}_{fiber_name}")
                parameter_features.extend(per_fiber)

            matrix = da.stack(parameter_features, axis=1)
            arrays.append(matrix)
            records.extend(
                (parameter_key, self.ephys.ch_names[int(channel)])
                for channel in self.neural_channels
            )

        if not arrays:
            return pd.DataFrame()

        values = da.concatenate(arrays, axis=0).compute()
        index = pd.MultiIndex.from_tuples(
            records,
            names=["parameter", "channel"],
        )
        return pd.DataFrame(values, index=index, columns=columns)

    def plot_average_recordings(
        self,
        amplitude,
        condition=None,
        relative_time_frame=None,
        *,
        display=False,
        save=False,
        save_directory=None,
    ):
        """Plot neural and EMG pulse means for parameters at one amplitude."""
        table = self.parameters.parameters
        mask = table["pulse amplitude (μA)"] == amplitude
        if condition is not None:
            mask &= table["condition"] == condition
        parameter_keys = list(table.index[mask])
        if not parameter_keys:
            raise ValueError("No matching stimulation parameters were found.")

        figures = []
        for parameter_key in parameter_keys:
            waveform = self.compute_mean_traces(parameter_key)
            time = self.time_axis(parameter_key)
            if relative_time_frame is not None:
                start, stop = map(float, relative_time_frame)
                selection = (time >= start) & (time < stop)
                time = time[selection]
                waveform = waveform[:, selection]

            figure, axes = plt.subplots(1, 2, squeeze=False)
            neural_axis, emg_axis = axes[0]
            for channel in self.neural_channels:
                neural_axis.plot(time, waveform[int(channel)])
            neural_axis.set_title("Neural channels")
            for channel in self.emg_channels:
                emg_axis.plot(time, waveform[int(channel)])
            emg_axis.set_title("EMG channels")
            for axis in axes[0]:
                axis.set_xlabel("Time (s)")
                axis.set_ylabel("Amplitude")
            figure.tight_layout()

            if save:
                directory = Path(save_directory or ".")
                directory.mkdir(parents=True, exist_ok=True)
                figure.savefig(directory / f"{parameter_key}_{amplitude}.png")
            if display:
                plt.show()
            figures.append((figure, axes))

        return figures

    @staticmethod
    def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
        """Return an RMS-smoothed array along the final axis."""
        if win <= 1:
            return np.abs(x)
        kernel = np.ones(int(win), dtype=float) / int(win)
        squared = np.asarray(x) ** 2
        pad = int(win) // 2
        padded = np.pad(
            squared,
            [(0, 0)] * (squared.ndim - 1) + [(pad, pad)],
            mode="reflect",
        )
        smoothed = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"),
            -1,
            padded,
        )
        return np.sqrt(smoothed)

    def _artifact_end_indices_block(self, block: np.ndarray,
                                    min_end: int,
                                    baseline_tail: int,
                                    smooth_win: int,
                                    hold: int,
                                    k_mad: float) -> np.ndarray:
        """
        block: (p, c, T) numpy
        returns t_end: (p,) numpy (conservative across channels)
        """
        p, c, T = block.shape

        # envelope on abs signal (RMS-smoothed)
        env = self._moving_rms(np.abs(block), smooth_win)  # (p, c, T)

        # baseline stats from tail
        tail0 = max(T - baseline_tail, 0)
        tail = env[..., tail0:]  # (p, c, Tb)

        # robust baseline: median + k*MAD
        med = np.median(tail, axis=-1)  # (p, c)
        mad = np.median(np.abs(tail - med[..., None]), axis=-1)  # (p, c)
        sigma = 1.4826 * mad
        thr = med + k_mad * sigma  # (p, c)

        # detect first time after min_end where env < thr for 'hold' consecutive samples
        t_end = np.full((p,), min_end, dtype=np.int32)

        below = env < thr[..., None]  # (p, c, T)

        # conservative across channels: require *all* channels below threshold
        below_all = np.all(below, axis=1)  # (p, T)

        # find first run of length hold
        for i in range(p):
            b = below_all[i, :]
            start = min_end
            found = False
            # scan for a consecutive run of 'hold' Trues
            # (T is only 977, so this is cheap)
            for t in range(start, T - hold):
                if b[t:t + hold].all():
                    t_end[i] = t
                    found = True
                    break
            if not found:
                # fall back: turn on near the end
                t_end[i] = max(min_end, T - hold - 1)

        return t_end

    def artifact_adaptive_common_median_reference(
            self,
            epochs: da.Array,  # (pulses, channels, samples)
            ref_pool: np.ndarray,  # 1D array of channel indices used to form the reference (exclude closest)
            min_end: int = 18,  # earliest possible artifact end (samples)
            baseline_tail: int = 200,  # samples used to estimate baseline (tail of epoch)
            smooth_win: int = 9,  # envelope smoothing window (samples)
            hold: int = 10,  # require below-threshold for this many samples
            k_mad: float = 8.0,  # threshold = median + k*MAD
            ramp: int = 20,  # taper length (samples) for gate
            return_t_end: bool = True
    ):
        """
        Returns:
          cleaned_epochs: dask array (same shape)
          t_end: numpy array (pulses,) if return_t_end else not returned
        """
        if epochs.ndim != 3:
            raise ValueError("epochs must be (pulses, channels, samples)")
        P, C, T = epochs.shape
        ref_pool = np.asarray(ref_pool, dtype=int)
        if ref_pool.ndim != 1 or ref_pool.size < 1:
            raise ValueError("ref_pool must be a non-empty 1D array of channel indices")
        if np.any(ref_pool < 0) or np.any(ref_pool >= C):
            raise ValueError("ref_pool has out-of-range channel indices")

        # Make sure time is not chunked (epochs are short, this is ideal)
        epochs2 = epochs.rechunk({2: -1})

        # ---- Step 1: compute t_end per pulse (small output; OK to compute eagerly) ----
        t_end_da = da.map_blocks(
            self._artifact_end_indices_block,
            epochs2,
            dtype=np.int32,
            drop_axis=(1, 2),  # drop channels and time -> output is (pulses,)
            new_axis=(),  # keep 1D
            min_end=min_end,
            baseline_tail=baseline_tail,
            smooth_win=smooth_win,
            hold=hold,
            k_mad=k_mad,
        )
        t_end = t_end_da.compute()

        # ---- Step 2: build a gate per pulse (numpy; small) ----
        gate = np.zeros((P, T), dtype=np.float32)
        for i in range(P):
            te = int(t_end[i])
            te = max(0, min(te, T - 1))
            r0 = te
            r1 = min(te + ramp, T)
            if r0 < T:
                gate[i, r0:r1] = np.linspace(0.0, 1.0, max(r1 - r0, 1), endpoint=False, dtype=np.float32)
                gate[i, r1:] = 1.0

        gate_da = da.from_array(gate, chunks=(epochs2.chunks[0], (T,)))  # align pulse chunking

        # ---- Step 3: subtract gated median reference in a Dask blockwise way ----
        def _apply_ref_block(block: np.ndarray, gate_block: np.ndarray) -> np.ndarray:
            # block: (p, c, T), gate_block: (p, T)
            # compute ref from restricted pool
            ref = np.median(block[:, ref_pool, :], axis=1)  # (p, T)
            # apply gated subtraction to all channels
            return block - (gate_block[:, None, :] * ref[:, None, :])

        cleaned = da.map_blocks(
            _apply_ref_block,
            epochs2,
            gate_da,
            dtype=epochs2.dtype
        )

        return (cleaned, t_end) if return_t_end else cleaned

