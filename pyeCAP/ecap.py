# neuro base class imports
from .base.epoch_data import _EpochData
from .base.utils.numeric import _to_numeric_array

# other imports

import dask.array as da
import numpy as np
import warnings
import pandas as pd
from scipy import ndimage
from scipy.signal import medfilt, find_peaks, savgol_filter, welch
import matplotlib.pyplot as plt
import sys


# TODO: edit docstrings
class ECAP(_EpochData):
    """
    This class represents ECAP data
    """

    def __init__(self, ephys_data, stim_data, distance_log=None, preload=False):
        """
        Constructor for the ECAP class.

        Parameters
        ----------
        ephys_data : _TsData or subclass instance
            Ephys data object.
        stim_data : Stim class instance
            Stimulation data object.
        distance_log : array of distances from recording electrode(s) to stimulating electrode [=] cm
        """
        # todo: differentiate recording channels
        # todo: improve unit checking i.e. [2 cm, 1 mm]
        # TODO: check len(distances) == len(rec_electrodes)

        super().__init__(ephys_data, stim_data, stim_data)

        if distance_log is not None:
            self.distance_log = _to_numeric_array(distance_log)
        else:
            self.distance_log = [0]

        # Lists to look through for ranges
        self.neural_fiber_names = ['A-alpha', 'A-beta', 'A-gamma', 'A-delta', 'B']
        self.emg_window = ["Total EMG"]

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        self.fiber_windows = None




        if 'EMG' in self.ephys.types:
            self.emg_channels = np.arange(0, self.ephys.shape[0])[self.ephys._ch_num_mask_by_type['EMG']]
        if 'ENG' in self.ephys.types:
            self.neural_channels = np.arange(0, self.ephys.shape[0])[self.ephys._ch_num_mask_by_type['ENG']]
        else:
            warnings.warn("Neural channels not implicitly stated. Assuming all channels are neural recordings")
            self.ephys = self.ephys.set_ch_types(["ENG"]*self.ephys.shape[0])
            self.neural_channels = np.arange(0, self.ephys.shape[0])

        self.neural_window_indices = self.calculate_neural_window_lengths()
        if 'EMG' in self.ephys.types:
            self.EMG_window_indicies = self.calculate_emg_window_lengths()

        if type(self.distance_log) == list and self.distance_log != [0]:
            if type(self.neural_window_indices) == np.ndarray and len(self.neural_window_indices.shape) > 1:
                if self.neural_window_indices.shape[0] != len(self.distance_log) and self.distance_log != [0]:
                    raise ValueError("Recording lengths don't match recording channel lengths")

            elif len(self.neural_window_indices) != len(self.distance_log):
                raise ValueError("Recording lengths don't match recording channel lengths")

        self.master_df = pd.DataFrame()


        self.log_path = distance_log

        if preload:
            self.preload()
            # self._window_dict_for_all_neural_channels()




    def _default_windows_for_channel(self, ch, parameter):
        """
        Return the appropriate default neural window dictionary for one channel
        and one stimulation parameter.

        Neural channels get neural fiber windows.
        EMG channels get the EMG window calculated for the current parameter.
        """

        fs = self.ts_data.sample_rate

        def _to_seconds(wins):
            wins = np.asarray(wins)

            if np.issubdtype(wins.dtype, np.integer):
                return wins.astype(float) / fs

            return wins.astype(float)

        neural_channels = np.asarray(getattr(self, "neural_channels", []), dtype=int)
        emg_channels = np.asarray(getattr(self, "emg_channels", []), dtype=int)

        ch = int(ch)

        if ch in neural_channels:
            neural_idx = np.where(neural_channels == ch)[0][0]

            wins = _to_seconds(self.neural_window_indices[neural_idx])
            names = list(self.neural_fiber_names)

            return {
                names[i]: tuple(wins[i])
                for i in range(len(names))
            }

        elif ch in emg_channels:
            emg_idx = np.where(emg_channels == ch)[0][0]

            # Important: calculate EMG window for THIS parameter
            emg_windows = self.calculate_emg_window_lengths(parameter=parameter)

            wins = _to_seconds(emg_windows[emg_idx])
            names = list(getattr(self, "emg_window", ["Total EMG"]))

            return {
                names[i]: tuple(wins[i])
                for i in range(len(names))
            }

        else:
            raise ValueError(
                f"No default window is defined for channel {ch} "
                f"({self.ephys.ch_names[ch]})."
            )



    def _time_window_to_indices(self, parameter, window_s):
        """
        Convert an epoch-relative time window to sample indices.

        Parameters
        ----------
        parameter
            Parameter key whose epoch dimensions should be used.
        window_s
            Two-element window in seconds relative to stimulation time.

            For example, (0.001, 0.008) selects 1-8 ms after the
            stimulation event.

        Returns
        -------
        tuple[int, int]
            Start and stop indices clipped to the epoch bounds.
        """
        if len(window_s) != 2:
            raise ValueError(
                "window_s must contain exactly two values: "
                "(start_seconds, stop_seconds)."
            )

        window_start_s = float(window_s[0])
        window_stop_s = float(window_s[1])

        if window_stop_s <= window_start_s:
            raise ValueError(
                "window_s must satisfy stop > start. "
                f"Received {window_s!r}."
            )

        sample_rate = float(self.ts_data.sample_rate)
        n_samples = self.epoch_sample_length(parameter)

        # Time represented by epoch index zero.
        if self.epoch_window is None or self.epoch_window == "auto":
            epoch_start_s = 0.0
        else:
            epoch_start_s = float(self.epoch_window[0])

        i0 = int(
            np.floor(
                (window_start_s - epoch_start_s) * sample_rate
            )
        )
        i1 = int(
            np.ceil(
                (window_stop_s - epoch_start_s) * sample_rate
            )
        )

        i0 = max(0, min(n_samples, i0))
        i1 = max(0, min(n_samples, i1))

        return i0, i1

    def _windows_dict_for_channel(self, neural_windows, ch_idx=None):
        """
        neural_windows: ndarray
            Shape: (n_channels, n_fibers, 2)  # start/stop per fiber

        ch_idx : int

        Returns dict: {'Aalpha': (t0, t1), ...} in SECONDS for a single channel.

        To have dict of all fibers for ALL channels, use:
            _window_dict_for_all_neural_channels
        """
        fs = self.ts_data.sample_rate
        fiber_names = list(self.neural_fiber_names)  # axis-1 order must match this
        wins = neural_windows[ch_idx]  # (n_fibers, 2)

        # If the array is in samples (ints), convert to seconds; if already float seconds, leave as-is.
        if np.issubdtype(wins.dtype, np.integer):
            wins_sec = wins.astype(float) / fs
        else:
            wins_sec = wins

        return {fiber_names[i]: (float(wins_sec[i, 0]), float(wins_sec[i, 1]))
                for i in range(wins_sec.shape[0])}

    # def _window_dict_for_all_neural_channels(self, return_seconds=True):
    #     """
    #     return_seconds: bool
    #         -True: return dictionary of start and stop TIMES
    #         -False: return dictionary of start and stop INDICIES
    #     Returns
    #     -------
    #     Returns dictionary of fibers of start and stop times or indicies
    #     """
    #     fws = {
    #         fiber_name: self.neural_window_indices[:, i, :]
    #         for i, fiber_name in enumerate(self.neural_fiber_names)
    #     }
    #
    #     if(return_seconds is True):
    #         fws = {
    #             fiber_name: windows / self.fs
    #             for fiber_name, windows in fws.items()
    #         }
    #
    #     self.fiber_windows = fws
    #
    #
    #     return fws

    # def _subtract_artifact(self, signal, window_jitter=0.01, ):

    def _calculate_AUC_single_param(self, parameter, windows, channels=None, method="RMS", baseline=None, absolute=False, artifact_subtraction=False):
        out = {}
        for name, w in windows.items():
            if method.upper() == "RMS":
                out[name] = self.rms_in_window(parameter, channels=channels, window_s=w, baseline_s=baseline)
            elif method.upper() == "AUC":
                out[name] = self.auc_in_window(parameter, channels=channels, window_s=w, baseline_s=baseline,
                                               absolute=absolute, artifact_subtraction=artifact_subtraction)
            else:
                raise ValueError("method must be 'RMS', 'AUC', or 'Peaks'")
        return out




    def preload(
            self,
            parameters=None,
            pulses=None,
            channels=None,
            samples=None,
    ):
        """
        Preconstruct lazy epoch and mean-waveform graphs.

        By default, this processes all parameters, all pulses, all
        channels, and all samples.

        Notes
        -----
        This builds lazy Dask graphs. It does not load all numerical data
        into RAM or call compute().
        """
        parameter_keys = self._normalize_parameter_keys(parameters)

        # Build raw epochs first. This populates the dask_array lru_cache.
        self.build_epoch_array(
            parameters=parameter_keys,
            pulses=pulses,
            channels=channels,
            samples=samples,
        )

        # This can reuse the cached dask_array graphs.
        self.build_mean_over_pulses(
            parameters=parameter_keys,
            channels=channels,
            samples=samples,
        )

        return self


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
        """
        Calculate EMG window indices.

        Window starts after stimulation artifact/capacitive discharge:
            onset = 6 * pulse_width

        Window ends at the shorter of:
            - one stimulation period
            - available epoch length

        Returns
        -------
        list
            Shape: n_emg_channels x n_emg_windows x 2
            Example: [[[onset, offset]], [[onset, offset]], ...]
        """

        if parameter is None:
            parameter = self.stim.parameters.index[0]

        fs = self.ts_data.sample_rate
        params = self.stim.parameters.loc[parameter]

        pw = params["pulse duration (ms)"] / 1000
        freq = params["frequency (Hz)"]

        onset = int(np.ceil(pw * 6 * fs))

        offset1 = int(np.floor(fs / freq)) - 1

        event_times = self.parameter_event_times.get(tuple(parameter), None)
        diffs = np.diff(event_times)
        offset2 = int(diffs.min() * self.ts_data.sample_rate)

        offset = min(offset1, offset2)

        if offset <= onset:
            offset = min(onset + 1, offset2)

        return [
            [[onset, offset]]
            for _ in self.emg_channels
        ]

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

    def filter_averages(self, filter_channels=None, filter_median_highpass=False, filter_median_lowpass=False,
                        filter_gaussian_highpass=False, filter_powerline=False):

        # First step: get an separate channels to be filtered.
        if ((filter_median_highpass is True) or
                (filter_median_lowpass is True) or
                (filter_gaussian_highpass is True) or
                (filter_powerline is True)):
            print("Begin filtering averages")
        if type(filter_channels) == int:
            filter_channels = [filter_channels]

        if filter_channels is None:
            filter_channels = slice(None, None, None)
            target_list = np.copy(self.mean_traces)
            exclusion_list = None

        else:
            target_list = np.copy(self.mean_traces[:, filter_channels])
            length_list = np.arange(0, len(self.mean_traces[0]))
            exclusion_list = [value for value in length_list if value not in filter_channels]

        # Second step: filter channels
        if filter_median_highpass is True:
            filtered_median_traces = np.apply_along_axis(lambda x: x - medfilt(x, 201), 2, target_list)
            target_list = filtered_median_traces

        if filter_median_lowpass is True:
            filtered_median_traces = np.apply_along_axis(lambda x: medfilt(x, 11), 2, target_list)
            target_list = filtered_median_traces

        if filter_gaussian_highpass is True:
            Wn = 4000
            s_c = Wn / self.fs
            sigma = (2 * np.pi * s_c) / np.sqrt(2 * np.log(2))
            filtered_gauss_traces = np.apply_along_axis(lambda x: ndimage.filters.gaussian_filter1d(x, sigma), 2,
                                                        target_list)
            target_list = filtered_gauss_traces

        # Second step: merge separated channels back into original list.
        # only applicable if filter_channels were selected
        if filter_channels != slice(None, None, None):
            new_list = []

            for first_idx in range(len(self.mean_traces)):
                new_sublist = []
                target_idx = 0
                for i in length_list:
                    if i in filter_channels:
                        new_sublist.append(target_list[first_idx, target_idx])
                        target_idx += 1
                    elif i in exclusion_list:
                        new_sublist.append(self.mean_traces[first_idx, i])
                new_list.append(new_sublist)

            self.mean_traces = np.array(new_list)

        else:
            self.mean_traces = np.array(target_list)
        if ((filter_median_highpass is True) or
                (filter_median_lowpass is True) or
                (filter_gaussian_highpass is True) or
                (filter_powerline is True)):
            print("Finished Filtering Averages")



    def plot_average_EMG(self, condition, amplitude, rec_channel):
        ### under construction
        df = self.stim.parameters
        parameter_indicies = df.index[
            (df['condition'] == condition) &
            (df['pulse amplitude (μA)'] == amplitude)][rec_channel]

        if len(parameter_indicies) == 0:
            raise ValueError("No such values specified found.")

        for p in parameter_indicies:
            idx = self.parameters_dictionary[p]
            fig, ax = plt.subplots()
            fig_title = "Condition: " + condition + " Amplitude: " + str(amplitude)
            fig.suptitle(fig_title)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            ax.plot(self.mean_traces[idx])


    def plot_recChannel_per_axes(self, condition, amplitude, stim_channel, relative_time_frame=None, display=False,
                                 save=False):
        """
        Plots all recording channels for a given condition, stimulation channel and amplitude.
        :param condition: Experimental Condition
        :param amplitude: Stimulation Amplitude
        :param stim_channel: Stimulation Channel
        :param relative_time_frame: Time frame in (s) to visualize
        :param display: Show figure
        :param save: Save figure
        """
        df = self.stim.parameters
        parameter_indicies = df.index[
            (df['condition'] == condition) &
            (df['pulse amplitude (μA)'] == amplitude) &
            (df['channel'] == stim_channel)]

        if len(parameter_indicies) == 0:
            raise ValueError("No such values specified found.")

        if relative_time_frame is None:
            relative_time_frame = slice(None, None, None)
            relative_time_ts = [i / self.fs for i in np.arange(0, self.mean_traces.shape[2])]

        elif type(relative_time_frame) is list:
            relative_time_ts = np.arange(relative_time_frame[0], relative_time_frame[1], 1 / self.fs)
            relative_time_frame = slice(int(round(relative_time_frame[0] * self.ephys.sample_rate)),
                                        int(round((relative_time_frame[-1] * self.ephys.sample_rate))))
            if len(relative_time_ts) == (relative_time_frame.stop - relative_time_frame.start) + 1:
                relative_time_ts = relative_time_ts[0:-1]

        for p in parameter_indicies:
            idx = self.parameters_dictionary[p]
            if self.stim.parameters.loc[p]['pulse amplitude (μA)'] < 0:
                amplitude = -1 * self.stim.parameters.loc[p]['pulse amplitude (μA)']
            else:
                amplitude = self.stim.parameters.loc[p]['pulse amplitude (μA)']

            fig, ax = plt.subplots(max(len(self.emg_channels), len(self.neural_channels)), 2, sharex=True, sharey='col',
                                   figsize=(10, 5 * max(len(self.emg_channels), len(self.neural_channels))))

            for rec_idx, channel in enumerate(self.neural_channels):
                min_h = np.amin(self.mean_traces[idx, self.neural_channels, relative_time_frame])
                max_h = np.amax(self.mean_traces[idx, self.neural_channels, relative_time_frame])

                ax[rec_idx, 0].plot(relative_time_ts, self.mean_traces[idx, channel, relative_time_frame].T)
                ax[rec_idx, 0].set_title("Channel: " + str(channel))
                if rec_idx < len(self.neural_channels):
                    for fiber_onsets in self.neural_window_indices[rec_idx]:
                        ax[rec_idx, 0].vlines(fiber_onsets[0] / self.fs, min_h, max_h)

            for rec_idx, channel in enumerate(self.emg_channels):
                min_h = np.amin(self.mean_traces[idx, self.emg_channels, relative_time_frame])
                max_h = np.amax(self.mean_traces[idx, self.emg_channels, relative_time_frame])

                ax[rec_idx, 1].plot(relative_time_ts, self.mean_traces[idx, channel, relative_time_frame].T)
                ax[rec_idx, 1].set_title("Channel: " + str(channel))
                if rec_idx < len(self.emg_channels):
                    for fiber_onsets in self.EMG_window_indicies[rec_idx]:
                        ax[rec_idx, 1].vlines(fiber_onsets[0] / self.fs, min_h, max_h)
                        ax[rec_idx, 1].vlines(fiber_onsets[1] / self.fs, min_h, max_h)

            fig_title = "Condition: " + condition + " Stim Channel:" + stim_channel + " Amplitude: " + str(amplitude)
            fig.suptitle(fig_title)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save:
                plt.savefig(condition + " " + stim_channel + " " + str(amplitude) + ".jpg")
            if display:
                plt.show()

    # def _ts(self):
    #     """Return the time-series array, preferring a persisted cache if present."""
    #     return getattr(self, "_persisted_ts_array", self.ts_data.array)
    #
    # def _autotune_time_chunk(self, *, k_window=6, target_block_mb=16, max_span_s=0.5):
    #     fs = float(self.ts_data.sample_rate)
    #     if self.epoch_window in (None, "auto"):
    #         win_s = 0.010
    #     else:
    #         win_s = float(self.epoch_window[1] - self.epoch_window[0])
    #     W = max(1, int(round(win_s * fs)))
    #     C = len(self.ts_data.ch_names)
    #     bytes_per_sample_all_ch = C * self.ts_data.array.dtype.itemsize
    #     min_chunk = max(W * int(k_window), 1)
    #     budget_samples = max(1, int((target_block_mb * 1024 ** 2) / bytes_per_sample_all_ch))
    #     cap_samples = max(1, int(max_span_s * fs))
    #     return int(max(min_chunk, min(budget_samples, cap_samples)))

    # def _replace_array(self, new_array):
    #     """Internal: replace the backing dask array."""
    #     self._array = new_array
    #     return self

    # def persist_ts(self, time_chunk: int | None = None, **tuner):
    #     """
    #     Rechunk and persist the continuous time-series array.
    #
    #     Warning
    #     -------
    #     This computes and retains the complete time-series array in memory.
    #     """
    #     arr = self.ts_data.array
    #
    #     if time_chunk is None:
    #         time_chunk = self._autotune_time_chunk(**tuner)
    #
    #     arr = arr.rechunk(
    #         {
    #             0: int(arr.shape[0]),
    #             1: int(time_chunk),
    #         }
    #     )
    #
    #     # This is the attribute dask_array() actually checks.
    #     self._persisted_array = arr.persist()
    #
    #     # Previously constructed epoch graphs point at the old source.
    #     self.clear_all_epoch_caches()
    #
    #     return self

    # def unpersist_ts(self):
    #     """Remove the persisted source and invalidate dependent caches."""
    #     if hasattr(self, "_persisted_array"):
    #         del self._persisted_array
    #
    #     self.clear_all_epoch_caches()
    #
    #     return self

    def features_per_fiber(self, parameters=None, baseline_s=None, methods=("RMS", "PEAKS"),
                           batch_params: int | None = None):
        """
        Compute ECAP features per fiber window for each parameter & channel.
        Returns a DataFrame (param, channel) x [RMS_Aα, ..., PEAKS_Aα, ...].
        """
        fs = float(self.ts_data.sample_rate)
        if parameters is None:
            parameters = list(self.parameters.parameters.index)

        # ensure window indices exist
        win = getattr(self, "neural_window_indices", None)
        if not win:
            raise ValueError("neural_window_indices missing; compute your fiber windows first.")
        fibers = list(win.keys())

        # optional baseline subtraction (common mean over a pre-stim window)
        if self.epoch_window in (None, "auto"):
            x0 = 0.0
        else:
            x0, _ = self.epoch_window
        if baseline_s is not None:
            b0 = int(np.floor((baseline_s[0] - x0) * fs))
            b1 = int(np.ceil((baseline_s[1] - x0) * fs))

        def _rms(seg):  # seg: (ch, W)
            return da.sqrt(da.mean(seg ** 2, axis=-1))

        def _peaks(seg):
            return da.maximum(da.abs(seg.max(axis=-1) - seg.min(axis=-1)), 0)

        cols = []
        mats = []

        # Optional batching to cap peak memory on huge param sets
        if batch_params is None or batch_params <= 0:
            batches = [parameters]
        else:
            batches = [parameters[i:i + batch_params] for i in range(0, len(parameters), batch_params)]

        for batch in batches:
            # Build per-param mean lazily and reduce immediately fiber-by-fiber
            per_method = {m: [] for m in methods}
            for p in batch:
                wf = self.mean_waveform(p)  # (ch, T) lazy
                if baseline_s is not None:
                    T = int(wf.shape[-1])
                    i0 = max(0, min(T - 2, b0))
                    i1 = max(i0 + 1, min(T, b1))
                    base = wf[..., i0:i1].mean(axis=-1, keepdims=True)
                    wf = wf - base

                # compute features for each fiber window
                feats = {}
                for f in fibers:
                    i0, i1 = win[f]
                    seg = wf[..., int(i0):int(i1)]
                    if "RMS" in methods:
                        feats.setdefault("RMS", []).append(_rms(seg))  # (ch,)
                    if "PEAKS" in methods:
                        feats.setdefault("PEAKS", []).append(_peaks(seg))  # (ch,)

                # stack to (ch, n_fibers) per method
                for m in feats:
                    per_method[m].append(da.stack(feats[m], axis=-1))  # (ch, F)

            # concat params in this batch → (P_batch, ch, F)
            for m in per_method:
                mats.append(da.stack(per_method[m], axis=0))  # (P_batch, ch, F)
                cols.extend([f"{m}_{f}" for f in fibers])

        # Combine all methods/batches along the last axis, compute once
        feat = da.concatenate(mats, axis=-1).compute()  # shape: (P_total, ch, n_cols)

        idx = pd.MultiIndex.from_product([parameters, list(self.ts_data.ch_names)],
                                         names=["parameter", "channel"])
        df = pd.DataFrame(feat.reshape(len(parameters) * len(self.ts_data.ch_names), -1),
                          index=idx, columns=cols)
        return df

    def plot_average_recordings(self, amplitude, condition=None, relative_time_frame=None, display=False,
                                save=False):
        df = self.stim.parameters

        if condition is None:
            parameter_indicies = df.index[df['pulse amplitude (μA)'] == amplitude]

        else:
            parameter_indicies = df.index[
                (df['condition'] == condition) &
                (df['pulse amplitude (μA)'] == amplitude)
                ]

        if len(parameter_indicies) == 0:
            raise ValueError("No such values specified found.")

        if relative_time_frame is None:
            relative_time_frame = slice(None, None, None)
            relative_time_ts = [i / self.fs for i in np.arange(0, self.mean_traces.shape[2])]

        elif type(relative_time_frame) is list:
            relative_time_ts = np.arange(relative_time_frame[0], relative_time_frame[1], 1 / self.fs)
            relative_time_frame = slice(int(round(relative_time_frame[0] * self.ephys.sample_rate)),
                                        int(round((relative_time_frame[-1] * self.ephys.sample_rate))))
            if len(relative_time_ts) == (relative_time_frame.stop - relative_time_frame.start) + 1:
                relative_time_ts = relative_time_ts[0:-1]

        fig, ax = plt.subplots(1, 2)
        for p in parameter_indicies:
            idx = self.parameters_dictionary[p]
            if self.stim.parameters.loc[p]['pulse amplitude (μA)'] < 0:
                amplitude = -1 * self.stim.parameters.loc[p]['pulse amplitude (μA)']
            else:
                amplitude = self.stim.parameters.loc[p]['pulse amplitude (μA)']

            for rec_idx, channel in enumerate(self.neural_channels):
                ax[0].plot(relative_time_ts, self.mean_traces[idx, channel, relative_time_frame].T)
                ax[0].set_title("Neural Channels")

            for rec_idx, channel in enumerate(self.emg_channels):
                ax[1].plot(relative_time_ts, self.mean_traces[idx, channel, relative_time_frame].T)
                ax[1].set_title("EMG Channels")

            if save:
                plt.savefig(condition + " " + self.stim.parameters.loc[p]['channel'] + " " + str(amplitude) + ".jpg")
            if display:
                plt.show()

    def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
        """x: (..., T) -> RMS-smoothed along last axis"""
        if win <= 1:
            return np.sqrt(x * x)
        k = np.ones(win, dtype=float) / win
        # apply along last axis
        x2 = x * x
        # pad reflect to avoid edge bias
        pad = win // 2
        x2p = np.pad(x2, [(0, 0)] * (x2.ndim - 1) + [(pad, pad)], mode="reflect")
        # convolution along last axis
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), -1, x2p)
        return np.sqrt(out)

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