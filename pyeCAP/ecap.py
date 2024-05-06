# neuro base class imports
import os
import sys
import time
import warnings

import dask

# other imports
import dask.bag as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks, medfilt, savgol_filter

from .base.epoch_data import _EpochData
from .base.utils.numeric import _to_numeric_array
from .base.utils.visualization import _plt_setup_fig_axis, _plt_show_fig
from .utilities.ancillary_functions import check_make_dir


# TODO: edit docstrings
class ECAP(_EpochData):
    """
    This class represents ECAP data
    """

    def __init__(self, ephys_data, stim_data, distance_log=None):
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

        if distance_log is not None:
            self.distance_log = _to_numeric_array(distance_log)
        else:
            warnings.warn(
                "Electrode distance information not provided. AUC calculation using standard neural windows cannot be completed."
            )
            self.distance_log = [0]

        # Lists to look through for ranges
        self.neural_fiber_names = ["A-alpha", "A-beta", "A-gamma", "A-delta", "B"]
        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        # self.master_df = pd.DataFrame()
        self.neural_window_indicies = self.calculate_neural_window_lengths()

        if type(self.distance_log) == list and self.distance_log != [0]:
            if (
                type(self.neural_window_indicies) == np.ndarray
                and len(self.neural_window_indicies.shape) > 1
            ):
                if self.neural_window_indicies.shape[0] != len(
                    self.distance_log
                ) and self.distance_log is not [0]:
                    raise ValueError(
                        "Recording lengths don't match recording channel lengths"
                    )

            elif len(self.neural_window_indicies) != len(self.distance_log):
                raise ValueError(
                    "Recording lengths don't match recording channel lengths"
                )

        if distance_log == "Experimental Log.xlsx":
            self.log_path = ephys_data.base_path + distance_log
        else:
            self.log_path = distance_log

        super().__init__(ephys_data, stim_data, stim_data)

    def calculate_neural_window_lengths(self):
        """
        :return: time_windows[recording_electrode][fiber_type][start/stop]
        """
        # min and max conduction velocities of various fiber types
        a_alpha = [
            120,
            70,
        ]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_beta = [
            70,
            30,
        ]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_gamma = [
            30,
            15,
        ]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_delta = [
            30,
            5,
        ]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        B = [
            15,
            3,
        ]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        velocity_matrix = np.array([a_alpha, a_beta, a_gamma, a_delta, B])
        # create time_windows based on fiber type activation for every recording
        # time_window[rec_electrode][fiber_type][start/stop]

        num_rec_electrode = len(self.ephys.array)
        fiber_type = len(self.neural_fiber_names)
        time_windows = np.zeros((num_rec_electrode, fiber_type, 2))
        for i, vel in enumerate(velocity_matrix):
            for j, length_cm in enumerate(self.distance_log):
                time_windows[j][i] = [length_cm / 100 * (1 / ii) for ii in vel]

        time_windows *= self.fs
        time_windows = np.round(time_windows).astype(int)

        return time_windows

    def calc_AUC(
        self,
        parameters=None,
        channels=None,
        window=None,
        window_units=None,
        method="mean",
        new_df=False,
    ):

        column_headers = self.stim.parameters.columns
        masterLIST = []

        if window is not None:
            if window_units is None:
                raise ValueError(
                    "User-specified limits for AUC calculation have been specified, however the units for window limits are not defined. Specify window_units in 'sec', 'ms','us' or 'samples'."
                )
            elif window_units == "samples":
                # print("Window units defined by sample #")
                start_idx = window[0]
                stop_idx = window[1]
            elif window_units == "sec":
                # print("Window units defined in seconds (sec).")
                start_idx = self._time_to_index(window[0])
                stop_idx = self._time_to_index(window[1])
            elif window_units == "ms":
                # print("Window units defined in milliseconds (ms).")
                start_idx = self._time_to_index(window[0], units="milliseconds")
                stop_idx = self._time_to_index(window[1], units="milliseconds")
            elif window_units == "us":
                # print("Window units defined in microseconds (us)")
                start_idx = self._time_to_index(window[0], units="microseconds")
                stop_idx = self._time_to_index(window[1], units="microseconds")
            else:
                raise ValueError(
                    "Unit type of AUC window not recognized. Acceptable units are 'sec', 'ms', 'us' or 'samples'."
                )
        else:
            print("Utilizing standard neural windows for calculation.")

        if parameters is not None:
            paramLIST = parameters
        else:
            paramLIST = self.stim.parameters.index.tolist()

        if channels is not None:
            if not isinstance(channels, list):
                chanLIST = [channels]
            else:
                chanLIST = channels
        else:
            chanLIST = self.ch_names

        data = self.mean(paramLIST, chanLIST)

        # Outer loop iterates through each parameter passed
        for param in data:
            mean_traces = data[param]

            # Inner loop iterates through each channel. Note, channels in chanLIST must be in order for the
            # arrays in mean_traces to correctly align with the channel.
            for idx, chan in enumerate(chanLIST):

                if isinstance(chan, int):
                    channel_name = self.ch_names[chan]
                    chanIDX = chan
                else:
                    channel_name = chan
                    chanIDX = np.where(self._ch_to_index(chan))[0].item()

                if window is None:

                    for standard_window_idx, standard_window in enumerate(
                        self.neural_window_indicies[chanIDX]
                    ):
                        start_idx = standard_window[0]
                        stop_idx = standard_window[1]
                        RMS = self._calc_RMS(
                            mean_traces[idx], window=(start_idx, stop_idx)
                        )

                        window_str = self.neural_fiber_names[standard_window_idx]
                        window_units = "Fiber type"
                        masterLIST.append(
                            [
                                RMS,
                                window_str,
                                window_units,
                                *self.stim.parameters.loc[param].tolist(),
                                channel_name,
                            ]
                        )
                else:
                    RMS = self._calc_RMS(mean_traces[idx], window=(start_idx, stop_idx))
                    window_str = str(window[0]) + " - " + str(window[1])
                    masterLIST.append(
                        [
                            RMS,
                            window_str,
                            window_units,
                            *self.stim.parameters.loc[param].tolist(),
                            channel_name,
                        ]
                    )

        data_df = pd.DataFrame(
            masterLIST,
            columns=[
                "AUC (Vs)",
                "Calculation Window",
                "Window Units",
                *column_headers,
                "Recording Electrode",
            ],
        )
        return data_df
        # if self.master_df.empty or new_df == True:
        #     self.master_df = data_df
        # else:
        #     self.master_df = pd.concat([self.master_df, data_df], ignore_index=True)

    def filter_averages(
        self,
        filter_channels=None,
        filter_median_highpass=False,
        filter_median_lowpass=False,
        filter_gaussian_highpass=False,
        filter_powerline=False,
    ):

        # First step: get an separate channels to be filtered.
        if (
            (filter_median_highpass is True)
            or (filter_median_lowpass is True)
            or (filter_gaussian_highpass is True)
            or (filter_powerline is True)
        ):
            print("Begin filtering averages")
        if type(filter_channels) == int:
            filter_channels = [filter_channels]

        if filter_channels == None:
            filter_channels = slice(None, None, None)
            target_list = np.copy(self.mean_traces)
            exclusion_list = None

        else:
            target_list = np.copy(self.mean_traces[:, filter_channels])
            length_list = np.arange(0, len(self.mean_traces[0]))
            exclusion_list = [
                value for value in length_list if value not in filter_channels
            ]

        # Second step: filter channels
        if filter_median_highpass is True:
            filtered_median_traces = np.apply_along_axis(
                lambda x: x - medfilt(x, 201), 2, target_list
            )
            target_list = filtered_median_traces

        if filter_median_lowpass is True:
            filtered_median_traces = np.apply_along_axis(
                lambda x: medfilt(x, 11), 2, target_list
            )
            target_list = filtered_median_traces

        if filter_gaussian_highpass is True:
            Wn = 4000
            s_c = Wn / self.fs
            sigma = (2 * np.pi * s_c) / np.sqrt(2 * np.log(2))
            filtered_gauss_traces = np.apply_along_axis(
                lambda x: ndimage.filters.gaussian_filter1d(x, sigma), 2, target_list
            )
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
        if (
            (filter_median_highpass is True)
            or (filter_median_lowpass is True)
            or (filter_gaussian_highpass is True)
            or (filter_powerline is True)
        ):
            print("Finished Filtering Averages")

    def plot_phase(
        self,
        channels,
        parameter,
        *args,
        method="mean",
        axis=None,
        x_lim=None,
        y_lim="auto",
        fig_size=(10, 3),
        show=True,
        fig_title=None,
        vlines=None,
        **kwargs
    ):

        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        calc_y_lim = [0, 0]
        print("Parameter index: " + str(parameter))

        for chan in channels:
            if method == "mean":
                plot_data = np.squeeze(
                    self.mean(parameter, channels=chan)[parameter] * 1e6
                )
            elif method == "median":
                plot_data = np.squeeze(
                    self.median(parameter, channels=chan)[parameter] * 1e6
                )
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'."
                )
            plot_time = self.time(parameter) * 1e3
            # compute appropriate y_limits
            if y_lim is None or y_lim == "auto":
                std_data = np.std(plot_data)
                calc_y_lim = [
                    np.min([-std_data * 6, calc_y_lim[0]]),
                    np.max([std_data * 6, calc_y_lim[1]]),
                ]
            elif y_lim == "max":
                calc_y_lim = None
            else:
                calc_y_lim = _to_numeric_array(y_lim)

            ax.plot(plot_time, plot_data, label=self.ch_names[chan], *args, **kwargs)

        ax.set_ylim(calc_y_lim)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude (uV)")
        ax.legend(loc="upper right")

        if fig_title is not None:
            ax.set_title(fig_title)

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        return _plt_show_fig(fig, ax, show)
