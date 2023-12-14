# neuro base class imports
from .base.epoch_data import _EpochData
from .base.utils.numeric import _to_numeric_array
from .utilities.ancillary_functions import check_make_dir

# other imports
import dask.bag as db
import openpyxl
import numpy as np
import warnings
import pandas as pd
from scipy import ndimage
from scipy.signal import medfilt, find_peaks, savgol_filter
import matplotlib.pyplot as plt
import sys
import dask
import os
import time


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
            warnings.warn("Electrode distance information not provided. AUC calculation using standard neural windows cannot be completed.")
            self.distance_log = [0]

        # Lists to look through for ranges
        self.neural_fiber_names = ['A-alpha', 'A-beta', 'A-gamma', 'A-delta', 'B']

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        self.ephys = self.ephys.set_ch_types(["ENG"]*self.ephys.shape[0])
        self.neural_channels = np.arange(0, self.ephys.shape[0])

        self.neural_window_indicies = self.calculate_neural_window_lengths()

        if type(self.distance_log) == list and self.distance_log != [0]:
            if type(self.neural_window_indicies) == np.ndarray and len(self.neural_window_indicies.shape) > 1:
                if self.neural_window_indicies.shape[0] != len(self.distance_log) and self.distance_log is not [0]:
                    raise ValueError("Recording lengths don't match recording channel lengths")

            elif len(self.neural_window_indicies) != len(self.distance_log):
                raise ValueError("Recording lengths don't match recording channel lengths")

        self.AUC_traces = []
        self.master_df = pd.DataFrame()
        self.parameters_dictionary = self.create_parameter_to_traces_dict()

        if distance_log == "Experimental Log.xlsx":
            self.log_path = ephys_data.base_path + distance_log
        else:
            self.log_path = distance_log

        super().__init__(ephys_data, stim_data, stim_data)

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
                   70]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_beta = [70,
                  30]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_gamma = [30,
                   15]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        a_delta = [30,
                   5]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        B = [15, 3]  # http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0004-282X2008000100033&lng=en&tlng=en
        velocity_matrix = np.array([a_alpha, a_beta, a_gamma, a_delta, B])
        # create time_windows based on fiber type activation for every recording
        # time_window[rec_electrode][fiber_type][start/stop]
        rec_electrode = len(self.neural_channels)
        fiber_type = len(self.neural_fiber_names)
        time_windows = np.zeros((rec_electrode, fiber_type, 2))
        for i, vel in enumerate(velocity_matrix):
            for j, length_cm in enumerate(self.distance_log):
                time_windows[j][i] = ([length_cm / 100 * (1 / ii) for ii in vel])

        time_windows *= self.fs
        time_windows = np.round(time_windows)

        np.round(time_windows)
        time_windows = time_windows.astype(int)
        return time_windows
    def calc_AUC(self, window=None, window_units=None, method='mean'):

        column_headers = self.stim.parameters.columns
        metadata_list = [self.stim.parameters[p].to_numpy() for p in column_headers]
        channelNAMES = np.array(self.ephys.ch_names)[self.neural_channels]
        master_list = []

        self._avg_data_for_AUC(method=method)

        if window is not None:
            if window_units is None:
                raise ValueError("User-specified limits for AUC calculation have been specified, however the units for window limits are not defined. Specify window_units in 'sec', 'ms','us' or 'samples'.")
            elif window_units == 'samples':
                print('Window units defined by sample #')
                start_idx = window[0]
                stop_idx = window[1]
            elif window_units == 'sec':
                print('Window units defined in seconds (sec).')
                start_idx = self.ts_data._time_to_index(window[0])
                stop_idx = self.ts_data._time_to_index(window[1])
            elif window_units == 'ms':
                print('Window units defined in milliseconds (ms).')
                start_idx = self.ts_data._time_to_index(window[0], units='milliseconds')
                stop_idx = self.ts_data._time_to_index(window[1], units='milliseconds')
            elif window_units == 'us':
                print('Window units defined in microseconds (us)')
                start_idx = self.ts_data._time_to_index(window[0], units='microseconds')
                stop_idx = self.ts_data._time_to_index(window[1], units='microseconds')
            else:
                raise ValueError("Unit type of AUC window not recognized. Acceptable units are 'sec', 'ms', 'us' or 'samples'.")
        else:
            print('Utilizing standard neural windows for calculation.')
            windowNAMES = self.neural_fiber_names
            print(windowNAMES)

        #Outer Loop iterates through each parameter/pulse train within the TDT block
        for stimIDX, avgDATA in enumerate(self.AUC_traces):

            #Inner loop iterates through each active recording electrode
            for chanIDX, trace in enumerate(avgDATA):

                metadata = [*[p[stimIDX] for p in metadata_list], channelNAMES[chanIDX]]
                if window is None:
                    RMS = []
                    for standard_window_idx, standard_window in enumerate(self.neural_window_indicies[chanIDX]):
                        start_idx = standard_window[0]
                        stop_idx = standard_window[1]
                        RMS.append(self._calc_RMS(trace, window=(start_idx, stop_idx)))

                    for idx, vals in enumerate(RMS):
                        master_list.append([vals, windowNAMES[idx], *metadata])
                else:
                    RMS = self._calc_RMS(trace, window=(start_idx, stop_idx))
                    customNAME = 'Custom: ' + str(window[0]) + ' to ' + str(window[1]) + ' ' + window_units
                    master_list.append([RMS, customNAME, *metadata])

        new_df = pd.DataFrame(master_list, columns=['AUC (Vs)', "Calculation Window", *column_headers, 'Recording Electrode'])

        if self.master_df.empty:
            self.master_df = new_df
        else:
            self.master_df = pd.concat([self.master_df, new_df], ignore_index=True)

    def _avg_data_for_AUC(self, parameter_index=None, method='mean'):
        # data frame with onset and offsets
        self.df_epoc_idx = self.epoc_index()

        if parameter_index is None:
            parameter_index = self.df_epoc_idx.index

        # print(toc-tic, "elapsed")
        #tic = time.perf_counter()
        bag_params = db.from_sequence(parameter_index.map(lambda x: self.dask_array(x)))

        #print(parameter_index)
        #print("Begin Averaging Data")
        if method == 'mean':
            print('Begin averaging data by mean.')
            self.AUC_traces = np.squeeze(dask.compute(bag_params.map(lambda x: np.nanmean(x, axis=0)).compute()))
        elif method == 'median':
            print('Begin averaging data by median.')
            self.AUC_traces = np.squeeze(dask.compute(bag_params.map(lambda x: np.nanmedian(x, axis=0)).compute()))
        print("Finished Averaging Data")
        #toc = time.perf_counter()
        #print(toc - tic, "seconds elapsed")

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

        if filter_channels == None:
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

    def create_parameter_to_traces_dict(self):
        return dict(zip(self.stim.parameters.index, range(len(self.stim.parameters.index))))

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
                    for fiber_onsets in self.neural_window_indicies[rec_idx]:
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
