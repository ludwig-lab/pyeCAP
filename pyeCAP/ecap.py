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
            self.distance_log = [0]

        # Lists to look through for ranges
        self.neural_fiber_names = ['A-alpha', 'A-beta', 'A-gamma', 'A-delta', 'B']
        self.emg_window = ["Total EMG"]

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate

        if 'EMG' in self.ephys.types:
            self.emg_channels = np.arange(0, self.ephys.shape[0])[self.ephys._ch_num_mask_by_type['EMG']]
        elif 'ENG' in self.ephys.types:
            self.neural_channels = np.arange(0, self.ephys.shape[0])[self.ephys._ch_num_mask_by_type['ENG']]
        else:
            warnings.warn("Neural channels not implicitly stated. Assuming all channels are neural recordings")
            self.ephys = self.ephys.set_ch_types(["ENG"]*self.ephys.shape[0])
            self.neural_channels = np.arange(0, self.ephys.shape[0])

        #self.neural_window_indicies = self.calculate_neural_window_lengths()

        if type(self.distance_log) == list and self.distance_log != [0]:
            if type(self.neural_window_indicies) == np.ndarray and len(self.neural_window_indicies.shape) > 1:
                if self.neural_window_indicies.shape[0] != len(self.distance_log) and self.distance_log is not [0]:
                    raise ValueError("Recording lengths don't match recording channel lengths")

            elif len(self.neural_window_indicies) != len(self.distance_log):
                raise ValueError("Recording lengths don't match recording channel lengths")

        self.mean_traces = []
        self.master_df = pd.DataFrame()
        self.parameters_dictionary = self.create_parameter_to_traces_dict()

        if distance_log == "Experimental Log.xlsx":
            self.log_path = ephys_data.base_path + distance_log
        else:
            self.log_path = distance_log

        super().__init__(ephys_data, stim_data, stim_data)

        if 'EMG' in self.ephys.types:
            self.EMG_window_indicies = self.calculate_emg_window_lengths()  # [[[75, 600]], [[75, 600]], [[75, 600]]]

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

    def calculate_emg_window_lengths(self):
        # Find pulse width. Multiply by six for onset to account for capacitive discharge. Go to end of sample for offset, or 1/frequency, whichever is shorter
        pw = self.stim.parameters.iloc[0]['pulse duration (ms)'] / 1000
        onset = int(pw * 6 * self.fs)
        offset1 = int(self.fs / self.stim.parameters.iloc[0]['frequency (Hz)']) - 1
        offset2 = len(self.dask_array(self.stim.parameters.index[0]))

        if offset1 < offset2:
            return [[[onset, offset1]] for c in self.emg_channels]
        else:
            return [[[onset, offset2]] for c in self.emg_channels]

    def calc_AUC(self, params=None, channels=None, window=None, window_units=None, method='mean'):

        column_headers = self.stim.parameters.columns
        self._avg_data_for_AUC(method=method)

        if window is not None:
            if window_units is None:
                raise ValueError("User-specified limits for AUC calculation have been specified, however the units for window limits are not defined. Specify window_units in 'ms' or 'samples'.")
            elif window_units == 'ms':
                print('Window units defined in milliseconds (ms)')
                #Need to convert the time in ms to sample# based on sampling freq
            elif window_units == 'samples':
                print('Window units defined by sample #')
                start_idx = window[0]
                stop_idx = window[1]
            else:
                raise ValueError("Unit type of AUC window not recognized. Acceptable units are either in 'ms' or 'samples'.")

        calculated_Values_list = []

        for stimIDX, signal in enumerate(self.AUC_traces):
            for chanIDX,trace in enumerate(signal):
                pass

        #utilize the neural windows steph previously used if there's no user specified input
        #window units to be defined in time or samples?  Could prompt user for that info

    def calc_AUC_method(self, signal, recording_idx, window_type, calculation_type, metadata, plot_AUCs=False,
                        save_path=None):
        if calculation_type == "RMS":
            if window_type.endswith("neural"):
                window_onset_idx = self.neural_window_indicies
            if window_type.endswith("EMG"):
                window_onset_idx = self.EMG_window_indicies
                print('EMG_window_indicies: ' + str(self.EMG_window_indicies))

            current_list = []
            self.windows = window_onset_idx


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

            def find_max(signal, start, stop):
                """
                Finds idx of maxima
                :param signal:
                :param start:
                :param stop:
                :return:
                """
                jitter_percentage = .01

                if stop > len(signal):
                    stop = len(signal) - 1

                peak_distance = (stop - start) / 10

                # distance in find_peaks must be at least 1. Check to makes sure:
                if peak_distance < 1:
                    peak_distance = 1

                peak_idx, _ = find_peaks(signal[start:stop], distance=peak_distance)
                peak_idx = [i + start for i in peak_idx]
                final_idx = []
                for idx, val in enumerate(peak_idx):
                    if start * (1 + jitter_percentage) < val < stop:
                        final_idx.append(val)
                maxima = [signal[i] for i in final_idx]

                if not maxima:
                    return stop
                max_value = max(maxima)
                max_idx = np.where(signal[start:stop] == max_value)
                max_idx = [i + start for i in max_idx]

                # returns the index of max location. Max index has been stored as a tuple, so the [0][0] is necessary
                return max_idx[0][0]

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

            def relevant_AUC(signal, recording_num, plot_AUC=False, save_path=None, recording_type='neural'):
                fiber_maxima = [find_max(signal, win_indicies[0], win_indicies[1]) for win_indicies in
                                self.neural_window_indicies[recording_idx]]

                # print(fiber_maxima)
                fiber_minima = [find_minima(signal, win_indicies[0], win_indicies[1], fiber_maxima[idx]) for
                                idx, win_indicies in enumerate(self.neural_window_indicies[recording_idx])]
                # print(fiber_minima)

                # Calculate AUC based on the maxima and their corresponding minima
                # Step one is integrating the area of the signal from point one to point two
                AUC1 = [np.trapz(signal[min_idx[0]:min_idx[1]]) for min_idx in fiber_minima]
                AUC1 = np.array(AUC1)
                # Step two is integrating the area underneath that section, to be subtracted later
                AUC2 = [np.trapz(np.linspace(signal[min_idx2[0]], signal[min_idx2[1]], min_idx2[1] - min_idx2[0]))
                        for
                        min_idx2
                        in
                        fiber_minima]
                AUC2 = np.array(AUC2)
                real_AUC = AUC1 - AUC2
                real_AUC = [0 if i < 0 else i for i in real_AUC]

                if plot_AUC:
                    min_y, max_y = determine_plotting_boundaries(signal, fiber_minima, fiber_maxima)
                    relative_ts = np.arange(0, len(signal)) / self.fs
                    # smoothed_curve = savgol_filter(signal, 7, 3)
                    # my_diff = np.diff(np.sign(np.diff(smoothed_curve)))

                    fig, ax = plt.subplots(1, figsize=(15, 15))

                    title = "{} {}, Amp {}, ENG {}".format(*metadata)
                    fig.suptitle(title)
                    # some cases have such close min and max, that it's best not windowed
                    if abs(max_y - min_y) > 1e-9:
                        ax.set_ylim(min_y, max_y)
                    ax.set_xlim(0, .015)
                    ax.plot(relative_ts, signal)
                    # ax.vlines(21/FS, min(signal), max(signal), color='red')
                    # ax.vlines(21 / FS + 10 / 70 / 100, min(signal), max(signal), color='red')
                    ax.vlines(self.neural_window_indicies[recording_num] / self.fs,
                              min(signal),
                              max(signal),
                              linewidth=3)
                    for idx, val in enumerate(fiber_maxima):
                        ax.scatter(val / self.fs,
                                   signal[fiber_maxima[idx]],
                                   marker='o',
                                   color='C1',
                                   s=150)
                        ax.scatter(fiber_minima[idx][0] / self.fs,
                                   signal[fiber_minima[idx][0]],
                                   marker='o',
                                   color='C6', s=150)
                        ax.scatter(fiber_minima[idx][1] / self.fs,
                                   signal[fiber_minima[idx][1]],
                                   marker='o',
                                   color='C6', s=150)
                    # ax.plot(relative_ts, smoothed_curve, linewidth=2)
                    # ax.plot(relative_ts,my_diff * .000005 - .00005)
                    if save_path is None:
                        meta_data = self.ephys.metadata
                        if len(meta_data) > 1:
                            base_path = meta_data[0]['file_location']
                        else:
                            base_path = meta_data[0]['file_location']
                        base_path += '/AUC Plots/'
                        check_make_dir(base_path)
                    else:
                        base_path = save_path
                    save_path = base_path + "/" + recording_type + " " + title + ".png"
                    plt.savefig(save_path)
                    plt.close('all')

                return real_AUC

            current_list = relevant_AUC(signal, recording_idx, plot_AUCs, save_path)
            return current_list

    def calculate_AUC(self, window_type="standard_neural", analysis_method="RMS", plot_AUC=False, save_path=None):
        # todo: other parameters
        """
        save_path:
            location to save plotting of AUCs
        window_type:
            "standard_neural": collects all data within entire range of neural window based on fiber type
            "standard_EMG",
            "dynamic_neural" : detects max peak and leading/lagging minima; up to 25% out of current window for leading.
            "dynamic_EMG" : detects onset and offset
        analysis_method:
            "RMS" for root mean square. "Integral" for integral
        plot_AUC :
            Can chose to plot calculation of AUCs
        """

        params = self.stim.parameters.columns
        # list of specifics: channels, condition, amplitude, stim type, etc...
        info_list = [self.stim.parameters[p].to_numpy() for p in params]

        # # Get pertinent info from dataframe
        # channel_list = self.stim.parameters["channel"].to_numpy()
        # condition_list = self.stim.parameters["condition"].to_numpy()
        # amplitude_list = self.stim.parameters["pulse amplitude (μA)"].to_numpy()
        # stim_type_list = self.stim.parameters["stimulation type"].to_numpy()

        # create empty array to store calculated values in
        master_list = []

        #if len(self.neural_window_indicies) != len(self.neural_channels):
        #    sys.exit("amounts of neural channels don't add up")

        if window_type.endswith("EMG"):
            window_nomenclature = self.emg_window
            recording_channels = self.emg_channels
            recording_channel_names = np.array(self.ephys.ch_names)[recording_channels]
        elif window_type.endswith("neural"):
            window_nomenclature = self.neural_fiber_names
            recording_channels = self.neural_channels
            recording_channel_names = np.array(self.ephys.ch_names)[recording_channels]

        # will iterate through each condition
        for signal_idx, signal_chain in enumerate(self.mean_traces):
            # will iterate through recording electrode
            ch_types = []
            for rec_idx, rec_chain in enumerate(signal_chain[recording_channels]):
                # iterate through each fiber type
                # plotting_metadata = [condition_list[signal_idx], channel_list[signal_idx], amplitude_list[signal_idx],
                #                      rec_idx]

                for m in self.ephys.metadata:
                    #print(len(m['types']))
                    #print(m)

                    if 'types' in m:
                        if len(m['types']) > 1:
                            ch_types.extend(m['types'])
                    else:
                        ch_types.append([''] * len(signal_chain[recording_channels]))

                plotting_metadata = [*[p[signal_idx] for p in info_list], recording_channel_names[rec_idx], ch_types[rec_idx]]

                calc_values = self.calc_AUC_method(rec_chain, rec_idx, window_type, analysis_method, plotting_metadata,
                                                   plot_AUCs=plot_AUC, save_path=save_path)

                for specific_window_idx, vals in enumerate(calc_values):
                    # master_list.append(
                    #     [vals, analysis_method, window_nomenclature[specific_window_idx], amplitude_list[signal_idx],
                    #
                    master_list.append(
                        [vals, analysis_method, window_nomenclature[specific_window_idx], *plotting_metadata])

        new_df = pd.DataFrame(master_list, columns=['AUC (Vs)', 'Calculation Type', "Calculation Window", *params,
                                                    'Recording Electrode', 'Recording Type'])

        if new_df['Recording Type'][0] == '':
            new_df.drop(columns='Recording Type', inplace=True)
        # if new_df.loc[0]["Stimulation Amplitude"] < 0:
        #     new_df['Stimulation Amplitude'] = new_df['Stimulation Amplitude'].apply(lambda x: x * -1)

        if self.master_df.empty:
            self.master_df = new_df
        else:
            self.master_df = pd.concat([self.master_df, new_df], ignore_index=True)

        # change negative amplitudes to positive

    def _avg_data_for_AUC(self, parameter_index=None, method='mean'):
        # data frame with onset and offsets
        self.df_epoc_idx = self.epoc_index()

        if parameter_index is None:
            parameter_index = self.df_epoc_idx.index

        # import time
        # tic = time.perf_counter()
        # reshaped_traces = db.from_sequence(parameter_index.map(lambda x: self.dask_array(x))).compute()
        # mean_traces_dasked = [np.mean(x, axis=0) for x in reshaped_traces]
        # bagged_traces = db.from_sequence(mean_traces_dasked)
        # self.mean_traces = dask.compute(bagged_traces.compute())
        # toc = time.perf_counter()
        #
        # print(toc-tic, "elapsed")
        tic = time.perf_counter()
        bag_params = db.from_sequence(parameter_index.map(lambda x: self.dask_array(x)))

        #print(parameter_index)
        #print("Begin Averaging Data")
        if method == 'mean':
            print('Begin averaging data by mean.')
            self.AUC_traces = np.squeeze(dask.compute(bag_params.map(lambda x: np.mean(x, axis=0)).compute()))
        elif method == 'median':
            print('Begin averaging data by median.')
            self.AUC_traces = np.squeeze(dask.compute(bag_params.map(lambda x: np.median(x, axis=0)).compute()))
        print("Finished Averaging Data")
        toc = time.perf_counter()
        print(toc - tic, "elapsed")
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
