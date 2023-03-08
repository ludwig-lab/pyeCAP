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
class EMG(_EpochData):
    """
    This class represents EMG data
    """

    def __init__(self, ephys_data, stim_data):
        """
        Constructor for the EMG class.

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

        # Lists to look through for ranges
        self.emg_window = ["Total EMG"]

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        self.mean_traces = []
        self.master_df = pd.DataFrame()
        self.parameters_dictionary = self.create_parameter_to_traces_dict()

        super().__init__(ephys_data, stim_data, stim_data)

        self.EMG_window_indicies = self.calculate_emg_window_lengths()

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

    def calc_AUC_method(self, signal, recording_idx, window_type, calculation_type, metadata, plot_AUCs=False,
                        save_path=None):

        if calculation_type == "RMS":
            if window_type.endswith("neural"):
                window_onset_idx = self.neural_window_indicies
            if window_type.endswith("EMG"):
                window_onset_idx = self.EMG_window_indicies

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

        if len(self.neural_window_indicies) != len(self.neural_channels):
            sys.exit("amounts of neural channels don't add up")

        window_nomenclature = self.emg_window
        recording_channels = self.emg_channels

        # will iterate through each condition
        for signal_idx, signal_chain in enumerate(self.mean_traces):

            # will iterate through recording electrode
            for rec_idx, rec_chain in enumerate(signal_chain[recording_channels]):
                # iterate through each fiber type
                # plotting_metadata = [condition_list[signal_idx], channel_list[signal_idx], amplitude_list[signal_idx],
                #                      rec_idx]

                ch_types = []
                for m in self.ephys.metadata:
                    if 'types' in m:
                        if len(m['ch_types']) > 1:
                            ch_types.extend(m['ch_types'])
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

    def average_data(self, parameter_index=None):
        # data frame with onset and offsets
        self.df_epoc_idx = self.epoc_index()

        if parameter_index is None:
            parameter_index = self.df_epoc_idx.index

        # import time
        #
        #
        # tic = time.perf_counter()
        # reshaped_traces = db.from_sequence(parameter_index.map(lambda x: self.dask_array(x))).compute()
        # mean_traces_dasked = [np.mean(x, axis=0) for x in reshaped_traces]
        # bagged_traces = db.from_sequence(mean_traces_dasked)
        # self.mean_traces = dask.compute(bagged_traces.compute())
        # toc = time.perf_counter()
        #
        # print(toc-tic, "elapsed")

        bag_params = db.from_sequence(parameter_index.map(lambda x: self.dask_array(x)))
        print("Begin Averaging Data")
        self.mean_traces = np.squeeze(dask.compute(bag_params.map(lambda x: np.mean(x, axis=0)).compute()))
        print("Finished Averaging Data")


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
