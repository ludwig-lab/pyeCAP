import pyeCAP
from tdt import read_block
import surgical_log_io
import mne
import datetime as dt
import scipy.io as sio
from scipy import signal, ndimage
from scipy.signal import find_peaks, savgol_filter, medfilt
from scipy.optimize import curve_fit
from heapq import nlargest
from ancillary_functions import calculate_neural_window_lengths, check_make_dir
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
import sys
from specific_plots import PlottingTDT
plt.style.use('seaborn-talk')




class TdtExperiment(PlottingTDT):
    def __init__(self, experiment_name, life_storage, emg_storage, storage_path, custom_surgical_log=False, tdt_lc_delay=0, **kwargs):
        # Set variables
        self.experiment_name = experiment_name
        self.life_storage = life_storage
        self.emg_storage = emg_storage
        self.storage_path = storage_path

        # Initialize some other variables held constant in our experiments.

        self.RELATIVE_TIME_RANGE = [-3E-3, 20E-3]
        self.FILT_FREQ = (60, 120)  # Notch filter frequencies
        self.BASELINE_TIME_BUFFER = 3  # time buffer before in seconds
        self.BASELINE_TIME_LENGTH = 2  # time in seconds for baseline to get recordings
        self.EXAMINE_LABCHART_CHANNELS = [1, 2]  # channel in which HR and mean BP is stored
        self.NUM_LIFE_ELECTRODES = 4
        self.NUM_EMG_ELECTRODES = 3
        self.NUM_FIBERS = 5
        self.NUM_CONFIGS = 10  # unique number of stim configs
        self.TDT_LC_DELAY = tdt_lc_delay
        # Todo: incorporate **kwargs into onboard storage.

        self.NUM_MONO = 6
        self.NUM_BI = 4
        self.LEGEND = surgical_log_io.create_legends(self.storage_path + '/Surgical Log.xlsx', **kwargs)
        self.NUM_LONG_RECORDINGS = self.NUM_CONFIGS
        self.EMG_REPORTING_AUCs = 3  # short, long, total
        self.fiber_type = 1  # 0: a-alpha 1: a-beta
        self.EMG_RESPONSE_INDEX = 0  # 0=total, 1=short 2=long
        self.fiber_names = [r'A-$\alpha$', r'A-$\beta$', r'A-$\gamma$', r'A-$\delta$', 'B']
        self.unix_fiber_names = ['A_\u03B1', 'A_\u03B2', 'A_\u03B3', 'A_\u03B4', 'B']
        self.NUM_VAGOTOMY_CONFIGS = 10  # number of stim configs post vagotomy
        self.ONSET_PERCENTAGE = .05
        self.SATURATION_PERCENTAGE = .85

        # Initialize storage variables
        self.all_amplitudes = np.array([])
        self.all_phys_response = []
        self.fs_LIFE = []
        self.fs_EMG = []
        self.tdt_scalars_key = []
        self.tdt_epocs_key = []
        self.filtered_LIFE_averages = []
        self.filtered_EMG_averages = []
        self.AUC_LIFE_calcs = []
        self.AUC_EMG_calcs = []

        # Create path variables for TDT and LC data. Will exit code if not present.
        self._find_create_input_path()

        # Create directories for storage. Also creates path for h5f storage
        self._find_create_storage()

        # Fill Surgical Log with tanks, create dictionary
        if not custom_surgical_log:
            self._read_write_surgical_log()
        else:
            warnings.warn("Warning: you will need to provide your own surgical log... Attempting to read")
            self.my_dictionary = surgical_log_io.create_dictionary(self.surgical_log_path)

    def info(self):
        return self.experiment_name

    def _find_create_input_path(self):
        initial_dirs = []
        all_files = []
        for (dir_path, dir_names, filenames) in os.walk(self.storage_path):
            initial_dirs.extend(dir_names)
            all_files.extend(filenames)
            break

        if len(initial_dirs) < 2:
            sys.exit("Could not find storage directories. Exiting")

        if "Surgical Log.xlsx" not in all_files:
            sys.exit("Surgical Log not found. Exiting")
        else:
            self.surgical_log_path = self.storage_path + "/Surgical Log.xlsx"

        if (initial_dirs[0].casefold() == "TDT" or initial_dirs[1] == "TDT") and (
                initial_dirs[0] == "LabChart" or initial_dirs[1] == "LabChart"):
            self.file_path_tdt = self.storage_path + "/TDT"
            self.file_path_lc = self.storage_path + "/LabChart/"
        else:
            sys.exit("Data does not contain both a \"TDT\" and a \"LabChart\" folder. Exiting")

        for (dir_path, dir_names, filenames) in os.walk(self.file_path_lc):
            all_files.extend(filenames)
            break
        if "STIM_HR_BP.mat" not in all_files:
            sys.exit("Labchart data has not yet been converted to 'STIM_HR_BP.mat'. Please convert. Exiting")
        else:
            self.file_path_lc = self.file_path_lc + "STIM_HR_BP.mat"

    def _find_create_storage(self):
        check_make_dir(self.experiment_name)
        self.traces_save_path = self.experiment_name + "/Filtered Traces"
        check_make_dir(self.traces_save_path)
        check_make_dir(self.experiment_name + "/AUC")
        self.save_auc_fig_path = self.experiment_name + "/AUC"
        check_make_dir(self.experiment_name + "/Min Max Analysis")
        self.min_max_save_path = self.experiment_name + "/Min Max Analysis"
        self.save_signal_h5_path = self.experiment_name + "/filter_average_signal.h5"
        self.save_auc_h5_path = self.experiment_name + "/AUC_calcs.h5"

    def _read_write_surgical_log(self):
        """Reads the surgical log to create the internal dictionary on which TDT blocks are associated with which stimulation configurations"""
        surgical_log_io.write_tanks_to_excel(self.file_path_tdt, self.surgical_log_path)
        self.my_dictionary = surgical_log_io.create_dictionary(self.surgical_log_path)

    def _create_recording_epocs(self, stim_dat):
        epocs = []
        onset = stim_dat.parameters['onset time (s)']
        offset = stim_dat.parameters['offset time (s)']

        # first, check to make sure we have the same number of onsets and offsets. If not, add the end time as offset
        if len(onset) != len(offset):
            warnings.warn("Warning: lengths of onset and offset time-points are not the same.")
            if len(onset) == len(offset) + 1:
                warnings.warn("Offset missing. Using end-time as last offset.")
                offset = np.append(stim_dat.epocs[self.tdt_epocs_key].offset,
                                                                         stim_dat.info.duration / dt.timedelta(
                                                                              seconds=1))
            else:
                warnings.warn("Checked, but they're not off by one. RIP bro")
        elif stim_dat.epocs[self.tdt_epocs_key].offset[-1] == float('inf'):
            warnings.warn("Offset listed as \"infinity\". Using end-time as last offset.")
            stim_dat.epocs[self.tdt_epocs_key]['offset'][-1] = stim_dat.info.duration / dt.timedelta(seconds=1)

        recording_start = stim_dat.info.start_date
        for i in range(len(stim_dat.epocs[self.tdt_epocs_key].onset)):
            start = recording_start + dt.timedelta(seconds=stim_dat.epocs[self.tdt_epocs_key].onset[i]) - dt.timedelta(
                seconds=self.BASELINE_TIME_BUFFER)
            stop = recording_start + dt.timedelta(seconds=stim_dat.epocs[self.tdt_epocs_key].offset[i])
            epocs.append([start, stop])

        # add time of start TDT to time of recording onset
        return epocs

    def gather_ephys(self, plot_signals=False):
        def matlab2datetime(matlab_datenum):
            day = dt.datetime.fromordinal(int(matlab_datenum))
            dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
            return day + dayfrac

        def notch_filter(freqs):
            for idx, channel in enumerate(current_LIFE_signals):
                current_LIFE_signals[idx] = mne.filter.notch_filter(channel, self.fs_LIFE, freqs, verbose=False)
            for idx, channel in enumerate(current_EMG_signals):
                current_EMG_signals[idx] = mne.filter.notch_filter(channel, self.fs_LIFE, freqs, verbose=False)

        def average_signal():
            def average_amplitude():
                for i in range(pulses_per_amplitude):
                    for ii in range(num_electrodes):
                        my_range = slice(idx_stims[i] + idx_relative_time_range[0],
                                         idx_stims[i] + idx_relative_time_range[1])
                        all_LIFE_averages[configuration_iterator][amp_idx][ii] += current_LIFE_signals[ii][
                                                                                      my_range] / pulses_per_amplitude

                        all_EMG_averages[configuration_iterator][amp_idx][ii] += current_EMG_signals[ii][
                                                                                     my_range] / pulses_per_amplitude

            nonlocal pulses_per_amplitude
            nonlocal data
            nonlocal current_amplitudes

            # Catch case where recordings end early and not all amplitudes were captured.
            if len(data.epocs[self.tdt_epocs_key].onset) < len(current_amplitudes):
                warnings.warn("Recordings stopped early. Not all amplitudes were captured")
                current_amplitudes = current_amplitudes[:len(data.epocs[self.tdt_epocs_key].onset)]
            for amp_idx, a in enumerate(current_amplitudes):
                self.STIM_FREQ = 1000 / data.scalars[tdt_scalars_key].data[0][amp_idx]  # Duration in ms to Hz
                first_onset = data.epocs[self.tdt_epocs_key].onset[amp_idx]
                first_onset_idx = int(round(first_onset * self.fs_LIFE))
                onset_iterator = int(round(self.fs_LIFE / self.STIM_FREQ))
                idx_stims = [first_onset_idx + onset_iterator * i for i in range(pulses_per_amplitude)]

                # There are times when the recording stops early. This results in current_LIFE_signals[ii] from not having enough items in the list. This statement will catch this case and pad the end of the list with NaNs
                if idx_stims[-1] + idx_relative_time_range[1] > len(current_LIFE_signals[0]):
                    warnings.warn("Warning: TDT stopped before stimulus/recording finished")
                    # Now check if we have more than half of viable recordings
                    midpoint_stop_idx = idx_stims[int(pulses_per_amplitude / 2)] + idx_relative_time_range[1]
                    if midpoint_stop_idx < len(current_LIFE_signals[0]):
                        warnings.warn("More than half of the stim sequence was captured. Averaging those values.")
                        # find the last viable recording and set it to pulses_per_amplitude
                        for idx, val_idx in enumerate(idx_stims):
                            if val_idx + idx_relative_time_range[1] > len(current_LIFE_signals[0]):
                                pulses_per_amplitude = idx
                                break
                        average_amplitude()  # average the signals across the amplitude up to the new index
                    else:
                        warnings.warn("Not enough recordings in interrupted sequence to reliably average. Placing NaNs")
                        nan_array = np.empty(idx_relative_time_range[1] - idx_relative_time_range[0])
                        nan_array[:] = np.nan
                        for ii in range(num_electrodes):
                            all_LIFE_averages[configuration_iterator][amp_idx][ii] = nan_array
                # Normal condition where recording did not end early:
                else:
                    average_amplitude()

        def filt1():
            for i in range(len(all_LIFE_averages[0])):
                for j in range(len(all_LIFE_averages[0][0])):

                    # check to make sure reading has acceptable values:
                    # Todo this can be solved with artifact rejection as well
                    if np.min(all_LIFE_averages[configuration_iterator][i][j]) < -1e6 or np.max(
                            all_LIFE_averages[configuration_iterator][i][j]) > 1e6:
                        warnings.warn(
                            "Warning: Recordings impossibly high or low (> or < 1e6) for particular configuration: {} {}, {} \nSetting them to 0".format(
                                configuration_iterator, i, j))
                        self.filtered_LIFE_averages[configuration_iterator][i][j] = np.full(
                            len(all_LIFE_averages[configuration_iterator][i][j]), np.nan)

                    # normal condition
                    else:
                        first_pass = []
                        second_pass = []
                        # first_pass.extend(mne.filter.notch_filter(all_LIFE_averages[configuration_iterator][i][j], fs, 60))
                        second_pass.extend(all_LIFE_averages[configuration_iterator][i][j] - signal.medfilt(
                            all_LIFE_averages[configuration_iterator][i][j], 201))
                        self.filtered_LIFE_averages[configuration_iterator][i][j] = ndimage.filters.gaussian_filter1d(
                            second_pass, np.std(second_pass))

                        second_pass = []
                        second_pass.extend(signal.medfilt(all_EMG_averages[configuration_iterator][i][j], 3))
                        self.filtered_EMG_averages[configuration_iterator][i][j] = second_pass - signal.medfilt(
                            second_pass, 201)

        all_LIFE_averages = np.array([])

        mat_contents = sio.loadmat(self.file_path_lc)

        configuration_iterator = 0

        # This for loop is really the meat and potatoes here. Goes through all the stimulation configurations
        for key, value in self.my_dictionary.items():
            def save_data(neural=False, phys=False):
                pass

                # h5f.close()

            def gather_phys_data():
                """
                :return: Gathers Physiological data from Labchart and appends to: self.all_phys_response
                """
                for new_idx, channel_idx in enumerate(self.EXAMINE_LABCHART_CHANNELS):

                    epoc_ind = []
                    time_shifted_epocs = []
                    for row in epocs:
                        time_shifted_epocs.append([val + self.TDT_LC_DELAY for val in row])
                    for epoc_idx, epoc_range in enumerate(time_shifted_epocs):
                        epocs_block_time = matlab2datetime(mat_contents['blocktimes'][0][0])

                        # Check to make sure first stimulation happens after first recording
                        if epoc_range[0] < epocs_block_time or epoc_range[1] < epocs_block_time:
                            sys.exit("Stimulation started before first recording. Consider checking your data.")

                        # find correct block start looking at index = 1 to make sure our current epoc is prior to to that. If it is, we found the correct block
                        # yes, the 0 index is weird, but it's a nested list inside an empty list
                        for LC_block_index, block_time in enumerate(mat_contents['blocktimes'][0]):
                            # check if start is less than next block. In this case, we found the correct block.
                            block_time = matlab2datetime(block_time)
                            if LC_block_index < len(mat_contents['blocktimes'][0]) - 1:
                                next_block_time = matlab2datetime(mat_contents['blocktimes'][0][LC_block_index + 1])
                            else:
                                next_block_time = matlab2datetime(
                                    mat_contents['blocktimes'][0][LC_block_index]) + dt.timedelta(weeks=100 * 52)
                            if epoc_range[0] > block_time and epoc_range[0] < next_block_time:
                                # this sets epoc_block_time to the human form time of the start time of the correct block
                                epoc_block_time = matlab2datetime(mat_contents['blocktimes'][0][LC_block_index])
                                time_diff_onset = dt.timedelta.total_seconds(epoc_range[0] - epoc_block_time)
                                sample_diff_onset = int(
                                    round(time_diff_onset * mat_contents['samplerate'][channel_idx][LC_block_index]))
                                onset_idx = int(
                                    sample_diff_onset + mat_contents['datastart'][channel_idx][LC_block_index])

                                # There are few cases where the labchart recording stopped and started in the middle of TDT stimulus.
                                # For these instances, only the first chunk of recording will be taken.
                                if epoc_range[1] > next_block_time:
                                    warnings.warn("Stimulus occurred within interrupted recording")
                                    offset_idx = int(mat_contents['dataend'][channel_idx][LC_block_index])
                                # Normal scenario:
                                else:
                                    time_diff_offset = dt.timedelta.total_seconds(epoc_range[1] - epoc_block_time)
                                    sample_diff_offset = int(
                                        round(
                                            time_diff_offset * mat_contents['samplerate'][channel_idx][LC_block_index]))
                                    offset_idx = int(
                                        sample_diff_offset + mat_contents['datastart'][channel_idx][LC_block_index])
                                epoc_ind.append(slice(onset_idx, offset_idx))
                                break

                    # Collect baseline prior to all stimulus
                    for i in range(len(epoc_ind)):
                        start_idx = epoc_ind[i].start
                        stop_idx = epoc_ind[i].start + int(
                            self.BASELINE_TIME_LENGTH * mat_contents['samplerate'][channel_idx][0])
                        # Check the condition where the the baseline wasn't fully recorded and therefore can not be used for comparision
                        if stop_idx > epoc_ind[i].stop:
                            warnings.warn(
                                "Couldn't establish baseline due to recording ending during baseline period. Setting values to 0")
                            current_phys_response[i][new_idx]

                        else:
                            baseline = np.array(mat_contents['data'][0][start_idx:stop_idx])
                            baseline_avg = np.nanmean(baseline)

                            trace_data = np.array(mat_contents['data'][0][epoc_ind[i]])
                            min_trace = np.nanmin(trace_data)
                            max_trace = np.nanmax(trace_data)

                            if abs(min_trace - baseline_avg) > max_trace - baseline_avg:
                                current_phys_response[i][new_idx] = (min_trace - baseline_avg)
                            else:
                                current_phys_response[i][new_idx] = (max_trace - baseline_avg)

                self.all_phys_response.append(current_phys_response)

            def plot_all_signals():
                # i iterates over amplitudes
                for i in range(len(current_amplitudes)):
                    # declare iterators for plotting windows and subwindow sizes
                    iter8r = [[0, 0], [0, 1], [1, 0], [1, 1]]
                    mini_iterator = [[0.25, 0.725, 0.2, 0.2], [0.725, 0.725, 0.2, 0.2], [0.25, 0.25, 0.2, 0.2],
                                     [0.725, 0.25, 0.2, 0.2]]
                    fig, ax = plt.subplots(2, 2)
                    fig.set_size_inches(12, 14)
                    fig.tight_layout(pad=5.0)
                    title = 'LIFE Recordings {}, {} \u03BCA'.format(meta_channel, current_amplitudes[i])
                    fig.suptitle(title, fontsize=14)

                    first_onset = data.epocs[self.tdt_epocs_key].onset[i]
                    first_onset_idx = int(round(first_onset * self.fs_LIFE))
                    stim_freq = 25
                    onset_iterator = int(round(self.fs_LIFE / stim_freq))
                    idx_stims = [first_onset_idx + onset_iterator * z for z in range(pulses_per_amplitude)]

                    # j iterates over LIFE channels
                    for j in range((len(all_LIFE_averages[0][0]))):
                        ax[tuple(iter8r[j])].plot(relative_ts, all_LIFE_averages[configuration_iterator][i][j])
                        ax[tuple(iter8r[j])].plot(relative_ts,
                                                  self.filtered_LIFE_averages[configuration_iterator][i][j])
                        ax[tuple(iter8r[j])].legend(('Raw, Average', 'Filtered, Average'), loc='lower right')
                        ax[tuple(iter8r[j])].set_title('LIFE {}'.format(j + 1))
                        ax[tuple(iter8r[j])].set_ylabel('(V)')
                        ax[tuple(iter8r[j])].set_xlabel('(s)')
                        ax[tuple(iter8r[j])].set_ylim([-.00004, .00015])

                        mini_ax = plt.axes(mini_iterator[j])

                        for k in range(len(idx_stims)):
                            my_range = slice(idx_stims[k] + idx_relative_time_range[0],
                                             idx_stims[k] + idx_relative_time_range[1])
                            mini_ax.plot(relative_ts, current_LIFE_signals[j][my_range], color='gray', linewidth=.1)
                            mini_ax.set_ylim(-.0004, .0004)
                        mini_ax.plot(relative_ts, all_LIFE_averages[configuration_iterator][i][j])
                    save_path = self.traces_save_path + "/" + "Raw and Filtered {}_{}microamp.png".format(meta_channel,
                                                                                                          -1 * int(
                                                                                                              current_amplitudes[
                                                                                                                  i]))
                    plt.savefig(save_path)
                    plt.close('all')

            path = key
            print("Configuration:", configuration_iterator + 1, "out of", len(self.my_dictionary.items()))
            data = pyeCAP.Ephys(path)
            stim_data = pyeCAP.Stim(path)
            meta_channel = value

            # Initialize all time series components
            self.fs_LIFE = data.sample_rate
            self.fs_EMG = data.sample_rate

            idx_relative_time_range = [int(round(i * self.fs_LIFE)) for i in self.RELATIVE_TIME_RANGE]
            idx_length = idx_relative_time_range[1] - idx_relative_time_range[0]
            relative_ts = np.arange(self.RELATIVE_TIME_RANGE[0], self.RELATIVE_TIME_RANGE[1], 1 / self.fs_LIFE)[0:-1]

            # pull basics from recording (LIFE data, # LIFEs, stim amplitudes, pulses per stim, stim onset/offsets)
            current_LIFE_signals = data[0:self.NUM_LIFE_ELECTRODES]
            current_EMG_signals = data[self.NUM_LIFE_ELECTRODES: self.NUM_LIFE_ELECTRODES + self.NUM_EMG_ELECTRODES]
            num_electrodes = current_LIFE_signals.shape[0]
            num_EMGs = current_EMG_signals.shape[0]

            # # if TDT scalars (amplitudes, pulses per amplitude, wavelengths, etc. aren't being read properly read.
            # # check here if they are being stored appropriately. should be the first item in the data.scalars structure
            # tdt_scalars_key = [key for key, value in data.scalars.items()]
            # tdt_scalars_key = tdt_scalars_key[0]
            # self.tdt_scalars_key = tdt_scalars_key

            current_amplitudes = [stim_data.parameters['pulse amplitude (μA)'][0][i] for i in range(len(stim_data.parameters['pulse amplitude (μA)'][0]))]
            pulses_per_amplitude = [stim_data.parameters['pulse count'][0][i] for i in range(len(stim_data.parameters['pulse amplitude (μA)'][0]))]
            epocs = stim_data.parameters[['onset time (s)', 'offset time (s)']]

            # Allocates sizes for arrays
            current_phys_response = np.zeros((len(current_amplitudes), len(self.EXAMINE_LABCHART_CHANNELS)))

            if configuration_iterator == 0:
                all_LIFE_averages = np.zeros(
                    (len(self.my_dictionary), len(current_amplitudes), num_electrodes, idx_length))
                self.filtered_LIFE_averages = np.zeros_like(all_LIFE_averages)

                all_EMG_averages = np.zeros((len(self.my_dictionary), len(current_amplitudes), num_EMGs, idx_length))
                self.filtered_EMG_averages = np.zeros_like(all_EMG_averages)

                self.all_amplitudes = np.zeros((len(self.my_dictionary), len(current_amplitudes)))

            # catch for when all current_amplitudes not present in all_amplitudes
            if len(current_amplitudes) != len(self.all_amplitudes[configuration_iterator]):
                # Catch the case when there were more than 10 stims. Resize arrays accordingly
                if len(current_amplitudes) > len(self.all_amplitudes[configuration_iterator]):
                    self.all_amplitudes.resize((len(self.all_amplitudes), len(current_amplitudes)), refcheck=False)
                    all_LIFE_averages.resize((len(self.my_dictionary), len(current_amplitudes), num_EMGs, idx_length),
                                             refcheck=False)
                    self.filtered_LIFE_averages.resize(
                        (len(self.my_dictionary), len(current_amplitudes), num_EMGs, idx_length), refcheck=False)
                    all_EMG_averages.resize((len(self.my_dictionary), len(current_amplitudes), num_EMGs, idx_length),
                                            refcheck=False)
                    self.filtered_EMG_averages.resize(
                        (len(self.my_dictionary), len(current_amplitudes), num_EMGs, idx_length), refcheck=False)

                filler_array = np.zeros(len(current_amplitudes))
                for i, val in enumerate(current_amplitudes):
                    self.all_amplitudes[configuration_iterator][i] = val
            else:
                self.all_amplitudes[configuration_iterator] = current_amplitudes

            gather_phys_data()
            notch_filter(self.FILT_FREQ)
            average_signal()
            filt1()
            if plot_signals:
                plot_all_signals()

            save_data(neural=True, phys=True)

            configuration_iterator += 1

    def calculate_AUCs(self, plot_AUCs=False):
        """
        This function will calculate the AUCs for all fiber types, as well as for EMG recordings.
        :return:
        """

        # Todo: check and load if data is unavailable.
        if len(self.filtered_EMG_averages) < 1 or len(self.filtered_LIFE_averages) < 1:
            sys.exit("No data found. Can't perform AUC calculations")

        def generate_amplitude_list(amplitudes):
            shape = amplitudes.shape
            amplitude_list = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if amplitudes[i][j] not in amplitude_list:
                        amplitude_list.append(amplitudes[i][j])
            amplitude_list = np.array(amplitude_list)
            amplitude_list.sort()
            return amplitude_list

        def tdt_find_max(signal, start, stop):
            if stop > len(signal):
                stop = len(signal) - 1
            jitter_percentage = .01

            # distance in find_peaks must be at least 1. Check to makes sure:
            my_distance = (stop - start) / 10
            if my_distance < 1:
                my_distance = 1
            peak_idx, _ = find_peaks(signal[start:stop], distance=my_distance)
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

        def tdt_find_minima(signal, start, stop, max_idx, overlap=True):
            try:
                smoothed_curve = savgol_filter(signal, 7, 3)
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
            return [min1, min2]

        def determine_boundaries(signal, fiber_minima, fiber_maxima):
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

        def tdt_relevant_AUC(signal, recording_num, plot_AUC=False, recording_type='neural'):
            window_offset = self.RELATIVE_TIME_RANGE[0]
            window_time_lengths = calculate_neural_window_lengths(self.surgical_log_path, self.fs_LIFE, window_offset)
            relative_ts = np.arange(self.RELATIVE_TIME_RANGE[0], self.RELATIVE_TIME_RANGE[1], 1 / self.fs_LIFE)[0:-1]

            # collects maxima per fiber type
            if recording_type == 'neural':
                fiber_maxima = [
                    tdt_find_max(signal, window_time_lengths[recording_num][win_len][0],
                                 window_time_lengths[recording_num][win_len][1]) for
                    win_len in range(len(window_time_lengths[0]))]

                # print(fiber_maxima)
                fiber_minima = [tdt_find_minima(signal, window_time_lengths[recording_num][win_len_min][0],
                                                window_time_lengths[recording_num][win_len_min][1],
                                                fiber_maxima[win_len_min])
                                for
                                win_len_min in range(len(window_time_lengths[0]))]
                # print(fiber_minima)

                # Calculate AUC based on the maxima and their corresponding minima
                # Step one is integrating the area of the signal from point one to point two
                AUC1 = [np.trapz(signal[min_idx[0]:min_idx[1]]) for min_idx in fiber_minima]
                AUC1 = np.array(AUC1)
                # Step two is integrating the area underneath that section, to be subtracted later
                AUC2 = [np.trapz(np.linspace(signal[min_idx2[0]], signal[min_idx2[1]], min_idx2[1] - min_idx2[0])) for
                        min_idx2
                        in
                        fiber_minima]
                AUC2 = np.array(AUC2)
                real_AUC = AUC1 - AUC2
                real_AUC = [0 if i < 0 else i for i in real_AUC]

                if plot_AUC:
                    min_y, max_y = determine_boundaries(signal, fiber_minima, fiber_maxima)
                    # smoothed_curve = savgol_filter(signal, 7, 3)
                    # my_diff = np.diff(np.sign(np.diff(smoothed_curve)))

                    fig, ax = plt.subplots(1, figsize=(15, 15))

                    title = "Config {}, Amp {}, LIFE {}".format(config, amp, rec)
                    fig.suptitle(title)
                    # some cases have such close min and max, that it's best not windowed
                    if abs(max_y - min_y) > 1e-9:
                        ax.set_ylim(min_y, max_y)
                    ax.set_xlim(0, .015)
                    ax.plot(relative_ts, signal)
                    # ax.vlines(21/FS, min(signal), max(signal), color='red')
                    # ax.vlines(21 / FS + 10 / 70 / 100, min(signal), max(signal), color='red')
                    ax.vlines(window_time_lengths[recording_num] / self.fs_LIFE + self.RELATIVE_TIME_RANGE[0],
                              min(signal),
                              max(signal),
                              linewidth=3)
                    for idx, val in enumerate(fiber_maxima):
                        ax.scatter(val / self.fs_LIFE + self.RELATIVE_TIME_RANGE[0], signal[fiber_maxima[idx]],
                                   marker='o',
                                   color='C1',
                                   s=150)
                        ax.scatter(fiber_minima[idx][0] / self.fs_LIFE + self.RELATIVE_TIME_RANGE[0],
                                   signal[fiber_minima[idx][0]],
                                   marker='o',
                                   color='C6', s=150)
                        ax.scatter(fiber_minima[idx][1] / self.fs_LIFE + self.RELATIVE_TIME_RANGE[0],
                                   signal[fiber_minima[idx][1]],
                                   marker='o',
                                   color='C6', s=150)
                    # ax.plot(relative_ts, smoothed_curve, linewidth=2)
                    # ax.plot(relative_ts,my_diff * .000005 - .00005)
                    save_path = self.min_max_save_path + "/" + recording_type + title + ".png"
                    plt.savefig(save_path)
                    plt.close('all')

            if recording_type == 'EMG':
                signal = abs(signal)
                EMG_maxima = EMG_find_max(signal, window_time_lengths[0][2][0], window_time_lengths[0][4][1])
                # print(fiber_maxima)
                EMG_minima = [
                    tdt_find_minima(signal, window_time_lengths[0][2][0], window_time_lengths[0][4][1], local_max) for
                    local_max in EMG_maxima]
                # print(fiber_minima)

                # Calculate AUC based on the maxima and their corresponding minima
                # Step one is integrating the area of the signal from point one to point two
                AUC_short = np.trapz(signal[EMG_minima[0][0]:EMG_minima[0][1]])
                AUC_long = np.trapz(signal[EMG_minima[1][0]:EMG_minima[1][1]])
                AUC_total = AUC_short + AUC_long

            if recording_type == 'neural':
                return real_AUC
            else:
                return AUC_total, AUC_short, AUC_long

        def EMG_find_max(signal, start, stop):
            if stop > len(signal):
                stop = len(signal) - 1
            jitter_percentage = .01
            peak_idx, _ = find_peaks(signal[start:stop], distance=(stop - start) / 10)
            peak_idx = [i + start for i in peak_idx]

            maxima = [signal[i] for i in peak_idx]

            if not maxima:
                return stop, stop

            max_values = nlargest(2, maxima)

            max_idx = [np.where(signal[start:stop] == i) for i in max_values]
            max_idx = [i[0] + start for i in max_idx]

            final_idx = [i[0] for i in max_idx]

            # returns the index of max location. Max index has been stored as a tuple, so the [0][0] is necessary
            return final_idx

        def save_AUCs():
            h5f = h5py.File(self.save_auc_h5_path, 'w')
            h5f.create_dataset('dataset_1', data=self.AUC_LIFE_calcs)
            h5f.create_dataset('dataset_2', data=self.AUC_EMG_calcs)
            h5f.close()

        num_stim_configs = len(self.filtered_LIFE_averages)
        num_amplitudes = len(self.filtered_LIFE_averages[0])

        self.AUC_LIFE_calcs = np.zeros(
            [num_stim_configs, num_amplitudes, self.NUM_LIFE_ELECTRODES, self.NUM_FIBERS])
        self.AUC_EMG_calcs = np.zeros(
            [num_stim_configs, num_amplitudes, self.NUM_EMG_ELECTRODES, self.EMG_REPORTING_AUCs])

        for config in range(len(self.AUC_LIFE_calcs)):
            for amp in range(len(self.AUC_LIFE_calcs[0])):
                for rec in range(len(self.AUC_LIFE_calcs[0][0])):
                    print(config, amp, rec)
                    self.AUC_LIFE_calcs[config][amp][rec] = np.array(
                        tdt_relevant_AUC(self.filtered_LIFE_averages[config][amp][rec], rec, plot_AUCs))

        for config in range(len(self.AUC_EMG_calcs)):
            for amp in range(len(self.AUC_EMG_calcs[0])):
                for rec in range(len(self.AUC_EMG_calcs[0][0])):
                    print(config, amp, rec)
                    self.AUC_EMG_calcs[config][amp][rec] = tdt_relevant_AUC(
                        self.filtered_EMG_averages[config][amp][rec], rec,
                        plot_AUC=False, recording_type='EMG')

        save_AUCs()

    def plot_AUCs(self, fiber_type):
        """
        :param fiber_type: 0: a-alpha 1: a-beta 2: a-gamma 3: a-delta 4: B
        :return: graph of AUCs for monopolar and bipolar configurations
        """

        non_vagotomy_configs = len(self.AUC_LIFE_calcs) - self.NUM_VAGOTOMY_CONFIGS

        def generate_amplitude_list(all_amplitudes):
            shape = all_amplitudes.shape
            amplitude_list = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if all_amplitudes[i][j] not in amplitude_list:
                        amplitude_list.append(all_amplitudes[i][j])
            amplitude_list = np.array(amplitude_list)
            amplitude_list.sort()
            # delete 0 amplitude
            if amplitude_list[-1] == 0:
                amplitude_list = np.delete(amplitude_list, -1)
            return amplitude_list

        def sigmoid(x, L, x0, k):
            y = L / (1 + np.exp(-k * (x - x0)))
            return y

        def sigmoid_given_y(y, L, x0, k):
            x = (k * x0 + np.log(-y / (y - L))) / k
            return x

        def rev_sigmoid(x, L, x0, k):
            y = L / (1 + np.exp(-k * (-x - x0)))
            return y

        def find_onset_and_saturation(x_vals, y_vals, onset, saturation, type="neural", my_debugger=False):
            max_y = max(y_vals)
            onset *= max_y
            saturation *= max_y

            if type == "neural":

                try:
                    p0 = [max(y_vals) * saturation, x_vals[3], 0.04]  # this is a mandatory initial guess
                    popt, pcov = curve_fit(sigmoid, x_vals, y_vals, p0,
                                           bounds=((-np.inf, 100, 0), (np.inf, 1000, np.inf)), method='dogbox')
                    # print(popt)
                    new_y = sigmoid(x_vals, *popt)
                    val1 = self.ONSET_PERCENTAGE * (max(new_y) - min(new_y)) + min(new_y)
                    val2 = self.SATURATION_PERCENTAGE * (max(new_y) - min(new_y)) + min(new_y)

                    x_val1 = sigmoid_given_y(val1, *popt)
                    x_val2 = sigmoid_given_y(val2, *popt)

                except:
                    return float('Nan'), float('Nan')

            if type == "phys":
                # print(x_vals, "\n", y_vals)
                p = np.polyfit(x_vals, y_vals, 4)
                f = np.poly1d(p)
                xp = np.linspace(min(x_vals), max(x_vals), 10000)
                new_y = f(xp)
                if my_debugger:
                    plt.plot(xp, new_y, '--')
                val1 = max(new_y) - self.ONSET_PERCENTAGE * abs(max(new_y) - min(new_y))
                val2 = max(new_y) - self.SATURATION_PERCENTAGE * abs(max(new_y) - min(new_y))

                index1 = (np.abs(new_y - val1)).argmin()
                index2 = (np.abs(new_y - val2)).argmin()

                x_val1 = xp[index1]
                x_val2 = xp[index2]

            #   plt.plot(x_vals, new_y, 'r-')
            #   plt.plot(x_vals, y_vals, 'b-')

            return x_val1, x_val2

        sorted_amplitudes = generate_amplitude_list(self.all_amplitudes)

        LIFE_auc_sorted = {}
        for idx_config, config in enumerate(self.all_amplitudes[:len(self.all_amplitudes) - self.NUM_VAGOTOMY_CONFIGS]):
            if idx_config < self.NUM_CONFIGS:
                LIFE_auc_sorted[idx_config] = {}
                for amp_idx, amp_val in enumerate(config):
                    if amp_val != 0:
                        LIFE_auc_sorted[idx_config][amp_val] = [
                            self.AUC_LIFE_calcs[idx_config][amp_idx][LIFE_idx][fiber_type] for LIFE_idx in
                            range(len(self.AUC_LIFE_calcs[0][0]))]
            else:
                current_config = idx_config % self.NUM_CONFIGS
                for amp_idx, amp_val in enumerate(config):  # j val
                    if amp_val in LIFE_auc_sorted[current_config] and amp_val != 0:
                        LIFE_auc_sorted[current_config][amp_val] = np.append(
                            LIFE_auc_sorted[current_config][amp_val], [
                                self.AUC_LIFE_calcs[idx_config][amp_idx][i][fiber_type] for i in
                                range(len(self.AUC_LIFE_calcs[0][0]))])
                    elif amp_val != 0:
                        LIFE_auc_sorted[current_config][amp_val] = [
                            self.AUC_LIFE_calcs[idx_config][amp_idx][LIFE_idx][fiber_type] for LIFE_idx in
                            range(len(self.AUC_LIFE_calcs[0][0]))]

        phys_sorted = {}
        for idx_config, config in enumerate(self.all_amplitudes[:self.NUM_LONG_RECORDINGS]):
            if idx_config < self.NUM_CONFIGS:
                phys_sorted[idx_config] = {}
                for amp_idx, amp_val in enumerate(config):
                    if amp_val != 0:
                        phys_sorted[idx_config][amp_val] = [i for i in self.all_phys_response[idx_config][amp_idx]]

        df_phys = pd.DataFrame.from_dict(phys_sorted)
        df_phys = df_phys.sort_index()

        df = pd.DataFrame.from_dict(LIFE_auc_sorted)

        median_df = df
        for conf in range(self.NUM_CONFIGS):
            for amp in sorted_amplitudes:
                median_df[conf][amp] = np.median(df[conf][amp])
        median_df = median_df.sort_index()

        EMG_auc_sorted = {}
        for idx_config, config in enumerate(self.all_amplitudes[:self.NUM_LONG_RECORDINGS]):
            if idx_config < self.NUM_CONFIGS:
                EMG_auc_sorted[idx_config] = {}
                for amp_idx, amp_val in enumerate(config):
                    if amp_val != 0:
                        EMG_auc_sorted[idx_config][amp_val] = [self.AUC_EMG_calcs[idx_config][amp_idx][muscle] for
                                                               muscle in
                                                               range(len(self.AUC_EMG_calcs[0][0]))]

        df_EMG = pd.DataFrame.from_dict(EMG_auc_sorted)
        df_EMG = df_EMG.sort_index()

        df_average_EMG = df_EMG
        for conf in range(self.NUM_CONFIGS):
            for amp in df_average_EMG[conf].keys():
                # check fo nan
                if df_average_EMG[conf][amp] == df_average_EMG[conf][amp]:
                    df_average_EMG[conf][amp] = (np.mean(df_average_EMG[conf][amp], axis=0))  # [EMG_response_index]
        df_average_EMG = df_average_EMG.sort_index()

        fig, ax = plt.subplots(6, 2, sharex=True, sharey="row", figsize=(30, 15))

        # fig.suptitle(fiber_names[fiber_type] + " Dose Response Curves", fontsize=20)
        # plt.setp(ax, xlim=(0, 2000))

        ax[0][0].set_title("Monopolar Stimulation, {} Response".format(self.fiber_names[fiber_type]))
        ax[1][0].set_title("Monopolar Stimulation, HR Response")
        ax[2][0].set_title("Monopolar Stimulation, BP Response")
        ax[3][0].set_title("Monopolar Stimulation, Average EMG Response")
        ax[4][0].set_title("Monopolar Stimulation, Short-Component, EMG Response")
        ax[5][0].set_title("Monopolar Stimulation, Long-Component, EMG Response")

        ax[0][1].set_title("Bipolar Stimulation, {} Response".format(self.fiber_names[fiber_type]))
        ax[1][1].set_title("Bipolar Stimulation, HR Response")
        ax[2][1].set_title("Bipolar Stimulation, BP Response")
        ax[3][1].set_title("Bipolar Stimulation, Average EMG Response")
        ax[4][1].set_title("Bipolar Stimulation, Short-Component, EMG Response")
        ax[5][1].set_title("Bipolar Stimulation, Long-Component, EMG Response")

        ax[0][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[0][0].set_ylabel(r'AUC ($V \cdot s$)')

        ax[1][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[1][0].set_ylabel('\u0394 HR')

        ax[2][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[2][0].set_ylabel('\u0394 MAP')

        ax[3][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[3][0].set_ylabel(r'AUC ($V \cdot s$)')

        ax[4][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[4][0].set_ylabel(r'AUC ($V \cdot s$)')

        ax[5][0].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[5][0].set_ylabel(r'AUC ($V \cdot s$)')

        ax[0][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[0][1].set_ylabel(r'AUC ($V \cdot s$)')

        ax[1][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[1][1].set_ylabel('\u0394 HR')

        ax[2][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[2][1].set_ylabel('\u0394 MAP')

        ax[3][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[3][1].set_ylabel(r'AUC ($V \cdot s$)')

        ax[4][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[4][1].set_ylabel(r'AUC ($V \cdot s$)')

        ax[5][1].set_xlabel(r'Stimulation Amplitude ($\mu$A)')
        ax[5][1].set_ylabel(r'AUC ($V \cdot s$)')

        # ENG Mono
        mono_iterator = 0
        ENG_mono_thresh = []
        for config in median_df.columns.values[0:self.NUM_MONO]:
            phys_data_per_config = median_df[config][:].astype(np.double)
            series_mask = np.isfinite(phys_data_per_config)

            x_vals = -1 * median_df.index.values[series_mask]
            y_vals = pd.Series.tolist(phys_data_per_config[series_mask])
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[mono_iterator][0].plot(x_vals, y_vals)
            ENG_mono_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")

        mono_iterator += 1

        # HR Mono
        HR_mono_thresh = []
        for config in df_phys.columns.values[0:self.NUM_MONO]:
            phys_index = 0  # 0 for HR, 1 for BP

            phys_data_per_config = df_phys[config].tolist()
            phys_response = [row for row in phys_data_per_config]

            series_mask = []
            for i in range(len(phys_response)):
                if phys_response[i] != phys_response[i]:
                    series_mask.append(False)
                elif phys_response[i][phys_index] is not phys_response[i][phys_index]:
                    series_mask.append(False)
                else:
                    series_mask.append(True)

            x_vals = -1 * df_phys.index.values[series_mask]
            y_vals = [row[phys_index] for row in np.array(phys_response)[series_mask]]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[mono_iterator][0].plot(x_vals, y_vals)
            HR_mono_thresh.append(
                find_onset_and_saturation(x_vals, y_vals, self.ONSET_PERCENTAGE, self.SATURATION_PERCENTAGE,
                                          type="phys"))
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")
        mono_iterator += 1

        # BP Mono
        MAP_mono_thresh = []
        for config in df_phys.columns.values[0:self.NUM_MONO]:
            phys_index = 1  # 0 for HR, 1 for BP

            phys_data_per_config = df_phys[config].tolist()
            phys_response = [row for row in phys_data_per_config]

            series_mask = []
            for i in range(len(phys_response)):
                if phys_response[i] != phys_response[i]:
                    series_mask.append(False)
                elif phys_response[i][phys_index] is not phys_response[i][phys_index]:
                    series_mask.append(False)
                else:
                    series_mask.append(True)

            x_vals = -1 * df_phys.index.values[series_mask]
            y_vals = [row[phys_index] for row in np.array(phys_response)[series_mask]]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[mono_iterator][0].plot(x_vals, y_vals)
            MAP_mono_thresh.append(
                find_onset_and_saturation(x_vals, y_vals, self.ONSET_PERCENTAGE, self.SATURATION_PERCENTAGE,
                                          type="phys"))
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")
        mono_iterator += 1

        # EMG Mono average
        EMG_mono_average_thresh = []
        EMG_response_index = 0
        for config in df_average_EMG.columns.values[0:self.NUM_MONO]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)
            EMG_mono_average_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[mono_iterator][0].plot(x_vals, y_vals)
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")
        mono_iterator += 1

        # EMG Mono short
        EMG_mono_short_thresh = []
        for config in df_average_EMG.columns.values[0:self.NUM_MONO]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index + 1] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[mono_iterator][0].plot(x_vals, y_vals)
            EMG_mono_short_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")
        mono_iterator += 1

        # EMG Mono Long
        EMG_mono_long_thresh = []
        for config in df_average_EMG.columns.values[0:self.NUM_MONO]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index + 2] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[mono_iterator][0].plot(x_vals, y_vals)
            EMG_mono_long_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[mono_iterator][0].legend((self.LEGEND[0]), loc="lower right")
        mono_iterator += 1

        bi_iterator = 0
        # ENG Bipolar
        ENG_bi_thresh = []
        for config in median_df.columns.values[self.NUM_MONO:]:
            phys_data_per_config = median_df[config][:].astype(np.double)
            series_mask = np.isfinite(phys_data_per_config)

            x_vals = -1 * median_df.index.values[series_mask]
            y_vals = pd.Series.tolist(phys_data_per_config[series_mask])
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[bi_iterator][1].plot(x_vals, y_vals)
            ENG_bi_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        # HR Bipolar
        HR_bi_thresh = []
        for config in df_phys.columns.values[self.NUM_MONO:]:
            phys_index = 0  # 0 for HR, 1 for BP

            phys_data_per_config = df_phys[config].tolist()
            phys_response = [row for row in phys_data_per_config]

            series_mask = []
            for i in range(len(phys_response)):
                if phys_response[i] != phys_response[i]:
                    series_mask.append(False)
                elif phys_response[i][phys_index] is not phys_response[i][phys_index]:
                    series_mask.append(False)
                else:
                    series_mask.append(True)

            x_vals = -1 * df_phys.index.values[series_mask]
            y_vals = [row[phys_index] for row in np.array(phys_response)[series_mask]]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[bi_iterator][1].plot(x_vals, y_vals)
            HR_bi_thresh.append(
                find_onset_and_saturation(x_vals, y_vals, self.ONSET_PERCENTAGE, self.SATURATION_PERCENTAGE,
                                          type="phys"))
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        # MAP Bipolar
        MAP_bi_thresh = []
        for config in df_phys.columns.values[self.NUM_MONO:]:
            phys_index = 1  # 0 for HR, 1 for BP

            phys_data_per_config = df_phys[config].tolist()
            phys_response = [row for row in phys_data_per_config]

            series_mask = []
            for i in range(len(phys_response)):
                if phys_response[i] != phys_response[i]:
                    series_mask.append(False)
                elif phys_response[i][phys_index] is not phys_response[i][phys_index]:
                    series_mask.append(False)
                else:
                    series_mask.append(True)

            x_vals = -1 * df_phys.index.values[series_mask]
            y_vals = [row[phys_index] for row in np.array(phys_response)[series_mask]]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[bi_iterator][1].plot(x_vals, y_vals)
            MAP_bi_thresh.append(
                find_onset_and_saturation(x_vals, y_vals, self.ONSET_PERCENTAGE, self.SATURATION_PERCENTAGE,
                                          type="phys"))
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        # EMG Bipolar average
        EMG_bi_average_thresh = []
        for config in df_average_EMG.columns.values[self.NUM_MONO:]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)
            EMG_bi_average_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[bi_iterator][1].plot(x_vals, y_vals)
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        # EMG Bipolar short
        EMG_bi_short_thresh = []
        for config in df_average_EMG.columns.values[self.NUM_MONO:]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index + 1] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)
            ax[bi_iterator][1].plot(x_vals, y_vals)
            EMG_bi_short_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        # EMG Bipolar long
        EMG_bi_long_thresh = []
        for config in df_average_EMG.columns.values[self.NUM_MONO:]:
            phys_data_per_config = df_average_EMG[config].to_numpy()
            series_mask = [i == i for i in phys_data_per_config]
            final_series = []
            plt_amps = []
            for idx, row in enumerate(phys_data_per_config):
                if not isinstance(row, float):
                    final_series.append(row)
                    plt_amps.append(-1 * (df_average_EMG.index.values)[idx])
            x_vals = plt_amps
            y_vals = [row[EMG_response_index + 2] for row in final_series]
            x_vals = np.flip(x_vals)
            y_vals = np.flip(y_vals)

            ax[bi_iterator][1].plot(x_vals, y_vals)
            EMG_bi_long_thresh.append(find_onset_and_saturation(x_vals, y_vals, .1, .85))
            ax[bi_iterator][1].legend((self.LEGEND[1]), loc="lower right")
        bi_iterator += 1

        plt.tight_layout()
        fig_save_title = self.save_auc_fig_path + "/" + "{0} {1} DRC.png".format(self.experiment_name,
                                                                                 self.unix_fiber_names[fiber_type])
        plt.savefig(fig_save_title)


if __name__ == '__main__':
    import pickle

    # """20200113"""
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20200113'
    # STORAGE_PATH = "D:\\20200113_SLB_ImtheraVNS"
    # plot_sigs = False
    # exp_20200113 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, mono=6, bi=4)
    # exp_20200113.gather_ephys(plot_signals=plot_sigs)
    # exp_20200113.calculate_AUCs(plot_AUCs=plot_sigs)
    # exp_20200113 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH)
    # exp_20200113.plot_AUCs(0)
    # exp_20200113.plot_AUCs(1)
    # exp_20200113.plot_one_mono_all_bi(mono_contact_num=6, fiber_type=0)
    # exp_20200113.plot_one_mono_all_bi(mono_contact_num=6, fiber_type=1)
    # pickle.dump(exp_20200113, open("20200113/20200113.p", "wb"))

    """20191119"""
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20191119'
    # STORAGE_PATH = "D:\\20191119_JaredNess_ImtheraVNS"
    # plot_sigs = False
    # custom_delay = dt.timedelta(hours=1, seconds=43395/1000)
    # exp_20191119 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, tdt_lc_delay=custom_delay, mono=6, bi=4)
    # exp_20191119.gather_ephys(plot_signals=plot_sigs)
    # exp_20191119.calculate_AUCs(plot_AUCs=plot_sigs)
    # exp_20191119.plot_AUCs(0)
    # exp_20191119.plot_AUCs(1)
    # exp_20191119.plot_one_mono_all_bi(mono_contact_num=2, fiber_type=0)

    """20191126"""
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20191126'
    # STORAGE_PATH = "D:\\20191126_SLB_LivaNovaAndImtheraVNS"
    # plot_sigs = False
    # exp_20191126 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, mono=6, bi=4)
    # exp_20191126.gather_ephys(plot_signals=plot_sigs)
    # exp_20191126.calculate_AUCs(plot_AUCs=plot_sigs)
    # exp_20191126.plot_AUCs(0)
    # exp_20191126.plot_AUCs(1)
    # pickle.dump(exp_20191126, open("20191126/20191126.p", "wb"))
    #
    # """20191216"""
    #
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20191216'
    # STORAGE_PATH = "D:/20191216_SLB_ImtheraVNS"
    # plot_sigs = True
    # exp_20191216 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, mono=6, bi=4)
    # exp_20191216.gather_ephys(plot_sigs)
    # exp_20191216.calculate_AUCs(plot_AUCs=plot_sigs)
    # exp_20191216.plot_AUCs(0)
    # exp_20191216.plot_AUCs(1)
    # pickle.dump(exp_20191216, open("20191216/20191216.p", "wb"))

    """20191204"""
    #
    # exp_20191204 = pickle.load(open("20191204/20191204.p", "rb"))
    # # exp_20191204.calculate_AUCs(plot_AUCs=False)
    # exp_20191204.plot_AUCs(fiber_type=1)
    #
    # # Save a experiment into a pickle file.
    # pickle.dump(exp_20191204, open("20191204/20191204.p", "wb"))

    # # Load the experiment from a pickle file
    # import pickle
    #
    # exp_20191204 = pickle.load(open("20191204/20191204.p", "rb"))
    # exp_20191204.plot_AUCs(fiber_type=1)

    # """20191119"""
    # # LIFE_STORAGE = 'RawE'
    # # EMG_STORAGE = 'RawG'
    # #
    # # EXPERIMENT_NAME = '20191119'
    # # STORAGE_PATH = "D:\\20191119_JaredNess_ImtheraVNS"
    # # plot_sigs = True
    # # exp_20191119 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, mono=6, bi=4)
    # # exp_20191119.gather_ephys(plot_sigs)
    # exp_20191119 = pickle.load(open("20191119/20191119.p", "rb"))
    # exp_20191119.calculate_AUCs(True)
    # exp_20191119.plot_AUCs(0)
    # exp_20191119.plot_AUCs(1)

    """20191009"""
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20191009'
    # STORAGE_PATH = "D:\\20191009_JaredNess_LivaNovaAndImtheraVNS"
    # plot_sigs = True
    # exp_20191009 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, custom_surgical_log=True, mono=6, bi=7)
    # exp_20191009.NUM_VAGOTOMY_CONFIGS = 0
    # exp_20191009.NUM_CONFIGS = 13
    # exp_20191009.NUM_LONG_RECORDINGS = 13
    # exp_20191009.gather_ephys(plot_sigs)
    # exp_20191009.calculate_AUCs(plot_sigs)
    # exp_20191009.plot_AUCs(0)
    # exp_20191009.plot_AUCs(1)
    # pickle.dump(exp_20191009, open("20191009/20191009.p", "wb"))
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'

    """20190925"""
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    # EXPERIMENT_NAME = '20190925'
    # STORAGE_PATH = "D:\\20190925_JaredNess_LivaNovaAndImtheraVNS"
    # plot_sigs = True
    # exp_20190925 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, mono=6, bi=6)
    # exp_20190925.NUM_VAGOTOMY_CONFIGS = 0
    # exp_20190925.NUM_CONFIGS = 12
    # exp_20190925.NUM_LONG_RECORDINGS = 12
    # exp_20190925.gather_ephys(plot_sigs)
    # exp_20190925.calculate_AUCs(plot_sigs)
    # exp_20190925.plot_AUCs(0)
    # exp_20190925.plot_AUCs(1)
    # # pickle.dump(exp_20190925, open("20190925/20190925.p", "wb"))

    """ 20190820 """
    # LIFE_STORAGE = 'RawE'
    # EMG_STORAGE = 'RawG'
    #
    # EXPERIMENT_NAME = '20190820'
    # STORAGE_PATH = "D:\\20190820_JaredNess_LivaNovaAndImtheraVNS"
    # plot_sigs = True
    # exp_20190820 = TdtExperiment(EXPERIMENT_NAME, LIFE_STORAGE, EMG_STORAGE, STORAGE_PATH, custom_surgical_log=True,
    #                              mono=2, bi=2)
    # exp_20190820.NUM_VAGOTOMY_CONFIGS = 0
    # exp_20190820.NUM_CONFIGS = 4
    # exp_20190820.NUM_LONG_RECORDINGS = 4
    # exp_20190820.gather_ephys(plot_sigs)
    # exp_20190820.gather_ephys(plot_signals=plot_sigs)

    """ More detailed plotting"""

    b = pickle.load(open("20190925/20190925.p", "rb"))
    # b.plot_one_mono_all_bi(mono_contact_num=6, fiber_type=1)



    #a.plot_one_mono_all_bi(1)
