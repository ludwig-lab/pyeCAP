from ancillary import surgical_log_io
from ancillary import ancillary_functions
import dask.array as da
import pyeCAP
import fnmatch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import dill
from pandas import Series
from pathlib import Path
from collections.abc import Iterable



class TdtExperiment:
    def __init__(
            self,
            experiment_name,
            experiment_storage_path,  # Name and path of exp
            tdt_chunk_size=2_000_000,
            create_storage_dictionary=True,
            time_difference=0,  # Accommodate for wrong times on recording devices
            num_recording_channels=None,
            name_recording_channels=None,
            name_stim_channels=None,
            # bool_examine_labchart_channels=None,
            stores=None,
            # rz_sample_rate=None,
            # si_sample_rate=None,
            sample_delay=None,
            baseline_time_length=2,
            bootstrap=False,
            **kwargs
    ):
        # Todo: vitals recorded true or false
        """
        # experiment_name:          name of experiment
        # experiment_storage_path:  storage path to your TDT and LabChart folders (if recorded).
        # create_storage_dictionary:if set to true this will create an Excel document with all storage banks. The user will then have the choice to configure this to assign unique attributes to each bank, or exclude a bank entirely.
        # NOTE: if create_storage_dictionary is not submitted, the user will have to manually
        #   supply a list of all stimulation and recording channels and names
        # num_recording_channels:   number of recording channels. If there are multiple types of recording channels,
        #                           this can be indicated by passing as an array ie
        #                           5 neural recordings, 3 emg recordings: [5, 3]
        # num_stim_channels:        number of stimulation channels. If there are multiple types of recording channels,
        #                           this can be indicated by passing as an array ie
        #                           5 monopolar channels, 3 bipolar channels: [5, 3]
        # name_recording_channels:  optional argument: for graphing purposes you may wish to name your channels
        # name_stim_channels:       optional argument: for graphing purposes you may wish to name your channels
        # vitals_recorded:          optional argument: set true if vital signs were recorded via LabChart
        # bootstrap:                number of times to bootstrap AUC recordings
        """
        # Declaration of required variables and constants

        self.experiment_name = experiment_name
        self.storage_path = experiment_storage_path
        self.num_recording_channels = num_recording_channels

        if type(num_recording_channels) == tuple or type(num_recording_channels) == list:
            self.total_recording_channels = 0
            for x in num_recording_channels:
                self.total_recording_channels += x
        elif type(num_recording_channels) == int or type(num_recording_channels) == float:
            self.total_recording_channels = int(num_recording_channels)

        self.name_recording_channels = name_recording_channels
        self.name_stim_channels = name_stim_channels
        self.BASELINE_TIME_LENGTH = baseline_time_length  # time in seconds for baseline to get recordings
        # self.EXAMINE_LABCHART_CHANNELS = bool_examine_labchart_channels

        # Initialize storage variables
        self.experimental_log_path = []
        self.all_amplitudes = np.array([])

        self.data_ephys = []
        self.data_stim = []

        if create_storage_dictionary:
            self._find_and_store_input_paths()

        # Todo: catch exception for when not create_storage_dictionary
        if not create_storage_dictionary:
            sys.exit()

        # Create directories for storage.
        self._find_create_storage()

        # Create dictionary from experimental log
        self.my_dictionary = surgical_log_io.create_dictionary(self.experimental_log_path)

        banks = [key for key in self.my_dictionary.keys()]
        self.data_ephys = pyeCAP.Ephys(banks, stores=stores,
                                       tdt_chunk_size=tdt_chunk_size,
                                       # rz_sample_rate=rz_sample_rate,
                                       # si_sample_rate=si_sample_rate,
                                       sample_delay=sample_delay, **kwargs)
        self.data_stim = pyeCAP.Stim(banks)
        self.data_stim.set_pulse_amplitude_abs()
        self.data_ecap = []
        self.data_phys = []
        self.data_phys_response = []

        # Modify params according to surgical_log
        condition_list = surgical_log_io.get_list(self.experimental_log_path, 4)
        stimulation_list = surgical_log_io.get_list(self.experimental_log_path, 3)
        channel_list = surgical_log_io.get_list(self.experimental_log_path, 2)

        for i in range(len(condition_list)):
            condition_series = Series([condition_list[i]], name='condition')
            stimulation_series = Series([stimulation_list[i]], name='stimulation type')
            channel_series = Series([channel_list[i]], name='channel')
            self.data_stim.add_series(i, condition_series)
            self.data_stim.add_series(i, stimulation_series)
            self.data_stim.add_series(i, channel_series)

    def _find_and_store_input_paths(self):
        target_filename = f"{self.experiment_name}_Experimental_Log.xlsx"
        matches = list(Path(self.storage_path).rglob(target_filename))

        if not matches:
            # No existing experimental log was found
            self.experimental_log_path = Path(
                ancillary_functions.create_experimental_log(self.storage_path)
            )

            os.startfile(str(self.experimental_log_path))
            time.sleep(2)

        elif len(matches) == 1:
            # Use the actual location of the file
            self.experimental_log_path = matches[0]

        else:
            # Multiple matching logs were found
            match_list = "\n".join(str(path) for path in matches)

            raise RuntimeError(
                f"Multiple experimental logs named {target_filename!r} were found:\n"
                f"{match_list}"
            )

        # Find labchart storage as .mat file
        mat_files = []
        for root, dirs, files in os.walk(self.storage_path):
            for name in files:
                if fnmatch.fnmatch(name, '*.mat'):
                    mat_files.append(os.path.join(root, name))

        if len(mat_files) == 0:
            sys.exit("Labchart data has not yet been converted to '*.mat'. Please convert. Exiting")
        # Todo: better way of finding LC mat file without being too restrictive ie. looking for vitals.mat
        elif len(mat_files) > 1:
            sys.exit("Too many '*.mat' files. Please only have one. Exiting")
        else:
            self.file_path_lc = mat_files

    def _find_create_storage(self):
        ancillary_functions.check_make_dir(self.experiment_name)
        # self.traces_save_path = self.experiment_name + "/Filtered Traces"
        # ancillary_functions.check_make_dir(self.traces_save_path)
        ancillary_functions.check_make_dir(self.experiment_name + "/AUC")
        self.save_auc_fig_path = self.experiment_name + "/AUC"
        # ancillary_functions.check_make_dir(self.experiment_name + "/Min Max Analysis")
        # self.min_max_save_path = self.experiment_name + "/Min Max Analysis"

    def curate_data(
            self,
            keep_channels=None,  # Existing channel names/indices to retain
            ch_names=None,  # Optional replacement names
            ch_types=None,  # Optional replacement channel types
            remove_channels=None,  # Existing channel names/indices to remove
            filter_median_low=False,
            filter_median=False,
            filter_gaussian_highpass=False,
            filter_powerline=False,
    ):
        """
        Curate electrophysiology channels.

        Parameters
        ----------
        keep_channels : str, int, or iterable, optional
            Existing channel names or indices to retain.

        ch_names : str or iterable of str, optional
            New names for the remaining channels.

        ch_types : str or iterable of str, optional
            New types for the remaining channels.

        remove_channels : str, int, or iterable, optional
            Existing channel names or indices to remove.
        """

        def _as_list(value):
            """Convert a scalar or iterable into a list without splitting strings."""
            if value is None:
                return []

            if isinstance(value, (str, np.str_)):
                return [str(value)]

            if isinstance(value, (int, np.integer)) and not isinstance(
                    value, (bool, np.bool_)
            ):
                return [int(value)]

            try:
                return list(value)
            except TypeError as exc:
                raise TypeError(
                    "Expected a channel name, channel index, or iterable; "
                    f"received {type(value).__name__}."
                ) from exc

        def _to_names(indices_or_names, all_names):
            """Resolve channel indices and names into existing channel names."""
            all_names = list(all_names)
            values = _as_list(indices_or_names)

            resolved_names = []

            for value in values:
                if isinstance(value, (bool, np.bool_)):
                    raise TypeError(
                        "Boolean values cannot be used as channel indices."
                    )

                if isinstance(value, (int, np.integer)):
                    index = int(value)

                    if not 0 <= index < len(all_names):
                        raise IndexError(
                            f"Channel index {index} is out of range. "
                            f"Valid indices are 0 through {len(all_names) - 1}."
                        )

                    resolved_names.append(all_names[index])

                elif isinstance(value, (str, np.str_)):
                    name = str(value)

                    if name not in all_names:
                        raise ValueError(
                            f"Unknown channel {name!r}. "
                            f"Available channels: {all_names}"
                        )

                    resolved_names.append(name)

                else:
                    raise TypeError(
                        "Channels must be string names or integer indices; "
                        f"received {value!r} "
                        f"({type(value).__name__})."
                    )

            return resolved_names

        def _apply(method_name, *args, **kwargs):
            """
            Call a data_ephys method while supporting either:

            1. Methods that return the modified object.
            2. Methods that modify the object in place and return None.
            """
            result = getattr(self.data_ephys, method_name)(*args, **kwargs)

            if result is not None:
                self.data_ephys = result

        keep_channels = _as_list(keep_channels)
        remove_channels = _as_list(remove_channels)
        ch_names = _as_list(ch_names)
        ch_types = _as_list(ch_types)

        current = list(self.data_ephys.ch_names)

        # 1. Remove explicitly requested channels.
        if remove_channels:
            remove_by_name = _to_names(remove_channels, current)
            _apply("remove_ch", remove_by_name)
            current = list(self.data_ephys.ch_names)

        # 2. Keep only explicitly requested channels.
        if keep_channels:
            keep_names = _to_names(keep_channels, current)

            if len(keep_names) != len(set(keep_names)):
                raise ValueError(
                    f"Duplicate channels were requested: {keep_names}"
                )

            complement = [
                name for name in current
                if name not in keep_names
            ]

            if complement:
                _apply("remove_ch", complement)

            current = list(self.data_ephys.ch_names)

            # Removing the complement does not necessarily reorder channels.
            if current != keep_names:
                raise ValueError(
                    "The requested channel order differs from the existing order. "
                    f"Requested: {keep_names}; resulting order: {current}. "
                    "A channel-reordering method must be used here."
                )

        # 3. Rename remaining channels.
        if ch_names:
            if len(ch_names) != len(current):
                raise ValueError(
                    f"Received {len(ch_names)} new channel names, but "
                    f"{len(current)} channels remain."
                )

            if not all(isinstance(name, str) for name in ch_names):
                raise TypeError("All new channel names must be strings.")

            if len(ch_names) != len(set(ch_names)):
                raise ValueError(
                    f"Channel names must be unique: {ch_names}"
                )

            _apply("set_ch_names", ch_names)
            current = list(self.data_ephys.ch_names)

        # 4. Set types for remaining channels.
        if ch_types:
            if len(ch_types) != len(current):
                raise ValueError(
                    f"Received {len(ch_types)} channel types, but "
                    f"{len(current)} channels remain."
                )

            _apply("set_ch_types", ch_types)

        self.total_recording_channels = len(current)

        if getattr(self, "num_recording_channels", None) is not None:
            expected = self.num_recording_channels

            if isinstance(expected, (list, tuple, np.ndarray)):
                expected = sum(expected)

            expected = int(expected)

            if expected and expected != len(current):
                raise ValueError(
                    f"num_recording_channels={expected}, but "
                    f"{len(current)} channels remain."
                )

        # 5. Filtering
        if filter_powerline:
            _apply(
                "filter_powerline_iir",
                frequencies=(60, 120, 180),
                Q=35.0,
                pad_seconds=0.5,
            )

        if filter_median_low:
            _apply(
                "filter_median",
                btype="lowpass",
                kernel_size=11,
            )

        if filter_median:
            _apply(
                "filter_median",
                btype="highpass",
            )

        if filter_gaussian_highpass:
            _apply(
                "filter_gaussian",
                Wn=2000,
                btype="lowpass",
            )

        return self

    def gather_ecap(self, experiment_log_path, filter_post_average=False, plot_AUCs=False):
        distances = ancillary_functions.electrode_distances(experiment_log_path)
        ecap = self.data_ecap = pyeCAP.ECAP(self.data_ephys, self.data_stim, distances, preload=False)

        if filter_post_average:
            # optional: persist for speed if you’ll compute a lot
            fs = ecap.ts_data.sample_rate
            ecap.persist_ts(time_chunk=int(0.01 * fs * 6))

            param_keys = list(ecap.parameters.parameters.index)

            # lazy stack of mean waveforms, one per parameter
            wf_lazy = [ecap.mean_waveform(p) for p in param_keys]  # each (channels, samples)
            wf3_lazy = da.stack(wf_lazy, axis=0)  # (params, channels, samples)

            # compute and cache if downstream expects a NumPy array in self.mean_traces
            self.data_ecap.mean_traces = np.asarray(wf3_lazy.compute())

            self.data_ecap.filter_averages(filter_median_lowpass=True, filter_channels=[4, 5, 6])
            self.data_ecap.filter_averages(filter_median_highpass=True, filter_gaussian_highpass=True,
                                           filter_channels=[0, 1, 2, 3, 4, 5, 6])

        # self.data_ecap.calculate_AUC(window_type="standard_neural", analysis_method="RMS")
        # self.data_ecap.calculate_AUC(window_type="standard_neural", analysis_method="Peaks", plot_AUC=plot_AUCs,
        #                              save_path=self.save_auc_fig_path)
        #
        # self.data_ecap.calculate_AUC(window_type="standard_EMG", analysis_method="RMS", plot_AUC=plot_AUCs)

    def gather_phys(self, phys_location, trigger_channel, time_difference=0):
        self.data_phys = pyeCAP.Phys(phys_location)
        self.data_phys_response = pyeCAP.PhysResponse(phys_data=self.data_phys, stim_data=self.data_stim,
                                                      trigger_channel=trigger_channel, time_difference=time_difference)
        self.data_phys.set_ch_names(['Stim', 'HR', 'BP'])


if __name__ == '__main__':

    def conduct_experiment(
            exp_name,
            plot_AUCs,
            ADI_trigger_channel,
            bool_filt_gauss,
            bool_filt_median_low, bool_filt_median, bool_filt_powerline, exp_path=None, collect_phys=False,
            stores=None,  # Default: ['RawE', 'RawG']
            remove_channels=None,
            rec_ch_names=None,  # Default: ['ENG', 'ENG', 'ENG', 'ENG', 'EMG', 'EMG', 'EMG']
            rec_ch_types=None,
            # Default: ['LIFE 1', 'LIFE 2', 'LIFE 3', 'LIFE 4', 'EMG 1', 'EMG 2', 'EMG 3']
            rz_sample_rate=25,
            si_sample_rate=25,
            sample_delay=None,
            bool_filter_post_average=False,
            time_difference=0
    ):
        if stores is None:
            stores = ['RawE', 'RawG']
        if rec_ch_types is None:
            rec_ch_types = ['ENG', 'ENG', 'ENG', 'ENG', 'EMG', 'EMG', 'EMG']
        if rec_ch_names is None:
            rec_ch_names = ['LIFE 1', 'LIFE 2', 'LIFE 3', 'LIFE 4', 'EMG 1', 'EMG 2', 'EMG 3']

        tic = time.perf_counter()
        if exp_path is not None:
            storage_path_test = exp_path
        else:
            storage_path_test = r"D:\ImThera\data_raw\\" + exp_name

        global exp
        exp = TdtExperiment(exp_name, storage_path_test, stores=stores,
                            # rz_sample_rate=rz_sample_rate,
                            # si_sample_rate=si_sample_rate,
                            sample_delay=sample_delay)
        exp.curate_data(
            ch_names=rec_ch_names,
            ch_types=rec_ch_types,
            remove_channels=remove_channels,
            filter_median_low=bool_filt_median_low,
            filter_median=bool_filt_median,
            filter_gaussian_highpass=bool_filt_gauss,
            filter_powerline=bool_filt_powerline,
        )

        exp.gather_ecap(experiment_log_path=exp.experimental_log_path, plot_AUCs=plot_AUCs,
                        filter_post_average=bool_filter_post_average)

        toc = time.perf_counter()
        print("total time elapsed:", toc - tic)

        if collect_phys:
            phys_location = os.path.join(
                storage_path_test,
                "LabChart",
                "STIM_HR_BP.mat",
            )
            exp.gather_phys(phys_location=phys_location, trigger_channel=ADI_trigger_channel,
                            time_difference=time_difference)
            exp.data_phys_response.create_all_phys_dataframe()


    ###### conduct_experiment("20190820", additional_delay=4, plot_AUCs=True, ADI_trigger_channel=0, bool_filt_gauss=True, bool_filt_median=True, bool_filt_powerline=True)

    #### Begin collective code
    # conduct_experiment("20190925", sample_delay=8, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median=True, bool_filt_median_low=True, bool_filt_powerline=True,
    #                    exp_path=r'D:\Data\cVNS\20190925',
    #                    filter_post_average=False, bootstrap=False)
    # save_path = r'D:\Data Analysis\cVNS\2019_09_25\\'
    # file1 = save_path + exp.experiment_name + "_master_df_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp
    #
    # conduct_experiment("20191009", sample_delay=4, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False)
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp

    # conduct_experiment("20191119", sample_delay=11, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median_low=False,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False)
    # dill.dump(exp.data_ecap, file=open(r'C:\Users\steph\Desktop\DF Collection\20191119_data_ecap.pkl', "wb"))
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp

    #
    # conduct_experiment("20191204", sample_delay=4, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median_low=True,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False, bootstrap=True)
    # save_path = "C:\\users\\steph\\Desktop\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp
    #
    # conduct_experiment("20191126", sample_delay=6, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False)
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp
    #

    conduct_experiment("20191204", sample_delay=4, remove_channels="RawG 4", plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
                       bool_filt_median_low=False,
                       bool_filt_median=True, bool_filt_powerline=True, bool_filter_post_average=False)

    save_path = Path(r"D:\ImThera\data_intermediate\Dask")
    save_path.mkdir(parents=True, exist_ok=True)

    file1 = save_path / f"{exp.experiment_name}_df_bootstrap_filterpreaverage.pkl"
    file2 = save_path / f"{exp.experiment_name}_bootstrap_filterpreaverage.pkl"

    exp.data_ecap.master_df.to_pickle(file1)

    with file2.open("wb") as file:
        dill.dump(exp, file=file)

    # conduct_experiment("20191216", sample_delay=4, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False)
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp
    #
    # conduct_experiment("20200113", sample_delay=5, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=True,
    #                    bool_filt_median=True, bool_filt_powerline=True, filter_post_average=False)
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + "_df_bootstrap_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_bootstrap_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))
    # del exp

    # ## short version for all
    # exps = ["20190925", "20191119", "20191126", "20191204", "20191216", "20200113"]
    # delays = [8,11,6,4,4,5]
    # for idx, e in enumerate(exps):
    #
    #     conduct_experiment(exps[idx], sample_delay=delays[idx], plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=False,
    #                        bool_filt_median=False, bool_filt_powerline=False, filter_post_average=True)
    #     save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    #     file1 = save_path + exp.experiment_name + "_shortFilter.xlsx"
    #     file2 = save_path + exp.experiment_name + "_short_Filter.pkl"
    #     exp.data_ecap.master_df.to_excel(file1, index=False, index_label=False)
    #     dill.dump(exp, file=open(file2, "wb"))
    #     del exp

    # conduct_experiment("20190925", sample_delay=4, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=False,
    #                    bool_filt_median=False, bool_filt_powerline=False, filter_post_average=True, collect_phys=False, time_difference=0)
    # save_path = "C:\\users\\steph\\desktop\\DF Collection\\"
    # file1 = save_path + exp.experiment_name + ".xlsx"
    # file2 = save_path + exp.experiment_name + ".pkl"
    # exp.data_ecap.master_df.to_excel(file1, index=False, index_label=False)
    # dill.dump(exp + "_short_HR_increase", file=open(file2, "wb"))

    # aVNS
    # conduct_experiment(exp_name="20210428", plot_AUCs=False, exp_path=r'D:\aVNS\20210428', stores=['life', 'Raws'],
    #                    rec_ch_names=['LIFE 1', 'LIFE 2', 'LIFE 3', 'CUFF 1', 'CUFF 2', 'CUFF 3', 'MICRO 1', 'MICRO 2', 'MICRO 3'],
    #                    rec_ch_types=['LIFE']*9,
    #                    sample_delay=9,
    #                    bool_filt_gauss=True, bool_filt_median=True, bool_filt_powerline=True,
    #                    filter_post_average=False,
    #                    ADI_trigger_channel=0
    #                    )
    # save_path = r'C:\Users\steph\Data Analysis\aVNS\20210428\Data\\'
    # file1 = save_path + exp.experiment_name + "_df_filterpreaverage.pkl"
    # file2 = save_path + exp.experiment_name + "_filterpreaverage.pkl"
    # exp.data_ecap.master_df.to_pickle(file1)
    # dill.dump(exp, file=open(file2, "wb"))

    ##### End collective code
    # conduct_experiment("Sample 20191204", sample_delay=4, plot_AUCs=False, ADI_trigger_channel=0, bool_filt_gauss=False,
    #                    bool_filt_median=False, bool_filt_powerline=True, filter_post_average=False)

    # # Use these to dump/load data after import
    # dill.dump(exp20191204_filtered, file=open("Dask/20191204/exp_20191204_filtered.pkl", "wb"))
    # # Use this to load
    # exp = dill.load(open(r"C:\Users\steph\Desktop\DF Collection\20191204.pkl", "rb"))

    # phys_location = r"D:\20190925\LabChart\STIM_HR_BP.mat"
    # exp.gather_phys(phys_location=phys_location, trigger_channel=0, time_difference=0)
    # exp.data_phys_response.create_all_phys_dataframe()
    # exp20191204_raw.data_ecap.window_onset_idx = exp20191204_raw.data_ecap.window_idx
    # exp20191204_raw.data_ecap.neural_fiber_names = [r'A-$\alpha$', r'A-$\beta$', r'A-$\gamma$', r'A-$\delta$', 'B']
    # exp20191204_raw.data_ecap.calculate_AUC()
