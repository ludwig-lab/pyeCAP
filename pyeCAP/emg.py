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

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        self.master_df = pd.DataFrame()
        super().__init__(ephys_data, stim_data, stim_data)

    def calc_AUC(self, window=None, window_units=None, method='mean'):

        column_headers = self.stim.parameters.columns
        metadata_list = [self.stim.parameters[p].to_numpy() for p in column_headers]
        channelNAMES = np.array(self.ephys.ch_names)
        master_list = []

        self._avg_data_for_AUC(method=method)

        if window_units is None:
            raise ValueError("EMG AUC calculation requires user specified 'window' and 'window_units'. Specify window_units in 'sec', 'ms','us' or 'samples'.")
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

        #Outer Loop iterates through each parameter/pulse train within the TDT block
        for stimIDX, avgDATA in enumerate(self.AUC_traces):

            #Inner loop iterates through each active recording electrode
            for chanIDX, trace in enumerate(avgDATA):
                metadata = [*[p[stimIDX] for p in metadata_list], channelNAMES[chanIDX]]
                RMS = self._calc_RMS(trace, window=(start_idx, stop_idx))
                master_list.append([RMS, window, window_units, *metadata])

        new_df = pd.DataFrame(master_list, columns=['AUC (Vs)', "Calculation Window", "Window Units", *column_headers, 'Recording Electrode'])

        if self.master_df.empty:
            self.master_df = new_df
        else:
            self.master_df = pd.concat([self.master_df, new_df], ignore_index=True)

    def _avg_data_for_AUC(self, parameter_index=None, method='mean'):
        # data frame with onset and offsets
        self.df_epoc_idx = self.epoc_index()

        if parameter_index is None:
            parameter_index = self.df_epoc_idx.index

        #tic = time.perf_counter()
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
        #toc = time.perf_counter()
        #print(toc - tic, "seconds elapsed")