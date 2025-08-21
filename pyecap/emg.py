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
import openpyxl
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks, medfilt, savgol_filter

from .base.epoch_data import _EpochData
from .base.utils.numeric import _to_numeric_array
from .utilities.ancillary_functions import check_make_dir


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
        """
        # todo: differentiate recording channels
        # todo: improve unit checking i.e. [2 cm, 1 mm]

        self.ephys = ephys_data
        self.stim = stim_data
        self.fs = ephys_data.sample_rate
        # self.master_df = pd.DataFrame()
        super().__init__(ephys_data, stim_data, stim_data)

    def calc_AUC(
        self,
        parameters=None,
        channels=None,
        window=None,
        window_units=None,
        bin=None,
        method="mean",
    ):

        column_headers = self.stim.parameters.columns
        masterLIST = []

        if window_units is None:
            raise ValueError(
                "EMG AUC calculation requires user specified 'window' and 'window_units'. Specify window_units in 'sec', 'ms','us' or 'samples'."
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

        # Returns a dictionary of mean values where the key is the parameter and the value is a numpyarray of 2
        # dimensions where the first dimension corresponding to channels and second dimension corresponding to data
        # points

        data = self.mean(paramLIST, chanLIST, bin)

        # Outer loop iterates through each parameter
        for param in data:
            mean_traces = data[param]

            # Inner loop iterates through each channel. Note, channels in chanLIST must be in order for the
            # arrays in mean_traces to correctly align with the channel.
            for idx, chan in enumerate(chanLIST):

                if isinstance(chan, int):
                    channel_name = self.ch_names[chan]
                else:
                    channel_name = chan

                RMS = self._calc_RMS(
                    data=mean_traces[idx], window=(start_idx, stop_idx)
                )
                window_str = str(window[0]) + " - " + str(window[1])

                masterLIST.append(
                    [
                        RMS,
                        window_str,
                        window_units,
                        bin,
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
                "Pulse Bin",
                *column_headers,
                "Recording Electrode",
            ],
        )
        return data_df
        # if self.master_df.empty or new_df == True:
        #     self.master_df = data_df
        # else:
        #     self.master_df = pd.concat([self.master_df, data_df], ignore_index=True)
