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

# pyCAP imports
from .base.ts_data import _TsData
from .base.event_data import _EventData
from .base.parameter_data import _ParameterData
from .base.utils.base import _to_array, _is_iterable
from .base.utils.numeric import _to_numeric_array, find_first
from .base.utils.visualization import (
    _plt_setup_fig_axis,
    _plt_show_fig,
    _plt_ax_to_pix,
    _plt_add_ax_connected_top,
    _plt_ax_aspect,
    _plt_add_cbar_axis,
)


# TODO: edit docstrings
class ECAP(_EpochData):
    """
    This class represents ECAP data
    """

    def __init__(self, ephys_data, stim_data):
        """
        Constructor for the ECAP class.

        Parameters
        ----------
        ephys_data : _TsData or subclass instance
            Ephys data object.
        stim_data : Stim class instance
            Stimulation data object.
        """

        if (
            isinstance(ephys_data, _TsData)
            and isinstance(stim_data, _EventData)
            and isinstance(stim_data, _ParameterData)
        ):
            self.ts_data = ephys_data
            self.event_data = stim_data
            self.parameters = stim_data
            self.x_lim = "auto"
        else:
            raise ValueError("Unrecognized input data types")
