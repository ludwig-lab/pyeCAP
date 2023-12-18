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
from .base.event_data import _EventData
from .base.parameter_data import _ParameterData

# pyCAP imports
from .base.ts_data import _TsData
from .base.utils.base import _is_iterable, _to_array
from .base.utils.numeric import _to_numeric_array, find_first
from .base.utils.visualization import (
    _plt_add_ax_connected_top,
    _plt_add_cbar_axis,
    _plt_ax_aspect,
    _plt_ax_to_pix,
    _plt_setup_fig_axis,
    _plt_show_fig,
)
from .utilities.ancillary_functions import check_make_dir


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

        super().__init__(ephys_data, stim_data, stim_data, "auto")
