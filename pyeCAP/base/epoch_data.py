# Python standard library imports
import hashlib
import math
import sys
import time
import warnings
from collections import OrderedDict
from functools import cached_property, lru_cache

import dask

# Scientific computing package imports
import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from cached_property import threaded_cached_property
from dask import delayed

# interactive plotting
from ipywidgets import AppLayout, Button, FloatSlider, Output, VBox, interact

# Plotting
from matplotlib import get_backend as plt_get_backend
from matplotlib.collections import LineCollection

from .event_data import _EventData
from .parameter_data import _ParameterData

# pyCAP imports
from .ts_data import _TsData
from .utils.base import _is_iterable, _to_array
from .utils.numeric import _get_size, _to_numeric_array, find_first_true
from .utils.visualization import (
    _plt_add_ax_connected_top,
    _plt_add_cbar_axis,
    _plt_ax_aspect,
    _plt_ax_to_pix,
    _plt_setup_fig_axis,
    _plt_show_fig,
)

sns.set_context(
    "paper", font_scale=1.4, rc={"lines.linewidth": 2.5, "axes.linewidth": 2.0}
)
sns.set_style("ticks")


def _to_parameters(parameters):
    if isinstance(parameters, tuple) and len(parameters) == 2:
        return [parameters]
    elif _is_iterable(parameters, element_type=tuple):
        return parameters
    else:
        return _to_array(parameters, dtype="int,int")


# TODO: check if all relevant methods have docstrings
# TODO: edit docstrings to be more informative
class _EpochData:
    """
    Class representing Epoch Data and parent class of ECAP. Contains the methods for the ECAP class.
    """

    @classmethod
    def clear_cache(cls):
        """
        Clears the cache across all instances of _EpochData.
        """
        cls._cache.clear()

    @classmethod
    def cache_size(cls):
        """
        Returns the current size of the cache in bytes.
        """
        return sum(sys.getsizeof(v) for v in cls._cache.values())

    def __init__(self, ts_data, event_data, parameters, x_lim="auto"):
        """
        Constructor for _Epochdata class.

        Parameters
        ----------
        ts_data : _TsData or subclass
            Time series or Ephys data.
        event_data : _EventData or subclass
            Stimulation event data.
        parameters : _ParameterData or subclass
            Stimulation parameter data.
        x_lim : str
            ???
        """
        self._cache = (
            OrderedDict()
        )  # Initialize a cahce across all instances of _EpochData
        self._cache_size = 0.5e9  # Set the maximum cache size in bytes
        if (
            isinstance(ts_data, _TsData)
            and isinstance(event_data, _EventData)
            and isinstance(parameters, _ParameterData)
        ):
            self.ts_data = ts_data
            self.event_data = event_data
            self.parameters = parameters
            self.x_lim = x_lim
        else:
            raise ValueError("Unrecognized input data types")

    @cached_property
    def _state_identifier(self):
        """
        Generate a state identifier that integrates the result from the 3 classes it inherits from.
        """

        # get_state_identifier properties
        ts_data_id = self.ts_data._state_identifier
        event_data_id = self.event_data._state_identifier
        parameter_data_id = self.parameters._state_identifier

        # Concatenate the identifiers and convert to bytes
        combined_id = (ts_data_id + event_data_id + parameter_data_id).encode()

        # Generate a SHA256 hash of the combined identifier
        return hashlib.sha256(combined_id).hexdigest()

    @property
    def sample_rate(self):
        return self.ts_data.sample_rate

    @property
    def ch_names(self):
        return self.ts_data.ch_names

    @lru_cache(maxsize=None)
    def time(self, parameter):
        """
        Returns an array of times starting at 0 seconds. These times correspond to times of data points from one pulse
        of the given parameter. The result is stored in a cache for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.

        Returns
        -------
        numpy.ndarray
            Array of times of the data points.

        Examples
        ________
        >>> ecap_data.time((0,0))   # first stimulation of first data set       # doctest: +SKIP
        >>> ecap_data.time((0,1))   # second stimulation of first data set      # doctest: +SKIP
        """
        sample_len = self.dask_array(parameter).shape[2]
        if self.x_lim is None or self.x_lim == "auto":
            return np.arange(0, sample_len) / self.sample_rate
        else:
            return (np.arange(0, sample_len) / self.sample_rate) - self.x_lim[0]

    def epoc_index(self):
        """
        :return: dataframe of onset indices stores in a  multiindex dataframe of mxn
            where m:conditions, n: amplitude
        """
        fs = self.sample_rate
        pulse_count = self.event_data.parameters["pulse count"]

        # epocs including offset: might have to include this later for checking. Leaving Commented out for now
        df_offset = pd.DataFrame(self.ts_data.start_indices)
        df_offset["offset index"] = df_offset[[0]]
        df_offset = df_offset.rename(columns={0: "onset index"})
        df_epocs = self.event_data.parameters[["onset time (s)", "offset time (s)"]]
        fs = self.sample_rate
        # df_epocs['time diff (s)'] = df_epocs['offset time (s)'] - df_epocs['onset time (s)']
        df_epocs_idx = round(df_epocs * fs).astype(int)
        df_epocs_idx = df_epocs_idx.rename(
            columns={"onset time (s)": "onset index", "offset time (s)": "offset index"}
        )

        return df_epocs_idx

    @lru_cache(maxsize=None)
    def dask_array(self, parameter):
        """
        Returns an array of data points from the data set from the given parameter. The array will contain
        the Ephys data with one dimension representing the pulse and another representing the channel. The remaining
        dimension specifies the number of data points in each pulse/channel combination. The result is stored in a cache
        for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        Returns
        -------
        dask.array.core.Array
            Array

        Examples
        ________
        >>> ecap_data.dask_array((0,0)) # doctest: +SKIP
        """
        # Initialize a list to store event times for each channel.
        event_times = []
        # ToDo: modify so data continues to be pulled out across channels if channels are equal between event_data
        #  and ephys data but if channels match between the two datasets then pull out on a perchannel basis

        # Iterate over each channel in the event data.
        for ch in self.event_data.ch_names:
            # Get events for the current channel based on start times.
            events = self.event_data.events(
                ch, start_times=self.ts_data.start_indices / self.sample_rate
            )
            # Get event indicators for the current channel.
            event_indicators = self.event_data.event_indicators(ch)
            # Filter events based on the provided stimulation parameter.
            event_indices = np.logical_and(
                event_indicators[:, 0] == parameter[0],
                event_indicators[:, 1] == parameter[1],
            )
            # Append the filtered events to the event times list.
            event_times.append(events[event_indices])

        # Concatenate event times from all channels into a single array.
        event_times = np.concatenate(event_times, axis=0)

        # Determine the length of each sample based on settings or automatically.
        if self.x_lim is None or self.x_lim == "auto":
            min_event_time = np.min(np.diff(event_times))
            sample_len = np.round(min_event_time * self.sample_rate)
        else:
            event_times = event_times + self.x_lim[0]
            sample_len = int(
                np.round((self.x_lim[1] - self.x_lim[0]) * self.sample_rate)
            )

        # Convert event times to indices.
        event_times = self.ts_data._time_to_index(event_times)

        # Create a boolean mask to select relevant times from the time series data.
        indices = np.zeros(self.ts_data.shape[1], dtype=bool)
        for ts in event_times:
            te = int(ts + sample_len)
            indices[ts:te] = True

        # Find the indices of the first and last onset in the boolean mask.
        first_onset_idx = find_first_true(indices)
        last_onset_idx = len(indices) - find_first_true(np.flip(indices)) - 1

        # Determine indices to remove from the data.
        removal_idx = np.logical_not(
            indices[first_onset_idx : last_onset_idx + 1]
        ).nonzero()

        # Extract the relevant event data.
        event_data = self.ts_data.array[:, first_onset_idx : last_onset_idx + 1]
        # Create a mask to filter the event data.
        idx_mask = np.repeat(
            indices[first_onset_idx : last_onset_idx + 1][np.newaxis, :],
            self.ts_data.shape[0],
            axis=0,
        )

        # If there are data points to remove, compute chunk sizes for the Dask array.
        # ToDo: Create Heuristic model with best way to extract data if it is tightly packed or not
        if len(removal_idx[0]) > 0:
            # ToDo: rewrite so that this happens blockwise in dask as opposed to all at once to speed up.
            event_data = event_data[idx_mask].compute_chunk_sizes()

        # Reshape the event data and rearrange axes for the final output.
        event_data_reshaped = da.reshape(
            event_data, (self.ts_data.shape[0], len(event_times), int(sample_len))
        )
        return da.moveaxis(event_data_reshaped, 1, 0)

    def get_parameters(
        self, channel=None, bipolar_ch=None, amplitude=None, amp_cutoff=None
    ):

        # TODO: make function work with lists of channels and bipolar channels

        # Bring up stimulation data to use for finding desired parameters
        df = self.parameters.parameters
        paramLIST = []

        if channel is None:
            # If no channel or bipolar channel are specified return all params matching amplitude arguments
            if bipolar_ch is None:
                if amp_cutoff is not None:
                    if isinstance(amplitude, list):
                        raise ValueError(
                            "If the 'amp_cutoff' argument is defined, the 'amplitude' argument must be a single value."
                        )
                    elif amp_cutoff == "high":
                        paramLIST = df.loc[
                            abs(df["pulse amplitude (μA)"]) >= amplitude
                        ].index
                    elif amp_cutoff == "low":
                        paramLIST = df.loc[
                            abs(df["pulse amplitude (μA)"]) <= amplitude
                        ].index
                    else:
                        raise ValueError(
                            "The 'amp_cutoff' argument must be 'high', 'low', or 'None'. Use 'None' when you wish to pass a specific value or list of amplitudes to look for."
                        )
                else:
                    if not isinstance(amplitude, list):
                        amplitude = [amplitude]
                    paramLIST = df.loc[df["pulse amplitude (μA)"].isin(amplitude)].index
            # Channel is not defined, but bipolar channel is then return params that satisfy amplitude arguments while matching the bipolar channel
            else:
                if (
                    amplitude is None
                ):  # If only bipolar channel is specified, return all params that use that bipolar channel
                    paramLIST = df.loc[df["bipolar channel"] == bipolar_ch].index
                elif amp_cutoff is not None:
                    if isinstance(amplitude, list):
                        raise ValueError(
                            "If the 'amp_cutoff' argument is defined, the 'amplitude' argument must be a single value."
                        )
                    elif amp_cutoff == "high":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) >= amplitude)
                            & (df["bipolar channel"] == bipolar_ch)
                        ].index
                    elif amp_cutoff == "low":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) <= amplitude)
                            & (df["bipolar channel"] == bipolar_ch)
                        ].index
                    else:
                        raise ValueError(
                            "The 'amp_cutoff' argument must be 'high', 'low', or 'None'. Use 'None' when you wish to pass a specific value or list of amplitudes to look for."
                        )
                else:
                    if not isinstance(amplitude, list):
                        amplitude = [amplitude]
                    paramLIST = df.loc[
                        (df["pulse amplitude (μA)"].isin(amplitude))
                        & (df["bipolar channel"] == bipolar_ch)
                    ].index

        # If statements for situations where 'channel' is not specified
        else:
            # Channel is defined, but bipolar channel is undefined, return all params using channel and matching amplitude arguments
            if bipolar_ch is None:
                if (
                    amplitude is None
                ):  # If only channel is specified, return all params that use that channel
                    paramLIST = df.loc[df["channel"] == channel].index
                elif amp_cutoff is not None:
                    if isinstance(amplitude, list):
                        raise ValueError(
                            "If the 'amp_cutoff' argument is defined, the 'amplitude' argument must be a single value."
                        )
                    elif amp_cutoff == "high":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) >= amplitude)
                            & (df["channel"] == channel)
                        ].index
                    elif amp_cutoff == "low":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) <= amplitude)
                            & (df["channel"] == channel)
                        ].index
                    else:
                        raise ValueError(
                            "The 'amp_cutoff' argument must be 'high', 'low', or 'None'. Use 'None' when you wish to pass a specific value or list of amplitudes to look for."
                        )
                else:
                    if not isinstance(amplitude, list):
                        amplitude = [amplitude]
                    paramLIST = df.loc[
                        (df["pulse amplitude (μA)"].isin(amplitude))
                        & (df["channel"] == channel)
                    ].index
            # Channel and bipolar_ch are defined, return all params that match all arguments
            else:
                if amplitude is None:
                    paramLIST = df.loc[
                        (df["channel"] == channel)
                        & (df["bipolar channel"] == bipolar_ch)
                    ].index
                elif amp_cutoff is not None:
                    if isinstance(amplitude, list):
                        raise ValueError(
                            "If the 'amp_cutoff' argument is defined, the 'amplitude' argument must be a single value."
                        )
                    elif amp_cutoff == "high":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) >= amplitude)
                            & (df["channel"] == channel)
                            & (df["bipolar channel"] == bipolar_ch)
                        ].index
                    elif amp_cutoff == "low":
                        paramLIST = df.loc[
                            (abs(df["pulse amplitude (μA)"]) <= amplitude)
                            & (df["channel"] == channel)
                            & (df["bipolar channel"] == bipolar_ch)
                        ].index
                    else:
                        raise ValueError(
                            "The 'amp_cutoff' argument must be 'high', 'low', or 'None'. Use 'None' when you wish to pass a specific value or list of amplitudes to look for."
                        )
                else:
                    if not isinstance(amplitude, list):
                        amplitude = [amplitude]
                    paramLIST = df.loc[
                        (df["pulse amplitude (μA)"].isin(amplitude))
                        & (df["channel"] == channel)
                        & (df["bipolar channel"] == bipolar_ch)
                    ].index

        # If only channel is specified, return all params that use that channel

        # If only bipolar channel is specified, return all params that use that bipolar channel

        # If channel and bipolar channel are specified, return all params that use that pairing

        # If amplitude is specified and has a cutoff type, return params based on the type. Above/Below etc., This cannot be used if a list of amplitudes is passed

        return paramLIST.tolist()

    def plot(
        self,
        channels,
        parameters,
        *args,
        parameter_label=None,
        method="mean",
        axis=None,
        x_lim=None,
        y_lim="auto",
        colors=sns.color_palette(),
        fig_size=(10, 5),
        show=True,
        spread_parameters=False,
        spread_factor=3.0,
        **kwargs,
    ):
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        calc_y_lim = [0, 0]

        if spread_parameters:
            channels = tuple(channels)
        else:
            channels = channels[0]

        spread_accumulator = 0
        max_std_data = 0
        std_data = 0
        custom_y_ticks = []
        custom_y_labels = []

        for p, c in zip(_to_parameters(parameters), colors):
            if parameter_label is not None:
                parameter_label_value = (
                    self.event_data.parameters.loc[p, parameter_label]
                    if parameter_label in self.event_data.parameters.columns
                    else None
                )

            if method == "mean":
                plot_data = self.mean(p, channels=channels)
            elif method == "median":
                plot_data = self.median(p, channels=channels)
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'."
                )
            plot_time = self.time(p)

            if spread_parameters:
                std_data = np.std(plot_data)
                if std_data > max_std_data:
                    max_std_data = std_data
                adjusted_plot_data = plot_data + (
                    np.ones_like(plot_data) * spread_accumulator
                )
                for i, ch in enumerate(channels):
                    ax.plot(
                        plot_time,
                        adjusted_plot_data[i, :],
                        *args,
                        color=colors[i],
                        **kwargs,
                    )

                # Save the position and label for the custom tick
                custom_y_ticks.append(spread_accumulator)
                if parameter_label is not None:
                    custom_y_labels.append(parameter_label_value)

                spread_accumulator += std_data * spread_factor
            else:
                ax.plot(plot_time, plot_data, *args, color=c, **kwargs)
        spread_accumulator -= std_data * spread_factor

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

        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")  # Label for the left y-axis
        ax.set_ylim(
            _to_numeric_array([calc_y_lim[0], calc_y_lim[1] + spread_accumulator])
        )

        if spread_parameters:
            ax2 = ax.twinx()
            ax.set_ylabel(parameter_label)
            ax2.set_ylabel("amplitude (V)")  # Label for the right y-axis
            ax2.set_ylim(
                _to_numeric_array([calc_y_lim[0], calc_y_lim[1] + spread_accumulator])
            )

            # Remove existing left y-axis ticks and set custom ticks
            ax.set_yticks(custom_y_ticks)
            ax.set_yticklabels(custom_y_labels)

            for i, ch in enumerate(channels):
                ax.plot(
                    [], [], color=colors[i], label=ch
                )  # Dummy plot for legend entry
            ax.legend(loc="lower right")

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        if spread_parameters:
            return _plt_show_fig(fig, [ax, ax2], show)
        else:
            return _plt_show_fig(fig, ax, show)

    def plot_channel(
        self,
        channel,
        parameters,
        *args,
        method="mean",
        axis=None,
        x_lim=None,
        y_lim="auto",
        colors=sns.color_palette(),
        fig_size=(10, 3),
        show=True,
        sort=None,
        fig_title=None,
        vlines=None,
        **kwargs,
    ):
        """
        Plots the data from a channel for the given stimulation parameters. Plotting occurs over the time interval of
        one pulse period, starting at 0 seconds. Plotting uses either the mean or median of each data point across all
        pulses.

        Parameters
        ----------
        channel : str, int
            Channel or channel index to be plotted.
        parameters : tuple, list
            Stimulation parameter or list of parameters.
        * args : Arguments
            See `mpl.axes.Axes.plot <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ for more
            information.
        method : str
            Use 'mean' to plot the mean values and 'median' to plot the median values.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis, or a matplotlib axis to plot on.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        y_lim : None, str, list, tuple, np.ndarray
            None or 'auto' to automatically calculate reasonable bounds based on standard deviation of data. 'max' to
            plot y axis limits encompassing all accessible data. Otherwise tuple, list, or numpy array of length 2
            containing limits for the y axis.
        colors : list
            Color palette or list of colors to use for the plot.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        ** kwargs : KeywordArguments
            See `mpl.axes.Axes.plot <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ for more
            information.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        Examples
        ________
        >>> ecap_data.plot("RawE 1", (0,0))     # doctest: +SKIP
        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        calc_y_lim = [0, 0]

        if sort is not None:
            if sort == "ascending":
                sorted_params = (
                    self.parameters.parameters.loc[parameters]
                    .sort_values("pulse amplitude (μA)", ascending=False)
                    .index
                )
            elif sort == "descending":
                sorted_params = (
                    self.parameters.parameters.loc[parameters]
                    .sort_values("pulse amplitude (μA)", ascending=True)
                    .index
                )
            else:
                raise ValueError(
                    "The 'sort' argument must be set to either 'ascending', 'descending', or 'None' (default)."
                )
        else:
            sorted_params = parameters

        for p, c in zip(_to_parameters(sorted_params), colors):
            if method == "mean":
                plot_data = self.mean(p, channels=channel)
                # print(plot_data.shape)
            elif method == "median":
                plot_data = self.median(p, channels=channel)
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'."
                )
            plot_time = self.time(p)

            # compute appropriate y_limits
            if y_lim is None or y_lim == "auto":
                std_data = np.std(plot_data)
                calc_y_lim = [
                    np.min([-std_data * 7, calc_y_lim[0]]),
                    np.max([std_data * 7, calc_y_lim[1]]),
                ]
            elif y_lim == "max":
                calc_y_lim = None
            else:
                calc_y_lim = _to_numeric_array(y_lim)

            plotLABEL = (
                str(self.parameters.parameters.loc[p]["pulse amplitude (μA)"])
                + " (Stim Ch. "
                + str(self.parameters.parameters.loc[p]["channel"])
                + ")"
            )
            ax.plot(
                plot_time, plot_data[0, :], label=plotLABEL, *args, color=c, **kwargs
            )

        ax.set_ylim(calc_y_lim)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")
        ax.legend(loc=1)

        if fig_title is not None:
            ax.set_title(fig_title)

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        if vlines is not None:
            # Add vline at specific sample # -- TODO: Incorporate adding it in at a specific time
            if isinstance(vlines, int):  # For case where only one line is passed
                ax.axvline(vlines * (1 / self.fs), linestyle="--", c="red")
            elif isinstance(vlines, list):
                for line in vlines:
                    ax.axvline(line * (1 / self.fs), linestyle="--", c="red")
            else:
                raise Exception(
                    "Vertical line inputs must be integer (for single line), or a list of integers."
                )

        return _plt_show_fig(fig, ax, show)

    def plot_raster(
        self,
        channel,
        parameters,
        *args,
        method="mean",
        axis=None,
        x_lim=None,
        c_lim="auto",
        c_map="RdYlBu",
        fig_size=(10, 4),
        show=True,
        **kwargs,
    ):
        """
        Generates a raster plot for a given channel and parameters.

        Parameters
        ----------
        channel : str
            Channel to be plotted.
        parameters : list
            List of stimulation parameters.
        * args :  Arguments
            See `mpl.axes.Axes.imshow <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html>`_ for more
            information.
        method : str
            Use 'mean' to plot the mean values and 'median' to plot the median values.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        c_lim : str
            Range of values for the color map to cover. By default, it covers data within +/- 6 standard deviations.
        c_map : str, matplotlib.color.Colormap
            Color map. See https://matplotlib.org/stable/tutorials/colors/colormaps.html.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        ** kwargs : KeywordArguments
            See `mpl.axes.Axes.imshow <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html>`_ for more
            information.


        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        calc_c_lim = [0, 0]

        plot_data = []

        for p in _to_parameters(parameters):
            plot_time = self.time(p)
            if method == "mean":
                p_data = self.mean(p, channels=channel)
            elif method == "median":
                p_data = self.median(p, channels=channel)
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'."
                )

            # compute appropriate y_limits
            if c_lim is None or c_lim == "auto":
                std_data = np.std(p_data)
                calc_c_lim = [
                    np.min([-std_data * 6, calc_c_lim[0]]),
                    np.max([std_data * 6, calc_c_lim[1]]),
                ]
            elif c_lim == "max":
                calc_c_lim = None
            else:
                calc_c_lim = _to_numeric_array(c_lim)

            # add p data to plot data list

            plot_data.append(p_data[0, :])

        plot_data = np.stack(plot_data)

        # Set up plot labels so that axis dimensions are correct
        ax.set_xlabel("time (s)")
        ax.set_ylabel("parameter")

        ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
        ax.set_yticklabels(parameters)
        ax.set_ylim(0, plot_data.shape[0])

        _plt_add_cbar_axis(
            fig, ax, c_label="amplitude (V)", c_lim=calc_c_lim, c_map=c_map
        )

        # if aspect is None:
        #     _plt_ax_aspect(fig, ax)
        #     a_ratio = ((plot_time[-1]-plot_time[0])/(plot_data.shape[0]))*_plt_ax_aspect(fig, ax)
        # else:
        #     a_ratio = ((plot_time[-1]-plot_time[0])/plot_data.shape[0])*aspect

        ax.imshow(
            plot_data,
            *args,
            cmap=c_map,
            extent=[plot_time[0], plot_time[-1], 0, plot_data.shape[0]],
            vmin=calc_c_lim[0],
            vmax=calc_c_lim[1],
            aspect="auto",
            **kwargs,
        )

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        return _plt_show_fig(fig, ax, show)

    def plot_binned_traces(
        self,
        channel,
        parameter,
        bin,
        *args,
        show_mean=False,
        method="mean",
        axis=None,
        opacity=1,
        x_lim=None,
        y_lim="auto",
        colors=sns.color_palette(),
        fig_size=(10, 3),
        show=True,
        fig_title=None,
        vlines=None,
        **kwargs,
    ):
        """
        Plots the raw data response to each individual stimulation pulse over a specified range for a given channel.

        Parameters
        ----------
        channel : str, int
            Channel or channel index to be plotted.
        parameter : tuple
            Stimulation parameter to be plotted.
        bin : tuple, list
            Range of stimulation pulses to be plotted.
        * args : Arguments
            See `mpl.axes.Axes.plot <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html>`_ for more
            information.
        show_mean : bool
            Set to 'True' to also plot the mean response to the specified channel/parameter pair.
        method : str
            Specify whether to plot the 'mean' or 'median' of the aggregate response when show_mean = True.
        opacity : float
            A value between 0 and 1 that adjusts the transparency of the plotted traces.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis, or a matplotlib axis to plot on.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        y_lim : None, str, list, tuple, np.ndarray
            None or 'auto' to automatically calculate reasonable bounds based on standard deviation of data. 'max' to
            plot y axis limits encompassing all accessible data. Otherwise tuple, list, or numpy array of length 2
            containing limits for the y axis.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        fig_title : str
            Adds a user-defined title to the plotted figure.
        vlines : int, list
            Adds a dashed red vertical line at the specified sample(s).
        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)
        calc_y_lim = [0, 0]
        plot_time = self.time(parameter)
        print(_to_parameters(parameter))
        print("Plotting trace #s " + str(bin[0]) + " to " + str(bin[1]))

        # Creates numpy array of binned traces for plotting
        bin_data = self.array(parameter, channel)[bin[0] : bin[1], :, :]

        for data in bin_data:
            ax.plot(plot_time, data[0, :], alpha=opacity)

        if show_mean == True:
            if method == "median":
                avg_trace = self.median(parameter, channel)
            else:
                avg_trace = self.mean(parameter, channel)
            ax.plot(plot_time, avg_trace[0, :], "r")

        if fig_title is not None:
            ax.set_title(fig_title)

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        # compute appropriate y_limits
        if y_lim is None or y_lim == "auto":
            std_data = np.std(bin_data)
            calc_y_lim = [
                np.min([-std_data * 7, calc_y_lim[0]]),
                np.max([std_data * 7, calc_y_lim[1]]),
            ]
        elif y_lim == "max":
            calc_y_lim = None
        else:
            calc_y_lim = _to_numeric_array(y_lim)
        ax.set_ylim(calc_y_lim)

        if vlines is not None:
            # TODO: Change so that this argument uses time instead of sample #
            if isinstance(vlines, int):  # For case where only one line is passed
                # ax[idx].axvline(vlines * self.fs)
                ax.axvline(vlines * (1 / self.fs), linestyle="--", c="red")
            elif isinstance(vlines, list):
                for line in vlines:
                    ax.axvline(line * (1 / self.fs), linestyle="--", c="red")
            else:
                raise Exception(
                    "Vertical line inputs must be integer (for single line), or a list of integers."
                )
        # ax.set_ylim(y_lim)
        return _plt_show_fig(fig, ax, show)

    def multiplot(
        self,
        channels,
        parameters,
        num_cols=1,
        *args,
        method="mean",
        x_lim=None,
        y_lim="auto",
        fig_size=(10, 3),
        show=True,
        show_window=None,
        fig_title=None,
        sort=None,
        save=False,
        vlines=None,
        baseline_ms=None,
        title_fontsize="medium",
        file_name=None,
        **kwargs,
    ):
        """
        Plots multiple epochs for specified channels and parameters.

        Parameters:
        - channels (list or str): List of channel names or a single channel name.
        - parameters (list): List of parameter names.
        - num_cols (int): Number of columns in the subplot grid (default: 1).
        - method (str): Averaging method for plotting data. Options are 'mean' and 'median' (default: 'mean').
        - x_lim (tuple): Tuple specifying the x-axis limits (default: None).
        - y_lim (str or tuple): y-axis limits. Options are 'auto' for automatic calculation, 'max' for maximum range, or a tuple specifying the limits (default: 'auto').
        - fig_size (tuple): Figure size in inches (default: (10, 3)).
        - show (bool): Whether to display the plot (default: True).
        - show_window (bool): Whether to display the neural fiber window on the plot (default: None).
        - fig_title (str): Title of the figure (default: None).
        - sort (str): Sorting order for parameters. Options are 'ascending' for ascending order, 'descending' for descending order (default: None).
        - save (bool): Whether to save the figure (default: False).
        - vlines (int or list): Vertical lines to be added to the plot at specific sample numbers (default: None).
        - baseline_ms (float): Baseline correction window in milliseconds (default: None).
        - title_fontsize (str): Font size of the figure title. Options are 'small', 'medium', 'large' and integar (default: 'medium').
        - file_name (str): Name of the file to save the figure (default: None).
        - **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
        - fig, ax: The figure and axes objects.
        """

        # If a string is passed because the user only specified a single channel, will convert to list before proceeding
        if isinstance(channels, str):
            channels = [channels]

        if show_window == True and len(channels) > 1:
            warnings.warn(
                "Multiple channels passed. Neural fiber window displays with respect to first channel only."
            )

        num_rows = math.ceil(len(parameters) / num_cols)
        fig_width = fig_size[0] * num_cols
        fig_height = fig_size[1] * num_rows

        fig, ax = plt.subplots(
            ncols=num_cols, nrows=num_rows, figsize=(fig_width, fig_height)
        )
        fig.suptitle(fig_title, fontsize=title_fontsize)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.25, wspace=0.15)

        print("Channels: " + str(channels))
        # print('Parameter indices: ' + str(parameters))
        ax = ax.ravel()
        axrange = range(len(parameters))

        if sort is not None:
            if sort == "ascending":
                sorted_params = (
                    self.parameters.parameters.loc[parameters]
                    .sort_values("pulse amplitude (μA)", ascending=False)
                    .index
                )
            elif sort == "descending":
                sorted_params = (
                    self.parameters.parameters.loc[parameters]
                    .sort_values("pulse amplitude (μA)", ascending=True)
                    .index
                )
        else:
            sorted_params = parameters

        for param, idx in zip(_to_parameters(sorted_params), axrange):
            # print('Parameter: ' + str(param))
            # print('Index: ' + str(idx))

            for chan in channels:
                name = (
                    str(self.parameters.parameters["pulse amplitude (μA)"][param])
                    + "μA "
                    + str(param)
                )

                if method == "mean":
                    plot_data = self.mean(param, channels=chan)
                    # print('mean')   #Check to make sure 'if' loop functions
                elif method == "median":
                    plot_data = self.median(param, channels=chan)
                    # print('median')
                elif method == "std":
                    plot_data = self.std(param, channels=chan)
                else:
                    raise ValueError(
                        "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                        "and 'median'."
                    )

                plot_time = self.time(param)
                # Baseline correction
                if baseline_ms is not None:
                    baseline_correction_window = int(baseline_ms * 0.001 * self.fs)
                    baseline = np.mean(
                        plot_data[:, :baseline_correction_window], axis=1
                    )
                    plot_data = plot_data - baseline[:, None]
                # compute appropriate y_limits
                calc_y_lim = [0, 0]
                if y_lim is None or y_lim == "auto":
                    std_data = np.std(plot_data)
                    # print(std_data.compute())
                    calc_y_lim = [
                        np.min([-std_data.compute() * 7, calc_y_lim[0]]),
                        np.max([std_data.compute() * 7, calc_y_lim[1]]),
                    ]
                    # print(calc_y_lim)
                    # calc_y_lim = [-std_data * 7, std_data * 7]
                elif y_lim == "max":
                    calc_y_lim = None
                else:
                    calc_y_lim = _to_numeric_array(y_lim)

                ax[idx].plot(plot_time, plot_data[0, :], label=chan, *args, **kwargs)

                ax[idx].set_ylim(calc_y_lim)
                ax[idx].set_xlabel("time (s)")
                ax[idx].set_ylabel("amplitude (V)")
                ax[idx].legend()
                ax[idx].set_title(name)

                if x_lim is None:
                    ax[idx].set_xlim(plot_time[0], plot_time[-1])
                else:
                    ax[idx].set_xlim(x_lim)

                if vlines is not None:
                    # Add vline at specific sample # -- Later: Incorporate adding it in at a specific time
                    if isinstance(
                        vlines, int
                    ):  # For case where only one line is passed
                        # ax[idx].axvline(vlines * self.fs)
                        ax[idx].axvline(vlines * (1 / self.fs), linestyle="--", c="red")
                    elif isinstance(vlines, list):
                        for line in vlines:
                            ax[idx].axvline(
                                line * (1 / self.fs), linestyle="--", c="red"
                            )
                    else:
                        raise Exception(
                            "Vertical line inputs must be integer (for single line), or a list of integers."
                        )

                # IN DEVELOPMENT -- Display integration windows on plot
                if show_window == True:
                    c_map = "bgrym"
                    label_scale = [0.9, 0.8, 0.7, 0.6, 0.5]

                    if self.neural_window_indicies is not None:
                        # print(self.neural_window_indicies)
                        window_array = self.neural_window_indicies[
                            np.where(self.ts_data._ch_to_index(channels[0]))
                        ]
                        # print(windows)
                        for i, win in enumerate(window_array[0]):
                            start = win[0] / self.fs
                            stop = win[1] / self.fs
                            ax[idx].axvspan(start, stop, alpha=0.1, color=c_map[i])
                            ax[idx].axvspan(start, stop, edgecolor="k", fill=False)

                            # if window_labels == True:
                            #    ax.hlines(label_scale[i] * np.max(y_lim), start, stop, color=c_map[i])
                            #    ax.text(start, label_scale[i] * np.max(y_lim), self.neural_fiber_names[i],
                            #            ha='right', va='center')
                    else:
                        raise ValueError(
                            "Neural window indicies values have not been calculated for this data yet."
                        )
            if save:
                if file_name is None:
                    file_name = fig_title
                plt.savefig(file_name)
        return _plt_show_fig(fig, ax, show)

    def plot_interactive(self, channels, parameter, *args, method="mean", **kwargs):

        """
        Creates an interactive plot of the specified channels and parameter. Hovering over lines in the plot
        displays data point specific information.

        Parameters
        ----------
        channel : str, int
            Channel or channel index to be plotted.
        parameter : tuple
            Stimulation parameter to be plotted.
        method : str
            Specify whether to plot the 'mean' or 'median' of the data.
        """
        # TODO: Make this function work with multiple parameters
        if isinstance(parameter, list):
            raise Exception(
                "Interactive plots can only contain a single parameter (amplitude)."
            )

        if not isinstance(channels, list):
            channels = [channels]

        plotDF = pd.DataFrame()
        nameLIST = []
        plotDF["Time (ms)"] = self.time(parameter) * 1e3
        plotDF["Sample Number"] = plotDF.index
        plotNAME = (
            "Amplitude (uA): "
            + str(self.parameters.parameters["pulse amplitude (μA)"][parameter])
            + " - Parameter ID: "
            + str(parameter)
        )

        # If channels are passed as integers, converts them to string name channels for purpose of figure legends
        if isinstance(channels[0], int):
            ch_names = np.array(self.ch_names)[self._ch_to_index(channels)].tolist()
        else:
            ch_names = channels

        # TODO: Make function work with multiple parameters
        for chan in ch_names:
            if method == "mean":
                plotDF[chan] = self.mean(parameter, channels=chan).T
            elif method == "median":
                plotDF[chan] = self.median(parameter, channels=chan).T
            nameLIST.append(chan)
        fig = px.line(
            plotDF,
            x="Time (ms)",
            y=ch_names,
            title=plotNAME,
            hover_data=["Sample Number"],
        )
        fig.update_xaxes(title_text="Time (ms)")
        fig.update_yaxes(title_text="Voltage (V)")
        fig.show()
        return

    def array(self, parameters, channels=None):
        # Ensure parameters and channels are lists for iteration
        if not isinstance(parameters, list):
            parameters = [parameters]
        if channels is None:
            channels = self.ch_names
        elif not isinstance(channels, list):
            channels = [channels]
        channels = np.where(self._ch_to_index(channels))[0]

        # Prepare a list of dask arrays to compute
        dask_arrays = {}
        data_to_compute = {}
        for parameter in parameters:
            # Check which channels are not in the cache
            channels_to_compute = [
                channel
                for channel in channels
                if self._generate_cache_key(self.array, parameter, channel)
                not in self._cache
            ]
            if channels_to_compute:
                data_to_compute[parameter] = channels_to_compute
                # Compute only the channels that are not in the cache
                dask_arrays[parameter] = self.dask_array(parameter)[
                    :, channels_to_compute, :
                ]

        # Compute all dask arrays at once
        computed_arrays = da.compute(dask_arrays, scheduler="threads", num_workers=8)[0]

        # Store computed arrays in the cache
        for parameter, channels_to_compute in data_to_compute.items():
            for i, channel in enumerate(channels_to_compute):
                key = self._generate_cache_key(self.array, parameter, channel)
                channel_array = computed_arrays[parameter][:, i, :]
                data_size = _get_size(channel_array)  # Get size of the entire array
                # If adding this data will exceed the cache size, remove least recently used items
                while (
                    len(self._cache) > 0
                    and (data_size + sum(_get_size(a) for a in self._cache.values()))
                    > self._cache_size
                ):
                    self._cache.popitem(last=False)
                self._cache[key] = channel_array

        result = {
            parameter: np.stack(
                [
                    self._cache[
                        self._generate_cache_key(self.array, parameter, channel)
                    ]
                    for channel in channels
                ],
                axis=1,
            )
            for parameter in parameters
        }
        if len(parameters) == 1:
            return result[parameters[0]]
        else:
            return result

    # @lru_cache(
    #     maxsize=None
    # )  # Caching this since results are small but computational cost high
    def mean(self, parameters, channels=None):
        """
        Computes an array of mean values of the data from a parameter for each pulse
        across given channels. The result is stored in a cache for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channels : None, str, int, tuple
            Channels or channel indices to include in the array.

        Returns
        -------
        dask.array.core.Array, numpy.ndarray
            Dask array if channels are specified, numpy array if no channels are specified.

        Examples
        ________
        >>> ecap_data.mean  ((0,0), channels = ['RawE 1'])        # doctest: +SKIP
        """

        if not isinstance(parameters, list):
            parameters = [parameters]

        if len(parameters) == 1:
            return np.mean(self.array(parameters, channels=channels), axis=0)
        else:
            return {
                p: np.mean(v, axis=0)
                for p, v in self.array(parameters, channels).items()
            }

    # @lru_cache(
    #     maxsize=None
    # )  # Caching this since results are small but computational cost high
    def median(self, parameters, channels=None):
        """
        Computes an array of median values of the data from a parameter  for each pulse across given channels. The
        result is stored in a cache for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channels : None, str, int, tuple
            Channels or channel indices to include in the array.

        Returns
        -------
        dask.array.core.Array, numpy.ndarray
            Dask array if channels are specified, numpy array if no channels are specified.

        Examples
        ________
        >>> ecap_data.median((0,0), channels = ['RawE 1'])        # doctest: +SKIP
        """
        if not isinstance(parameters, list):
            parameters = [parameters]

        if len(parameters) == 1:
            return np.median(self.array(parameters, channels=channels), axis=0)
        else:
            return {
                p: np.median(v, axis=0)
                for p, v in self.array(parameters, channels).items()
            }

    # @lru_cache(
    #     maxsize=None
    # )  # Caching this since results are small but computational cost high
    def std(self, parameters, channels=None):
        """
        Computes an array of standard deviation values of the data from a parameter across each pulse for the given
        channels. The result is stored in a cache for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channels : None, str, int, tuple
            Channels or channel indices to include in the array.

        Returns
        -------
        dask.array.core.Array, numpy.ndarray
            Dask array if channels are specified, numpy array if no channels are specified.

        Examples
        ________
        >>> ecap_data.std((0,0), channels = ['RawE 1'])        # doctest: +SKIP
        """
        if not isinstance(parameters, list):
            parameters = [parameters]

        if len(parameters) == 1:
            return np.std(self.array(parameters, channels=channels), axis=0)
        else:
            return {
                p: np.std(v, axis=0)
                for p, v in self.array(parameters, channels).items()
            }

    def _time_to_index(self, time, units="seconds"):
        # TODO: calculate index accounting for
        """
        Converts an elapsed time into an index. This index corresponds to the array index of the data point at the
        specified time.

        Parameters
        ----------
        time : int, str
            Elapsed time.
        units : str
            Units of the 'time' parameter. Enter 'seconds', 'milliseconds', or 'microseconds'.


        Returns
        -------
        int
            Array index corresponding to the data at the time input.

        Examples
        ________
        >>> ephys_data._time_to_index(5)
        122070

        """
        # Convert time units once at the beginning.
        if units == "milliseconds":
            time = time / 1e3
        elif units == "microseconds":
            time = time / 1e6

        return np.round(np.multiply(time, self.sample_rate)).astype(int)

    def _ch_to_index(self, channels):
        return self.ts_data._ch_to_index(channels)

    def _first_onset_index(self):
        """
        :return: dataframe of onset indices stores in a  multiindex dataframe of mxn
            where m:conditions, n: amplitude
        """
        fs = self.sample_rate
        df_epocs = pd.copy(self.event_data.parameters[["onset time (s)"]], deep=True)

        # convert onset time to onset index
        df_epocs_idx = round(df_epocs * fs).astype(int)
        df_epocs_idx = df_epocs_idx.rename(
            columns={"onset time (s)": "onset index", "offset time (s)": "offset index"}
        )

        return df_epocs_idx

    def _generate_cache_key(self, function_name, parameter, channel):
        # Create a combined string
        return f"{self._state_identifier}_{function_name}_{parameter}_{channel}"

    def _calc_RMS(self, data, window=None):
        if window is not None:
            RMS = np.sqrt(np.mean(data[window[0] : window[1]] ** 2))
        else:
            RMS = np.sqrt(np.mean(data**2))
        return RMS
