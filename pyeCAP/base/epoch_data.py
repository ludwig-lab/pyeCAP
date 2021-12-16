# Python standard library imports
import sys
import pandas as pd
from functools import lru_cache
import warnings
from cached_property import threaded_cached_property

# Scientific computing package imports
import dask.array as da
from dask import delayed
import numpy as np
import time

# Plotting
from matplotlib import get_backend as plt_get_backend
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
# interactive plotting
from ipywidgets import interact, AppLayout, FloatSlider, VBox, Button, Output

# pyCAP imports
from .ts_data import _TsData
from .event_data import _EventData
from .parameter_data import _ParameterData
from .utils.base import _to_array, _is_iterable
from .utils.numeric import _to_numeric_array, find_first
from .utils.visualization import _plt_setup_fig_axis, _plt_show_fig, _plt_ax_to_pix, _plt_add_ax_connected_top, \
    _plt_ax_aspect, _plt_add_cbar_axis

sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 2.5, "axes.linewidth": 2.0})
sns.set_style("ticks")


def _to_parameters(parameters):
    if isinstance(parameters, tuple) and len(parameters) == 2:
        return [parameters]
    elif _is_iterable(parameters, type=tuple):
        return parameters
    else:
        return _to_array(parameters, dtype='int,int')

# TODO: check if all relevant methods have docstrings
# TODO: edit docstrings to be more informative
class _EpochData:
    """
    Class representing Epoch Data and parent class of ECAP. Contains the methods for the ECAP class.
    """
    def __init__(self, ts_data, event_data, parameters, x_lim='auto'):
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
        if isinstance(ts_data, _TsData) \
                and isinstance(event_data, _EventData) \
                and isinstance(parameters, _ParameterData):
            self.ts_data = ts_data
            self.event_data = event_data
            self.parameters = parameters
            self.x_lim = x_lim
        else:
            raise ValueError("Unrecognized input data types")

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
        if self.x_lim is None or self.x_lim == 'auto':
            return np.arange(0, sample_len) / self.ts_data.sample_rate
        else:
            return (np.arange(0, sample_len) / self.ts_data.sample_rate) - self.x_lim[0]

    # Todo: move to end
    def _first_onset_index(self):
        """
        :return: dataframe of onset indices stores in a  multiindex dataframe of mxn
            where m:conditions, n: amplitude
        """
        fs = self.ts_data.sample_rate
        df_epocs = pd.copy(self.event_data.parameters[['onset time (s)']], deep=True)

        # convert onset time to onset index
        df_epocs_idx = round(df_epocs * fs).astype(int)
        df_epocs_idx = df_epocs_idx.rename(columns={"onset time (s)": "onset index", "offset time (s)": "offset index"})

        return df_epocs_idx

    def epoc_index(self):
        """
        :return: dataframe of onset indices stores in a  multiindex dataframe of mxn
            where m:conditions, n: amplitude
        """
        fs = self.ts_data.sample_rate
        pulse_count = self.event_data.parameters['pulse count']

        # epocs including offset: might have to include this later for checking. Leaving Commented out for now
        df_offset = pd.DataFrame(self.ts_data.start_indices)
        df_offset['offset index'] = df_offset[[0]]
        df_offset = df_offset.rename(columns={0: "onset index"})
        df_epocs = self.event_data.parameters[['onset time (s)', 'offset time (s)']]
        fs = self.ts_data.sample_rate
        # df_epocs['time diff (s)'] = df_epocs['offset time (s)'] - df_epocs['onset time (s)']
        df_epocs_idx = round(df_epocs * fs).astype(int)
        df_epocs_idx = df_epocs_idx.rename(columns={"onset time (s)": "onset index", "offset time (s)": "offset index"})

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
        event_times = []
        # ToDo: modify so data continues to be pulled out across channels if channels are equal between event_data
        #  and ephys data but if channels match between the two datasets then pull out on a perchannel basis
        for ch in self.event_data.ch_names:
            events = self.event_data.events(ch, start_times=self.ts_data.start_indices / self.ts_data.sample_rate)
            event_indicators = self.event_data.event_indicators(ch)
            event_indices = np.logical_and(event_indicators[:, 0] == parameter[0],
                                           event_indicators[:, 1] == parameter[1])
            event_times.append(events[event_indices])
        event_times = np.concatenate(event_times, axis=0)
        if self.x_lim is None or self.x_lim == 'auto':
            min_event_time = np.min(np.diff(event_times))
            sample_len = np.round(min_event_time * self.ts_data.sample_rate)
        else:
            event_times = event_times + self.x_lim[0]
            sample_len = int(np.round((self.x_lim[1] - self.x_lim[0]) * self.ts_data.sample_rate))
        event_times = self.ts_data._time_to_index(event_times)

        indices = np.zeros(self.ts_data.shape[1], dtype=bool)
        for ts in event_times:
            te = int(ts + sample_len)
            indices[ts:te] = True

        first_onset_idx = find_first(True, indices)
        last_onset_idx = len(indices) - find_first(True, np.flip(indices)) - 1
        removal_idx = np.logical_not(indices[first_onset_idx:last_onset_idx + 1]).nonzero()

        event_data = self.ts_data.array[:, first_onset_idx:last_onset_idx + 1]
        idx_mask = np.repeat(indices[first_onset_idx:last_onset_idx + 1][np.newaxis,:], self.ts_data.shape[0], axis=0)

        # ToDo: Create Heuristic model with best way to extract data if it is tightly packed or not
        if len(removal_idx[0]) > 0:
            # ToDo: rewrite so that this happens blockwise in dask as opposed to all at once to speed up.
            event_data = event_data[idx_mask].compute_chunk_sizes()

        event_data_reshaped = da.reshape(event_data, (self.ts_data.shape[0], len(event_times), int(sample_len)))
        return da.moveaxis(event_data_reshaped, 1, 0)

    def plot(self, axis=None, channels=None, x_lim=None, y_lim='auto', ch_labels=None, colors=sns.color_palette(),
             fig_size=(12, 3), show=True):
        """
        Plotting method for ECAP data.

        Parameters
        ----------
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis, or a matplotlib axis to plot on.
        channels :  int, str, list, tuple, np.ndarray
            Channels to plot. Can be a boolean numpy array with the same length as the number of channels, an integer
            array, or and array of strings containing channel names.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        y_lim : None, str, list, tuple, np.ndarray
            None or 'auto' to automatically calculate reasonable bounds based on standard deviation of data. 'max' to
            plot y axis limits encompassing all accessible data. Otherwise tuple, list, or numpy array of length 2
            containing limits for the y axis.
        ch_labels : list, tuple, np.ndarray
            Stings to use as channel labels in the plot. Must match length of channels being displayed.
        colors : list
            Color palette or list of colors to use for channels.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool, str
            String 'notebook' to plot interactively in a jupyter notebook or boolean value indicating if the plot should
            be displayed.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.
        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        # Get channels to plot
        if channels is None:
            channels = slice(None, None, None)
        else:
            channels = self._ch_to_index(channels)

        return _plt_show_fig(fig, ax, show)

    def plot_channel(self, channel, parameters, *args, method='mean', axis=None, x_lim=None, y_lim='auto',
                     colors=sns.color_palette(), fig_size=(10, 3), show=True, **kwargs):
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
        print(_to_parameters(parameters))
        for p, c in zip(_to_parameters(parameters), colors):
            if method == 'mean':
                plot_data = self.mean(p, channels=channel)
            elif method == 'median':
                plot_data = self.median(p, channels=channel)
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'.")
            plot_time = self.time(p)

            # compute appropriate y_limits
            if y_lim is None or y_lim == 'auto':
                std_data = np.std(plot_data)
                calc_y_lim = [np.min([-std_data * 6, calc_y_lim[0]]),
                              np.max([std_data * 6, calc_y_lim[1]])]
            elif y_lim == 'max':
                calc_y_lim = None
            else:
                calc_y_lim = _to_numeric_array(y_lim)

            ax.plot(plot_time, plot_data[0, :], *args, color=c, **kwargs)

        ax.set_ylim(calc_y_lim)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (V)')

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        return _plt_show_fig(fig, ax, show)

    def plot_raster(self, channel, parameters, *args, method='mean', axis=None, x_lim=None, c_lim='auto',
                    c_map='RdYlBu', fig_size=(10, 4), show=True, **kwargs):
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
            if method == 'mean':
                p_data = self.mean(p, channels=channel)
            elif method == 'median':
                p_data = self.median(p, channels=channel)
            else:
                raise ValueError(
                    "Unrecognized value received for 'method'. Implemented averaging methods include 'mean' "
                    "and 'median'.")

            # compute appropriate y_limits
            if c_lim is None or c_lim == 'auto':
                std_data = np.std(p_data)
                calc_c_lim = [np.min([-std_data * 6, calc_c_lim[0]]),
                              np.max([std_data * 6, calc_c_lim[1]])]
                print((p_data[0, :].shape, std_data, calc_c_lim))
            elif c_lim == 'max':
                calc_c_lim = None
                print((p_data[0, :].shape))
            else:
                calc_c_lim = _to_numeric_array(c_lim)

            # add p data to plot data list

            plot_data.append(p_data[0, :])

        plot_data = np.stack(plot_data)

        # Set up plot labels so that axis dimensions are correct
        ax.set_xlabel('time (s)')
        ax.set_ylabel('parameter')

        ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
        ax.set_yticklabels(parameters)
        ax.set_ylim(0, plot_data.shape[0])

        _plt_add_cbar_axis(fig, ax, c_label='amplitude (V)', c_lim=calc_c_lim, c_map=c_map)

        # if aspect is None:
        #     _plt_ax_aspect(fig, ax)
        #     a_ratio = ((plot_time[-1]-plot_time[0])/(plot_data.shape[0]))*_plt_ax_aspect(fig, ax)
        # else:
        #     a_ratio = ((plot_time[-1]-plot_time[0])/plot_data.shape[0])*aspect

        ax.imshow(plot_data, *args, cmap=c_map, extent=[plot_time[0], plot_time[-1], 0, plot_data.shape[0]],
                  vmin=calc_c_lim[0], vmax=calc_c_lim[1], aspect='auto', **kwargs)

        if x_lim is None:
            ax.set_xlim(plot_time[0], plot_time[-1])
        else:
            ax.set_xlim(x_lim)

        return _plt_show_fig(fig, ax, show)

    @lru_cache(maxsize=None)
    def array(self, parameter, channels=None):
        """
        Returns a numpy or dask array of the raw data with the specified parameter and channels. The array will contain
        the time series data with one dimension representing the pulse and another representing the channel. The remaining
        dimension specifies the number of data points in each pulse/channel combination. The result is stored in a cache
        for faster computing.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channels : None, str, int, tuple
            Channels or channel indices to include in the array.

        Returns
        -------
        numpy.ndarray, dask.array.core.
            Three dimensional array containing raw Ephys data.

        Examples
        ________
        >>> ecap_data.array((0,0), channels = ['RawE 1'])        # doctest: +SKIP
        """
        if channels is None:
            return self.dask_array(parameter).compute()
        else:
            return self.dask_array(parameter)[:, self.ts_data._ch_to_index(channels), :]

    @lru_cache(maxsize=None)  # Caching this since results are small but computational cost high
    def mean(self, parameter, channels=None):
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
        return np.mean(self.array(parameter, channels=channels), axis=0)

    @lru_cache(maxsize=None)  # Caching this since results are small but computational cost high
    def median(self, parameter, channels=None):
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
        return np.median(self.array(parameter, channels=channels), axis=0)

    @lru_cache(maxsize=None)  # Caching this since results are small but computational cost high
    def std(self, parameter, channels=None):
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
        return np.std(self.array(parameter, channels=channels), axis=0)
