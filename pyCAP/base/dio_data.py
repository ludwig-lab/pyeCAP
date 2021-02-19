import numpy as np
import matplotlib.pyplot as plt

from .utils.numeric import _to_numeric_array, largest_triangle_three_buckets
from .utils.visualization import _plt_setup_fig_axis, _plt_show_fig, _plt_ax_to_pix, _plt_add_ax_connected_top


class _DioData:
    """
    Class for stimulation start/stop data.
    """
    def __init__(self, dio, metadata, indicators=None):
        """

        Parameters
        ----------
        dio : list
            List of dictionaries containing channel name and stimulation start/stop times in an array.
        metadata : list
            List of dictionaries containing stimulation experiment metadata.
        indicators : None, list
            List of dictionaries containing channel name and pandas integer array that relates the stimulation parameter
            to the starting and stopping times.
        """
        if not isinstance(dio, list):
            dio = [dio]
        if not isinstance(metadata, list):
            metadata = [metadata]
        if indicators is not None:
            if not isinstance(indicators, list):
                indicators = [indicators]
        self._dio = dio
        self._metadata = metadata
        self._dio_indicators = indicators

    @property
    def metadata(self):
        """
        Property getter method for metadata including experiment start and stop times, channel names, and
        stimulation starting times.

        Returns
        -------
        list
            List of dictionaries containing metadata for each data set.
        """
        return self._metadata

    @property
    def ch_names(self):
        """
        Property getter method for channel names.

        Returns
        -------
        list
            List of channel names.
        """
        ch_names = [tuple(meta['ch_names']) for meta in self._metadata]
        if len(set(ch_names)) == 1:
            return list(ch_names[0])
        else:
            raise ValueError("Import data sets do not have consistent channel names.")

    @property
    def start_times(self):
        """
        Property getter method for the experiment start time of each data set.

        Returns
        -------
        list
            List of start times in seconds since epoch.
        """
        start_times = [meta['start_time'] for meta in self.metadata]
        return start_times

    def dio(self, channel, start_times=None, reference=None, remove_gaps=False):
        """
        Outputs an array containing starting and stopping times for stimulation periods for a given channel.

        Parameters
        ----------
        channel : str
            Name of channel.
        start_times : None, list
            Specify a list of start times or specify None to use the start_times property default.
        reference : None, pyCAP.base.ts_data._TsData or subclass
            If start_times is None, specify a reference object to match start times. This is useful when removing gaps
            between data sets.
        remove_gaps: bool
            Uses the object specified in the 'reference' parameter to take gaps in the data into account. Set to True to
            remove time gaps in the data.

        Returns
        -------
        numpy.ndarray
            Array containing start and stop times of the stimulation data.
        """
        if hasattr(self, "TDT_delay") and reference is not None:
            offset = self.TDT_delay / reference.sample_rate
        else:
            offset = 0.0

        if start_times is None:
            if reference is None:
                start_times = self.start_times
            else:
                if remove_gaps:
                    start_times = reference.start_indices/reference.sample_rate
                else:
                    start_times = reference.start_times

        if isinstance(channel, str):
            if channel in self.ch_names:
                start_times = [s + offset - start_times[0] for s in start_times]
                events = [e[channel] + s for e, s in zip(self._dio, start_times)]
                return np.concatenate(events)
        else:
            raise TypeError("_EventData class can only be indexed using 'str' or 'int' types")

    def dio_indicators(self, channel):
        """
        Returns an array of indicators that link stimulation start/stop times to stimulation parameters for a given
        channel.

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        pandas.core.indexes.numeric.Int64Index
            Pandas integer index.

        """
        if isinstance(channel, str):
            if channel in self.ch_names:
                # TODO: Make work when there are multiple dio files
                return self._dio_indicators[0][channel]

    def plot_raster(self, *args, axis=None, start_times=None, reference=None, remove_gaps=False, ch_names=None,
                    x_lim=None, fig_size=(10, 1.5), display='span', show=True, dio=None, **kwargs):
        """
        Plots electrical stimulation time periods in a raster format.

        Parameters
        ----------
        * args : Arguments
            See `mpl.axes.Axes.axvspan <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvspan.html>`_
            and `mpl.axes.Axes.axvline <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvline.html>`_
            for details on plot customization.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        start_times: None, list
            List of start times for each experiment or 'None' to use start times in metadata.
        reference : None, pyCAP.base.ts_data._TsData or subclass
            If start_times is None, specify a reference object to match start times. This is useful when removing gaps
            between data sets.
        remove_gaps: bool
            Uses the object specified in the 'reference' parameter to take gaps in the data into account. Set to True to
            remove time gaps in the data.
        ch_names : None, list
            Names of channels to plot, or None to plot all channels.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        display : str
            Use 'span' or 'lines' to specify the type of raster plot to create.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        dio : None, dict
            Use None to use the dio data stored in the object, or replace with custom data in the same format.
        ** kwargs : KeywordArguments
            See `mpl.axes.Axes.axvspan <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvspan.html>`_
            and `mpl.axes.Axes.axvline <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvline.html>`_
            for details on plot customization.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)
        if ch_names is None:
            ch_names = self.ch_names
        channels = len(self.ch_names)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for k in range(0, len(lst), n):
                yield lst[k:k + n]

        # plot avx lines in appropriate places
        x_max = 0
        for i, key in enumerate(ch_names):
            if dio is None:
                events = self.dio(key, start_times=start_times, reference=reference, remove_gaps=remove_gaps)
            else:
                events = dio[key]
            if len(events) > 0:
                if display == 'span':
                    for event in chunks(events, 2):
                        ax.axvspan(event[0], event[1], *args, ymin=i * 1 / channels, ymax=(i+1) * 1 / channels, **kwargs)
                        ax.axvline(event[0], *args, ymin=i * 1 / channels, ymax=(i+1) * 1 / channels, **kwargs)
                elif display == 'lines':
                    ax.vlines(events[::2], i+0.5, i+1.5)
                else:
                    raise ValueError("Unrecognized value of input 'display'")
                if x_lim is None:
                    max_ = events[-1]  # assumes events are in order but does not check
                    if max_ > x_max:
                        x_max = max_

        # set x_lims via either user defined limits of 0 to last event
        if x_lim is None:
            ax.set_xlim(0, x_max)
        else:
            ax.set_xlim(x_lim[0], x_lim[1])

        # set channel names
        ax.set_yticks(np.arange(len(ch_names)) + 1)
        ax.set_yticklabels(ch_names)

        # set y_lims
        ax.set_ylim(0.5, len(ch_names) + 0.5)

        ax.set_xlabel('time(s)')
        return _plt_show_fig(fig, ax, show)

    def append(self, new_data):
        """
        Creates a class instance with new data added to the original data.

        Parameters
        ----------
        new_data : _DioData or subclass
            New data to be added to the original data.

        Returns
        -------
        _DioData or subclass
            New class instance with new data.
        """
        if isinstance(new_data, type(self)):
            dio = self._dio + new_data._dio
            metadata = self._metadata + new_data._metadata
            if self._dio_indicators is None:
                if new_data._dio_indicators is None:
                    indicators = None
                else:
                    indicators = new_data._dio_indicators
            else:
                if new_data._dio_indicators is None:
                    indicators = self._dio_indicators
                else:
                    indicators = self._dio_indicators + new_data._dio_indicators
            return type(self)(dio, metadata, indicators)
        else:
            raise TypeError("Appended data is not of the same type.")

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.ch_names:
                # TODO: Make work when there are multiple dio files
                return self.dio(item)
        else:
            raise TypeError("_DioData class can only be indexed using 'str' or 'int' types")