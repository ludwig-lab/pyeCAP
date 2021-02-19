import numpy as np
import matplotlib.pyplot as plt

from .utils.numeric import _to_numeric_array, largest_triangle_three_buckets
from .utils.visualization import _plt_setup_fig_axis, _plt_show_fig, _plt_ax_to_pix, _plt_add_ax_connected_top


class _EventData:
    """
    Class for stimulation event data.
    """
    def __init__(self, events, metadata, indicators=None):
        """

        Parameters
        ----------
        events : list
            List of dictionaries containing  name and an array with stimulation event times.
        metadata : list
            List of dictionaries containing stimulation experiment metadata.
        indicators : None, list
            List of dictionaries containing channel name and a pandas integer array relating each stimulation event to
            the stimulation parameter.
        """
        # TODO: deal with properly incrementing even indicators if there are multiple files
        if not isinstance(events, list):
            events = [events]
        if not isinstance(metadata, list):
            metadata = [metadata]
        if indicators is not None:
            if not isinstance(indicators, list):
                indicators = [indicators]
        self._events = events
        self._metadata = metadata
        self._event_indicators = indicators

    @property
    def metadata(self):
        """
        Property getter method for metadata for the data sets including start and stop time, channel names, number of
        stimulation times, and more.

        Returns
        -------
        list
            list of metadata dictionaries for each stimulation data set.
        """
        return self._metadata

    @property
    def ch_names(self):
        """
        Property getter method for viewing channel names.

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
        Property getter method for experiment start times of stimulation data sets.

        Returns
        -------
        list
            List of start times in seconds since epoch.
        """
        start_times = [meta['start_time'] for meta in self.metadata]
        return start_times

    def events(self, channel, start_times=None, reference=None, remove_gaps=False):
        """
        Outputs a one dimensional array of elapsed times corresponding to stimulation pulses from the specified channel.
        Times are in seconds and the first recorded data set is assumed to start at 0 seconds while other data sets
        start times are specified with the start_times argument.

        Parameters
        ----------
        channel : str
            Channel name.
        start_times : None, list
            Specify a list of start times or specify None to use the start_times property method default.
        reference : None, pyCAP.base.ts_data._TsData or subclass
            If start_times is None, specify a reference object to match start times. This is useful when removing gaps
            between data sets.
        remove_gaps: bool
            Uses the object specified in the 'reference' parameter to take gaps in the data into account. Set to True to
            remove time gaps in the data.

        Returns
        -------
        numpy.ndarray
            Array of elapsed times.
        """
        if hasattr(self, "TDT_delay") and reference is not None:
            offset = self.TDT_delay/reference.sample_rate
        else:
            offset = 0.0

        if start_times is None:
            if reference is None:
                start_times = self.start_times
            else:
                if remove_gaps:
                    start_times = reference.start_indices / reference.sample_rate
                else:
                    start_times = reference.start_times

        if isinstance(channel, str):
            if channel in self.ch_names:
                start_times = [s + offset - start_times[0] for s in start_times]
                events = [e[channel] + s for e, s in zip(self._events, start_times)]
                return np.concatenate(events)
        else:
            raise TypeError("_EventData class can only be indexed using 'str' or 'int' types")

    def event_indicators(self, channel):
        """
        Outputs a numpy array of stimulation parameters for each pulse given a channel name.

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        numpy.ndarray
            Two dimensional array of integers.
        """
        if isinstance(channel, str):
            if channel in self.ch_names:
                # TODO: Make work when there are multiple dio files
                event_indicators = [list(zip(np.asarray(i).repeat(len(e_ind[channel])), e_ind[channel])) for i, e_ind in enumerate(self._event_indicators)]
                return np.concatenate(event_indicators).astype(int)
        else:
            raise TypeError("_DioData class can only be indexed using 'str' or 'int' types")

    def plot_raster(self, axis=None, start_times=None, reference=None, remove_gaps=False, x_lim=None, fig_size=(10, 1.5), show=True, lw=1, **kwargs):
        """
        Plots stimulation data showing the time periods with and without stimulation in raster format.

        Parameters
        ----------
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        start_times : None, list
            List of start times for each data set in timestamp format (seconds since epoch). Leaving this as None will
            default to start times stored in the metadata.
        reference : None, pyCAP.base.ts_data._TsData or subclass
            If start_times is None, specify a reference object to match start times. This is useful when removing gaps
            between data sets.
        remove_gaps: bool
            Uses the object specified in the 'reference' parameter to take gaps in the data into account. Set to True to
            remove time gaps in the data.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        lw : int
            Line width.
        **kwargs : KeywordArguments
            See `mpl.axes.Axes.vlines <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html>`_ for more
            information.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        # plot avx lines in appropriate places
        x_max = 0
        for i, key in enumerate(self.ch_names):
            events = self.events(key, start_times=start_times, reference=reference, remove_gaps=remove_gaps)
            ax.vlines(events, i+0.5, i+1.5, lw=lw, **kwargs)
            if x_lim is None:
                max_ = events[-1]  # assumes events are in order but does not check
                if max_ > x_max:
                    x_max = max_

        # set x_lims via either user defined limits of 0 to last event
        if x_lim is None:
            ax.set_xlim(0,x_max)
        else:
            ax.set_xlim(x_lim[0], x_lim[1])

        # set channel names
        ax.set_yticks(np.arange(len(self.ch_names))+1)
        ax.set_yticklabels(self.ch_names)

        #set y_lims
        ax.set_ylim(0.5, len(self.ch_names)+0.5)

        ax.set_xlabel('time(s)')
        return _plt_show_fig(fig, ax, show)

    def append(self, new_data):
        """
        Creates a new class instance with new data added to the original data.

        Parameters
        ----------
        new_data : _EventData or subclass
            New data to be added.

        Returns
        -------
        _EventData or subclass
            Class instance with new data included.
        """
        if isinstance(new_data, type(self)):
            events = self._events + new_data._events
            metadata = self._metadata + new_data._metadata
            if self._event_indicators is None:
                if new_data._event_indicators is None:
                    indicators = None
                else:
                    indicators = new_data._event_indicators
            else:
                if new_data._event_indicators is None:
                    indicators = self._event_indicators
                else:
                    indicators = self._event_indicators + new_data._event_indicators
            return type(self)(events, metadata, indicators)
        else:
            raise TypeError("Appended data is not of the same type.")

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.ch_names:
                # TODO: Make work when there are multiple dio files
                return self.events[item]
        else:
            raise TypeError("_DioData class can only be indexed using 'str' or 'int' types")
