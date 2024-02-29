import copy
from datetime import datetime
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from .utils.numeric import _to_numeric_array, largest_triangle_three_buckets
from .utils.visualization import (
    _plt_add_ax_connected_top,
    _plt_ax_to_pix,
    _plt_setup_fig_axis,
    _plt_show_fig,
)


class _DioData:
    """
    Class for stimulation start/stop data.
    """

    def __init__(self, dio, metadata, dio_indicators=None):
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
        # metadata expects ch_names and start_time
        if not isinstance(dio, list):
            dio = [dio]
        if not isinstance(metadata, list):
            metadata = [metadata]
        if dio_indicators is not None:
            if not isinstance(dio_indicators, list):
                dio_indicators = [dio_indicators]
        self._dio = dio
        self._metadata = metadata
        self._dio_indicators = dio_indicators

    @cached_property
    def _state_identifier(self):
        from .utils.base import _generate_state_identifier

        properties = [
            self._dio,
            self._metadata,
            self._dio_indicators,
        ]  # Add more properties to this list if needed
        return _generate_state_identifier(properties)

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
        ch_names = [tuple(meta["ch_names"]) for meta in self._metadata]
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
        start_times = [meta["start_time"] for meta in self.metadata]
        return start_times

    def _ch_to_keys(self, channels):
        """
        Converts channel identifiers (names or indexes) to the channel names.

        Parameters
        ----------
        channels : str, int, or list of str/int
            Channel name(s) or index(es).

        Returns
        -------
        list
            The channel names corresponding to the given identifiers.

        Raises
        ------
        ValueError
            If any of the channels do not exist.
        TypeError
            If the channel identifiers are not of type str, int, or list of str/int.
        """
        if not isinstance(channels, list):
            channels = [channels]

        channel_names = []
        for channel in channels:
            if isinstance(channel, int):
                try:
                    channel_names.append(self.ch_names[channel])
                except IndexError:
                    raise ValueError(f"Channel index {channel} is out of range.")
            elif isinstance(channel, str):
                if channel in self.ch_names:
                    channel_names.append(channel)
                else:
                    raise ValueError(f"Channel name '{channel}' does not exist.")
            else:
                raise TypeError(
                    "Channel identifier must be a string (channel name), an integer (channel index), or a list of such identifiers."
                )

        return channel_names

    def dio(self, channel, start_times=None, reference=None, remove_gaps=False):
        """
        Outputs an array containing starting and stopping times for stimulation periods for a given channel.

        Parameters
        ----------
        channel : str or int
            Name or index of the channel. Only one channel can be specified.
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
        # Ensure channel is a single identifier, not a list
        if isinstance(channel, list):
            raise TypeError(
                "Multiple channels are not supported. Please specify a single channel as a string or integer."
            )

        # Convert channel identifier to channel name
        channel_name = self._ch_to_keys([channel])[
            0
        ]  # _ch_to_keys now always receives a list and returns the first element

        # If start_times is None, use the start_times property method default or specify a reference object to match start times.
        if start_times is None:
            if reference is None:
                start_times = self.start_times
            else:
                if remove_gaps:
                    start_times = reference.start_indices / reference.sample_rate
                else:
                    start_times = reference.start_times

        # Return the events for the channel.
        events = [
            np.asarray(e[channel_name]) + s
            for e, s in zip(self._dio, start_times)
            if e.get(channel_name)
        ]
        if events:  # Check if events list is not empty
            return np.concatenate(events)
        else:
            return np.array([])  # Return an empty numpy array if no events

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

    def plot_raster(self, *args, **kwargs):
        self.plot_dio(*args, **kwargs)

    def plot_dio(
        self,
        *args,
        axis=None,
        start_times=None,
        reference=None,
        remove_gaps=False,
        ch_names=None,
        x_lim=None,
        fig_size=(10, 1.5),
        display="span",
        show=True,
        dio=None,
        **kwargs,
    ):
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
                yield lst[k : k + n]

        # plot avx lines in appropriate places
        for i, key in enumerate(ch_names):
            if dio is None:
                events = self.dio(
                    key,
                    start_times=start_times,
                    reference=reference,
                    remove_gaps=remove_gaps,
                )
            else:
                events = dio[key]
            if len(events) > 0:
                if display == "span":
                    for event in chunks(events, 2):
                        if remove_gaps:
                            ax.axvspan(
                                event[0],
                                event[1],
                                *args,
                                ymin=i * 1 / channels,
                                ymax=(i + 1) * 1 / channels,
                                **kwargs,
                            )
                            ax.axvline(
                                event[0],
                                *args,
                                ymin=i * 1 / channels,
                                ymax=(i + 1) * 1 / channels,
                                **kwargs,
                            )
                        else:
                            ax.axvspan(
                                datetime.fromtimestamp(event[0]),
                                datetime.fromtimestamp(event[1]),
                                *args,
                                ymin=i * 1 / channels,
                                ymax=(i + 1) * 1 / channels,
                                **kwargs,
                            )
                            ax.axvline(
                                datetime.fromtimestamp(event[0]),
                                *args,
                                ymin=i * 1 / channels,
                                ymax=(i + 1) * 1 / channels,
                                **kwargs,
                            )
                elif display == "lines":
                    if remove_gaps:
                        ax.vlines(events[::2], i + 0.5, i + 1.5)
                    else:
                        ax.vlines(datetime.fromtimestamp(events[::2]), i + 0.5, i + 1.5)
                else:
                    raise ValueError("Unrecognized value of input 'display'")

        # set x_lims via either user defined limits of 0 to last event
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])

        if remove_gaps:
            ax.set_xlabel("time(s)")

        # set channel names
        ax.set_yticks(np.arange(len(ch_names)) + 1)
        ax.set_yticklabels(ch_names)

        # set y_lims
        ax.set_ylim(0.5, len(ch_names) + 0.5)

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

    def merge(self, other):
        """
        Merges channels and metadata from two _DioData instances if their start_times are the same and their channels are different.

        Parameters
        ----------
        other : _DioData
            The other _DioData instance to merge with.

        Returns
        -------
        _DioData
            A new _DioData instance with merged data.
        """
        if not isinstance(other, _DioData):
            raise TypeError("The 'other' parameter must be an instance of _DioData.")

        if self.start_times != other.start_times:
            raise ValueError(
                "The start_times of the two _DioData instances must be the same."
            )

        if set(self.ch_names).intersection(set(other.ch_names)):
            raise ValueError(
                "The channels of the two _DioData instances must be different."
            )

        # Merge dio and indicators
        merged_dio = [{**d, **other_d} for d, other_d in zip(self._dio, other._dio)]
        merged_indicators = [
            {**i, **other_i}
            for i, other_i in zip(
                self._dio_indicators or [], other._dio_indicators or []
            )
        ]

        # Merge metadata
        merged_metadata = []
        for self_meta, other_meta in zip(self._metadata, other._metadata):
            merged_ch_names = self_meta["ch_names"] + other_meta["ch_names"]
            merged_meta = {**self_meta, **other_meta, "ch_names": merged_ch_names}
            merged_metadata.append(merged_meta)

        return type(self)(merged_dio, merged_metadata, merged_indicators)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.ch_names:
                # TODO: Make work when there are multiple dio files
                return self.dio(item)
        else:
            raise TypeError(
                "_DioData class can only be indexed using 'str' or 'int' types"
            )

    def remove_events(self, events_to_remove, channels=None, use_timestamps=True):
        """
        Removes events and associated indicators from a copy of the _DioData object based on a list of event times to remove.
        Can remove events from specific channels if the channels parameter is provided. Can operate in two modes:
        direct removal of specified times or removal based on matching timestamps, ensuring matching pairs of on/off events
        and their indicators are always removed together.

        Parameters
        ----------
        events_to_remove : list
            A list of event times to remove from the _DioData object.
        channels : list, optional
            A list of channel names from which events should be removed. If None, events will be removed from all channels.
        use_timestamps : bool, optional
            If True, uses self.dio to convert _dio data to timestamps and removes entries where the timestamps match.
            If False, operates on direct event times removal.

        Returns
        -------
        _DioData
            A new _DioData instance with specified events removed.
        """
        # Manually create a new instance of _DioData
        new_obj = _DioData([], [], [])

        events_to_remove = np.array(events_to_remove)

        if channels is not None:
            channels = self._ch_to_keys(
                channels
            )  # Convert channels to a list of channel names using _ch_to_keys

        for channel in channels if channels is not None else self.ch_names:
            if use_timestamps:
                # Retrieve the timestamps for the channel
                timestamps = self.dio(channel)
                # Determine which timestamps to keep by checking pairs
                keep_pairs = []
                for i in range(0, len(timestamps), 2):  # Process in pairs
                    start, stop = timestamps[i], timestamps[i + 1]
                    # Use np.isin to check if start or stop is in events_to_remove
                    if (
                        not np.isin(start, events_to_remove).any()
                        and not np.isin(stop, events_to_remove).any()
                    ):
                        print((start, stop))
                        keep_pairs.extend([start, stop])
            else:
                # Direct removal mode, not using timestamps
                keep_pairs = None  # This will be used to indicate direct mode

            # Process each dio and dio_indicator for the channel
            for i, dio_channel in enumerate(self._dio):
                if channel in dio_channel:
                    if keep_pairs is not None:
                        # Filter events based on the keep_pairs
                        filtered_events = [
                            event
                            for event in dio_channel[channel]
                            if event in keep_pairs
                        ]
                    else:
                        # Direct removal mode
                        events = dio_channel[channel]
                        filtered_events = []
                        for j in range(0, len(events), 2):
                            if (
                                events[j] not in events_to_remove
                                and events[j + 1] not in events_to_remove
                            ):
                                filtered_events.extend([events[j], events[j + 1]])

                    # Update the new object's _dio
                    if i >= len(new_obj._dio):
                        new_obj._dio.append({})
                    new_obj._dio[i][channel] = filtered_events

                    # Update the new object's _dio_indicators if necessary
                    if (
                        self._dio_indicators
                        and len(self._dio_indicators) > i
                        and channel in self._dio_indicators[i]
                    ):
                        if i >= len(new_obj._dio_indicators):
                            new_obj._dio_indicators.append({})
                        if keep_pairs is not None:
                            filtered_indicators = [
                                self._dio_indicators[i][channel][j // 2]
                                for j, event in enumerate(dio_channel[channel])
                                if event in keep_pairs
                            ]
                        else:
                            indicators = self._dio_indicators[i][channel]
                            filtered_indicators = [
                                indicators[k]
                                for k in range(len(indicators))
                                if 2 * k in range(0, len(filtered_events), 2)
                            ]
                        new_obj._dio_indicators[i][channel] = filtered_indicators

        # Manually copy metadata and other necessary attributes
        new_obj._metadata = [
            dict(meta) for meta in self._metadata
        ]  # Deep copy each metadata dictionary

        return new_obj
