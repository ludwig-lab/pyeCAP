import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import copy
import pandas as pd
import dask as da

from .ts_data import _TsData
from .dio_data import _DioData
from .parameter_data import _ParameterData
from .utils.visualization import _plt_add_ax_connected_top, _plt_setup_fig_axis
from functools import lru_cache


class _DioEpoch:
    """Class for analyzing data based on stimulation onset and offset times"""

    def __init__(self, ts_data, dio_data, trigger_channel, parameter_data=None, threshold=1, time_difference=0,
                 search=1, compute=True, trigger_offsets=None):
        """
        Constructor for the _DioEpoch class.

        Parameters
        ----------
        ts_data : _TsData or subclass
            Time series data such as Phys or Ephys data.
        dio_data : _DioData or subclass
            Data referencing stimulation start and stop times.
        trigger_channel : str
            Channel that contains data for stimulation trigger pulses. Used to sync the Phys and Stim data times.
        parameter_data : _ParameterData or subclass, None
            Data referencing stimulation parameters. Use None if dio_data and parameter_data reference the same object.
        threshold : float
            Minimum value to register as a stimulation pulse.
        time_difference : float
            Time difference in seconds between the clocks of the computers that recorded the stimulation and
            physiological data. A time_difference of 3600 would correct an error where the time of the phys_data leads
            the stim_data by 1 hour.
        search : float
            Maximum time difference to search for a pulse in the trigger channel in seconds. With a search of 1, pyCAP
            will search all data points between 1 second before and after the time given by the stimulation data.
        compute : bool
            True to compute trigger offsets for each pulse when instantiating, False to ignore trigger offsets, or pass
            in custom trigger offsets
        trigger_offsets : dict
            Dictionary matching parameter with trigger offset. Only used if compute is False or trigger_channel is None.
        """
        self._time_difference = time_difference
        self._threshold = threshold
        self._trigger_channel = trigger_channel
        self._search = search

        if isinstance(ts_data, _TsData) and isinstance(dio_data, _DioData):
            self.ts_data = ts_data
            self.dio_data = dio_data
            self.parameters = parameter_data
        else:
            raise ValueError("Unrecognized stimulation and physiological data types")
        if parameter_data is None and isinstance(dio_data, _ParameterData):
            self.parameters = dio_data
        elif isinstance(parameter_data, _ParameterData):
            self.parameters = parameter_data
        else:
            raise ValueError("Unrecognized stimulation and physiological data types")

        if compute and trigger_channel is not None:
            t_o = self._initialize_all_pulses()
            self._trigger_offsets = dict(zip(list(self.parameters.parameters.index), t_o))
        elif not compute and trigger_offsets is not None:
            self._trigger_offsets = trigger_offsets
        else:
            self._trigger_offsets = {p: 0 for p in self.parameters.parameters.index}

    def get_ts_data(self):
        """
        Getter method for the time series data.

        Returns
        -------
        _TsData or subclass
            Time series data.

        Examples
        ________
        >>> response_data.get_ts_data()     # doctest: +SKIP
        """
        return self.ts_data

    def get_stim_data(self):
        """
        Getter Method for stimulation data. Specifically returns the dio_data used to create the object.

        Returns
        -------
        _DioData or subclass
            Stimulation data.
        """
        return self.dio_data

    def get_parameter_data(self):
        """
        Getter method for stimulation parameter data in DataFrame format.

        Returns
        -------
        pandas.core.frame.DataFrame
            Pandas DataFrame containing stimulation parameters.
        """
        # TODO: it may be useful to use this method in some of the code below
        return self.parameters.parameters

    def get_time_difference(self):
        """
        Getter method for the user-specified time difference between the Physiological and Stimulation data sets.

        Returns
        -------
        int, float
            Time difference in seconds
        """
        return self._time_difference

    def get_trigger_channel(self):
        """
        Getter method for the user-specified stimulation trigger channel.

        Returns
        -------
        str
            Channel name.
        """
        return self._trigger_channel

    def get_threshold(self):
        """
        Getter method for the user-specified threshold to define a stimulation pulse.

        Returns
        -------
        int, float
            Stimulation threshold.
        """
        return self._threshold

    def get_trigger_offsets(self, parameters=None):
        """
        Getter method for stimulation trigger offsets (in seconds). Trigger offsets are defined as the time difference
        between the detected pulse in the trigger channel and the stimulation data. A trigger offset of 0.5 means that
        that the detected pulse leads the stimulation data by 0.5 seconds.

        Parameters
        ----------
        parameters : None, list
            List of parameters to get the trigger offsets for, or None to get the trigger offsets of all parameters.

        Returns
        -------
        dict
            Python dictionary with keys as parameters and values as trigger offsets.
        """
        if parameters is None:
            return self._trigger_offsets
        else:
            return {key: value for key, value in self._trigger_offsets if key in parameters}

    def get_search(self):
        """
        Getter method for the user-specified search window for stimulation trigger pulses.

        Returns
        -------
        int, float
            Maximum time difference to search for a pulse in the trigger channel in seconds.
        """
        return self._search

    def set_trigger_channel(self, new_channel):
        """
        Resets the trigger channel to new_channel. Warning : This will re-compute the trigger offsets which may be time
        consuming and will reset any user-specified trigger offsets.

        Parameters
        ----------
        new_channel : str
            Channel from the physiological data to search for

        Returns
        -------
        _DioEpoch or subclass
            New class instance with a different trigger_channel and recomputed trigger offsets.
        """
        return type(self)(self.ts_data, self.dio_data, new_channel, threshold=self._threshold,
                          time_difference=self._time_difference, search=self._search)

    def set_threshold(self, new_threshold):
        """
        Resets the threshold for stimulation detection to new_threshold. Warning : This will re-compute the trigger
        offsets which may be time consuming and will reset any user-specified trigger offsets.

        Parameters
        ----------
        new_threshold : int, float
            New Threshold to detect stimulation pulses.

        Returns
        -------
        _DioEpoch or subclass
            New class instance with a different threshold and recomputed trigger offsets.

        Examples
        ________
        >>> # create a new object with a threshold of 3.
        >>> new_response_data = response_data.set_threshold(3)      # doctest: +SKIP
        """
        return type(self)(self.ts_data, self.dio_data, self._trigger_channel, threshold=new_threshold,
                          time_difference=self._time_difference, search=self._search)

    def set_time_difference(self, new_time_difference):
        """
        Resets time difference between the physiological and stimulation data sets. Warning : This will re-compute
        the trigger offsets which may be time consuming and will reset any user-specified trigger offsets.

        Parameters
        ----------
        new_time_difference : int, float
            New time difference to reset the time difference.

        Returns
        -------
        _DioEpoch or subclass
            New class instance with a different time_difference and recomputed trigger offsets.
        """
        return type(self)(self.ts_data, self.dio_data, self._trigger_channel, threshold=self._threshold,
                          time_difference=new_time_difference, search=self._search)

    @lru_cache(maxsize=None)
    def set_trigger_offsets(self, parameters, offsets):
        """
        Resets trigger offsets for the specified parameters with the specified offsets. Parameters and offsets must line
        up by index in order for them to be matched properly.

        Parameters
        ----------
        parameters : list
            List of parameters to reset trigger offsets for.
        offsets : list
            List of offsets to use when resetting the parameters.

        Returns
        -------
        _DioEpoch or subclass
            New class instance with the specified trigger offsets changed.

        Examples
        ________
        >>> # create a new object with trigger offsets reset to 0.5s for parameters (0,0), (0,1), and (0,2).
        >>> new_response_data = response_data.set_trigger_offsets([(0,0), (0,1), (0,2)], [0.5, 0.5, 0.5])      # doctest: +SKIP
        """
        new_trigger_offsets = copy.deepcopy(self._trigger_offsets)
        for idx, p in enumerate(parameters):
            new_trigger_offsets[p] = offsets[idx]
        return type(self)(self.ts_data, self.dio_data, self._trigger_channel, threshold=self._threshold,
                          time_difference=self._time_difference, search=self._search, compute=False,
                          trigger_offsets=new_trigger_offsets)

    def set_search(self, new_window):
        """
        Resets search window to new_window. The search window is defined as the maximum time difference relative to the
        stimulation data to search for a trigger pulse. Warning : This will re-compute the trigger offsets which may be
        time consuming and will reset any user-specified trigger offsets.

        Parameters
        ----------
        new_window : int, float
            New search window.

        Returns
        -------
        _DioEpoch or subclass
            New class instance with a different search and recomputed trigger offsets.
        """
        return type(self)(self.ts_data, self.dio_data, self._trigger_channel, threshold=self._threshold,
                          time_difference=self._time_difference, search=new_window)

    def parameter_time(self, parameter):
        """
        Finds the start and end time of the given parameter. Times are in seconds relative to the start time of the
        physiological data.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.

        Returns
        -------
        tuple
            Start time followed by end time in a python tuple.

        Examples
        ________
        >>> # get start time and end time of the parameter (0,0)
        >>> start_time, end_time = response_data.parameter_time((0,0))      # doctest: +SKIP
        """
        # get onset/offset times, and errors due to computer time syncing
        trigger_offset = self._trigger_offsets[parameter]
        if np.isnan(trigger_offset):
            trigger_offset = 0
        phys_start_time = self.ts_data.start_times[0]
        stim_start_time = self.dio_data.start_times[parameter[0]] + self._time_difference
        onset_time = stim_start_time + self.parameters.parameters.loc[parameter, "onset time (s)"] - phys_start_time
        offset_time = stim_start_time + self.parameters.parameters.loc[parameter, "offset time (s)"] - phys_start_time

        # compute start/end indices and times
        start_index = self.ts_data._time_to_index(onset_time, remove_gaps=False)
        end_index = self.ts_data._time_to_index(offset_time, remove_gaps=False)
        start_time = start_index / self.ts_data.sample_rate + trigger_offset
        end_time = end_index / self.ts_data.sample_rate + trigger_offset

        return start_time, end_time

    def array(self, parameter, onset=0, offset=0, channels=None, delayed=False):
        """
        Generates an array of raw physiological data from the specified parameter with the specified onset and offset.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        onset : int, float
            Time to include before the stimulation start time (-2.0 will include 2.0 seconds of data before stimulation
            onset).
        offset : int, float
            Time to include after the stimulation end time (2.0 will include 2.0 seconds of data after the stimulation
            ending).
        channels : str, list
            Channel name or list of channel names to include in the data array

        Returns
        -------
        dask.array.core.Array
            Array of raw time series data.

        Examples
        ________
        >>> # return array of raw data for the parameter (0,0) across all channels and include 5 seconds of data before
        >>> # the start time of the parameter.
        >>> response_data.array((0,0), onset=-5, offset=0)      # doctest: +SKIP
        """
        # slice time series data by channels
        if channels is None:
            channels = slice(None, None, None)
        else:
            channels = self.ts_data._ch_to_index(channels)

        # slice time series data by start/end time
        start_time, end_time = self.parameter_time(parameter)
        start_idx = self.ts_data._time_to_index(start_time + onset)
        end_idx = self.ts_data._time_to_index(end_time + offset)
        array_indices = slice(start_idx, end_idx, None)
        if delayed is False:
            return self.ts_data.array[channels, array_indices]
        else:
            return da.delayed(self.ts_data.array[channels, array_indices])

    def time(self, parameter, onset=0, offset=0):
        """
        Creates an array of points representing time where each point corresponds to a data point in the raw data array.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        onset : int, float
            Time to include before the stimulation start time (-2.0 will include 2.0 seconds of data before stimulation
            onset).
        offset : int, float
            Time to include after the stimulation end time (2.0 will include 2.0 seconds of data after the stimulation
            ending).

        Returns
        -------
        numpy.ndarray
            Array of time points in seconds corresponding to each data point starting at the onset time.
        """
        # get number of data points by calling the dio_array method
        dio_length = self.array(parameter=parameter, onset=onset, offset=offset).shape[1]
        # return an array starting at the onset time, incremented by the time between samples
        return np.add(np.arange(0, dio_length) / self.ts_data.sample_rate, onset)

    def plot_parameter(self, parameter, channels=None, onset=-1, offset=1, show=True, events=False, axis=None,
                       fig_size=(10, 6), **kwargs):
        """
        Plots a parameter with the given channels and onset and offset times.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channels : str, list
            Channel name or list of channel names to plot.
        onset : int, float
            Time to include before the stimulation start time (-2.0 will include 2.0 seconds of data before stimulation
            onset).
        offset : int, float
            Time to include after the stimulation end time (2.0 will include 2.0 seconds of data after the stimulation
            ending).
        show : bool
            Set to True to display the plot, False to return the axis and not display the plot.
        events : bool
            Set to True to display stimulation event data alongside the parameter data, False to omit event data.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        **kwargs : KeywordArguments
            See :ref:`_TsData (parent class)` for more details.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot, None
            Returns subplot and plots nothing, or returns nothing and displays the plot.

        Examples
        ________
        >>> # plot the data for Channel 7 across parameter (0,0) with stimulation raster shown
        >>> response_data.plot_parameter((0,0), channels=[0], events=True)        # doctest: +SKIP
        """
        # get start/end times
        start_time, end_time = self.parameter_time(parameter=parameter)
        elapsed_time = end_time + offset - start_time - onset
        # call the _TsData plotting method, but return the axis instead of showing the plot
        fig, ax = _plt_setup_fig_axis(axis=axis, fig_size=fig_size)
        self.ts_data.plot(axis=ax, channels=channels, x_lim=(start_time + onset, end_time + offset), y_lim='max',
                          show=False, **kwargs)

        # reset the tick labels to reflect time 0 being stimulation start time
        order = int(math.log10(elapsed_time))
        # ensure that the tick steps are of sufficient size based on elapsed time
        step_size = int(elapsed_time * 10 ** (-1 * order)) * 10 ** (order - 1)
        ticklabels = np.round(np.arange(onset, onset + elapsed_time, step_size), decimals=-1 * order + 3)
        ticks = np.add(ticklabels, start_time)
        plt.xticks(ticks=ticks, labels=ticklabels)

        # handle event data plotting
        if events:
            top_ax = _plt_add_ax_connected_top(fig, fig.axes[1])
            self.plot_dio(axis=top_ax, show=False, color='orange', zorder=-1)
            top_ax.set_xlim((start_time + onset, end_time + offset))

        if show:
            plt.show()
        else:
            return fig, fig.axes[0]

    def baseline(self, parameter, channel, first_onset=-3, second_onset=-1):
        """
        Calculates baseline values for a parameter from the period before stimulation given the starting and ending
        onset times before stimulation occurs.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channel : str
            Channel name.
        first_onset : int, float
            Time to start measuring baseline before stimulation onset (Negative numbers reference times before
            stimulation onset).
        second_onset : int, float
            Time to finish measuring baseline before stimulation onset (Negative numbers reference times before
            stimulation onset).

        Returns
        -------
        numpy.ndarray
            Array of floats representing the baseline for each channel.

        Examples
        ________
        >>> # get baseline for Channel 7 of parameter (0,0) for the time period 5 to 1 seconds prior to stimulation
        >>> baseline = response_data.baseline((0,0), channel=0, first_onset=-5)       # doctest: +SKIP
        """
        # get array of data to compute baseline from, find and return the mean
        start_time, stop_time = self.parameter_time(parameter)
        elapsed_time = stop_time - start_time
        data = self.array(parameter=parameter, onset=first_onset, offset=second_onset - elapsed_time, channels=channel)
        baseline = float(np.nanmean(data, axis=1))
        return baseline

    def delta(self, parameter, channel, first_onset=-3, second_onset=0, offset=0, method=None):
        """
        Calculates changes in values at the specified channels due to stimulation. First calculates a baseline from the
        onset times, then compares the baseline to a maximum or minimum value across the stimulation period.

        Parameters
        ----------
        parameter : tuple
            Stimulation parameter. Composed of index for the data set and index for the stimulation.
        channel : str
            Channel name.
        first_onset : int, float
            Time to start measuring baseline before stimulation onset (Negative numbers reference times before
             stimulation onset).
        second_onset : int, float
            Time to finish measuring baseline before stimulation onset (Negative numbers reference times before
             stimulation onset).
        offset : int, float
            Time after stimulation period finishes to search for a maximum(positive numbers reference times after
             stimulation ends).
        method : None, str
            Use "maximum" to find the maximum and "minimum" to find the minimum. Default None will find the largest
            change which could be maximum or minimum.

        Returns
        -------
        dict
            Dictionary containing baseline, extreme value, and absolute change.

        Examples
        ________
        >>> # get largest change from Channel 7 on parameter (0,0)
        >>> delta_dict = response_data.baseline((0,0), "Channel 7", first_onset=-5)       # doctest: +SKIP
        >>> change = delta_dict["absolute change"]      # doctest: +SKIP
        """

        # create empty dictionary, fill in with baseline
        delta_dict = {}
        data = self.array(parameter, channels=channel, onset=first_onset, offset=offset, delayed=True)
        on_idx = self.ts_data._time_to_index(first_onset)
        start_idx = self.ts_data._time_to_index(second_onset)
        shift_idx = start_idx - on_idx
        baseline = np.mean(data[:, 0:shift_idx], axis=1).compute()[0]
        data = data[:, shift_idx:].compute()
        delta_dict["baseline"] = baseline

        highest = np.nanmax(data)
        lowest = np.nanmin(data)
        # compute change based on method
        if method == "maximum":
            delta_dict["extreme"] = highest
            delta_dict["absolute change"] = highest - baseline
        elif method == "minimum":
            delta_dict["extreme"] = lowest
            delta_dict["absolute change"] = lowest - baseline
        elif method is None:
            delta_dict["extreme max"] = highest
            delta_dict["extreme min"] = lowest
            delta_dict["absolute change max"] = highest - baseline
            delta_dict["absolute change min"] = lowest - baseline

        else:
            raise (ValueError("Unrecognized method type"))

        return delta_dict

    def verify_parameters(self, parameters=None, baseline=3, plot=False):
        """
        Method to verify that stimulation parameters are referring to the correct time in the physiological data set.
        This method will generate warnings for inadequate baseline, inability to find a trigger pulse, or interruptions
        by start and end time of blocks. This method will plot all of the parameters in the order specified to aid in
        debugging.

        Parameters
        ----------
        parameters : list
            List of parameters to verify.
        baseline : int, float
            Seconds of baseline to verify. A baseline of 3 will verify that there are 3 seconds of uninterrupted
            baseline before the parameter start time.
        plot : bool
            Set to True to create plot figures for each parameter and store them in the output dictionary.

        Returns
        -------
        dict
            Dictionary containing verification information.

        Examples
        ________
        >>> # check for any errors in the parameters (0,0) and (0,1) with a 5 second baseline period
        >>> response_data.verify_parameters([(0,0), (0,1)], baseline=5)     # doctest: +SKIP
        """
        # initialize parameter verification
        missing_triggers = []
        block_interruptions = []
        baseline_interruptions = []
        baseline_overlap = []

        if parameters is None:
            parameters = list(self.parameters.parameters.index)

        stim_end_times = [self.parameter_time(p)[1] for p in self.parameters.parameters.index]

        for parameter in parameters:
            # find initial parameters
            start_time, end_time = self.parameter_time(parameter)
            phys_start_time = self.ts_data.start_times[0]
            stim_start_time = self.dio_data.start_times[parameter[0]] + self._time_difference
            onset_time = stim_start_time + self.parameters.parameters.loc[parameter, "onset time (s)"] - phys_start_time
            offset_time = stim_start_time + self.parameters.parameters.loc[parameter, "offset time (s)"] - phys_start_time

            # check for trigger pulse
            if np.isnan(self._trigger_offsets[parameter]):
                missing_triggers.append(parameter)

            # check for block start/end time interruption
            in_block = False
            for i in range(self.ts_data.ndata):
                if self.ts_data.start_times[i] - self.ts_data.start_times[0] < onset_time < offset_time < \
                        self.ts_data.end_times[i] - self.ts_data.start_times[0]:
                    in_block = True
            if not in_block:
                block_interruptions.append(parameter)

            # check for inadequate baseline
            for st in self.ts_data.start_indices / self.ts_data.sample_rate:
                if start_time - baseline < st < start_time:
                    baseline_interruptions.append(parameter)
            for et in stim_end_times:
                if start_time - baseline < et < start_time:
                    baseline_overlap.append(parameter)

        # generate warnings
        warnings.warn("Missing trigger pulse in parameters : {}".format(missing_triggers))
        warnings.warn("Stimulation period interruption by start/end of block in parameters: {}".format(block_interruptions))
        warnings.warn("Baseline period interruption by start/end of block in parameters: {}".format(baseline_interruptions))
        warnings.warn("Baseline period overlap with other stimulation in parameters: {}".format(baseline_overlap))

        warnings_dict = {"Missing Triggers": missing_triggers, "Stimulation Interruption": block_interruptions,
                         "Baseline Interruptions": baseline_interruptions, "Stim During Baseline": baseline_overlap}

        # return dictionary of parameter information
        if plot:
            warnings_dict["Plot Figures"] = []
            for p in parameters:
                fig, ax = self.plot_parameter(p, onset=(-1 * baseline), offset=5, show=False, events=True)
                plt.title(p)
                warnings_dict["Plot Figures"].append(fig)
            return warnings_dict
        else:
            return warnings_dict

    def dio(self, channel):
        """
        Outputs an array containing starting and stopping times for stimulation periods for a given channel. Start/stop
        times are relative to the start of the time series data object with no gaps between data sets.

        Parameters
        ----------
        channel : str
            Name of channel.

        Returns
        -------
        numpy.ndarray
            Array containing start and stop times of the stimulation data.

        See Also
        ________
        pyCAP.base.dio_data._DioData.dio
         """
        # TODO: test this method on data with multiple channels
        indicators = self.dio_data.dio_indicators(channel)
        new_dio = np.ndarray((0,))
        for i in range(len(self.dio_data.metadata)):
            for j in range(len(indicators) // 2):
                start_time, end_time = self.parameter_time((i, j))
                new_dio = np.append(new_dio, [start_time, end_time])
        return new_dio

    def plot_dio(self, *args, axis=None, ch_names=None, x_lim=None, fig_size=(10, 1.5), display='span', show=True,
                 **kwargs):
        """
        Plots electrical stimulation time periods in a raster format with start/stop times relative to the start of the
        time series data set with no gaps.

        Parameters
        ----------
        * args : Arguments
            See `mpl.axes.Axes.axvspan <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvspan.html>`_
            and `mpl.axes.Axes.axvline <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvline.html>`_
            for details on plot customization.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
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
        ** kwargs : KeywordArguments
            See `mpl.axes.Axes.axvspan <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvspan.html>`_
            and `mpl.axes.Axes.axvline <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvline.html>`_
            for details on plot customization.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        See Also
        ________
        pyCAP.base.dio_data._DioData.plot_raster
        """
        # create custom data to pass
        new_dio = {}
        for channel in self.dio_data.ch_names:
            new_dio[channel] = self.dio(channel)

        # plot the data using the dio data raster plot method
        self.dio_data.plot_dio(*args, axis=axis, ch_names=ch_names, x_lim=x_lim, fig_size=fig_size, display=display,
                               show=show, dio=new_dio, **kwargs)

    def create_all_phys_dataframe(self, calc_type=min):

        master_list = []
        for idx in self.dio_data.parameters.index.to_list():
            for chan in self.ts_data.ch_names:
                columns = self.dio_data.parameters.columns
                unique_columns = [c for c in columns if len(self.data_phys_response.dio_data.parameters[c].unique()) > 1]
                key_vals = self.dio_data.parameters.loc[idx, unique_columns]

                if key_vals['pulse amplitude (μA)'] < 0:
                    key_vals['pulse amplitude (μA)'] *= -1
                master_list.append([self.delta(parameter=idx, channel=chan, method='maximum')['absolute change'], chan,
                                    *key_vals])
        self.master_df = pd.DataFrame(master_list, columns=['delta', "Recording Channel", "Stimulation Amplitude", "Stimulation Channel", "Condition", "Stimulation Configuration"])
        # bag = da.bag.from_delayed(self.master_df['delta'].tolist())
        # deltas = bag.compute()
        # self.master_df.drop(columns='delta')
        # self.master_df['delta'] = deltas

    def _expected_indices(self, parameter):
        # returns the expected indices for the phys data array to search for a stimulation
        phys_start_time = self.ts_data.start_times[0]
        stim_start_time = self.dio_data.start_times[parameter[0]] + self._time_difference
        onset_time = stim_start_time + self.parameters.parameters.loc[parameter, "onset time (s)"] - phys_start_time

        start_index = self.ts_data._time_to_index(onset_time, remove_gaps=False)
        return start_index - int(self.ts_data.sample_rate * self._search), start_index + int(
            self.ts_data.sample_rate * self._search)

    def _verify_pulse(self, parameter):
        # check for a stimulation pulse in the phys data set
        i1, i2 = self._expected_indices(parameter)
        data = self.ts_data.remove_ch(self._trigger_channel, invert=True)
        array = data.array[0, i1:i2 + 1].compute()
        expected_index = int((i1 + i2) / 2) - i1

        # find the index of the trigger pulse that is the closest to the estimated stimulation period
        smallest_offset = i2 - i1
        for idx, point in enumerate(array):
            if point > self._threshold and abs(idx - expected_index) < abs(smallest_offset):
                smallest_offset = idx - expected_index

        # convert this trigger pulse to seconds of offset or NaN if no pulse is found.
        if smallest_offset == i2 - i1:
            return np.NaN
        else:
            return smallest_offset / self.ts_data.sample_rate

    def _initialize_all_pulses(self):
        # initialize all parameters by finding offsets for each parameter with the _verify_pulse method.
        parameters_list = list(self.parameters.parameters.index)
        offsets = []
        for p in parameters_list:
            offset = self._verify_pulse(p)
            if np.isnan(offset):
                warnings.warn("No trigger pulse found for stimulation parameter {}".format(p))
            offsets.append(offset)
        return offsets

# TODO: steps to make this class better:
# implement the remove_gaps parameter similarly to the _TsData class
# trigger channels with NaN values will produce lots of warnings
# Stimulation times in a block gap may pick up trigger pulses on either side of the gap
