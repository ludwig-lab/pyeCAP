# python standard library imports
import os
import glob

# scientific computing library imports
import sys

import numpy as np
import pandas as pd

# neuro base class imports
from .base.event_data import _EventData
from .base.dio_data import _DioData
from .base.parameter_data import _ParameterData
from .base.utils.base import _is_iterable
from .base.utils.numeric import _to_numeric_array
# eba_toolkit io class imports
from .io.ripple_io import RippleIO, RippleEvents
from .io.tdt_io import TdtIO, TdtStim


class Stim(_EventData, _DioData, _ParameterData):
    """
    Class for working with stimulation data.
    """
    def __init__(self, file_path, io=None, events=None, event_indicators=None, dio=None, dio_indicators=None,
                 parameters=None, metadata=None):
        """
        Constructor for stimulation data objects.

        Parameters
        ----------
        file_path : str, list
            Directory or list of directories containing TDT data sets.
        io : None, list
            List of eba_toolkit io objects to read ripple/tdt data for each experiment.
        events : None, list
            List of dictionaries containing  name and an array with stimulation event times.
        event_indicators : None, list
            List of dictionaries containing channel name and a pandas integer array relating each stimulation event to
            the stimulation parameter.
        dio : None, list
            List of dictionaries containing channel name and stimulation start/stop times in an array.
        dio_indicators : None, list
            List of dictionaries containing channel name and pandas integer array that relates the stimulation parameter
            to the starting and stopping times.
        parameters : None, list
            List of pandas DataFrames containing stimulation parameter data.
        metadata : None, list
            List of dictionaries containing stimulation experiment metadata.

        Examples
        ________
        >>> eba_toolkit.Stim(pathname1)   # replace pathnames with paths to data      # doctest: +SKIP

        >>> eba_toolkit.Stim([pathname1, pathname2, pathname3])                       # doctest: +SKIP
        """

        if isinstance(file_path, list) \
                and isinstance(io, list) \
                and isinstance(events, list) \
                and isinstance(event_indicators, list) \
                and isinstance(dio, list) \
                and isinstance(dio_indicators, list) \
                and isinstance(parameters, list) \
                and isinstance(metadata, list):
            self.file_path = file_path
            self.io = io
            _EventData.__init__(self, events, metadata, indicators=event_indicators)
            _DioData.__init__(self, dio, metadata, indicators=dio_indicators)
            _ParameterData.__init__(self, parameters, metadata)
        elif isinstance(file_path, str):
            # Read in Ripple data files
            if file_path.endswith('.nev'):
                # TODO: Changes for tdt will break this code for ripple, need to repair
                self.file_path = [file_path]
                self.io = [RippleIO(file_path)]
                events = RippleEvents(self.io[0], type='stim')

            # Read in file types that point to a directory (i.e. tdt)
            elif os.path.isdir(file_path):
                # Check if directory is for tdt data
                tev_files = glob.glob(file_path + '/*.tev')  # There should only be one
                if len(tev_files) == 0:
                    raise FileNotFoundError("Could not located '*.tev' file expected for tdt tank.")
                elif len(tev_files) > 1:
                    raise FileExistsError("Multiple '*.tev' files found in tank, 1 expected.")
                else:
                    self.file_path = [file_path]
                    self.io = [TdtIO(file_path)]
                    tdt_stim = TdtStim(self.io[0])
                    parameters = tdt_stim.parameters
                    metadata = tdt_stim.metadata
                    events = tdt_stim.events()
                    event_indicators = tdt_stim.events(indicators=True)
                    dio = tdt_stim.dio()
                    dio_indicators = tdt_stim.dio(indicators=True)
            # File type not found
            else:
                print(file_path)
                _, file_extension = os.path.splitext(file_path)
                raise IOError('"' + file_extension + '" is not a supported file extension')
            _EventData.__init__(self, events, metadata, indicators=event_indicators)
            _DioData.__init__(self, dio, metadata, indicators=dio_indicators)
            _ParameterData.__init__(self, parameters, metadata)
        elif _is_iterable(file_path, str):
            data = [type(self)(f) for f in file_path]
            file_path = [item for d in data for item in d.file_path]
            io = [item for d in data for item in d.io]
            events = [item for d in data for item in d._events]
            event_indicators = [item for d in data for item in d._event_indicators]
            dio = [item for d in data for item in d._dio]
            dio_indicators = [item for d in data for item in d._dio_indicators]
            parameters = [item for d in data for item in d._parameters]
            metadata = [item for d in data for item in d._metadata]
            self.__init__(file_path, io, events, event_indicators, dio, dio_indicators, parameters, metadata)
        else:
            raise ValueError("Input expected to be string or list of strings")

    @property
    def raw_stores(self):
        """
        Returns data for raw stimulation waveforms ('eS1r' stores) and raw voltage monitoring data ('MonA' stores) if
        they exist.

        Returns
        -------
        list
            list of dictionaries that contain the raw data structs.
        """
        raw_data = []
        for i in range(len(self.metadata)):
            raw_data.append({key: getattr(self.io[i].tdt_block.stores, key) for key in self.metadata[i]['raw_stores']})
        return raw_data

    def plot_dio(self, *args, **kwargs):
        """
        Creates a plot of stimulation data showing the time periods with and without stimulation in raster format. See the
        _DioData.plot_raster method for more detail.

        Parameters
        ----------
        *args : Arguments
            Arguments to be passed to the _DioData.plot_raster method.
        **kwargs : KeywordArguments
            Keyword arguments to be passed to the _DioData.plot_raster method.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        See Also
        ________
        _DioData.plot_raster

        Examples
        ________
        >>> stim_data.plot_dio()        # doctest: +SKIP
        """
        _DioData.plot_raster(self, *args, **kwargs)

    def plot_events(self, *args, **kwargs):
        """
        Creates a plot of stimulation data showing the time periods with and without stimulation in raster format. See the
        _EventData.plot_raster method for more detail.

        Parameters
        ----------
        *args : Arguments
            Arguments to be passed to the _EventData.plot_raster method.
        **kwargs : KeywordArguments
            Keyword arguments to be passed to the _EventData.plot_raster method.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        See Also
        ________
        _EventData.plot_raster

        """
        _EventData.plot_raster(self, *args, **kwargs)

    def set_parameters(self, parameter, values):
        """
        Resets the values of the 'channel' and 'polarity' parameters in the stimulation parameters DataFrame.
        ?? object is modified in place??

        Parameters
        ----------
        parameter : str
            Potential parameters are 'polarity' and 'channel'.
        values : list
            List of values to reset the parameters.

        Returns
        -------
        None
        """
        print(len(values))
        potential_parameters = ['polarity',
                                'channel']
        if parameter in potential_parameters:
            if _is_iterable(values):
                if len(values) == 1:
                    for p_df in self._parameters:
                        p_df[parameter] = values[1]
                elif len(values) == len(self._parameters):
                    for p_df, v in zip(self._parameters, values):
                        p_df[parameter] = v
                else:
                    raise ValueError("Number of values must be 1 or equal to the number of data sets")
            else:
                for p_df in self._parameters:
                    p_df[parameter] = values

    def set_channels(self, values):
        """
        Resets values in the stimulation parameters DataFrame. Updates 'channel', 'polarity', 'anode' and 'cathode'
        columns.
        ?? Object is modified in place ??

        Parameters
        ----------
        values : list, int
            Values to use in the stimulation parameters DataFrame. Use a list of integers equal to the number of data
            sets to reset the channels for each data set. Use a list of sequences of length 2 to reset 'anode' and
            'cathode' values for bipolar stimulation.
        Returns
        -------
        None
        """
        if _is_iterable(values):
            if len(values) == 1:
                values = list(values)*len(self._parameters)
            elif len(values) == len(self._parameters):
                pass
            else:
                raise ValueError("Number of values must be 1 or equal to the number of data sets")
            for p_df, v in zip(self._parameters, values):
                v = _to_numeric_array(v, dtype=int)
                p_df['polarity'] = len(v)
                if len(v) == 1:
                    p_df['channel'] = v[0]
                    if 'cathode' in p_df.columns:
                        p_df['cathode'] = np.NaN
                    if 'anode' in p_df.columns:
                        p_df['anode'] = np.NaN
                elif len(v) == 2:
                    p_df['cathode'] = v[0]
                    p_df['anode'] = v[1]
                    if 'channel' in p_df.columns:
                        p_df['channel'] = np.NaN
                else:
                    raise NotImplementedError('Only monopolar and bipolar stimulation implemented at this time')
        else:
            self.set_channels([values])

    def add_series(self, num_condition, series_to_add: pd.Series):
        """
        Used to add series to stimulation parameters. commonly used with: Channel, Condition, Stimulation Type
        """
        num_amplitudes = len(self._parameters[num_condition])
        if len(series_to_add) == 1:
            series_to_add = series_to_add.append([series_to_add]*(num_amplitudes-1), ignore_index=True)
        if len(series_to_add) != num_amplitudes:
            sys.exit('length of amplitudes not the same')
        else:
            self._parameters[num_condition][series_to_add.name] = series_to_add

    def append(self, new_data):
        """
        Adds new data to a Stim class instance.
        ?? object is modified in place??

        Parameters
        ----------
        new_data : Stim
            New data to be added to the Stim class instance.

        Returns
        -------
        None
        """
        if isinstance(new_data, type(self)):
            file_names = self.file_path + new_data.file_path
            io = self.io + new_data.io
            events = self._events + new_data._events
            if self._event_indicators is None:
                if new_data._event_indicators is None:
                    event_indicators = None
                else:
                    event_indicators = new_data._event_indicators


            else:
                if new_data._event_indicators is None:
                    event_indicators = self._event_indicators
                else:
                    event_indicators = self._event_indicators + new_data._event_indicators
            dio = self._dio + new_data._dio
            if self._dio_indicators is None:
                if new_data._dio_indicators is None:
                    dio_indicators = None
                else:
                    dio_indicators = new_data._dio_indicators
            else:
                if new_data._dio_indicators is None:
                    dio_indicators = self._dio_indicators
                else:
                    dio_indicators = self._dio_indicators + new_data._dio_indicators
            self._parameters += new_data._parameters
            self._metadata += new_data._metadata
        else:
            raise TypeError("Appended data is not of the same type.")
