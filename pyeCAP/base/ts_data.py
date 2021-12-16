# python standard library imports
import os.path
import warnings
from datetime import datetime
import copy
from collections import Iterable

# scientific computing library imports
import dask.array as da
from dask.diagnostics import ProgressBar
import dask.multiprocessing
from dask.cache import Cache
import numpy as np
import scipy
from scipy import signal
import mne

# plotting and figure generation
from matplotlib import get_backend as plt_get_backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

# interactive plotting
from ipywidgets import interact, AppLayout, FloatSlider, VBox, Button, Output

# neuro base class imports
from .dio_data import _DioData
from .event_data import _EventData
from .utils.numeric import _to_numeric_array, largest_triangle_three_buckets, _group_consecutive
from .utils.visualization import _plt_setup_fig_axis, _plt_show_fig, _plt_ax_to_pix, _plt_add_ax_connected_top, \
    _plt_check_interactive

from multiprocessing.pool import ThreadPool
dask.config.set(scheduler='threads', pool=ThreadPool(8))

sns.set_context("paper", font_scale=1.4, rc={"lines.linewidth": 2.5, "axes.linewidth": 2.0})
sns.set_style("ticks")

cache = Cache(2e9)  # Leverage two gigabytes of memory
cache.register()  # Turn cache on globally


class _TsData:
    """
    Class for time series data. Contains many of the methods for the pyCAP.Ephys child class.
    """
    def __init__(self, data, metadata, chunks=None, daskify=True, thread_safe=True, fancy_index=True, order=True,
                 ch_offsets=None):
        """
        Constructor for time series data objects.

        Parameters
        ----------
        data : list
            List of arrays of input data.
        metadata : list
            List of experiment metadata dictionaries.
        chunks : None, list
            Dask chunks of the input data, or None to pull the chunks from the input data.
        daskify : bool
            Set to True to convert input data into a dask array.
        thread_safe : bool
            Set to True to create a lock if daskify is True.
        fancy_index : bool
            Set to True to support fancy indexing if daskify is True.
        order : bool
            Set to True to order data sets by start time. Since data from the same file is read in chronological order,
            this will only have an effect when reading in data from multiple files.
        ch_offsets: None, list
            List of the same length as the number of channels. Each number corresponds to a channel offset in units of
            samples. This corrects for errors in timing between channels in TDT systems. Channel offsets are negative,
            and delete the first samples of the corresponding channel in order to change the timing difference between
            channels. Set to None to ignore any channel offsets.
        """
        # TODO: add better checks on data - make sure they have the same sampling rate and num_channels, warn when combining things with different channel names
        # Expects list of different input data types, immediately convert for data/metadata
        if not isinstance(data, list):
            data = [data]
        if not isinstance(metadata, list):
            metadata = [metadata]

        # Setup expected chunking values for each data set
        if chunks is None:
            self.chunks = [d.chunks for d in data]
        else:
            if not isinstance(chunks, list):
                chunks = [chunks]
            self.chunks = chunks

        # Setup metadata for different data sets
        self.metadata = metadata

        # Setup data, converting to dask if required
        if daskify:
            self.data = [da.from_array(d, lock=thread_safe, fancy=fancy_index) for d in data]
        else:
            self.data = data
        if ch_offsets is not None:
            for meta in metadata:
                if 'ch_offsets' not in meta.keys():
                    meta['ch_offsets'] = ch_offsets
                meta['stream_lengths'] = meta['stream_lengths'] - min([min(ch_offsets), 0]) - max([min(ch_offsets), 0])
            self.orig_data = data
            if len(ch_offsets) != self.orig_data[0].shape[0]:       # check for channel offsets of incorrect length
                raise ValueError("Channel offsets must be the same length as the number of channels")
            # rechunk and rearrange arrays to account for channel offsets if necessary
            self.data = self._introduce_offsets(ch_offsets)

        # sort all data sets by start time, place in chronological order
        if order:
            unsorted = [(a, m) for a, m in zip(self.data, self.metadata)]
            srted = sorted(unsorted, key=lambda x: x[1]['start_time'])
            self.data = [s[0] for s in srted]
            self.metadata = [s[1] for s in srted]

    @property
    def array(self):
        """
        Property getter method for a dask array of the raw data for all data sets. All data sets are concatenated into
        one array by channel.

        Returns
        -------
        dask.array.core.Array
            Array of the class instance data sets.

        Examples
        ________
        >>> ephys_data.array     # doctest: +SKIP

        """
        return da.concatenate(self.data, axis=1)

    @property
    def ch_offsets(self):
        """
        Property getter method for channel offsets. Channel offsets are in units of samples. Use channel offsets to
        account for any sampling delays between channels. Any data sets without channel offsets will show up as None.

        Returns
        -------
        list, None
            list of offsets for each channel, or None if channels are not specified.
        """
        if "ch_offsets" in self.metadata[0]:
            return self.metadata[0]["ch_offsets"]
        else:
            return None

    @property
    def shape(self):
        """
        Property getter method for the dimensions of the raw data array. Shows number of channels as the first
        dimension and number of data points as the second dimension.

        Returns
        _______
        tuple
            Dimensions of the array.
        See Also
        ________
        array

        Examples
        ________
        >>> ephys_data.shape
        (16, 155648)

        """
        return self.array.shape

    @property
    def ndim(self):
        """
        Property getter method for the  number of dimensions in the raw data array.

        Returns
        _______
        int
            Number of dimensions in the array.
        See Also
        ________
        array

        Examples
        ________
        >>> ephys_data.ndim
        2
        """
        return self.array.ndim

    @property
    def dtype(self):
        """
        Property getter method for the data type of each element in the raw data array.

        Returns
        -------
        numpy.dtpye
            Type of data present in the array.
        See Also
        --------
        array

        Examples
        ________
        >>> ephys_data.dtype
        dtype('float32')
        """
        return self.array.dtype

    @property
    def size(self):
        """
        Property getter method for the total number of elements in the raw data array.

        Returns
        -------
        int
            Total size of the array.
        See Also
        --------
        array

        Examples
        ________
        >>> ephys_data.size
        2490368
        """
        return self.array.size

    @property
    def itemsize(self):
        """
        Property getter method for the storage taken up by a single element in the raw data array.

        Returns
        -------
        int
            Size of each element in bytes.
        See Also
        --------
        array

        Examples
        ________
        >>> ephys_data.itemsize
        4
        """
        return self.array.itemsize

    @property
    def shapes(self):
        """
        Property getter method for the dimensions of each data set.

        Returns
        -------
        list
            List of tuples, with each tuple corresponding to the shape property for each data set.
        See Also
        --------
        shape

        Examples
        ________
        >>> ephys_data.shapes
        [(16, 155648)]
        """
        return [d.shape for d in self.data]

    def remove_data(self, datasets, invert=False):
        pass

    def remove_ch(self, channels, invert=False):
        """
        Method for removing channels from time series data objects.

        Parameters
        ----------
        channels : int, str, list
            Name of a channel in the form of a string or index of the channel list from property method ch_names.
            This method will also accept a list of channel names or indices.
        invert : bool
            Default false will remove the specified channels. Invert = True will remove all channels except the ones
            specified in the channels parameter.

        Returns
        -------
        _TsData or subclass
            New class instance with specified channel names removed.

        Examples
        ________
        >>> ephys_data.remove_ch("RawE 1")       # doctest: +SKIP

        """
        ch_idx = self._ch_to_index(channels)
        if not invert:
            ch_idx = np.logical_not(ch_idx)
        data = [d[ch_idx, :] for d in self.data]
        metadata = copy.deepcopy(self.metadata)
        for m in metadata:
            for k in m.keys():
                if isinstance(m[k], list) and len(m[k]) == len(ch_idx):
                    m[k] = [ch_m for ch_m, idx_bool in zip(m[k], ch_idx) if idx_bool]
        return type(self)(data, metadata, chunks=self.chunks, daskify=False)

    @property
    def ch_names(self):
        """
        Property getter method for channel names.

        Returns
        -------
        list
            A list of the channel names.

        Examples
        ________

        >>> ephys_data.ch_names
        ['RawE 1', 'RawE 2', 'RawE 3', 'RawE 4', 'RawG 1', 'RawG 2', 'RawG 3', 'RawG 4', 'LIFt 1', 'LIFt 2', 'LIFt 3', 'LIFt 4', 'EMGt 1', 'EMGt 2', 'EMGt 3', 'EMGt 4']

        """
        ch_names = [tuple(meta['ch_names']) for meta in self.metadata]
        if len(set(ch_names)) == 1:
            return list(ch_names[0])
        else:
            raise ValueError("Import data sets do not have consistent channel names.")

    def set_ch_names(self, ch_names):
        """
        Method for renaming channels.

        Parameters
        ----------
        ch_names : list
            List of channel names.

        Returns
        -------
        _TsData or subclass
            New class instance with renamed channels.

        See Also
        ________
        ch_names

        Examples
        ________
        >>> new_channels = ["LIFE1", "LIFE2", "LIFE3", "LIFE4", "EMG1", "EMG2", "EMG3", "EMG4", "T1", "T2", "T3", "T4", "E1", "E2", "E3", "E4"]
        >>> renamed_data = ephys_data.set_ch_names(new_channels)
        >>> renamed_data.ch_names
        ['LIFE1', 'LIFE2', 'LIFE3', 'LIFE4', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'T1', 'T2', 'T3', 'T4', 'E1', 'E2', 'E3', 'E4']
        """
        # TODO: make this accept a dictionary with channel mask which is more convenient for large ch counts
        if len(ch_names) != len(self.ch_names):
            raise ValueError("Number of channels in input 'types' does not match number of channels in data array.")
        if len(set(ch_names)) != len(self.ch_names):
            raise ValueError("Multiple channels can not be assigned the same channel name.")

        metadata = copy.deepcopy(self.metadata)
        for m in metadata:
            m['ch_names'] = ch_names
        return type(self)(self.data, metadata, chunks=self.chunks, daskify=False)

    @property
    def types(self):
        """
        Returns the types present in the ephys dataset. For datasets that contain multiple types of ephys recordings
        on differnt channels. For example a single dataset could contiain single unit data, LFP data, neurography
        data or other specific ephys datasets on different channels. Defining types in a dataset enables operations
        to be performed only on specific types of data.

        Returns
        -------
        list
            List of channel types without repeats.

        Examples
        ________
        >>> ephys_data = ephys_data.set_ch_types(['L','L','L','L','E','E','E','E','L','L','L','L','E','E','E','E'])
        >>> sorted(ephys_data.types)     # sort this list to ensure the order is always the same
        ['L','L','L','L','E','E','E','E','L','L','L','L','E','E','E','E']
        """
        ch_types = [tuple(meta['types']) if 'types' in meta.keys() else tuple() for meta in self.metadata]
        if len(set(ch_types)) == 1:
            return list(set(ch_types[0]))
        else:
            raise ValueError("Import data sets do not have consistent channel types.")

    @property
    def ch_types(self):
        """
        Returns the types of each channel in the ephys dataset. For datasets that contain multiple types of ephys recordings
        on differnt channels. For example a single dataset could contiain single unit data, LFP data, neurography
        data or other specific ephys datasets on different channels. Defining types in a dataset enables operations
        to be performed only on specific types of data.

        Returns
        -------
        list
            List of channel types. Matches length of number of channels in the dataset.

        Examples
        ________
        >>> ephys_data = ephys_data.set_ch_types(['L','L','L','L','E','E','E','E','L','L','L','L','E','E','E','E'])
        >>> sorted(ephys_data.types)     # sort this list to ensure the order is always the same
        ['L','E']
        """
        ch_types = [tuple(meta['types']) if 'types' in meta.keys() else tuple() for meta in self.metadata]
        if len(set(ch_types)) == 1:
            return ch_types[0]
        else:
            raise ValueError("Import data sets do not have consistent channel types.")

    def set_ch_types(self, ch_types, rename=False):
        """
        Method for setting the type of each channel.

        Parameters
        ----------
        ch_types : list
            List of channel types

        rename : bool
            Boolean value indicating if the channels should be renamed based on the channel types.

        Returns
        -------
        _TsData or subclass
            New class instance with channel types reset.

        Examples
        ________
        >>> ephys_data.set_ch_types(['L','L','L','L','E','E','E','E','L','L','L','L','E','E','E','E'])  # doctest: +SKIP

        See Also
        ________
        types

        """
        # TODO: make this accept a dictionary with channel mask which is more convenient for large ch counts
        metadata = copy.deepcopy(self.metadata)
        if len(ch_types) != len(self.ch_names):
            raise ValueError("Number of channels in input 'types'", len(ch_types), "does not match number of channels in data array", len(self.ch_names), ".")
        for m in metadata:
            m['types'] = ch_types
        if rename:
            ch_names = ch_types.copy()
            for t in set(ch_types):
                t_count = 1
                for i, ch in enumerate(ch_names):
                    if ch == t:
                        ch_names[i] = ch_names[i] + " " + str(int(t_count))
                        t_count += 1
            return type(self)(self.data, metadata, chunks=self.chunks, daskify=False).set_ch_names(ch_names)
        else:
            return type(self)(self.data, metadata, chunks=self.chunks, daskify=False)

    @property
    def sample_rate(self):
        """
        Property getter method for the sampling rate of the data set.

        Returns
        -------
        float
            Sample rate of the experiment in Hz.

        Examples
        ________
        >>> ephys_data.sample_rate
        24414.0625
        """
        # returns single value if all data sets have the same sampling rate, otherwise returns list
        rates = [meta['sample_rate'] for meta in self.metadata]
        if len(set(rates)) == 1:
            return rates[0]
        else:
            raise ValueError("Import data sets do not have consistent sample rates.")

    @property
    def start_times(self):
        """
        Property getter method for start times of each data set.

        Returns
        -------
        list
            List of start times for each data set in seconds since epoch.

        Examples
        ________
        >>> ephys_data.start_times
        [1576541104.999999]
        """
        start_times = [meta['start_time'] for meta in self.metadata]
        return start_times

    @property
    def start_indices(self):
        """
        Property getter method for array indices that represent the start of each data set within the dask
        array. Using these indices allows the raw data in the array property to be divided by data set.

        Returns
        -------
        numpy.ndarray
            Array containing the start indices of each data set.

        See Also
        --------
        array

        Examples
        ________
        >>> ephys_data.start_indices
        array([0])

        # Example code to split the array back into component data sets
        >>> import numpy as np                                                                  # doctest: +SKIP
        >>> np.split(ephys_data.array.compute(), ephys_data.start_indices[1:], axis = 1)        # doctest: +SKIP
        """
        start_indices = np.zeros(len(self.shapes))
        data_lengths = [s[1] for s in self.shapes[0:-1]]
        start_indices[1:] = np.cumsum(data_lengths)
        return start_indices.astype(int)

    @property
    def end_times(self):
        """
        Property getter method for the end time of each data set in seconds since the epoch.

        Returns
        -------
        list
            List containing end times of each data set.

        Examples
        ________
        >>> ephys_data.end_times
        [1576541111.3753412]
        """
        return [t + (s[1] / self.sample_rate) for t, s in zip(self.start_times, self.shapes)]

    @property
    def ndata(self):
        """
        Property getter method for the the number of data sets included.

        Returns
        -------
        int
            Number of data sets in the class instance.

        Examples
        ________
        >>> ephys_data.ndata
        1
        """
        return len(self.data)

    def time(self, remove_gaps=True):
        """
        Property getter method for a dask array of time points for each data set starting at 0 seconds. Each point in
        the time array corresponds to a point in the raw data array. Times are derived from the sample rate.

        Parameters
        ----------
        remove_gaps : bool
            Set to False to take gaps between data sets into account.

        Returns
        -------
        dask.array.core.Array
            Array of time points.

        Examples
        ________
        >>> ephys_data.time      # doctest: +SKIP

        """
        # Chunk size is arbitrary
        time_points = [da.arange(d.shape[1], chunks=c[1]*10) for d, c in zip(self.data, self.chunks)]
        sample_rates = self.sample_rate
        for i in range(1, self.ndata):
            time_points[i] = time_points[i] + time_points[i-1][-1] + 1
            if not remove_gaps and isinstance(sample_rates, list):
                time_points[i] = time_points[i] + sample_rates[i] * (self.start_times[i] - self.end_times[i - 1])
            elif not remove_gaps:
                time_points[i] = time_points[i] + sample_rates * (self.start_times[i] - self.end_times[i - 1])
        if isinstance(sample_rates, list):
            return da.concatenate([t / s for t, s in zip(time_points, sample_rates)])
        else:
            return da.concatenate(time_points) / sample_rates

    def channel_reference(self, channel, ch_type=None):
        """
        Creates a new object with specified channel types re-referenced to the specified channel.

        Parameters
        ----------
        channel : int, str
            Name or index of the channel to be used as the reference channel.
        ch_type : str, list
            Specifies channel type or list of types to apply the channel reference to. Use 'None' to re-reference all
            channels.

        Returns
        -------
        _TsData or subclass
            Class instance with channels referenced to specified channel.

        Examples
        ________
        >>> ephys_data.channel_reference(channel = 'RawE 1')         # doctest: +SKIP

        """
        channel = self._ch_to_index(channel)
        if ch_type is None:
            data = [d - d[channel, :] for d in self.data]
        else:
            if isinstance(ch_type, str):
                if ch_type in self.types:
                    channels = self._ch_type_to_index(ch_type)[:, None]
                else:
                    raise Warning(ch_type + " not found in types")
            elif isinstance(ch_type, (list, np.ndarray)):
                channels = np.zeros((len(self.ch_names), 1))
                for ch in ch_type:
                    if isinstance(ch, str):
                        if ch in self.types:
                            channels = np.logical_or(self._ch_type_to_index(ch)[:, None], channels)
                        else:
                            raise Warning(ch + " not found in types")
                    else:
                        raise ValueError("Input 'ch_type' is expected to be list of str, not " + type(ch))
            else:
                raise ValueError("Input 'ch_type' is expected to be of type str, list, or numpy array")
            data = [d - d[channel, :].repeat(d.shape[0], axis=0) * channels for d in self.data]
            # Both return and remove channel used as reference.
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False).remove_ch(channel)

    def common_reference(self, ch_type=None, method='mean'):
        """
        Creates a new object with specified channel types re-referenced to a common mean or median.

        Parameters
        ----------
        ch_type : None, str, list
            Channel type or list of types that the common reference will be applied to. Use 'None' to re-reference all
            channels.
        method : str
            Enter in 'mean' use a mean common reference and 'median' to use a median common reference.

        Returns
        -------
        _TsData or subclass
            Class instance with channels referenced to mean or median.

        Examples
        ________
        >>> ephys_data.common_reference(method = 'mean')    # doctest: +SKIP

        """
        if ch_type is None:
            if method == 'mean':
                data = [(d - da.mean(d, axis=0)[None, :] + d / d.shape[0]) * d.shape[0] / (d.shape[0] - 1) for d in
                        self.data]
            elif method == 'median':
                data = []
                for d in self.data:
                    d_array = []
                    for i in range(d.shape[0]):
                        median_idx = np.ones(d.shape[0], dtype=bool)
                        median_idx[i] = 0
                        new_chunks = (sum(median_idx), d.chunks[1])
                        median_data = da.map_blocks(lambda x: np.median(x, axis=0)[None, :],
                                                    da.rechunk(d[median_idx, :], new_chunks), chunks=(1, new_chunks[1]))
                        d_array.append(d[i, :] - median_data[0, :])

                    data.append(da.stack(d_array))
            else:
                raise ValueError(str(method) + " is not a valid option for input 'method'")
        else:
            if isinstance(ch_type, str):
                if ch_type in self.types:
                    channels = self._ch_type_to_index(ch_type)[:, None]
                else:
                    raise Warning(ch_type + " not found in types")
            elif isinstance(ch_type, (list, tuple, np.ndarray)):
                channels = np.zeros((len(self.ch_names), 1))
                for ch in ch_type:
                    if isinstance(ch, str):
                        if ch in self.types:
                            channels = np.logical_or(self._ch_type_to_index(ch)[:, None], channels)
                        else:
                            raise Warning(ch + " not found in types")
                    else:
                        raise ValueError("Input 'ch_type' is expected to be list of str, not " + type(ch))
            else:
                raise ValueError("Input 'ch_type' is expected to be of type str, list, tuple, or numpy array")
            if method == 'mean':
                ch_num = sum(channels)[0]

                data = []
                for d in self.data:
                    d_mean = da.repeat(da.mean(d[channels[:, 0], :], axis=0)[None, :], d.shape[0], axis=0) * channels
                    data.append(d - (d_mean + (d * channels / d.shape[0]) * (channels * ch_num / (ch_num - 1))))
            elif method == 'median':
                data = []
                for d in self.data:
                    d_array = []
                    for i in range(d.shape[0]):
                        if channels[i]:
                            median_idx = np.ones(d.shape[0], dtype=bool)
                            median_idx[i] = 0
                            median_idx[np.logical_not(channels[:, 0])] = 0
                            new_chunks = (sum(median_idx), d.chunks[1])
                            median_data = da.map_blocks(lambda x: np.median(x, axis=0)[None, :],
                                                        da.rechunk(d[median_idx, :], new_chunks),
                                                        chunks=(1, new_chunks[1]))
                            d_array.append(d[i, :] - median_data[0, :])
                        else:
                            d_array.append(d[i, :])
                    data.append(da.stack(d_array))
            else:
                raise ValueError(str(method) + " is not a valid option for input 'method'")
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def filter_iir(self, Wn, rp=None, rs=None, btype='band', order=1, ftype='butter'):
        """
        Filters the data with an infinite impulse response (iir) filter with the scipy.signal.iirfilter method.

        Parameters
        ----------
        Wn : list, tuple
            Sequence with 2 elements containing critical frequencies.
        rp : None, float
            Maximum ripple in the passband in decibels for Chebyshev and ellipitic filters.
        rs : None, float
             Minimum attenuation in the stop band for Chebyshev and ellipitic filters.
        btype : str
            Use 'band', 'bandpass’, ‘lowpass’, ‘highpass’, or ‘bandstop’ to specify type of filter.
        order : int
            Order of the filter. Default is 1.
        ftype : str
            Use 'butter' for Butterworth filter, 'cheby1' for Chebyshev I filter, 'cheby2' for Chebyshev II filter,
            'ellip' for elliptic filter, 'bessel' for Bessel filter.

        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """
        sos = signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype, output='sos', fs=self.sample_rate)
        a, b = signal.sos2tf(sos)
        overlap = max(len(a), len(b))
        data = [da.map_overlap(d, lambda x: signal.sosfiltfilt(sos, x), (0, overlap), dtype=d.dtype) for d in self.data]
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def filter_fir(self, cutoff, width=None, filter_length='auto', window='hamming', pass_zero=True):
        """
        Filters the frequencies specified in the 'cutoff' parameter with a finite impulse response (fir) filter.

        Parameters
        ----------
        cutoff : float, list, tuple
            Cutoff frequency or increasing sequence specifying band edge frequencies (Hz). Frequencies must be between 0
            and self.sample_rate/2.
        width : None, float
            Width of the transition region in Hz.
        filter_length : str
            [Not yet implemented].
        window : str, tuple
            Window type. See scipy.signal.get_window documentation for more information.
            for more information.
        pass_zero : bool, str
            If True, gain at 0 is 1, if False, the gain is 0.
        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """
        # TODO: implement the 'filter_length' parameter

        def convfft(x):
            conv1 = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(x[0, :], filter_weights, mode='same')), filter_weights,
                                   mode='same'))[None, :]
            return conv1

        numtaps = int((3.3 * self.sample_rate) / (2 * width)) * 2 + 1
        filter_weights = signal.firwin(numtaps, cutoff, width=width, window=window, pass_zero=pass_zero,
                                       fs=self.sample_rate)
        min_chunk_size = min([min(d.chunks[1]) for d in self.data])
        chunk_size = min([d.chunks[1][0] for d in self.data])
        data = []
        if min_chunk_size < 3 * numtaps:
            for d in self.data:
                n_chunk_size = int(chunk_size * np.ceil(3 * numtaps / chunk_size))
                r_chunks = tuple([1] * d.shape[0])
                c_chunks = [n_chunk_size] * (int(d.shape[1] // n_chunk_size) - 1)
                last_chunk = int(n_chunk_size + (d.shape[1] % n_chunk_size))
                c_chunks = tuple(c_chunks + [last_chunk])
                data.append(da.map_overlap(da.rechunk(d, (r_chunks, c_chunks)), convfft, (0, numtaps), dtype=d.dtype))

        else:
            data = [da.map_overlap(d, lambda x: np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(x[0, :], filter_weights, mode='same')), filter_weights,
                                   mode='same'))[None, :], (0, numtaps), dtype=d.dtype) for d in self.data]
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def filter_median(self, kernel_size=201, btype='lowpass'):
        """
        Filters the channels using a median filter. The median filter slides across the data set and replaces each data
        point with the median of surrounding entries using the scipy.ndimage.median_filter method. The median filter is
        a nonlinear filter useful for noise and spike elimination.

        Parameters
        ----------
        kernel_size : int
            The size of the 'window' used in the median filter.
        btype : str
            Use 'lowpass' or 'low' to attenuate high frequency signals. Use 'highpass' or 'high to attenuate low
            frequency signals.

        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """
        if btype in ('lowpass', 'low'):
            data = [da.map_overlap(d, lambda x: scipy.ndimage.median_filter(x, size=(1, kernel_size)), (0, kernel_size),
                                   dtype=d.dtype) for d in self.data]
        elif btype in ('highpass', 'high'):
            data = [
                da.map_overlap(d, lambda x: x - scipy.ndimage.median_filter(x, size=(1, kernel_size)), (0, kernel_size),
                               dtype=d.dtype) for d in self.data]
        else:
            raise ValueError("Value of input 'btype'")
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def filter_gaussian(self, Wn, btype='lowpass', order=0, truncate=4.0):
        """
        Filters the channels using a gaussian filter using the scipy.ndimage.guassian_filter1d method.

        Parameters
        ----------
        Wn : int, float
            Corner frequency
        btype : str
            Use 'lowpass' or 'low' to attenuate high frequency signals. Use 'highpass' or 'high to attenuate low
            frequency signals.
        order : int
            The order of the kernel. 0 corresponds to a filter with a Gaussian kernel. 1 corresponds to the derivative,
            ect.
        truncate : float
            Number of standard deviations to include in Gaussian filter kernel

        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """
        s_c = Wn / self.sample_rate
        sigma = (2 * np.pi * s_c) / np.sqrt(2 * np.log(2))
        lw = int(truncate * sigma + 0.5)
        if btype in ('lowpass', 'low'):
            data = [da.map_overlap(d, lambda x: scipy.ndimage.gaussian_filter1d(x, sigma, axis=1, order=order), (0, lw),
                                   dtype=d.dtype) for d in self.data]
        elif btype in ('highpass', 'high'):
            data = [
                da.map_overlap(d, lambda x: x - scipy.ndimage.gaussian_filter1d(x, sigma, axis=1, order=order), (0, lw),
                               dtype=d.dtype) for d in self.data]
        else:
            raise ValueError("Value of input 'btype'")
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def filter_powerline(self, frequencies=[60, 120, 180], notch_width=None, trans_bandwidth=1.0):
        """Filter powerline noise from time series data.

        This function filters data with a series of notch filters at defined frequencies. Filtering frequencies
        default to 60, 120, and 180 Hz, which are appropriate for 60Hz line noise common in the United States. A fir
        filter is constructed with a hamming window and filter design is performed with scipy.signal.firwin.
        Filtering is performed with self.filter_fir. This function is set up with similar default parameters to
        mne.notch_filter from the mne-python toolkit.

        Parameters
        ----------
        frequencies : list, tuple, np.ndarray, int, float
            Frequencies at which to filter in Hz. Defaults to 60, 120, and 180 Hz.
        notch_width : list, tuple, np.ndarray, int, float
            Width of notch at each filter frequency. Defaults to freq / 200 if set to None.
        trans_bandwidth : int, float
            Width of transition band in Hz.
        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        See Also
        --------
        filter_fir
        filter_iir

        """
        frequencies = _to_numeric_array(frequencies)
        if notch_width is None:
            # if notch widths are not defined, set them to freq / 200
            widths = frequencies / 200
        else:
            widths = _to_numeric_array(notch_width)
        if len(frequencies) != len(widths):
            if len(widths) == 1:
                widths = np.repeat(widths, len(frequencies))
            else:
                raise ValueError("Number of notch_widths must match number of frequencies in powerline notch filter.")
        cutoffs = []
        for f, w in zip(frequencies, widths):
            cutoffs.append(f - (w / 2.0) - (trans_bandwidth / 2.0))
            cutoffs.append(f + (w / 2.0) + (trans_bandwidth / 2.0))
        return self.filter_fir(cutoffs, width=trans_bandwidth)

    def filter_Spike(self, Wn=[300,5000], rp=0.01, rs=100, btype='bandpass', order=4):
        """
        Filters the data with an infinite impulse response (iir) filter with the scipy.signal.iirfilter method.
        Parameters
        ----------
        Wn : list, tuple
            Sequence with 2 elements containing critical frequencies.
        rp : None, float
            Maximum ripple in the passband in decibels for Chebyshev and ellipitic filters.
        rs : None, float
             Minimum attenuation in the stop band for Chebyshev and ellipitic filters.
        btype : str
            Use 'band', 'bandpass’, ‘lowpass’, ‘highpass’, or ‘bandstop’ to specify type of filter.
        order : int
            Order of the filter. Default is 1.
        ftype : str
            Use 'butter' for Butterworth filter, 'cheby1' for Chebyshev I filter, 'cheby2' for Chebyshev II filter,
            'ellip' for elliptic filter, 'bessel' for Bessel filter.
        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """
        lowFreq = Wn[0]
        highFreq = Wn[1]
        sos = signal.ellip(order, rp, rs,[2*np.pi*lowFreq,2*np.pi*highFreq], btype, analog=True)
        b,a = signal.ellip(order, rp, rs,[2*np.pi*lowFreq,2*np.pi*highFreq], btype, analog=True)
        overlap = max(len(a), len(b))
        data = [da.map_overlap(d, lambda x: signal.sosfiltfilt(sos, x), (0, overlap), dtype=d.dtype) for d in self.data]
        return type(self)(data, self.metadata, chunks=self.chunks, daskify=False)

    def plot_times(self, *args, axis=None, events=None, x_lim=None, fig_size=(10, 2), show=True, **kwargs):
        """
        Plots the times when experiments represented by the data sets were conducted.
        Useful for visualizing experiment duration and experiment timing relative to other experiments.

        Parameters
        ----------
        * args : Arguments
            y_min, y_max are possible arguments. Float from 0-1 representing the height of the time bar relative to the
            height of the plot.
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        events : _DioData, _EventData, or subclass
            Event data to plot alongside time series data.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        ** kwargs : KeywordArguments
            See `mpl.axes.Axes.axvspan <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvspan.html>`_
            and `mpl.axes.Axes.axvline <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axvline.html>`_
            for details on plot customization.

        Returns
        -------
        # Todo: check return
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.

        Examples
        ________
        >>> ephys_data.plot_times()     # doctest: +SKIP

        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        for ts, te in zip(self.start_times, self.end_times):
            ts = datetime.fromtimestamp(ts)
            te = datetime.fromtimestamp(te)
            ax.axvspan(ts, te, *args, **kwargs)
            ax.axvline(ts, *args, **kwargs)

        # Handle event data if passed to plot function
        if isinstance(events, (_DioData, _EventData)):
            if isinstance(events, _DioData):
                events.plot_dio(axis=ax, show=False, color='grey', zorder=-1)
            if isinstance(events, _EventData):
                events.plot_events(axis=ax, show=False, color='orange', lw=1)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim(x_lim)

        return _plt_show_fig(fig, ax, show)

    def plot(self, axis=None, channels=None, events=None, x_lim=None, y_lim='auto', ch_labels=None,
             colors=sns.color_palette(), fig_size=(10, 6), down_sample=True,
             show=True, remove_gaps=True):
        """Method for plotting time series data.

        Method for plotting time series data using matplotlib. Also allows for interactive plots within a jupyter
        notebook, or jupyter lab based on ipython widgets. If show is set to 'notebook' this method will display an
        interactive plot in a jupyter notebook (only works within a jupyter notebook). If show is True a plot will be
        diplayed. If show is False then a plot will not be displayed but a matplotlib axis will be returned.

        Parameters
        ----------
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        channels : int, str, list, tuple, np.ndarray
            Channels to plot. Can be a boolean numpy array with the same length as the number of channels, an integer
            array, or and array of strings containing channel names.
        events : _DioData, _EventData, or subclass
            Event data to plot alongside time series data.
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
        down_sample : bool
            Down sample data to optimize display speed. WARNING: This changes the frequency components of the plot.
            Defaults to True.
        show : str, bool
            String 'notebook' to plot interactively in a jupyter notebook or boolean value indicating if the plot should
            be displayed.
        remove_gaps : bool
            Set to False to plot gaps in the data

        Returns
        -------
        matplotlib.axis.Axis, ipywidgets.widgets.widget_templates.AppLayout
            If show is 'notebook' returns an ipython app. Otherwise returns a matplotlib axis.

        Examples
        ________
        >>> ephys_data.plot()       # plots data with default settings      # doctest: +SKIP

        >>> # plots data with x limit and stimulation data
        >>> ephys_data.plot(events=stim_data, x_lim=(0,5), y_lim=(0,1))     # doctest: +SKIP
        """
        show = _plt_check_interactive(show)

        # Set up figure and axis for plotting
        if show == 'notebook':
            fig, axes = _plt_setup_fig_axis(axis, fig_size, subplots=(2, 1), gridspec_kw={'height_ratios': [29, 1]})
            ax = axes[0]
            scroll_ax = axes[1]
        else:
            fig, ax = _plt_setup_fig_axis(axis, fig_size)

        # Get channels to plot
        if channels is None:
            channels = slice(None, None, None)
        else:
            channels = self._ch_to_index(channels)

        # validate the x_limits that were received
        # x_limits are expected to be received in terms of time
        x_lim = self._time_lim_validate(x_lim, remove_gaps=remove_gaps)

        x_index = (self._time_to_index(x_lim[0], remove_gaps=remove_gaps), self._time_to_index(x_lim[1], remove_gaps=remove_gaps)+1)
        plot_array = self.array[channels, x_index[0]:x_index[1]].compute()

        # get plot data
        ax.set_xlim(x_lim)

        if y_lim is not None and y_lim != 'auto' and y_lim != 'max':
            y_lim = _to_numeric_array(y_lim)
            if len(y_lim) == 1:
                d_r = y_lim[0]
                y_lim = None
            elif len(y_lim) == 2:
                d_r = np.median(np.std(plot_array, axis=1), axis=0) * 6

            # Todo: len('max') == 3. Not sure if this is your way of checking to see if it's max, but seems ambiguous if it is.
            elif len(y_lim) == 3:
                d_r = y_lim[1]
                y_lim = (y_lim[0], y_lim[2])
            else:
                raise AttributeError(
                    "Input y_lim is expected to be None, 'auto', 'max', or iterable with length less than 3.")
        else:
            # plot +- 6 standard deviation of the median std.dev in the dataset
            d_r = np.median(np.std(plot_array, axis=1), axis=0) * 6
        tick_locations = np.arange(plot_array.shape[0]) * d_r
        offsets = np.zeros((plot_array.shape[0], 2), dtype=float)
        offsets[:, 1] = tick_locations

        if y_lim == 'auto' or y_lim is None:
            d_min = tick_locations[0] - d_r
            d_max = tick_locations[-1] + d_r
        elif y_lim == 'max':
            d_min = np.min(np.min(plot_array, axis=1) + tick_locations)
            d_max = np.max(da.max(plot_array, axis=1) + tick_locations)
        else:
            d_min = y_lim[0]
            d_max = y_lim[1] + tick_locations[-1]
        ax.set_ylim(d_min, d_max)

        px_width, _ = _plt_ax_to_pix(fig, ax)

        plot_data = self._to_plt_line_collection(x_lim, channels, px_width, down_sample=down_sample, remove_gaps=remove_gaps)
        for data in plot_data:
            # TODO: Matplotlib 3.5.0 has changed how offsets work. The easy solution for right now is to exclude
            #  matplotlib version > 3.5 from the install list, however the best long-term solution is likely to shift
            #  this to use matplotlibs transforms instead which should be compatible across versions.
            lines = LineCollection(data[0], offsets=offsets, colors=colors, linewidths=np.ones(plot_array.shape[0]), transOffset=None)
            current_lines = ax.add_collection(lines)

        ax.set_yticks(tick_locations)
        if ch_labels is not None:
            ax.set_yticklabels(np.asarray(ch_labels))
        else:
            ax.set_yticklabels(np.asarray(self.ch_names)[channels])
        ax.set_xlabel('time (s)')

        # Handle event data if passed to plot function
        if isinstance(events, (_DioData, _EventData)):
            top_ax = _plt_add_ax_connected_top(fig, ax)
            if isinstance(events, _DioData):
                events.plot_dio(axis=top_ax, reference=self, remove_gaps=remove_gaps, show=False, color='grey', zorder=-1)
            if isinstance(events, _EventData):
                events.plot_events(axis=top_ax, reference=self, remove_gaps=remove_gaps, show=False, color='orange', lw=1)
            top_ax.set_xlim(x_lim)

        # show the plot if appropriate
        if show == "notebook":
            # Handle event data if passed to plot function
            if isinstance(events, _DioData):
                events.plot_dio(axis=scroll_ax, reference=self, remove_gaps=remove_gaps, show=False, color='orange', zorder=-1)
            scroll_ax.set_xlim(self._time_lim_validate(None))
            ax.set_xlabel(None)
            scroll_ax.set_xlabel('Time (s)')
            scroll_ax.get_yaxis().set_ticks([])
            scroll_ax.get_yaxis().set_visible(False)
            scroll_span = scroll_ax.axvspan(x_lim[0], x_lim[1], color='green', zorder=11, alpha=0.7)
            scroll_line = scroll_ax.axvline(x_lim[0],  color='green', zorder=11, alpha=0.7)
            # Add data set start points to the scrollbar axis

            if remove_gaps:
                start_times = self.start_indices / self.sample_rate
            else:
                start_times = [st - self.start_times[0] for st in self.start_times]

            for ts in start_times:
                scroll_ax.axvline(ts, lw=2, color='black', zorder=10)

            # display data gaps on scrollbar axis if applicable
            if not remove_gaps:
                gaps = [self.start_times[i] - self.end_times[i-1] for i in range(1, len(self.start_times))]
                for g in range(len(gaps)):
                    scroll_ax.axvspan(self.end_times[g] - self.start_times[0], self.end_times[g]+gaps[g] - self.start_times[0], color='red', zorder=9, alpha=0.5)

            plt.ioff()
            fig.tight_layout()
            slider = FloatSlider(
                orientation='horizontal',
                description='Start Time:',
                value=x_lim[0],
                min=self.time(remove_gaps=remove_gaps)[0],
                max=self.time(remove_gaps=remove_gaps)[-1]
            )
            debug_view = Output()

            slider.layout.margin = '0px 0% 0px 0%'
            slider.layout.width = '100%'

            # fig.canvas.toolbar_visible = False
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.canvas.layout.min_height = '400px'
            fig.canvas.layout.min_width = '400px'

            @debug_view.capture(clear_output=True)
            def update_lines(change):
                nonlocal current_lines  # Grab nonlocal variable current_lines so that it is accessable within this scope
                nonlocal scroll_span
                nonlocal scroll_line
                # Get new x_limits
                x_lim_n = (change.new, change.new + x_lim[1] - x_lim[0])

                # Calculate new line segments
                px_width_n, _ = _plt_ax_to_pix(fig, ax)
                plot_data_n = self._to_plt_line_collection(x_lim_n, channels, px_width, down_sample=down_sample, remove_gaps=remove_gaps)

                # Create line collection and add to plot.
                current_lines.remove()
                scroll_span.remove()
                scroll_line.remove()
                down_sampled_n = []
                for data_n in plot_data_n:
                    down_sampled_n.append(data_n[1])
                    lines_n = LineCollection(data_n[0], offsets=offsets, colors=colors, linewidths=np.ones(plot_array.shape[0]), transOffset=None)
                    current_lines = ax.add_collection(lines_n)
                scroll_span = scroll_ax.axvspan(x_lim_n[0], x_lim_n[1], color='green', zorder=11, alpha=0.7)
                scroll_line = scroll_ax.axvline(x_lim_n[0],  color='green', zorder=11, alpha=0.7)

                # Set new x_limits
                ax.set_xlim(x_lim_n)
                print((x_lim_n, down_sampled_n))
                fig.canvas.draw()
                fig.canvas.flush_events()
            # TODO: make sure this function is called less often so the slider can move faster
            slider.observe(update_lines, names='value')

            app = AppLayout(
                header=debug_view,
                center=fig.canvas,
                footer=slider,
                pane_heights=[0.3, 6, 0.5]
            )

            return app
        else:
            return _plt_show_fig(fig, ax, show)

    def plot_psd(self, axis=None, x_lim=None, y_lim=None, show=True, *args, fig_size=(10, 3), nperseg=None,
                 colors=sns.color_palette(), **kwargs):
        """
        Plots Power Spectral Density (PSD) (V**2/Hz) vs. Frequency (Hz) using the Welch method and 50 percent overlap.

        Parameters
        ----------
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        y_lim : None, list, tuple, np.ndarray
            Use None to have the y axis span the entire distribution. Otherwise, use a tuple or other two element
            sequence to specify limits for the y axis. This function plots the y-axis on a log scale.
        show : bool
            Set to True to display the plot and return nothing, set to False to return the plotting axis and display
            nothing.
        * args : Arguments
            See `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_ for
            more information.
        fig_size : list, tuple, np.ndarray
            The size of the matplotlib figure to plot axis on if axis=None.
        nperseg : None, int
            The segment length. Sample rate is used if not specified.
        colors : list
            Color palette or list of colors to use for channels.
        ** kwargs : KeywordArguments
            See `scipy.signal.welch <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_ for
            more information.

        Returns
        -------
        matplotlib.axis.Axis, None
            If show is False, returns a matplotlib axis. Otherwise, plots the figure and returns None.
        """
        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        if nperseg is None:
            nperseg = int(self.sample_rate)

        psd = da.apply_along_axis(signal.welch, 1, self.array, self.sample_rate, *args, shape=(2, nperseg),
                                  dtype=np.float64, nperseg=nperseg, **kwargs)
        psd = da.swapaxes(psd, 1, 2).compute()

        lines = LineCollection(psd, linewidths=np.ones(psd.shape[0]), colors=colors, transOffset=None)
        ax.add_collection(lines)
        # ax.semilogy(psd[0,:,0], psd[0,:,1])

        if x_lim is None:
            ax.set_xlim(psd[0, 0, 0], psd[0, -1, 0])
        else:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is None:
            if x_lim is None:
                y_min = np.min(psd[:, :, 1])
                y_max = np.max(psd[:, :, 1])
                ax.set_ylim(y_min, y_max)
            else:
                freq_array = psd[0, :, 0]
                if x_lim[0] == 0:
                    arg_x_min = 0
                else:
                    arg_x_min = np.argmax(freq_array > x_lim[0])
                arg_x_max = np.argmax(freq_array > x_lim[1])
                y_min = np.min(psd[:, arg_x_min:arg_x_max, 1])
                y_max = np.max(psd[:, arg_x_min:arg_x_max, 1])
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_yscale('log')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD [V**2/Hz]')

        ax.set_title(None)
        return _plt_show_fig(fig, ax, show)

    def save(self, path, *args, scale=1, dtype=None, store='data', compression='gzip', method='hdf5', **kwargs):
        data = self.array
        if scale != 1:
            data = da.multiply(self.array, scale)
        if dtype is not None:
            data = data.astype(dtype)
        if method == 'hdf5':
            if not (path.endswith(".h5") or path.endswith(".hdf5")):
                path = os.path.splitext(path)[0] + '.h5'
        elif method == 'mat':
            if not (path.endswith(".mat")):
                path = os.path.splitext(path)[0] + '.mat'
            data = data.transpose()

            import h5py
            h5py.File(path, mode='x', userblock_size=512)

            ### The follwoing 43 lines of code are reproduced form the hdf5storage library, which can be found here:
            # https://github.com/frejanordsiek/hdf5storage/blob/main/COPYING.txt
            # These 43 lines of code are reproduced under the following conditions:
            # Copyright (c) 2013-2021, Freja Nordsiek
            # All rights reserved.
            #
            # Redistribution and use in source and binary forms, with or without
            # modification, are permitted provided that the following conditions are met:
            #
            # 1. Redistributions of source code must retain the above copyright notice,
            # this list of conditions and the following disclaimer.
            #
            # 2. Redistributions in binary form must reproduce the above copyright
            # notice, this list of conditions and the following disclaimer in the
            # documentation and/or other materials provided with the distribution.
            #
            # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
            # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
            # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
            # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
            # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
            # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
            # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
            # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
            # POSSIBILITY OF SUCH DAMAGE.
            ###
            # Get the time.
            now = datetime.now()
            # Construct the leading string. The MATLAB one looks like
            #
            # s = 'MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: ' \
            #     + now.strftime('%a %b %d %H:%M:%S %Y') \
            #     + ' HDF5 schema 1.00 .'
            #
            # Platform is going to be changed to CPython version. The
            # version is just gotten from sys.version_info, which is a class
            # for Python >= 2.7, but a tuple before that.
            import sys
            v = sys.version_info
            if sys.hexversion >= 0x02070000:
                v = {'major': v.major, 'minor': v.minor, 'micro': v.micro}
            else:
                v = {'major': v[0], 'minor': v[1], 'micro': v[1]}

            s = 'MATLAB 7.3 MAT-file, Platform: CPython ' \
                + '{0}.{1}.{2}'.format(v['major'], v['minor'], v['micro']) \
                + ', Created on: ' \
                + now.strftime('%a %b %d %H:%M:%S %Y') \
                + ' HDF5 schema 1.00 .'

            # Make the bytearray while padding with spaces up to 128-12
            # (the minus 12 is there since the last 12 bytes are special.

            b = bytearray(s + (128-12-len(s))*' ', encoding='utf-8')

            # Add 8 nulls (0) and the magic number (or something) that
            # MATLAB uses. Lengths must be gone to to make sure the argument
            # to fromhex is unicode because Python 2.6 requires it.

            b.extend(bytearray.fromhex(
                b'00000000 00000000 0002494D'.decode()))

            # Now, write it to the beginning of the file.

            try:
                fd = open(path, 'r+b')
                fd.write(b)
            except:
                raise
            finally:
                fd.close()
            ### This is the end of the reproduced section of code for modifying hdf5 files to be read as .mat files.
        else:
            raise ValueError("Save mehtod '{}' is not recognized. Implemented save methods include 'hdf5'.".format(method))
        with ProgressBar():
            data.to_hdf5(path, "/"+store, *args, compression=compression, **kwargs)




    def _ch_type_to_index(self, ch_type):
        """
        Returns a boolean array with indices matching the self.ch_names property and values indicating whether the
        channel is of the type specified in the ch_type argument.

        Parameters
        ----------
        ch_type : str
            Channel type for the method to search for.

        Returns
        -------
        numpy.ndarray
            A numpy array of boolean values.
        See Also
        --------
        ch_names
        """
        ch_types = [tuple(meta['types']) for meta in self.metadata]
        if len(set(ch_types)) == 1:
            return np.array((ch_types[0])) == ch_type
        else:
            raise ValueError("Import data sets do not have consistent channel types.")

    def _ch_to_index(self, ch):
        """
        Returns a boolean array with indices corresponding to the self.ch_names property. A value in the array will be
        'True' if the channel name is present in the 'ch' argument, otherwise the value will be 'False'.

        Parameters
        ----------
        ch : str, list
            Channel name or names for the method to search for.

        Returns
        -------
        numpy.ndarray
            A numpy array of boolean values.
        See Also
        --------
        ch_names
        """
        # Use numpy to check of ch is string or list of strings
        ch = np.asarray(ch).flatten()
        if ch.dtype.type is np.bool_:
            if len(ch) == len(self.ch_names):
                return ch
            else:
                raise ValueError("Length of boolean np.ndarray does not match length of channels")
        else:
            if ch.dtype.type is np.str_:
                ch_compare = np.asarray(self.ch_names)
            else:
                ch = _to_numeric_array(ch)
                ch_compare = np.arange(0, len(self.ch_names))
            extra_ch = np.isin(ch, ch_compare, invert=True)
            if np.any(extra_ch):
                warnings.warn(str(ch[extra_ch]) + " not found in channel list or is outside range.")
            return np.isin(ch_compare, ch)

    def _time_lim_validate(self, x_lim, remove_gaps=True):
        """
        Checks whether a time limit with a start and end time is valid. Returns a tuple of the input time
        limit or raises an error. Useful for checking x-limits in plotting functions.

        Parameters
        ----------
        x_lim : None, list, tuple
            Range of times in seconds. Use None to calculate the time limit for the entire data set. Use an iterable
            such as a list or tuple to input start and end times.
        remove_gaps : bool
            Set to False to take gaps in the data into account.

        Returns
        -------
        tuple
            Valid time limit containing start and end time.

        """
        if x_lim is None:
            # TODO: Check if speed for this is important, this requires calculating chunks of the time array and
            #  would be slower than directly calculating.
            x_lim = (self.time(remove_gaps=remove_gaps)[0], self.time(remove_gaps=remove_gaps)[-1])
            x_lim = _to_numeric_array(x_lim)
        else:
            x_lim = _to_numeric_array(x_lim)
            # Check x_lim values are positive
            if np.any(x_lim < 0):
                raise ValueError("Time limits for time series data cannot be negative. Received limits: "
                                 + str(x_lim))
            # Check of x_lim[0] is less than x_lim[1]
            elif x_lim[1] <= x_lim[0]:
                raise ValueError("End time cannot precede start time limit. Received limits: " + str(x_lim))
            # Check if both time limits are greater than end time of recording
            elif np.all(x_lim > self.time(remove_gaps=remove_gaps)[-1]):
                raise ValueError("Time limits are not within expected limits of (" + str(self.time(remove_gaps=remove_gaps)[0]) + ", "
                                 + str(self.time(remove_gaps=remove_gaps)[-1]) + "). Received limits: " + str(x_lim))
            # Check if x_lim[1] is greater than end time of recording. Based on previous checks x_lim[0] is already known to be within time bounds for recording.
            elif x_lim[1] > self.time(remove_gaps=remove_gaps)[-1]:
                x_lim[1] = self.time(remove_gaps=remove_gaps)[-1] # replace x_lim[1] with end time of array.
            # All checks have passed return time limits as received.
            else:
                pass
        return tuple(x_lim)

    def _time_to_index(self, time, units='seconds', remove_gaps=True):
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
        remove_gaps : bool
            Set to False to take into account time gaps in the data.


        Returns
        -------
        int
            Array index corresponding to the data at the time input.

        Examples
        ________
        >>> ephys_data._time_to_index(5)
        122070

        """
        # recording gaps - also allow for different time formats
        if units == 'milliseconds':
            time = time / 1e3
        elif units == 'microseconds':
            time = time / 1e6

        if not remove_gaps:
            sts = [st - self.start_times[0] for st in self.start_times]
            ets = [et - self.start_times[0] for et in self.end_times]
            sis = self.start_indices
            sr = self.sample_rate

            # converts each time to an index that takes gaps into account
            def tti_with_gaps(elapsed_time, start_times=sts, end_times=ets, start_indices=sis, sample_rate=sr):
                for t in range(len(start_times)):
                    if start_times[t] <= elapsed_time <= end_times[t]:
                        return round(start_indices[t] + sample_rate * (elapsed_time - start_times[t]))
                    elif end_times[t-1] <= elapsed_time <= start_times[t]:
                        return start_indices[t] - 1 + round((elapsed_time - end_times[t-1])/(start_times[t] - end_times[t-1]))
                return round(start_indices[-1] + sample_rate * (elapsed_time - start_times[-1]))

            if isinstance(time, Iterable):
                return np.array([tti_with_gaps(t) for t in time]).astype(int)
            else:
                return int(tti_with_gaps(time))

        else:
            return np.round(np.multiply(time, self.sample_rate)).astype(int)

    def _to_mne_raw(self):  # TODO make this work with all data if list of dask arrays
        """
        Loads raw data set into an array. This method is very computationally and memory intensive.

        Returns
        -------
        mne.io.array.array.RawArray
            Mne array of the raw data set.

        """
        info = mne.create_info(ch_names=self.data[0].shape[0], sfreq=self.sample_rate, ch_types='ecog')
        with ProgressBar():
            return mne.io.RawArray(np.asarray(self.array.compute()), info, verbose=False)

    def _to_plt_line_collection(self, x_lim, channels, d_l, down_sample=True, remove_gaps=True):
        """

        Converts raw data into an array that can be used to create a matplotlib line collection. With gaps removed, the
        list this method returns will be one element. Otherwise, each tuple in the list will correspond to a chunk of
        data without any gaps.

        Parameters
        ----------
        x_lim : list, tuple, numpy.ndarray
            Sequence containing the start and end times of the plot (in seconds).
        channels: int, slice
            Channel index or slice of indices.
        d_l : int ??
            ??
        down_sample : bool
            Down sample to plot fewer points if True.

        Returns
        -------

        list
            List containging tuples of line collection arrays and boolean indicators.

        """
        # Convert time limits to indices and then get time array
        x_index = self._time_to_index(x_lim, remove_gaps=remove_gaps)
        x_slice = slice(x_index[0], x_index[1])
        time_arr = self.time(remove_gaps=remove_gaps)[x_slice].compute()
        # Get array of plot data
        plot_arr = self.array[channels, x_slice].compute()

        # Down sample data if there more that 8x data points than the number of pixels
        # 8x sampling is arbitrary but found to have a good tradeoff between visual appearance and speed

        def to_line_array(plot_array, time):
            if down_sample and len(time) > d_l*8:
                plot_data = np.zeros((plot_array.shape[0], d_l*4, 2))
                # Simultaneously down_sample data and reshape array to the correct shape for a matplotlib LineCollection
                for channel in range(plot_array.shape[0]):
                    full_sampled = np.stack([plot_array[channel], time]).T
                    down_sampled = largest_triangle_three_buckets(full_sampled, d_l*4)
                    plot_data[channel, :, :] = np.flip(down_sampled, axis=1)
                down_sampled_bool = True  # indicate that the data has been down sampled
            else:
                # Reshape array to the correct shape for a matplotlib LineCollection
                time_array = time[np.newaxis, :, np.newaxis].repeat(plot_array.shape[0], 0)
                plot_data = np.concatenate([time_array, plot_array[:, :, np.newaxis]], axis=2)
                down_sampled_bool = False  # indicate that the data has not been down sampled
            return plot_data, down_sampled_bool

        if remove_gaps:
            return [to_line_array(plot_arr, time_arr)]
        else:
            # TODO: investigate down_sampled_bool and effects of downsampling some data sets but not others
            # split arrays up by data sets and make a list of linecollection arrays
            splitters = [i - x_index[0] for i in self.start_indices if i > x_index[0] and i < x_index[1]]
            plot_arr = np.split(plot_arr, splitters, axis=1)
            time_arr = np.split(time_arr, splitters, axis=0)
            return [to_line_array(plot_arr[i], time_arr[i]) for i in range(len(plot_arr))]

    @property
    def _ch_num_mask_by_type(self):
        """
        Displays the number mask of channels by type.
        :return: dictionary
            Dictionary of channel types and counr
            Example: LIFE: 1, 1, 1, 1, 0,0,0; EMG:0,0,0,0,1,1,1
        """
        vals = self.types
        d = {}
        for val in vals:
            count = [ch_type == val for ch_type in self.metadata[0]['types']]
            d[val] = count
        return d

    def _introduce_offsets(self, ch_offsets):
        # TODO: this only works with negative channel offsets
        # Introduces offsets for channels in array
        ch_offsets = _to_numeric_array(ch_offsets, dtype=int)
        unique_offsets = set(ch_offsets)
        data_lengths = [d.shape[1] for d in self.data]

        low_channel = []  # create empty array for storing info on reordering arrays
        temp_data = [[] for d in self.data]  # create empty arrays for reordering data and stacking to dask arrays
        min_offset = min(unique_offsets)
        for o in unique_offsets:
            ch_pos = ch_offsets == o
            ch_pos = _group_consecutive(np.arange(len(ch_pos))[ch_pos])
            ch_pos = [(c[0], c[-1] + 1) for c in ch_pos]
            for c in ch_pos:
                low_channel.append(c)
                for i, d in enumerate(self.data):
                    data_chunk = d[c[0]:c[1], -o:data_lengths[i] + (min_offset - o)]
                    temp_data[i].append((data_chunk, c[0]))   # appends a tuple to keep track of the original position

        # sorting function to reorder arrays for concatenation
        def orderpos(row):
            return row[1]

        output_data = []
        for t in temp_data:
            if len(t) > 1:
                t.sort(key=orderpos)
                t = [row[0] for row in t]
                output_data.append(da.concatenate(t, axis=0))
            else:
                output_data.append(t[0][0])
        return output_data

    # def _chop(self, block_id):
    #     # chops off data in one data set to line up channels by channel offset
    #     start = min(self.metadata[block_id]["ch_offsets"])
    #     end = max(self.metadata[block_id]["ch_offsets"])
    #     original_shape = self.orig_data[block_id].shape
    #     # chop off excess data at start/end of each channel
    #     chopped_data = []
    #     for i, offset in enumerate(self.metadata[block_id]["ch_offsets"]):
    #         subset = slice(end - offset, original_shape[1] + start - offset)
    #         chopped_row = self.orig_data[block_id][i, subset]
    #         chopped_data.append(chopped_row)
    #     return da.stack(chopped_data)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        pass

    def __getitem__(self, items):
        return self.array[items].compute()
