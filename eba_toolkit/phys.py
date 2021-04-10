
# scientific library imports
import scipy.io as sio
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

# neuro base class imports
from .base.ts_data import _TsData
from .base.utils.base import _is_iterable
from .base.utils.visualization import _plt_ax_to_pix, _plt_setup_fig_axis, _plt_show_fig, _plt_add_ax_connected_top

# eba_toolkit io class imports
from .io.adinstruments_io import AdInstrumentsIO, convert_time, ADInstrumentsBin

# TODO: Comments could be used for event data


def pad_array(filename):
    """
    Pads a .mat file with NaN values when blocks of data are missing on one or more channels.

    Parameters
    ----------
    filename: str
        Path to the .mat file.

    Returns
    _______
    str
        Path of the new padded .mat file.
    """
    # read in data
    raw_dict = sio.loadmat(filename)
    datastart = raw_dict['datastart']
    data = raw_dict['data']
    dataend = raw_dict['dataend']

    # find block lengths and account for any missing data
    num_channels, num_blocks = datastart.shape
    lens = dataend - datastart + 1
    block_lengths = lens[0]
    for b in range(len(block_lengths)):
        if block_lengths[b] <= 1:
            for c in range(num_channels):
                if lens[c, b] != -1:
                    block_lengths[b] = lens[c, b]
                    break

    # new datastart/dataend vectors
    new_datastart = np.ones((num_channels, num_blocks))
    for c in range(num_channels):
        for b in range(num_blocks):
            new_datastart[c, b] += num_channels * sum(block_lengths[:b]) + c * block_lengths[b]

    new_dataend = np.zeros((num_channels, num_blocks))
    for c in range(num_channels):
        new_dataend[c] += new_datastart[c] + block_lengths - 1

    # pad the data array
    def datastart_sort(idxs):
        return new_datastart[idxs[0], idxs[1]]

    missing_blocks = []
    for c in range(num_channels):
        for b in range(num_blocks):
            if datastart[c, b] == -1:
                missing_blocks.append((c, b))

    missing_blocks.sort(key=datastart_sort)

    for m in missing_blocks:
        pad_arr = np.zeros((1, int(block_lengths[m[1]])))
        pad_arr[:] = np.nan
        data_left = data[:, :int(new_datastart[m[0], m[1]]) - 1]
        data_right = data[:, int(new_datastart[m[0], m[1]]) - 1:]
        data = np.concatenate((data_left, pad_arr, data_right), axis=1)

    # replace old values in dictionary with new ones
    raw_dict['data'] = data
    raw_dict['datastart'] = new_datastart
    raw_dict['dataend'] = new_dataend
    raw_dict['unittext'] = np.append(raw_dict['unittext'], "N/A")
    raw_dict['unittextmap'] = np.where(raw_dict['unittextmap'] == -1, len(raw_dict['unittext']), raw_dict['unittextmap'])

    # write dictionary to a matlab file
    sio.savemat(filename.replace(".mat", "_padded.mat"), raw_dict, do_compression=True)
    return filename.replace(".mat", "_padded.mat")


def get_comments(file, comtype=None):
    """
    Reads the comments in the specified file path. Creates a dictionary with keys as channel names. Values are
    dictionaries with comment names as key names and arrays of timestamps as values.

    Parameters
    ----------
    file : str
        Path of a .mat file.
    comtype : None, str
        Use None to default to any comment type. Specify 'user', 'event', or 'both'.

    Returns
    -------
    dict
        Python dictionary. Index with channel name and comment name.

    Examples
    ________
    >>> eba_toolkit.phys.get_comments(pathname)    # insert pathname to data   # doctest: +SKIP

    """
    # initialize file reader
    raw = sio.loadmat(file)
    comdict = {}
    ch_names = raw['titles']
    for ch in ch_names:
        comdict[ch] = {}

    # return empty comment dictionary if there are no comments
    if 'com' not in raw:
        return comdict

    # set up comments dictionary for each block in format {channel_name: {comment_name}: []}
    com_array = raw['com']
    com_names = raw['comtext'].flatten()

    # load data needed to determine comment timestamps
    times_list = [convert_time(t) for t in raw['blocktimes'].flatten()]
    sample = raw['tickrate'].flatten()

    # set up a list to filter comment types
    types_key = {None: [1,2], "both": [1,2], "user": [1], "event": [2]}
    if comtype not in types_key:
        raise(ValueError("Invalid comment type"))
    types = types_key[comtype]

    # loop over each comment and add it to comdict
    for arr in com_array:
        channel_idx = int(arr[0] - 1)
        block_idx = int(arr[1] - 1)
        com_time = times_list[block_idx] + arr[2] / sample[block_idx]
        try:
            name = com_names[int(arr[-1]) - 1]
        except IndexError:
            name = 'unnamed'

        for ch in comdict:
            if name not in comdict[ch]:
                comdict[ch][name]= []

        if arr[3] in types:
            if channel_idx == -2:
                # checks for comments that apply to all channels
                for ch in comdict:
                    comdict[ch][name].append(com_time)
            else:
                comdict[ch_names[channel_idx]][name].append(com_time)

    # convert lists to numpy arrays to save memory, return result
    for ch in comdict:
        for com_name in comdict[ch]:
            comdict[ch][com_name] = np.array(comdict[ch][com_name])
    return comdict


class Phys(_TsData):
    """Class for physiological data"""
    def __init__(self, data, *args, mult_data=True, check=True, order=True, **kwargs):
        """
        Constructor for the Phys class

        Parameters
        ----------
        data : str, list
            Path name or list of path names to .mat files.
        * args : Arguments
            See :ref:`_TsData (parent class)` constructor.
        mult_data : bool
            Set to True to treat each block as a separate data set, false to treat each all blocks as one data set.
        check : bool
            Set to True to allow warnings about improperly formatted files.
        order : bool
            Set to True to order data sets by start time. Since data from the same file is read in chronological order,
            this will only have an effect when reading in data from multiple files.
        ** kwargs : KeywordArguments
            See :ref:`_TsData (parent class)` constructor

        Examples
        ________
        >>> eba_toolkit.Phys(pathname1)   # replace pathnames with paths to data      # doctest: +SKIP

        >>> eba_toolkit.Phys([pathname1, pathname2, pathname3])                       # doctest: +SKIP

        """

        if isinstance(data, str):
            self.adinstruments = AdInstrumentsIO(data, mult_data, check)
            super().__init__(self.adinstruments.array, self.adinstruments.metadata, daskify=False, chunks=self.adinstruments.chunks)
        elif _is_iterable(data, str):
            self.adinstruments = [AdInstrumentsIO(path, mult_data, check) for path in data]
            array = []
            metadata = []
            chunks = []
            for dataset in self.adinstruments:
                array.extend(dataset.array)
                metadata.extend(dataset.metadata)
                chunks.extend(dataset.chunks)
            super().__init__(array, metadata, daskify=False, chunks=chunks, order=order)
        else:
            # TODO: transfer the comments in metadata properly
            super().__init__(data, *args, **kwargs)

    @property
    def units(self):
        """
        Property getter method for channel units.

        Returns
        -------
        list
            List of dictionaries linking channel name to units for each data set.

        Examples
        ________
        >>> phys_data.units # doctest: +SKIP
        """
        return [dict(zip(self.ch_names, meta['units'])) for meta in self.metadata]

    def select(self, datasets, select=True):
        """
        Select or remove data sets from a Phys object. Creates a new object with the desired data sets. Warning: indices
        and time from the time array may be inconsistent with the origninal object.

        Parameters
        ----------
        datasets : list
            List of data sets to select or remove. Use integer indices to specify data sets.
        select : bool
            Set to True to select the specified data sets or False to remove the specified data sets.

        Returns
        -------
        eba_toolkit.Phys
            New Phys object with desired data sets.

        Examples
        ________
        >>> phys_data.select([0,1,2,4,6])   # doctest: +SKIP
        """
        new_data = []
        new_metadata = []
        chunks = []
        for i in range(self.ndata):
            if (i in datasets and select) or (not select and i not in datasets):
                new_data.append(self.data[i])
                new_metadata.append(self.metadata[i])
                chunks.append(self.chunks[i])

        return type(self)(new_data, new_metadata, chunks=chunks, daskify=False)

    def plot(self, axis=None, channels=None, events=False, x_lim=None, y_lim='max', ch_labels=None,
             colors=sns.color_palette(), fig_size=(15, 6), down_sample=True,
             show=True, remove_gaps=True):
        """
        Plotting method for Phys data. This method overrides the method in _TsData to provide better scaling for Phys
        objects.

        Parameters
        ----------
        axis : None, matplotlib.axis.Axis
            Either None to use a new axis or matplotlib axis to plot on.
        channels : int, str, list, tuple, np.ndarray
            Channels to plot. Can be a boolean numpy array with the same length as the number of channels, an integer
            array, or and array of strings containing channel names.
        events : _DioData, _EventData, or subclass
            Event data plotting for Phys objects not yet supported. Consider using the Phys_Response class to plot
             stimulation events.
        x_lim : None, list, tuple, np.ndarray
            None to plot the entire data set. Otherwise tuple, list, or numpy array of length 2 containing the start of
            end times for data to plot.
        y_lim : str, list
            String 'max' to show all data within the y-limits, or List of two item tuples to assign a y-limit to each
            channel individually.
        ch_labels : list, tuple, np.ndarray
            Iterable of strings to use as channel labels in the plot. Must match length of channels being displayed.
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
         matplotlib.axis.Axis
            matplotlib axis containing the plot.

        """
        # initialize x limits, figure, and channels
        if channels is None:
            channels = self.ch_names
        fig, ax = _plt_setup_fig_axis(axis, fig_size=fig_size)
        x_lim = self._time_lim_validate(x_lim, remove_gaps=remove_gaps)
        x_index = (self._time_to_index(x_lim[0], remove_gaps=remove_gaps), self._time_to_index(x_lim[1], remove_gaps=remove_gaps)+1)
        plt.yticks([])

        # create a subplot for each channel
        for i, channel in enumerate(channels):
            # set up axes limits and labels
            plt.xticks([])
            channel_ax = fig.add_subplot(len(channels) + events, 1, i + 1 + events)
            channel_ax.set_xlim(x_lim)
            if ch_labels is not None:
                channel_ax.set_ylabel(ch_labels[i])
            else:
                channel_ax.set_ylabel("{} ({})".format(channel, self.units[-1][channel]))
            ch_slice = self._ch_to_index(channel)
            plot_array = self.array[ch_slice, x_index[0]:x_index[1]].compute()
            try:
                if y_lim == "max":
                    datamax = np.nanmax(plot_array)
                    datamin = np.nanmin(plot_array)
                    channel_ax.set_ylim(datamin - (datamax-datamin) * .1, datamax + (datamax-datamin) * .1)
                else:
                    channel_ax.set_ylim(y_lim[i])
            except ValueError:
                channel_ax.set_ylim(0, 1)

            # plot data on the axes
            px_width, _ = _plt_ax_to_pix(fig, channel_ax)
            plot_data = self._to_plt_line_collection(x_lim, ch_slice, px_width, remove_gaps=remove_gaps)
            for data in plot_data:
                lines = LineCollection(data[0], linewidths=np.ones(plot_array.shape[0]), transOffset=None,
                                       colors=colors[i % len(colors)])
                channel_ax.add_collection(lines)

        return _plt_show_fig(fig, ax, show)
