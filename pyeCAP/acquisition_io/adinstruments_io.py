
# scientific imports
import dask.array as da
import scipy.io as sio
import numpy as np

import struct
import datetime as dt
import warnings


def check_data(raw_data):
    """
    Warns user of potential errors in reading in the data.

    Parameters
    ----------
    raw_data : dict
        Python dictionary with keys corresponding to matlab variables and numpy data arrays as values.

    Returns
    -------
    None
    """

    # TODO: add more tests to catch unworkable data

    # check for proper array dimensions
    num_blocks = len(raw_data['blocktimes'].flatten())
    num_channels = len(raw_data['titles'].flatten())
    # short circuit each and operator to check if arrays are named properly before using them as keys
    if 'data' in raw_data and raw_data['data'].shape[0] != 1:
        warnings.warn("Improperly formatted array 'data'")
    if 'datastart' in raw_data and raw_data['datastart'].shape != (num_channels, num_blocks):
        warnings.warn("Improperly formatted array 'datastart'")
    if 'dataend' in raw_data and raw_data['dataend'].shape != (num_channels, num_blocks):
        warnings.warn("Improperly formatted array 'dataend'")
    if 'samplerate' in raw_data and raw_data['samplerate'].shape != (num_channels, num_blocks):
        warnings.warn("Improperly formatted array 'samplerate'")
    if 'firstsampleoffset' in raw_data and raw_data['firstsampleoffset'].shape != (num_channels, num_blocks):
        warnings.warn("Improperly formatted array 'firstsampleoffset'")
    if 'tickrate' in raw_data and raw_data['tickrate'].shape != (1, num_blocks):
        warnings.warn("Improperly formatted vector 'tickrate'")
    if 'blocktimes' in raw_data and raw_data['blocktimes'].shape != (1, num_blocks):
        warnings.warn("Improperly formatted vector 'blocktimes'")
    if 'unittext' in raw_data and raw_data['unittext'].ndim != 1:
        warnings.warn("Improperly formatted vector 'unittext'")
    if 'unittextmap' in raw_data and raw_data['unittextmap'].shape != (num_channels, num_blocks):
        warnings.warn("Improperly formatted array 'unittextmap'")

    # check for consistent sample rates
    sample = raw_data['samplerate']
    for block_idx in range(len(sample.transpose())):
        if len(np.unique(sample.transpose()[block_idx])) != 1:
            warnings.warn("Block {} has inconsistent sample rates".format(block_idx))
    if len(np.unique(raw_data['tickrate'][0])) != 1:
        warnings.warn("Inconsistent sample rates between blocks")

    # check for nan values
    warnings.filterwarnings(action="ignore", module="numpy")
    if np.isnan(np.sum(raw_data['data'])):
        warnings.warn("Encountered NaN values in data")

    # check for offsets in samples
    if np.sum(raw_data['firstsampleoffset'].flatten()) != 0:
        warnings.warn("Offset in data block(s)")

    # check for unit inconsistencies
    for channel in range(len(raw_data['unittextmap'])):
        if len(np.unique(raw_data['unittextmap'][channel])) != 1:
            warnings.warn("Inconsistent units in channel {}".format(raw_data["titles"][channel]))

    # check for missing data
    if True in np.unique(raw_data['datastart'] == -1):
        warnings.warn("Channel data does not align into a rectangular array."
                      " Try padding the data with the pyeCAP.phys.pad_array function")


def convert_time(matlab_time):
    """
    Converts matlab datetime format to python timestamp.

    Parameters
    ----------
    matlab_time : float
        Matlab datetime format

    Returns
    -------
    float
        Time since the Unix Epoch in seconds.
    """
    py_time = dt.datetime.fromordinal(int(matlab_time)) + dt.timedelta(days=matlab_time % 1) - dt.timedelta(days=366)
    return py_time.timestamp()


def to_array(raw_array, index_start_array, index_end_array, mult_data):
    """
    Creates a properly formatted dask array to insert into the Phys or _TsData constructor.

    Parameters
    ----------
    raw_array : numpy.ndarray
        Array of all raw data points. ('data')
    index_start_array : numpy.ndarray
        Array of matlab start indices for each channel and block combination. ('datastart')
    index_end_array : numpy.ndarray
        Array of matlab end indices for each channel and block combination. ('dataend')
    mult_data : bool
        Indicator of whether or not to treat the file as one data set or multiple.

    Returns
    -------
    dask.array.core.Array
        Raw data formatted into an array.
    """
    # get shape of the array
    shape = index_start_array.shape
    num_channels = shape[0]
    num_divisions = shape[1]

    # data sets by blocks
    da_blocks = []
    for i in range(num_divisions):
        # for each block, create an array
        da_col = []
        for j in range(num_channels):
            start_idx = int(index_start_array[j][i] - 1)    # matlab indices start from 1
            end_idx = int(index_end_array[j][i])
            if start_idx >= 0 and end_idx > 0:     # ensures channel is not empty
                da_col.append(raw_array[0, start_idx: end_idx])
        da_blocks.append(da.rechunk(da.stack(da_col, axis=0), (1, 204800)))

    if mult_data:
        # return list of block arrays
        return da_blocks
    else:
        # return concatenated array
        return [da.rechunk(da.concatenate(da_blocks, axis=1), (1, 204800))]


def to_meta(start_indices, end_indices, tick, channels, units, units_map, start_times, mult_data):
    """
    Generates metadata for a Phys object.

    Parameters
    ----------
    start_indices : numpy.ndarray
        Array of matlab start indices for each channel and block combination. ('datastart')
    end_indices : numpy.ndarray
        Array of matlab end indices for each channel and block combination. ('dataend')
    tick : numpy.ndarray
        Array of highest sample rates for each block. ('tickrate')
    channels : numpy.ndarray
        Array of channel names. ('titles')
    units : numpy.ndarray
        Array of channel units. ('unittext')
    units_map : numpy.ndarray
        Array of indices linking units to channel and block. ('unittextmap')
    start_times : numpy.ndarray
        Array of start times for each block (matlab datetime format). ('blocktimes')
    mult_data : bool
        Indicator of whether or not to treat the file as one data set or multiple.

    Returns
    -------
    list
        Metadata dictionaries for each data set.

    """
    # generate lists for each metadata value
    sample_rate = list(tick.flatten())
    ch_names = [list(channels)]*len(sample_rate)
    unit_names = []
    for i in range(len(sample_rate)):
        unit_names.append([units[int(units_map[j][i])-1] for j in range(len(channels))])
    starts = [convert_time(st) for st in list(start_times.flatten())]
    ends = [starts[i] + (end_indices[0, i] - start_indices[0, i]) / float(sample_rate[i]) for i in range(len(sample_rate))] #TODO: eliminate hardcoding

    # create metadata dictionaries for each block
    metas = ['start_time', 'end_time', 'ch_names', 'sample_rate', 'units']
    meta_dicts = []
    for i in range(len(sample_rate)):
        meta_dicts.append(dict(zip(metas, [starts[i], ends[i], ch_names[i], sample_rate[i], unit_names[i]])))

    if mult_data:
        # return metadata for each block
        return meta_dicts
    else:
        # return metadata for the entire data set, use end time of last block
        meta_dicts[0]['end_time'] = meta_dicts[-1]['end_time']
        return [meta_dicts[0]]


def read_headers(file_header, channel_headers):
    # function for reading ADInstruments binary file headers
    num_samples = file_header[14]
    data_format = file_header[16]
    num_channels = len(channel_headers)

    # get time/sample rate metadata
    start_time = dt.datetime(file_header[6], file_header[7], file_header[8], file_header[9], file_header[10], int(file_header[11]),
                                   int(np.round((np.round(file_header[11], 6) - int(file_header[11])) * 10 ** 6))).timestamp()
    end_time = start_time + file_header[5] * num_samples
    metadata = {'start_time': start_time,
                'end_time': end_time,
                'ch_names': [],
                'sample_rate': 1 / file_header[5],
                'units': []
                }
    # read channel headers to add to metadata
    for channel_header in channel_headers:
        channel_text = "".join([c.decode("iso-8859-1") for c in channel_header[:32] if c != b'\x00'])
        unit_text = "".join([c.decode("iso-8859-1") for c in channel_header[32:64] if c != b'\x00'])
        metadata['ch_names'].append(channel_text)
        metadata['units'].append(unit_text)

    return metadata, num_samples, num_channels, data_format


class AdInstrumentsIO:
    # class for AD instruments files IO
    def __init__(self, data, mult_data, check):
        # create a reader for the file
        if data.endswith(".mat"):
            self.reader = ADInstrumentsMAT(data, mult_data, check)
        elif data.endswith(".adibin"):
            self.reader = ADInstrumentsBin(data)
        else:
            raise IOError('Unreadable file. Files must be .mat or .adibin')

        # get data from the reader
        self.array = self.reader.array
        self.metadata = self.reader.metadata
        self.chunks = self.reader.chunks


class ADInstrumentsMAT:
    # class for AD Instruments .mat files
    def __init__(self, data, mult_data, check):
        try:
            raw = sio.loadmat(data)
        except ValueError as e:
            raise IOError(f"{e} \n \n Unreadable .mat file, file may be too large")
        except Exception as e:
            raise IOError(f"Exception {e} occured during file reading")

        # check data for bad structure, ect
        if check:
            # TODO: add check for missing data
            check_data(raw)

            # map parameters in as numpy arrays
            mapping = [raw['data'], raw['datastart'], raw['dataend'], raw['tickrate'], raw['titles'], raw['blocktimes'],
                       raw['unittext'], raw['unittextmap']]
            raw_array, start_indices, end_indices, tick, channels, start_times, units, unit_map = mapping

            # define properties by calling array and metadata functions
            self.array = to_array(raw_array, start_indices, end_indices, mult_data)
            self.metadata = to_meta(start_indices, end_indices, tick, channels, units, unit_map, start_times, mult_data)
            self.chunks = [(1, 204800)] * len(self.metadata)


class ADInstrumentsBin:
    # ADInstruments binary file reader class
    # info on binary files:
    # http://cdn.adinstruments.com/adi-web/manuals/translatebinary/LabChartBinaryFormat.pdf

    def __init__(self, filename):
        # read in binary headers
        with open(filename, 'rb') as f:
            file_head = struct.unpack('<4cld5l2d4l', f.read(68))

            if file_head[0].decode("iso-8859-1") != 'C':    # file header always begins with "CWFB"
                raise ValueError("File is missing header")

            channel_heads = []
            for i in range(file_head[13]):
                channel_heads.append(struct.unpack('<64c4d', f.read(96)))

        # get metadata from headers
        metadata, num_samples, num_channels, data_format = read_headers(file_head, channel_heads)
        dtype_dict = {1: "double", 2: "float32", 3: "short"}

        # read in the binary array from metadata
        if data_format == 1:
            # TODO: implement scaling and offset for adinstruments integer files
            raise NotImplementedError("Reading of integer binary files is not yet implemented")

        raw_array = np.memmap(filename, mode='r', dtype=dtype_dict[data_format], offset=68+96*num_channels)
        if not raw_array.size == num_samples*num_channels:  # check for improperly formatted data
            raise ValueError("Improper array size. Ensure that arrays are exported without time data")
        self.array = [da.from_array(raw_array.reshape(num_samples, num_channels).T, (1, 204800))]
        self.metadata = [metadata]
        self.chunks = [(1, 204800)]*num_channels
