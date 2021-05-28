from tdt import read_block, epoc_filter
import os
import glob  # for file and directory handling
import pprint
import numpy as np
import pandas as pd
import dask
import dask.array as da
from dask.cache import Cache
import itertools
import threading  # to lock file for thread safe reading
import warnings

# This can be removed in python 3.8 as they are adding cached properties as a built-in decorator
from cached_property import threaded_cached_property

cache = Cache(0.5e9)  # Leverage 500mb of memory for fast access to data


def gather_sample_delay(rz_sample_rate, si_sample_rate):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base", "TDTSampleDelay.csv"))
    df.set_index('Unnamed: 0', inplace=True)
    if str(rz_sample_rate) in df.columns:
        if si_sample_rate in df.index:
            try:
                return int(df.loc[si_sample_rate, str(rz_sample_rate)])
            except ValueError:
                print('RZ or SI Sample rate not recognized')
        else:
            raise ValueError('SI Sample rate invalid')
    else:
        raise ValueError('RZ Sample rate invalid')

class TdtIO:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock = threading.Lock()

        if isinstance(self.file_path, str):
            if os.path.isdir(self.file_path):
                try:
                    # var, set to 1 to return only the headers for this block, if you need to
                    # make fast consecutive calls to read_block
                    self.tdt_block = read_block(file_path, headers=1)
                except IOError:
                    print(self.file_path + " is not accessible.")
            else:
                raise IOError("File path not found")
        else:
            raise IOError("Path was not formatted as a string.")

    @property
    def stores(self):
        return list(self.tdt_block.stores.keys())


class TdtStim:
    def __init__(self, tdt_io, type='stim'):
        if isinstance(tdt_io, TdtIO):
            self.tdt_io = tdt_io
        else:
            raise TypeError("input expected to be of type TdtIO")
        self.type = type

    @threaded_cached_property
    def metadata(self):
        metadata = {'start_time': self.tdt_io.tdt_block['start_time'][0],
                    'stop_time': self.tdt_io.tdt_block['stop_time'][0]}

        stores = self.tdt_io.tdt_block.stores
        metadata['stores'] = []
        for idx, k in enumerate(stores.keys()):
            # Stimulation data is by default recorded in a pair of stores withe the same first 4 characters followed
            # by a p or r. Store ending with 'p' is for stimulation parameters and is 'scalar' data, 'r' is a raw
            # recording of stimulation waveform and is a 'stream'
            if stores[k]['type_str'] == 'scalars' and (k[-1] == 'p' and k[:-1] + 'r' in stores.keys()):
                metadata['stores'].append(k)
                metadata['num_simulations'] = stores[k].size
                metadata['stimulation_onsets'] = np.sort(np.unique(stores[k].ts))
        channels = []
        for k in metadata['stores']:
            for onset in metadata['stimulation_onsets']:
                ch_idx = stores[k].ts == onset
                ch_sorted = np.argsort(stores[k].chan[ch_idx])
                channels.append(stores[k].data[ch_idx][ch_sorted][5])
        metadata['channels'] = list(np.unique(np.array(channels).astype(int)))
        metadata['ch_names'] = ['Stim ' + str(ch) for ch in metadata['channels']]
        return metadata

    # @threaded_cached_property
    # def parameters(self):
    #     stores = self.tdt_io.tdt_block.stores
    #     stim_stores = {}
    #
    #     for k in self.metadata['stores']:
    #         stim_data = []
    #         for onset in self.metadata['stimulation_onsets']:
    #             # This has nothing to do with the stimulation channel but tdt refers to each stim parameter as a channel
    #             # in their storage format
    #             ch_idx = stores[k].ts == onset
    #             ch_sorted = np.argsort(stores[k].chan[ch_idx])
    #             # only first 6 parameters are used append to onset time to match order of stim_data DataFrame.
    #             parameters = [onset] + list(stores[k].data[ch_idx][ch_sorted])
    #             stim_data.append(parameters)
    #         stim_data = np.array(stim_data)
    #         print(stim_data)
    #         # If the 'electrical stim' gizmo in tdt was used to aquire data then the first parameter should be 0
    #         zero_columns = np.all(stim_data == 0.0, axis=0)
    #         if zero_columns[1]:
    #             parameter_names = ['onset time (s)', 'channel', 'stimulation gain', 'pulse count', 'period (ms)',
    #                                'pulse amplitude (μA)', 'pulse duration (ms)']
    #             # Pulse with segment A only (Monophasic)
    #             if np.all(zero_columns[7:11]):
    #                 channels = [0] + list(range(1, 7))
    #             # Pulse with segments A, B (Biphasic)
    #             elif np.all(zero_columns[9:11]):
    #                 channels = [0] + list(range(1, 9))
    #                 # Add additional parameter names for channel 2
    #                 parameter_names += ['pulse amplitude 2 (μA)',
    #                                     'pulse duration 2 (ms)']
    #             # Pulse with segments A, B, C (Biphasic + Interphase delay)
    #             else:
    #                 raise NotImplementedError("Stim format with biphasic pulse and interphase delay using the TDT "
    #                                           "'Electrical Stimulation' gizmo is not yet implemented.")
    #                 channels = [0] + list(range(1, 11))
    #             stim_data = stim_data[:, channels]
    #         # If the 'electrical stim driver' gizmo is used then the first parameter is never zero, since it
    #         # represents the period of the waveform
    #         else:
    #             stim_data = stim_data[:, 0:7]
    #             parameter_names = ['onset time (s)', 'period (ms)', 'pulse count', 'pulse amplitude (μA)',
    #                                'pulse duration (ms)', 'interphase delay (ms)', 'channel']
    #         parameter_dataframe = pd.DataFrame(stim_data, index=range(len(stim_data)), columns=parameter_names)
    #         parameter_dataframe = parameter_dataframe.astype({'pulse count': int, 'channel': int})
    #         parameter_dataframe.insert(2, "frequency (Hz)", 1000/parameter_dataframe['period (ms)'])
    #         parameter_dataframe.insert(5, "duration (ms)",
    #                                    parameter_dataframe['period (ms)']*parameter_dataframe['pulse count'])
    #         parameter_dataframe.insert(1, "offset time (s)",
    #                                    parameter_dataframe['onset time (s)']+parameter_dataframe['duration (ms)']/1000)
    #         stim_stores[k] = parameter_dataframe
    #         self._parameters = stim_stores
    #         if len(self.metadata['stores']) == 1:
    #             return self._parameters[self.metadata['stores'][0]]
    #         else:
    #             return self._parameters

    @threaded_cached_property
    def parameters(self):
        stores = self.tdt_io.tdt_block.stores

        # TODO: generate error messages for missing eS1p
        # TODO: check for 'eS1p/' or 'eS1e/'
        # TODO: generate proper channel names with the 'Electrical Stimulation' gizmo
        # convert data in stores to eS1p format
        stim_data = []
        for onset in self.metadata['stimulation_onsets']:
            ch_idx = stores['eS1p'].ts == onset
            ch_sorted = np.argsort(stores['eS1p'].chan[ch_idx])
            parameters = list(stores['eS1p'].data[ch_idx][ch_sorted])
            stim_data.append(parameters)

        stim_parameters = np.array(stim_data)
        onsets = np.reshape(self.metadata['stimulation_onsets'], (-1, 1))
        zero_cols = np.all(stim_parameters == 0.0, axis=0)
        voices = []

        if zero_cols[0]:
            # 'Electrical Stimulation' Gizmo was used
            col_names = ['onset time (s)', 'channel', 'stimulation gain', 'pulse count', 'period (ms)',
                         'pulse amplitude A (μA)', 'pulse duration A (ms)']
            channels = np.zeros((stim_parameters.shape[0], 1))
            voices.append("")
            if np.all(zero_cols[6:10]):
                # A segment only
                stim_data = np.concatenate((onsets, channels, stim_parameters[:, 1:6]), axis=1)
            elif np.all(zero_cols[8:10]):
                col_names += ['pulse amplitude B (μA)', 'pulse duration B (ms)']
                stim_data = np.concatenate((onsets, channels, stim_parameters[:, 1:8]), axis=1)
            else:
                # A, B, and C segments
                col_names += ['pulse amplitude B (μA)', 'pulse duration B (ms)', 'pulse amplitude C (μA)',
                              'pulse duration C (ms)']
                stim_data = np.concatenate((onsets, channels, stim_parameters[:, 1:10]), axis=1)
        else:
            # 'Electrical Stim Driver was used'
            col_names = ['onset time (s)']
            stim_data = onsets
            if np.all(zero_cols[6:11]) and not (zero_cols[11]):
                # Bipolar up to 2 voices
                possible_voices = [" A", " C"]
                bipolar = True
            else:
                # Monopolar up to 4 voices
                possible_voices = [" A", " B", " C", " D"]
                bipolar = False

            # for each possible voice, get appropriate data and column names
            for i, voice in enumerate(possible_voices):
                if not np.all(zero_cols[i * 24 // len(possible_voices):(i + 1) * 24 // len(possible_voices)]):
                    voices.append(voice)
                    col_names += [f'period{voice} (ms)', f'pulse count{voice}', f'pulse amplitude{voice} (μA)',
                                  f'pulse duration{voice} (ms)', f'interphase delay{voice} (ms)', f'channel{voice}']
                    if bipolar:
                        stim_data = np.concatenate((stim_data, stim_parameters[:, i * 12:i * 12 + 6],
                                                    stim_parameters[:, (i + 1) * 12 - 1:(i + 1) * 12]), axis=1)
                        col_names += [f'bipolar channel{voice}']
                    else:
                        stim_data = np.concatenate((stim_data, stim_parameters[:, i * 6:(i + 1) * 6]), axis=1)

        # add in new calculated columns for each vocie
        parameter_dataframe = pd.DataFrame(stim_data, columns=col_names)
        for i, voice in enumerate(voices):
            parameter_dataframe = parameter_dataframe.astype({f'pulse count{voice}': int, f'channel{voice}': int})
            parameter_dataframe.insert(2 + i * 6, f"frequency{voice} (Hz)", 1000 / parameter_dataframe[f'period{voice} (ms)'])
            parameter_dataframe.insert(5 + i * 6, f"duration{voice} (ms)", parameter_dataframe[f'period{voice} (ms)'] *
                                       parameter_dataframe[f'pulse count{voice}'])
            parameter_dataframe.insert(1, f"offset time{voice} (s)", parameter_dataframe['onset time (s)'] +
                                       parameter_dataframe[f'duration{voice} (ms)'] / 1000)
        return parameter_dataframe

    def dio(self, indicators=False):
        dio = {}
        params = self.parameters
        for ch, name in zip(self.metadata['channels'], self.metadata['ch_names']):
            ch_params = params[params['channel'] == ch]
            if indicators:
                indicator_data = np.repeat(ch_params.index, 2)
                dio[name] = indicator_data
            else:
                dio_data = np.zeros((len(ch_params)*2,), dtype=float)
                onsets = np.array(ch_params['onset time (s)'])
                offsets = np.array(ch_params['offset time (s)'])
                dio_data[0::2] = onsets
                dio_data[1::2] = offsets
                dio[name] = dio_data
        return dio

    def events(self, indicators=False):
        events = {}
        params = self.parameters
        for ch, name in zip(self.metadata['channels'], self.metadata['ch_names']):
            ch_params = params[params['channel'] == ch]
            num_events = np.sum(ch_params['pulse count'])
            ch_events = np.zeros((num_events,), dtype=float)
            idx_pointer = 0
            if indicators:
                for index, row in params.iterrows():
                    num_stim_events = int(row['pulse count'])
                    stim_indicators = np.ones((num_stim_events,))*index
                    ch_events[idx_pointer:idx_pointer+num_stim_events] = stim_indicators
                    idx_pointer += num_stim_events
            else:
                for index, row in params.iterrows():
                    stim_events = row['onset time (s)'] + np.arange(0, row['pulse count'])*row['period (ms)']/1000
                    num_stim_events = int(row['pulse count'])
                    ch_events[idx_pointer:idx_pointer+num_stim_events] = stim_events
                    idx_pointer += num_stim_events
            events[name] = ch_events
        return events

    def __getitem__(self, items):
        if len(self.metadata['stores']) == 1:
            if isinstance(items, (int, list, slice)):
                return self.parameters.loc[items, :]
        else:
            raise ValueError('expected 1 stim store, received ' + str(len(self.metadata['stores'])))


# class TdtEvents:
#     def __init__(self, tdt_io, type='stim'):
#         # check if input tdt_io is instance of TdtIO class
#         if isinstance(tdt_io, TdtIO):
#             self.tdt_io = tdt_io
#         else:
#             raise TypeError("input expected to be of type TdtIO")
#
#         # check if type input is correct
#         if type in ['stim', 'stim_onset', 'stim_pulses']:
#             self.type = type
#         else:
#             raise AttributeError("Unexpected event type attribute received")
#
#
#     @threaded_cached_property
#     def metadata(self):
#         pass


class TdtArray:
    """ Stores data in  an array by channel and index
        Default: All data from all stores and channels

        Streams are alphanumerically iterated through and stored accordingly in metadata['streams']

    """

    def __init__(self, tdt_io, type='ephys', stores=None, chunk_size=2040800):

        # Check and handle inputs
        # Validate tdt_io input
        if isinstance(tdt_io, TdtIO):
            self.tdt_io = tdt_io
        else:
            raise TypeError("input expected to be of type TdtIO")

        # Validate type input
        # TODO: Need to update array reading so that it uses ephys vs. stim to pull appropriate data
        if type in ['ephys', 'stim']:
            self.type = type
        else:
            raise AttributeError("Unexpected event type attribute received")

        # Set up stores list
        if stores is None:
            self.stores = self.tdt_io.stores
        else:
            stores = np.array(stores)
            self.stores = list(stores[np.isin(stores, self.tdt_io.stores)])

        # Validate the selected stores are valid for conversion to an array
        s_freqs = self.metadata['sample_rate']
        s_lengths = self.metadata['stream_lengths']
        if isinstance(s_freqs, list):
            raise IOError("Selected stores contain arrays with different sampling rates. Use input 'stores' to select "
                          "a subset of TDT stores with compatible sampling rates.")

        if isinstance(s_lengths, list):
            # ToDo: yell at TDT for having incorrect block size.
            if max(s_lengths) - min(s_lengths) == self.metadata['block_size'] - 10:
                warnings.warn("Block size mismatch by one block. Adjusting stream length to shortest length.")
                self.metadata['stream_lengths'] = min(s_lengths)
                for len_idx, lens in enumerate(s_lengths):
                    if lens > min(s_lengths):
                        current_stream = self.stores[len_idx]
                        # use numpy to delete last block within TdTStruct of data and channel (removes last block entirely)
                        # TdTStruct were not able to be modified in situ
                        num_channels = int(max(self.tdt_io.tdt_block.stores[current_stream].chan))
                        self.tdt_io.tdt_block.stores[current_stream].data = self.tdt_io.tdt_block.stores[current_stream].data[:-num_channels]
                        self.tdt_io.tdt_block.stores[current_stream].chan = self.tdt_io.tdt_block.stores[current_stream].chan[:-num_channels]
                        dump = 1
            else:
                raise IOError("Selected stores contain arrays with different numbers of samples. Use input 'stores' to "
                              "select a subset of TDT stores with compatible sample numbers.")

        # TODO: need to correctly handle tdt types properly
        # Create aray in dask format
        # Calculate array size
        self.shape = (len(self.metadata['ch_names']), self.metadata['stream_lengths'])

        # Get tev file path to pull data from
        tev_files = glob.glob(self.tdt_io.file_path + '/*.tev')  # There should only be one
        if len(tev_files) == 0:
            raise FileNotFoundError("Could not located '*.tev' file expected for tdt tank.")
        elif len(tev_files) > 1:
            raise FileExistsError("Multiple '*.tev' files found in tank, 1 expected.")
        else:
            self.tev_file = tev_files[0]

            # create list of data locations
            data_arrays = []  # place to store lists of dask delayed object for each segment on disk
            stores = self.tdt_io.tdt_block.stores
            # Todo: yell at TDT about block_size
            self.block_size = int(self.metadata['block_size'] - 10)
            self.np_dtype = np.dtype('f')

            tev_file = self.tev_file
            np_dtype = self.np_dtype
            block_size = self.block_size
            block_bits = block_size * np_dtype.itemsize

            @dask.delayed
            def load_block(offsets):
                f = open(tev_file, 'r')
                n_offsets = np.copy(offsets)
                n_offsets[1:] = np.diff(offsets) - block_bits
                # print((f, np_dtype, block_size, offsets))
                block = [np.fromfile(f, dtype=np_dtype, count=block_size, offset=offset) for offset in n_offsets]
                block = np.concatenate(block)
                return block

            n = int(chunk_size / block_size)
            for key in np.asarray(self.metadata['streams']).flatten():
                channel_list = np.sort(np.unique(self.tdt_io.tdt_block.stores[key].chan))
                for channel in channel_list:
                    data_offsets = np.array(self.tdt_io.tdt_block.stores[key].data[self.tdt_io.tdt_block.stores[key].chan == channel], dtype=int)
                    # Check for sev file that will exist if files saved seperately
                    sev_file = os.path.splitext(tev_file)[0] + '_' + key + '_Ch' + str(channel) + '.sev'
                    if os.path.isfile(sev_file):
                        # if sev file exists then data is stored sequentially on disk and can be quickly memmaped (this should be faster)
                        f_offset = data_offsets[0]
                        # f = open(tev_file, 'r')
                        mm_array = np.memmap(sev_file, shape=len(data_offsets)*block_size, dtype=np_dtype, offset=f_offset)
                        ch_array = da.from_array(mm_array, chunks=(chunk_size,))
                        data_arrays.append(ch_array)
                    else:
                        # if not then we need to individually map each block
                        data_offsets = [data_offsets[i:i + n] for i in range(0, len(data_offsets), n)]
                        data_len = [len(offsets) for offsets in data_offsets]
                        data_delayed = [(load_block(np.array(offsets)), l) for offsets, l in zip(data_offsets, data_len)]
                        dask_arrays = [da.from_delayed(d, (self.block_size * l,), dtype=self.np_dtype) for d, l in data_delayed]
                        data_arrays.append(da.concatenate(dask_arrays))
            self.data = da.stack(data_arrays)
            self.chunks = (1, chunk_size)

    @dask.delayed
    def load_block(self, offset):
        return np.fromfile(self.tev_file, dtype=self.np_dtype, count=self.block_size, offset=offset)

    @threaded_cached_property
    def metadata(self):
        metadata = {'start_time': self.tdt_io.tdt_block['start_time'][0],
                    'stop_time': self.tdt_io.tdt_block['stop_time'][0]}

        stores = self.tdt_io.tdt_block.stores
        metadata['block_size'] = []
        metadata['streams'] = []
        metadata['stream_lengths'] = []
        metadata['sample_rate'] = []
        metadata['file_location'] = self.tdt_io.file_path
        for idx, k in enumerate(stores.keys()):
            # TODO: Deal with streams that have different sampling rates, lengths, onset times, data types or other
            #  differences that prevent them from being accessed in the same array

            # Stores have a 4-5 character string identifier. Stimulation data is by default recorded in a pair of
            # stores withe the same first 4 characters followed by a p or r. Store ending with 'p' is for stimulation
            # parameters and is 'scalar' data, 'r' is a raw recording of stimulation waveform and is a 'stream'
            if k in self.stores and stores[k]['type_str'] == 'streams' and (k[-1] != 'r' and k[:-1] + 'p' not in stores.keys()):
                metadata['streams'].append(k)
                metadata['sample_rate'].append(stores[k]['fs'])
                metadata['block_size'].append(stores[k]['size'])

        # get number of channels within each stream
        metadata['channels'] = []
        for k in metadata['streams']:
            # noinspection PyTypeChecker
            metadata['channels'].append([k + " " + str(i + 1) for i in range(np.max(stores[k]['chan']))])
        metadata['ch_names'] = list(itertools.chain.from_iterable(metadata['channels']))

        # Setup variables and code to determine enumeration through streams. That is a recording with 4 RawE (LIFE)
        # channels, 4 RawG (EMG) channels and 2 _. channels will have the shape [4, 4, 2]
        channel_array = metadata['channels']
        metadata['channels_per_stream'] = [len(i) for i in channel_array]
        metadata['cumulative_channel_count'] = [metadata['channels_per_stream'][0]]
        for i in range(1, len(metadata['channels_per_stream'])):
            metadata['cumulative_channel_count'].append(
                metadata['channels_per_stream'][i] + metadata['cumulative_channel_count'][i - 1])
        metadata['stream_lengths'] = []

        for idx, name in enumerate(metadata['streams']):
            header_length = len(self.tdt_io.tdt_block.stores[name].data)
            # Case where waveform is present. It has two channels, but one is the anodic portion and the other is the
            # cathodic. So really, 1 portion
            if name.startswith('_'):
                channels_in_stream = 1
            else:
                channels_in_stream = metadata['channels_per_stream'][idx]
            metadata['stream_lengths'].append(int((header_length / channels_in_stream) * 2 ** 11))

        # if we expect each channel to have been recorded with the same parameters we will merge metadata into a
        # single dict with lists with entries like name that differ between channels.
        # Leaving this commented out, because channels may have different sampling rates
        for key in metadata.keys():
            if isinstance(metadata[key], list) \
                    and not isinstance(metadata[key][0], list) \
                    and len(set(metadata[key])) == 1:
                metadata[key] = metadata[key][0]

        return metadata

    @threaded_cached_property
    def dtype(self):
        return self.np_dtype

    @property
    def ndim(self):
        return 2

    def __getitem__(self, items):
        return self.data[items]
