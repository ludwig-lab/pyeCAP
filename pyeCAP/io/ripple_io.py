# standard imports
import os
from itertools import compress
import warnings

# scientific computing library imports
import numpy as np

# to lock file for thread safe reading
import threading
from cached_property import threaded_cached_property # This can be removed in python 3.8 as they are adding cached properties as a built-in decorator


# TODO: set this up to store event data and metadata after loading the first time.
class RippleIO:
    """ """

    def __init__(self, file_path):
        self.file_path = file_path
        self.lock = threading.Lock()

        # import statements from th pyneuroshare library from Ripple that mush be obtained directly from them.
        try:
            from pyns.nsfile import NSFile
            from pyns.nsentity import EntityType
        except ModuleNotFoundError:
            warnings.warn("No Neuroshare package found, Ripple files will not work")

        if isinstance(self.file_path, str):
            if self.file_path.endswith('.nev'):
                try:
                    self.nsfile = NSFile(file_path)
                except IOError:
                    print(self.file_path + " is not accessible.")
                except NameError:
                    warnings.warn("Ripple files not yet implemented")
            else:
                _, file_extension = os.path.splitext(self.file_path)
                raise IOError("Expected '.nev' file as the input. Received a '" + file_extension + "' file.")
        else:
            raise IOError("Expected path formatted as a string.")


class RippleEvents:
    """ Return timing of stimulation events"""

    def __init__(self, ripple_io, type='stim'):
        if isinstance(ripple_io, RippleIO):
            self.ripple_io = ripple_io
            self.type = type
        else:
            raise TypeError("input expected to by of type RippleIO")

    @property
    def metadata(self):
        channels = []
        if self.type == 'stim':
            # get the appropriate information for each channel
            for i, entity in enumerate(self.ripple_io.nsfile.get_entities(EntityType.segment)):
                with self.ripple_io.lock:
                    info_raw = entity.get_entity_info()._asdict()
                    segment_raw = entity.get_segment_info()._asdict()
                    source_raw = entity.get_seg_source_info()._asdict()
                    # header_raw = entity.get_extended_headers()

                info = dict(info_raw)
                segment = dict(segment_raw)
                source = dict(source_raw)
                # header = dict(header_raw)

                dict_info = {"info_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in info.items()}
                dict_segment = {"segment_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in
                                segment.items()}
                dict_source = {"source_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in
                               source.items()}
                # dict_header = {"header_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in
                #                header.items()}
                merged_dict = {**dict_info, **dict_source, **dict_segment}
                channels.append(merged_dict)
            # since we expect each channel to have been recorded with the same parameters we will merge metadata into a
            # single dict with lists with entries like name that differ between channels

            metadata = {}
            for key in channels[0].keys():
                metadata[key] = [channel[key] for channel in channels]
                if len(set(metadata[key])) == 1:
                    metadata[key] = metadata[key][0]
            # For stim channels we will also remove excess entries with no stimulation applied
            stim_bool = [True if count > 0 else False for count in metadata['info_item_count']]
            for key in metadata.keys():
                if isinstance(metadata[key], list):
                    # remove entries with no item count (no stimulations applied)
                    metadata[key] = list(compress(metadata[key], stim_bool))
            # Add information (boolean list) on what entities contain data
            metadata['source_channels'] = [i for i, channel in enumerate(stim_bool, start=1) if channel]

            # set up expected metadata for ts_data class
            metadata['ch_names'] = metadata['info_label']
        else:
            raise ValueError("Value of 'type' is not recognized")
        return metadata

    def keys(self):
        return self.metadata['info_label']

    def event_data(self, channel, index=None):
        entities = list(self.ripple_io.nsfile.get_entities(EntityType.segment))
        if isinstance(channel, int) and channel in self.metadata['source_channels']:
            channel = self.metadata['info_label'][self.metadata['source_channels'].index(channel)]
        elif isinstance(channel, str) and channel in self.metadata['info_label']:
            pass
        else:
            raise ValueError("channel is either not of type 'str' or 'int' or was not found")

        for entity in entities:
            entity_info = entity.get_entity_info()._asdict()
            if isinstance(channel, str) and entity_info['label'] == channel:
                if index is None:
                    return np.asarray([entity.get_segment_data(idx)[0] for idx in range(entity_info['item_count'])])
                elif isinstance(index, int):
                    return entity.get_segment_data(index)
                else:
                    raise TypeError("index expected to be  'int', or None")

    def __getitem__(self, item):
        if isinstance(item, (str, int)):
            return self.event_data(item)
        else:
            raise TypeError("RippleEvent class can only be indexed using 'str' or 'int' types")


class RippleArray:
    """ """

    def __init__(self, ripple_io, type='ephys'):
        if isinstance(ripple_io, RippleIO):
            self.ripple_io = ripple_io
            self.type = type
            self.shape = (len(self.metadata['info_label']), self.metadata['info_item_count'])
        else:
            raise TypeError("input expected to by of type RippleIO")

    @threaded_cached_property
    def metadata(self):
        channels = []
        if self.type == 'ephys':  # TODO: Refactor this to eliminate duplication of concepts for loading metadata
            # between types
            # get the appropriate information for each channel
            for i, entity in enumerate(self.ripple_io.nsfile.get_entities(EntityType.analog)):
                with self.ripple_io.lock:
                    info_raw = entity.get_entity_info()._asdict()
                    analog_raw = entity.get_analog_info()._asdict()
                    header_raw = entity.get_extended_header()._asdict()

                info = dict(info_raw)
                analog = dict(analog_raw)
                header = dict(header_raw)

                dict_info = {"info_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in info.items()}
                dict_analog = {"analog_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in
                               analog.items()}
                dict_header = {"header_" + k: (v.split('\0')[0] if isinstance(v, str) else v) for k, v in
                               header.items()}
                merged_dict = {**dict_info, **dict_analog, **dict_header}
                channels.append(merged_dict)

            # since we expect each channel to have been recorded with the same parameters we will merge metadata into a
            # single dict with lists with entries like name that differ between channels
            metadata = {}
            for key in channels[0].keys():
                metadata[key] = [channel[key] for channel in channels]
                if len(set(metadata[key])) == 1:
                    metadata[key] = metadata[key][0]

            # set up expected metadata for ts_data class
            metadata['ch_names'] = metadata['header_electrode_label']
            metadata['sample_rate'] = metadata['analog_sample_rate']

        else:
            raise ValueError("Value of 'type' is not recognized")
        return metadata

    @threaded_cached_property
    def dtype(self):
        return self[0, 0].dtype

    @property
    def ndim(self):
        return 2

    def __getitem__(self, items):
        if self.type == 'ephys':
            # This is potentially slower than I would like because of the overhead of recreating entity types every time
            # they are needed. This could be easily resolved by modifying the base pyns class to return direct access to
            # the parser as opposed to forcing access through individual entities.
            # TODO: test speed of lots of small reads with this method to see if overhead of creating entities is
            #  significant
            # TODO: write test cases for proper indexing of arrays from ripple files
            entities = list(self.ripple_io.nsfile.get_entities(EntityType.analog))
            if isinstance(items, tuple):
                if len(items) == 0:
                    return self[:]
                elif len(items) == 1:
                    return self[items[0]]
                elif len(items) == 2:
                    if isinstance(items[1], (list, int, np.ndarray)):
                        item_list = np.array([items[1]]) if isinstance(items[1], int) else np.array(items[1])
                        if issubclass(item_list.dtype.type, np.integer):
                            start_index = np.min(item_list)
                            stop_index = np.max(item_list) + 1
                            data_slice = item_list - start_index
                    elif isinstance(items[1], slice):
                        if items[1].start is None and items[1].stop is None and items[1].step is None:
                            return self[items[0]]
                        else:
                            # TODO: handle negative start and stop indices
                            start_index = 0 if items[1].start is None else items[1].start
                            stop_index = self.shape[1] if items[1].stop is None else items[1].stop
                            data_slice = slice(None, None, items[1].step)
                    else:
                        raise IndexError('only integers, slices (`:`), and integer arrays are valid indices')

                    if isinstance(items[0], (list, int, np.ndarray)):
                        item_list = np.array([items[0]]) if isinstance(items[0], int) else np.array(items[0])
                        if issubclass(item_list.dtype.type, np.integer):
                            data_list = []
                            for i in item_list:
                                with self.ripple_io.lock:
                                    data = entities[i].get_analog_data(start_index, stop_index - start_index,
                                                                       use_scale=True)
                                data_list.append(data)
                            data_array = np.array(data_list)
                        else:
                            raise IndexError('only integers, slices (`:`), and integer arrays are valid indices')
                    elif isinstance(items[0], slice):
                        data_list = []
                        for entity in entities[items[0]]:
                            with self.ripple_io.lock:
                                data_list.append(
                                    entity.get_analog_data(start_index, stop_index - start_index, use_scale=True))
                        data_array = np.array(data_list)
                    else:
                        raise IndexError('only integers, slices (`:`), and integer arrays are valid indices')
                    return data_array[:, data_slice]

                else:
                    raise IndexError("too many indices for array")
            elif isinstance(items, (list, int, np.ndarray)):
                item_list = np.array([items]) if isinstance(items, int) else np.array(items)
                if issubclass(item_list.dtype.type, np.integer):
                    data_list = []
                    for i in item_list:
                        with self.ripple_io.lock:
                            data_list.append(entities[i].get_analog_data(0, self.shape[1], use_scale=True))
                    return np.array(data_list)
                else:
                    raise IndexError('only integers, slices (`:`), and integer arrays are valid indices')
            elif isinstance(items, slice):
                with self.ripple_io.lock:
                    data_list = [entity.get_analog_data(0, self.shape[1], use_scale=True) for entity in entities[items]]
                return np.array(data_list)
            else:
                raise IndexError('only integers, slices (`:`), and integer arrays are valid indices')
        else:
            raise ValueError("Value of 'type' is not recognized")
