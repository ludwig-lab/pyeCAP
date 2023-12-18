# Standard library imports
import functools
import glob  # for file and directory handling
import io
import itertools
import mmap
import os
import sys
import threading  # to lock file for thread safe reading
import warnings
from collections import defaultdict
from contextlib import redirect_stdout
from typing import List, Optional

import dask
import dask.array as da
import numpy as np
import pandas as pd

# Third party imports
import tdt
from dask.cache import Cache
from tdt import epoc_filter, read_block

cache = Cache(0.5e9)  # Leverage 500mb of memory for fast access to data


def gather_sample_delay(rz_sample_rate, si_sample_rate):
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "base",
            "TDTSampleDelay.csv",
        )
    )
    df.set_index("Unnamed: 0", inplace=True)
    if str(rz_sample_rate) in df.columns:
        if si_sample_rate in df.index:
            try:
                return int(df.loc[si_sample_rate, str(rz_sample_rate)])
            except ValueError:
                print("RZ or SI Sample rate not recognized")
        else:
            raise ValueError("SI Sample rate invalid")
    else:
        raise ValueError("RZ Sample rate invalid")


class TdtIO:
    """
    This class is responsible for handling TDT data files.
    It reads the data, handles errors, and extracts metadata.
    """

    def __init__(self, file_path: str):
        """
        Initialize TdtIO with a file path.

        :param file_path: The path to the TDT data file.
        :type file_path: str
        :raises TypeError: If the file path is not a string.
        :raises FileNotFoundError: If the file path does not point to a directory.
        :raises IOError: If the file path is not accessible or does not appear to be a TDT tank.
        """
        self.file_path = file_path
        self.lock = threading.Lock()  # Lock for thread-safe reading

        # Check if the file path is a string and if it points to a directory
        if not isinstance(self.file_path, str):
            raise TypeError("Path was not formatted as a string.")
        if not os.path.isdir(self.file_path):
            raise FileNotFoundError("File path not found")

        # Redirect print output from read_block to buffer
        try:
            with io.StringIO() as buf, redirect_stdout(buf):
                # 'headers' var, set to 1 to return only the headers for the block.
                # This enables fast consecutive calls to read_block
                self.tdt_block = read_block(self.file_path, headers=1)
                output = buf.getvalue()
        except IOError as e:
            raise IOError(f"{self.file_path} is not accessible. {str(e)}")

        if self.tdt_block is None:
            raise IOError("File path does not appear to be a TDT tank.")

    @functools.cached_property
    def metadata(self) -> dict:
        """
        Extract metadata from the StoresListing file.
        This method is cached for efficiency as the metadata won't change after being read in.

        :return: A dictionary containing the metadata.
        :rtype: dict
        """
        metadata = {}
        gizmo_dict = {}
        obj_id = {}
        txt_path = os.path.join(self.file_path, "StoresListing.txt")
        try:
            with open(txt_path, "r") as f:
                txt = f.read()
        except FileNotFoundError:
            warnings.warn(
                "No StoresListing file found, pyeCAP will assume default store names"
            )

        # Parse StoresListing file
        # This parsing is not ideal and should be improved to better account for all possible variations
        for n, txtblock in enumerate(txt.split("\n\n")):
            # Read in experiment metadata text block
            if (
                n == 0 and "Experiment" in txtblock
            ):  # The first block of text should be the experiment information
                metadata.update(
                    {
                        line.split(":")[0]: line.split(":")[1].strip()
                        for line in txtblock.split("\n")
                    }
                )
                metadata.pop("Time")
            # Read in storage data from each tdtgizmo
            elif txtblock.startswith("Object ID") or txtblock.startswith("ObjectID"):
                store_ids = []
                for txt_line in txtblock.split("\n"):
                    if txt_line.startswith("Object ID") or txt_line.startswith(
                        "ObjectID"
                    ):
                        object_id = txt_line.split("-")[0].split(":")[1].strip()
                        gizmo_name = txt_line.split("-")[1].strip()
                    elif txt_line.startswith(" Store ID") or txt_line.startswith(
                        " StoreID"
                    ):
                        store_ids.append(txt_line.split(":")[1].strip())
                gizmo_dict.update({store_id: gizmo_name for store_id in store_ids})
                obj_id.update({store_id: object_id for store_id in store_ids})

        metadata["Gizmo Name"] = gizmo_dict
        metadata["Gizmo ID"] = obj_id
        return metadata

    @property
    def stores(self) -> list:
        """
        Return a list of keys from the tdt_block stores.

        :return: A list of keys from the tdt_block stores.
        :rtype: list
        """
        return list(self.tdt_block.stores.keys())


class TdtStim:
    """
    Class for handling TDT stimulation data.
    """

    def __init__(self, tdt_io: TdtIO, type: str = "stim"):
        """
        Initialize TdtStim object.

        :param tdt_io: TdtIO object
        :type tdt_io: TdtIO
        :param type: Type of stimulation, defaults to "stim"
        :type type: str, optional
        """
        if not isinstance(tdt_io, TdtIO):
            raise TypeError("Input expected to be of type TdtIO")
        self.tdt_io = tdt_io
        self.type = type

        # check for stimulation stores
        stores = self.tdt_io.tdt_block.stores
        self.raw_stores = []
        self.parameter_stores = []

        for key, value in self.tdt_io.metadata["Gizmo Name"].items():
            if (
                key in self.tdt_io.stores or "_" + key in self.tdt_io.stores
            ) and value in (
                "Electrical Stim Driver",
                "Electrical Stimulation",
            ):
                # TODO: Use the data types instead of assumptions about defualt naming conventions to check if the keys are stim parameter or stream data.
                # TODO: Fallback to searching based on default names and giving warning.
                if key in self.tdt_io.stores:
                    if "p" in key:
                        self.parameter_stores.append(key)
                    elif "r" in key:
                        self.raw_stores.append(key)
                    else:
                        warnings.warn(
                            "Parameter data stored by the electrical stim driver could not be found."
                        )
                if "_" + key in self.tdt_io.stores:
                    if "p" in key:
                        self.parameter_stores.append("_" + key)
                    elif "r" in key:
                        self.raw_stores.append("_" + key)
                    else:
                        warnings.warn(
                            "Parameter data stored by the electrical stim driver could not be found."
                        )
            elif (
                key == "MonA"
            ):  # TODO: check if any other monitoring channels exist and for new data use the IV10 or other stimulator to check if monitoring data is there
                self.raw_stores.append(key)

        # raise errors/warnings for certain data
        if not self.parameter_stores and not self.raw_stores:
            raise ValueError("No electrical stimulation detected")
        elif not self.parameter_stores:
            warnings.warn("No electrical stimulation parameters detected")

        # get parameter data into recognizable format
        self.stim_parameters = {}
        for i, par in enumerate(self.parameter_stores):
            stim_data = []
            for onset in np.unique(stores[self.parameter_stores[i]].ts):
                ch_idx = stores[self.parameter_stores[i]].ts == onset
                ch_sorted = np.argsort(stores[self.parameter_stores[i]].chan[ch_idx])
                parameters = list(
                    stores[self.parameter_stores[i]].data[ch_idx][ch_sorted]
                )
                stim_data.append(parameters)
            self.stim_parameters[par] = np.array(stim_data)

        self.voices = {}
        for k in self.parameter_stores:
            self.voices[k] = []

    @functools.cached_property
    def metadata(self) -> dict:
        """
        Get metadata for the TdtStim object.

        :return: Metadata dictionary
        :rtype: dict
        """
        metadata = self.tdt_io.metadata
        metadata.update(
            {
                "start_time": self.tdt_io.tdt_block["start_time"][0],
                "stop_time": self.tdt_io.tdt_block["stop_time"][0],
            }
        )

        stores = self.tdt_io.tdt_block.stores
        metadata["raw_stores"] = self.raw_stores
        metadata["parameter_stores"] = self.parameter_stores

        metadata["num_stimulations"] = {}
        metadata["stimulation_onsets"] = {}
        for k in self.parameter_stores:
            metadata["num_stimulations"][k] = stores[k].size
            metadata["stimulation_onsets"][k] = np.sort(np.unique(stores[k].ts))

        metadata["channels"] = list(np.unique(self.parameters["channel"]))
        metadata["ch_names"] = ["Stim " + str(ch) for ch in metadata["channels"]]
        return metadata

    @functools.cached_property
    def parameters(self) -> pd.DataFrame:
        """
        Get stimulation parameters.

        :return: DataFrame of stimulation parameters
        :rtype: pd.DataFrame
        """
        stim_stores = []

        for k in self.parameter_stores:
            stim_parameters = self.stim_parameters[k]
            onsets = np.reshape(np.unique(self.tdt_io.tdt_block.stores[k].ts), (-1, 1))
            zero_cols = np.all(stim_parameters == 0.0, axis=0)
            if zero_cols[0]:
                # 'Electrical Stimulation' Gizmo was used
                col_names = [
                    "onset time (s)",
                    "channel",
                    "stimulation gain",
                    "pulse count",
                    "period (ms)",
                    "pulse amplitude A (μA)",
                    "pulse duration A (ms)",
                ]
                channels = np.ones((stim_parameters.shape[0], 1))
                self.voices[k].append("")
                if np.all(zero_cols[6:10]):
                    # A segment only
                    stim_data = np.concatenate(
                        (onsets, channels, stim_parameters[:, 1:6]), axis=1
                    )
                elif np.all(zero_cols[8:10]):
                    col_names += ["pulse amplitude B (μA)", "pulse duration B (ms)"]
                    stim_data = np.concatenate(
                        (onsets, channels, stim_parameters[:, 1:8]), axis=1
                    )
                else:
                    # A, B, and C segments
                    col_names += [
                        "pulse amplitude B (μA)",
                        "pulse duration B (ms)",
                        "pulse amplitude C (μA)",
                        "pulse duration C (ms)",
                    ]
                    stim_data = np.concatenate(
                        (onsets, channels, stim_parameters[:, 1:10]), axis=1
                    )
            else:
                # 'Electrical Stim Driver was used'
                col_names = [
                    "onset time (s)",
                    "period (ms)",
                    "pulse count",
                    "pulse amplitude (μA)",
                    "pulse duration (ms)",
                    "delay (ms)",
                    "channel",
                ]
                voice_data = ["A"] * onsets.shape[0]
                stim_data = np.concatenate((onsets, stim_parameters[:, 0:6]), axis=1)
                if np.all(zero_cols[6:11]) and not (zero_cols[11]):
                    # Bipolar up to 2 voices
                    possible_voices = ["A", "C"]
                    stim_data = np.concatenate(
                        (stim_data, stim_parameters[:, 11:12]), axis=1
                    )
                    polarity = "Bipolar"
                    col_names += ["bipolar channel"]
                else:
                    # Monopolar up to 4 voices
                    possible_voices = ["A", "B", "C", "D"]
                    polarity = "Monopolar"

                # TODO: Figure out what is going on here and finish reading in all possible data from TDT stim driver.
                # # Add extra data if necessary
                # for i, voice in enumerate(possible_voices[1:]):
                #     if not np.all(zero_cols[(i + 1) * 24 // len(possible_voices):(i + 2) * 24 // len(possible_voices)]):
                #         self.voices[k].append(voice)
                #         voice_data += [voice] * onsets.shape[0]
                #         if polarity == "bipolar":
                #             new_data = np.concatenate((onsets, stim_parameters[:, (i + 1) * 12:(i + 1) * 12 + 6],
                #                                        stim_parameters[:, (i + 2) * 12 - 1:(i + 2) * 12]), axis=1)
                #         else:
                #             new_data = np.concatenate((onsets, stim_parameters[:, (i + 1) * 6:(i + 2) * 6]), axis=1)
                #         stim_data = np.concatenate((stim_data, new_data), axis=0)

            # add in new calculated columns for each voice
            parameter_dataframe = pd.DataFrame(stim_data, columns=col_names)
            if not zero_cols[0]:
                parameter_dataframe["voice"] = voice_data
            parameter_dataframe["store"] = k
            parameter_dataframe = parameter_dataframe.astype(
                {"pulse count": int, "channel": int}
            )
            parameter_dataframe.insert(
                2, "frequency (Hz)", 1000 / parameter_dataframe["period (ms)"]
            )
            parameter_dataframe.insert(
                5,
                "duration (ms)",
                parameter_dataframe["period (ms)"] * parameter_dataframe["pulse count"],
            )
            parameter_dataframe.insert(
                1,
                "offset time (s)",
                parameter_dataframe["onset time (s)"]
                + parameter_dataframe["duration (ms)"] / 1000,
            )
            stim_stores.append(parameter_dataframe)

        return pd.concat(stim_stores).reset_index(drop=True)

    def dio(self, indicators: bool = False) -> dict:
        """
        Get digital input/output data.

        :param indicators: If True, return indicators, defaults to False
        :type indicators: bool, optional
        :return: Dictionary of digital input/output data
        :rtype: dict
        """
        # defaultdict is used here to automatically create lists for keys that do not yet exist in the dictionary.
        # This is useful when we want to append items to lists for certain keys in a loop,
        # and we don't know in advance what the keys will be.
        dio = defaultdict(list)
        params = self.parameters
        for ch, name in zip(self.metadata["channels"], self.metadata["ch_names"]):
            ch_params = params[params["channel"] == ch]
            # Preallocate an array of zeros for the digital input/output data.
            # The size of the array is twice the length of the channel parameters,
            # because for each parameter, we will store both an onset and an offset.
            dio_data = np.zeros((len(ch_params) * 2,), dtype=float)
            if indicators:
                dio_data[:] = np.repeat(ch_params.index, 2)
            else:
                onsets = np.array(ch_params["onset time (s)"])
                offsets = np.array(ch_params["offset time (s)"])
                # Fill the preallocated array with the onset and offset times.
                # The onset times are stored at even indices, and the offset times are stored at odd indices.
                dio_data[0::2] = onsets
                dio_data[1::2] = offsets
            # Here, we append the dio_data to the list associated with the key 'name'.
            # If 'name' does not yet exist in the dictionary, defaultdict automatically creates a new list for it.
            dio[name].extend(dio_data)
        # Convert the defaultdict back to a regular dictionary before returning.
        return dict(dio)

    def events(self, indicators: bool = False) -> dict:
        """
        Get stimulation events. If indicators is True, the function will return an array of
        integers where each integer corresponds to the index of the stimulation within the
        the stimulation parameter DataFrame returned by the parameters method.

        :param indicators: If True, return indicators, defaults to False
        :type indicators: bool, optional
        :return: Dictionary of stimulation events
        :rtype: dict
        """
        events = {}
        params = self.parameters
        for ch, name in zip(self.metadata["channels"], self.metadata["ch_names"]):
            ch_params = params[params["channel"] == ch]
            total_events = ch_params["pulse count"].sum()
            # Preallocate an array of the required size for the stimulation events.
            # This is done for efficiency reasons, as it is faster to allocate memory for an array once,
            # rather than dynamically resizing the array every time we add an event.
            ch_events = np.empty(total_events, dtype=int if indicators else float)
            event_idx = 0
            if indicators:
                for index, row in ch_params.iterrows():
                    num_stim_events = int(row["pulse count"])
                    ch_events[event_idx : event_idx + num_stim_events] = index
                    event_idx += num_stim_events
            else:
                for index, row in ch_params.iterrows():
                    stim_events = (
                        row["onset time (s)"]
                        + np.arange(0, row["pulse count"]) * row["period (ms)"] / 1000
                    )
                    stim_events += row["delay (ms)"] / 1000
                    num_stim_events = len(stim_events)
                    ch_events[event_idx : event_idx + num_stim_events] = stim_events
                    event_idx += num_stim_events
            events[name] = ch_events
        return events


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
    def __init__(
        self,
        tdt_io: TdtIO,
        type: str = "ephys",
        stores: Optional[List] = None,
        chunk_size: int = 2040800,
    ):
        """
        Stores data in an array by channel and index.
        Streams are alphanumerically iterated through and stored accordingly in metadata['streams'].

        Args:
            tdt_io (TdtIO): An instance of TdtIO class.
            type (str, optional): Type of data to be pulled. Defaults to "ephys".
            stores (list, optional): List of stores to be used. Defaults to None.
            chunk_size (int, optional): Size of the chunk to be used. Defaults to 2040800.

        Raises:
            TypeError: If tdt_io is not an instance of TdtIO class.
            AttributeError: If type is not "ephys" or "stim".
            IOError: If selected stores contain arrays with different sampling rates or different numbers of samples.
            FileNotFoundError: If '*.tev' file expected for tdt tank is not found.
            FileExistsError: If multiple '*.tev' files are found in tank.
        """

        # Check and handle inputs
        # Validate tdt_io input
        if isinstance(tdt_io, TdtIO):
            self.tdt_io = tdt_io
        else:
            raise TypeError("input expected to be of type TdtIO")

        # Validate type input
        # TODO: Need to update array reading so that it uses ephys vs. stim to pull appropriate data
        if type in ["ephys", "stim"]:
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
        s_freqs = self.metadata["sample_rate"]
        s_lengths = self.metadata["stream_lengths"]
        if isinstance(s_freqs, list):
            store_options = ""
            for s, fs in zip(self.metadata["streams"], self.metadata["sample_rate"]):
                store_options = (
                    store_options + s + " [sampling rate = " + str(fs) + " Hz]\n"
                )
            raise IOError(
                "Selected stores contain arrays with different sampling rates. Use input 'stores=' to select "
                "a subset of TDT stores with compatible sampling rates.\n"
                "Available stores are: \n" + store_options
            )

        if isinstance(s_lengths, list):
            # ToDo: figure out why block side is 10 bit larger than stroed data
            if max(s_lengths) - min(s_lengths) == self.metadata["block_size"] - 10:
                warnings.warn(
                    "Block size mismatch by one block. Adjusting stream length to shortest length."
                )
                self.metadata["stream_lengths"] = min(s_lengths)
                for len_idx, lens in enumerate(s_lengths):
                    if lens > min(s_lengths):
                        current_stream = self.stores[len_idx]
                        # use numpy to delete last block within TdTStruct of data and channel (removes last block entirely)
                        # TdTStruct were not able to be modified in situ
                        num_channels = int(
                            max(self.tdt_io.tdt_block.stores[current_stream].chan)
                        )
                        self.tdt_io.tdt_block.stores[
                            current_stream
                        ].data = self.tdt_io.tdt_block.stores[current_stream].data[
                            :-num_channels
                        ]
                        self.tdt_io.tdt_block.stores[
                            current_stream
                        ].chan = self.tdt_io.tdt_block.stores[current_stream].chan[
                            :-num_channels
                        ]
                        dump = 1
            else:
                store_options = ""
                for s, sl in zip(
                    self.metadata["streams"], self.metadata["stream_lengths"]
                ):
                    store_options = (
                        store_options + s + " [length = " + str(sl) + " samples]\n"
                    )
                raise IOError(
                    "Selected stores contain arrays with different numbers of samples. Use input 'stores' to "
                    "select a subset of TDT stores with compatible sample numbers.\n"
                    "Available stores are: \n" + store_options
                )

        # TODO: need to handle tdt types other than floats properly
        # Create aray in dask format
        # Calculate array size
        self.shape = (len(self.metadata["ch_names"]), self.metadata["stream_lengths"])

        # Get tev file path to pull data from
        tev_files = glob.glob(
            self.tdt_io.file_path + "/*.tev"
        )  # There should only be one
        if len(tev_files) == 0:
            raise FileNotFoundError(
                "Could not located '*.tev' file expected for tdt tank."
            )
        elif len(tev_files) > 1:
            raise FileExistsError("Multiple '*.tev' files found in tank, 1 expected.")
        else:
            self.tev_file = tev_files[0]

            # create list of data locations
            data_arrays = (
                []
            )  # place to store lists of dask delayed object for each segment on disk
            stores = self.tdt_io.tdt_block.stores
            # Todo: yell at TDT about block_size
            self.block_size = int(self.metadata["block_size"] - 10)
            self.np_dtype = np.dtype("f")

            tev_file = self.tev_file
            np_dtype = self.np_dtype
            block_size = self.block_size
            block_bits = block_size * np_dtype.itemsize

            @dask.delayed
            def load_block(offsets):
                f = open(tev_file, "rb")
                mm = mmap.mmap(
                    f.fileno(), 0, access=mmap.ACCESS_READ
                )  # Create a memory-mapped file object
                n_offsets = np.copy(offsets)
                # n_offsets[1:] = np.diff(offsets) - block_bits
                block = [
                    np.frombuffer(mm, dtype=np_dtype, count=block_size, offset=offset)
                    for offset in n_offsets
                ]
                block = np.concatenate(block)
                return block

            n = int(chunk_size / block_size)
            for key in np.asarray(self.metadata["streams"]).flatten():
                channel_list = np.sort(
                    np.unique(self.tdt_io.tdt_block.stores[key].chan)
                )
                for channel in channel_list:
                    data_offsets = np.array(
                        self.tdt_io.tdt_block.stores[key].data[
                            self.tdt_io.tdt_block.stores[key].chan == channel
                        ],
                        dtype=np.int64,
                    )  # needs to by an int64 for datasets greater than >4GB
                    # Check for sev file that will exist if files saved seperately
                    sev_file = (
                        os.path.splitext(tev_file)[0]
                        + "_"
                        + key
                        + "_Ch"
                        + str(channel)
                        + ".sev"
                    )
                    if os.path.isfile(sev_file):
                        # if sev file exists then data is stored sequentially on disk and can be quickly memmaped (this should be faster)
                        f_offset = data_offsets[0]
                        # f = open(tev_file, 'r')
                        mm_array = np.memmap(
                            sev_file,
                            shape=len(data_offsets) * block_size,
                            dtype=np_dtype,
                            offset=f_offset,
                        )
                        ch_array = da.from_array(mm_array, chunks=(chunk_size,))
                        data_arrays.append(ch_array)
                    else:
                        # if not then we need to individually map each block
                        data_offsets = [
                            data_offsets[i : i + n]
                            for i in range(0, len(data_offsets), n)
                        ]
                        data_len = [len(offsets) for offsets in data_offsets]
                        data_delayed = [
                            (load_block(np.array(offsets)), l)
                            for offsets, l in zip(data_offsets, data_len)
                        ]
                        dask_arrays = [
                            da.from_delayed(
                                d, (self.block_size * l,), dtype=self.np_dtype
                            )
                            for d, l in data_delayed
                        ]
                        data_arrays.append(da.concatenate(dask_arrays))
            self.data = da.stack(data_arrays)
            self.chunks = (1, chunk_size)

    @dask.delayed
    def load_block(self, offset: int) -> np.ndarray:
        """
        Load a block of data from the file.

        Args:
            offset (int): The offset to start reading from.

        Returns:
            np.ndarray: The block of data read from the file.
        """
        return np.fromfile(
            self.tev_file, dtype=self.np_dtype, count=self.block_size, offset=offset
        )

    @functools.cached_property
    def metadata(self) -> dict:
        """
        Get the metadata of the TdtArray.

        Returns:
            dict: The metadata of the TdtArray.
        """
        metadata = self.tdt_io.metadata
        metadata.update(
            {
                "start_time": self.tdt_io.tdt_block["start_time"][0],
                "stop_time": self.tdt_io.tdt_block["stop_time"][0],
            }
        )

        stores = self.tdt_io.tdt_block.stores
        metadata["block_size"] = []
        metadata["streams"] = []
        metadata["stream_lengths"] = []
        metadata["sample_rate"] = []
        metadata["file_location"] = self.tdt_io.file_path
        for idx, k in enumerate(stores.keys()):
            # TODO: Deal with streams that have different sampling rates, lengths, onset times, data types or other
            #  differences that prevent them from being accessed in the same array

            # Stores have a 4-5 character string identifier. Stimulation data is by default recorded in a pair of
            # stores withe the same first 4 characters followed by a p or r. Store ending with 'p' is for stimulation
            # parameters and is 'scalar' data, 'r' is a raw recording of stimulation waveform and is a 'stream'
            if (
                k in self.stores
                and stores[k]["type_str"] == "streams"
                and (k[-1] != "r" and k[:-1] + "p" not in stores.keys())
            ):
                metadata["streams"].append(k)
                metadata["sample_rate"].append(stores[k]["fs"])
                metadata["block_size"].append(stores[k]["size"])

        # get number of channels within each stream
        metadata["channels"] = []
        for k in metadata["streams"]:
            # noinspection PyTypeChecker
            metadata["channels"].append(
                [k + " " + str(i + 1) for i in range(np.max(stores[k]["chan"]))]
            )
        metadata["ch_names"] = list(itertools.chain.from_iterable(metadata["channels"]))

        # Setup variables and code to determine enumeration through streams. That is a recording with 4 RawE (LIFE)
        # channels, 4 RawG (EMG) channels and 2 _. channels will have the shape [4, 4, 2]
        channel_array = metadata["channels"]
        metadata["channels_per_stream"] = [len(i) for i in channel_array]
        metadata["cumulative_channel_count"] = [metadata["channels_per_stream"][0]]
        for i in range(1, len(metadata["channels_per_stream"])):
            metadata["cumulative_channel_count"].append(
                metadata["channels_per_stream"][i]
                + metadata["cumulative_channel_count"][i - 1]
            )
        metadata["stream_lengths"] = []

        for idx, name in enumerate(metadata["streams"]):
            header_length = len(self.tdt_io.tdt_block.stores[name].data)
            # Case where waveform is present. It has two channels, but one is the anodic portion and the other is the
            # cathodic. So really, 1 portion
            if name.startswith("_"):
                channels_in_stream = 1
            else:
                channels_in_stream = metadata["channels_per_stream"][idx]
            metadata["stream_lengths"].append(
                int((header_length / channels_in_stream) * 2**11)
            )

        # if we expect each channel to have been recorded with the same parameters we will merge metadata into a
        # single dict with lists with entries like name that differ between channels.
        # Leaving this commented out, because channels may have different sampling rates
        for key in metadata.keys():
            if (
                isinstance(metadata[key], list)
                and not isinstance(metadata[key][0], list)
                and len(set(metadata[key])) == 1
            ):
                metadata[key] = metadata[key][0]

        return metadata

    @functools.cached_property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the TdtArray.

        Returns:
            np.dtype: The data type of the TdtArray.
        """
        return self.np_dtype

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the TdtArray.

        Returns:
            int: The number of dimensions of the TdtArray.
        """
        return 2

    def __getitem__(self, items):
        """
        Get the item(s) from the TdtArray using array-like indexing.

        Args:
            items: The index or slice to get from the TdtArray.

        Returns:
            The item(s) from the TdtArray.
        """
        return self.data[items]
