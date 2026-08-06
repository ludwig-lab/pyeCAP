from tdt import read_block
import os
from pathlib import Path
import glob  # for file and directory handling
import io
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import dask
import dask.array as da
from dask.cache import Cache
import itertools
import threading  # to lock file for thread safe reading
import warnings
from cached_property import threaded_cached_property
import mmap
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_sample_delay_table():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base", "TDTSampleDelay.csv")
    df = pd.read_csv(path)
    df.set_index('Unnamed: 0', inplace=True)
    return df

def gather_sample_delay(rz_sample_rate, si_sample_rate):
    df = _load_sample_delay_table()
    rz = str(rz_sample_rate)
    if rz not in df.columns:
        raise ValueError("RZ Sample rate invalid")
    if si_sample_rate not in df.index:
        raise ValueError("SI Sample rate invalid")
    val = df.loc[si_sample_rate, rz]
    try:
        return int(val)
    except Exception:
        raise ValueError("RZ or SI Sample rate not recognized")


class TdtIO:
    def __init__(self, file_path: str | os.PathLike[str]):
        """
        Load the headers from a TDT tank directory.

        Parameters
        ----------
        file_path : str or os.PathLike
            Path to the TDT tank directory.
        """

        if not isinstance(file_path, (str, os.PathLike)):
            raise TypeError(
                "file_path must be a string or pathlib.Path-like object, "
                f"not {type(file_path).__name__}."
            )

        self.file_path = Path(file_path).expanduser()
        self.lock = threading.Lock()

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"TDT tank path does not exist: {self.file_path}"
            )

        if not self.file_path.is_dir():
            raise NotADirectoryError(
                f"TDT tank path is not a directory: {self.file_path}"
            )


        self.tdt_block = None
        try:
            with io.StringIO() as buf, redirect_stdout(buf):
                # headers=1 avoids loading waveform samples while retaining TEV
                # offsets required by TdtArray.
                self.tdt_block = read_block(str(self.file_path), headers=1)
        except OSError as exc:
            raise OSError(
                f"TDT tank is not accessible: {self.file_path}"
            ) from exc

        if self.tdt_block is None:
            raise OSError(
                "File path does not appear to be a TDT tank: "
                f"{self.file_path}"
            )


    @threaded_cached_property  # We can cache this because none of this metadata will change after being read in.
    def metadata(self):
        # read in StoresListing file to get more metadata
        metadata = {}
        gizmo_dict = {}
        obj_id = {}
        txt_path = os.path.join(self.file_path, "StoresListing.txt")
        txt = ""
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except FileNotFoundError:
            warnings.warn(
                "No StoresListing file found; pyeCAP will use default store-name heuristics.",
                stacklevel=2,
            )

        # parse StoresListing file
        # TODO: This parsing is not great it is hastily modified from a previous version to account for differences in
        #  the StoresListing.txt file between versions of TDT's software. It should be generalized and cleaned up to
        #  better accout for all possible variations
        for n, txtblock in enumerate(txt.split("\n\n")):
            # read in experiment metadata text block
            if n == 0 and "Experiment" in txtblock:
                for line in txtblock.splitlines():
                    key, separator, value = line.partition(":")
                    if separator:
                        metadata[key.strip()] = value.strip()
                metadata.pop("Time", None)
            # read in storage data from each tdtgizmo
            elif txtblock.startswith("Object ID") or txtblock.startswith("ObjectID"):
                store_ids = []
                for txt_line in txtblock.split("\n"):
                    if txt_line.startswith("Object ID") or txt_line.startswith("ObjectID"):
                        object_id = txt_line.split("-")[0].split(":")[1].strip()
                        gizmo_name = txt_line.split("-")[1].strip()
                    elif txt_line.startswith(" Store ID") or txt_line.startswith(" StoreID"):
                        store_ids.append(txt_line.split(":")[1].strip())
                gizmo_dict.update({store_id: gizmo_name for store_id in store_ids})
                obj_id.update({store_id: object_id for store_id in store_ids})

        metadata["Gizmo Name"] = gizmo_dict
        metadata["Gizmo ID"] = obj_id
        return metadata


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

        # check for stimulation stores
        stores = self.tdt_io.tdt_block.stores
        self.raw_stores = []
        self.parameter_stores = []

        # Case-insensitive mapping for store IDs
        store_map_lower_to_orig = {k.lower(): k for k in self.tdt_io.stores}

        def _has_store(store_id_str: str):
            return store_map_lower_to_orig.get(store_id_str.lower())

        gname = self.tdt_io.metadata.get("Gizmo Name", {})
        for key_txt, gizmo_name in gname.items():
            orig_key = _has_store(key_txt)
            if not orig_key:
                continue
            if "electrical stim" in gizmo_name.lower():
                kl = orig_key.lower()
                if kl.endswith("p"):
                    self.parameter_stores.append(orig_key)
                elif kl.endswith("r") or orig_key == "MonA":
                    self.raw_stores.append(orig_key)

        # Fallback: heuristic scan if mapping didn’t yield anything
        if not self.parameter_stores and not self.raw_stores:
            for k in self.tdt_io.stores:
                kl = k.lower()
                if kl.endswith("p"):
                    self.parameter_stores.append(k)
                elif kl.endswith("r") or k == "MonA":
                    self.raw_stores.append(k)

        if len(self.parameter_stores) == 0 and len(self.raw_stores) == 0:
            raise ValueError("No electrical stimulation detected")

        elif len(self.parameter_stores) == 0:
            warnings.warn("No electrical stimulation parameters detected")

        # get parameter data into recognizable format
        self.stim_parameters = {}
        for i, par in enumerate(self.parameter_stores):
            stim_data = []
            for onset in np.unique(stores[self.parameter_stores[i]].ts):
                ch_idx = stores[self.parameter_stores[i]].ts == onset
                ch_sorted = np.argsort(stores[self.parameter_stores[i]].chan[ch_idx])
                parameters = list(stores[self.parameter_stores[i]].data[ch_idx][ch_sorted])
                stim_data.append(parameters)
            self.stim_parameters[par] = np.array(stim_data)

        self.voices = {}
        for k in self.parameter_stores:
            self.voices[k] = []

    @staticmethod
    def _add_source_id(parameter_dataframe):
        def make_source_id(row):
            parts = [str(row["store"])]

            voice = row.get("voice")
            if pd.notna(voice) and voice != "":
                parts.append(str(voice))

            tdt_channel = row.get("tdt_channel")
            if pd.notna(tdt_channel):
                parts.append(f"ch{int(tdt_channel)}")

            return ":".join(parts)

        parameter_dataframe["source_id"] = parameter_dataframe.apply(
            make_source_id,
            axis=1,
        )

        return parameter_dataframe

    @threaded_cached_property
    def metadata(self):
        # Copy instead of modifying TdtIO metadata in place.
        metadata = dict(self.tdt_io.metadata)

        metadata.update(
            {
                "start_time": self.tdt_io.tdt_block["start_time"][0],
                "stop_time": self.tdt_io.tdt_block["stop_time"][0],
                "raw_stores": list(self.raw_stores),
                "parameter_stores": list(self.parameter_stores),
            }
        )

        stores = self.tdt_io.tdt_block.stores

        metadata["num_stimulations"] = {
            store: stores[store].size
            for store in self.parameter_stores
        }

        metadata["stimulation_onsets"] = {
            store: np.sort(np.unique(stores[store].ts))
            for store in self.parameter_stores
        }

        sources = (
            self.parameters["source_id"]
            .drop_duplicates()
            .tolist()
        )

        metadata["stim_sources"] = sources
        metadata["ch_names"] = [
            f"Stim {source}"
            for source in sources
        ]

        return metadata

    @threaded_cached_property
    def parameters(self):
        stim_stores = []

        for k in self.parameter_stores:
            stim_parameters = self.stim_parameters[k]
            onsets = np.reshape(np.unique(self.tdt_io.tdt_block.stores[k].ts), (-1, 1))
            zero_cols = np.all(stim_parameters == 0.0, axis=0)
            if zero_cols[0]:
                # 'Electrical Stimulation' Gizmo was used
                col_names = ['onset time (s)', 'stimulation gain', 'pulse count', 'period (ms)',
                             'pulse amplitude A (μA)', 'pulse duration A (ms)']
                channels = np.ones((stim_parameters.shape[0], 1))
                self.voices[k].append("")
                if np.all(zero_cols[6:10]):
                    # A segment only
                    stim_data = np.concatenate((onsets, stim_parameters[:, 1:6]), axis=1)
                elif np.all(zero_cols[8:10]):
                    col_names += ['pulse amplitude B (μA)', 'pulse duration B (ms)']
                    stim_data = np.concatenate((onsets, stim_parameters[:, 1:8]), axis=1)
                else:
                    # A, B, and C segments
                    col_names += ['pulse amplitude B (μA)', 'pulse duration B (ms)', 'pulse amplitude C (μA)',
                                  'pulse duration C (ms)']
                    stim_data = np.concatenate((onsets, stim_parameters[:, 1:10]), axis=1)
            else:
                # 'Electrical Stim Driver was used'
                # TODO: Code for electrical stim driver
                col_names = ['onset time (s)', 'period (ms)', 'pulse count', 'pulse amplitude (μA)',
                             'pulse duration (ms)', 'delay (ms)', 'tdt_channel']
                voice_data = ["A"] * onsets.shape[0]
                stim_data = np.concatenate((onsets, stim_parameters[:, 0:6]), axis=1)
                if np.all(zero_cols[6:11]) and not (zero_cols[11]):
                    # Bipolar up to 2 voices
                    stim_data = np.concatenate((stim_data, stim_parameters[:, 11:12]), axis=1)
                    col_names += ["tdt_bipolar_channel"]
                else:
                    # Monopolar up to 4 voices
                    possible_voices = ["A", "B", "C", "D"]
                    polarity = "Monopolar"


            # add in new calculated columns for each voice
            parameter_dataframe = pd.DataFrame(stim_data, columns=col_names)

            if not zero_cols[0]:
                parameter_dataframe['voice'] = voice_data

            parameter_dataframe['store'] = k

            parameter_dataframe["pulse count"] = (
                parameter_dataframe["pulse count"].astype(int)
            )

            if "tdt_channel" in parameter_dataframe.columns:
                parameter_dataframe["tdt_channel"] = (
                    parameter_dataframe["tdt_channel"].astype(int)
                )

            parameter_dataframe.insert(2, "frequency (Hz)", 1000 / parameter_dataframe['period (ms)'])
            parameter_dataframe.insert(5, "duration (ms)", parameter_dataframe['period (ms)'] *
                                       parameter_dataframe['pulse count'])
            parameter_dataframe.insert(1, "offset time (s)", parameter_dataframe['onset time (s)'] +
                                       parameter_dataframe['duration (ms)'] / 1000)

            parameter_dataframe = self._add_source_id(parameter_dataframe)

            stim_stores.append(parameter_dataframe)

        return pd.concat(stim_stores).reset_index(drop=True)

    def _source_groups(self):
        params = self.parameters

        for source_id, source_params in params.groupby(
                "source_id",
                sort=False,
        ):
            name = f"Stim {source_id}"
            yield source_id, name, source_params

    def dio(self, indicators=False):
        dio = {}

        for source_id, name, source_params in self._source_groups():
            if indicators:
                dio_data = np.repeat(
                    source_params.index.to_numpy(),
                    2,
                )
            else:
                dio_data = np.empty(
                    len(source_params) * 2,
                    dtype=float,
                )

                dio_data[0::2] = source_params[
                    "onset time (s)"
                ].to_numpy()

                dio_data[1::2] = source_params[
                    "offset time (s)"
                ].to_numpy()

            dio[name] = dio_data

        return dio

    def events(self, indicators=False):
        events = {}

        for source_id, name, source_params in self._source_groups():
            source_events = []

            for index, row in source_params.iterrows():
                pulse_count = int(row["pulse count"])

                if indicators:
                    values = np.full(
                        pulse_count,
                        index,
                        dtype=int,
                    )
                else:
                    delay_ms = row.get("delay (ms)", 0.0)

                    if pd.isna(delay_ms):
                        delay_ms = 0.0

                    values = (
                            row["onset time (s)"]
                            + np.arange(pulse_count)
                            * row["period (ms)"]
                            / 1000
                            + delay_ms
                            / 1000
                    )

                source_events.append(values)

            if source_events:
                events[name] = np.concatenate(source_events)
            else:
                dtype = int if indicators else float
                events[name] = np.array([], dtype=dtype)

        return events


def detect_stim_stores(all_store_ids, gizmo_name_map):
    """
    Decide which stores are stimulation parameter stores and raw/monitor stores.

    Parameters
    ----------
    all_store_ids : list[str]
        e.g., ['251p','251r','RawE','RawG',...]
    gizmo_name_map : dict[str, str]
        Mapping StoreID -> Gizmo name (from StoresListing.txt parsing), e.g.
        {'251p': 'Electrical Stim Driver', '251r': 'Electrical Stim Driver', ...}

    Returns
    -------
    (parameter_stores, raw_stores) : (list[str], list[str])
    """
    # Case-insensitive matching
    lower_to_orig = {k.lower(): k for k in all_store_ids}

    param, raw = [], []

    # Pass 1: use gizmo map if available
    for store_id, gizmo in (gizmo_name_map or {}).items():
        k = lower_to_orig.get(store_id.lower())
        if not k:
            continue
        if "electrical stim" in (gizmo or "").lower():
            kl = k.lower()
            if kl.endswith("p"):
                param.append(k)
            elif kl.endswith("r") or k == "MonA":
                raw.append(k)

    # Pass 2: fallback heuristic if pass 1 found nothing
    if not param and not raw:
        for k in all_store_ids:
            kl = k.lower()
            if kl.endswith("p"):
                param.append(k)
            elif kl.endswith("r") or k == "MonA":
                raw.append(k)

    return param, raw


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



def _read_tdt_multichannel_chunk(tev_path, channel_sources, block_size, dtype):
    """Read one time chunk for all channels into a single 2D NumPy array.

    Parameters
    ----------
    tev_path : str
        Path to the shared TEV file.
    channel_sources : sequence
        Each entry is either ``("tev", offsets)`` or
        ``("sev", path, sample_start, sample_count)``.
    block_size : int
        Number of samples in each TEV block after its header.
    dtype : numpy dtype specifier
        Output/sample dtype.
    """
    dtype = np.dtype(dtype)
    block_size = int(block_size)
    sources = list(channel_sources)
    if not sources:
        return np.empty((0, 0), dtype=dtype)

    def source_sample_count(source):
        if source[0] == "tev":
            return int(len(source[1]) * block_size)
        if source[0] == "sev":
            return int(source[3])
        raise ValueError(f"Unknown TDT channel source type: {source[0]!r}")

    sample_counts = [source_sample_count(source) for source in sources]
    if len(set(sample_counts)) != 1:
        raise ValueError(
            "All channels in a TDT read task must contain the same number of samples."
        )

    n_samples = sample_counts[0]
    output = np.empty((len(sources), n_samples), dtype=dtype)
    block_bytes = int(block_size * dtype.itemsize)

    # SEV channels are contiguous files and can be read directly.
    for channel_index, source in enumerate(sources):
        if source[0] != "sev":
            continue
        _, path, sample_start, sample_count = source
        byte_start = int(sample_start) * dtype.itemsize
        byte_count = int(sample_count) * dtype.itemsize
        with open(path, "rb", buffering=0) as file:
            file.seek(byte_start)
            buffer = file.read(byte_count)
        if len(buffer) != byte_count:
            raise IOError(
                f"Short SEV read from {path!r}: expected {byte_count} bytes, "
                f"received {len(buffer)}."
            )
        output[channel_index, :] = np.frombuffer(
            buffer,
            dtype=dtype,
            count=int(sample_count),
        )

    tev_channels = [
        (channel_index, np.asarray(source[1], dtype=np.int64))
        for channel_index, source in enumerate(sources)
        if source[0] == "tev"
    ]
    if not tev_channels:
        return output

    binary_flag = getattr(os, "O_BINARY", 0)

    # POSIX: pread avoids shared file-position state and does not map the full TEV.
    if hasattr(os, "pread"):
        descriptor = os.open(tev_path, os.O_RDONLY | binary_flag)
        try:
            blocks_in_chunk = len(tev_channels[0][1])
            for block_index in range(blocks_in_chunk):
                start = block_index * block_size
                for channel_index, offsets in tev_channels:
                    offset = int(offsets[block_index])
                    buffer = os.pread(descriptor, block_bytes, offset)
                    if len(buffer) != block_bytes:
                        raise IOError(
                            f"Short TEV read at byte offset {offset}: expected "
                            f"{block_bytes} bytes, received {len(buffer)}."
                        )
                    output[channel_index, start:start + block_size] = np.frombuffer(
                        buffer,
                        dtype=dtype,
                        count=block_size,
                    )
            return output
        finally:
            os.close(descriptor)

    # Windows: map the TEV once for the entire multichannel task rather than once
    # per channel. Assignment copies each view into the output before the map closes.
    descriptor = os.open(tev_path, os.O_RDONLY | binary_flag)
    try:
        with mmap.mmap(descriptor, length=0, access=mmap.ACCESS_READ) as mapped:
            mapped_size = len(mapped)
            blocks_in_chunk = len(tev_channels[0][1])
            for block_index in range(blocks_in_chunk):
                start = block_index * block_size
                for channel_index, offsets in tev_channels:
                    offset = int(offsets[block_index])
                    if offset < 0 or offset + block_bytes > mapped_size:
                        raise IOError(
                            f"TEV block at byte offset {offset} falls outside file bounds."
                        )
                    output[channel_index, start:start + block_size] = np.frombuffer(
                        mapped,
                        dtype=dtype,
                        count=block_size,
                        offset=offset,
                    )
    finally:
        os.close(descriptor)

    return output


class TdtArray:
    """Lazy channel-by-time access to selected TDT stream stores.

    The TEV graph is constructed as one delayed two-dimensional read per time
    chunk. Each read opens/maps the TEV file once and fills all selected
    channels, rather than constructing a separate read graph for every channel.
    """

    def __init__(self, tdt_io, type="ephys", stores=None, chunk_size=1_000_000):
        if not isinstance(tdt_io, TdtIO):
            raise TypeError("input expected to be of type TdtIO")
        if type not in ("ephys", "stim"):
            raise AttributeError("Unexpected event type attribute received")
        if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer number of samples.")

        self.tdt_io = tdt_io
        self.type = type

        available_stores = self.tdt_io.stores
        if stores is None:
            self.stores = available_stores
        else:
            if isinstance(stores, str):
                requested = [stores]
            else:
                requested = list(stores)
            self.stores = [store for store in requested if store in available_stores]
            missing = [store for store in requested if store not in available_stores]
            if missing:
                warnings.warn(
                    "Ignoring TDT stores that were not found: " + ", ".join(map(str, missing)),
                    stacklevel=2,
                )

        stores_dict = self.tdt_io.tdt_block.stores
        stream_keys = [
            key
            for key in stores_dict.keys()
            if key in self.stores
            and stores_dict[key]["type_str"] == "streams"
            and key[-1] != "r"
            and (key[:-1] + "p") not in stores_dict.keys()
        ]
        if not stream_keys:
            raise ValueError("No compatible TDT stream stores were selected.")

        sample_rates = [float(stores_dict[key]["fs"]) for key in stream_keys]
        if len(set(sample_rates)) != 1:
            options = "\n".join(
                f"{key} [sampling rate = {rate} Hz]"
                for key, rate in zip(stream_keys, sample_rates)
            )
            raise IOError(
                "Selected stores have different sampling rates. Select a compatible "
                "subset with stores=.\nAvailable stores:\n" + options
            )

        raw_block_sizes = [int(stores_dict[key]["size"]) for key in stream_keys]
        if len(set(raw_block_sizes)) != 1:
            options = "\n".join(
                f"{key} [block size = {size}]"
                for key, size in zip(stream_keys, raw_block_sizes)
            )
            raise IOError(
                "Selected stores have different TDT block sizes and cannot share one "
                "array.\nAvailable stores:\n" + options
            )

        self.block_size = int(raw_block_sizes[0] - 10)
        if self.block_size <= 0:
            raise ValueError(
                f"Invalid effective TDT block size: {self.block_size}."
            )
        self.np_dtype = np.dtype(np.float32)
        self.tev_file = self._find_single_tev(self.tdt_io.file_path)

        # Build channel specifications once, in the same stream/channel order used
        # by metadata. Boolean selection preserves the original chronological
        # header order and avoids a full stable sort of every store header table.
        channel_sources = []
        channel_names = []
        channels_by_stream = []
        block_counts = []

        tev_stem = os.path.splitext(self.tev_file)[0]
        for key in stream_keys:
            store = stores_dict[key]
            channel_ids = np.asarray(store.chan)
            offsets = np.asarray(store.data, dtype=np.int64)
            if channel_ids.shape != offsets.shape:
                raise ValueError(
                    f"Store {key!r} has mismatched channel and offset arrays."
                )

            unique_channels = np.unique(channel_ids)
            stream_names = []
            for channel in unique_channels:
                channel_offsets = offsets[channel_ids == channel]
                if channel_offsets.size == 0:
                    continue

                channel_number = int(channel)
                name = f"{key} {channel_number}"
                stream_names.append(name)
                channel_names.append(name)
                block_counts.append(int(channel_offsets.size))

                sev_file = f"{tev_stem}_{key}_Ch{channel_number}.sev"
                if os.path.isfile(sev_file):
                    channel_sources.append(("sev", sev_file))
                else:
                    channel_sources.append(("tev", channel_offsets))

            channels_by_stream.append(stream_names)

        if not channel_sources:
            raise ValueError("Selected TDT streams contain no readable channels.")

        min_blocks = min(block_counts)
        max_blocks = max(block_counts)
        if min_blocks != max_blocks:
            # Preserve the previous supported case: some channels/stores may have
            # exactly one trailing block more than the shortest recording. Trim
            # locally without mutating the TDT header object.
            if max_blocks - min_blocks == 1:
                warnings.warn(
                    "Channel lengths differ by one TDT block; trimming to the "
                    "shortest channel.",
                    stacklevel=2,
                )
                trimmed_sources = []
                for source in channel_sources:
                    if source[0] == "tev":
                        trimmed_sources.append(("tev", source[1][:min_blocks]))
                    else:
                        trimmed_sources.append(source)
                channel_sources = trimmed_sources
            else:
                details = ", ".join(
                    f"{name}={count} blocks"
                    for name, count in zip(channel_names, block_counts)
                )
                raise IOError(
                    "Selected TDT channels have different recording lengths: " + details
                )

        n_channels = len(channel_sources)
        n_blocks = min_blocks
        n_samples = int(n_blocks * self.block_size)

        # Populate/cache metadata from the validated channel plan. This avoids a
        # second pass with potentially different assumptions about channel count.
        metadata = dict(self.tdt_io.metadata)
        metadata.update(
            {
                "start_time": self.tdt_io.tdt_block["start_time"][0],
                "stop_time": self.tdt_io.tdt_block["stop_time"][0],
                "streams": list(stream_keys),
                "channels": channels_by_stream,
                "ch_names": channel_names,
                "channels_per_stream": [len(names) for names in channels_by_stream],
                "sample_rate": sample_rates[0],
                "block_size": raw_block_sizes[0],
                "stream_lengths": n_samples,
                "file_location": self.tdt_io.file_path,
            }
        )
        metadata["cumulative_channel_count"] = list(
            np.cumsum(metadata["channels_per_stream"], dtype=int)
        )
        self._metadata = metadata
        self.shape = (n_channels, n_samples)

        blocks_per_task = max(1, int(int(chunk_size) // self.block_size))
        self.chunk_size = int(blocks_per_task * self.block_size)

        array_chunks = []
        for block_start in range(0, n_blocks, blocks_per_task):
            block_stop = min(block_start + blocks_per_task, n_blocks)
            chunk_blocks = block_stop - block_start
            chunk_samples = int(chunk_blocks * self.block_size)
            sample_start = int(block_start * self.block_size)

            chunk_sources = []
            for source_type, payload in channel_sources:
                if source_type == "tev":
                    chunk_sources.append(
                        ("tev", np.asarray(payload[block_start:block_stop], dtype=np.int64))
                    )
                else:
                    chunk_sources.append(
                        ("sev", os.fspath(payload), sample_start, chunk_samples)
                    )

            delayed_chunk = dask.delayed(
                _read_tdt_multichannel_chunk,
                pure=False,
            )(
                os.fspath(self.tev_file),
                chunk_sources,
                self.block_size,
                self.np_dtype.str,
            )
            array_chunks.append(
                da.from_delayed(
                    delayed_chunk,
                    shape=(n_channels, chunk_samples),
                    dtype=self.np_dtype,
                )
            )

        self.data = (
            array_chunks[0]
            if len(array_chunks) == 1
            else da.concatenate(array_chunks, axis=1)
        )
        # The graph is born with one full channel chunk. No stack/rechunk layer is
        # needed, and `.chunks` is only a compatibility snapshot.
        self.chunks = self.data.chunks

    def _find_single_tev(self, directory: str | os.PathLike[str]) -> str:
        tev_path = None
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".tev"):
                    if tev_path is not None:
                        raise FileExistsError(
                            "Multiple '*.tev' files found in tank, 1 expected."
                        )
                    tev_path = entry.path

        if tev_path is None:
            raise FileNotFoundError(
                "Could not locate '*.tev' file expected for TDT tank."
            )
        return tev_path

    @dask.delayed
    def load_block(self, offset):
        return np.fromfile(
            self.tev_file,
            dtype=self.np_dtype,
            count=self.block_size,
            offset=int(offset),
        )

    @threaded_cached_property
    def metadata(self):
        # `_metadata` is constructed during initialization from the same validated
        # channel plan that constructs the Dask array.
        return self._metadata

    @threaded_cached_property
    def dtype(self):
        return self.np_dtype

    @property
    def ndim(self):
        return 2

    def __getitem__(self, items):
        return self.data[items]

