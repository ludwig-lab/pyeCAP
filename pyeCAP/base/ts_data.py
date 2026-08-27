# python standard library imports
import os.path
from datetime import datetime
import copy
from collections.abc import Mapping, Sequence

# scientific computing library imports
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
from scipy import signal, ndimage

# plotting and figure generation

# interactive plotting

# neuro base class imports
from .dio_data import _DioData
from .event_data import _EventData
from .utils.numeric import _to_numeric_array, largest_triangle_three_buckets, _group_consecutive
from .utils.visualization import _plt_setup_fig_axis, _plt_show_fig, _plt_ax_to_pix, _plt_add_ax_connected_top, \
    _plt_check_interactive

# from multiprocessing.pool import ThreadPool
# dask.config.set(scheduler='threads', pool=ThreadPool(8))
# cache = Cache(2e9)  # Leverage two gigabytes of memory
# cache.register()  # Turn cache on globally





class _TsData:
    """
    Class for time series data. Contains many of the methods for the pyCAP.Ephys child class.
    """

    def __init__(self, data, metadata, chunks=None, daskify=True, thread_safe=True, fancy_index=True, order=True,
                 ch_offsets=None):
        """
        Construct a lazy collection of 2-D time-series arrays.

        Parameters
        ----------
        data : array-like or sequence of array-like
            One or more arrays with shape ``(channels, samples)``.
        metadata : mapping or sequence of mappings
            One metadata mapping per input array.
        chunks : optional
            Chunk specification for non-Dask inputs. For multiple datasets,
            provide one specification per dataset.
        daskify : bool
            Retained for API compatibility. Non-Dask inputs are always wrapped
            lazily so that ``self.data`` has a consistent Dask-array invariant.
        thread_safe, fancy_index : bool
            Forwarded to ``dask.array.from_array`` for non-Dask inputs.
        order : bool
            Sort datasets by ``metadata['start_time']``.
        ch_offsets : sequence or sequence of sequences, optional
            Negative sample offsets. A flat sequence applies to every dataset;
            a nested sequence supplies one offset vector per dataset.
        """
        data_list = list(data) if isinstance(data, (list, tuple)) else [data]
        metadata_list = list(metadata) if isinstance(metadata, (list, tuple)) else [metadata]

        if not data_list:
            raise ValueError("data must contain at least one dataset.")
        if len(data_list) != len(metadata_list):
            raise ValueError(
                f"data contains {len(data_list)} dataset(s), but metadata contains "
                f"{len(metadata_list)} entry/entries."
            )
        if not all(isinstance(md, Mapping) for md in metadata_list):
            raise TypeError("Each metadata entry must be a mapping.")

        # Copy only the top-level mappings. Acquisition metadata may contain
        # large nested arrays or tables that the constructor never mutates.
        # A full deepcopy adds substantial initialization time and RSS.
        metadata_list = [dict(md) for md in metadata_list]
        chunk_specs = self._normalize_chunk_specs(chunks, len(data_list))

        arrays = []
        for source, chunk_spec in zip(data_list, chunk_specs):
            if isinstance(source, da.Array):
                arr = source
            else:
                if chunk_spec is None:
                    chunk_spec = getattr(source, "chunks", "auto")
                arr = da.from_array(
                    source,
                    chunks=chunk_spec,
                    lock=thread_safe,
                    fancy=fancy_index,
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"Time-series datasets must be 2-D (channels, samples); got shape {arr.shape}."
                )
            arrays.append(arr)

        offsets_by_dataset = self._normalize_offsets(ch_offsets, len(arrays))
        for i, offsets in enumerate(offsets_by_dataset):
            if offsets is None:
                continue
            offsets = np.asarray(offsets, dtype=int).ravel()
            if offsets.size != int(arrays[i].shape[0]):
                raise ValueError(
                    f"Channel-offset length ({offsets.size}) does not match dataset {i} "
                    f"channel count ({arrays[i].shape[0]})."
                )
            if np.any(offsets > 0):
                raise ValueError("Channel offsets must be zero or negative sample counts.")

            arrays[i] = self._apply_channel_offsets(arrays[i], offsets)
            metadata_list[i]["ch_offsets"] = offsets.tolist()
            crop = int(-offsets.min(initial=0))
            if crop and "stream_lengths" in metadata_list[i]:
                value = metadata_list[i]["stream_lengths"]
                adjusted = np.asarray(value) - crop
                if np.any(adjusted < 0):
                    raise ValueError("Channel offsets exceed the recorded stream length.")
                if np.isscalar(value):
                    metadata_list[i]["stream_lengths"] = int(adjusted)
                elif isinstance(value, np.ndarray):
                    metadata_list[i]["stream_lengths"] = adjusted.astype(value.dtype, copy=False)
                else:
                    metadata_list[i]["stream_lengths"] = adjusted.tolist()

        channel_counts = {int(arr.shape[0]) for arr in arrays}
        if len(channel_counts) != 1:
            raise ValueError("All datasets must contain the same number of channels.")

        n_channels = next(iter(channel_counts))
        for i, md in enumerate(metadata_list):
            if "ch_names" not in md:
                raise KeyError(f"metadata[{i}] is missing required key 'ch_names'.")
            if len(md["ch_names"]) != n_channels:
                raise ValueError(
                    f"metadata[{i}]['ch_names'] has length {len(md['ch_names'])}; "
                    f"expected {n_channels}."
                )
            if "types" in md and len(md["types"]) != n_channels:
                raise ValueError(
                    f"metadata[{i}]['types'] has length {len(md['types'])}; expected {n_channels}."
                )

        if order:
            combined = sorted(
                zip(arrays, metadata_list),
                key=lambda pair: pair[1].get("start_time", 0),
            )
            arrays = [pair[0] for pair in combined]
            metadata_list = [pair[1] for pair in combined]

        self.data = arrays
        self.metadata = metadata_list
        self._array = None
        self._rebuild_channel_maps()

    @staticmethod
    def _normalize_chunk_specs(chunks, n_datasets):
        if chunks is None:
            return [None] * n_datasets

        if n_datasets == 1:
            if isinstance(chunks, list) and len(chunks) == 1:
                return [chunks[0]]
            return [chunks]

        if not isinstance(chunks, (list, tuple)) or len(chunks) != n_datasets:
            raise ValueError(
                "For multiple datasets, chunks must contain one chunk specification per dataset."
            )
        return list(chunks)

    @staticmethod
    def _normalize_offsets(ch_offsets, n_datasets):
        if ch_offsets is None:
            return [None] * n_datasets

        if (
            n_datasets > 1
            and isinstance(ch_offsets, (list, tuple))
            and len(ch_offsets) == n_datasets
            and all(
                item is None
                or (isinstance(item, Sequence) and not isinstance(item, (str, bytes)))
                for item in ch_offsets
            )
        ):
            return list(ch_offsets)

        return [ch_offsets] * n_datasets

    @staticmethod
    def _apply_channel_offsets(data, offsets):
        offsets = np.asarray(offsets, dtype=int).ravel()
        min_offset = int(offsets.min(initial=0))
        if min_offset == 0 and np.all(offsets == 0):
            return data

        n_time = int(data.shape[1])

        # Common TDT case: every channel has the same acquisition delay.
        # Use one direct time slice instead of building channel groups and a
        # concatenate layer. The slice still contributes one task per source
        # time chunk, which is required to expose the corrected logical array.
        if np.all(offsets == offsets[0]):
            start = int(-offsets[0])
            return data[:, start:n_time]

        pieces = []
        for offset in np.unique(offsets):
            selected = np.flatnonzero(offsets == offset)
            for group in _group_consecutive(selected):
                start_ch = int(group[0])
                stop_ch = int(group[-1]) + 1
                start = int(-offset)
                stop = int(n_time + min_offset - offset)
                pieces.append((start_ch, data[start_ch:stop_ch, start:stop]))

        pieces.sort(key=lambda item: item[0])
        return da.concatenate([piece for _, piece in pieces], axis=0)

    def _spawn(self, data=None, metadata=None):
        """Create a same-type lazy object without rerunning acquisition loading."""
        new = copy.copy(self)
        _TsData.__init__(
            new,
            self.data if data is None else data,
            copy.deepcopy(self.metadata) if metadata is None else metadata,
            chunks=None,
            daskify=False,
            order=False,
            ch_offsets=None,
        )
        return new

    def _rebuild_channel_maps(self):
        """
        Build fast lookup structures for channel selection.
        """
        # name -> index
        self._ch_name_to_idx = {name: i for i, name in enumerate(self.ch_names)}

        # optional: type -> boolean mask (handy elsewhere)
        try:
            ct = self.ch_types  # list-like, length n_channels
            self._ch_type_to_mask = {t: (np.asarray(ct) == t) for t in set(ct)}
        except Exception:
            self._ch_type_to_mask = {}

    @property
    def array(self):
        """
        Cached concatenated dask array across datasets (channels x time).
        Rebuilt only when self.data changes.

        Returns
        -------
        dask.array.core.Array
            Array of the class instance data sets.

        Examples
        ________
        >>> ephys_data.array     # doctest: +SKIP

        """
        a = self._array
        if a is None:
            # NOTE: concatenation along time axis
            a = da.concatenate(self.data, axis=1) if len(self.data) > 1 else self.data[0]
            self._array = a
        return a

    @property
    def chunks(self):
        """Actual Dask chunks for each component dataset."""
        return [d.chunks for d in self.data]

    @property
    def ch_offsets(self):
        """Return shared channel offsets, or one value per dataset when they differ."""
        offsets = [metadata.get("ch_offsets") for metadata in self.metadata]
        normalized = [None if value is None else tuple(value) for value in offsets]
        if len(set(normalized)) == 1:
            return None if normalized[0] is None else list(normalized[0])
        return [None if value is None else list(value) for value in normalized]

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

    @property
    def ndim(self):
        """
        Property getter method for the number of dimensions in the raw data array.

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

    def remove_data(self, datasets, invert=False):
        """Return a new object with selected component datasets removed or retained."""
        n = self.ndata
        arr = np.asarray(datasets)
        if arr.dtype == np.bool_:
            selected = arr.ravel().astype(bool, copy=False)
            if selected.size != n:
                raise ValueError("Dataset boolean mask length does not match ndata.")
        else:
            selected = np.zeros(n, dtype=bool)
            for idx in np.atleast_1d(arr).tolist():
                idx = int(idx)
                if idx < 0:
                    idx += n
                if not 0 <= idx < n:
                    raise IndexError(f"Dataset index {idx} is out of range for {n} datasets.")
                selected[idx] = True

        keep = selected if invert else ~selected
        if not np.any(keep):
            raise ValueError("The operation would remove every dataset.")

        data = [d for d, flag in zip(self.data, keep) if flag]
        metadata = [md for md, flag in zip(self.metadata, keep) if flag]
        result = self._spawn(data=data, metadata=metadata)
        if hasattr(self, "io") and len(self.io) == n:
            result.io = [item for item, flag in zip(self.io, keep) if flag]
        return result

    def _metadata_after_channel_mask(self, keep):
        keep = np.asarray(keep, dtype=bool).ravel()
        n_before = len(keep)
        metadata = copy.deepcopy(self.metadata)
        per_channel_keys = ("ch_names", "types", "ch_offsets", "units", "channel_units")

        for md in metadata:
            for key in per_channel_keys:
                value = md.get(key)
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) == n_before:
                    md[key] = [item for item, flag in zip(value, keep) if flag]

            streams = md.get("channels")
            if (
                isinstance(streams, list)
                and all(isinstance(stream, (list, tuple, np.ndarray)) for stream in streams)
            ):
                if sum(len(stream) for stream in streams) != n_before:
                    raise ValueError(
                        "Metadata 'channels' does not align with the data channel axis."
                    )
                new_streams = []
                cursor = 0
                for stream in streams:
                    local_keep = keep[cursor:cursor + len(stream)]
                    new_streams.append(
                        [item for item, flag in zip(stream, local_keep) if flag]
                    )
                    cursor += len(stream)
                md["channels"] = new_streams
                md["channels_per_stream"] = [len(stream) for stream in new_streams]
                md["cumulative_channel_count"] = list(
                    np.cumsum(md["channels_per_stream"], dtype=int)
                )

            for key in ("n_channels", "num_channels"):
                if key in md:
                    md[key] = int(keep.sum())

        return metadata

    @staticmethod
    def _update_metadata_list(metadata, op_record, n_arrays):
        """
        Ensure metadata is a list[dict] of length n_arrays.
        Append op_record to each entry's 'history' list.
        """
        if isinstance(metadata, Mapping) or metadata is None:
            metadata = [metadata] * n_arrays
        elif not isinstance(metadata, (list, tuple)):
            raise TypeError("metadata must be a mapping or a sequence of mappings.")

        if len(metadata) != n_arrays:
            if len(metadata) == 1:
                metadata = list(metadata) * n_arrays
            else:
                raise ValueError(
                    f"Expected {n_arrays} metadata entries, received {len(metadata)}."
                )

        # Deep-ish copy & append history
        out = []
        for md in metadata:
            md = {} if md is None else dict(md)
            hist = list(md.get("history", []))
            hist.append(op_record)
            md["history"] = hist
            out.append(md)
        return out

    def remove_ch(self, channels, invert=False):
        """Return a new object with channels removed, or retained when ``invert=True``."""
        selected = self._ch_to_index(channels)
        keep = selected if invert else ~selected
        if not np.any(keep):
            raise ValueError("The operation would remove every channel.")

        data = [d[keep, :] for d in self.data]
        metadata = self._metadata_after_channel_mask(keep)
        return self._spawn(data=data, metadata=metadata)

    def set_ch_names(self, ch_names):
        """Return a new object with renamed channels."""
        ch_names = list(ch_names)
        if len(ch_names) != len(self.ch_names):
            raise ValueError("Number of channel names must match the data channel count.")
        if len(set(ch_names)) != len(ch_names):
            raise ValueError("Channel names must be unique.")

        metadata = copy.deepcopy(self.metadata)
        for md in metadata:
            md["ch_names"] = ch_names.copy()
        return self._spawn(metadata=metadata)

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
        ['L','E']
        """
        ch_types = [tuple(meta['types']) if 'types' in meta.keys() else tuple() for meta in self.metadata]
        if len(set(ch_types)) == 1:
            return list(dict.fromkeys(ch_types[0]))
        else:
            raise ValueError("Import data sets do not have consistent channel types.")

    @property
    def ch_types(self):
        # TODO: Repeat property same as types - eliminate one.
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
            return list(ch_types[0])
        else:
            raise ValueError("Import data sets do not have consistent channel types.")

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

    def set_ch_types(self, ch_types, rename=False):
        """Return a new object with channel types assigned."""
        ch_types = list(ch_types)
        if len(ch_types) != len(self.ch_names):
            raise ValueError("Number of channel types must match the data channel count.")

        metadata = copy.deepcopy(self.metadata)
        for md in metadata:
            md["types"] = ch_types.copy()

        result = self._spawn(metadata=metadata)
        if not rename:
            return result

        counts = {}
        names = []
        for channel_type in ch_types:
            counts[channel_type] = counts.get(channel_type, 0) + 1
            names.append(f"{channel_type} {counts[channel_type]}")
        return result.set_ch_names(names)

    def time(self, remove_gaps=True, chunk_size=1_000_000):
        """Return a lazy elapsed-time axis in seconds."""
        sample_rate = float(self.sample_rate)
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        axes = []
        cumulative_samples = 0
        first_start = float(self.start_times[0])
        for data, start_time in zip(self.data, self.start_times):
            n_samples = int(data.shape[1])
            chunks = min(chunk_size, max(n_samples, 1))
            local = da.arange(n_samples, chunks=chunks, dtype=np.float64) / sample_rate
            if remove_gaps:
                local = local + cumulative_samples / sample_rate
                cumulative_samples += n_samples
            else:
                local = local + (float(start_time) - first_start)
            axes.append(local)

        return da.concatenate(axes) if len(axes) > 1 else axes[0]

    def channel_reference(self, channel, ch_type=None):
        """Reference selected channels to one channel, then remove the reference channel."""
        reference_mask = self._ch_to_index(channel)
        if int(reference_mask.sum()) != 1:
            raise ValueError("channel must identify exactly one reference channel.")
        reference_index = int(np.flatnonzero(reference_mask)[0])

        if ch_type is None:
            apply_mask = np.ones(len(self.ch_names), dtype=bool)
        else:
            requested = [ch_type] if isinstance(ch_type, str) else list(ch_type)
            unknown = [item for item in requested if item not in self.types]
            if unknown:
                raise ValueError(f"Unknown channel type(s): {unknown}")
            apply_mask = np.zeros(len(self.ch_names), dtype=bool)
            for item in requested:
                apply_mask |= self._ch_type_to_index(item)

        mask_column = apply_mask[:, None]
        data = []
        for arr in self.data:
            reference = arr[reference_index, :][None, :]
            data.append(da.where(mask_column, arr - reference, arr))

        return self._spawn(data=data).remove_ch(reference_index)

    def common_reference(self, ch_type=None, method="mean", *, exclude_self=False):
        """Reference selected channels to their common mean or median."""
        method = str(method).lower()
        if method not in {"mean", "median"}:
            raise ValueError("method must be 'mean' or 'median'.")

        if ch_type is None:
            mask = np.ones(len(self.ch_names), dtype=bool)
        else:
            requested = [ch_type] if isinstance(ch_type, str) else list(ch_type)
            unknown = [item for item in requested if item not in self.types]
            if unknown:
                raise ValueError(f"Unknown channel type(s): {unknown}")
            mask = np.zeros(len(self.ch_names), dtype=bool)
            for item in requested:
                mask |= self._ch_type_to_index(item)

        n_selected = int(mask.sum())
        if n_selected < 2:
            raise ValueError("At least two selected channels are required for common referencing.")
        mask_column = mask[:, None]
        data = []

        if method == "mean":
            for arr in self.data:
                common = da.mean(arr[mask, :], axis=0)[None, :]
                referenced = arr - common
                if exclude_self:
                    referenced = referenced * (n_selected / (n_selected - 1))
                data.append(da.where(mask_column, referenced, arr))
        else:
            out_dtype = np.result_type(np.float32, *(arr.dtype for arr in self.data))

            def median_reference_block(block, apply_mask, leave_one_out):
                x = block.astype(out_dtype, copy=False)
                out = x.copy()
                selected = np.flatnonzero(apply_mask)
                if leave_one_out:
                    for index in selected:
                        others = selected[selected != index]
                        out[index] = x[index] - np.median(x[others], axis=0)
                else:
                    common = np.median(x[selected], axis=0)
                    out[selected] = x[selected] - common[None, :]
                return out

            for arr in self.data:
                if len(arr.chunks[0]) != 1:
                    arr = arr.rechunk({0: -1})
                data.append(
                    da.map_blocks(
                        median_reference_block,
                        arr,
                        dtype=out_dtype,
                        chunks=arr.chunks,
                        apply_mask=mask,
                        leave_one_out=exclude_self,
                        meta=np.empty((0, 0), dtype=out_dtype),
                    )
                )

        return self._spawn(data=data)

    def filter_iir(
            self,
            Wn,
            rp=None,
            rs=None,
            btype="band",
            order=1,
            ftype="butter",
            *,
            overlap=None,
            transient_tol=1e-6,
    ):
        """
        Apply a zero-phase IIR filter lazily along the time axis.

        Filtering uses :func:`scipy.signal.sosfiltfilt` inside
        :func:`dask.array.map_overlap`. The Dask overlap is estimated from the
        slowest-decaying filter pole rather than from ``sosfiltfilt``'s
        ``padlen``. ``padlen`` only controls endpoint padding and is generally
        too short to suppress transients at internal Dask chunk boundaries.

        Parameters
        ----------
        Wn : float or array-like
            Critical frequency or frequencies in Hz.
        rp : float, optional
            Maximum passband ripple in decibels for applicable filter types.
        rs : float, optional
            Minimum stopband attenuation in decibels for applicable filters.
        btype : {'band', 'bandpass', 'bandstop', 'lowpass', 'highpass'}
            Filter type.
        order : int
            Filter order.
        ftype : str
            IIR design family accepted by :func:`scipy.signal.iirfilter`.
        overlap : int, optional
            Number of neighboring samples supplied on each side of every Dask
            block. By default it is estimated from the maximum pole radius and
            ``transient_tol``.
        transient_tol : float
            Target pole-decay magnitude used for automatic overlap estimation.
            Must lie strictly between 0 and 1. Smaller values use more overlap
            and more closely reproduce whole-array filtering.

        Returns
        -------
        _TsData or subclass
            New object containing the lazily filtered data.
        """
        btype_map = {
            "band": "bandpass",
            "bandpass": "bandpass",
            "bandstop": "bandstop",
            "low": "lowpass",
            "lowpass": "lowpass",
            "high": "highpass",
            "highpass": "highpass",
        }
        try:
            btype = btype_map[str(btype).lower()]
        except KeyError as exc:
            raise ValueError(
                "btype must be 'band'/'bandpass', 'bandstop', "
                "'low'/'lowpass', or 'high'/'highpass'."
            ) from exc

        Wn = np.atleast_1d(np.asarray(Wn, dtype=float))
        fs = float(self.sample_rate)

        if np.any(~np.isfinite(Wn)) or np.any(Wn <= 0) or np.any(Wn >= fs / 2):
            raise ValueError(
                f"Wn must contain finite frequencies in (0, fs/2). "
                f"Got Wn={Wn}, fs/2={fs / 2:.3f} Hz."
            )
        if btype in {"bandpass", "bandstop"} and Wn.size != 2:
            raise ValueError(f"{btype} requires two critical frequencies; got {Wn.size}.")
        if btype in {"lowpass", "highpass"} and Wn.size != 1:
            raise ValueError(f"{btype} requires one critical frequency; got {Wn.size}.")
        if Wn.size == 2 and Wn[0] >= Wn[1]:
            raise ValueError(f"Band edges must be strictly increasing; got Wn={Wn}.")

        order = int(order)
        if order < 1:
            raise ValueError(f"order must be a positive integer; got {order}.")

        transient_tol = float(transient_tol)
        if not 0.0 < transient_tol < 1.0:
            raise ValueError(
                f"transient_tol must lie strictly between 0 and 1; got {transient_tol}."
            )

        sos = signal.iirfilter(
            order,
            Wn,
            rp=rp,
            rs=rs,
            btype=btype,
            ftype=ftype,
            output="sos",
            fs=fs,
        )

        # Match scipy.signal.sosfiltfilt's default pad-length calculation.
        n_sections = sos.shape[0]
        ntaps = 2 * n_sections + 1
        ntaps -= min(
            int(np.count_nonzero(sos[:, 2] == 0.0)),
            int(np.count_nonzero(sos[:, 5] == 0.0)),
        )
        padlen = max(3 * ntaps, 1)

        # A stable IIR has poles strictly inside the unit circle. The slowest
        # pole determines how much neighboring data is needed before a block
        # edge transient decays below transient_tol.
        _, poles, _ = signal.sos2zpk(sos)
        max_pole_radius = float(np.max(np.abs(poles))) if poles.size else 0.0
        if not np.isfinite(max_pole_radius) or max_pole_radius >= 1.0:
            raise ValueError(
                "The designed IIR filter is unstable or numerically invalid: "
                f"maximum pole radius={max_pole_radius!r}."
            )

        if overlap is None:
            if max_pole_radius <= 0.0:
                estimated_overlap = padlen
            else:
                estimated_overlap = int(
                    np.ceil(np.log(transient_tol) / np.log(max_pole_radius))
                )
            overlap = max(padlen, estimated_overlap)
        else:
            if isinstance(overlap, bool):
                raise TypeError("overlap must be an integer number of samples, not bool.")
            overlap = int(overlap)
            if overlap < padlen:
                raise ValueError(
                    f"overlap must be at least padlen ({padlen}) samples; got {overlap}."
                )

        out_dtype = np.result_type(np.float32, *(d.dtype for d in self.data))

        def _filt_last_axis(block):
            x = block.astype(out_dtype, copy=False)
            if x.shape[-1] <= padlen:
                raise ValueError(
                    "An overlapped IIR block must contain more samples than "
                    f"padlen={padlen}; got {x.shape[-1]}."
                )
            return signal.sosfiltfilt(
                sos,
                x,
                axis=-1,
                padlen=padlen,
            ).astype(out_dtype, copy=False)

        filtered = []
        for d in self.data:
            time_axis = d.ndim - 1
            n_time = int(d.shape[time_axis])
            if n_time <= padlen:
                raise ValueError(
                    f"Dataset length ({n_time}) must exceed IIR padlen ({padlen})."
                )

            # Avoid relying on map_overlap's implicit rechunk when the requested
            # overlap is larger than one or more existing time chunks.
            if min(d.chunks[time_axis]) <= overlap:
                target_time_chunk = min(
                    n_time,
                    max(65_536, 4 * overlap + 1),
                )
                d = d.rechunk({time_axis: target_time_chunk})

            depth = {time_axis: overlap}
            f = da.map_overlap(
                _filt_last_axis,
                d,
                depth=depth,
                boundary="none",
                trim=True,
                dtype=out_dtype,
                allow_rechunk=True,
            )
            filtered.append(f)

        op = {
            "op": "filter_iir",
            "order": order,
            "ftype": ftype,
            "btype": btype,
            "Wn": Wn.tolist(),
            "rp": rp,
            "rs": rs,
            "fs": fs,
            "padlen": int(padlen),
            "overlap": int(overlap),
            "transient_tol": transient_tol,
            "max_pole_radius": max_pole_radius,
        }
        new_meta = self._update_metadata_list(
            self.metadata,
            op,
            n_arrays=len(filtered),
        )

        return self._spawn(data=filtered, metadata=new_meta)

    def filter_fir(self, cutoff, width=None, filter_length="auto", window="hamming",
                   pass_zero=True, tap_limit=None):
        """Apply a zero-phase FIR filter lazily along the time axis."""
        fs = float(self.sample_rate)
        cutoff = np.atleast_1d(np.asarray(cutoff, dtype=float))
        if np.any(cutoff <= 0) or np.any(cutoff >= fs / 2):
            raise ValueError("cutoff frequencies must lie strictly between 0 and Nyquist.")

        if filter_length == "auto":
            if width is None or float(width) <= 0:
                raise ValueError("A positive width is required when filter_length='auto'.")
            numtaps = int((3.3 * fs) / (2 * float(width))) * 2 + 1
        else:
            numtaps = int(filter_length)
            if numtaps < 3:
                raise ValueError("filter_length must be at least 3 taps.")

        if tap_limit is not None:
            numtaps = min(numtaps, int(tap_limit))
        if numtaps < 3:
            raise ValueError("The resulting FIR filter must contain at least 3 taps.")
        if numtaps % 2 == 0:
            numtaps += 1

        weights = signal.firwin(
            numtaps,
            cutoff,
            width=width,
            window=window,
            pass_zero=pass_zero,
            fs=fs,
        )
        radius = numtaps // 2
        overlap = 2 * radius
        out_dtype = np.result_type(np.float32, *(arr.dtype for arr in self.data))
        weights = weights.astype(out_dtype, copy=False)

        def zero_phase_fft(block):
            x = block.astype(out_dtype, copy=False)
            y = signal.fftconvolve(x, weights[None, :], mode="same", axes=(-1,))
            y = signal.fftconvolve(y[..., ::-1], weights[None, :], mode="same", axes=(-1,))
            return y[..., ::-1].astype(out_dtype, copy=False)

        filtered = []
        for arr in self.data:
            if min(arr.chunks[-1]) <= 2 * overlap:
                target = max(65_536, 8 * radius)
                arr = arr.rechunk({arr.ndim - 1: int(target)})
            filtered.append(
                da.map_overlap(
                    zero_phase_fft,
                    arr,
                    depth={arr.ndim - 1: overlap},
                    boundary="reflect",
                    trim=True,
                    dtype=out_dtype,
                )
            )

        op = {
            "op": "filter_fir",
            "cutoff": cutoff.tolist(),
            "width": width,
            "numtaps": int(numtaps),
            "window": window,
            "pass_zero": pass_zero,
        }
        metadata = self._update_metadata_list(self.metadata, op, len(filtered))
        return self._spawn(data=filtered, metadata=metadata)

    def filter_median(
            self,
            kernel_size=201,
            btype="lowpass",
            *,
            boundary="reflect",
            persist_input=False,
    ):
        """
        Apply a median filter along the time axis.

        The median is computed independently for each channel using SciPy's
        optimized 1-D median-filter path.

        Parameters
        ----------
        kernel_size : int
            Width of the median-filter window in samples. Even values are
            increased by one so that the window is symmetric.
        btype : {"lowpass", "low", "highpass", "high"}
            ``lowpass`` returns the rolling median.
            ``highpass`` subtracts the rolling median from the original data.
        boundary : str
            Boundary handling passed to Dask ``map_overlap``.
        persist_input : bool
            Persist the input Dask array before filtering.

        Returns
        -------
        _TsData or subclass
            New object containing the lazily filtered data.
        """

        if not isinstance(kernel_size, int) or kernel_size < 3:
            raise ValueError(
                f"kernel_size must be integer >= 3, got {kernel_size}."
            )

        if kernel_size % 2 == 0:
            kernel_size += 1

        btype = str(btype).lower()

        if btype not in (
                "low",
                "lowpass",
                "high",
                "highpass",
        ):
            raise ValueError(
                "btype must be 'lowpass'/'low' or "
                "'highpass'/'high'."
            )

        radius = kernel_size // 2

        in_dtype = np.result_type(
            *(d.dtype for d in self.data)
        )

        if btype in ("high", "highpass"):
            out_dtype = np.result_type(
                np.float32,
                in_dtype,
            )
        else:
            out_dtype = in_dtype

        # --------------------------------------------------------
        # Fast median implementation
        # --------------------------------------------------------

        def _median(block):
            """
            Median filter independently along the final axis.

            Using scipy.ndimage.median_filter on each 1-D trace is
            dramatically faster than calling the N-D implementation with
            size=(1, kernel_size).
            """

            block = np.asarray(block)

            original_shape = block.shape

            # Flatten all non-time dimensions.
            traces = block.reshape(
                -1,
                original_shape[-1],
            )

            filtered = np.empty_like(traces)

            for index in range(traces.shape[0]):
                filtered[index] = ndimage.median_filter(
                    traces[index],
                    size=kernel_size,
                    mode="nearest",
                )

            return filtered.reshape(original_shape)

        def _median_high(block):
            x = block.astype(
                out_dtype,
                copy=False,
            )

            return x - _median(x)

        if btype in ("high", "highpass"):
            op = _median_high
        else:
            op = _median

        # --------------------------------------------------------
        # Construct lazy Dask arrays
        # --------------------------------------------------------

        filtered = []

        for d in self.data:
            x = (
                d.persist()
                if persist_input
                else d
            )

            y = da.map_overlap(
                op,
                x,
                depth={
                    x.ndim - 1: radius
                },
                boundary=boundary,
                trim=True,
                dtype=out_dtype,
            )

            filtered.append(y)

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        op_meta = {
            "op": "filter_median",
            "kernel_size": int(kernel_size),
            "btype": btype,
            "boundary": boundary,
        }

        new_meta = self._update_metadata_list(
            self.metadata,
            op_meta,
            n_arrays=len(filtered),
        )

        return self._spawn(
            data=filtered,
            metadata=new_meta,
        )

    def filter_masked_gaussian_baseline(
            self,
            sigma,
            *,
            stim=None,
            stim_times=None,
            stim_indices=None,
            pre=0.0003,
            post=0.0015,
            truncate=4.0,
            mode="nearest",
            fill_method="nearest",
            baseline_only=False,
            out_dtype=None,
    ):
        """Subtract a Gaussian baseline while excluding stimulation intervals."""
        if sigma is None or sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")

        if stim_indices is None:
            if stim_times is None:
                if stim is None:
                    raise ValueError("Provide one of stim, stim_times, or stim_indices.")
                stim_times = stim.all_event_times(reference=self)
            stim_times = np.asarray(stim_times, dtype=float).ravel()
        else:
            stim_indices = np.asarray(stim_indices, dtype=np.int64).ravel()

        input_dtype = np.result_type(*(arr.dtype for arr in self.data))
        if out_dtype is None:
            out_dtype = np.result_type(np.float32, input_dtype)

        sample_rate = float(self.sample_rate)
        sigma_samples = float(sigma * sample_rate)
        pre_samples = int(round(pre * sample_rate))
        post_samples = int(round(post * sample_rate))
        radius = max(1, int(np.ceil(truncate * sigma_samples)))

        def masked_gaussian(block, mask_block):
            x = block.astype(out_dtype, copy=False)
            weights = (~mask_block.astype(bool, copy=False)).astype(out_dtype, copy=False)
            numerator = ndimage.gaussian_filter1d(
                x * weights, sigma=sigma_samples, axis=-1, mode=mode, truncate=truncate
            )
            denominator = ndimage.gaussian_filter1d(
                weights, sigma=sigma_samples, axis=-1, mode=mode, truncate=truncate
            )
            result = np.empty_like(numerator, dtype=out_dtype)
            good = denominator > np.finfo(np.float32).eps
            result[good] = numerator[good] / denominator[good]
            if np.any(~good):
                if fill_method == "nearest":
                    fallback = ndimage.gaussian_filter1d(
                        x, sigma=sigma_samples, axis=-1, mode=mode, truncate=truncate
                    )
                    result[~good] = fallback[~good]
                elif fill_method == "constant":
                    result[~good] = 0
                else:
                    raise ValueError("fill_method must be 'nearest' or 'constant'.")
            return result

        filtered = []
        global_starts = self.start_indices
        for dataset_index, (arr, metadata) in enumerate(zip(self.data, self.metadata)):
            n_time = int(arr.shape[-1])
            if stim_indices is None:
                start_time = float(metadata.get("start_time", 0.0) or 0.0)
                local_events = np.rint((stim_times - start_time) * sample_rate).astype(np.int64)
            else:
                local_events = stim_indices - int(global_starts[dataset_index])

            local_events = local_events[
                (local_events + post_samples >= 0)
                & (local_events - pre_samples < n_time)
            ]

            template = da.empty((n_time,), chunks=arr.chunks[-1], dtype=bool)

            def build_mask(block, block_info=None, events=local_events):
                if block_info is None:
                    return np.zeros_like(block, dtype=bool)
                location = block_info[None]["array-location"][0]
                block_start, block_stop = map(int, location)
                output = np.zeros(block_stop - block_start, dtype=bool)
                relevant = events[
                    (events + post_samples > block_start)
                    & (events - pre_samples < block_stop)
                ]
                for event in relevant:
                    start = max(block_start, int(event) - pre_samples) - block_start
                    stop = min(block_stop, int(event) + post_samples) - block_start
                    if start < stop:
                        output[start:stop] = True
                return output

            mask_1d = da.map_blocks(
                build_mask, template, dtype=bool, meta=np.empty((0,), dtype=bool)
            )
            shape = (1,) * (arr.ndim - 1) + (n_time,)
            mask = da.broadcast_to(mask_1d.reshape(shape), arr.shape).rechunk(arr.chunks)
            baseline = da.map_overlap(
                masked_gaussian,
                arr,
                mask,
                depth={arr.ndim - 1: radius},
                boundary=mode,
                trim=True,
                dtype=out_dtype,
            )
            filtered.append(
                baseline if baseline_only else arr.astype(out_dtype, copy=False) - baseline
            )

        op = {
            "op": "filter_masked_gaussian_baseline",
            "sigma_s": float(sigma),
            "pre_s": float(pre),
            "post_s": float(post),
            "truncate": float(truncate),
            "mode": mode,
            "baseline_only": bool(baseline_only),
        }
        metadata = self._update_metadata_list(self.metadata, op, len(filtered))
        return self._spawn(data=filtered, metadata=metadata)

    def filter_gaussian(self, Wn, btype='lowpass', order=0, truncate=4.0):
        """
        Filters the channels using a gaussian filter using the scipy.ndimage.guassian_filter1d method.

        Parameters
        ----------
        Wn : int, float
            Corner frequency in Hz (−3 dB for order=0). Must be in (0, fs/2).
        btype : str
            Use 'lowpass' or 'low' to attenuate high frequency signals. Use 'highpass' or 'high to attenuate low
            frequency signals.
        order : int
            The order of the kernel. 0 corresponds to a filter with a Gaussian kernel. 1 corresponds to the derivative,
            ect.
        truncate : float
            Radius in standard deviations. Kernel radius ≈ truncate * sigma.

        Returns
        -------
        _TsData or subclass
            New class instance of the same type as self which contains the filtered data.
        """

        fs = float(self.sample_rate)
        fc = float(Wn)

        # ---- validate ----
        if not (0.0 < fc < fs / 2):
            raise ValueError(f"Wn (fc) must be in (0, fs/2). Got {fc} with fs/2={fs / 2:.3f} Hz.")
        btype = btype.lower()
        if btype not in ('lowpass', 'low', 'highpass', 'high'):
            raise ValueError("btype must be 'lowpass'/'low' or 'highpass'/'high'.")
        if (btype in ('highpass', 'high')) and order != 0:
            raise ValueError("Highpass is defined as x - lowpass(x) and requires order=0.")

        # ---- convert fc (Hz) -> sigma (samples) at -3 dB for order=0 ----
        # sigma_samples = sqrt(ln 2)/(2*pi) * (fs/fc)
        sigma = (np.sqrt(np.log(2.0)) / (2.0 * np.pi)) * (fs / fc)

        # map_overlap depth on time axis
        radius = int(truncate * sigma + 0.5)
        radius = max(radius, 1)

        def _depth_tuple(ndim):
            d = [0] * ndim
            d[-1] = radius
            return tuple(d)

        def _min_time_chunk(darr):
            ch = darr.chunks[-1]
            return min(ch) if isinstance(ch, tuple) else ch

        # ops
        def _gauss_low(block):
            return ndimage.gaussian_filter1d(block, sigma=sigma, axis=-1,
                                             order=order, mode='reflect',
                                             truncate=truncate)

        # highpass (order=0 only): x - lowpass(x)
        def _gauss_high(block):
            x = block.astype(np.result_type(np.float32, block.dtype), copy=False)
            m = ndimage.gaussian_filter1d(x, sigma=sigma, axis=-1,
                                          order=0, mode='reflect', truncate=truncate)
            return x - m

        op = _gauss_low if btype in ('lowpass', 'low') else _gauss_high
        out_dtype = (np.result_type(np.float32, *(d.dtype for d in self.data))
                     if btype in ('highpass', 'high') else
                     np.result_type(*(d.dtype for d in self.data)))

        filtered = []
        for d in self.data:
            # ensure enough interior after trimming
            if _min_time_chunk(d) < (2 * radius + 1):
                new_time = max(4 * radius + 1, 65536)  # pragmatic default
                d = d.rechunk({d.ndim - 1: int(new_time)})

            f = da.map_overlap(
                op, d,
                depth=_depth_tuple(d.ndim),
                boundary="none",  # ndimage handles padding (mode='reflect')
                trim=True,
                dtype=out_dtype,
            )
            filtered.append(f)

        # provenance
        # update metadata + provenance
        op = {
            "op": "filter_gaussian",
            "order": order,
            "ftype": f"gaussian {btype}",
            "Wn": Wn,
        }
        new_meta = self._update_metadata_list(self.metadata, op, n_arrays=len(filtered))

        return self._spawn(data=filtered, metadata=new_meta)

    def filter_powerline(self, frequencies=(60, 120, 180), notch_width=None, trans_bandwidth=1, tap_limit=None):
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
        return self.filter_fir(cutoffs, width=trans_bandwidth, tap_limit=tap_limit)

    def filter_powerline_iir(self, frequencies=(60,), Q=35.0, pad_seconds=0.5):
        fs = float(self.sample_rate)
        freqs = np.atleast_1d(frequencies).astype(float)

        # Build SOS cascade of notches
        sos_blocks = []
        for f0 in freqs:
            if 0 < f0 < fs / 2:
                b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
                sos_blocks.append(signal.tf2sos(b, a))
        if not sos_blocks:
            return self._spawn()

        sos = np.vstack(sos_blocks)

        # Choose padlen (samples) and overlap depth (samples)
        padlen = int(round(pad_seconds * fs))
        padlen = max(padlen, 256)

        # Ensure padlen is comfortably smaller than the smallest time-chunk
        min_time_chunk = min(min(d.chunks[1]) for d in self.data)
        # sosfiltfilt needs padlen < n_samples-1 for each block; we’ll be conservative:
        padlen = min(padlen, min_time_chunk // 10)

        depth = (0, padlen)
        out_dtype = np.result_type(np.float32, *(data.dtype for data in self.data))

        def _filt(x):
            values = x.astype(out_dtype, copy=False)
            return signal.sosfiltfilt(sos, values, axis=1, padlen=padlen).astype(
                out_dtype, copy=False
            )

        data_filt = [
            da.map_overlap(_filt, d, depth=depth, boundary="reflect", dtype=out_dtype)
            for d in self.data
        ]

        return self._spawn(data=data_filt)

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
             colors=None, fig_size=(10, 6), down_sample=True,
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
        from matplotlib.collections import LineCollection

        if colors is None:
            try:
                import seaborn as sns
                colors = sns.color_palette()
            except ImportError:
                colors = None

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

        x_index = (int(self._time_to_index(x_lim[0], remove_gaps=remove_gaps)),
                   int(self._time_to_index(x_lim[1], remove_gaps=remove_gaps)) + 1)
        x_slice = slice(x_index[0], x_index[1])
        time_array = self.time(remove_gaps=remove_gaps)[x_slice].compute()
        plot_array = self.array[channels, x_slice].compute()

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
            d_max = np.max(np.max(plot_array, axis=1) + tick_locations)
        else:
            d_min = y_lim[0]
            d_max = y_lim[1] + tick_locations[-1]
        ax.set_ylim(d_min, d_max)

        px_width, _ = _plt_ax_to_pix(fig, ax)

        plot_data = self._to_plt_line_collection(
            x_lim, channels, px_width, down_sample=down_sample, remove_gaps=remove_gaps,
            plot_arr=plot_array, time_arr=time_array, x_index=x_index,
        )
        current_lines = []
        for data in plot_data:
            # TODO: Matplotlib 3.5.0 has changed how offsets work. The easy solution for right now is to exclude
            #  matplotlib version > 3.5 from the install list, however the best long-term solution is likely to shift
            #  this to use matplotlibs transforms instead which should be compatible across versions.
            lines = LineCollection(data[0], offsets=offsets, colors=colors, linewidths=np.ones(plot_array.shape[0]),
                                   transOffset=None)
            current_lines.append(ax.add_collection(lines))

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
                events.plot_dio(axis=top_ax, reference=self, remove_gaps=remove_gaps, show=False, color='grey',
                                zorder=-1)
            if isinstance(events, _EventData):
                events.plot_events(axis=top_ax, reference=self, remove_gaps=remove_gaps, show=False, color='orange',
                                   lw=1)
            top_ax.set_xlim(x_lim)

        # show the plot if appropriate
        if show == "notebook":
            # Handle event data if passed to plot function
            if isinstance(events, _DioData):
                events.plot_dio(axis=scroll_ax, reference=self, remove_gaps=remove_gaps, show=False, color='orange',
                                zorder=-1)
            scroll_ax.set_xlim(self._time_lim_validate(None, remove_gaps=remove_gaps))
            ax.set_xlabel(None)
            scroll_ax.set_xlabel('Time (s)')
            scroll_ax.get_yaxis().set_ticks([])
            scroll_ax.get_yaxis().set_visible(False)
            scroll_span = scroll_ax.axvspan(x_lim[0], x_lim[1], color='green', zorder=11, alpha=0.7)
            scroll_line = scroll_ax.axvline(x_lim[0], color='green', zorder=11, alpha=0.7)
            # Add data set start points to the scrollbar axis

            if remove_gaps:
                start_times = self.start_indices / self.sample_rate
            else:
                start_times = [st - self.start_times[0] for st in self.start_times]

            for ts in start_times:
                scroll_ax.axvline(ts, lw=2, color='black', zorder=10)

            # display data gaps on scrollbar axis if applicable
            if not remove_gaps:
                gaps = [self.start_times[i] - self.end_times[i - 1] for i in range(1, len(self.start_times))]
                for g in range(len(gaps)):
                    scroll_ax.axvspan(self.end_times[g] - self.start_times[0],
                                      self.end_times[g] + gaps[g] - self.start_times[0], color='red', zorder=9,
                                      alpha=0.5)

            import matplotlib.pyplot as plt
            from ipywidgets import AppLayout, FloatSlider, Output

            plt.ioff()
            fig.tight_layout()
            slider = FloatSlider(
                orientation='horizontal',
                description='Start Time:',
                value=x_lim[0],
                min=0.0,
                max=max(0.0, self._time_lim_validate(None, remove_gaps=remove_gaps)[1] - (x_lim[1] - x_lim[0]))
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
                nonlocal current_lines
                nonlocal scroll_span
                nonlocal scroll_line
                # Get new x_limits
                window_width = x_lim[1] - x_lim[0]
                full_end = self._time_lim_validate(None, remove_gaps=remove_gaps)[1]
                start = min(float(change.new), max(0.0, full_end - window_width))
                x_lim_n = (start, start + window_width)

                # Calculate new line segments
                px_width_n, _ = _plt_ax_to_pix(fig, ax)
                plot_data_n = self._to_plt_line_collection(
                    x_lim_n, channels, px_width_n, down_sample=down_sample,
                    remove_gaps=remove_gaps,
                )

                # Create line collection and add to plot.
                for collection in current_lines:
                    collection.remove()
                current_lines = []
                scroll_span.remove()
                scroll_line.remove()
                down_sampled_n = []
                for data_n in plot_data_n:
                    down_sampled_n.append(data_n[1])
                    lines_n = LineCollection(data_n[0], offsets=offsets, colors=colors,
                                             linewidths=np.ones(plot_array.shape[0]), transOffset=None)
                    current_lines.append(ax.add_collection(lines_n))
                scroll_span = scroll_ax.axvspan(x_lim_n[0], x_lim_n[1], color='green', zorder=11, alpha=0.7)
                scroll_line = scroll_ax.axvline(x_lim_n[0], color='green', zorder=11, alpha=0.7)

                # Set new x_limits
                ax.set_xlim(x_lim_n)
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

    def _welch_psd_array(self, data, nperseg=None, noverlap=None, detrend="constant",
                         window="hann", batch_windows=128):
        fs = float(self.sample_rate)
        n_time = int(data.shape[-1])
        if nperseg is None:
            nperseg = int(fs)
        nperseg = min(int(nperseg), n_time)
        if nperseg < 2:
            raise ValueError("At least two samples are required for Welch PSD.")
        if noverlap is None:
            noverlap = nperseg // 2
        noverlap = int(noverlap)
        if not 0 <= noverlap < nperseg:
            raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg.")

        step = nperseg - noverlap
        n_windows = 1 + (n_time - nperseg) // step
        batch_windows = max(1, int(batch_windows))
        frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)

        def welch_batch(block, window_count):
            _, psd = signal.welch(
                block,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                window=window,
                scaling="density",
                axis=-1,
            )
            return np.asarray(psd, dtype=np.float64) * window_count, window_count

        def combine(left, right):
            return left[0] + right[0], left[1] + right[1]

        tasks = []
        for first_window in range(0, n_windows, batch_windows):
            count = min(batch_windows, n_windows - first_window)
            start = first_window * step
            stop = start + (count - 1) * step + nperseg
            segment = data[:, start:stop].rechunk({0: -1, 1: -1})
            block = segment.to_delayed().ravel()[0]
            tasks.append(delayed(welch_batch)(block, count))

        while len(tasks) > 1:
            reduced = []
            iterator = iter(tasks)
            for left in iterator:
                right = next(iterator, None)
                reduced.append(left if right is None else delayed(combine)(left, right))
            tasks = reduced

        psd_sum, count = tasks[0].compute()
        return frequencies, psd_sum / count

    def welch_psd_time(self, time_on, time_off, nperseg=None, noverlap=None,
                       detrend="constant", window="hann", batch_windows=128):
        limits = self._time_lim_validate((time_on, time_off))
        indices = self._time_to_index(limits)
        data = self.array[:, int(indices[0]):int(indices[1]) + 1]
        return self._welch_psd_array(
            data,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            window=window,
            batch_windows=batch_windows,
        )

    def plot_psd(self, axis=None, x_lim=None, y_lim=None, show=True, fig_size=(10, 3),
                 nperseg=None, noverlap=None, detrend="constant", window="hann",
                 colors=None, batch_windows=128, **kwargs):
        from matplotlib.collections import LineCollection

        if colors is None:
            try:
                import seaborn as sns
                colors = sns.color_palette()
            except ImportError:
                colors = None

        fig, ax = _plt_setup_fig_axis(axis, fig_size)
        frequencies, psd = self._welch_psd_array(
            self.array,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            window=window,
            batch_windows=batch_windows,
        )

        lines = [np.column_stack([frequencies, psd[channel]]) for channel in range(psd.shape[0])]
        collection = LineCollection(lines, linewidths=np.ones(psd.shape[0]), colors=colors, transOffset=None)
        ax.add_collection(collection)

        if x_lim is None:
            ax.set_xlim(frequencies[0], frequencies[-1])
            band = slice(None)
        else:
            ax.set_xlim(x_lim[0], x_lim[1])
            low = np.searchsorted(frequencies, x_lim[0], side="left")
            high = np.searchsorted(frequencies, x_lim[1], side="right")
            band = slice(low, high)

        if y_lim is None:
            visible = psd[:, band]
            positive = visible[visible > 0]
            if positive.size:
                ax.set_ylim(positive.min(), visible.max())
        else:
            ax.set_ylim(y_lim[0], y_lim[1])

        ax.set_yscale("log")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("PSD [V**2/Hz]")
        ax.set_title(None)
        return _plt_show_fig(fig, ax, show)

    def save(self, path, *args, scale=1, dtype=None, store='data', compression='gzip', method='hdf5', **kwargs):
        path = os.fspath(path)
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
            with h5py.File(path, mode='x', userblock_size=512):
                pass

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

            b = bytearray(s + (128 - 12 - len(s)) * ' ', encoding='utf-8')

            # Add 8 nulls (0) and the magic number (or something) that
            # MATLAB uses. Lengths must be gone to to make sure the argument
            # to fromhex is unicode because Python 2.6 requires it.

            b.extend(bytearray.fromhex(
                b'00000000 00000000 0002494D'.decode()))

            # Now, write it to the beginning of the file.

            with open(path, 'r+b') as fd:
                fd.write(b)
            ### This is the end of the reproduced section of code for modifying hdf5 files to be read as .mat files.
        else:
            raise ValueError(
                "Save method '{}' is not recognized. Implemented save methods include 'hdf5'.".format(method))
        with ProgressBar():
            data.to_hdf5(path, "/" + store, *args, compression=compression, **kwargs)

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
        """Return a strict boolean channel-selection mask."""
        n_channels = len(self.ch_names)
        if isinstance(ch, slice):
            mask = np.zeros(n_channels, dtype=bool)
            mask[ch] = True
            return mask

        if isinstance(ch, np.ndarray) and ch.dtype == np.bool_:
            values = ch.ravel()
            if values.size != n_channels:
                raise ValueError("Boolean channel mask length does not match channel count.")
            return values.astype(bool, copy=False)

        if isinstance(ch, (str, int, np.integer)) and not isinstance(ch, (bool, np.bool_)):
            values = [ch]
        else:
            try:
                values = list(ch)
            except TypeError as exc:
                raise TypeError("Channel selectors must be names, indices, a slice, or a boolean mask.") from exc

        if values and all(isinstance(value, (bool, np.bool_)) for value in values):
            mask = np.asarray(values, dtype=bool)
            if mask.size != n_channels:
                raise ValueError("Boolean channel mask length does not match channel count.")
            return mask

        if not hasattr(self, "_ch_name_to_idx"):
            self._rebuild_channel_maps()

        indices = []
        invalid = []
        for value in values:
            if isinstance(value, str):
                if value not in self._ch_name_to_idx:
                    invalid.append(value)
                else:
                    indices.append(self._ch_name_to_idx[value])
            elif isinstance(value, (bool, np.bool_)):
                invalid.append(value)
            else:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    invalid.append(value)
                    continue
                if index < 0:
                    index += n_channels
                if not 0 <= index < n_channels:
                    invalid.append(value)
                else:
                    indices.append(index)

        if invalid:
            raise ValueError(f"Unknown or out-of-range channel selector(s): {invalid}")
        if not indices:
            raise ValueError("No channels were selected.")

        mask = np.zeros(n_channels, dtype=bool)
        mask[np.unique(indices)] = True
        return mask

    def _time_lim_validate(self, x_lim, remove_gaps=True):
        """Validate and clamp a two-element elapsed-time range."""
        if remove_gaps:
            end = (int(self.shape[1]) - 1) / float(self.sample_rate)
        else:
            end = (
                float(self.start_times[-1]) - float(self.start_times[0])
                + (int(self.shapes[-1][1]) - 1) / float(self.sample_rate)
            )
        end = max(0.0, float(end))

        if x_lim is None:
            return 0.0, end

        limits = np.asarray(_to_numeric_array(x_lim), dtype=float).ravel()
        if limits.size != 2:
            raise ValueError("x_lim must contain exactly two values.")
        if np.any(limits < 0):
            raise ValueError(f"Time limits cannot be negative: {limits}")
        if limits[1] <= limits[0]:
            raise ValueError(f"End time must be greater than start time: {limits}")
        if limits[0] > end:
            raise ValueError(f"Time limits begin after the recording ends at {end:.9g} s.")
        limits[1] = min(limits[1], end)
        return float(limits[0]), float(limits[1])

    def _time_to_index(self, time, units="seconds", remove_gaps=True):
        """Convert elapsed time values to indices on the concatenated data axis."""
        values = np.asarray(time, dtype=float)
        if units == "milliseconds":
            values = values / 1e3
        elif units == "microseconds":
            values = values / 1e6
        elif units != "seconds":
            raise ValueError("units must be 'seconds', 'milliseconds', or 'microseconds'.")

        sample_rate = float(self.sample_rate)
        if remove_gaps:
            result = np.rint(values * sample_rate).astype(int)
        else:
            starts = np.asarray(self.start_times, dtype=float) - float(self.start_times[0])
            lengths = np.asarray([shape[1] for shape in self.shapes], dtype=int)
            last_sample_times = starts + (lengths - 1) / sample_rate
            start_indices = self.start_indices

            def convert_one(elapsed):
                for i, (start, stop) in enumerate(zip(starts, last_sample_times)):
                    if start <= elapsed <= stop:
                        return int(round(start_indices[i] + (elapsed - start) * sample_rate))
                    if i < len(starts) - 1 and stop < elapsed < starts[i + 1]:
                        fraction = (elapsed - stop) / (starts[i + 1] - stop)
                        return int(start_indices[i + 1] - 1 + round(fraction))
                if elapsed < starts[0]:
                    return 0
                return int(round(start_indices[-1] + (elapsed - starts[-1]) * sample_rate))

            flat = np.asarray([convert_one(value) for value in values.ravel()], dtype=int)
            result = flat.reshape(values.shape)

        return int(result) if result.ndim == 0 else result

    def _to_mne_raw(self):  # TODO make this work with all data if list of dask arrays
        """
        Loads raw data set into an array. This method is very computationally and memory intensive.

        Returns
        -------
        mne.acquisition_io.array.array.RawArray
            Mne array of the raw data set.

        """
        import mne

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sample_rate, ch_types='ecog')
        with ProgressBar():
            return mne.io.RawArray(np.asarray(self.array.compute()), info, verbose=False)

    def _to_plt_line_collection(self, x_lim, channels, d_l, down_sample=True, remove_gaps=True,
                                plot_arr=None, time_arr=None, x_index=None):
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
        if x_index is None:
            x_index = self._time_to_index(x_lim, remove_gaps=remove_gaps)
            x_index = (int(x_index[0]), int(x_index[1]) + 1)
        if time_arr is None or plot_arr is None:
            x_slice = slice(x_index[0], x_index[1])
            time_arr = self.time(remove_gaps=remove_gaps)[x_slice].compute()
            plot_arr = self.array[channels, x_slice].compute()

        # Down sample data if there more that 8x data points than the number of pixels
        # 8x sampling is arbitrary but found to have a good tradeoff between visual appearance and speed

        def to_line_array(plot_array, time):
            if down_sample and len(time) > d_l * 8:
                plot_data = np.zeros((plot_array.shape[0], d_l * 4, 2))
                # Simultaneously down_sample data and reshape array to the correct shape for a matplotlib LineCollection
                for channel in range(plot_array.shape[0]):
                    full_sampled = np.stack([plot_array[channel], time]).T
                    down_sampled = largest_triangle_three_buckets(full_sampled, d_l * 4)
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
        """Apply one offset vector to every component dataset."""
        offsets = np.asarray(ch_offsets, dtype=int).ravel()
        if np.any(offsets > 0):
            raise ValueError("Channel offsets must be zero or negative.")
        return [self._apply_channel_offsets(arr, offsets) for arr in self.data]

    def as_float32(self):
        """Return a lazily cast float32 copy."""
        data = [arr.astype(np.float32) if arr.dtype != np.float32 else arr for arr in self.data]
        return self._spawn(data=data)

    def select_channels(self, names_or_idx):
        """Return a lazy subset of channels in their original dataset order."""
        return self.remove_ch(names_or_idx, invert=True)

    def persist(self):
        """Persist component datasets and return a new same-type object."""
        return self._spawn(data=[arr.persist() for arr in self.data])

    def base_rechunk(self, time_chunk_samples: int | None = None):
        """Return a copy with full channel chunks and the requested time chunk size."""
        if time_chunk_samples is None:
            return self
        time_chunk_samples = int(time_chunk_samples)
        if time_chunk_samples <= 0:
            raise ValueError("time_chunk_samples must be positive.")
        data = [
            arr.rechunk({0: int(arr.shape[0]), 1: time_chunk_samples})
            for arr in self.data
        ]
        return self._spawn(data=data)

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        """Iterate lazily over channel rows."""
        return iter(self.array)

    def __getitem__(self, items):
        """Return a lazy Dask selection; call ``.compute()`` explicitly when needed."""
        return self.array[items]

