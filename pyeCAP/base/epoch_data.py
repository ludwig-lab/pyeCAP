from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
import re

import dask.array as da
from dask import delayed

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch

from .event_data import _EventData
from .parameter_data import _ParameterData
from .ts_data import _TsData
from .utils.visualization import (
    _plt_add_cbar_axis,
    _plt_setup_fig_axis,
    _plt_show_fig,
)


EPOCH_DATA_VERSION = "2026-08-05-max-filter-v4"

def _assemble_and_extract_epoch_cluster(
    chunk_grid: list[list[np.ndarray]],
    local_starts: np.ndarray,
    sample_length: int,
) -> np.ndarray:
    """Assemble source chunks and extract one cluster of epochs.

    Parameters
    ----------
    chunk_grid
        Nested source chunks arranged as ``[channel_chunk][time_chunk]``.
        Dask resolves each item to a NumPy array before this function runs.
    local_starts
        Epoch start indices relative to the assembled source block.
    sample_length
        Number of samples in each epoch.

    Returns
    -------
    np.ndarray
        Array shaped ``(pulses, channels, samples)``.
    """
    channel_blocks: list[np.ndarray] = []

    for time_chunks in chunk_grid:
        chunks = [np.asarray(chunk) for chunk in time_chunks]
        channel_block = (
            chunks[0]
            if len(chunks) == 1
            else np.concatenate(chunks, axis=1)
        )
        channel_blocks.append(channel_block)

    block = (
        channel_blocks[0]
        if len(channel_blocks) == 1
        else np.concatenate(channel_blocks, axis=0)
    )

    local_starts = np.asarray(local_starts, dtype=np.int64).reshape(-1)
    n_pulses = int(local_starts.size)
    n_channels = int(block.shape[0])

    if n_pulses == 0:
        return np.empty(
            (0, n_channels, sample_length),
            dtype=block.dtype,
        )

    # Fast path for consecutive, nonoverlapping epochs.
    expected_starts = (
        local_starts[0]
        + np.arange(n_pulses, dtype=np.int64) * sample_length
    )

    if (
        np.array_equal(local_starts, expected_starts)
        and int(local_starts[0]) == 0
        and block.shape[1] == n_pulses * sample_length
    ):
        return (
            block.reshape(n_channels, n_pulses, sample_length)
            .transpose(1, 0, 2)
        )

    output = np.empty(
        (n_pulses, n_channels, sample_length),
        dtype=block.dtype,
    )

    for pulse_index, start in enumerate(local_starts):
        start = int(start)
        stop = start + sample_length
        output[pulse_index] = block[:, start:stop]

    return output


@dataclass(slots=True)
class EpochQueryResult:
    """Epochs returned by a parameter-table query.

    Parameters
    ----------
    parameters
        Matching rows from ``_EpochData.parameters.parameters`` in the same
        order used by ``arrays``.
    arrays
        Mapping from parameter key to a lazy 3-D Dask array shaped
        ``(pulses, recording_channels, samples)``.
    filters
        Resolved parameter-table filters used to produce the result.

    Notes
    -----
    ``array`` returns a 3-D array when exactly one parameter matched. When
    multiple matching arrays have identical shapes, it stacks them as a 4-D
    array shaped ``(parameters, pulses, recording_channels, samples)``.
    Ragged matches remain available through ``arrays`` and can sometimes be
    combined along the pulse axis with ``concatenate_pulses()``.
    """

    parameters: pd.DataFrame
    arrays: dict[object, da.Array]
    filters: dict[object, object]

    def __len__(self) -> int:
        """Return the number of matching parameter rows."""
        return len(self.arrays)

    def __iter__(self):
        """Iterate over matching parameter keys."""
        return iter(self.arrays)

    def __getitem__(self, parameter: object) -> da.Array:
        """Return the 3-D array associated with one matching parameter key."""
        return self.arrays[parameter]

    @property
    def keys(self) -> tuple[object, ...]:
        """Return matching parameter keys in parameter-table order."""
        return tuple(self.arrays)

    @property
    def shapes(self) -> dict[object, tuple[int, ...]]:
        """Return the lazy array shape associated with every match."""
        return {
            key: tuple(int(size) for size in array.shape)
            for key, array in self.arrays.items()
        }

    @property
    def summary(self) -> pd.DataFrame:
        """Return matching parameter rows with epoch-shape information."""
        summary = self.parameters.copy()

        summary["epoch shape"] = [
            tuple(int(size) for size in self.arrays[key].shape)
            for key in summary.index
        ]
        summary["number of pulses"] = [
            int(self.arrays[key].shape[0])
            for key in summary.index
        ]
        summary["number of recording channels"] = [
            int(self.arrays[key].shape[1])
            for key in summary.index
        ]
        summary["number of samples"] = [
            int(self.arrays[key].shape[2])
            for key in summary.index
        ]

        return summary

    @property
    def is_stackable(self) -> bool:
        """Return whether all matching arrays can form one regular 4-D array."""
        if len(self.arrays) < 2:
            return True
        return len({tuple(array.shape) for array in self.arrays.values()}) == 1

    @property
    def array(self) -> da.Array:
        """Return one 3-D array or a stacked 4-D array.

        Returns
        -------
        da.Array
            One match produces ``(pulses, recording_channels, samples)``.
            Multiple equally shaped matches produce
            ``(parameters, pulses, recording_channels, samples)``.

        Raises
        ------
        ValueError
            If there are no matches or the matching arrays have different
            shapes. In that case, use ``arrays`` or ``concatenate_pulses()``.
        """
        arrays = list(self.arrays.values())

        if not arrays:
            raise ValueError("The query did not match any epoch arrays.")

        if len(arrays) == 1:
            return arrays[0]

        shape_counts = Counter(tuple(array.shape) for array in arrays)
        if len(shape_counts) != 1:
            summary = ", ".join(
                f"{count} parameter(s) with shape {shape}"
                for shape, count in sorted(shape_counts.items())
            )
            raise ValueError(
                "The matching epoch arrays have different shapes and cannot "
                "be stacked into one regular array: "
                f"{summary}. Use result.arrays or "
                "result.concatenate_pulses() instead."
            )

        return da.stack(arrays, axis=0)

    @property
    def data(self) -> da.Array:
        """Alias for ``array``."""
        return self.array

    def compute(self):
        """Compute ``array`` and return the resulting NumPy array."""
        return self.array.compute()

    def concatenate_pulses(self) -> tuple[da.Array, pd.DataFrame]:
        """Concatenate matches into one 3-D pulse array when possible.

        Returns
        -------
        array
            Lazy array shaped
            ``(all_pulses, recording_channels, samples)``.
        pulse_parameters
            One row per output pulse. The table contains the source parameter
            key, the original pulse index, and all matching parameter columns.

        Raises
        ------
        ValueError
            If there are no matches or the recording-channel/sample dimensions
            differ across matching arrays.
        """
        if not self.arrays:
            raise ValueError("The query did not match any epoch arrays.")

        trailing_shapes = {
            tuple(array.shape[1:])
            for array in self.arrays.values()
        }
        if len(trailing_shapes) != 1:
            raise ValueError(
                "The matching arrays cannot be concatenated because their "
                "recording-channel or sample dimensions differ. Use "
                "result.arrays instead."
            )

        combined = da.concatenate(list(self.arrays.values()), axis=0)
        pulse_rows: list[dict[str, object]] = []

        for parameter_key, array in self.arrays.items():
            parameter_values = self.parameters.loc[parameter_key].to_dict()

            for pulse_index in range(int(array.shape[0])):
                pulse_rows.append(
                    {
                        "parameter key": parameter_key,
                        "pulse index": pulse_index,
                        **parameter_values,
                    }
                )

        return combined, pd.DataFrame(pulse_rows)


class _EpochData:
    """Create and analyze event-aligned epochs from continuous time-series data.

    A single parameter produces a lazy Dask array with shape
    ``(pulses, channels, samples)``.
    """

    def __init__(
        self,
        ts_data: _TsData,
        event_data: _EventData,
        parameters: _ParameterData,
        epoch_window: tuple[float, float] | str | None = "auto",
        *,
        epoch_cluster_gap: int = 500_000,
    ) -> None:
        if not isinstance(ts_data, _TsData):
            raise TypeError("ts_data must be an instance of _TsData.")
        if not isinstance(event_data, _EventData):
            raise TypeError("event_data must be an instance of _EventData.")
        if not isinstance(parameters, _ParameterData):
            raise TypeError("parameters must be an instance of _ParameterData.")

        self.ts_data = ts_data
        self.event_data = event_data
        self.parameters = parameters

        self._epoch_cache: dict[object, da.Array] = {}
        self._sample_length_cache: dict[object, int] = {}
        self._event_index_cache: dict[object, np.ndarray] = {}
        self._persisted_array: da.Array | None = None

        self._epoch_cluster_gap = self._validate_cluster_gap(epoch_cluster_gap)
        self._epoch_window = self._validate_epoch_window(epoch_window)

        self.parameter_event_times = self._create_parameter_event_times()

    # ------------------------------------------------------------------
    # Configuration and cache management
    # ------------------------------------------------------------------

    @property
    def epoch_window(self) -> tuple[float, float] | str | None:
        return self._epoch_window

    @epoch_window.setter
    def epoch_window(self, value: tuple[float, float] | str | None) -> None:
        self._epoch_window = self._validate_epoch_window(value)
        if hasattr(self, "_epoch_cache"):
            self.clear_epoch_cache()

    @property
    def epoch_cluster_gap(self) -> int:
        """Maximum gap, in samples, between epochs grouped into one source block."""
        return self._epoch_cluster_gap

    @epoch_cluster_gap.setter
    def epoch_cluster_gap(self, value: int) -> None:
        self._epoch_cluster_gap = self._validate_cluster_gap(value)
        if hasattr(self, "_epoch_cache"):
            self._epoch_cache.clear()

    @staticmethod
    def _validate_cluster_gap(value: int) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("epoch_cluster_gap must be nonnegative.")
        return value

    @staticmethod
    def _validate_epoch_window(
        value: tuple[float, float] | str | None,
    ) -> tuple[float, float] | str | None:
        if value is None:
            return None

        if isinstance(value, str):
            if value == "auto":
                return value
            raise ValueError("epoch_window must be 'auto', None, or (start, end).")

        try:
            start, end = value
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "epoch_window must contain exactly two values: (start, end)."
            ) from exc

        start = float(start)
        end = float(end)

        if not np.isfinite([start, end]).all():
            raise ValueError("epoch_window values must be finite.")
        if end <= start:
            raise ValueError(
                f"epoch_window must satisfy end > start; received {value!r}."
            )

        return start, end

    def clear_epoch_cache(self) -> None:
        """Clear cached epoch graphs and cached sample lengths."""
        self._epoch_cache.clear()
        self._sample_length_cache.clear()

    def refresh_event_times(self) -> None:
        """Rebuild event-time mappings after ``event_data`` changes."""
        self.parameter_event_times = self._create_parameter_event_times()
        self._event_index_cache.clear()
        self.clear_epoch_cache()

    def persist_ts(self, time_chunk: int | None = None) -> _EpochData:
        """Persist the continuous source array for faster repeated epoch access.

        Notes
        -----
        Persisting the complete recording may consume substantial memory.
        """
        source = self.ts_data.array
        if not isinstance(source, da.Array):
            source = da.asarray(source)

        if time_chunk is not None:
            time_chunk = int(time_chunk)
            if time_chunk <= 0:
                raise ValueError("time_chunk must be a positive integer.")
            source = source.rechunk({1: time_chunk})

        self._persisted_array = source.persist()
        self._epoch_cache.clear()
        return self

    def clear_persisted_ts(self) -> None:
        """Return to reading epochs from ``ts_data.array``."""
        self._persisted_array = None
        self._epoch_cache.clear()

    # ------------------------------------------------------------------
    # Parameter normalization
    # ------------------------------------------------------------------

    @property
    def _parameter_index(self) -> pd.Index:
        return self.parameters.parameters.index

    def _normalize_parameter_key(self, parameter: object) -> object:
        """Validate one parameter key and return its hashable representation."""
        index = self._parameter_index

        if isinstance(index, pd.MultiIndex):
            if isinstance(parameter, list):
                raise TypeError(
                    "A single MultiIndex parameter must be a tuple, such as "
                    "(0, 1), not a list such as [0, 1]."
                )
            if not isinstance(parameter, tuple):
                raise TypeError(
                    f"A parameter key must be a {index.nlevels}-item tuple."
                )
            if len(parameter) != index.nlevels:
                raise ValueError(
                    f"Expected a {index.nlevels}-item parameter tuple; "
                    f"received {parameter!r}."
                )
            key = parameter
        else:
            if isinstance(parameter, (list, dict, set, np.ndarray)):
                raise TypeError("A single parameter key must be hashable.")
            key = parameter

        try:
            exists = key in index
        except TypeError as exc:
            raise TypeError(f"Parameter key must be hashable; received {key!r}.") from exc

        if not exists:
            raise KeyError(f"Unknown parameter key: {key!r}")

        return key

    def _normalize_parameter_keys(self, parameters: object = None) -> list[object]:
        """Normalize ``None``, ``'all'``, one key, or an iterable of keys."""
        index = self._parameter_index

        if parameters is None:
            return list(index)

        if isinstance(parameters, str):
            if parameters == "all":
                return list(index)
            raise ValueError("The only accepted string is parameters='all'.")

        if isinstance(index, pd.MultiIndex):
            if isinstance(parameters, tuple):
                if len(parameters) == 0:
                    return []

                # A tuple of tuples is interpreted as multiple MultiIndex keys.
                if all(isinstance(key, tuple) for key in parameters):
                    keys = list(parameters)
                else:
                    # An unknown single tuple should raise KeyError rather than
                    # being incorrectly split into scalar keys.
                    return [self._normalize_parameter_key(parameters)]
            else:
                try:
                    keys = list(parameters)
                except TypeError as exc:
                    raise TypeError(
                        "A MultiIndex selection must be one tuple key or an "
                        "iterable of tuple keys."
                    ) from exc
        else:
            # Scalar keys are valid for a regular Index.
            if not isinstance(parameters, (list, tuple, set, np.ndarray, pd.Index)):
                return [self._normalize_parameter_key(parameters)]

            # A tuple can itself be a valid key in a regular Index.
            try:
                if isinstance(parameters, tuple) and parameters in index:
                    return [self._normalize_parameter_key(parameters)]
            except TypeError:
                pass

            keys = list(parameters)

        return [self._normalize_parameter_key(key) for key in keys]

    # ------------------------------------------------------------------
    # Event times and epoch geometry
    # ------------------------------------------------------------------

    def _create_parameter_event_times(self) -> dict[object, np.ndarray]:
        """Map each valid parameter key to sorted event times in seconds.

        This method intentionally avoids calling ``_normalize_parameter_key``
        for every pulse. Repeated pandas MultiIndex membership checks inside
        the event loop are very expensive for large recordings.
        """
        parameter_times: dict[object, list[float]] = {}
        valid_keys = set(self._parameter_index.tolist())
        start_times = (
            np.asarray(self.ts_data.start_indices, dtype=float)
            / float(self.ts_data.sample_rate)
        )

        for channel in self.event_data.ch_names:
            events = self.event_data.events(channel, start_times=start_times)
            indicators = self.event_data.event_indicators(channel)

            try:
                n_events = len(events)
                n_indicators = len(indicators)
            except TypeError:
                events = list(events)
                indicators = list(indicators)
                n_events = len(events)
                n_indicators = len(indicators)

            if n_events != n_indicators:
                raise ValueError(
                    f"Event/indicator length mismatch for channel {channel!r}: "
                    f"{n_events} events and {n_indicators} indicators."
                )

            for indicator, event_time in zip(indicators, events):
                if isinstance(indicator, tuple):
                    key = indicator
                else:
                    try:
                        key = tuple(indicator)
                    except TypeError:
                        key = indicator

                if key not in valid_keys:
                    continue

                event_time = float(event_time)
                if np.isfinite(event_time):
                    parameter_times.setdefault(key, []).append(event_time)

        return {
            key: np.sort(np.asarray(times, dtype=float))
            for key, times in parameter_times.items()
        }

    def _event_times(self, parameter_key: object) -> np.ndarray:
        """Return already-sorted finite event times for one parameter."""
        event_times = self.parameter_event_times.get(parameter_key)
        if event_times is None:
            return np.empty(0, dtype=float)

        event_times = np.asarray(event_times, dtype=float).reshape(-1)
        if np.isfinite(event_times).all():
            return event_times
        return event_times[np.isfinite(event_times)]

    def _event_indices(self, parameter_key: object) -> np.ndarray:
        """Return sorted event sample indices, cached per parameter."""
        cached = self._event_index_cache.get(parameter_key)
        if cached is not None:
            return cached

        event_times = self._event_times(parameter_key)
        if event_times.size == 0:
            indices = np.empty(0, dtype=np.int64)
        else:
            indices = np.asarray(
                self.ts_data._time_to_index(event_times),
                dtype=np.int64,
            ).reshape(-1)
            indices.sort()

        self._event_index_cache[parameter_key] = indices
        return indices

    def _epoch_start_sample(self) -> int:
        if self.epoch_window is None or self.epoch_window == "auto":
            return 0

        sample_rate = float(self.ts_data.sample_rate)
        return int(np.rint(self.epoch_window[0] * sample_rate))

    def epoch_sample_length(self, parameter: object) -> int:
        """Return samples per epoch without constructing the Dask graph."""
        parameter_key = self._normalize_parameter_key(parameter)

        if parameter_key in self._sample_length_cache:
            return self._sample_length_cache[parameter_key]

        sample_rate = float(self.ts_data.sample_rate)

        if self.epoch_window is not None and self.epoch_window != "auto":
            start_sample = int(np.rint(self.epoch_window[0] * sample_rate))
            stop_sample = int(np.rint(self.epoch_window[1] * sample_rate))
            sample_length = stop_sample - start_sample

            if sample_length < 1:
                raise ValueError(
                    "epoch_window is shorter than one sample at the current "
                    f"sample rate ({sample_rate:g} Hz)."
                )
        else:
            event_times = self._event_times(parameter_key)

            if event_times.size == 0:
                sample_length = 0
            elif event_times.size == 1:
                sample_length = 1
            else:
                event_indices = self._event_indices(parameter_key)
                positive_differences = np.diff(event_indices)
                positive_differences = positive_differences[positive_differences > 0]
                sample_length = (
                    int(positive_differences.min())
                    if positive_differences.size
                    else 1
                )

        self._sample_length_cache[parameter_key] = sample_length
        return sample_length

    def time_axis(self, parameter: object) -> np.ndarray:
        """Return the relative epoch time axis in seconds."""
        parameter_key = self._normalize_parameter_key(parameter)
        sample_length = self.epoch_sample_length(parameter_key)
        sample_rate = float(self.ts_data.sample_rate)
        start_sample = self._epoch_start_sample()

        return (
            start_sample + np.arange(sample_length, dtype=np.int64)
        ) / sample_rate

    def _time_window_to_indices(
        self,
        parameter: object,
        window_s: tuple[float, float],
    ) -> tuple[int, int]:
        """Convert a relative time window to clipped, half-open sample indices."""
        try:
            start_s, stop_s = map(float, window_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("window_s must be a two-value (start, stop) tuple.") from exc

        if stop_s <= start_s:
            raise ValueError("window_s must satisfy stop > start.")

        time = self.time_axis(parameter)
        if time.size == 0:
            raise ValueError(f"Parameter {parameter!r} has an empty epoch.")

        sample_rate = float(self.ts_data.sample_rate)
        epoch_start = float(time[0])
        epoch_stop = float(time[-1] + 1.0 / sample_rate)

        clipped_start = max(start_s, epoch_start)
        clipped_stop = min(stop_s, epoch_stop)

        if clipped_stop <= clipped_start:
            raise ValueError(
                f"Requested window {window_s!r} does not overlap the epoch "
                f"window ({epoch_start:g}, {epoch_stop:g})."
            )

        index_start = int(np.searchsorted(time, clipped_start, side="left"))
        index_stop = int(np.searchsorted(time, clipped_stop, side="left"))
        index_stop = min(index_stop, time.size)

        if index_stop <= index_start:
            raise ValueError("The selected window contains no samples.")

        return index_start, index_stop

    # ------------------------------------------------------------------
    # Lazy epoch construction and selection
    # ------------------------------------------------------------------

    def _empty_epoch_array(self, sample_length: int) -> da.Array:
        return da.zeros(
            (0, int(self.ts_data.shape[0]), sample_length),
            dtype=self.ts_data.dtype,
        )

    def _construct_epoch_array(self, parameter_key: object) -> da.Array:
        """Construct one lazy ``(pulses, channels, samples)`` array."""
        sample_length = self.epoch_sample_length(parameter_key)
        event_indices = self._event_indices(parameter_key)

        if event_indices.size == 0 or sample_length == 0:
            return self._empty_epoch_array(sample_length)

        starts = event_indices + self._epoch_start_sample()
        stops = starts + sample_length
        n_time = int(self.ts_data.shape[1])

        valid = (starts >= 0) & (stops <= n_time)
        starts = np.sort(starts[valid])

        if starts.size == 0:
            return self._empty_epoch_array(sample_length)

        source = (
            self._persisted_array
            if self._persisted_array is not None
            else self.ts_data.array
        )
        if not isinstance(source, da.Array):
            source = da.asarray(source)

        split_indices = np.flatnonzero(
            np.diff(starts) > self.epoch_cluster_gap
        ) + 1
        clusters = np.split(starts, split_indices)

        cluster_arrays = []

        for cluster_starts in clusters:
            block_start = int(cluster_starts[0])
            block_stop = int(cluster_starts[-1] + sample_length)

            block = source[:, block_start:block_stop]

            local_starts = (
                    cluster_starts - block_start
            ).astype(np.int64)

            # Do not pass the Dask array directly to delayed().
            # Instead, expose its individual chunks as Delayed objects.
            block_chunks = block.to_delayed(
                optimize_graph=False,
            ).tolist()

            delayed_cluster = delayed(
                _assemble_and_extract_epoch_cluster,
                pure=False,
            )(
                block_chunks,
                local_starts,
                sample_length,
            )

            cluster_array = da.from_delayed(
                delayed_cluster,
                shape=(
                    len(cluster_starts),
                    int(self.ts_data.shape[0]),
                    sample_length,
                ),
                dtype=self.ts_data.dtype,
            )

            cluster_arrays.append(cluster_array)

        if len(cluster_arrays) == 1:
            return cluster_arrays[0]

        return da.concatenate(cluster_arrays, axis=0)

    def dask_array(self, parameter: object) -> da.Array:
        """Return the cached lazy epoch array for one parameter."""
        parameter_key = self._normalize_parameter_key(parameter)

        if parameter_key not in self._epoch_cache:
            self._epoch_cache[parameter_key] = self._construct_epoch_array(parameter_key)

        return self._epoch_cache[parameter_key]

    @staticmethod
    def _preserve_axis(index: object, axis_size: int | None = None) -> object:
        """Preserve a dimension without triggering one-item fancy indexing."""
        if not isinstance(index, (Integral, np.integer)):
            return index

        index = int(index)
        if axis_size is not None:
            if index < 0:
                index += int(axis_size)
            if index < 0 or index >= int(axis_size):
                raise IndexError("index is out of bounds")

        return slice(index, index + 1)

    def _channel_index(self, channels: object = None) -> object:
        if channels is None:
            return slice(None)
        return self._preserve_axis(
            self.ts_data._ch_to_index(channels),
            axis_size=int(self.ts_data.shape[0]),
        )

    @staticmethod
    def _normalize_parameter_column_name(name: object) -> str:
        """Convert a parameter column name to a Python-friendly identifier.

        Examples
        --------
        ``"pulse amplitude (μA)"`` becomes ``"pulse_amplitude"``.
        ``"Channel Name"`` becomes ``"channel_name"``.
        """
        normalized = str(name).strip().lower()
        normalized = normalized.replace("μ", "u").replace("µ", "u")

        # Keyword arguments cannot conveniently include units, so remove one
        # or more parenthesized unit/annotation groups before snake-casing.
        normalized = re.sub(r"\([^)]*\)", "", normalized)
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return normalized

    def _parameter_column_lookup(self) -> dict[str, list[object]]:
        """Return normalized parameter names mapped to real table columns."""
        lookup: dict[str, list[object]] = {}

        for column in self.parameters.parameters.columns:
            normalized = self._normalize_parameter_column_name(column)
            lookup.setdefault(normalized, []).append(column)

        return lookup

    def _resolve_parameter_column(self, requested_name: str) -> object:
        """Resolve a query keyword or exact name to a parameter-table column."""
        table = self.parameters.parameters

        # Exact names supplied through parameter_filters take precedence.
        if requested_name in table.columns:
            return requested_name

        requested = self._normalize_parameter_column_name(requested_name)
        lookup = self._parameter_column_lookup()

        direct_matches = lookup.get(requested, [])
        if len(direct_matches) == 1:
            return direct_matches[0]
        if len(direct_matches) > 1:
            raise KeyError(
                f"Parameter filter {requested_name!r} is ambiguous. It matches "
                f"columns {direct_matches!r}. Use the exact column name in "
                "parameter_filters={...}."
            )

        aliases: dict[str, tuple[str, ...]] = {
            "amplitude": (
                "pulse_amplitude",
                "stimulation_amplitude",
                "stim_amplitude",
            ),
            "pulse_amplitude": (
                "pulse_amplitude",
                "stimulation_amplitude",
                "stim_amplitude",
                "amplitude",
            ),
            "stim_amplitude": (
                "stimulation_amplitude",
                "pulse_amplitude",
                "stim_amplitude",
                "amplitude",
            ),
            "channel": (
                "channel_name",
                "stimulation_channel",
                "stim_channel",
                "channel",
            ),
            "stim_channel": (
                "stimulation_channel",
                "stim_channel",
                "channel_name",
                "channel",
            ),
            "stimulation_channel": (
                "stimulation_channel",
                "stim_channel",
                "channel_name",
                "channel",
            ),
        }

        alias_matches: list[object] = []
        for alias in aliases.get(requested, ()):
            alias_matches.extend(lookup.get(alias, []))

        # Preserve table order while removing duplicates.
        alias_matches = list(dict.fromkeys(alias_matches))

        if len(alias_matches) == 1:
            return alias_matches[0]
        if len(alias_matches) > 1:
            raise KeyError(
                f"Parameter filter {requested_name!r} is ambiguous. It could "
                f"refer to columns {alias_matches!r}. Use the exact column "
                "name in parameter_filters={...}."
            )

        available = sorted(lookup)
        raise KeyError(
            f"No parameter column matches {requested_name!r}. Available "
            f"query names are: {available}"
        )

    @staticmethod
    def _parameter_filter_mask(
        series: pd.Series,
        expected: object,
        *,
        case_sensitive: bool,
    ) -> pd.Series:
        """Return a Boolean mask for one parameter-column filter.

        Scalar values use equality. Lists, sets, NumPy arrays, and pandas
        indexes use membership. A slice represents an inclusive range.
        String matching is case-insensitive unless ``case_sensitive=True``.
        """
        if expected is None:
            return series.isna()

        if isinstance(expected, slice):
            mask = pd.Series(True, index=series.index, dtype=bool)

            if expected.start is not None:
                mask &= series >= expected.start
            if expected.stop is not None:
                mask &= series <= expected.stop

            return mask.fillna(False)

        collection_types = (list, set, np.ndarray, pd.Index)
        if isinstance(expected, collection_types):
            expected_values = list(expected)

            if (
                not case_sensitive
                and expected_values
                and all(isinstance(value, str) for value in expected_values)
            ):
                expected_folded = {
                    value.strip().casefold()
                    for value in expected_values
                }
                values = series.astype("string").str.strip().str.casefold()
                return values.isin(expected_folded).fillna(False)

            return series.isin(expected_values).fillna(False)

        if isinstance(expected, str) and not case_sensitive:
            values = series.astype("string").str.strip().str.casefold()
            return values.eq(expected.strip().casefold()).fillna(False)

        return series.eq(expected).fillna(False)

    def _filter_parameter_table(
        self,
        filters: Mapping[str, object],
        *,
        case_sensitive: bool,
    ) -> tuple[pd.DataFrame, dict[object, object]]:
        """Filter the parameter table and return resolved column names.

        The special string ``"max"`` is evaluated after all ordinary filters.
        Therefore, ``condition="Intact", pulse_amplitude="max"`` selects the
        highest numeric pulse amplitude available within the intact subset.
        All rows tied at that maximum are retained.
        """
        table = self.parameters.parameters

        if not table.index.is_unique:
            raise ValueError(
                "The parameter table index must be unique before epochs can "
                "be queried by parameter values."
            )

        mask = pd.Series(True, index=table.index, dtype=bool)
        resolved_filters: dict[object, object] = {}
        maximum_filters: list[object] = []

        # Resolve every requested keyword first so duplicate aliases are caught
        # before any filtering is performed.
        for requested_name, expected_value in filters.items():
            column = self._resolve_parameter_column(requested_name)

            if column in resolved_filters:
                raise TypeError(
                    f"Multiple query arguments resolve to parameter column "
                    f"{column!r}."
                )

            resolved_filters[column] = expected_value

            if (
                isinstance(expected_value, str)
                and expected_value.strip().casefold() == "max"
            ):
                maximum_filters.append(column)
                continue

            mask &= self._parameter_filter_mask(
                table[column],
                expected_value,
                case_sensitive=case_sensitive,
            )

        # Evaluate maxima only within the subset produced by the ordinary
        # filters. This is what makes the selector condition/channel aware.
        for column in maximum_filters:
            candidate_source = table.loc[mask, column]
            candidate_values = pd.to_numeric(
                candidate_source,
                errors="coerce",
            )

            # Also accept values stored as strings with commas or units, such
            # as "2,000", "2000 uA", or "2000 μA".
            if candidate_values.isna().any():
                extracted = (
                    candidate_source.astype("string")
                    .str.replace(",", "", regex=False)
                    .str.extract(
                        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                        expand=False,
                    )
                )
                extracted_numeric = pd.to_numeric(extracted, errors="coerce")
                candidate_values = candidate_values.fillna(extracted_numeric)

            valid_values = candidate_values.dropna()

            if valid_values.empty:
                ordinary_description = ", ".join(
                    f"{name!r}={value!r}"
                    for name, value in resolved_filters.items()
                    if not (
                        isinstance(value, str)
                        and value.strip().casefold() == "max"
                    )
                )
                subset_description = (
                    f" after applying {ordinary_description}"
                    if ordinary_description
                    else ""
                )
                raise ValueError(
                    f"Cannot select the maximum of parameter column "
                    f"{column!r}{subset_description}: the matching rows "
                    "contain no numeric values."
                )

            maximum_value = valid_values.max()
            maximum_mask = pd.Series(False, index=table.index, dtype=bool)
            maximum_mask.loc[candidate_values.index] = candidate_values.eq(
                maximum_value
            )
            mask &= maximum_mask

        return table.loc[mask].copy(), resolved_filters

    def _select_epoch_array(
        self,
        parameter: object,
        *,
        pulses: object = None,
        recording_channels: object = None,
        samples: object = None,
    ) -> da.Array:
        """Select one parameter while preserving all three epoch dimensions."""
        array = self.dask_array(parameter)

        if pulses is not None:
            array = array[
                self._preserve_axis(pulses, axis_size=int(array.shape[0])),
                :,
                :,
            ]

        array = array[:, self._channel_index(recording_channels), :]

        if samples is not None:
            array = array[
                :,
                :,
                self._preserve_axis(samples, axis_size=int(array.shape[2])),
            ]

        return array

    def query_epochs(
        self,
        *,
        pulses: object = None,
        recording_channels: object = None,
        samples: object = None,
        parameter_filters: Mapping[str, object] | None = None,
        case_sensitive: bool = False,
        **filters: object,
    ) -> EpochQueryResult:
        """Query epochs using values from columns in the parameter table.

        Parameters
        ----------
        pulses
            Optional pulse-axis selection applied to every matching array.
        recording_channels
            Optional recorded-signal channel selection. This is separate from
            a parameter filter such as ``channel="Channel 1"``, which selects
            the stimulated channel/contact stored in the parameter table.
        samples
            Optional sample-axis selection applied to every matching array.
        parameter_filters
            Optional mapping of parameter-table column names to desired values.
            Use this form for exact column names containing spaces or units,
            for example ``{"pulse amplitude (μA)": 2000}``.
        case_sensitive
            Whether string-valued filters should be case-sensitive.
        **filters
            Python-friendly parameter filters such as
            ``condition="Intact"``, ``pulse_amplitude=2000``, or
            ``channel="Channel 1"``.

        Returns
        -------
        EpochQueryResult
            Matching parameter rows and one lazy 3-D epoch array per row.
        """
        combined_filters: dict[str, object] = {}

        if parameter_filters is not None:
            combined_filters.update(dict(parameter_filters))

        overlap = set(combined_filters).intersection(filters)
        if overlap:
            raise TypeError(
                "The following parameter filters were supplied twice: "
                f"{sorted(overlap)}"
            )

        combined_filters.update(filters)
        if not combined_filters:
            raise TypeError(
                "query_epochs() requires at least one parameter-table filter."
            )

        matched_parameters, resolved_filters = self._filter_parameter_table(
            combined_filters,
            case_sensitive=case_sensitive,
        )

        if matched_parameters.empty:
            description = ", ".join(
                f"{name}={value!r}"
                for name, value in combined_filters.items()
            )
            raise KeyError(f"No parameter rows matched: {description}")

        arrays = {
            parameter_key: self._select_epoch_array(
                parameter_key,
                pulses=pulses,
                recording_channels=recording_channels,
                samples=samples,
            )
            for parameter_key in matched_parameters.index
        }

        return EpochQueryResult(
            parameters=matched_parameters,
            arrays=arrays,
            filters=resolved_filters,
        )

    def epoch(
        self,
        parameter: object = None,
        *,
        pulses: object = None,
        channels: object = None,
        recording_channels: object = None,
        samples: object = None,
        parameter_filters: Mapping[str, object] | None = None,
        case_sensitive: bool = False,
        **filters: object,
    ) -> da.Array | EpochQueryResult:
        """Return one parameter's epochs or query epochs by parameter values.

        Single-parameter mode
        ---------------------
        Passing ``parameter`` returns one lazy 3-D Dask array shaped
        ``(pulses, recording_channels, samples)``. The existing ``channels``
        argument remains an alias for ``recording_channels``.

        Query mode
        ----------
        Omit ``parameter`` and provide one or more parameter-table filters.
        Query mode returns :class:`EpochQueryResult`.

        Examples
        --------
        Select one existing parameter key::

            epochs = data.epoch((0, 0), channels="LIFE 1")

        Query one exact stimulation configuration::

            result = data.epoch(
                condition="Intact",
                pulse_amplitude=2000,
                channel="Channel 1",
            )
            epochs = result.array

        Query all amplitudes for one stimulation channel::

            result = data.epoch(
                condition="Intact",
                channel="Channel 1",
            )
            print(result.summary)
        """
        if channels is not None and recording_channels is not None:
            raise TypeError(
                "Pass either channels or recording_channels, not both."
            )

        if recording_channels is None:
            recording_channels = channels

        query_requested = parameter_filters is not None or bool(filters)

        if query_requested:
            if parameter is not None:
                raise TypeError(
                    "Pass either one parameter key or parameter-table filters, "
                    "not both."
                )

            return self.query_epochs(
                pulses=pulses,
                recording_channels=recording_channels,
                samples=samples,
                parameter_filters=parameter_filters,
                case_sensitive=case_sensitive,
                **filters,
            )

        if parameter is None:
            raise TypeError(
                "epoch() requires either a parameter key or at least one "
                "parameter-table filter."
            )

        return self._select_epoch_array(
            parameter,
            pulses=pulses,
            recording_channels=recording_channels,
            samples=samples,
        )

    def epoch_arrays(
        self,
        parameters: object = None,
        *,
        pulses: object = None,
        channels: object = None,
        samples: object = None,
    ) -> dict[object, da.Array]:
        """Return one selected 3-D epoch array per parameter."""
        return {
            key: self._select_epoch_array(
                key,
                pulses=pulses,
                recording_channels=channels,
                samples=samples,
            )
            for key in self._normalize_parameter_keys(parameters)
        }

    def epoch_4d(
        self,
        parameters: object = None,
        *,
        pulses: object = None,
        channels: object = None,
        samples: object = None,
    ) -> da.Array:
        """Stack matching parameter arrays as ``(parameters, pulses, channels, samples)``."""
        arrays = self.epoch_arrays(
            parameters,
            pulses=pulses,
            channels=channels,
            samples=samples,
        )

        if not arrays:
            raise ValueError("No parameters were selected.")

        shape_counts = Counter(tuple(array.shape) for array in arrays.values())
        if len(shape_counts) != 1:
            summary = ", ".join(
                f"{count} parameter(s) with shape {shape}"
                for shape, count in sorted(shape_counts.items())
            )
            raise ValueError(
                "Selected parameters cannot be stacked because their shapes differ: "
                f"{summary}. Use epoch_arrays() instead."
            )

        return da.stack(list(arrays.values()), axis=0)

    # ------------------------------------------------------------------
    # Pulse reductions
    # ------------------------------------------------------------------

    def _reduce_pulses(
        self,
        parameter: object,
        *,
        channels: object = None,
        method: str,
    ) -> da.Array:
        array = self.epoch(parameter, channels=channels)

        if array.shape[0] == 0:
            raise ValueError(f"Parameter {parameter!r} contains no valid pulses.")

        if method == "mean":
            return array.mean(axis=0, split_every=8)
        if method == "median":
            return da.median(array, axis=0)
        if method == "std":
            return array.std(axis=0)

        raise ValueError("method must be 'mean', 'median', or 'std'.")

    def mean_waveform(self, parameter: object, channels: object = None) -> da.Array:
        """Return the pulse mean with shape ``(channels, samples)``."""
        return self._reduce_pulses(parameter, channels=channels, method="mean")

    def median_waveform(self, parameter: object, channels: object = None) -> da.Array:
        """Return the pulse median with shape ``(channels, samples)``."""
        return self._reduce_pulses(parameter, channels=channels, method="median")

    def std_waveform(self, parameter: object, channels: object = None) -> da.Array:
        """Return the pulse standard deviation with shape ``(channels, samples)``."""
        return self._reduce_pulses(parameter, channels=channels, method="std")

    def mean_waveforms(
        self,
        parameters: object = None,
        *,
        channels: object = None,
        samples: object = None,
    ) -> da.Array:
        """Stack pulse means as ``(parameters, channels, samples)``."""
        means: list[da.Array] = []

        for key in self._normalize_parameter_keys(parameters):
            mean = self.mean_waveform(key, channels=channels)
            if samples is not None:
                mean = mean[
                    :,
                    self._preserve_axis(samples, axis_size=int(mean.shape[1])),
                ]
            means.append(mean)

        if not means:
            channel_index = self._channel_index(channels)
            selected_channels = np.arange(
                int(self.ts_data.shape[0]),
                dtype=np.int64,
            )[channel_index]
            n_channels = int(np.asarray(selected_channels).size)
            return da.zeros((0, n_channels, 0), dtype=self.ts_data.dtype)

        shapes = {tuple(mean.shape) for mean in means}
        if len(shapes) != 1:
            raise ValueError(
                "Mean waveform shapes differ across parameters: "
                f"{sorted(shapes)}"
            )

        return da.stack(means, axis=0)

    # ------------------------------------------------------------------
    # Windowed measurements
    # ------------------------------------------------------------------

    def _mean_segment(
        self,
        parameter: object,
        *,
        channels: object,
        window_s: tuple[float, float],
        baseline_s: tuple[float, float] | None,
    ) -> da.Array:
        waveform = self.mean_waveform(parameter, channels=channels)

        if baseline_s is not None:
            baseline_start, baseline_stop = self._time_window_to_indices(
                parameter,
                baseline_s,
            )
            baseline = waveform[:, baseline_start:baseline_stop].mean(
                axis=1,
                keepdims=True,
            )
            waveform = waveform - baseline

        window_start, window_stop = self._time_window_to_indices(
            parameter,
            window_s,
        )
        return waveform[:, window_start:window_stop]

    def rms_in_window(
        self,
        parameter: object,
        channels: object = None,
        window_s: tuple[float, float] = (0.0, 0.008),
        baseline_s: tuple[float, float] | None = None,
    ) -> da.Array:
        """Return channel-wise RMS of the pulse-mean waveform in ``window_s``."""
        segment = self._mean_segment(
            parameter,
            channels=channels,
            window_s=window_s,
            baseline_s=baseline_s,
        )
        return da.sqrt(da.mean(segment**2, axis=1))

    def auc_in_window(
        self,
        parameter: object,
        channels: object = None,
        window_s: tuple[float, float] = (0.0, 0.008),
        baseline_s: tuple[float, float] | None = None,
        *,
        absolute: bool = False,
    ) -> da.Array:
        """Return channel-wise trapezoidal AUC of the pulse-mean waveform."""
        segment = self._mean_segment(
            parameter,
            channels=channels,
            window_s=window_s,
            baseline_s=baseline_s,
        )

        if segment.shape[1] < 2:
            raise ValueError("AUC requires at least two samples in window_s.")

        if absolute:
            segment = da.absolute(segment)

        dt = 1.0 / float(self.ts_data.sample_rate)
        return da.sum(
            (segment[:, :-1] + segment[:, 1:]) * (0.5 * dt),
            axis=1,
        )

    # ------------------------------------------------------------------
    # Other analyses
    # ------------------------------------------------------------------

    def welch_psd_parameter(
        self,
        parameter: object,
        *,
        channels: object = None,
        nperseg: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Welch PSD over the parameter's continuous onset/offset interval."""
        parameter_key = self._normalize_parameter_key(parameter)
        row = self.parameters.parameters.loc[parameter_key]

        required = {"onset time (s)", "offset time (s)"}
        missing = required.difference(row.index)
        if missing:
            raise KeyError(
                "Parameter table is missing required columns: "
                f"{sorted(missing)}"
            )

        start_index, stop_index = np.asarray(
            self.ts_data._time_to_index(
                [row["onset time (s)"], row["offset time (s)"]]
            ),
            dtype=np.int64,
        )

        if stop_index <= start_index:
            raise ValueError(
                "Parameter offset must occur after its onset; received "
                f"indices ({start_index}, {stop_index})."
            )

        source = (
            self._persisted_array
            if self._persisted_array is not None
            else self.ts_data.array
        )
        if not isinstance(source, da.Array):
            source = da.asarray(source)

        data = source[:, start_index:stop_index]
        data = data[self._channel_index(channels), :].compute()
        data = data - np.mean(data, axis=-1, keepdims=True)

        sample_rate = float(self.ts_data.sample_rate)
        if nperseg is None:
            nperseg = min(int(sample_rate), data.shape[-1])
        else:
            nperseg = min(int(nperseg), data.shape[-1])

        if nperseg < 1:
            raise ValueError("The selected parameter interval contains no samples.")

        return welch(
            data,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
            scaling="density",
            axis=-1,
        )

    def event_sample_indices(self) -> pd.DataFrame:
        """Return event onset/offset times converted to integer sample indices."""
        columns = ["onset time (s)", "offset time (s)"]
        missing = set(columns).difference(self.event_data.parameters.columns)
        if missing:
            raise KeyError(
                "Event table is missing required columns: "
                f"{sorted(missing)}"
            )

        result = self.event_data.parameters.loc[:, columns].copy()
        result = np.rint(result * float(self.ts_data.sample_rate)).astype(np.int64)
        return result.rename(
            columns={
                "onset time (s)": "onset index",
                "offset time (s)": "offset index",
            }
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_channel(
        self,
        channel: object,
        parameters: object,
        *args,
        method: str = "mean",
        axis=None,
        x_lim=None,
        y_lim="auto",
        colors=None,
        fig_size=(10, 3),
        show=True,
        **kwargs,
    ):
        """Plot one channel's mean, median, or standard-deviation waveform."""
        parameter_keys = self._normalize_parameter_keys(parameters)
        if not parameter_keys:
            raise ValueError("No parameters were selected.")

        if colors is None:
            colors = list(sns.color_palette(n_colors=len(parameter_keys)))
        elif isinstance(colors, str):
            colors = [colors]
        else:
            colors = list(colors)

        if len(colors) < len(parameter_keys):
            raise ValueError(
                f"Received {len(colors)} color(s) for "
                f"{len(parameter_keys)} parameter(s)."
            )

        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        y_scale = 0.0
        time_min = np.inf
        time_max = -np.inf

        for parameter_key, color in zip(parameter_keys, colors):
            data = self._reduce_pulses(
                parameter_key,
                channels=channel,
                method=method,
            ).compute()

            if data.shape[0] != 1:
                raise ValueError("plot_channel() requires exactly one channel.")

            time = self.time_axis(parameter_key)
            ax.plot(time, data[0], *args, color=color, **kwargs)

            y_scale = max(y_scale, float(np.std(data[0])))
            time_min = min(time_min, float(time[0]))
            time_max = max(time_max, float(time[-1]))

        if y_lim is None or y_lim == "auto":
            if y_scale > 0:
                ax.set_ylim(-6.0 * y_scale, 6.0 * y_scale)
        elif y_lim != "max":
            ax.set_ylim(np.asarray(y_lim, dtype=float))

        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (V)")
        ax.set_xlim((time_min, time_max) if x_lim is None else x_lim)

        return _plt_show_fig(fig, ax, show)

    def plot_raster(
        self,
        channel: object,
        parameters: object,
        *args,
        method: str = "mean",
        axis=None,
        x_lim=None,
        c_lim="auto",
        c_map="RdYlBu",
        fig_size=(10, 4),
        show=True,
        **kwargs,
    ):
        """Plot one waveform row per parameter for a single channel."""
        parameter_keys = self._normalize_parameter_keys(parameters)
        if not parameter_keys:
            raise ValueError("No parameters were selected.")

        fig, ax = _plt_setup_fig_axis(axis, fig_size)

        rows: list[np.ndarray] = []
        reference_time: np.ndarray | None = None

        for parameter_key in parameter_keys:
            data = self._reduce_pulses(
                parameter_key,
                channels=channel,
                method=method,
            ).compute()

            if data.shape[0] != 1:
                raise ValueError("plot_raster() requires exactly one channel.")

            time = self.time_axis(parameter_key)
            if reference_time is None:
                reference_time = time
            elif time.shape != reference_time.shape or not np.allclose(time, reference_time):
                raise ValueError(
                    "All selected parameters must have the same time axis for a raster plot."
                )

            rows.append(data[0])

        if reference_time is None:
            raise RuntimeError("No raster time axis was constructed.")

        plot_data = np.stack(rows)

        if c_lim is None or c_lim == "auto":
            scale = 6.0 * float(np.std(plot_data))
            color_limits = (-scale, scale) if scale > 0 else (None, None)
        elif c_lim == "max":
            color_limits = (None, None)
        else:
            limits = np.asarray(c_lim, dtype=float).reshape(-1)
            if limits.size != 2:
                raise ValueError("c_lim must contain exactly two values.")
            if not np.isfinite(limits).all():
                raise ValueError("c_lim values must be finite.")
            if limits[1] <= limits[0]:
                raise ValueError("c_lim must satisfy upper > lower.")
            color_limits = (float(limits[0]), float(limits[1]))

        image = ax.imshow(
            plot_data,
            *args,
            cmap=c_map,
            extent=[
                reference_time[0],
                reference_time[-1],
                0,
                plot_data.shape[0],
            ],
            vmin=color_limits[0],
            vmax=color_limits[1],
            aspect="auto",
            **kwargs,
        )

        ax.set_xlabel("time (s)")
        ax.set_ylabel("parameter")
        ax.set_yticks(np.arange(plot_data.shape[0]) + 0.5)
        ax.set_yticklabels(parameter_keys)
        ax.set_ylim(0, plot_data.shape[0])
        ax.set_xlim(
            (reference_time[0], reference_time[-1])
            if x_lim is None
            else x_lim
        )

        _plt_add_cbar_axis(
            fig,
            ax,
            c_label="amplitude (V)",
            c_lim=color_limits,
            c_map=c_map,
        )

        # Keep a reference for callers that need the image artist.
        ax._epoch_raster_image = image
        return _plt_show_fig(fig, ax, show)
