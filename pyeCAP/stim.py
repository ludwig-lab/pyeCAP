import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .base.event_data import _EventData
from .base.dio_data import _DioData
from .base.parameter_data import _ParameterData
from .base.utils.base import _is_iterable

from .acquisition_io.ripple_io import RippleIO, RippleEvents
from .acquisition_io.tdt_io import TdtIO, TdtStim


class Stim(_EventData, _DioData, _ParameterData):
    """
    Class for working with stimulation data.

    Physical stimulation contacts are assigned after loading using
    ``set_channels()`` rather than during object construction.
    """

    def __init__(self, file_path):
        """
        Load stimulation data from one or more Ripple files or TDT tanks.

        Parameters
        ----------
        file_path : str, os.PathLike, or iterable of path-like
            Ripple file, TDT tank directory, or a sequence of either.

        Examples
        --------
        >>> stim = Stim(pathname)  # doctest: +SKIP

        >>> stim = Stim([pathname1, pathname2])  # doctest: +SKIP

        >>> stim.set_channels([[6], [6, 3]])  # doctest: +SKIP
        """
        paths = self._normalize_paths(file_path)

        loaded = [
            self._load_one_path_single(path)
            for path in paths
        ]

        (
            file_paths,
            io,
            events,
            event_indicators,
            dio,
            dio_indicators,
            parameters,
            metadata,
        ) = map(list, zip(*loaded))

        self.file_path = file_paths
        self.io = io

        event_indicators = self._normalize_optional(
            event_indicators,
            name="event indicators",
        )

        dio = self._normalize_optional(
            dio,
            name="DIO data",
        )

        dio_indicators = self._normalize_optional(
            dio_indicators,
            name="DIO indicators",
        )

        parameters = self._normalize_optional(
            parameters,
            name="parameter data",
        )

        self._validate_metadata(metadata)

        self._initialize_base_classes(
            events=events,
            event_indicators=event_indicators,
            dio=dio,
            dio_indicators=dio_indicators,
            parameters=parameters,
            metadata=metadata,
        )

        self._update_derived_attributes()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_base_classes(
        self,
        *,
        events,
        event_indicators,
        dio,
        dio_indicators,
        parameters,
        metadata,
    ):
        """Initialize the data-oriented base classes."""

        _EventData.__init__(
            self,
            events,
            metadata,
            indicators=event_indicators,
        )

        _DioData.__init__(
            self,
            dio,
            metadata,
            indicators=dio_indicators,
        )

        _ParameterData.__init__(
            self,
            parameters,
            metadata,
        )

    def _update_derived_attributes(self):
        """Update attributes calculated from the loaded stimulation data."""

        self.stim_times = self.all_event_times()
        self.param_to_times = self.parameter_to_event_times()

    @staticmethod
    def _validate_metadata(metadata):
        """Require one metadata dictionary per dataset."""

        invalid = [
            (i, type(value).__name__)
            for i, value in enumerate(metadata)
            if not isinstance(value, dict)
        ]

        if invalid:
            raise TypeError(
                "Each dataset must provide a metadata dictionary. "
                f"Invalid entries: {invalid}"
            )

    @staticmethod
    def _normalize_optional(values, *, name):
        """
        Normalize optional per-dataset data.

        Returns
        -------
        list or None
            None when every dataset lacks the value. Otherwise, returns
            the original per-dataset list.

        Raises
        ------
        ValueError
            If only some datasets provide the requested data.
        """

        if all(value is None for value in values):
            return None

        if any(value is None for value in values):
            missing = [
                i
                for i, value in enumerate(values)
                if value is None
            ]

            raise ValueError(
                f"Only some datasets contain {name}. "
                f"Missing for dataset indices: {missing}"
            )

        return values

    # ------------------------------------------------------------------
    # Path handling and loading
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_paths(data):
        """
        Normalize path-like input into a list of Path objects.
        """

        if isinstance(data, (str, os.PathLike)):
            return [Path(data)]

        try:
            items = list(data)
        except TypeError as exc:
            raise TypeError(
                "file_path must be path-like or an iterable of paths."
            ) from exc

        if not items:
            raise ValueError("file_path cannot be empty.")

        paths = []

        for item in items:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(
                    "Every file_path item must be path-like. "
                    f"Received {type(item).__name__}."
                )

            paths.append(Path(item))

        return paths

    @staticmethod
    def _load_one_path_single(path):
        """
        Load one Ripple file or one TDT tank.

        Returns
        -------
        tuple
            path, acquisition_io, events, event indicators, DIO, DIO indicators,
            parameters, metadata

        Notes
        -----
        Each returned value represents exactly one dataset. Values are
        not wrapped in lists here; __init__ collects one value per path.
        """

        path = Path(path)

        # --------------------------------------------------------------
        # Ripple
        # --------------------------------------------------------------

        if path.suffix.lower() == ".nev":
            io_obj = RippleIO(str(path))
            ripple_events = RippleEvents(io_obj, type="stim")

            events = ripple_events
            event_indicators = None
            dio = None
            dio_indicators = None
            parameters = None
            metadata = getattr(ripple_events, "metadata", None)

            return (
                path,
                io_obj,
                events,
                event_indicators,
                dio,
                dio_indicators,
                parameters,
                metadata,
            )

        # --------------------------------------------------------------
        # TDT
        # --------------------------------------------------------------

        if path.is_dir():
            tev_files = list(path.glob("*.tev"))

            if not tev_files:
                child_tanks = sorted({
                    file.parent
                    for file in path.glob("*/*.tev")
                })

                if not child_tanks:
                    raise FileNotFoundError(
                        f'Could not locate a "*.tev" file under "{path}".'
                    )

                if len(child_tanks) == 1:
                    return Stim._load_one_path_single(child_tanks[0])

                raise IsADirectoryError(
                    f'"{path}" contains multiple TDT tanks. '
                    "Pass the individual tank directories as a list."
                )

            if len(tev_files) > 1:
                raise FileExistsError(
                    f'"{path}" contains multiple "*.tev" files; '
                    "exactly one was expected."
                )

            io_obj = TdtIO(str(path))
            tdt_stim = TdtStim(io_obj)

            parameters = tdt_stim.parameters
            metadata = tdt_stim.metadata
            events = tdt_stim.events()
            event_indicators = tdt_stim.events(indicators=True)
            dio = tdt_stim.dio()
            dio_indicators = tdt_stim.dio(indicators=True)

            return (
                path,
                io_obj,
                events,
                event_indicators,
                dio,
                dio_indicators,
                parameters,
                metadata,
            )

        if path.exists():
            extension = path.suffix or path.name

            raise IOError(
                f'"{extension}" is not a supported file type.'
            )

        raise FileNotFoundError(
            f'"{path}" does not exist.'
        )

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    @property
    def raw_stores(self):
        """
        Return raw stimulation waveform and monitoring stores.

        Returns
        -------
        list of dict
            One dictionary of raw stores per dataset.
        """

        raw_data = []

        for dataset_index, metadata in enumerate(self.metadata):
            store_names = metadata.get("raw_stores")

            if store_names is None:
                warnings.warn(
                    f"Dataset {dataset_index} does not provide raw "
                    "TDT stimulation stores.",
                    stacklevel=2,
                )
                raw_data.append({})
                continue

            stores = self.io[dataset_index].tdt_block.stores

            raw_data.append({
                store_name: getattr(stores, store_name)
                for store_name in store_names
            })

        return raw_data

    # ------------------------------------------------------------------
    # Parameter setters
    # ------------------------------------------------------------------

    def set_parameters(self, parameter, values):
        """
        Set one parameter column for one or more datasets.

        Parameters
        ----------
        parameter : str
            Parameter column to update.
        values : scalar or sequence
            A scalar is applied to every dataset. A one-item sequence is
            broadcast to every dataset. Otherwise, provide one value per
            dataset.

        Returns
        -------
        Stim
            The modified object.
        """

        if self._parameters is None:
            raise ValueError(
                "This Stim object does not contain parameter data."
            )

        dataset_values = self._broadcast_dataset_values(values)

        for index, (dataframe, value) in enumerate(
            zip(self._parameters, dataset_values)
        ):
            dataframe = dataframe.copy()
            dataframe.loc[:, parameter] = value
            self._parameters[index] = dataframe

        return self

    def set_stimulation_contacts(self, values):
        """
        Assign physical stimulation electrode montages.

        Parameters
        ----------
        values : dict or sequence of dict
            Provide one montage per dataset. Each montage must contain
            ``cathodes`` and ``anodes``.

            Examples
            --------
            Monopolar:
                {"cathodes": [6], "anodes": []}

            Bipolar:
                {"cathodes": [6], "anodes": [3]}

            Tripolar:
                {"cathodes": [6], "anodes": [3, 4]}

            Multiple datasets:
                [
                    {"cathodes": [6], "anodes": []},
                    {"cathodes": [6], "anodes": [3]},
                    {"cathodes": [6], "anodes": [3, 4]},
                ]

        Returns
        -------
        Stim
            The modified object.
        """
        if self._parameters is None:
            raise ValueError(
                "This Stim object does not contain parameter data."
            )

        montages = self._normalize_montages(values)

        for dataset_index, (dataframe, montage) in enumerate(
                zip(self._parameters, montages)
        ):
            cathodes = self._normalize_contacts(
                montage.get("cathodes", ()),
                name="cathodes",
                dataset_index=dataset_index,
            )

            anodes = self._normalize_contacts(
                montage.get("anodes", ()),
                name="anodes",
                dataset_index=dataset_index,
            )

            if not cathodes and not anodes:
                raise ValueError(
                    f"Dataset {dataset_index} must contain at least "
                    "one cathode or anode."
                )

            overlapping = set(cathodes) & set(anodes)

            if overlapping:
                raise ValueError(
                    f"Dataset {dataset_index} assigns contacts "
                    f"{sorted(overlapping)} as both cathodes and anodes."
                )

            dataframe = dataframe.copy()
            row_count = len(dataframe)

            # Tuples are preferable to mutable lists inside DataFrames.
            dataframe.loc[:, "cathodes"] = pd.Series(
                [cathodes] * row_count,
                index=dataframe.index,
                dtype=object,
            )

            dataframe.loc[:, "anodes"] = pd.Series(
                [anodes] * row_count,
                index=dataframe.index,
                dtype=object,
            )

            dataframe.loc[:, "n_contacts"] = (
                    len(cathodes) + len(anodes)
            )

            dataframe.loc[:, "contact_name"] = self._format_channel_name(cathodes, anodes)

            self._parameters[dataset_index] = dataframe

        return self

    @staticmethod
    def _normalize_contacts(values, *, name, dataset_index):
        """Convert one electrode-role assignment to a tuple of integers."""
        if values is None:
            return ()

        if np.isscalar(values):
            values = [values]

        try:
            contacts = tuple(int(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} for dataset {dataset_index} must contain "
                "integer contact numbers."
            ) from exc

        if len(set(contacts)) != len(contacts):
            raise ValueError(
                f"{name} for dataset {dataset_index} contains "
                "duplicate contacts."
            )

        return contacts

    def _normalize_montages(self, values):
        """Return one montage dictionary per parameter dataset."""
        number_of_datasets = len(self._parameters)

        if isinstance(values, dict):
            return [values.copy() for _ in range(number_of_datasets)]

        try:
            montages = list(values)
        except TypeError as exc:
            raise TypeError(
                "Montage values must be a dictionary or a sequence "
                "of dictionaries."
            ) from exc

        if not montages:
            raise ValueError("At least one montage must be supplied.")

        if not all(isinstance(value, dict) for value in montages):
            raise TypeError(
                "Each montage must be a dictionary containing "
                "'cathodes' and 'anodes'."
            )

        if len(montages) == 1:
            return [
                montages[0].copy()
                for _ in range(number_of_datasets)
            ]

        if len(montages) != number_of_datasets:
            raise ValueError(
                f"Expected one montage for each of the "
                f"{number_of_datasets} datasets, but received "
                f"{len(montages)}."
            )

        return montages

    def _broadcast_dataset_values(self, values):
        """
        Normalize values to one value per parameter DataFrame.
        """

        number_of_datasets = len(self._parameters)

        if not _is_iterable(values):
            return [values] * number_of_datasets

        values = list(values)

        if len(values) == 1:
            return values * number_of_datasets

        if len(values) != number_of_datasets:
            raise ValueError(
                f"Expected one value for each of the "
                f"{number_of_datasets} datasets, but received "
                f"{len(values)}."
            )

        return values

    @staticmethod
    def _format_channel_name(cathodes, anodes):
        """
        Create a readable stimulation-contact label.

        Examples
        --------
        cathodes=(6,), anodes=()
            -> "6"

        cathodes=(6,), anodes=(3,)
            -> "6c-3a"

        cathodes=(6,), anodes=(3, 4)
            -> "6c-3a-4a"

        cathodes=(6, 7), anodes=(3,)
            -> "6c-7c-3a"
        """
        cathodes = tuple(cathodes)
        anodes = tuple(anodes)

        contacts = cathodes + anodes

        if len(contacts) == 1:
            return str(contacts[0])

        cathode_names = [
            f"{channel}c"
            for channel in cathodes
        ]

        anode_names = [
            f"{channel}a"
            for channel in anodes
        ]

        return "-".join(cathode_names + anode_names)

    def set_pulse_amplitude_abs(
        self,
        col="pulse amplitude (μA)",
    ):
        """
        Replace pulse amplitudes with their absolute values.

        Returns
        -------
        Stim
            The modified object.
        """

        if self._parameters is None:
            raise ValueError(
                "This Stim object does not contain parameter data."
            )

        missing = [
            index
            for index, dataframe in enumerate(self._parameters)
            if col not in dataframe.columns
        ]

        if missing:
            raise KeyError(
                f'Column "{col}" is missing from parameter datasets '
                f"{missing}."
            )

        for index, dataframe in enumerate(self._parameters):
            dataframe = dataframe.copy()
            dataframe.loc[:, col] = dataframe[col].abs()
            self._parameters[index] = dataframe

        return self

    def add_series(
        self,
        dataset_index,
        series_to_add: pd.Series,
    ):
        """
        Add a Series as a column to one parameter DataFrame.

        Parameters
        ----------
        dataset_index : int
            Parameter dataset to modify.
        series_to_add : pandas.Series
            Values to add. A one-value Series is broadcast over all
            parameter rows.

        Returns
        -------
        Stim
            The modified object.
        """

        dataframe = self._parameters[dataset_index].copy()
        row_count = len(dataframe)

        if len(series_to_add) == 1:
            values = np.repeat(
                series_to_add.iloc[0],
                row_count,
            )
        elif len(series_to_add) == row_count:
            values = series_to_add.to_numpy()
        else:
            raise ValueError(
                f"Series length must be 1 or {row_count}; "
                f"received {len(series_to_add)}."
            )

        dataframe.loc[:, series_to_add.name] = values
        self._parameters[dataset_index] = dataframe

        return self

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_dio(self, *args, **kwargs):
        """Plot stimulation DIO periods."""

        return _DioData.plot_raster(
            self,
            *args,
            **kwargs,
        )

    def plot_events(self, *args, **kwargs):
        """Plot stimulation event times."""

        return _EventData.plot_raster(
            self,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Combining Stim objects
    # ------------------------------------------------------------------

    def append(self, new_data):
        """
        Append another Stim object in place.

        Parameters
        ----------
        new_data : Stim
            Object containing datasets to append.

        Returns
        -------
        Stim
            The modified object.
        """

        if not isinstance(new_data, type(self)):
            raise TypeError(
                "Appended data must be another Stim object."
            )

        self.file_path.extend(new_data.file_path)
        self.io.extend(new_data.io)
        self._events.extend(new_data._events)
        self._dio.extend(new_data._dio)
        self._parameters.extend(new_data._parameters)
        self._metadata.extend(new_data._metadata)

        self._event_indicators = self._append_optional_lists(
            self._event_indicators,
            new_data._event_indicators,
            name="event indicators",
        )

        self._dio_indicators = self._append_optional_lists(
            self._dio_indicators,
            new_data._dio_indicators,
            name="DIO indicators",
        )

        self._update_derived_attributes()

        return self

    @staticmethod
    def _append_optional_lists(left, right, *, name):
        """Append two optional per-dataset lists."""

        if left is None and right is None:
            return None

        if left is None or right is None:
            raise ValueError(
                f"Cannot append objects with inconsistent {name}."
            )

        return left + right

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def all_event_times(
        self,
        start_times=None,
        reference=None,
        remove_gaps=False,
        unique=True,
    ):
        """
        Return sorted pulse times across all stimulation sources.
        """

        times = []

        for channel_name in self.ch_names:
            channel_times = self.events(
                channel_name,
                start_times=start_times,
                reference=reference,
                remove_gaps=remove_gaps,
            )

            if channel_times is not None and len(channel_times):
                times.append(
                    np.asarray(channel_times, dtype=float)
                )

        if not times:
            return np.array([], dtype=float)

        output = np.sort(np.concatenate(times))

        if unique:
            output = np.unique(output)

        return output

    def parameter_to_event_times(
        self,
        start_times=None,
        reference=None,
        remove_gaps=False,
    ):
        """
        Map each parameter row to its stimulation event times.

        Returns
        -------
        dict
            Keys are ``(dataset_index, parameter_index)`` tuples.
        """

        parameter_times = {}

        for channel_name in self.ch_names:
            channel_events = self.events(
                channel_name,
                start_times=start_times,
                reference=reference,
                remove_gaps=remove_gaps,
            )

            indicators = self.event_indicators(channel_name)

            for indicator, event_time in zip(
                indicators,
                channel_events,
            ):
                key = tuple(indicator)

                parameter_times.setdefault(
                    key,
                    [],
                ).append(float(event_time))

        return {
            key: np.sort(np.asarray(values, dtype=float))
            for key, values in parameter_times.items()
        }

    def all_event_indices(
        self,
        reference,
        start_times=None,
        remove_gaps=False,
        unique=True,
        clip=True,
    ):
        """
        Return pulse sample indices on another object's timebase.
        """

        times = self.all_event_times(
            start_times=start_times,
            reference=reference,
            remove_gaps=remove_gaps,
            unique=unique,
        )

        indices = np.rint(
            times * reference.sample_rate
        ).astype(np.int64, copy=False)

        if clip:
            if hasattr(reference, "shape"):
                number_of_samples = reference.shape[-1]
            else:
                number_of_samples = reference.array.shape[-1]

            indices = indices[
                (indices >= 0)
                & (indices < number_of_samples)
            ]

        if unique:
            indices = np.unique(indices)

        return indices