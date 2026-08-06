# python standard library imports
from pathlib import Path
from collections.abc import Mapping, Sequence
import os
import glob

# neuro base class imports
from .base.ts_data import _TsData
# from .base.utils.base import _is_iterable

# pyeCAP acquisition_io class imports
from .acquisition_io.ripple_io import RippleIO, RippleArray
from .acquisition_io.tdt_io import TdtIO, TdtArray, gather_sample_delay


class Ephys(_TsData):
    def __init__(self, data, *args, stores=None, order=True,
                 rz_sample_rate=None, si_sample_rate=None, sample_delay=None,
                 user_metadata=None, tdt_chunk_size=1_000_000, **kwargs):
        self.exp_path = data

        if user_metadata is not None and not isinstance(user_metadata, Mapping):
            raise TypeError("user_metadata must be a mapping or None.")

        try:
            requested_paths = self._normalize_paths(data)
        except TypeError:
            requested_paths = None

        if requested_paths is None:
            self.file_path = None
            super().__init__(data, *args, order=order, **kwargs)
            return

        self.file_path = requested_paths
        io_all = []
        data_all = []
        metadata_all = []
        chunks_all = []

        for requested_path in requested_paths:
            io_list, data_list, metadata_list, chunk_list = self._load_one_path(
                requested_path,
                stores=stores,
                rz_sample_rate=rz_sample_rate,
                si_sample_rate=si_sample_rate,
                sample_delay=sample_delay,
                user_metadata=user_metadata,
                tdt_chunk_size=tdt_chunk_size,
            )
            io_all.extend(io_list)
            data_all.extend(data_list)
            metadata_all.extend(metadata_list)
            chunks_all.extend(chunk_list)

        if order:
            records = sorted(
                zip(io_all, data_all, metadata_all, chunks_all),
                key=lambda record: record[2].get("start_time", 0),
            )
            io_all = [record[0] for record in records]
            data_all = [record[1] for record in records]
            metadata_all = [record[2] for record in records]
            chunks_all = [record[3] for record in records]

        self.io = io_all
        offsets = [metadata.get("ch_offsets") for metadata in metadata_all]
        offset_argument = offsets[0] if len(offsets) == 1 else offsets

        super().__init__(
            data_all,
            metadata_all,
            *args,
            chunks=chunks_all,
            daskify=True,
            order=False,
            ch_offsets=offset_argument,
            **kwargs,
        )

    @staticmethod
    def _expand_tdt_tanks_if_needed(path: str) -> list[str]:
        """
        If path is a folder containing subfolders each with a .tev, return those subfolders.
        Otherwise, return [path].
        """
        if not os.path.isdir(path):
            return [path]
        tev_files = glob.glob(os.path.join(path, "*.tev"))
        if len(tev_files) == 1:
            return [path]
        if len(tev_files) > 1:
            raise FileExistsError("Multiple '*.tev' files found in tank, 1 expected.")
        # tank-of-tanks
        tev_files = glob.glob(os.path.join(path, "*", "*.tev"))
        if not tev_files:
            raise FileNotFoundError("Could not locate '*.tev' file expected for tdt tank.")
        return sorted({os.path.dirname(f) for f in tev_files})

    @classmethod
    def _load_one_path(cls, path, *, stores=None, rz_sample_rate=None,
                       si_sample_rate=None, sample_delay=None, user_metadata=None,
                       tdt_chunk_size=1_000_000):
        """Load one path, expanding a directory containing multiple TDT tanks."""
        paths = cls._expand_tdt_tanks_if_needed(os.fspath(path))
        io_all, data_all, metadata_all, chunks_all = [], [], [], []

        for expanded_path in paths:
            io_list, data_list, metadata_list, chunk_list, _ = cls._load_one_path_single(
                expanded_path,
                stores=stores,
                rz_sample_rate=rz_sample_rate,
                si_sample_rate=si_sample_rate,
                sample_delay=sample_delay,
                user_metadata=user_metadata,
                tdt_chunk_size=tdt_chunk_size,
            )
            io_all.extend(io_list)
            data_all.extend(data_list)
            metadata_all.extend(metadata_list)
            chunks_all.extend(chunk_list)

        return io_all, data_all, metadata_all, chunks_all

    @staticmethod
    def _load_one_path_single(
            path,
            *,
            stores=None,
            rz_sample_rate=None,
            si_sample_rate=None,
            sample_delay=None,
            user_metadata=None,
            tdt_chunk_size=1_000_000,
    ):
        """Load exactly one Ripple file or TDT tank directory."""
        path = Path(path)
        if user_metadata is not None and not isinstance(user_metadata, Mapping):
            raise TypeError("user_metadata must be a mapping or None.")

        if path.suffix.lower() == ".nev":
            io_object = RippleIO(str(path))
            array = RippleArray(io_object, type="ephys")
            metadata = dict(array.metadata)
            if user_metadata:
                metadata.update(user_metadata)
            return [io_object], [array], [metadata], [array.chunks], True

        if path.is_dir():
            try:
                io_object = TdtIO(str(path))
                data_store = TdtArray(
                    io_object,
                    type="ephys",
                    stores=stores,
                    chunk_size=tdt_chunk_size,
                )
            except Exception as exc:
                raise IOError(f'"{path}" is not a valid TDT tank.') from exc

            data = data_store.data
            metadata = dict(data_store.metadata)
            chunks = data_store.chunks
            n_channels = len(metadata["ch_names"])
            channel_offsets = None

            if isinstance(sample_delay, Sequence) and not isinstance(sample_delay, (str, bytes)):
                channel_offsets = [-int(delay) for delay in sample_delay]
                if len(channel_offsets) != n_channels:
                    raise ValueError(
                        f"sample_delay length ({len(channel_offsets)}) does not match "
                        f"number of channels ({n_channels})."
                    )
            elif sample_delay is not None:
                channel_offsets = [-int(sample_delay)] * n_channels

            if rz_sample_rate is not None or si_sample_rate is not None:
                if channel_offsets is None:
                    channel_offsets = [0] * n_channels
                rate_offset = -int(gather_sample_delay(rz_sample_rate, si_sample_rate))
                channel_offsets = [offset + rate_offset for offset in channel_offsets]

            if channel_offsets is not None:
                metadata["ch_offsets"] = channel_offsets
            if user_metadata:
                metadata.update(user_metadata)

            return [io_object], [data], [metadata], [chunks], False

        if path.exists():
            extension = path.suffix or path.name
            raise IOError(f'"{extension}" is not a supported file extension.')
        raise IOError(f'"{path}" is not a file or directory.')

    @staticmethod
    def _normalize_paths(data):
        """
        Normalize path-like input to a list[Path].

        Accepts:
            - single str / PathLike
            - iterable of str / PathLike

        Raises:
            TypeError if `data` is not path-like input.
        """
        if isinstance(data, (str, os.PathLike)):
            return [Path(data)]

        try:
            items = list(data)
        except TypeError:
            raise TypeError("data is not path-like input")

        if not items:
            raise TypeError("data is an empty iterable, not path-like input")

        paths = []
        for x in items:
            if not isinstance(x, (str, os.PathLike)):
                raise TypeError(f"non-path element in iterable: {type(x)}")
            paths.append(Path(x))

        return paths