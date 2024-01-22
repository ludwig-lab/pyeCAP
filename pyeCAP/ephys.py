# python standard library imports
import glob
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# scientific computing library imports
import numpy as np

# neuro base class imports
from .base.ts_data import _TsData
from .base.utils.base import _is_iterable

# pyeCAP io class imports
from .io.ripple_io import RippleArray, RippleIO
from .io.tdt_io import TdtArray, TdtIO, gather_sample_delay


def process_file(
    data_type,
    file,
    *args,
    stores=None,
    rz_sample_rate=None,
    si_sample_rate=None,
    sample_delay=None,
    **kwargs
):
    if isinstance(file, str):
        ephys_data_set = data_type(
            file,
            *args,
            stores=stores,
            rz_sample_rate=rz_sample_rate,
            si_sample_rate=si_sample_rate,
            sample_delay=sample_delay,
            **kwargs
        )
        return ephys_data_set
    else:
        raise IOError(
            "Input is expected to be either a string containing a file path, or a list of file paths."
        )


class Ephys(_TsData):
    """
    Class for working with Ephys objects.
    """

    def __init__(
        self,
        data,
        *args,
        stores=None,
        order=True,
        rz_sample_rate=None,
        si_sample_rate=None,
        sample_delay=None,
        **kwargs
    ):
        """
        Ephys constructor.

        Parameters
        ----------
        data : str, list
            path name or list of pathnames for a directory of TDT data
        * args : Arguments
            Arguments to be passed to :ref:`_TsData (parent class)` constructor.
        stores : None, list, tuple
            Sequence of tdt store names to include in the object, or None to include all stores.
        order : bool
            Set to True to order data sets by start time. Since data from the same file is read in chronological order,
            this will only have an effect when reading in data from multiple files.
        rz_sample_rate : int
            Sample rate (in kHz) of the rz processor of a TDT system.
        si_sample_rate : int
            Sample rate (in kHz) of the PZ5 processor of a TDT system.
        sample_delay : int, list
            Int to add one sample delay to all channels, or a list to add individual sample delays. The list must be the
            same length as the number of channels and each sample in the list matches the channels by order. Specify
            offsets as positive to show

        ** kwargs : KeywordArguments
            Keyword arguments to be passed to :ref:`_TsData (parent class)` constructor.

        Examples
        ________
        >>> pyeCAP.Ephys(pathname1)   # replace pathnames with paths to data      # doctest: +SKIP

        >>> pyeCAP.Ephys([pathname1, pathname2, pathname3])                       # doctest: +SKIP
        """
        self.exp_path = data
        # Work with file path
        if isinstance(data, str):
            file_path = data

            # Read in Ripple data files
            if file_path.endswith(".nev"):
                self.file_path = [file_path]
                self.io = [RippleIO(file_path)]
                data = RippleArray(self.io[0], type="ephys")
                chunks = data.chunks
                metadata = data.metadata
                daskify = True

            # Read in file types that point to a directory (i.e. tdt)
            elif os.path.isdir(file_path):
                # Check if directory is for tdt data
                tev_files = glob.glob(file_path + "/*.tev")  # There should only be one
                if len(tev_files) == 0:
                    # Check if this is a folder of tanks, look for tev files one live deep
                    tev_files = glob.glob(file_path + "/*/*.tev")
                    if len(tev_files) == 0:
                        raise FileNotFoundError(
                            "Could not located '*.tev' file expected for tdt tank."
                        )
                    else:
                        data = [os.path.split(f)[0] for f in tev_files]
                        self.__init__(
                            data,
                            *args,
                            stores=stores,
                            rz_sample_rate=rz_sample_rate,
                            si_sample_rate=si_sample_rate,
                            sample_delay=sample_delay,
                            **kwargs
                        )
                        return
                elif len(tev_files) > 1:
                    raise FileExistsError(
                        "Multiple '*.tev' files found in tank, 1 expected."
                    )
                else:
                    self.file_path = [file_path]
                    self.io = [TdtIO(file_path)]
                    data_store = TdtArray(self.io[0], type="ephys", stores=stores)
                    chunks = data_store.chunks
                    data = data_store.data
                    metadata = data_store.metadata
                    daskify = False
                    # set up sample delay to be ch_offsets parameter
                    if isinstance(sample_delay, list):
                        sample_delay = [-sd for sd in sample_delay]
                    elif sample_delay is not None:
                        sample_delay = [-int(sample_delay)] * len(metadata["ch_names"])

                    # add in delays from rz and si sample rate
                    if rz_sample_rate is not None or si_sample_rate is not None:
                        if sample_delay is None:
                            sample_delay = 0
                        # set tdt delays based on rz and si sample rates
                        rate_offsets = [
                            -gather_sample_delay(rz_sample_rate, si_sample_rate)
                        ] * len(metadata["ch_names"])
                        sample_delay = np.add(sample_delay, rate_offsets)

            # File type not found
            else:
                if os.path.exists(file_path):
                    if os.path.isdir(file_path):
                        raise IOError('"' + file_path + '"  - is not a tdt tank.')
                    else:
                        _, file_extension = os.path.splitext(file_path)
                        raise IOError(
                            '"' + file_extension + '" is not a supported file extension'
                        )
                else:
                    raise IOError('"' + file_path + '"  - is not a file or directory.')

            super().__init__(
                data,
                metadata,
                *args,
                chunks=chunks,
                daskify=daskify,
                ch_offsets=sample_delay,
                **kwargs
            )

        # Work with iterable of file paths
        elif _is_iterable(data, str):
            self.file_path = data

            # Reading in the initial data files (particularly TDT files) can be slow and can be parallelized.

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_file,
                        type(self),
                        file,
                        *args,
                        stores=stores,
                        rz_sample_rate=rz_sample_rate,
                        si_sample_rate=si_sample_rate,
                        sample_delay=sample_delay,
                        **kwargs
                    )
                    for file in self.file_path
                ]
                ephys_files = [future.result() for future in futures]

            self.io = [item for d in ephys_files for item in d.io]
            data = [item for d in ephys_files for item in d.data]
            metadata = [item for d in ephys_files for item in d.metadata]
            chunks = [item for d in ephys_files for item in d.chunks]
            super().__init__(
                data,
                metadata,
                *args,
                chunks=chunks,
                daskify=False,
                order=order,
                **kwargs
            )

        else:
            super().__init__(data, *args, **kwargs)
