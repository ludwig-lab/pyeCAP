from datetime import datetime, timedelta
from typing import Union

import adi
import dask
import dask.array as da
import numpy as np

from ..base.dio_data import _DioData
from ..base.ts_data import _TsData


class AdiIO:
    """
    This class is responsible for handling Analog Devices data files in an adict format.
    It reads the data, handles errors, and extracts metadata.
    """

    def __init__(self, file_path: str):
        """
        Initializes the AdiIO object with a file path.

        This constructor checks if the provided file path is a string and if it ends with the '.adicht' extension,
        indicating it is an adinstruments file. If these conditions are met, it reads the file using the adi library.
        Otherwise, it raises appropriate errors.

        Parameters:
        - file_path (str): The path to the adinstruments file.

        Raises:
        - TypeError: If the file_path is not a string.
        - ValueError: If the file_path does not end with '.adicht'.
        """
        if not isinstance(file_path, str):
            raise TypeError("The file path must be a string.")
        self.file_path = file_path
        if file_path.endswith(".adicht"):
            self.file_object = adi.read_file(file_path)
        else:
            raise ValueError("The file is not an adinstruments file.")

    def close_file(self):
        """
        Closes the file associated with this AdiIO object.
        """
        self.file_object.close_file()

    @property
    def num_channels(self):
        """
        Returns the number of channels in the ADI file.

        This property accesses the underlying file object to retrieve the total number of channels available in the ADI file.

        Returns:
            int: The number of channels.
        """
        return self.file_object.get_n_channels()

    @property
    def num_data(self):
        """
        Returns the number of records in the ADI file.

        This property accesses the underlying file object to retrieve the total number of records available in the ADI file.

        Returns:
            int: The number of records.
        """
        return self.file_object.get_n_records()

    @property
    def records(self):
        """
        Provides access to the records in the ADI file.

        This property accesses the underlying file object to retrieve the records available in the ADI file. Each record contains data and metadata for a specific recording session.

        Returns:
            list: A list of records from the ADI file.
        """
        return self.file_object.records

    @property
    def channels(self):
        """
        Provides access to the channel information in the ADI file.

        This property accesses the underlying file object to retrieve information about the channels available in the ADI file. Each channel represents a separate data stream within the file.

        Returns:
            list: A list of channels from the ADI file.
        """
        return self.file_object.channels

    @property
    def ch_names(self):
        """
        Retrieves the names of all channels in the ADI file.

        This method accesses the underlying file object to extract the names of all channels available in the ADI file.

        Returns:
            list[str]: A list containing the names of all channels.
        """
        return [channel.name for channel in self.file_object.channels]


class AdiChannel(_TsData):
    """
    Class for handling ADI channel data, inheriting from _TsData for time series data manipulation.
    Data is loaded lazily using Dask.
    """

    def __init__(
        self, file_path_or_adiio: Union[str, AdiIO], channel_id: Union[int, str]
    ):
        """
        Initializes the AdiChannel object with data from an ADI file for a specific channel.
        Data loading is deferred until explicitly requested.

        Parameters:
        - file_path_or_adiio (Union[str, AdiIO]): The path to the ADI file or an AdiIO object.
        - channel_id (Union[int, str]): The ID or name of the channel to extract data from.
        """
        # Determine if file_path_or_adiio is a string or an AdiIO object and act accordingly
        if isinstance(file_path_or_adiio, str):
            self.file_path = file_path_or_adiio
            self.adi_file = AdiIO(file_path_or_adiio)
        elif isinstance(file_path_or_adiio, AdiIO):
            self.file_path = (
                file_path_or_adiio.file_path
            )  # Assuming AdiIO object has a file_path attribute
            self.adi_file = file_path_or_adiio
        else:
            raise ValueError("file_path_or_adiio must be a string or an AdiIO object.")

        # Handle channel_id as either an integer or a string
        if isinstance(channel_id, int):
            channel_index = channel_id - 1  # Adjust for 0-based indexing
        elif isinstance(channel_id, str):
            channel_names = [channel.name for channel in self.adi_file.channels]
            if channel_names.count(channel_id) == 1:
                channel_index = channel_names.index(channel_id)
            elif channel_names.count(channel_id) > 1:
                raise ValueError(
                    f"Multiple channels found with the name '{channel_id}'. Please specify by channel ID."
                )
            else:
                raise ValueError(f"No channel found with the name '{channel_id}'.")
        else:
            raise ValueError("channel_id must be an integer or a string.")

        # Extract the specified channel
        channel = self.adi_file.channels[channel_index]

        # Prepare a list to hold Dask arrays for each record and metadata
        dask_arrays = []
        metadata = []
        for record_id in range(1, channel.n_records + 1):
            # Get the number of samples for the current record to define the Dask array shape
            n_samples = channel.n_samples[record_id - 1]

            # Create a Dask array for the record data
            record_dask_array = da.from_delayed(
                dask.delayed(
                    lambda ch_id, r_id: self.adi_file.channels[ch_id].get_data(r_id)
                )(channel_index, record_id),
                shape=(n_samples,),
                dtype=np.float32,
            )
            dask_arrays.append(record_dask_array)

            # Assuming self.adi_file.records[record_id-1].record_time returns a RecordTime object
            record_time = self.adi_file.records[record_id - 1].record_time

            # Example metadata extraction with start time and trigger time
            record_metadata = {
                "sample_rate": channel.fs[record_id - 1],
                "start_time": record_time.rec_datetime.timestamp(),  # Convert recording start datetime to timestamp
                "trigger_time": record_time.trig_datetime.timestamp(),  # Convert trigger datetime to timestamp
                "ch_names": [channel.name],
                "ch_types": [None],  # Placeholder for actual channel type
                "units": channel.units,
            }
            metadata.append(record_metadata)

        # Initialize the parent class with the Dask array and metadata
        super().__init__(dask_arrays, metadata)


class AdiDIO(_DioData):
    """
    Class for handling ADI digital input/output (DIO) channel data, inheriting from _DioData.
    Processes the time series data to identify on/off periods based on a specified voltage threshold,
    ensuring that transitions are always correctly paired as on/off events.
    """

    def __init__(
        self,
        file_path_or_adiio: Union[str, AdiIO],
        channel_id: Union[int, str],
        threshold=2.5,
    ):
        """
        Initializes the AdiDIO object with data from an ADI file for a specific channel.
        Identifies on/off periods in the signal based on a threshold and creates an object inheriting from _DioData,
        ensuring transitions are always paired as on/off events.

        Parameters:
        - file_path_or_adiio (Union[str, AdiIO]): The path to the ADI file or an AdiIO object.
        - channel_id (Union[int, str]): The ID or name of the channel to extract data from.
        """
        # Determine if file_path_or_adiio is a string or an AdiIO object and act accordingly
        if isinstance(file_path_or_adiio, str):
            self.adi_file = AdiIO(file_path_or_adiio)
        elif isinstance(file_path_or_adiio, AdiIO):
            self.adi_file = file_path_or_adiio
        else:
            raise ValueError("file_path_or_adiio must be a string or an AdiIO object.")

        # Handle channel_id as either an integer or a string
        if isinstance(channel_id, int):
            channel_index = channel_id - 1  # Adjust for 0-based indexing
        elif isinstance(channel_id, str):
            channel_names = [channel.name for channel in self.adi_file.channels]
            if channel_names.count(channel_id) == 1:
                channel_index = channel_names.index(channel_id)
            elif channel_names.count(channel_id) > 1:
                raise ValueError(
                    f"Multiple channels found with the name '{channel_id}'. Please specify by channel ID."
                )
            else:
                raise ValueError(f"No channel found with the name '{channel_id}'.")
        else:
            raise ValueError("channel_id must be an integer or a string.")

        # Extract the specified channel
        channel = self.adi_file.channels[channel_index]

        # Prepare the dio list and metadata list
        dio = []
        metadata = []

        # Loop through each record in the channel
        for record_id in range(1, channel.n_records + 1):
            # Load the data for the current record
            data = channel.get_data(record_id)

            # Find indices where the signal crosses the threshold (default is 2.5V)
            above_threshold = data > threshold
            transitions = np.diff(above_threshold.astype(int))
            start_indices = (
                np.where(transitions == 1)[0] + 1
            )  # +1 to correct for diff offset
            stop_indices = np.where(transitions == -1)[0] + 1

            # Ensure the first index is a start index and the last index is a stop index
            if start_indices.size and stop_indices.size:
                if start_indices[0] > stop_indices[0]:
                    # Remove the first stop index if it comes before the first start index
                    stop_indices = stop_indices[1:]
                if stop_indices.size and start_indices[-1] > stop_indices[-1]:
                    # Remove the last start index if it comes after the last stop index
                    start_indices = start_indices[:-1]

            # Adjust for differences in the sizes of the lists, if any remain
            if start_indices.size > stop_indices.size:
                # Remove extra start indices from the end
                start_indices = start_indices[: stop_indices.size]
            elif stop_indices.size > start_indices.size:
                # Remove extra stop indices from the end
                stop_indices = stop_indices[: start_indices.size]

            # Convert indices to times relative to the start of the record
            times = []
            for start_idx, stop_idx in zip(start_indices, stop_indices):
                start_time = start_idx / channel.fs[record_id - 1]
                stop_time = stop_idx / channel.fs[record_id - 1]
                times.extend([start_time, stop_time])

            # Create a dictionary for this record with channel name as key and times as value
            dio.append({channel.name: times})

            # Assuming self.adi_file.records[record_id-1].record_time returns a RecordTime object with recording start datetime
            record_time = self.adi_file.records[record_id - 1].record_time.rec_datetime

            # Add metadata for this record
            metadata.append(
                {"ch_names": [channel.name], "start_time": record_time.timestamp()}
            )

        # Initialize the parent class with the dio and metadata
        super().__init__(dio, metadata)
