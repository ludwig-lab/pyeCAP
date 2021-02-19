import pandas as pd


class _ParameterData:
    """
    Class representing stimulation parameter data.
    """
    def __init__(self, parameters, metadata):
        """
        Constructor for stimulation parameter data class.

        Parameters
        ----------
        parameters: list
            List of pandas DataFrames containing stimulation parameter data.
        metadata : list
            List of dictionaries containing stimulation experiment metadata.
        """
        if not isinstance(parameters, list):
            parameters = [parameters]
        if not isinstance(metadata, list):
            metadata = [metadata]
        self._parameters = parameters
        self._metadata = metadata

    @property
    def parameters(self):
        """
        Property getter method for information about stimulation time periods, pulses, frequencies, and more.

        Returns
        -------
        pandas.core.frame.DataFrame
            Pandas DataFrame containing stimulation parameters.
        """
        return pd.concat(self._parameters, keys=range(len(self._parameters)))

    def append(self, new_data):
        """
        Creates a new class instance with new parameter data added to the original parameter data.

        Parameters
        ----------
        new_data : _ParameterData or subclass
            New data

        Returns
        -------
        _ParameterData or subclass
            Class instance containing original data with new data.
        """
        if isinstance(new_data, type(self)):
            parameters = self._parameters + new_data._parameters
            metadata = self._metadata + new_data._metadata
            return type(self)(parameters, metadata)
        else:
            raise TypeError("Appended data is not of the same type.")
