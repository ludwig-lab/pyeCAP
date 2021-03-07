# neuro base class imports
from .base.dio_epoch import _DioEpoch


class PhysResponse(_DioEpoch):
    """Class for indexing physiological data with stimulation parameters and viewing the response to stimulation."""
    def __init__(self, phys_data, stim_data, trigger_channel, threshold=1, time_difference=0, search=1, **kwargs):
        """
        Constructor for the PhysResponse class.

        Parameters
        ----------
        phys_data : _TsData or subclass
            Phys object containing the raw data.
        stim_data : Stim or PhysStim object
            Object containing stimulation data.
        trigger_channel : str, None
            Channel that contains data for stimulation trigger pulses. Used to sync the Phys and Stim data times. Use
            None to create an object without a trigger channel (not recommended).
        threshold : float
            Minimum value to register as a stimulation pulse
        time_difference : float
            Time difference in seconds between the clocks of the computers that recorded the stimulation and
            physiological data. A time_difference of 3600 would correct an error where the time of the phys_data leads
            the stim_data by 1 hour.
        search : float
            Maximum time difference to search for a pulse in the trigger channel in seconds. With a search of 1, eba-toolkit
            will search all data points between 1 second before and after the time given by the stimulation data.
        ** kwargs : KeywordArguments
            See :ref:`_DioEpoch` for more information

        Examples
        ________
        >>> # Set up a PhysResponse object with trigger as "Channel 6" and default parameters.
        >>> response_data = eba_toolkit.PhysResponse(phys_data, stim_data, "Channel 6")    # doctest: +SKIP
        """
        _DioEpoch.__init__(self, phys_data, stim_data, trigger_channel, threshold=threshold,
                           time_difference=time_difference, search=search, **kwargs)
