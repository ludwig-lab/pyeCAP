ECAP
====

The ECAP class combines Ephys and Stim objects. Below is documentation for the ECAP class along with its parent class
_EpochData.

.. testsetup::

    import eba_toolkit, os
    with eba_toolkit.eba_toolkit.hide_print():
        ephys_data = eba_toolkit.Ephys(os.path.join('..', 'tests', 'data', 'TDT', 'SmallData'))
        stim_data = eba_toolkit.Stim(os.path.join('..', 'tests', 'data', 'TDT', 'SmallData'))
        ecap_data = eba_toolkit.ECAP(ephys_data, stim_data)

.. autoclass:: eba_toolkit.ecap.ECAP
    :members:
    :special-members:

ECAP parent class
^^^^^^^^^^^^^^^^^

The _EpochData class is the parent class of the ECAP class. The ECAP class inherits all methods from this class. Methods
in this class take one or more stimulation parameters as input. Stimulation parameters are tuples containing two
integer indices for the data. The first integer represents the data set, and the second integer represents the
stimulation period within the experiment. For example, the parameter (1,3) would refer to the fourth stimulation in the
second experiment.

There are two main types of methods present in the _EpochData class: methods for working with the data as an array, and
plotting methods.

- Array Methods:

    These Methods work with the data as a dask array or numpy array. With the Ephys class, arrays of the data were divided
    into two dimensions, one for data points and one for channels. The _EpochData class re-dimensions the array into three
    dimensions using the stimulation parameters. One dimension for channels, one for stimulation pulses, and one for data
    points. Each array is generated using one parameter and extra data is ignored. This means that each array only
    contains the data for one stimulation period within one experiment. All of the methods for working with arrays are
    wrapped with the functools.lru_cache decorator for faster computing.

    Visualization of the dask array for one parameter. The ECAP object shown below has 7 channels, 750 pulses in the
    stimulation period, and 977 data points in one pulse.

    .. image:: ../../images/array_example.png

- Plotting Methods:

    These methods help to visualize the relevant Ephys data.


.. autoclass:: eba_toolkit.base.epoch_data._EpochData

    .. automethod:: __init__

    .. centered:: Array Methods
    .. automethod:: time
    .. automethod:: dask_array
    .. automethod:: array
    .. automethod:: mean
    .. automethod:: median
    .. automethod:: std

    .. centered:: Plotting Methods
    .. automethod:: plot(axis=None, channels=None, x_lim=None, y_lim='auto',  ch_labels=None, colors=sns.color_palette(), fig_size=(12, 3), show=True)
    .. automethod:: plot_channel(channel, parameters, *args, method='mean', axis=None, x_lim=None, y_lim='auto', colors=sns.color_palette(), fig_size=(10, 3), show=True, **kwargs)
    .. automethod:: plot_raster(channel, parameters, *args, method='mean', axis=None, x_lim=None, c_lim='auto', c_map='RdYlBu', fig_size=(10, 4), show=True, **kwargs)
