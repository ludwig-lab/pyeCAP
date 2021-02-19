Ephys
=====

The Ephys class provides several ways to work with electrophysiology data. An Ephys object is usually instantiated with
only a path name to a ripple file or TDT tank or a list of path names.

.. automodule:: pyCAP.ephys
    :members:
    :special-members:

Sample Delays
^^^^^^^^^^^^^

.. TODO: Insert more details on how to compute sample delays.

TDT recording devices are sometimes configured in a way that creates delays between different Ephys channels or between
a stimulation channel and Ephys channels. The delay is expressed in terms of number of samples and can depend on sample
rate and gizmo configuration. More information can be found at https://www.tdt.com/files/fastfacts/IODelays.pdf

In order to take these delays into account, the rz_sample_rate, si_sample_rate, and sample_delay parameters are passed
into the Ephys constructor. The rz and si sample rate parameters refer to the table found at
https://www.tdt.com/files/fastfacts/IODelays.pdf . Inputting these parameters will correct for uniform delays for each
channel by removing samples from the start of each data set in order to line up stimulation timing with Ephys data
timing. Additional sample delays can be specified with the sample_delay parameter. Input sample delays as an integer to
apply the sample delay to all channels or as a list (with same length as number of channels) to specify a sample delay
for individual channels. Specify sample delays as positive to chop off samples from the start of the data set. Although
negative sample delays will not cause the code to crash, no samples will be eliminated and the data will be treated as
if no sample delays were entered.

_TsData (parent class)
^^^^^^^^^^^^^^^^^^^^^^

The Ephys and Phys classes inherit the majority of methods from the _TsData class. There are several categories of methods including:

- Plotting Methods:

    Methods for plotting raw or processed data and experiment times.

- Channel Methods:

    Methods that set channel names, types, or re-reference channels by creating a new class instance with different
    channels.

- Filtering Methods:

    These methods are used to filter time series data. They are useful for eliminating noise and unwanted frequencies
    from the data. These methods return a new class instance with filtered data.

- Utility Methods:

    These methods are mainly used by other _TsData methods. However, they can be used to process the data at a lower
    level with a greater degree of control.

- Property Methods:

    Getter Methods using the @property decorator. These contain important metadata or ways to work with the time series
    data as an array.


.. testsetup::

    import pyCAP, os
    ephys_path = os.path.join('..', 'tests', 'data', 'TDT', 'SmallData')
    with pyCAP.pyCAP.hide_print():
        ephys_data = pyCAP.Ephys(ephys_path)

.. autoclass:: pyCAP.base.ts_data._TsData
    :members: array, shape, ndim, dtype, size, itemsize, shapes, start_indices, time, ch_names, ch_types, start_times, end_times, sample_rate, ndata,

    .. automethod:: __init__

    .. centered:: Plotting methods
    .. automethod:: plot(axis=None, channels=None, events=None, x_lim=None, y_lim='auto',  ch_labels=None, colors=sns.color_palette(), fig_size=(10, 6), down_sample=True, show=True)
    .. automethod:: plot_times(*args, axis=None, events=None, x_lim=None, fig_size=(10, 2), show=True, **kwargs)
    .. automethod:: plot_psd(axis=None, x_lim=None, y_lim=None, show=True, *args, fig_size=(10, 3), nperseg=None, colors=sns.color_palette(), **kwargs)

    .. centered:: Channel Methods
    .. automethod:: set_ch_names
    .. automethod:: set_ch_types
    .. automethod:: remove_ch
    .. automethod:: channel_reference
    .. automethod:: common_reference

    .. centered:: Filtering Methods
    .. automethod:: filter_fir
    .. automethod:: filter_iir
    .. automethod:: filter_median
    .. automethod:: filter_gaussian
    .. automethod:: filter_powerline

    .. centered:: Utility Methods
    .. automethod:: _time_to_index
    .. automethod:: _time_lim_validate
    .. automethod:: _ch_to_index
    .. automethod:: _ch_type_to_index
    .. automethod:: _to_plt_line_collection
    .. automethod:: _to_mne_raw

    .. centered:: Property Methods




