Stim
====

The Stim class along with parent classes handle stimulation metadata, parameters, and times of stimulation events. This
data can be used to view times at which stimulation occurred, stimulation amplitudes, stimulation frequencies, and more.
The Stim class uses pandas DataFrames to view these parameters. Similar to the Ephys class, there are plotting methods
as well as methods for working with channels.

.. TODO: Get stim data that is not empty, use it to add more doctests

.. testsetup::

    import pyeCAP, os
    with pyeCAP.pyeCAP.hide_print():
        stim_data = pyeCAP.Stim(os.path.join('..', 'tests', 'data', 'TDT', 'SmallData'))

.. automodule:: pyeCAP.stim
    :members:
    :special-members:


Stim parent classes
^^^^^^^^^^^^^^^^^^^
The Stim class has three parent classes to handle stimulation: events, parameters, and Dio data. The inheritance order
for Stim objects is _EventData first, _DioData second, and _ParameterData third. The event data class keeps track of the
time of each individual stimulation pulse. The Digital I/O (Dio) class keeps track of the start and stop time of a group
of pulses. The parameter data class keeps track of stimulation parameters such as pulse amplitude and pulse count.

Event Data class
................

.. automodule:: pyeCAP.base.event_data
    :members:
    :private-members: _EventData

Dio Data class
..............

.. automodule:: pyeCAP.base.dio_data
    :members:
    :private-members: _DioData

Parameter Data class
....................

.. automodule:: pyeCAP.base.parameter_data
    :members:
    :private-members: _ParameterData