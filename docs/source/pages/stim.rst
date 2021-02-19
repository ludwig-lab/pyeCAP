Stim
====

The Stim class along with parent classes handle stimulation metadata, parameters, and times of stimulation events. This
data can be used to view times at which stimulation occurred, stimulation amplitudes, stimulation frequencies, and more.
The Stim class uses pandas DataFrames to view these parameters. Similar to the Ephys class, there are plotting methods
as well as methods for working with channels.

.. TODO: Get stim data that is not empty, use it to add more doctests

.. testsetup::

    import pyCAP, os
    with pyCAP.pyCAP.hide_print():
        stim_data = pyCAP.Stim(os.path.join('..', 'tests', 'data', 'TDT', 'SmallData'))

.. automodule:: pyCAP.stim
    :members:
    :special-members:


Stim parent classes
^^^^^^^^^^^^^^^^^^^
The Stim class has three parent classes to handle stimulation events, parameters, and dio data. The inheritance order
for Stim objects is _EventData first, _DioData second, and _ParameterData third.

Event Data class
................

.. automodule:: pyCAP.base.event_data
    :members:
    :private-members: _EventData

Dio Data class
..............

.. automodule:: pyCAP.base.dio_data
    :members:
    :private-members: _DioData

Parameter Data class
....................

.. automodule:: pyCAP.base.parameter_data
    :members:
    :private-members: _ParameterData