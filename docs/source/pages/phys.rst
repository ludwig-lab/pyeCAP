Phys
====

The Phys class works with physiological data such as heart rate and blood pressure. It inherits from the _TsData class
similarly to the Ephys class so the same methods will work on both classes. The Phys class reads data from ADInstruments
technology. Before attempting to read this data, export the raw .adicht file to a .mat file using software such as
LabChart. More information about exporting files can be found on the ADInstruments website at
https://www.adinstruments.com/

Phys objects can be instantiated with a path name to a file similarly to Ephys objects. Phys objects also have the option
to read one file into multiple data sets. Set mult_data to True in order to read in one file into multiple data sets.
The data sets are automatically split up by 'blocks' in the recording.

.. TODO: Add a physiological data sample to github, uncomment the test setup

..    .. testsetup::

        import eba_toolkit, os
        path = os.path.join()
        phys_data = eba_toolkit.Phys(path)

.. autoclass:: eba_toolkit.phys.Phys
    :members: units
    :special-members: __init__

Parent class
^^^^^^^^^^^^

The Parent class of the Phys class is the _TsData class. The majority of methods available to Phys objects are inherited
from this class.

* :ref:`_TsData (parent class)`

Reading data with the Phys class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Phys class reads data from non-hdf5 MatLab (.mat) files. Pre-processing of raw data is necessary to transform files
into this format. Software such as LabChart https://www.adinstruments.com/support/software will transform raw .adicht
files into .mat files. To create the MatLab file, open LabChart and open the raw data file. Click on the 'File'
dropdown in the upper left corner and select export. Set the file type to be "MATLAB" and select a file name and location.
A window will pop up with different export options. Choose the desired channels and make sure the 'upsample to same rate'
option is selected. This will ensure a consistent sample rate by interpolating data points. eba-toolkit currently does not
support data with different sampling rates between blocks or channels.

More information about how Labchart exports this data can be found here: https://www.adinstruments.com/support/knowledge-base/how-does-matlab-open-exported-data

Automatic Warnings
..................

When reading .mat files, eba-toolkit can generate warnings that will catch data that is improperly formatted
or not implemented. To enable these warnings, set the 'check' parameter to True when instantiating a new Phys object.
This parameter is only intended to generate warnings by reading data directly from files. This will not check data when
data and metadata are manually inputted into a Phys object. The constructor calls the check_data function documented below.
The current warnings are shown below:

- Array formatted improperly: This warning is generated if data is not exported to the matlab file correctly and the arrays have incorrect dimensions.

- NaN values in data: This warning is generated if there are NaN values in the data. This may cause problems when plotting data. Ensure that y limits are specified manually.

- Inconsistent sample rates: This warning is generated when the data has inconsistent sample rates. Reading this data is not yet implemented.

- Inconsistent units: This warning is generated when a channel has more than one unit across the experiment.

- Offset in data blocks: Warns about offsets in the start time of each channel. See the 'firstsampleoffset' array at https://www.adinstruments.com/support/knowledge-base/how-does-matlab-open-exported-data for more information.

Functions for reading in data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following functions can be used to read in data at a lower level.

.. automodule:: eba_toolkit.io.adinstruments_io
    :members: check_data, to_array, to_meta

Comment reading
...............

The following function can be used to read the comments from a file and record the timestamp, channel, and name of each
comment.

.. automodule:: eba_toolkit.phys
    :members: get_comments


