pyeCAP Modules
===================
.. TODO: write docstrings for each class and method to include in the documentation
.. TODO: create more automodules if necessary

This page documents the pyeCAP classes and their methods.

*Note that pyeCAP objects are never modified in place. Each 'property' method is a getter method that allows properties
to be accessed as normal attributes but not modified. All other methods will return new pyeCAP objects with different
properties or non-pyeCAP objects.*

.. toctree::
    :maxdepth: 2

    ephys.rst
    phys.rst
    stim.rst
    ecap.rst
    physresponse.rst
