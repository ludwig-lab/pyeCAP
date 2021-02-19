pyCAP Modules
=============
.. TODO: write docstrings for each class and method to include in the documentation
.. TODO: create more automodules if necessary

This page documents the pyCAP classes and their methods.

*Note that pyCAP objects are never modified in place. Each 'property' method is a getter method that allows properties
to be accessed as normal attributes but not modified. All other methods will return new pyCAP objects with different
properties or non-pyCAP objects.*

.. toctree::
    :maxdepth: 2

    ephys.rst
    phys.rst
    stim.rst
    ecap.rst
    physresponse.rst
