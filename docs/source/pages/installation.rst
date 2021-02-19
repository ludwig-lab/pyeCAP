Installation Guide
==================

.. TODO: add pyns-python3 to installation for complete install.
.. TODO: investigate a requirements.txt install 

Through pyPI
^^^^^^^^^^^^

The simplest way to install pyCAP is through pyPI, which is often automatically installed with python. To install pyCAP,
open a terminal window such as bash or powershell. Run 'pip' or 'pip3' to verify that pyPI is installed correctly. A
list of pip commands should appear. Run the command 'pip install pyCAP' or 'pip3 install pyCAP' to automatically
download and install pyCAP along with all the latest versions of all the dependencies.

It is not recommended to install pyCAP in the base python environment because pyPI may break other installations or
dependencies for other programs during the installation. Instead, it is recommended to install pyCAP and other packages
in a fresh virtual environment. Anaconda and the python package venv provide ways to manage virtual environments.
For more information, see https://www.anaconda.com/ and https://docs.python.org/3/library/venv.html.

Through GitHub
^^^^^^^^^^^^^^

To view the source code on GitHub as well as examples in jupyter notebook, go to [github link here]. Download the
repository into a file folder of your choice. Ensure that jupyter notebook is installed in the python environment,
either with anaconda or the command 'pip3 install jupyter'. pyCAP can also be installed after downloading the source
code by navigating to the pyCAP directory in a terminal window, ensuring that the terminal is operating in the correct
virtual environment, and then running 'pip install .' or 'python setup.py install'.
