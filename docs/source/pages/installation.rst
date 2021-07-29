Installation Guide
==================

.. TODO: add pyns-python3 to installation for complete install.

Through pyPI
^^^^^^^^^^^^

The simplest way to install pyeCAP is through pyPI, which is often automatically installed with python. To install pyeCAP,
open a terminal window such as bash, powershell, or cmd.exe. Run the command 'pip install pyeCAP' or 'pip3 install pyeCAP' to automatically
download and install pyeCAP along with all the dependencies.

It is not recommended to install pyeCAP in the base python environment because pyPI may interfere with other packages and create dependency conflicts.
Instead, it is recommended to install pyeCAP in a new virtual environment. Anaconda and the python package venv
provide ways to manage virtual environments. For more information, see https://www.anaconda.com/ and https://docs.python.org/3/library/venv.html.

Through GitHub
^^^^^^^^^^^^^^

To view the source code on GitHub as well as examples in jupyter notebook, go to https://github.com/ludwig-lab/pyeCAP.
Download the repository into a file folder of your choice. pyeCAP can also be installed after downloading the source
code by navigating to the pyeCAP directory in a terminal window, ensuring that the terminal is operating in the correct
virtual environment, and then running 'pip install .' or 'python setup.py install'.

Through a Web Browser
^^^^^^^^^^^^^^^^^^^^^

pyeCAP is coming soon to Google Colab.
.. To run pyeCAP in a web browser, go to the github page at https://github.com/ludwig-lab/pyeCAP, then click
on the Binder link in the ReadMe file. It will take a few minutes to load the environment. After the environment is loaded,
it is now possible to run the example jupyter notebooks in a web browser. A small example data set is provided.
