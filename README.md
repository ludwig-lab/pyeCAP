
# Introduction to pyeCAP

## Disclaimer : version 0.0 of pyeCAP is not a stable version and many of the methods have not been validated. This program is currently under development and all classes and methods may be subject to change.

pyeCAP is a python package for the analysis of evoked compound action potentials (CAPs) in elecytrophysiology data sets.

pyeCAP was built with the goal of simplifying the collection and analysis of CAP data. As such this toolkit abstract the technicalities of loading, saving, and working with ephys data into an efficient class structure that forms the basis for fast and interactive analysis of CAPs. This enables real-time visualization of data during an ephys recording session to allow for optimizing experimental time and resources. The pCAP package also contains many visualization tools for fast and interactive visualization and analysis of data after an experiment.


## Organization of pyeCAP

pyeCAP is organization into three major classes that you user will interact with. These are:
* __Ephys__ - A class for working with Ephys data. This class handles mapping ephys data sets from disk for fast analysis, preprocessing, and visualization of data. Important features include:
 - _Loading Data_
 - _Interactive Visualization and Plotting_
 - _Re-referencing strategies_
 - _Filtering_
* __Stim__ - A class for working with Stimulation data. This class handles the details of reading in stimulation data from different stimulation/recording system. Timing of stimulation pulse trains, individual pulses, and stimulation parameters can be easily worked with through the Stim class.
* __ECAP__ - The ECAP class works with Ephys and Stim data and provides an interface for analysis and visualization of ECAP data. 
* __Phys__ - The Phys class analyzes physiological data along with stimulation data

## Supported Data Types

pyeCAP currently supports analysis of data collected with systems developed by Tucker Davis Technologies (https://www.tdt.com/), and ADInstruments (https://www.adinstruments.com/). Ripple Neuro is coming soon (https://rippleneuro.com/).


## Enabling Interactive Data Analysis

In order to enable interactive data analysis and both real-time data as well as those stored on your computers hard disk, pyeCAP is built on the concept of lazy execution. Electrophysiology analysis scripts written by scientists are often time consuming and memory intensive to run. This is because we are often working with large data sets and in order to preprocess our data we usually read the entire data set into memory and often create copies of the data after various filtering, referencing, artifact rejection, or other procedures. Additionally, although we normally preprocess an entire data set we often only work with and plot small portions of our data at a time. 

pyeCAP is built from the ground up to work differently. Data is mapped to it's location on your hard disk, without ever reading the data into memory. A data analysis pipeline, wich can consist of various preprocessing or data analysis techniques is then built by the user. At this point zero computation has been performed. 

To interact with your data, at some point you will want to reduce processed data into some kind of summary statistics or visualization. At this point pyeCAP evaluates your data analysis procedure and determines with data needs to be accessed to return the request result. In this way it only reads from disk and analyzes the section of data that are relevant to the reduced data set on which you are performing statistics or creating visualizations from.

Chunking data in this way allows for minimizes unnecessary computations, but also allows us to easily parallelize computations across multiple chunks of data in order to take advantage of parallel processing and further accelerate your data analysis. Lastly, because chunks of data are not all stored in memory at once. pyeCAP allows you to work with large data sets on even modest computer hardware and even work on data sets that are larger than your available memory. 

For those interested. pyeCAP mainly takes advantage of a python project called Dask to parallelize operations on chunked data sets. More information on Dask and lazy evaluation can be found here: https://docs.dask.org/en/latest/

## Installation

Installation is currently available through pyPI with the command 'pip install pyeCAP'. Note that when importing the package, the command 'import eba_toolkit' is used instead of the hyphen.

pyeCAP is also installable by downloading the repository, navigating to the repository with a command window, and running the command "python setup.py install" or "python3 setup.py install".

## Documentation

Documentation is available on readthedocs:
https://pyecap.readthedocs.io/en/latest/

## Example Code
pyeCAP currently has several example jupyter notebooks in the "examples" folder of the repository.
Examples 1-4 are updated and examples 5-7 need updating.

Example notebooks can be accessed on binder:
(Warning: example data must be downloaded and unzipped which will take around 5 minutes. binder will time out after 10 minutes of inactivity and the notebooks will have to be restarted and the examples must be re-downloaded. Ensure that the notebook is being actively used to avoid having to re-download and re-create the virtual environment.)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ludwig-lab/pyeCAP/main)

or google colab:
https://colab.research.google.com/drive/1Y8j4L4T-_kpFrinS3OQhDpuxHW7VK9cO?usp=sharing (notebook 1)
https://colab.research.google.com/drive/1mZsibfkB_KoDlEeC2P0XIOZfCzqoTJ5v?usp=sharing (notebook 2)
https://colab.research.google.com/drive/16kwh1xFga7xYQWy2kuInwzE4qWjSNxzJ?usp=sharing (notebook 3)
https://colab.research.google.com/drive/1OhFqSQ8rvKrrdA_M83baACsxXuoWCRJk?usp=sharing (notebook 4)
