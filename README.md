
# Introduction to eba-toolkit

## Disclaimer : version 0.0 of eba-toolkit is not a stable version and many of the methods have not been validated. This program is currently under development and all classes and methods may be subject to change.

Python is a python package for the analysis of evoked compound action potentials (CAPs) in elecytrophysiology data sets.

eba-toolkit was built with the goal of simplifying the collection and analysis of CAP data. As such this toolkit abstract the technicalities of loading, saving, and working with ephys data into an efficient class structure that forms the basis for fast and interactive analysis of CAPs. This enables real-time visualization of data during an ephys recording session to allow for optimizing experimental time and resources. The pCAP package also contains many visualization tools for fast and interactive visualization and analysis of data after an experiment.


## Organization of eba-toolkit

eba-toolkit is organization into three major classes that you user will interact with. These are:
* __Ephys__ - A class for working with Ephys data. This class handles mapping ephys data sets from disk for fast analysis, preprocessing, and visualization of data. Important features include:
 - _Loading Data_
 - _Interactive Visualization and Plotting_
 - _Re-referencing strategies_
 - _Filtering_
* __Stim__ - A class for working with Stimulation data. This class handles the details of reading in stimulation data from different stimulation/recording system. Timing of stimulation pulse trains, individual pulses, and stimulation parameters can be easily worked with through the Stim class.
* __ECAP__ - The ECAP class works with Ephys and Stim data and provides an interface for analysis and visualization of ECAP data. 
* __Phys__ - The Phys class analyzes physiological data along with stimulation data

## Supported Data Types

eba-toolkit currently supports analysis of data collected with systems developed by Tucker Davis Technologies (https://www.tdt.com/), and ADInstruments (https://www.adinstruments.com/). Ripple Neuro is coming soon (https://rippleneuro.com/).


## Enabling Interactive Data Analysis

In order to enable interactive data analysis and both real-time data as well as those stored on your computers hard disk, eba-toolkit is built on the concept of lazy execution. Electrophysiology analysis scripts written by scientists are often time consuming and memory intensive to run. This is because we are often working with large data sets and in order to preprocess our data we usually read the entire data set into memory and often create copies of the data after various filtering, referencing, artifact rejection, or other procedures. Additionally, although we normally preprocess an entire data set we often only work with and plot small portions of our data at a time. 

eba-toolkit is built from the ground up to work differently. Data is mapped to it's location on your hard disk, without ever reading the data into memory. A data analysis pipeline, wich can consist of various preprocessing or data analysis techniques is then built by the user. At this point zero computation has been performed. 

To interact with your data, at some point you will want to reduce processed data into some kind of summary statistics or visualization. At this point eba-toolkit evaluates your data analysis procedure and determines with data needs to be accessed to return the request result. In this way it only reads from disk and analyzes the section of data that are relevant to the reduced data set on which you are performing statistics or creating visualizations from.

Chunking data in this way allows for minimizes unnecessary computations, but also allows us to easily parallelize computations across multiple chunks of data in order to take advantage of parallel processing and further accelerate your data analysis. Lastly, because chunks of data are not all stored in memory at once. eba-toolkit allows you to work with large data sets on even modest computer hardware and even work on data sets that are larger than your available memory. 

For those interested. eba-toolkit mainly takes advantage of a python project called Dask to parallelize operations on chunked data sets. More information on Dask and lazy evaluation can be found here: https://docs.dask.org/en/latest/

## Installation

Installation is currently available through pyPI with the command 'pip install eba-toolkit'. Note that when importing the package, the command 'import eba_toolkit' is used instead of the hyphen.

eba-toolkit is also installable by downloading the repository, navigating to the repository with a command window, and running the command "python setup.py" or "python3 setup.py".

## Documentation

Coming soon to readthedocs.

Documentation can be built locally by downloading the repository. After downloading, open a command window and ensure that sphinx readthedocs theme is installed with the command "pip install sphinx-rtd-theme". Navigate to the docs folder of the repository with the terminal window and run the command "make html". The built documentation should be available as an html page in the docs/build directory. 

## Example Code
eba-toolkit currently has several example jupyter notebooks in the "examples" folder of the repository. Coming soon to Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ludwig-lab/eba-toolkit/new_dependencies)
