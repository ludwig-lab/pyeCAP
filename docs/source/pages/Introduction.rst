Introduction
=============

Organization
^^^^^^^^^^^^

PyCAP is organized into four major classes that you user will interact with. These are:

- Ephys  A class for working with Ephys data. This class handles mapping ephys data sets from disk for fast analysis, preprocessing, and visualization of data. Important features include: Loading Data
- Interactive Visualization and Plotting
- Re-referencing strategies
- Filtering
- Stim - A class for working with Stimulation data. This class handles the details of reading in stimulation data from different stimulation/recording system. Timing of stimulation pulse trains, individual pulses, and stimulation parameters can be easily worked with through the Stim class.
- ECAP - The ECAP class works with Ephys and Stim data and provides an interface for analysis and visualization of ECAP data.
- Phys - A class for working with physiological data such as heart rate and blood pressure.

Data type support
^^^^^^^^^^^^^^^^^
PyCAP currently supports analysis of data collected with systems developed by Tucker Davis Technologies https://www.tdt.com/, Ripple Neuro https://rippleneuro.com/, and ADInstruments https://www.adinstruments.com/.

Interactive data analysis
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to enable interactive data analysis and both real-time data as well as those stored on your computers hard disk, PyCAP is built on the concept of lazy execution. Electrophysiology analysis scripts written by scientists are often time consuming and memory intensive to run. This is because we are often working with large data sets and in order to preprocess our data we usually read the entire data set into memory and often create copies of the data after various filtering, referencing, artifact rejection, or other procedures. Additionally, although we normally preprocess an entire data set we often only work with and plot small portions of our data at a time.

PyCAP is built from the ground up to work differently. Data is mapped to it's location on your hard disk, without ever reading the data into memory. A data analysis pipeline, wich can consist of various preprocessing or data analysis techniques is then built by the user. At this point zero computation has been performed.

To interact with your data, at some point you will want to reduce processed data into some kind of summary statistics or visualization. At this point PyCAP evaluates your data analysis procedure and determines with data needs to be accessed to return the request result. In this way it only reads from disk and analyzes the section of data that are relevant to the reduced data set on which you are performing statistics or creating visualizations from.

Chunking data in this way allows for minimizes unnecessary computations, but also allows us to easily parallelize computations across multiple chunks of data in order to take advantage of parallel processing and further accelerate your data analysis. Lastly, because chunks of data are not all stored in memory at once. PyCAP allows you to work with large data sets on even modest computer hardware and even work on data sets that are larger than your available memory.

For those interested. PyCAP mainly takes advantage of a python project called Dask to parallelize operations on chunked data sets. More information on Dask and lazy evaluation can be found here: https://docs.dask.org/en/latest/