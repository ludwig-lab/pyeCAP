# python standard library imports
import math
from typing import List, Union

import dask.array as da

# scientific computing library imports
import numpy as np
from numba import float32, int32, jit, njit
from numba.typed import List


def _to_numeric_array(array, dtype=float):
    """
    Convert python objects to a 1D numeric array.

    Converts a python object into a numeric numpy array. Utilizes numpy's np.asarray and np.astype in order to
    gracefully handle different object types as well as raise appropriate error messages. Always flattens result to
    a 1D array.

    Parameters
    ----------
    array : object
        Input object which will be converted to a numpy array.
    dtype : str, type
        The dtype of the array that will be returned.

    Returns
    -------
    np.ndarray
        A 1D numeric array with type given by 'dtype'.

    Raises
    ------
    ValueError
        If the conversion to a numpy array or the dtype conversion fails.
    """
    try:
        np_array = np.asarray(array)

        # Convert type if necessary
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)

        # Flatten the array if it's not already 1D
        if np_array.ndim != 1:
            np_array = np_array.flatten()

        return np_array
    except Exception as e:
        raise ValueError(f"Conversion to numeric array failed: {e}")


@njit(nogil=True)
def largest_triangle_three_buckets(data, threshold):
    """Return a downsampled version of data.

    Parameters
    ----------
    data: list of lists/tuples
        data must be formated this way: [[x,y], [x,y], [x,y], ...]
                                    or: [(x,y), (x,y), (x,y), ...]
    threshold: int
        threshold must be >= 2 and <= to the len of data
    Returns
    -------
    data, but downsampled using threshold
    """
    # chop off any NaN values at the beginning of the data because triangles can not be computed with NaN values
    i = 0
    found_pt = False
    while i < len(data) - 1 and not found_pt:
        if not np.isnan(data[i][0]):
            found_pt = True
        else:
            i += 1
    data = data[i:]

    # Bucket size. Leave room for start and end data points
    every = (len(data) - 2) / (threshold - 2)

    a = 0  # Initially a is the first point in the triangle
    next_a = 0
    max_area_point = np.zeros(2)

    sampled = np.zeros((threshold, 2))
    sampled[0] = data[0]  # Always add the first point

    for i in range(0, threshold - 2):
        # Calculate point average for next bucket (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_rang_end = avg_range_end if avg_range_end < len(data) else len(data)

        avg_range_length = avg_rang_end - avg_range_start

        while avg_range_start < avg_rang_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        # Get the range for this bucket
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        # Point a
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1

        while range_offs < range_to:
            # Calculate triangle area over three buckets
            area = (
                math.fabs(
                    (point_ax - avg_x) * (data[range_offs][1] - point_ay)
                    - (point_ax - data[range_offs][0]) * (avg_y - point_ay)
                )
                * 0.5
            )

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1

        sampled[i + 1] = max_area_point  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)

    sampled[-1] = data[-1]  # Always add last

    return sampled


@njit(nogil=True)
def find_first_true(vec: np.ndarray) -> int:
    """
    Return the index of the first occurrence of True in a boolean numpy array using Numba.

    This function uses Numba for just-in-time compilation, offering performance benefits
    for large arrays.

    Parameters:
    vec (np.ndarray): Boolean Numpy array in which to find the first occurrence of True.

    Returns:
    int: The index of the first True in the array, or -1 if no True value is found.
    """
    for i in range(vec.size):
        if vec[i]:
            return i
    return -1


@njit(nogil=True)
def find_first(vec: np.ndarray) -> int:
    """
    Return the index of the first occurrence of True in a boolean numpy array using Numba.

    This function uses Numba for just-in-time compilation, offering performance benefits
    for large arrays.

    Parameters:
    vec (np.ndarray): Boolean Numpy array in which to find the first occurrence of True.

    Returns:
    int: The index of the first True in the array, or -1 if no True value is found.
    """
    for i in range(array.size):
        if array[i] == value:
            return i
    return -1


def _group_consecutive(data: np.ndarray, stepsize: int = 1) -> list:
    """
    Groups consecutive chunks of integers from an integer array into subarrays.

    This function takes an array of integers and groups consecutive integers into subarrays.
    The consecutive integers are defined based on a specified step size. By default, consecutive
    integers are those which differ by 1. The function returns a list of numpy arrays, each
    containing a chunk of consecutive integers.

    Parameters:
    -----------
    data : np.ndarray
        An array of integers which will be grouped into consecutive chunks.

    stepsize : int, optional
        The step size to define consecutiveness. Consecutive integers are those which differ
        by the step size. The default is 1.

    Returns:
    --------
    list of np.ndarray
        A list containing subarrays of the input array. Each subarray consists of consecutive
        integers as defined by the stepsize.

    Examples:
    ---------
    >>> _group_consecutive(np.array([1, 2, 3, 5, 6, 7]))
    [array([1, 2, 3]), array([5, 6, 7])]

    >>> _group_consecutive(np.array([10, 20, 30, 50]), stepsize=10)
    [array([10, 20, 30]), array([50])]

    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def _get_size(obj):
    """
    Estimate the size of a numpy or dask array in bytes.

    Parameters
    ----------
    obj : numpy.ndarray or dask.array
        The array for which to estimate the size.

    Returns
    -------
    int
        The estimated size of the array in bytes.

    Raises
    ------
    TypeError
        If the input object is not a numpy or dask array.
    """

    if isinstance(obj, (np.ndarray, da.Array)):
        return obj.size * obj.dtype.itemsize
    else:
        raise TypeError("Input object must be a numpy or dask array.")
