# python standard library imports
import math

# scientific computing library imports
import numpy as np
from numba import jit


def _to_numeric_array(array, dtype=float):
    """Convert python objects to a 1D numeric array.

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
    """
    return np.asarray(array).astype(dtype).flatten()

@jit(nopython=True)
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
            area = math.fabs(
                (point_ax - avg_x)
                * (data[range_offs][1] - point_ay)
                - (point_ax - data[range_offs][0])
                * (avg_y - point_ay)
            ) * 0.5

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1

        sampled[i+1] = max_area_point  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)

    sampled[-1] = data[-1]  # Always add last

    return sampled

@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for idx, v in enumerate(vec):
        if item == v:
            return idx
    return -1

def _group_consecutive(data, stepsize=1):
    # Groups consecutive chunks of integers from an integer array into subarrays
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
