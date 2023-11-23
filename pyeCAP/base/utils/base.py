# python standard library imports
from collections.abc import Iterable

# scientific computing library imports
import numpy as np


def _is_iterable(obj, element_type=None):
    """
    Check if an object is an iterable and, if element_type is specified, 
    whether all elements in it are of the specified type.

    Parameters:
    obj (any): The object to check.
    element_type (type, optional): The type to check against each element in the iterable. Defaults to None.

    Returns:
    bool: True if obj is an iterable and, if element_type is specified, all elements are of that type.
    """
    if element_type is None:
        return isinstance(obj, Iterable)
    else:
        return isinstance(obj, Iterable) and all(isinstance(element, element_type) for element in obj)


def _to_array(array, dtype=None):
    """
    Convert an input to a flattened numpy array of a specified type.

    Parameters:
    array (any): The input to convert. Should be an iterable or a scalar value.
    dtype (type, optional): The desired data type of the numpy array. Defaults to None.

    Returns:
    numpy.ndarray: A flattened numpy array of the specified type.

    Raises:
    ValueError: If the conversion to a numpy array fails.
    """
    try:
        np_array = np.asarray(array)

        # Check if dtype conversion is necessary
        if dtype is not None and np_array.dtype != dtype:
            np_array = np_array.astype(dtype)

        # Flatten the array if it's not already 1D
        if np_array.ndim != 1:
            np_array = np_array.flatten()

        return np_array
    except Exception as e:
        raise ValueError(f"Conversion to numpy array failed: {e}")
