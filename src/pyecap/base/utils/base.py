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
        return isinstance(obj, Iterable) and all(
            isinstance(element, element_type) for element in obj
        )


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


def _generate_state_identifier(properties):
    """
    Generate a unique identifier for a given set of properties.

    This function serializes each property into bytes. If a property has a '_state_identifier' attribute,
    it is encoded into bytes. Otherwise, the property is pickled into bytes. The bytes of all properties
    are then concatenated and hashed using SHA256 to generate a unique identifier.

    Parameters:
    properties (Iterable): An iterable of properties to generate the identifier for.

    Returns:
    str: A unique identifier for the given properties as a hexadecimal string.
    """
    import hashlib
    import pickle

    bytes_list = []
    for prop in properties:
        if hasattr(prop, "_state_identifier"):
            bytes_list.append(prop._state_identifier.encode())
        else:
            bytes_list.append(pickle.dumps(prop))

    # Hash the concatenated bytes
    return hashlib.sha256(b"".join(bytes_list)).hexdigest()
