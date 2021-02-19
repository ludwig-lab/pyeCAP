# python standard library imports
from collections.abc import Iterable

# scientific computing library imports
import numpy as np


def _is_iterable(obj, type=None):
    if type is None:
        return isinstance(obj, Iterable)
    else:
        return all([isinstance(o, type) for o in obj])

def _to_array(array, dtype=None):
    if dtype is None:
        return np.asarray(array).flatten()
    else:
        return np.asarray(array).astype(dtype).flatten()