from collections.abc import Iterable
import os
import numpy as np


def _is_iterable(obj, types=None):
    try:
        iter(obj)
    except TypeError:
        return False

    if types is None:
        return True

    return all(isinstance(x, types) for x in obj)

def _to_array(array, dtype=None):
    if dtype is None:
        return np.asarray(array).flatten()
    else:
        return np.asarray(array).astype(dtype).flatten()