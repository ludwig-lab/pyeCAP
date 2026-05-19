import dask.array as da
import numpy as np
import pytest

from pyeCAP.base.utils.numeric import (
    _get_size,
    _to_numeric_array,
    find_first,
    find_first_true,
)

# Test _to_numeric_array


def test_to_numeric_array_with_int_list():
    result = _to_numeric_array([1, 2, 3], int)
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_to_numeric_array_with_float_conversion():
    result = _to_numeric_array([1, 2, 3], float)
    assert result.dtype == float
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_numeric_array_with_non_iterable():
    result = _to_numeric_array(5, int)
    assert np.array_equal(result, np.array([5]))


def test_to_numeric_array_with_no_dtype_change():
    input_array = np.array([1.0, 2.0, 3.0], dtype=float)
    result = _to_numeric_array(input_array)
    assert result is input_array  # Ensures no unnecessary copy was made


def test_to_numeric_array_error_handling():
    with pytest.raises(ValueError):
        _to_numeric_array("invalid", dtype=int)


# Tests for find_first_true


def test_find_first_true_with_true_present():
    array = np.array([False, False, True, False])
    assert find_first_true(array) == 2


def test_find_first_true_with_no_true():
    array = np.array([False, False, False, False])
    assert find_first_true(array) == -1


def test_find_first_true_empty_array():
    array = np.array([])
    assert find_first_true(array) == -1


# Tests for find_first


def test_find_first_with_value_present():
    array = np.array([1, 2, 3, 4])
    assert find_first(array, 3) == 2


def test_find_first_with_value_absent():
    array = np.array([1, 2, 3, 4])
    assert find_first(array, 5) == -1


def test_find_first_empty_array():
    array = np.array([])
    assert find_first(array, 1) == -1


def test_find_first_with_string():
    array = np.array(["apple", "banana", "cherry"])
    assert find_first(array, "banana") == 1


# Tests for _get_size


def test_get_size_numpy():
    # Create a numpy array
    np_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    # Calculate expected size: number of elements * size of each element
    expected_size = np_array.size * np_array.dtype.itemsize
    assert _get_size(np_array) == expected_size


def test_get_size_dask():
    # Create a dask array
    da_array = da.from_array([1, 2, 3, 4, 5], chunks=2)
    # Calculate expected size: number of elements * size of each element
    expected_size = da_array.size * da_array.dtype.itemsize
    assert _get_size(da_array) == expected_size


def test_get_size_invalid_input():
    # Test with an invalid input (not a numpy or dask array)
    with pytest.raises(TypeError):
        _get_size([1, 2, 3, 4, 5])


# Test median filter 1d function
