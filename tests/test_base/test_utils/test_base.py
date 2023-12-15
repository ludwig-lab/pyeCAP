import numpy as np
import pytest

from pyeCAP.base.utils.base import _is_iterable, _to_array

# Tests for is_homogeneous_iterable


def test_is_iterable_with_homogeneous_list():
    assert _is_iterable([1, 2, 3], int)


def test_is_iterable_with_heterogeneous_list():
    assert not _is_iterable([1, "2", 3], int)


def test_is_iterable_with_non_iterable():
    assert not _is_iterable(42, int)


def test_is_iterable_without_type():
    assert _is_iterable([1, 2, 3])


# Tests for _to_array


def test_to_array_with_list():
    result = _to_array([1, 2, 3], int)
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_to_array_with_type_conversion():
    result = _to_array([1, 2, 3], float)
    assert result.dtype == float
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_array_with_non_iterable():
    result = _to_array(5, int)
    assert np.array_equal(result, np.array([5]))


def test_to_array_with_no_dtype():
    result = _to_array([1.0, 2.0, 3.0])
    assert result.dtype == float
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_array_error_handling():
    with pytest.raises(ValueError):
        _to_array("invalid", dtype=int)
