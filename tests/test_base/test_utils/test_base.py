import hashlib
import pickle

import numpy as np
import pytest

from pyeCAP.base.utils.base import _generate_state_identifier, _is_iterable, _to_array

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


# Tests for _generate_state_identifier

# Define TestClass at the module level
# Test if the function correctly uses pickle.dumps for objects without _state_identifier attribute
class TestClassWithStateIdentifier:
    _state_identifier = "test"
    pass


class TestClassNoStateIdentifier:
    pass


def test_generate_state_identifier():
    properties = ["prop1", "prop2", {"prop3": "value3"}]
    # Test if the function returns a string
    assert isinstance(_generate_state_identifier(properties), str)


def test_generate_state_identifier_with_state_identifier():
    # Test if the function correctly uses pickle.dumps for objects without _state_identifier attribute
    properties = ["prop1", "prop2", {"prop3": "value3"}, TestClassWithStateIdentifier()]

    # Serialize properties into bytes in the same way as in _generate_state_identifier
    bytes_list = []
    for prop in properties:
        if hasattr(prop, "_state_identifier"):
            bytes_list.append(prop._state_identifier.encode())
        else:
            bytes_list.append(pickle.dumps(prop))

    # Compute the hash of the serialized properties
    expected_hash = hashlib.sha256(b"".join(bytes_list)).hexdigest()

    assert _generate_state_identifier(properties) == expected_hash


def test_generate_state_identifier_without_state_identifier():
    # Test if the function correctly uses pickle.dumps for objects without _state_identifier attribute
    properties = ["prop1", "prop2", {"prop3": "value3"}, TestClassNoStateIdentifier()]

    # Serialize properties into bytes in the same way as in _generate_state_identifier
    bytes_list = []
    for prop in properties:
        bytes_list.append(pickle.dumps(prop))

    # Compute the hash of the serialized properties
    expected_hash = hashlib.sha256(b"".join(bytes_list)).hexdigest()

    assert _generate_state_identifier(properties) == expected_hash
