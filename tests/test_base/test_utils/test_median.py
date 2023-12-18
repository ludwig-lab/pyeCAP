import dask.array as da
import numpy as np
import pytest
from scipy.ndimage import median_filter

from pyeCAP.base.utils.median import *


def simple_median(arr):
    return np.sort(arr)[arr.size // 2]


def simple_rolling_median(data, kernel_size):
    edge = kernel_size // 2
    median_filtered = np.empty_like(data)
    for i in range(edge, data.size - edge):
        median_filtered[i] = simple_median(data[i - edge : i + edge + 1])
    # Manually implement padding
    median_filtered[:edge] = data[:edge]
    median_filtered[-edge:] = data[-edge:]
    return median_filtered


# Test heap functions


def test_heap_operations():
    # Test heappush_min and heappop_min
    heap = [0] * 10
    end = 0
    for item in [5, 3, 8, 1, 2]:
        add_to_min_heap(heap, item, end)
        end += 1
        assert is_min_heap(
            heap[:end]
        ), f"Min heap property violated after heappush_min with item {item}"

    sorted_items = []
    while end > 0:
        item = pop_from_min_heap(heap, end)
        end -= 1
        sorted_items.append(item)
        assert is_min_heap(heap[:end]), "Min heap property violated after heappop_min"

    assert sorted_items == sorted(
        sorted_items
    ), "Heappop_min did not return items in sorted order"

    # Test heappush_max and heappop_max
    heap = [0] * 10
    end = 0
    for item in [5, 3, 8, 1, 2]:
        add_to_max_heap(heap, item, end)
        end += 1
        assert is_max_heap(
            heap[:end]
        ), f"Max heap property violated after heappush_max with item {item}"

    sorted_items = []
    while end > 0:
        item = pop_from_max_heap(heap, end)
        end -= 1
        sorted_items.append(item)
        assert is_max_heap(heap[:end]), "Max heap property violated after heappop_max"

    assert sorted_items == sorted(
        sorted_items, reverse=True
    ), "Heappop_max did not return items in sorted order"

    return "All heap operation tests passed"


# Test median_of_stream function


def test_median_of_stream():
    # Test case 1: median of a single-element stream
    data = np.array([1])
    assert median_of_stream(data)[0] == 1, "Test case 1 failed"

    # Test case 2: median of a two-element stream
    data = np.array([1, 2])
    assert median_of_stream(data)[-1] == 1.5, "Test case 2 failed"

    # Test case 3: median of a three-element stream
    data = np.array([1, 2, 3])
    assert median_of_stream(data)[-1] == 2, "Test case 3 failed"

    # Test case 4: median of a stream with repeated elements
    data = np.array([1, 1, 1])
    assert median_of_stream(data)[-1] == 1, "Test case 4 failed"

    # Test case 5: median of a stream with negative elements
    data = np.array([-1, -2, -3])
    assert median_of_stream(data)[-1] == -2, "Test case 5 failed"


def test_median_of_stream_long():
    # Test case 1: median of a long stream of consecutive integers
    data = np.arange(1, 101)
    assert np.isclose(median_of_stream(data)[-1], np.median(data)), "Test case 1 failed"

    # Test case 2: median of a long stream of random integers
    data = np.random.randint(1, 101, size=100)
    assert np.isclose(median_of_stream(data)[-1], np.median(data)), "Test case 2 failed"

    # Test case 3: median of a long stream of random floats
    data = np.random.uniform(1, 101, size=100)
    assert np.isclose(median_of_stream(data)[-1], np.median(data)), "Test case 3 failed"


def test_calculate_median():
    # Test case 1: median of an odd-sized array
    data = np.array([1, 2, 3])
    assert calculate_median(data) == np.median(data), "Test case 1 failed"

    # Test case 2: median of an even-sized array
    data = np.array([1, 2, 3, 4])
    assert calculate_median(data) == np.median(data), "Test case 2 failed"

    # Test case 3: median of an array with negative numbers
    data = np.array([-1, -2, -3, -4])
    assert calculate_median(data) == np.median(data), "Test case 3 failed"

    # Test case 4: median of an array with repeated numbers
    data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    assert calculate_median(data) == np.median(data), "Test case 4 failed"

    # Test case 5: median of a large array
    data = np.random.rand(10000)
    assert np.isclose(calculate_median(data), np.median(data)), "Test case 5 failed"


def test_heappush_heappop():
    heap = np.zeros(10)
    end = 0

    # Test heappush
    values_to_push = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    for val in values_to_push:
        heappush_min(heap, val, end)
        end += 1
        assert heap[0] == min(heap[:end]), "Heap property violated after heappush"

    # Test heappop
    sorted_values = []
    while end > 0:
        val = heappop_min(heap, end)
        end -= 1
        sorted_values.append(val)

    assert sorted_values == sorted(
        values_to_push
    ), "Heappop did not return elements in sorted order"

    return "Heappush and Heappop tests passed"


# Test cases for heappush and heappop
def test_heap_operations():
    # Test heappush
    heap = [0] * 10
    end = 0
    for item in [5, 3, 8, 1, 2]:
        heappush_min(heap, item, end)
        end += 1
        assert is_min_heap(
            heap[:end]
        ), f"Heap property violated after heappush with item {item}"

    # Test heappop
    sorted_items = []
    while end > 0:
        item = heappop_min(heap, end)
        end -= 1
        sorted_items.append(item)
        assert is_min_heap(heap[:end]), "Heap property violated after heappop"

    # Check if the sorted_items list is sorted
    assert sorted_items == sorted(
        sorted_items
    ), "Heappop did not return items in sorted order"

    return "All heap operation tests passed"


def test_heappush():
    # Test heappush_min
    heap = np.zeros(10)
    end = 0
    for item in [5, 3, 8, 1, 2]:
        heappush_min(heap, item, end)
        end += 1
        print(heap)
        assert is_min_heap(
            heap[:end]
        ), f"Min heap property violated after heappush_min with item {item}"
        assert (
            len(heap[:end]) == end
        ), f"Heap length incorrect after heappush_min with item {item}"
        assert heap[0] == min(
            heap[:end]
        ), f"Top of min heap incorrect after heappush_min with item {item}"

    # Test heappush_max
    heap = np.zeros(10)
    end = 0
    for item in [5, 3, 8, 1, 2]:
        heappush_max(heap, item, end)
        end += 1
        print(heap)
        assert is_max_heap(
            heap[:end]
        ), f"Max heap property violated after heappush_max with item {item}"
        assert (
            len(heap[:end]) == end
        ), f"Heap length incorrect after heappush_max with item {item}"
        assert heap[0] == max(
            heap[:end]
        ), f"Top of max heap incorrect after heappush_max with item {item}"


def test_siftup_max_and_siftdown_max():
    # Testing _siftup_max and _siftdown_max individually
    test_results = []

    # Test _siftdown_max
    for item in [5, 3, 8, 1, 2]:
        heap = [0] * 5
        heap[0] = item
        siftdown_max(heap, 0, 1)
        test_results.append((heap[0] == item, f"_siftdown_max with item {item}"))

    # Test _siftup_max
    for item in [5, 3, 8, 1, 2]:
        heap = [item] + [0] * 4
        end = 1
        siftup_max(heap, 0, end)
        test_results.append((heap[0] == item, f"_siftup_max with item {item}"))

    # Collect results
    failed_tests = [result for result in test_results if not result[0]]
    if failed_tests:
        return f"Failed tests: {failed_tests}"
    else:
        return "All _siftup_max and _siftdown_max tests passed"


def test_remove_from_heap():
    # Test case 1: remove from min heap
    heap = np.array([1, 2, 3, 4, 5])
    end = len(heap)
    item = 3
    expected = np.array([1, 2, 5, 4])
    end = remove_from_min_heap(heap, end, item)
    assert np.all(heap[:end] == expected), "Test case 1 failed"

    # Test case 2: remove from max heap
    heap = np.array([5, 4, 3, 2, 1])
    end = len(heap)
    item = 3
    expected = np.array([5, 4, 1, 2])
    end = remove_from_max_heap(heap, end, item)
    assert np.all(heap[:end] == expected), "Test case 2 failed"

    # Test case 3: remove item not in heap
    heap = np.array([1, 2, 3, 4, 5])
    end = len(heap)
    item = 6
    expected = np.array([1, 2, 3, 4, 5])  # heap should remain unchanged
    end = remove_from_min_heap(heap, end, item)
    assert np.all(heap[:end] == expected), "Test case 3 failed"

    # Test case 4: remove from empty heap
    heap = np.array([])
    end = len(heap)
    item = 1
    expected = np.array([])  # heap should remain unchanged
    end = remove_from_max_heap(heap, end, item)
    assert np.all(heap[:end] == expected), "Test case 4 failed"


def test_add_number_to_heaps():
    min_heap = np.zeros(10)
    max_heap = np.zeros(10)
    min_heap_end = 0
    max_heap_end = 0

    # Test adding to an empty heap
    min_heap, max_heap, min_heap_end, max_heap_end = add_number_to_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end, 5
    )
    assert max_heap_end == 1
    assert min_heap_end == 0
    assert get_max_heap_top(max_heap) == 5

    # Test adding a smaller number
    min_heap, max_heap, min_heap_end, max_heap_end = add_number_to_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end, 3
    )
    assert max_heap_end == 2
    assert min_heap_end == 0
    assert get_max_heap_top(max_heap) == 5

    # Test adding a larger number
    min_heap, max_heap, min_heap_end, max_heap_end = add_number_to_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end, 8
    )
    assert max_heap_end == 2
    assert min_heap_end == 1
    assert get_min_heap_top(min_heap) == 8

    # Test adding a number equal to the max heap top
    min_heap, max_heap, min_heap_end, max_heap_end = add_number_to_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end, 5
    )
    assert max_heap_end == 2
    assert min_heap_end == 2
    assert get_min_heap_top(min_heap) == 5


def test_rebalance_heaps():
    # Initialize two heaps
    min_heap = np.array([5, 6, 7, 8, 9, 0])
    max_heap = np.array([1, 2, 3, 4, 0, 0])
    min_heap_end = 5
    max_heap_end = 4

    # Call the function to rebalance the heaps
    min_heap, max_heap, min_heap_end, max_heap_end = rebalance_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end
    )

    # Check if the heaps are correctly rebalanced
    assert max_heap_end - min_heap_end in [0, 1]
    assert min_heap_end == 4
    assert max_heap_end == 5
    assert np.all(np.sort(min_heap[:min_heap_end]) == np.array([6, 7, 8, 9]))
    assert np.all(np.sort(max_heap[:max_heap_end]) == np.array([1, 2, 3, 4, 5]))

    # Now, let's make max_heap larger than min_heap and rebalance again
    max_heap = np.array([6, 1, 2, 3, 4, 5])
    max_heap_end = 6

    # Call the function to rebalance the heaps
    min_heap, max_heap, min_heap_end, max_heap_end = rebalance_heaps(
        min_heap, max_heap, min_heap_end, max_heap_end
    )

    # Check if the heaps are correctly rebalanced
    assert max_heap_end - min_heap_end in [0, 1]
    assert min_heap_end == 5
    assert max_heap_end == 5
    assert np.all(np.sort(min_heap[:min_heap_end]) == np.array([6, 6, 7, 8, 9]))
    assert np.all(np.sort(max_heap[:max_heap_end]) == np.array([1, 2, 3, 4, 5]))


def test_remove_oldest_number():
    # Test case 1: oldest_number is in max_heap
    max_heap = np.array([5, 3, 2])
    min_heap = np.array([6, 7, 8])
    max_heap_end = 3
    min_heap_end = 3
    oldest_number = 5
    min_heap, max_heap, min_heap_end, max_heap_end = remove_oldest_number(
        min_heap, max_heap, min_heap_end, max_heap_end, oldest_number
    )
    assert 5 not in max_heap[:max_heap_end]

    # Test case 2: oldest_number is in min_heap
    max_heap = np.array([5, 3, 2])
    min_heap = np.array([6, 7, 8])
    max_heap_end = 3
    min_heap_end = 3
    oldest_number = 6
    min_heap, max_heap, min_heap_end, max_heap_end = remove_oldest_number(
        min_heap, max_heap, min_heap_end, max_heap_end, oldest_number
    )
    assert 6 not in min_heap[:min_heap_end]


def test_calculate_roll_median():
    # Test case 1: max_heap has more elements than min_heap
    max_heap = np.array([5, 3, 2])
    min_heap = np.array([6, 7, 8])
    max_heap_end = 3
    min_heap_end = 2
    assert calculate_roll_median(min_heap, max_heap, min_heap_end, max_heap_end) == 5

    # Test case 2: min_heap has more elements than max_heap
    max_heap = np.array([5, 3, 2])
    min_heap = np.array([6, 7, 8, 9])
    max_heap_end = 3
    min_heap_end = 4
    assert calculate_roll_median(min_heap, max_heap, min_heap_end, max_heap_end) == 5.5

    # Test case 3: max_heap and min_heap have equal number of elements
    max_heap = np.array([5, 3, 2])
    min_heap = np.array([6, 7, 8])
    max_heap_end = 3
    min_heap_end = 3
    assert calculate_roll_median(min_heap, max_heap, min_heap_end, max_heap_end) == 5.5

    # Test case 5: max_heap has one element, min_heap is empty
    max_heap = np.array([5])
    min_heap = np.array([])
    max_heap_end = 1
    min_heap_end = 0
    assert calculate_roll_median(min_heap, max_heap, min_heap_end, max_heap_end) == 5


def test_rolling_median():
    # Create a 1D numpy array
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    kernel_size = 3

    # Apply the median filter
    result = rolling_median(data, kernel_size)

    # Expected result after applying the median filter
    expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Check if the result matches the expected result
    assert np.array_equal(
        result, expected_result
    ), "The median filter did not produce the expected result"

    # Test with a larger kernel size
    kernel_size = 5
    result = rolling_median(data, kernel_size)
    expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.array_equal(
        result, expected_result
    ), "The median filter did not produce the expected result with a larger kernel size"

    # Test with an array of odd length
    data = np.array([1, 2, 3, 4, 5])
    kernel_size = 3
    result = rolling_median(data, kernel_size)
    expected_result = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(
        result, expected_result
    ), "The median filter did not produce the expected result"

    # Test with an array of even length
    data = np.array([1, 2, 3, 4, 5, 6])
    result = rolling_median(data, kernel_size)
    expected_result = np.array([1, 2, 3, 4, 5, 6])
    assert np.array_equal(
        result, expected_result
    ), "The median filter did not produce the expected result"

    # Test with a larger kernel size
    kernel_size = 5
    result = rolling_median(data, kernel_size)
    expected_result = np.array([1, 2, 3, 4, 5, 6])
    assert np.array_equal(
        result, expected_result
    ), "The median filter did not produce the expected result with a larger kernel size"

    # Test on short data set
    numbers = np.array([1, 3, -1, 5, 2, 6, 3, 7, 4, 8])
    window_size = 3
    # Corrected expected medians
    expected_medians = np.array([1, 1, 3, 2, 5, 3, 6, 4, 7, 8])
    calculated_medians = rolling_median(numbers, window_size)
    assert np.allclose(
        calculated_medians, expected_medians
    ), "Calculated medians do not match expected medians"


@pytest.mark.parametrize("array_size, kernel_size", [(10000, 13), (20000, 21)])
def test_rolling_median_large(array_size, kernel_size):
    # Generate a large random array
    data = np.random.rand(array_size).astype(np.float32) * 200

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Calculate rolling medians
    result1 = rolling_median(data, kernel_size)
    result2 = simple_rolling_median(data, kernel_size)

    # Check if the results are close enough
    if not np.allclose(result1, result2, atol=1e-5):
        diff_indices = np.where(np.abs(result1 - result2) > 1e-5)
        assert (
            False
        ), f"Differences found at indices: {diff_indices} /n Values in result1: {result1[diff_indices]} /n Values in result2: {result2[diff_indices]}"


def test_heap_empty():
    heap = []

    # Test heappop on empty heap
    with pytest.raises(IndexError):
        heappop_min(heap)

    # Test _siftdown on empty heap
    with pytest.raises(IndexError):
        siftdown_min(heap, 0, 0)

    # Test _siftup on empty heap
    with pytest.raises(IndexError):
        siftup_min(heap, 0)
