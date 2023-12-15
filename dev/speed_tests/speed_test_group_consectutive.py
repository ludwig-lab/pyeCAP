import itertools
import time
import types  # Add this line

import numba
import numpy as np
from tqdm import tqdm


# Definitions of various implementations of the function
def _group_consecutive_numpy(data: np.ndarray, stepsize: int = 1) -> list:
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


@numba.jit(nopython=True)
def _group_consecutive_numba(data, stepsize=1):
    start_idx = 0
    for i in range(1, len(data)):
        if data[i] - data[i - 1] != stepsize:
            yield data[start_idx:i]
            start_idx = i
    yield data[start_idx:]


def _group_consecutive_itertools(data, stepsize=1):
    groups = []
    for k, g in itertools.groupby(enumerate(data), lambda ix: ix[1] - stepsize * ix[0]):
        groups.append([x for i, x in g])
    return groups


def _group_consecutive_python(data, stepsize=1):
    # Check if the array is empty
    if data.size == 0:
        return []
    groups = [[data[0]]]
    for x in data[1:]:
        if x - groups[-1][-1] == stepsize:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def test_functions():
    # Define test cases
    test_cases = [
        (np.array([1, 2, 3, 5, 6, 7]), [np.array([1, 2, 3]), np.array([5, 6, 7])]),
        (
            np.array([1, 2, 3, 5, 6, 7, 9]),
            [np.array([1, 2, 3]), np.array([5, 6, 7]), np.array([9])],
        ),
        (
            np.array([1, 3, 5, 7]),
            [np.array([1]), np.array([3]), np.array([5]), np.array([7])],
        ),
        (np.array([]), []),
    ]

    # List of functions to test
    functions = [
        _group_consecutive_numpy,
        _group_consecutive_numba,
        _group_consecutive_itertools,
        _group_consecutive_python,
    ]

    # Test each function
    for func in functions:
        for input_array, expected_output in test_cases:
            output = func(input_array)
            if isinstance(output, np.ndarray):
                output = output.tolist()
            try:
                assert all(
                    np.array_equal(a, b) for a, b in zip(output, expected_output)
                ), f"Function {func.__name__} failed for input {input_array}. Expected {expected_output}, got {output}."
                print(
                    f"Test passed for function {func.__name__} with input {input_array}."
                )
            except AssertionError as e:
                print(str(e))


def generate_test_data(size, max_segment_length=100, max_step=1):
    """
    Generate an array of the given size with random consecutive segments.

    Parameters:
    size (int): The size of the array to generate.
    max_segment_length (int): The maximum length of a consecutive segment.
    max_step (int): The maximum step size between consecutive elements.

    Returns:
    np.ndarray: Array with random consecutive segments.
    """
    data = []
    while len(data) < size:
        segment_length = np.random.randint(1, max_segment_length + 1)
        start = np.random.randint(0, 100)
        segment = np.arange(start, start + segment_length * max_step, max_step)
        data.extend(
            segment[: max(0, size - len(data))]
        )  # Ensure we don't exceed the desired size
    return np.array(data)


# Function to benchmark the implementations
def benchmark_function(func, data):
    start_time = time.time()
    result = func(data)
    if isinstance(result, types.GeneratorType):  # Check if the result is a generator
        print("generator")
        result = list(result)  # Convert generator to list to force execution
    duration = time.time() - start_time
    return duration


# Call the test function
test_functions()

array_sizes = [
    100000,
    1000000,
    10000000,
    50000000,
    100000000,
]  # Added larger sizes for numpy and numba versions

results = []

compile_test_array = np.array([1, 2, 3, 5, 6, 7])
compilation_start = time.time()
list(_group_consecutive_numba(compile_test_array))
compilation_duration = time.time() - compilation_start
results.append(("Numba Compilation", "N/A", compilation_duration))

total_steps = len(array_sizes) * 4
with tqdm(total=total_steps, desc="Benchmarking Functions") as pbar:
    for size in array_sizes:
        test_array = generate_test_data(size)
        # Benchmark each function
        for func in [
            _group_consecutive_numpy,
            _group_consecutive_numba,
            _group_consecutive_itertools,
            _group_consecutive_python,
        ]:
            # Skip the larger sizes for itertools and python versions
            if size > 10000000 and func.__name__ in [
                "_group_consecutive_itertools",
                "_group_consecutive_python",
            ]:
                pbar.update(1)
                continue
            duration = benchmark_function(func, test_array)
            results.append((func.__name__, size, duration))
            pbar.update(1)

print("Function Name, Array Size, Duration")
for result in results:
    print(", ".join(str(x) for x in result))
