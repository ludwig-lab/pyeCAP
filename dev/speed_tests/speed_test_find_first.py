import numpy as np
from numba import jit
import timeit
from tqdm import tqdm
from typing import Union
import random

# Function definitions

def find_first_true(vec: np.ndarray) -> int:
    """
    Return the index of the first occurrence of True in a boolean numpy array.

    Parameters:
    vec (np.ndarray): Boolean Numpy array in which to find the first occurrence of True.

    Returns:
    int: The index of the first True in the array, or -1 if no True value is found.
    """
    true_indices = np.where(vec)[0]
    return true_indices[0] if true_indices.size > 0 else -1

@jit(nopython=True)
def find_first_true_numba(vec: np.ndarray) -> int:
    """
    Return the index of the first occurrence of True in a boolean numpy array using Numba.

    This function uses Numba for just-in-time compilation, offering performance benefits
    for large arrays.

    Parameters:
    vec (np.ndarray): Boolean Numpy array in which to find the first occurrence of True.

    Returns:
    int: The index of the first True in the array, or -1 if no True value is found.
    """
    for idx in range(vec.size):
        if vec[idx]:
            return idx
    return -1

def find_first_general(array: np.ndarray, value: Union[int, float, str]) -> int:
    """
    Return the index of the first occurrence of a specified value in a numpy array.

    Parameters:
    array (np.ndarray): Numpy array in which to find the value.
    value (int, float, str): Value to find in the array.

    Returns:
    int: The index of the first occurrence of the value in the array, or -1 if not found.
    """
    indices = np.where(array == value)[0]
    return indices[0] if indices.size > 0 else -1

@jit(nopython=True)
def find_first_general_numba(array: np.ndarray, value: Union[int, float, str]) -> int:
    """
    Return the index of the first occurrence of a specified value in a numpy array using Numba.

    This function uses Numba for just-in-time compilation to enhance performance,
    especially useful for large arrays.

    Parameters:
    array (np.ndarray): Numpy array in which to find the value.
    value (int, float, str): Value to find in the array.

    Returns:
    int: The index of the first occurrence of the value in the array, or -1 if not found.
    """
    for idx in range(array.size):
        if array[idx] == value:
            return idx
    return -1

def create_boolean_array(size, num_true, true_positions):
    arr = np.zeros(size, dtype=bool)
    for pos in true_positions:
        arr[pos] = True
    return arr

# Parameters to test
array_lengths = [10000, 100000, 1000000, 10000000]
num_trues = [1, 10, 100, 1000, 5000]
iterations = 100

# Measure compilation time for Numba functions once
test_array = create_boolean_array(100, 1, [50])  # Small array for compilation
start_time = timeit.default_timer()
find_first_true_numba(test_array)
compilation_time_true_numba = timeit.default_timer() - start_time

start_time = timeit.default_timer()
find_first_general_numba(test_array, True)
compilation_time_general_numba = timeit.default_timer() - start_time

# Store results
results = []

# Single progress bar for all benchmarks
total_steps = len(array_lengths) * len(num_trues) * 4 * iterations
with tqdm(total=total_steps, desc="Benchmarking Functions") as pbar:
    for length in array_lengths:
        for num_true in num_trues:
            true_positions = sorted(random.sample(range(length), num_true))
            test_array = create_boolean_array(length, num_true, true_positions)

            # Initialize times
            numba_time = general_numba_time = general_time = find_first_true_time = 0

            for _ in range(iterations):
                numba_time += timeit.timeit(lambda: find_first_true_numba(test_array), number=1)
                pbar.update(1)
                general_numba_time += timeit.timeit(lambda: find_first_general_numba(test_array, True), number=1)
                pbar.update(1)
                general_time += timeit.timeit(lambda: find_first_general(test_array, True), number=1)
                pbar.update(1)
                find_first_true_time += timeit.timeit(lambda: find_first_true(test_array), number=1)
                pbar.update(1)

            results.append((length, num_true, numba_time, general_numba_time, general_time, find_first_true_time))

# Print compilation times
print("Compilation time for find_first_true_numba:", compilation_time_true_numba)
print("Compilation time for find_first_general_numba:", compilation_time_general_numba)

# Print benchmark results in an organized way
for length, num_true, numba_time, general_numba_time, general_time, find_first_true_time in results:
    print(f"\nArray Length: {length}, Number of Trues: {num_true}")
    print(f"Specialized Boolean Numba function time: {numba_time}")
    print(f"General Numba function time: {general_numba_time}")
    print(f"General function time: {general_time}")
    print(f"find_first_true function time: {find_first_true_time}")