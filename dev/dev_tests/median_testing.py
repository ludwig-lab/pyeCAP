import numpy as np

from pyeCAP.base.utils.median import rolling_median


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


# Generate a small random dataset
np.random.seed(0)  # for reproducibility
data = np.random.rand(300).astype(np.float32) * 200

# Set the window size
window_size = 21

# Print the input data
print("Input data:")
print(data)

# Call the rolling_median function
result = rolling_median(data, window_size)

# Print the result
print("\nResult from rolling_median:")
print(result)

# Call the simple_rolling_median function
simple_result = simple_rolling_median(data, window_size)

# Print the result
print("\nResult from simple_rolling_median:")
print(simple_result)

# Find the differences
diff_indices = np.where(result != simple_result)

# Print the differences
print("\nDifferences found at indices:")
print(diff_indices)
print("Values in result:")
print(result[diff_indices])
print("Values in simple_result:")
print(simple_result[diff_indices])
