# python standard library imports
import math
from typing import List, Union

import dask.array as da

# scientific computing library imports
import numpy as np
from numba import jit, njit
from numba.typed import List


@njit(nogil=True)
def heappush_min(heap, item, end):
    heap[end] = item
    siftdown_min(heap, 0, end)


@njit(nogil=True)
def heappop_min(heap, end):
    lastelt = heap[end - 1]
    if end > 1:
        returnitem = heap[0]
        heap[0] = lastelt
        siftup_min(heap, 0, end - 1)
        return returnitem
    return lastelt


def is_min_heap(heap, end):
    if len(heap) == 0:
        return True
    for i in range(end):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < end and heap[i] > heap[left]:
            return False
        if right < end and heap[i] > heap[right]:
            return False
    return True


@njit(nogil=True)
def heapreplace_min(heap, end, item):
    """Pop and return the current smallest value, and add the new item."""
    if end == 0:
        raise IndexError("Heap is empty")
    returnitem = heap[0]
    heap[0] = item
    siftup_min(heap, 0, end)
    return returnitem


@njit(nogil=True)
def heappushpop_min(heap, end, item):
    """Fast version of a heappush followed by a heappop."""
    if end > 0 and heap[0] < item:
        item, heap[0] = heap[0], item
        siftup_min(heap, 0, end)
    return item


@njit(nogil=True)
def siftdown_min(heap, startpos, pos):
    newitem = heap[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


@njit(nogil=True)
def siftup_min(heap, pos, endpos):
    startpos = pos
    newitem = heap[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    siftdown_min(heap, startpos, pos)


@njit(nogil=True)
def remove_from_min_heap(heap, end, item):
    index = -1
    for i in range(end):
        if heap[i] == item:
            index = i
            break
    if index == -1:
        return end
    elif index == end - 1:
        end -= 1
        return end
    else:
        heap[index] = heap[end - 1]
        end -= 1
        if index == 0 or heap[index] > heap[(index - 1) >> 1]:
            siftup_min(heap, index, end)
            return end
        else:
            # Check if we need to sift up
            parent_index = (index - 1) >> 1
            while index > 0 and heap[index] < heap[parent_index]:
                heap[index], heap[parent_index] = heap[parent_index], heap[index]
                index = parent_index
                parent_index = (index - 1) >> 1
        return end


@njit(nogil=True)
def rolling_median_very_naive(numbers, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    min_heap = np.zeros(window_size, dtype=numbers.dtype)
    max_heap = np.zeros(window_size, dtype=numbers.dtype)
    medians = np.zeros_like(numbers)
    edge = window_size // 2
    # Start heap counters at 1 - accounting for first two valus being added
    min_heap_end = 1
    max_heap_end = 1
    # Initialize first two values manually
    if numbers[0] <= numbers[1]:
        # Smaller number should start in max heap
        max_heap[0] = -numbers[0]  # Negate the number
        min_heap[0] = numbers[1]
    else:
        max_heap[0] = -numbers[1]  # Negate the number
        min_heap[0] = numbers[0]

    for i in range(2, len(numbers)):
        # print(f"{i} - {i - edge} - Max heap end: {max_heap_end}-{is_min_heap(max_heap, max_heap_end)}, Min heap end: {min_heap_end}-{is_min_heap(min_heap,min_heap_end)}, Next: {numbers[i]}, Oldest: {numbers[i - window_size]}")
        # print(f"Max heap: {max_heap}\nMin heap: {min_heap}")
        if i >= window_size:
            # Remove oldest number from heaps
            oldest_number = numbers[i - window_size]
            if oldest_number <= -max_heap[0]:
                max_heap_end = remove_from_min_heap(
                    max_heap, max_heap_end, -oldest_number
                )  # Negate the number
                # The heaps now have equal size, which is good
            else:
                min_heap_end = remove_from_min_heap(
                    min_heap, min_heap_end, oldest_number
                )

        # Add new number into heaps
        newest_number = numbers[i]
        if (
            newest_number <= min_heap[0]
        ):  # If newest number is less the smallest value in the min_heap add to the max heap
            heappush_min(
                max_heap, -newest_number, max_heap_end
            )  # Store the negative of the number
            max_heap_end += 1
            # The max heap is now 1 element larger than the min_heap - which is what we expect
        else:  # Otherwise put on the min_heap
            heappush_min(min_heap, newest_number, min_heap_end)
            min_heap_end += 1

        # Rebalance Heaps
        while max_heap_end > min_heap_end + 1:
            heappush_min(
                min_heap, -heappop_min(max_heap, max_heap_end), min_heap_end
            )  # Negate the popped number
            max_heap_end -= 1
            min_heap_end += 1

        while min_heap_end > max_heap_end:
            heappush_min(
                max_heap, -heappop_min(min_heap, min_heap_end), max_heap_end
            )  # Negate the popped number
            min_heap_end -= 1
            max_heap_end += 1

        if i >= window_size - 1:
            medians[i - edge] = -max_heap[0]  # Negate the result

        # Handle edge cases by inserting origional data
        medians[:edge] = numbers[:edge]
        medians[-edge:] = numbers[-edge:]
    return medians
