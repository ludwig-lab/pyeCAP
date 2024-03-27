# python standard library imports
import math
from typing import List, Union

import dask.array as da

# scientific computing library imports
import numpy as np
from numba import jit, njit, prange
from numba.typed import List


@njit(nogil=True)
def heappush_min(heap, item, end, heap_indices, tracking_index, index_window):
    heap[end] = item
    heap_indices[end] = tracking_index
    # Update index_window for the newly added item
    index_window[tracking_index] = end
    siftdown_min(heap, 0, end, heap_indices, index_window)
    return end + 1


@njit(nogil=True)
def heappop_min(heap, end, heap_indices, index_window):
    lastelt = heap[end - 1]
    lastindex = heap_indices[end - 1]
    if end > 1:
        returnitem = heap[0]
        returnindex = heap_indices[0]
        index_window[returnindex] = -1  # Element is removed
        heap[0] = lastelt
        heap_indices[0] = lastindex
        index_window[lastindex] = 0
        simple_siftup_min(heap, 0, end - 1, heap_indices, index_window)
        # Update index_window for the swapped elements
        return returnitem, returnindex, end - 1
    else:
        index_window[lastindex] = -1  # Element is removed
    return lastelt, lastindex, end - 1


@njit(nogil=True)
def heappushpop_min(heap, item, end, heap_indices, tracking_index, index_window):
    """Fast version of a heappush followed by a heappop."""
    if end > 1 and heap[0] < item:
        returnitem = heap[0]
        returnindex = heap_indices[0]
        index_window[returnindex] = -1  # Element is removed
        heap[0] = item
        heap_indices[0] = tracking_index
        index_window[tracking_index] = 0
        simple_siftup_min(heap, 0, end, heap_indices, index_window)
        return returnitem, returnindex
    return item, tracking_index


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
def siftdown_min(heap, startpos, pos, min_heap_indices, index_window):
    newitem = heap[pos]
    newindex = min_heap_indices[pos]
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos], min_heap_indices[pos] = (
                heap[parentpos],
                min_heap_indices[parentpos],
            )
            index_window[min_heap_indices[pos]] = pos
            pos = parentpos
            continue
        break
    heap[pos], min_heap_indices[pos] = newitem, newindex
    index_window[newindex] = pos


@njit(nogil=True)
def simple_siftup_min(heap, pos, endpos, min_heap_indices, index_window):
    newitem = heap[pos]
    newindex = min_heap_indices[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and heap[childpos] > heap[rightpos]:
            childpos = rightpos
        if newitem > heap[childpos]:
            heap[pos], min_heap_indices[pos] = (
                heap[childpos],
                min_heap_indices[childpos],
            )
            index_window[min_heap_indices[pos]] = pos
            pos = childpos
            childpos = 2 * pos + 1
            continue
        break
    heap[pos], min_heap_indices[pos] = newitem, newindex
    index_window[newindex] = pos
    # siftdown_min(heap, startpos, pos, min_heap_indices, index_window)


@njit(nogil=True)
def siftup_min(heap, pos, endpos, min_heap_indices, index_window):
    startpos = pos
    newitem = heap[pos]
    newindex = min_heap_indices[pos]
    childpos = 2 * pos + 1
    while childpos < endpos:
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos], min_heap_indices[pos] = heap[childpos], min_heap_indices[childpos]
        index_window[min_heap_indices[pos]] = pos
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos], min_heap_indices[pos] = newitem, newindex
    index_window[newindex] = pos
    siftdown_min(heap, startpos, pos, min_heap_indices, index_window)


@njit(nogil=True)
def remove_from_min_heap(heap, end, item, min_heap_indices, index_window):
    index = -1
    for i in range(end):
        if heap[i] == item:
            index = i
            break

    if index == -1:
        return end

    # Swap the last element with the one to be removed
    heap[index], min_heap_indices[index] = heap[end - 1], min_heap_indices[end - 1]
    end -= 1
    index_window[
        min_heap_indices[index]
    ] = index  # Update index_window for the swapped element

    if index == 0 or heap[index] > heap[(index - 1) >> 1]:
        simple_siftup_min(heap, index, end, min_heap_indices, index_window)
    else:
        # Sift down if necessary
        siftdown_min(heap, 0, index, min_heap_indices, index_window)

    return end


@njit(nogil=True)
def remove_index_from_min_heap(
    heap, end, index, heap_indices, index_window, tracking_index
):
    # Swap the last element with the one to be removed
    heap[index], heap_indices[index] = heap[end - 1], heap_indices[end - 1]
    end -= 1
    index_window[
        heap_indices[index]
    ] = index  # Update index_window for the swapped element

    if index == 0 or heap[index] > heap[(index - 1) >> 1]:
        simple_siftup_min(heap, index, end, heap_indices, index_window)
    else:
        # Sift down if necessary
        siftdown_min(heap, 0, index, heap_indices, index_window)
    index_window[tracking_index] = -1
    return end


@njit(nogil=True)
def swap_index_from_min_heap(
    heap,
    end,
    index,
    heap_indices,
    index_window,
    new_item,
    new_tracking_index,
    tracking_index,
):
    # Swap the last element with the one to be removed
    index_window[tracking_index] = -1
    heap[index], heap_indices[index] = new_item, new_tracking_index
    index_window[
        new_tracking_index
    ] = index  # Update index_window for the swapped element

    if index == 0 or heap[index] > heap[(index - 1) >> 1]:
        simple_siftup_min(heap, index, end, heap_indices, index_window)
    else:
        # Sift down if necessary
        siftdown_min(heap, 0, index, heap_indices, index_window)
    return end


@njit(nogil=True)
def rolling_median(numbers, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if window_size == 1:
        return numbers
    elif window_size == 3:
        current_median = numbers[0]
        next_median = numbers[0]
        last_median = numbers[0]
        if (
            numbers[-3] <= numbers[-2] <= numbers[-1]
            or numbers[-1] <= numbers[-2] <= numbers[-3]
        ):
            last_median = numbers[-2]
        elif (
            numbers[-2] <= numbers[-3] <= numbers[-1]
            or numbers[-1] <= numbers[-3] <= numbers[-2]
        ):
            last_median = numbers[-3]
        else:
            last_median = numbers[-1]
        for i in range(len(numbers) - 2):
            if (
                numbers[i] <= numbers[i + 1] <= numbers[i + 2]
                or numbers[i + 2] <= numbers[i + 1] <= numbers[i]
            ):
                next_median = numbers[i + 1]
            elif (
                numbers[i + 1] <= numbers[i] <= numbers[i + 2]
                or numbers[i + 2] <= numbers[i] <= numbers[i + 1]
            ):
                next_median = numbers[i]
            else:
                next_median = numbers[i + 2]
            numbers[i] = current_median
            current_median = next_median
        numbers[-2] = last_median
        return numbers
    else:
        array_size = window_size // 2 + 1
        min_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values > median first value is always smallest
        min_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        max_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values < median first value is always smallest but stores as negative numebrs
        max_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        min_index_window = np.ones(window_size, np.int16) * -1
        max_index_window = np.ones(window_size, np.int16) * -1
        edge = window_size // 2
        # Start heap counters at 1 - accounting for first two valus being added
        min_heap_end = 1
        max_heap_end = 1
        # Initialize first two values manually
        if numbers[0] <= numbers[1]:
            # Smaller number should start in max heap
            max_heap[0] = -numbers[0]  # Negate the number
            max_heap_indices[0] = 0
            max_index_window[0] = 0
            min_heap[0] = numbers[1]
            min_heap_indices[0] = 1
            min_index_window[1] = 0
        else:
            max_heap[0] = -numbers[1]  # Negate the number
            max_heap_indices[0] = 1
            max_index_window[1] = 0
            min_heap[0] = numbers[0]
            min_heap_indices[0] = 0
            min_index_window[0] = 0

        # for i in range(2, len(numbers)):
        # if i >= window_size:
        #     # Remove oldest number from heaps
        #     # oldest_number = numbers[i - window_size]
        #     oldest_number = min_heap[min_index_window[tracking_index]] if min_index_window[tracking_index] != -1 else -max_heap[max_index_window[tracking_index]]
        #     if oldest_number <= -max_heap[0]:
        #         max_heap_end = remove_index_from_min_heap(
        #             max_heap, max_heap_end, max_index_window[tracking_index], max_heap_indices, max_index_window
        #         )  # Negate the number
        #         max_index_window[tracking_index] = -1
        #         # The heaps now have equal size, which is good
        #     else:
        #         min_heap_end = remove_index_from_min_heap(
        #             min_heap, min_heap_end, min_index_window[tracking_index], min_heap_indices, min_index_window
        #         )
        #         min_index_window[tracking_index] = -1

        # # Add new number into heaps
        # newest_number = numbers[i]
        # if (
        #     newest_number <= min_heap[0]
        # ):  # If newest number is less the smallest value in the min_heap add to the max heap
        #     heappush_min(
        #         max_heap, -newest_number, max_heap_end, max_heap_indices, tracking_index, max_index_window
        #     )  # Store the negative of the number
        #     max_heap_end += 1
        #     # The max heap is now 1 element larger than the min_heap - which is what we expect
        # else:  # Otherwise put on the min_heap
        #     heappush_min(min_heap, newest_number, min_heap_end, min_heap_indices, tracking_index, min_index_window)
        #     min_heap_end += 1
        for i in range(2, len(numbers)):
            # print(f"Max heap: {max_heap}\nMin heap: {min_heap}")
            # print(f"Max heap: {max_heap_indices}\nMin heap: {min_heap_indices}")
            # print(f"Max heap: {max_index_window}\nMin heap: {min_index_window}")

            tracking_index = i % window_size
            if i >= window_size:
                oldest_number = (
                    min_heap[min_index_window[tracking_index]]
                    if min_index_window[tracking_index] != -1
                    else -max_heap[max_index_window[tracking_index]]
                )
                newest_number = numbers[i]

                # print(f"{i} - {i - edge} - Max heap end: {max_heap_end}-{is_min_heap(max_heap, max_heap_end)}, Min heap end: {min_heap_end}-{is_min_heap(min_heap,min_heap_end)}, Next: {numbers[i]}, Oldest: {oldest_number}")

                if oldest_number <= -max_heap[0]:
                    if newest_number <= -max_heap[0]:
                        max_heap_end = swap_index_from_min_heap(
                            max_heap,
                            max_heap_end,
                            max_index_window[tracking_index],
                            max_heap_indices,
                            max_index_window,
                            -newest_number,
                            tracking_index,
                            tracking_index,
                        )
                    else:
                        value, value_index = heappushpop_min(
                            min_heap,
                            newest_number,
                            min_heap_end,
                            min_heap_indices,
                            tracking_index,
                            min_index_window,
                        )
                        max_heap_end = swap_index_from_min_heap(
                            max_heap,
                            max_heap_end,
                            max_index_window[tracking_index],
                            max_heap_indices,
                            max_index_window,
                            -value,
                            value_index,
                            tracking_index,
                        )
                        # min_heap_end = heappush_min(min_heap, newest_number, min_heap_end, min_heap_indices, tracking_index, min_index_window)
                        # value, value_index, min_heap_end = heappop_min(min_heap, min_heap_end, min_heap_indices, min_index_window)
                        # max_heap_end = remove_index_from_min_heap(max_heap, max_heap_end, max_index_window[tracking_index], max_heap_indices, max_index_window, tracking_index)
                        # max_heap_end = heappush_min(max_heap, -value, max_heap_end, max_heap_indices, value_index, max_index_window)  # Negate the popped number
                else:
                    if newest_number <= -max_heap[0]:
                        value, value_index = heappushpop_min(
                            max_heap,
                            -newest_number,
                            max_heap_end,
                            max_heap_indices,
                            tracking_index,
                            max_index_window,
                        )
                        min_heap_end = swap_index_from_min_heap(
                            min_heap,
                            min_heap_end,
                            min_index_window[tracking_index],
                            min_heap_indices,
                            min_index_window,
                            -value,
                            value_index,
                            tracking_index,
                        )
                        # max_heap_end = heappush_min(max_heap, -newest_number, max_heap_end, max_heap_indices, tracking_index, max_index_window)
                        # value, value_index, max_heap_end = heappop_min(max_heap, max_heap_end, max_heap_indices, max_index_window)
                        # min_heap_end = remove_index_from_min_heap(min_heap, min_heap_end, min_index_window[tracking_index], min_heap_indices, min_index_window, tracking_index)
                        # min_heap_end = heappush_min(min_heap, -value, min_heap_end, min_heap_indices, value_index, min_index_window)  # Negate the popped number
                    else:
                        min_heap_end = swap_index_from_min_heap(
                            min_heap,
                            min_heap_end,
                            min_index_window[tracking_index],
                            min_heap_indices,
                            min_index_window,
                            newest_number,
                            tracking_index,
                            tracking_index,
                        )
            else:
                # Add new number into heaps
                newest_number = numbers[i]
                if (
                    newest_number <= -max_heap[0]
                ):  # If newest number is less the smallest value in the min_heap add to the max heap
                    max_heap_end = heappush_min(
                        max_heap,
                        -newest_number,
                        max_heap_end,
                        max_heap_indices,
                        tracking_index,
                        max_index_window,
                    )  # Store the negative of the number
                else:  # Otherwise put on the min_heap
                    min_heap_end = heappush_min(
                        min_heap,
                        newest_number,
                        min_heap_end,
                        min_heap_indices,
                        tracking_index,
                        min_index_window,
                    )

                # Rebalance Heaps
                while max_heap_end > min_heap_end + 1:
                    value, value_index, max_heap_end = heappop_min(
                        max_heap, max_heap_end, max_heap_indices, max_index_window
                    )
                    min_heap_end = heappush_min(
                        min_heap,
                        -value,
                        min_heap_end,
                        min_heap_indices,
                        value_index,
                        min_index_window,
                    )  # Negate the popped number

                while min_heap_end > max_heap_end:
                    value, value_index, min_heap_end = heappop_min(
                        min_heap, min_heap_end, min_heap_indices, min_index_window
                    )
                    max_heap_end = heappush_min(
                        max_heap,
                        -value,
                        max_heap_end,
                        max_heap_indices,
                        value_index,
                        max_index_window,
                    )  # Negate the popped number

            if i >= window_size - 1:
                numbers[i - edge] = -max_heap[0]  # Negate the result
        return numbers


@njit(nogil=True)
def rolling_median_naive(numbers, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if window_size == 1:
        return numbers
    elif window_size == 3:
        current_median = numbers[0]
        next_median = numbers[0]
        last_median = numbers[0]
        if (
            numbers[-3] <= numbers[-2] <= numbers[-1]
            or numbers[-1] <= numbers[-2] <= numbers[-3]
        ):
            last_median = numbers[-2]
        elif (
            numbers[-2] <= numbers[-3] <= numbers[-1]
            or numbers[-1] <= numbers[-3] <= numbers[-2]
        ):
            last_median = numbers[-3]
        else:
            last_median = numbers[-1]
        for i in range(len(numbers) - 2):
            if (
                numbers[i] <= numbers[i + 1] <= numbers[i + 2]
                or numbers[i + 2] <= numbers[i + 1] <= numbers[i]
            ):
                next_median = numbers[i + 1]
            elif (
                numbers[i + 1] <= numbers[i] <= numbers[i + 2]
                or numbers[i + 2] <= numbers[i] <= numbers[i + 1]
            ):
                next_median = numbers[i]
            else:
                next_median = numbers[i + 2]
            numbers[i] = current_median
            current_median = next_median
        numbers[-2] = last_median
        return numbers
    else:
        array_size = window_size // 2 + 1
        min_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values > median first value is always smallest
        min_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        max_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values < median first value is always smallest but stores as negative numebrs
        max_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        min_index_window = np.ones(window_size, np.int16) * -1
        max_index_window = np.ones(window_size, np.int16) * -1
        edge = window_size // 2
        # Start heap counters at 1 - accounting for first two valus being added
        min_heap_end = 1
        max_heap_end = 1
        # Initialize first two values manually
        if numbers[0] <= numbers[1]:
            # Smaller number should start in max heap
            max_heap[0] = -numbers[0]  # Negate the number
            max_heap_indices[0] = 0
            max_index_window[0] = 0
            min_heap[0] = numbers[1]
            min_heap_indices[0] = 1
            min_index_window[1] = 0
        else:
            max_heap[0] = -numbers[1]  # Negate the number
            max_heap_indices[0] = 1
            max_index_window[1] = 0
            min_heap[0] = numbers[0]
            min_heap_indices[0] = 0
            min_index_window[0] = 0

        for i in range(2, len(numbers)):
            tracking_index = i % window_size
            newest_number = numbers[i]
            if i >= window_size:
                # Remove oldest number from heaps
                # oldest_number = numbers[i - window_size]
                oldest_number = (
                    min_heap[min_index_window[tracking_index]]
                    if min_index_window[tracking_index] != -1
                    else -max_heap[max_index_window[tracking_index]]
                )
                if oldest_number <= -max_heap[0]:
                    max_heap_end = remove_index_from_min_heap(
                        max_heap,
                        max_heap_end,
                        max_index_window[tracking_index],
                        max_heap_indices,
                        max_index_window,
                        tracking_index,
                    )  # Negate the number
                    # The heaps now have equal size, which is good
                else:
                    min_heap_end = remove_index_from_min_heap(
                        min_heap,
                        min_heap_end,
                        min_index_window[tracking_index],
                        min_heap_indices,
                        min_index_window,
                        tracking_index,
                    )

            # Add new number into heaps
            if (
                newest_number <= -max_heap[0]
            ):  # If newest number is less the smallest value in the min_heap add to the max heap
                max_heap_end = heappush_min(
                    max_heap,
                    -newest_number,
                    max_heap_end,
                    max_heap_indices,
                    tracking_index,
                    max_index_window,
                )  # Store the negative of the number
            else:  # Otherwise put on the min_heap
                min_heap_end = heappush_min(
                    min_heap,
                    newest_number,
                    min_heap_end,
                    min_heap_indices,
                    tracking_index,
                    min_index_window,
                )

            # Rebalance Heaps
            while max_heap_end > min_heap_end + 1:
                value, value_index, max_heap_end = heappop_min(
                    max_heap, max_heap_end, max_heap_indices, max_index_window
                )
                min_heap_end = heappush_min(
                    min_heap,
                    -value,
                    min_heap_end,
                    min_heap_indices,
                    value_index,
                    min_index_window,
                )  # Negate the popped number

            while min_heap_end > max_heap_end + 1:
                value, value_index, min_heap_end = heappop_min(
                    min_heap, min_heap_end, min_heap_indices, min_index_window
                )
                max_heap_end = heappush_min(
                    max_heap,
                    -value,
                    max_heap_end,
                    max_heap_indices,
                    value_index,
                    max_index_window,
                )  # Negate the popped number

            if i >= window_size - 1:
                if max_heap_end > min_heap_end:
                    numbers[i - edge] = -max_heap[0]  # Negate the result
                else:
                    numbers[i - edge] = min_heap[0]  # Negate the result
        return numbers


@njit(nogil=True)
def rolling_median_column(numbers, window_size, highpass=False):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    if window_size == 1:
        return numbers
    elif window_size == 3:
        current_median = numbers[0, 0]
        next_median = numbers[0, 0]
        last_median = numbers[0, 0]
        if (
            numbers[0, -3] <= numbers[0, -2] <= numbers[0, -1]
            or numbers[0, -1] <= numbers[0, -2] <= numbers[0, -3]
        ):
            last_median = numbers[0, -2]
        elif (
            numbers[0, -2] <= numbers[0, -3] <= numbers[0, -1]
            or numbers[0, -1] <= numbers[0, -3] <= numbers[0, -2]
        ):
            last_median = numbers[0, -3]
        else:
            last_median = numbers[0, -1]
        for i in range(len(numbers) - 2):
            if (
                numbers[0, i] <= numbers[0, i + 1] <= numbers[0, i + 2]
                or numbers[0, i + 2] <= numbers[0, i + 1] <= numbers[0, i]
            ):
                next_median = numbers[0, i + 1]
            elif (
                numbers[0, i + 1] <= numbers[0, i] <= numbers[0, i + 2]
                or numbers[0, i + 2] <= numbers[0, i] <= numbers[0, i + 1]
            ):
                next_median = numbers[0, i]
            else:
                next_median = numbers[0, i + 2]
            numbers[0, i] = current_median
            current_median = next_median
        numbers[0, -2] = last_median
        return numbers
    else:
        array_size = window_size // 2 + 1
        min_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values > median first value is always smallest
        min_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        max_heap = np.empty(
            array_size, numbers.dtype
        )  # Heap to store values < median first value is always smallest but stores as negative numebrs
        max_heap_indices = np.empty(
            array_size, np.int16
        )  # Array matchin heap that stores index in the index_window array for updating
        min_index_window = np.ones(window_size, np.int16) * -1
        max_index_window = np.ones(window_size, np.int16) * -1
        edge = window_size // 2
        # Start heap counters at 1 - accounting for first two valus being added
        min_heap_end = 1
        max_heap_end = 1
        # Initialize first two values manually
        if numbers[0, 0] <= numbers[0, 1]:
            # Smaller number should start in max heap
            max_heap[0] = -numbers[0, 0]  # Negate the number
            max_heap_indices[0] = 0
            max_index_window[0] = 0
            min_heap[0] = numbers[0, 1]
            min_heap_indices[0] = 1
            min_index_window[1] = 0
        else:
            max_heap[0] = -numbers[0, 1]  # Negate the number
            max_heap_indices[0] = 1
            max_index_window[1] = 0
            min_heap[0] = numbers[0, 0]
            min_heap_indices[0] = 0
            min_index_window[0] = 0

        for i in range(2, numbers.shape[1]):
            # print(f"Max heap: {max_heap}\nMin heap: {min_heap}")
            # print(f"Max heap: {max_heap_indices}\nMin heap: {min_heap_indices}")
            # print(f"Max heap: {max_index_window}\nMin heap: {min_index_window}")

            tracking_index = i % window_size
            newest_number = numbers[0, i]
            if i >= window_size:
                oldest_number = (
                    min_heap[min_index_window[tracking_index]]
                    if min_index_window[tracking_index] != -1
                    else -max_heap[max_index_window[tracking_index]]
                )

                # print(f"{i} - {i - edge} - Max heap end: {max_heap_end}-{is_min_heap(max_heap, max_heap_end)}, Min heap end: {min_heap_end}-{is_min_heap(min_heap,min_heap_end)}, Next: {numbers[i]}, Oldest: {oldest_number}")

                if oldest_number <= -max_heap[0]:
                    if newest_number <= -max_heap[0]:
                        max_heap_end = swap_index_from_min_heap(
                            max_heap,
                            max_heap_end,
                            max_index_window[tracking_index],
                            max_heap_indices,
                            max_index_window,
                            -newest_number,
                            tracking_index,
                            tracking_index,
                        )
                    else:
                        value, value_index = heappushpop_min(
                            min_heap,
                            newest_number,
                            min_heap_end,
                            min_heap_indices,
                            tracking_index,
                            min_index_window,
                        )
                        max_heap_end = swap_index_from_min_heap(
                            max_heap,
                            max_heap_end,
                            max_index_window[tracking_index],
                            max_heap_indices,
                            max_index_window,
                            -value,
                            value_index,
                            tracking_index,
                        )
                        # min_heap_end = heappush_min(min_heap, newest_number, min_heap_end, min_heap_indices, tracking_index, min_index_window)
                        # value, value_index, min_heap_end = heappop_min(min_heap, min_heap_end, min_heap_indices, min_index_window)
                        # max_heap_end = remove_index_from_min_heap(max_heap, max_heap_end, max_index_window[tracking_index], max_heap_indices, max_index_window, tracking_index)
                        # max_heap_end = heappush_min(max_heap, -value, max_heap_end, max_heap_indices, value_index, max_index_window)  # Negate the popped number
                else:
                    if newest_number <= -max_heap[0]:
                        value, value_index = heappushpop_min(
                            max_heap,
                            -newest_number,
                            max_heap_end,
                            max_heap_indices,
                            tracking_index,
                            max_index_window,
                        )
                        min_heap_end = swap_index_from_min_heap(
                            min_heap,
                            min_heap_end,
                            min_index_window[tracking_index],
                            min_heap_indices,
                            min_index_window,
                            -value,
                            value_index,
                            tracking_index,
                        )
                        # max_heap_end = heappush_min(max_heap, -newest_number, max_heap_end, max_heap_indices, tracking_index, max_index_window)
                        # value, value_index, max_heap_end = heappop_min(max_heap, max_heap_end, max_heap_indices, max_index_window)
                        # min_heap_end = remove_index_from_min_heap(min_heap, min_heap_end, min_index_window[tracking_index], min_heap_indices, min_index_window, tracking_index)
                        # min_heap_end = heappush_min(min_heap, -value, min_heap_end, min_heap_indices, value_index, min_index_window)  # Negate the popped number
                    else:
                        min_heap_end = swap_index_from_min_heap(
                            min_heap,
                            min_heap_end,
                            min_index_window[tracking_index],
                            min_heap_indices,
                            min_index_window,
                            newest_number,
                            tracking_index,
                            tracking_index,
                        )
            else:
                # Add new number into heaps
                if (
                    newest_number <= -max_heap[0]
                ):  # If newest number is less the smallest value in the min_heap add to the max heap
                    max_heap_end = heappush_min(
                        max_heap,
                        -newest_number,
                        max_heap_end,
                        max_heap_indices,
                        tracking_index,
                        max_index_window,
                    )  # Store the negative of the number
                else:  # Otherwise put on the min_heap
                    min_heap_end = heappush_min(
                        min_heap,
                        newest_number,
                        min_heap_end,
                        min_heap_indices,
                        tracking_index,
                        min_index_window,
                    )

                # Rebalance Heaps
                while max_heap_end > min_heap_end + 1:
                    value, value_index, max_heap_end = heappop_min(
                        max_heap, max_heap_end, max_heap_indices, max_index_window
                    )
                    min_heap_end = heappush_min(
                        min_heap,
                        -value,
                        min_heap_end,
                        min_heap_indices,
                        value_index,
                        min_index_window,
                    )  # Negate the popped number

                while min_heap_end > max_heap_end:
                    value, value_index, min_heap_end = heappop_min(
                        min_heap, min_heap_end, min_heap_indices, min_index_window
                    )
                    max_heap_end = heappush_min(
                        max_heap,
                        -value,
                        max_heap_end,
                        max_heap_indices,
                        value_index,
                        max_index_window,
                    )  # Negate the popped number

            if i >= window_size - 1:
                if highpass:
                    numbers[0, i - edge] += max_heap[
                        0
                    ]  # Subtract (add negative of median stored in max_heap) to numbers
                else:
                    numbers[0, i - edge] = -max_heap[0]  # Negate the result
        return numbers


# @njit(nogil=True)
# def median_of_stream(numbers):
#     min_heap = np.zeros_like(numbers)  # Min-heap for the larger half of the numbers
#     max_heap = np.zeros_like(numbers)  # Max-heap (simulated using negatives) for the smaller half of the numbers

#     medians = np.zeros_like(numbers, dtype=np.float32)

#     min_heap_end = 0
#     max_heap_end = 0

#     for i, number in enumerate(numbers):
#         if max_heap_end == 0 or number < -max_heap[0]:
#             heappush_min(max_heap, number, max_heap_end, is_min_heap=False)
#             max_heap_end += 1
#         else:
#             heappush_min(min_heap, number, min_heap_end, is_min_heap=True)
#             min_heap_end += 1

#         # Rebalance heaps
#         if min_heap_end > max_heap_end + 1:
#             heappush_min(max_heap, heappop_min(min_heap, min_heap_end, is_min_heap=True), max_heap_end, is_min_heap=False)
#             min_heap_end -= 1
#             max_heap_end += 1
#         elif max_heap_end > min_heap_end + 1:
#             heappush_min(min_heap, heappop_min(max_heap, max_heap_end, is_min_heap=False), min_heap_end, is_min_heap=True)
#             max_heap_end -= 1
#             min_heap_end += 1

#         # Calculate median
#         if min_heap_end == max_heap_end:
#             median = (get_min_heap_top(min_heap, is_min_heap=True) + get_min_heap_top(max_heap, is_min_heap=False)) / 2.0
#         elif min_heap_end > max_heap_end:
#             median = get_min_heap_top(min_heap, is_min_heap=True)
#         else:
#             median = get_min_heap_top(max_heap, is_min_heap=False)

#         medians[i] = median

#     return medians

# @njit(nogil=True)
# def calculate_median(numbers):
#     min_heap = np.zeros_like(numbers)  # Min-heap for the larger half of the numbers
#     max_heap = np.zeros_like(numbers)  # Max-heap (simulated using negatives) for the smaller half of the numbers

#     min_heap_end = 0
#     max_heap_end = 0

#     for number in numbers:
#         if max_heap_end == 0 or number < -max_heap[0]:
#             heappush_min(max_heap, number, max_heap_end, is_min_heap=False)
#             max_heap_end += 1
#         else:
#             heappush_min(min_heap, number, min_heap_end, is_min_heap=True)
#             min_heap_end += 1

#         # Rebalance heaps
#         if min_heap_end > max_heap_end + 1:
#             heappush_min(max_heap, heappop_min(min_heap, min_heap_end, is_min_heap=True), max_heap_end, is_min_heap=False)
#             min_heap_end -= 1
#             max_heap_end += 1
#         elif max_heap_end > min_heap_end + 1:
#             heappush_min(min_heap, heappop_min(max_heap, max_heap_end, is_min_heap=False), min_heap_end, is_min_heap=True)
#             max_heap_end -= 1
#             min_heap_end += 1

#     # Calculate median
#     if min_heap_end == max_heap_end:
#         median = (get_min_heap_top(min_heap, is_min_heap=True) + get_min_heap_top(max_heap, is_min_heap=False)) / 2.0
#     elif min_heap_end > max_heap_end:
#         median = get_min_heap_top(min_heap, is_min_heap=True)
#     else:
#         median = get_min_heap_top(max_heap, is_min_heap=False)

#     return median

# @njit(nogil=True)
# def sliding_median_two_heaps(arr, window_size):
#     n = len(arr)
#     medians = np.zeros(n)

#     max_heap, min_heap = [], []
#     invalid_elements = {}

#     # Initialize the first window
#     for i in range(window_size):
#         heappush_min(max_heap, -arr[i])

#     # Balance the heaps
#     for _ in range((window_size - 1) // 2):
#         heappush_min(min_heap, -heappush_min(max_heap))

#     for i in range(window_size, n):
#         new_element = arr[i]
#         exiting_element = arr[i - window_size]

#         # Add the new element to the correct heap
#         if new_element <= -max_heap[0]:
#             heappush_min(max_heap, -new_element)
#         else:
#             heappush_min(min_heap, new_element)

#         # Mark the exiting element as invalid
#         invalid_elements[exiting_element] = invalid_elements.get(exiting_element, 0) + 1

#         # Remove invalid elements from the top of the heaps
#         while max_heap and invalid_elements.get(-max_heap[0], 0) > 0:
#             invalid_elements[-max_heap[0]] -= 1
#             heappop_min(max_heap)
#         while min_heap and invalid_elements.get(min_heap[0], 0) > 0:
#             invalid_elements[min_heap[0]] -= 1
#             heappop_min(min_heap)

#         # Rebalance the heaps
#         if len(max_heap) > len(min_heap) + 1:
#             heappop_min(min_heap, -heappop_min(max_heap))
#         elif len(min_heap) > len(max_heap):
#             heappop_min(max_heap, -heappop_min(min_heap))

#         # Calculate the median
#         if window_size % 2 == 0:
#             median = (-max_heap[0] + min_heap[0]) / 2.0
#         else:
#             median = -max_heap[0]

#         medians[i - window_size + 1] = median

#     return medians
