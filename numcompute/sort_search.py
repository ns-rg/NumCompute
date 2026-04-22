import numpy as np


def topk(values, k, largest=True, return_indices=True):
    """
    Return top-k elements from array.

    Parameters
    ----------
    values : np.ndarray
    k : int
    largest : bool
        If True → largest k, else smallest k
    return_indices : bool

    Returns
    -------
    values or (values, indices)
    """

    values = np.asarray(values)

    if k <= 0 or k > values.size:
        raise ValueError("k must be between 1 and len(values)")

    if largest:
        indices = np.argpartition(-values, k - 1)[:k]
    else:
        indices = np.argpartition(values, k - 1)[:k]

    top_values = values[indices]

    # sort result
    sorted_idx = np.argsort(-top_values) if largest else np.argsort(top_values)

    top_values = top_values[sorted_idx]
    indices = indices[sorted_idx]

    return (top_values, indices) if return_indices else top_values


def binary_search(arr, x):
    """
    Binary search on sorted array.

    Parameters
    ----------
    arr : np.ndarray (sorted)
    x : value to search

    Returns
    -------
    (index, found)
    """

    arr = np.asarray(arr)

    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == x:
            return mid, True
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1

    return left, False  # insertion index
