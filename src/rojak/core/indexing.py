from typing import Sequence


def make_value_based_slice(coordinate: Sequence, min_value: float | None, max_value: float | None) -> slice:
    """
    Make a slice from coordinate while accounting for whether values are increasing or decreasing.
    Args:
        coordinate: Sequence of values
        min_value: Minimum value in slice
        max_value: Maximum value in slice

    Returns:
        slice: Slice to be used for indexing

    >>> import numpy as np
    >>> make_value_based_slice(np.arange(10), 0, 5)
    slice(0, 5, None)
    >>> make_value_based_slice(np.arange(10, 0, -1), 0, 5)
    slice(5, 0, None)
    """
    is_increasing: bool = coordinate[0] < coordinate[-1]
    return slice(min_value, max_value) if is_increasing else slice(max_value, min_value)
