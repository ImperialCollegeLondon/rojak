#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def get_regular_grid_spacing[T: np.number | np.inexact | np.datetime64](array: "NDArray[T]") -> None | T:
    """
    Determines if array has a regular grid spacing
    Args:
        array: Array to be checked

    Returns:
        Grid spacing if on a regular grid. If not, it returns None

    >>> get_regular_grid_spacing(np.linspace(5, 10, 11))
    0.5
    >>> get_regular_grid_spacing(np.arange(np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), \
    dtype="datetime64[h]"))
    datetime.timedelta(seconds=3600)
    >>> get_regular_grid_spacing(np.asarray([9, 3, 7]))
    """
    # No need to check for ndim == 0 as np.asarray([]).ndim == 1
    if array.ndim > 1:
        raise NotImplementedError("Test to determine regular grid spacing only supported for 1D arrays")

    difference = np.diff(array)

    # See https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html for character code
    match array.dtype.kind:
        case "f" | "c":  # f => float, c => complex floating-point
            if np.allclose(difference, difference[0]):
                return difference.item(0)
        case "M" | "i" | "u":  # M => datetime, i => signed integer, u => unsigned integer
            # Use exact comparison for these data types
            if np.all(difference == difference[0]):
                return difference.item(0)
        case _:
            raise NotImplementedError(f"Other dtypes ({array.dtype}) are not yet supported")

    return None
