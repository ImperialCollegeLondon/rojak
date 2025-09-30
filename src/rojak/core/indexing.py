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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import dask.dataframe as dd
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


def get_regular_grid_spacing[T: np.number | np.inexact | np.datetime64 | np.timedelta64](
    array: "NDArray[T]",
) -> None | T:
    """
    Determines if array has a regular grid spacing

    Args:
        array: Array to be checked

    Returns:
        Grid spacing if on a regular grid. If not, it returns None

    >>> np.set_printoptions(legacy=False)
    >>> get_regular_grid_spacing(np.linspace(5, 10, 11))
    np.float64(0.5)
    >>> get_regular_grid_spacing(np.arange(np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), \
    dtype="datetime64[h]"))
    np.timedelta64(1,'h')

    None is returned if array does not have a regular grid spacing
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
                return difference[0]
        case "M" | "i" | "u":  # M => datetime, i => signed integer, u => unsigned integer
            # Use exact comparison for these data types
            if np.all(difference == difference[0]):
                return difference[0]
        case _:
            raise NotImplementedError(f"Other dtypes ({array.dtype}) are not yet supported")

    return None


def map_values_to_nearest_coordinate_index[T: np.datetime64 | np.number | np.inexact](
    series: "dd.Series | pd.Series",
    coordinate: "NDArray[T]",
    valid_window: np.timedelta64 | np.number | np.inexact | None = None,
) -> "dd.Series | pd.Series[int]":
    """
    Assuming the coordinate is a regular grid, compute the closest index that the values in the series correspond to.

    Args:
        series: Data which corresponds to a given value in the coordinate array and needs to be mapped to the closest
            index in the coordinate array
        coordinate: Array which is to be indexed into
        valid_window: Symmetric window (e.g. +- 3 hrs => np.timedelta64(3, "h"))

    Returns:
        Series which contains the closest index the data maps to

    >>> import pandas as pd
    >>> time_coordinate = np.arange(np.datetime64("2018-08-01"), np.datetime64("2018-08-03"), np.timedelta64(6, "h"))
    >>> data_series = pd.Series([np.datetime64("2018-08-02T16:06"), np.datetime64("2018-08-01T07:37"), \
    np.datetime64("2018-08-02T09:12"), np.datetime64("2018-08-02T07:27"), np.datetime64("2018-08-02T19:09")])
    >>> map_values_to_nearest_coordinate_index(data_series, time_coordinate, valid_window=np.timedelta64(3, "h"))
        0    7
        1    1
        2    6
        3    5
        4    7
        dtype: int64
    >>> map_values_to_nearest_coordinate_index(data_series, time_coordinate, valid_window=np.timedelta64(2, "h"))
    Traceback (most recent call last):
    NotImplementedError: Function currently only supports regular grids with a symmetric window specified. \
    And the window must correspond to half of the grid spacing

    Not specifying the valid window, forces the minimum and maximum of the data in the Series to be strictly within
    the range of the coordinate

    >>> map_values_to_nearest_coordinate_index(data_series, time_coordinate)
    Traceback (most recent call last):
    ValueError: Values in series must be within the range of the coordinate

    By extending the time coordiante to include the last 6 hours on 2018-08-03 places the 2018-08-02T19:09 within
    the range of the coordinate

    >>> time_coordinate = np.arange(np.datetime64("2018-08-01"), np.datetime64("2018-08-03T06"), np.timedelta64(6, "h"))
    >>> map_values_to_nearest_coordinate_index(data_series, time_coordinate)
        0    7
        1    1
        2    6
        3    5
        4    7
        dtype: int64
    """
    if coordinate.ndim > 1:
        raise ValueError(f"Coordinate must be 1D not {coordinate.ndim}D")

    if valid_window is None:
        if series.min() < coordinate.min() or series.max() > coordinate.max():
            raise ValueError("Values in series must be within the range of the coordinate")
    elif series.min() < coordinate.min() - valid_window or series.max() > coordinate.max() + valid_window:
        raise ValueError("Values in series must be within the window of the coordinate")

    # Regular grid => monotonic function (in fact, it should be stricter)
    spacing = get_regular_grid_spacing(coordinate)
    if spacing is None:
        raise NotImplementedError(
            "Optimisation to map values to index into coordinate is only supported for regular grids"
        )
    if valid_window is not None and 2 * valid_window != spacing:
        raise NotImplementedError(
            "Function currently only supports regular grids with a symmetric window specified."
            " And the window must correspond to half of the grid spacing"
        )

    approximate_index = (series - coordinate[0]) / spacing  # pyright: ignore[reportOperatorIssue]
    # rint - rounds to the closest integer => gives closest index
    return np.rint(approximate_index).astype(int)


def map_index_to_coordinate_value(
    indices: "pd.Series | pd.Index", coordinate: "NDArray", series_name: str | None = None
) -> "pd.Series":
    """
    Retrieve original value based on index

    This function is the inverse of :py:func`map_values_to_nearest_coordinate_index`

    Args:
        indices: Indices of the values to map
        coordinate: Values which the indices correspond to
        series_name: Name of the new pandas series

    Returns: new pandas series with values

    """
    if coordinate.ndim > 1:
        raise ValueError(f"Coordinate must be 1D not {coordinate.ndim}D")
    if indices.min() < 0:
        raise ValueError("Index must be non-negative")
    if indices.max() > coordinate.size:
        raise ValueError("Index must be within the range of the coordinate")

    as_numpy_array = indices.to_numpy(dtype=int)
    return pd.Series(data=coordinate[as_numpy_array], name=series_name)


def map_order[T](on: list[T], by: list[int]) -> list[T]:
    """
    Maps order of ``on`` based on the order specified in ``by``.

    Args:
        on: List which order is to be mapped onto.
        by: List which specifies the new order of ``on``

    >>> descending = [5, 4, 3, 2, 1, 0]
    >>> ascending = list(range(10, 16))
    >>> map_order(ascending, descending)
    [15, 14, 13, 12, 11, 10]
    """
    length: int = len(by)
    by_as_set: set = set(by)

    if len(on) != length:
        raise ValueError("Order mapping must be on lists of the same length")
    if len(by_as_set) != length:
        raise ValueError("Order of items must have unique values")
    # if max(by) != length - 1:
    #     raise ValueError(f"Maximum value of the order to impose must be length - 1 not {max(by)}")
    # if min(by) != 0:
    #     raise ValueError("Minimum value of the order to impose must be 0")
    if set(range(length)) != by_as_set:
        raise ValueError("Order of items must be increasing by 1 from 0 to length")

    in_order: list = [None] * length
    for position, value in zip(by, on, strict=True):
        in_order[position] = value

    return in_order
