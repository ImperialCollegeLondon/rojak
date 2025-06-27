from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
import pytest

from rojak.core.indexing import get_regular_grid_spacing, make_value_based_slice, map_values_to_nearest_coordinate_index

if TYPE_CHECKING:
    import dask.dataframe as dd
    from numpy.typing import NDArray


@pytest.mark.parametrize(
    ("coordinate", "min_value", "max_value", "expected_slice"),
    [
        pytest.param(np.arange(10), 5, None, slice(5, None), id="increasing_from_min"),
        pytest.param(np.arange(10), None, 5, slice(None, 5), id="increasing_up_to_max"),
        pytest.param(np.arange(-10, 0, 1), -5, None, slice(-5, None), id="negative_increasing_from_min"),
        pytest.param(np.arange(-10, 0, 1), None, -5, slice(None, -5), id="negative_increasing_to_max"),
        pytest.param(np.arange(10, 0, -1), 5, None, slice(None, 5), id="decreasing_from_min"),
        pytest.param(np.arange(10, 0, -1), None, 5, slice(5, None), id="decreasing_to_max"),
        pytest.param(np.arange(0, -10, -1), None, -5, slice(-5, None), id="negative_decreasing_to_max"),
        pytest.param(np.arange(0, -10, -1), -5, None, slice(None, -5), id="negative_decreasing_from_min"),
    ],
)
def test_make_value_based_slice(
    coordinate: Sequence, min_value: float | None, max_value: float | None, expected_slice: slice
) -> None:
    computed_slice = make_value_based_slice(coordinate, min_value, max_value)
    assert computed_slice == expected_slice


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.arange(10), 1),
        (np.arange(10, 0, -1), -1),
        (np.arange(0, -10, -1), -1),
        (np.arange(-10, 1, 1), 1),
        (np.arange(0, 10, 1.0 / 3), 1 / 3),
        (np.linspace(10, 5, 11), -0.5),
        (
            np.arange(np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), dtype="datetime64[h]"),
            np.timedelta64(1, "h"),
        ),
        (
            np.arange(
                np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), np.timedelta64(6, "h"), dtype="datetime64[h]"
            ),
            np.timedelta64(6, "h"),
        ),
        (
            np.arange(
                np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), np.timedelta64(30, "m"), dtype="datetime64[s]"
            ),
            np.timedelta64(30, "m"),
        ),
        (np.asarray([3, 5.6, 6]), None),
    ],
)
def test_get_regular_grid_spacing(array: "NDArray", expected: float | None) -> None:
    spacing = get_regular_grid_spacing(array)
    assert spacing == expected


@pytest.mark.parametrize("array", [np.eye(3, 3), np.asarray([True, False])])
def test_get_regular_grid_spacing_not_implemented(array: "NDArray") -> None:
    with pytest.raises(NotImplementedError):
        get_regular_grid_spacing(array)


@pytest.mark.parametrize(
    ("series", "array", "window", "expected"),
    [
        pytest.param(
            None, np.eye(3, 3), None, pytest.raises(ValueError, match="Coordinate must be 1D not 2D"), id="not_1d"
        ),
        pytest.param(
            pd.Series(np.arange(10)),
            np.arange(8),
            1,
            pytest.raises(ValueError, match="Values in series must be within the window of the coordinate"),
            id="outside_window_upper",
        ),
        pytest.param(
            pd.Series(np.arange(11)),
            np.arange(10),
            None,
            pytest.raises(ValueError, match="Values in series must be within the range of the coordinate"),
            id="outside_strict_window",
        ),
        pytest.param(
            pd.Series(np.arange(10)),
            np.arange(8) + 2,
            1,
            pytest.raises(ValueError, match="Values in series must be within the window of the coordinate"),
            id="outside_window_lower",
        ),
        pytest.param(
            pd.Series(np.arange(10)),
            np.arange(5) + 3,
            1,
            pytest.raises(ValueError, match="Values in series must be within the window of the coordinate"),
            id="outside_window_lower_and_upper",
        ),
        pytest.param(
            pd.Series(np.arange(5)),
            np.asarray([0, 4, 4.5, 4.75]),
            None,
            pytest.raises(
                NotImplementedError,
                match="Optimisation to map values to index into coordinate is only supported for regular grids",
            ),
            id="irregular_grid",
        ),
        pytest.param(
            pd.Series(np.arange(5)),
            np.asarray([5, 8, 4, -1]),
            None,
            pytest.raises(
                NotImplementedError,
                match="Optimisation to map values to index into coordinate is only supported for regular grids",
            ),
            id="irregular_and_not_monotonic",
        ),
        pytest.param(
            pd.Series(np.arange(10)),
            np.arange(10),
            1,
            pytest.raises(
                NotImplementedError,
                match="Function currently only supports regular grids with a symmetric window specified",
            ),
            id="window_not_half",
        ),
        pytest.param(
            pd.Series(np.arange(np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), np.timedelta64(2, "h"))),
            np.arange(np.datetime64("1970-01-01"), np.datetime64("1970-01-02"), np.timedelta64(2, "h")),
            np.timedelta64(2, "h"),
            pytest.raises(
                NotImplementedError,
                match="Function currently only supports regular grids with a symmetric window specified",
            ),
            id="window_not_half_time",
        ),
    ],
)
def test_map_value_to_coordinate_index_fails(series: "dd.Series", array: "NDArray", window, expected) -> None:
    with expected:
        map_values_to_nearest_coordinate_index(series, array, valid_window=window)
