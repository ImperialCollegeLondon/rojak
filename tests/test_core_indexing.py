from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rojak.core.indexing import (
    get_regular_grid_spacing,
    make_value_based_slice,
    map_order,
    map_values_to_nearest_coordinate_index,
    shift_and_combine,
)

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
    coordinate: Sequence,
    min_value: float | None,
    max_value: float | None,
    expected_slice: slice,
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
                np.datetime64("1970-01-01"),
                np.datetime64("1970-01-02"),
                np.timedelta64(6, "h"),
                dtype="datetime64[h]",
            ),
            np.timedelta64(6, "h"),
        ),
        (
            np.arange(
                np.datetime64("1970-01-01"),
                np.datetime64("1970-01-02"),
                np.timedelta64(30, "m"),
                dtype="datetime64[s]",
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
            None,
            np.eye(3, 3),
            None,
            pytest.raises(ValueError, match="Coordinate must be 1D not 2D"),
            id="not_1d",
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


@pytest.mark.parametrize(
    ("on", "by", "matches"),
    [
        (list(range(3)), list(range(5)), "Order mapping must be on lists of the same length"),
        (list(range(5)), list(range(3)), "Order mapping must be on lists of the same length"),
        (list(range(3)), [None] * 3, "Order of items must have unique values"),
        (list(range(3)), [50] * 3, "Order of items must have unique values"),
        (list(range(3)), list(range(1, 4)), "Order of items must be increasing by 1 from 0 to length"),
    ],
)
def test_map_order_failure(on: list, by: list[int], matches: str) -> None:
    with pytest.raises(ValueError, match=matches):
        map_order(on, by)


@pytest.mark.parametrize(
    ("on", "by", "matches"),
    [
        ([np.ones(3), np.ones(3) * 5, np.ones(9) * 15], [2, 0, 1], [np.ones(3) * 5, np.ones(9) * 15, np.ones(3)]),
        ([da.ones(3), da.ones(3) * 5, da.ones(9) * 15], [2, 0, 1], [da.ones(3) * 5, da.ones(9) * 15, da.ones(3)]),
    ],
)
def test_map_order(on: list, by: list[int], matches: list) -> None:
    reordered = map_order(on, by)
    for computed, expected in zip(reordered, matches, strict=True):
        np.testing.assert_array_equal(computed, expected)


def add_combine[T: (xr.Dataset, xr.DataArray)](start: T, end: T):
    return start + end


def mean_combine[T: (xr.Dataset, xr.DataArray)](start: T, end: T):
    return (start + end) / 2


def min_combine[T: (xr.Dataset, xr.DataArray)](start: T, end: T):
    return xr.where(start.fillna(np.inf) < end.fillna(np.inf), start, end)


def max_combine[T: (xr.Dataset, xr.DataArray)](start: T, end: T):
    return xr.where(start.fillna(-np.inf) > end.fillna(-np.inf), start, end)


class TestShiftAndCombine:
    """Some tests initially generate by copilot using Claude Haiku 4.5
    All type hinting has been manually added and code has been modified to maintain code quality and to get tests to
    work.
    """

    @pytest.fixture
    def simple_dataarray(self):
        data = np.ones(10, dtype=bool)
        return xr.DataArray(
            data,
            dims=["pressure_level"],
            coords={"pressure_level": np.arange(10)},
            name="temperature",
        )

    @pytest.fixture
    def pressure_level_dataarray(self):
        data = np.arange(10, dtype=float)
        return xr.DataArray(
            data,
            dims=["pressure_level"],
            coords={"pressure_level": np.arange(10)},
            name="temperature",
        )

    @pytest.fixture
    def simple_dataset(self):
        data_var = np.ones(10, dtype=bool)
        return xr.Dataset(
            {"temperature": (["pressure_level"], data_var)},
            coords={"pressure_level": np.arange(10)},
        )

    @pytest.fixture
    def multi_var_dataset(self):
        return xr.Dataset(
            {
                "temperature": (["pressure_level"], np.arange(10, dtype=float)),
                "pressure": (["pressure_level"], np.arange(100, 110, dtype=float)),
                "humidity": (["pressure_level"], np.arange(50, 60, dtype=float)),
            },
            coords={"pressure_level": np.arange(10)},
        )

    @pytest.fixture
    def multi_dim_dataarray(self):
        data = np.random.default_rng().random((5, 10, 3))
        return xr.DataArray(
            data,
            dims=["lat", "pressure_level", "lon"],
            coords={
                "lat": np.arange(5),
                "pressure_level": np.arange(10),
                "lon": np.arange(3),
            },
            name="temperature",
        )

    def test_dataarray_returns_dataarray(self, simple_dataarray: xr.DataArray):
        result = shift_and_combine(simple_dataarray, shift_fill=0)
        assert isinstance(result, xr.DataArray)

    def test_dataset_returns_dataset(self, simple_dataset: xr.Dataset):
        result = shift_and_combine(simple_dataset, shift_fill=0)
        assert isinstance(result, xr.Dataset)

    def test_dataarray_preserves_name(self, simple_dataarray: xr.DataArray):
        result = shift_and_combine(simple_dataarray, shift_fill=0)
        assert result.name == simple_dataarray.name

    def test_dataarray_dimension_preserved(self, simple_dataarray: xr.DataArray):
        result = shift_and_combine(simple_dataarray)
        assert result.dims == simple_dataarray.dims

    def test_dataarray_coordinates_sliced(self, simple_dataarray: xr.DataArray):
        result = shift_and_combine(simple_dataarray, offset_start=1, offset_end=1)
        assert len(result.pressure_level) == result.size

    @pytest.mark.parametrize("start_offset", [-5, -1, 0, 1])
    @pytest.mark.parametrize(
        "negative_offset",
        [-1, -5, -10, -100],
        ids=["minus_one", "minus_five", "minus_ten", "minus_hundred"],
    )
    def test_negative_offset_end_raises_error(
        self, simple_dataarray: xr.DataArray, negative_offset: int, start_offset: int
    ):
        with pytest.raises(ValueError, match=r".*offset.*must be non-negative"):
            _ = shift_and_combine(simple_dataarray, offset_end=negative_offset, offset_start=start_offset)

    def test_shift_dim_not_in_coords(self, simple_dataarray: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="Dimension to shift must be a coordinate"):
            _ = shift_and_combine(simple_dataarray, shift_dim="blah", shift_fill=0)

    @pytest.mark.parametrize(
        ("offset_start", "offset_end", "expected_size"),
        [
            (0, 0, 10),  # No offsets, full size
            (1, 1, 8),  # Symmetric offsets
            (2, 1, 7),  # Asymmetric offsets
            (3, 2, 5),  # Larger offsets
            (1, 0, 9),  # Only start offset
            (0, 1, 9),  # Only end offset
        ],
        ids=[
            "no_offsets",
            "symmetric_offsets",
            "asymmetric_offsets",
            "large_offsets",
            "start_only",
            "end_only",
        ],
    )
    def test_dataarray_offset_combinations(
        self, simple_dataarray: xr.DataArray, offset_start: int, offset_end: int, expected_size: int
    ):
        result = shift_and_combine(
            simple_dataarray,
            offset_start=offset_start,
            offset_end=offset_end,
            shift_fill=0,
        )
        assert result.size == expected_size

    @pytest.mark.parametrize(
        "combine_func",
        [
            add_combine,
            mean_combine,
            min_combine,
            max_combine,
        ],
        ids=["addition", "mean", "min", "max"],
    )
    @pytest.mark.parametrize(
        "fill_value",
        [np.nan, 0.0, -999.0, 1e10, -1e10],
        ids=["nan", "zero", "negative", "large_positive", "large_negative"],
    )
    def test_dataarray_with_different_fill_values_and_combine_func(
        self, pressure_level_dataarray: xr.DataArray, fill_value: float, combine_func: Callable
    ):
        result = shift_and_combine(
            pressure_level_dataarray,
            shift_fill=fill_value,
            combine_func=combine_func,
        )
        assert isinstance(result, xr.DataArray)
        assert result.size > 0
        assert result.size == pressure_level_dataarray.size - 2

    @pytest.mark.parametrize(
        "combine_func",
        [
            add_combine,
            mean_combine,
            min_combine,
            max_combine,
        ],
        ids=["addition", "mean", "min", "max"],
    )
    @pytest.mark.parametrize(
        ("offset_start", "offset_end"),
        [
            (0, 0),  # No offsets, full size
            (1, 1),  # Symmetric offsets
            (2, 1),  # Asymmetric offsets
            (3, 2),  # Larger offsets
            (1, 0),  # Only start offset
            (0, 1),  # Only end offset
        ],
        ids=[
            "no_offsets",
            "symmetric_offsets",
            "asymmetric_offsets",
            "large_offsets",
            "start_only",
            "end_only",
        ],
    )
    def test_dataset_offset_combinations(
        self, multi_var_dataset: xr.DataArray, offset_start: int, offset_end: int, combine_func: Callable
    ):
        original_sizes: dict[Hashable, int] = dict(multi_var_dataset.sizes)
        result = shift_and_combine(
            multi_var_dataset,
            offset_start=offset_start,
            offset_end=offset_end,
            combine_func=combine_func,
            shift_fill=0,
        )
        for name, var_size in result.sizes.items():
            assert (original_sizes[name] - (offset_start + offset_end)) == var_size

    @pytest.mark.parametrize("offset", [1, 2, 3, 4])
    def test_dataarray_symmetric_offsets(self, simple_dataarray: xr.DataArray, offset: int):
        result = shift_and_combine(
            simple_dataarray,
            offset_start=offset,
            offset_end=offset,
        )
        assert result.size == simple_dataarray.size - 2 * offset

    @pytest.mark.parametrize("shift_dim", ["pressure_level"])
    def test_dataarray_custom_shift_dim_single_dim(self, simple_dataarray: xr.DataArray, shift_dim: str):
        result = shift_and_combine(
            simple_dataarray,
            shift_dim=shift_dim,
        )
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize(
        "combine_func",
        [
            add_combine,
            mean_combine,
            min_combine,
            max_combine,
        ],
        ids=["addition", "mean", "min", "max"],
    )
    @pytest.mark.parametrize(
        ("shift_dim", "affected_axis"),
        [
            ("lat", 0),
            ("pressure_level", 1),
            ("lon", 2),
        ],
    )
    def test_dataarray_multidimensional_shift_dims(
        self, multi_dim_dataarray: xr.DataArray, shift_dim: str, affected_axis: int, combine_func: Callable
    ):
        result = shift_and_combine(
            multi_dim_dataarray,
            shift_dim=shift_dim,
            offset_start=1,
            offset_end=1,
            combine_func=combine_func,
        )

        # Only the affected axis should be reduced
        expected_shape = list(multi_dim_dataarray.shape)
        expected_shape[affected_axis] -= 2
        assert result.shape == tuple(expected_shape)
