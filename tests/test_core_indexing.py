from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rojak.core.indexing import (
    get_regular_grid_spacing,
    make_value_based_slice,
    map_order,
    map_values_to_nearest_coordinate_index,
    map_values_to_nearest_index_irregular_grid,
    nearest_index,
    shift_and_combine,
)

if TYPE_CHECKING:
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


def _brute_force_nearest_index(coordinate: "NDArray", values: "NDArray") -> "NDArray":
    # Reference implementation: smallest distance, with ties broken towards the larger coordinate value
    distance = np.abs(values[:, np.newaxis] - coordinate[np.newaxis, :])
    is_closest = distance == distance.min(axis=1, keepdims=True)
    return np.where(is_closest, coordinate[np.newaxis, :], -np.inf).argmax(axis=1)


class TestNearestIndex:
    @pytest.mark.parametrize("is_descending", [False, True], ids=["ascending", "descending"])
    def test_matches_pandas_get_indexer(self, is_descending: bool) -> None:
        coordinate = np.arange(0.0, 100.1, 10.0)
        if is_descending:
            coordinate = coordinate[::-1]
        rng = np.random.default_rng(6)
        # Include values outside of the coordinate and exact ties halfway between coordinate values
        values = np.concatenate([rng.uniform(-20, 120, 200), np.arange(-5.0, 106.0, 10.0)])

        np.testing.assert_array_equal(
            nearest_index(coordinate, values), pd.Index(coordinate).get_indexer(pd.Index(values), method="nearest")
        )

    def test_unsorted_irregular_coordinate_matches_brute_force(self) -> None:
        rng = np.random.default_rng(7)
        coordinate = rng.permutation(np.cumsum(rng.uniform(1, 10, 30)))
        # Include values outside of the coordinate and exact ties halfway between coordinate values
        ties = (np.sort(coordinate)[1:] + np.sort(coordinate)[:-1]) / 2
        values = np.concatenate([rng.uniform(coordinate.min() - 20, coordinate.max() + 20, 500), ties])

        np.testing.assert_array_equal(nearest_index(coordinate, values), _brute_force_nearest_index(coordinate, values))

    def test_datetime_coordinate(self) -> None:
        coordinate = np.arange(np.datetime64("2025-01-01T00"), np.datetime64("2025-01-02T00"), np.timedelta64(6, "h"))
        values = np.array(["2024-12-31T20", "2025-01-01T03", "2025-01-01T08", "2025-01-01T21"], dtype="datetime64[h]")
        np.testing.assert_array_equal(nearest_index(coordinate, values), [0, 1, 1, 3])

    def test_single_point_coordinate(self) -> None:
        np.testing.assert_array_equal(nearest_index(np.array([5.0]), np.array([-10.0, 5.0, 20.0])), [0, 0, 0])

    @pytest.mark.parametrize(
        ("coordinate", "values", "expected"),
        [
            pytest.param(np.array([0.0, 10.0, 20.0]), np.array([np.nan, 19.0]), [0, 2], id="nan"),
            pytest.param(
                np.arange(np.datetime64("2025-01-01T00"), np.datetime64("2025-01-02T00"), np.timedelta64(6, "h")),
                np.array(["NaT", "2025-01-01T17"], dtype="datetime64[h]"),
                [0, 3],
                id="nat",
            ),
        ],
    )
    def test_missing_values_map_to_first_index(
        self, coordinate: "NDArray", values: "NDArray", expected: list[int]
    ) -> None:
        np.testing.assert_array_equal(nearest_index(coordinate, values), expected)

    def test_preserves_shape_of_values(self) -> None:
        values = np.array([[1.0, 12.0], [27.0, -3.0]])
        np.testing.assert_array_equal(nearest_index(np.array([0.0, 10.0, 20.0, 30.0]), values), [[0, 1], [3, 0]])

    def test_raises_if_coordinate_not_1d(self) -> None:
        with pytest.raises(ValueError, match="Coordinate must be 1D"):
            nearest_index(np.zeros((2, 2)), np.array([1.0]))


class TestMapValuesToNearestIndexIrregularGrid:
    coordinate: "NDArray" = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 250.0, 200.0])

    def test_pandas_series(self) -> None:
        series = pd.Series([990.0, 775.0, 260.0, 100.0, 1100.0])
        result = map_values_to_nearest_index_irregular_grid(series, self.coordinate)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0, 1, 5, 6, 0])

    def test_dask_series_is_lazy_and_matches_pandas(self) -> None:
        rng = np.random.default_rng(8)
        series = pd.Series(rng.uniform(150, 1050, 100))
        result = map_values_to_nearest_index_irregular_grid(dd.from_pandas(series, npartitions=4), self.coordinate)

        assert isinstance(result, da.Array)
        np.testing.assert_array_equal(
            result.compute(), map_values_to_nearest_index_irregular_grid(series, self.coordinate)
        )

    def test_ties_broken_towards_larger_value(self) -> None:
        # 775 is halfway between 850 and 700, and 225 is halfway between 250 and 200
        result = map_values_to_nearest_index_irregular_grid(pd.Series([775.0, 225.0]), self.coordinate)
        np.testing.assert_array_equal(result, [1, 5])
