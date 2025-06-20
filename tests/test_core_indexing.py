from typing import TYPE_CHECKING, Sequence

import numpy as np
import pytest

from rojak.core.indexing import get_regular_grid_spacing, make_value_based_slice

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
        (np.linspace(5, 10, 11), 0.5),
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
        (np.asarray([3, 5.6, 6]), None),
    ],
)
def test_get_regular_grid_spacing(array: "NDArray", expected: float | None) -> None:
    spacing = get_regular_grid_spacing(array)
    assert spacing == expected
