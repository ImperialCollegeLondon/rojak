from typing import Sequence

import numpy as np
import pytest

from rojak.core.indexing import make_value_based_slice


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
