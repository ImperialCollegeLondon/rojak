from typing import Tuple

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from rojak.turbulence.calculations import (
    magnitude_of_vector,
    shearing_deformation,
    stretching_deformation,
    total_deformation,
)


@pytest.fixture()
def generate_random_array_pair() -> Tuple[xr.DataArray, xr.DataArray]:
    return xr.DataArray(np.random.default_rng().random((3, 3))), xr.DataArray(np.random.default_rng().random((3, 3)))


def test_shear_deformation(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(x + y, shearing_deformation(x, y))


def test_shearing_deformation(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(x - y, stretching_deformation(x, y))


def test_magnitude_of_vector_squared(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(x * x + y * y, magnitude_of_vector(x, y, is_squared=True))


def test_magnitude_of_vector_default(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(np.hypot(x, y), magnitude_of_vector(x, y))
    xrt.assert_equal(np.hypot(x, y), magnitude_of_vector(x, y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, y), magnitude_of_vector(-x, y, is_abs=True))
    xrt.assert_equal(np.hypot(x, -y), magnitude_of_vector(x, -y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, -y), magnitude_of_vector(-x, -y, is_abs=True))
    xrt.assert_equal(np.hypot(x, -y), magnitude_of_vector(-x, y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, y), magnitude_of_vector(x, -y, is_abs=True))


def test_magnitude_of_vector_abs_squared(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(np.abs(x * x) + np.abs(y * y), magnitude_of_vector(x, y, is_abs=True, is_squared=True))
    xrt.assert_equal(np.abs(x * x) + np.abs(y * y), magnitude_of_vector(x, y, is_squared=True))


def test_total_deformation(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(np.hypot(x - x, y + y), total_deformation(x, y, y, x, False))
    xrt.assert_equal(np.hypot(x - y, x + y), total_deformation(x, y, x, y, False))
    xrt.assert_equal(np.hypot(x - y, x + y), total_deformation(x, x, y, y, False))
    xrt.assert_equal(np.hypot(x - x, x + x), total_deformation(x, x, x, x, False))
    xrt.assert_equal(np.hypot(y - y, y + y), total_deformation(y, y, y, y, False))

    xrt.assert_equal((x - x) ** 2 + (y + y) ** 2, total_deformation(x, y, y, x, True))
    xrt.assert_equal((x - y) ** 2 + (x + y) ** 2, total_deformation(x, y, x, y, True))
    xrt.assert_equal((x - y) ** 2 + (x + y) ** 2, total_deformation(x, x, y, y, True))
    xrt.assert_equal((x - x) ** 2 + (x + x) ** 2, total_deformation(x, x, x, x, True))
    xrt.assert_equal((y - y) ** 2 + (y + y) ** 2, total_deformation(y, y, y, y, True))
