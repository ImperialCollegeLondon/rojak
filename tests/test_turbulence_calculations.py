from typing import Tuple

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from rojak.turbulence.calculations import (
    EARTH_AVG_RADIUS,
    coriolis_parameter,
    latitudinal_derivative,
    magnitude_of_vector,
    potential_temperature,
    shearing_deformation,
    stretching_deformation,
    total_deformation,
    vertical_component_vorticity,
    wind_direction,
    wind_speed,
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


@pytest.mark.parametrize("function_to_test", [magnitude_of_vector, wind_speed])
def test_magnitude_of_vector_squared(generate_random_array_pair, function_to_test):
    x, y = generate_random_array_pair
    xrt.assert_equal(x * x + y * y, function_to_test(x, y, is_squared=True))


@pytest.mark.parametrize("function_to_test", [magnitude_of_vector, wind_speed])
def test_magnitude_of_vector_default(generate_random_array_pair, function_to_test):
    x, y = generate_random_array_pair
    xrt.assert_equal(np.hypot(x, y), function_to_test(x, y))
    xrt.assert_equal(np.hypot(x, y), function_to_test(x, y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, y), function_to_test(-x, y, is_abs=True))
    xrt.assert_equal(np.hypot(x, -y), function_to_test(x, -y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, -y), function_to_test(-x, -y, is_abs=True))
    xrt.assert_equal(np.hypot(x, -y), function_to_test(-x, y, is_abs=True))
    xrt.assert_equal(np.hypot(-x, y), function_to_test(x, -y, is_abs=True))


@pytest.mark.parametrize("function_to_test", [magnitude_of_vector, wind_speed])
def test_magnitude_of_vector_abs_squared(generate_random_array_pair, function_to_test):
    x, y = generate_random_array_pair
    xrt.assert_equal(np.abs(x * x) + np.abs(y * y), function_to_test(x, y, is_abs=True, is_squared=True))
    xrt.assert_equal(np.abs(x * x) + np.abs(y * y), function_to_test(x, y, is_squared=True))


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


def test_vorticity(generate_random_array_pair):
    x, y = generate_random_array_pair
    xrt.assert_equal(x - y, vertical_component_vorticity(x, y))
    xrt.assert_equal(x - y, stretching_deformation(x, y))


def test_potential_temperature_ncl_values():
    # Values from https://www.ncl.ucar.edu/Document/Functions/Contributed/pot_temp.shtml
    temperature = xr.DataArray([302.45, 301.25, 296.65, 294.05, 291.55, 289.05, 286.25, 283.25, 279.85, 276.25, 272.65,
                                268.65, 264.15, 258.35, 251.65, 243.45, 233.15, 220.75, 213.95, 206.65, 199.05, 194.65,
                                197.15, 201.55, 206.45, 211.85, 216.85, 221.45, 222.45, 225.65])  # fmt: skip
    pressure = xr.DataArray([100800, 100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000, 60000, 55000, 50000,
                             45000, 40000, 35000, 30000, 25000, 20000, 17500, 15000, 12500, 10000, 8000, 7000, 6000,
                             5000, 4000, 3000, 2500, 2000]) / 100  # fmt: skip
    theta = xr.DataArray([301.762, 301.25, 301.034, 303.045, 305.421, 308.098, 310.798, 313.669, 316.543, 319.706,
                          323.491, 327.553, 331.919, 335.753, 339.777, 343.521, 346.597, 349.789, 352.211, 355.528,
                          360.783, 376.058, 405.988, 431.206, 461.598, 499.026, 544.465, 603.697, 638.883, 690.782
                          ])  # fmt: skip

    xrt.assert_allclose(theta, potential_temperature(temperature, pressure), atol=0.8)


def test_potential_temperature_metpy_values():
    # https://github.com/Unidata/MetPy/blob/5a009293896d3fd14fe52718726c29b6c59e941e/tests/calc/test_thermo.py#L122
    temperature = xr.DataArray([278.0, 283.0, 291.0, 298.0])
    pressure = xr.DataArray([900.0, 500.0, 300.0, 100.0])
    theta = xr.DataArray([286.496, 344.981, 410.475, 575.348])
    xrt.assert_allclose(theta, potential_temperature(temperature, pressure))


def test_wind_direction():
    # Values from https://www.ncl.ucar.edu/Document/Functions/Contributed/wind_direction.shtml
    u = xr.DataArray([10, 0, 0, -10, 10, 10, -10, -10, 0])
    v = xr.DataArray([0, 10, -10, 0, 10, -10, 10, -10, 0])
    direction = xr.DataArray([270.0, 180.0, 360.0, 90.0, 225.0, 315.0, 135.0, 45.0, 0.0])
    xrt.assert_allclose(direction, np.rad2deg(wind_direction(u, v)))


def test_coriolis_param_and_derivative():
    # Values from https://www.ncl.ucar.edu/Document/Functions/Contributed/coriolis_param.shtml
    single_val = xr.DataArray([35])
    single_param = xr.DataArray([8.365038e-05])
    xrt.assert_allclose(single_param, coriolis_parameter(single_val))
    xrt.assert_allclose(single_param / EARTH_AVG_RADIUS, latitudinal_derivative(coriolis_parameter(single_val)))

    multiple_vals = xr.DataArray(np.linspace(-81, 81, 10))
    param = xr.DataArray([-0.0001440445, -0.0001299444, -0.0001031244, -6.62E-05, -2.28E-05, 2.28E-05, 6.62E-05,
                          0.0001031244, 0.0001299444, 0.0001440445])  # fmt: skip
    xrt.assert_allclose(param, coriolis_parameter(multiple_vals), atol=1e-4)
    xrt.assert_allclose(param / EARTH_AVG_RADIUS, latitudinal_derivative(coriolis_parameter(multiple_vals)))
