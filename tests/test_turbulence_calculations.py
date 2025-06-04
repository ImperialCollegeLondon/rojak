from typing import Tuple

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from rojak.turbulence.calculations import (
    EARTH_AVG_RADIUS,
    WrapAroundAngleArray,
    angles_gradient,
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


@pytest.fixture
def generate_random_array_pair() -> Tuple[xr.DataArray, xr.DataArray]:
    return xr.DataArray(np.random.default_rng().random((50, 50))), xr.DataArray(
        np.random.default_rng().random((50, 50))
    )


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


def test_coriolis_param_and_derivative_ncl_rossby_values():
    # Values from https://www.ncl.ucar.edu/Document/Functions/Contributed/beta_dfdy_rossby.shtml
    latitudes = xr.DataArray(np.arange(-90, 95, 5))
    coriolis = xr.DataArray([-1.46E-04, -1.45E-04, -1.44E-04, -1.41E-04, -1.37E-04, -1.32E-04, -1.26E-04, -1.19E-04,
                             -1.12E-04, -1.03E-04, -9.37E-05, -8.37E-05, -7.29E-05, -6.16E-05, -4.99E-05, -3.77E-05,
                             -2.53E-05, -1.27E-05, 0.00E+00, 1.27E-05, 2.53E-05, 3.77E-05, 4.99E-05, 6.16E-05,
                             7.29E-05, 8.37E-05, 9.37E-05, 1.03E-04, 1.12E-04, 1.19E-04, 1.26E-04, 1.32E-04, 1.37E-04,
                             1.41E-04, 1.44E-04, 1.45E-04, 1.46E-04])  # fmt: skip
    beta = xr.DataArray([-1.00E-18, 2.00E-12, 3.97E-12, 5.92E-12, 7.83E-12, 9.67E-12, 1.14E-11, 1.31E-11, 1.47E-11,
                         1.62E-11, 1.75E-11, 1.88E-11, 1.98E-11, 2.07E-11, 2.15E-11, 2.21E-11, 2.25E-11, 2.28E-11,
                         2.29E-11, 2.28E-11, 2.25E-11, 2.21E-11, 2.15E-11, 2.07E-11, 1.98E-11, 1.88E-11, 1.75E-11,
                         1.62E-11, 1.47E-11, 1.31E-11, 1.14E-11, 9.67E-12, 7.83E-12, 5.92E-12, 3.97E-12, 2.00E-12,
                         -1.00E-18])  # fmt: skip
    xrt.assert_allclose(coriolis, coriolis_parameter(latitudes), rtol=1e-2)
    xrt.assert_allclose(beta, latitudinal_derivative(coriolis_parameter(latitudes)))


def test_direction_class_type(generate_random_array_pair):
    arr, _ = generate_random_array_pair
    assert issubclass(WrapAroundAngleArray, np.ndarray)
    direction: np.ndarray = WrapAroundAngleArray(arr)
    assert isinstance(direction, WrapAroundAngleArray)
    # See https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy-asarray
    assert np.shares_memory(np.asarray(direction), arr)
    # np.asarray() does not pass through ndarray subclasses
    assert np.asarray(direction) is not direction
    # np.asanyarray() passes through ndarray subclasses
    assert np.asanyarray(direction) is direction


def test_direction_cross_zero_single_value():
    np.testing.assert_array_equal(
        np.array([np.pi / 2]),
        WrapAroundAngleArray(np.array([np.pi / 4])) - WrapAroundAngleArray(np.array([7.0 * np.pi / 4])),
    )
    np.testing.assert_array_equal(
        np.array([np.pi / 2]),
        WrapAroundAngleArray(np.array([7.0 * np.pi / 4])) - WrapAroundAngleArray(np.array(np.pi / 4)),
    )


def test_direction_cross_zero_array():
    initial: np.ndarray = np.array([np.pi / 4, 1.0, np.pi / 2])
    other: np.ndarray = initial + (5 * np.pi / 4)

    initial_direction: np.ndarray = WrapAroundAngleArray(initial)
    other_direction: np.ndarray = initial_direction + (5 * np.pi / 4)

    assert isinstance(initial_direction - other_direction, WrapAroundAngleArray)
    assert isinstance(initial - other, np.ndarray)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, np.abs(initial - other), initial_direction - other_direction
    )


@pytest.fixture
def angles_data():
    return [np.pi / 4, 1.0, np.pi / 2, 7.0 * np.pi / 4, 0, 0, 5.0 * np.pi / 4]


def test_angle_array_gradient(angles_data):
    # Values in angle_gradient = WrapAroundAngleArray([0.21460184, 0.39269908, 0.89269908, 0.78539816,
    #                                                  0.39269908, 1.17809725, 3.92699082])
    desired_gradient: WrapAroundAngleArray = np.gradient(WrapAroundAngleArray(np.array(angles_data)))
    # Values in normal_gradient = array([ 0.21460184,  0.39269908,  2.24889357, -0.78539816, -2.74889357,
    #                                     1.96349541,  3.92699082])
    normal_gradient: np.ndarray = np.gradient(np.array(angles_data))

    assert np.all(desired_gradient > 0)
    assert np.any(normal_gradient < 0)
    assert isinstance(desired_gradient, WrapAroundAngleArray)


# Add tests which show which cases of nesting WrapAroundAngleArray don't work


def test_angle_array_gradient_ufunc_simple(angles_data):
    import xarray as xr

    target_array: np.ndarray = np.array(
        [
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
            angles_data,
        ]
    )
    data_array: xr.DataArray = xr.DataArray(data=target_array)
    chunked: xr.DataArray = data_array.chunk({"dim_0": 3, "dim_1": 7})
    axis_which_varies: int = chunked.get_axis_num("dim_1")

    parallelised_ufunc: xr.DataArray = xr.apply_ufunc(
        angles_gradient,
        chunked,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"target_axis": axis_which_varies},
    ).compute()

    single_thread: xr.DataArray = np.gradient(WrapAroundAngleArray(target_array), axis=1)
    np.testing.assert_array_equal(angles_gradient(target_array, target_axis=axis_which_varies), single_thread)
    np.testing.assert_array_equal(single_thread, parallelised_ufunc.data)


def test_direction_behaves_like_normal_abs_sub(generate_random_array_pair):
    initial_angles, subsequent_angles = generate_random_array_pair
    initial_angles = initial_angles * np.pi
    subsequent_angles = subsequent_angles * np.pi
    initial_as_direction: np.ndarray = WrapAroundAngleArray(initial_angles)
    subsequent_as_direction: np.ndarray = WrapAroundAngleArray(subsequent_angles)

    np.testing.assert_array_equal(
        np.abs(initial_angles - subsequent_angles), initial_as_direction - subsequent_as_direction
    )
    np.testing.assert_array_equal(
        np.zeros_like(initial_angles) - subsequent_angles, np.zeros_like(initial_angles) - subsequent_as_direction
    )
    np.testing.assert_array_equal(
        np.abs(initial_angles - np.zeros_like(subsequent_angles)),
        initial_as_direction - np.zeros_like(subsequent_as_direction),
    )
    np.testing.assert_array_equal(
        np.ones_like(initial_angles) - subsequent_angles, np.ones_like(initial_angles) - subsequent_as_direction
    )
    np.testing.assert_array_equal(
        np.abs(initial_angles - np.ones_like(subsequent_angles)),
        initial_as_direction - np.ones_like(subsequent_as_direction),
    )


def test_direction_not_behave_like_normal_sub(generate_random_array_pair):
    initial_angles, subsequent_angles = generate_random_array_pair
    initial_angles = initial_angles * (np.pi / 2)
    subsequent_angles = subsequent_angles * (3 * np.pi / 2)
    initial_as_direction = WrapAroundAngleArray(initial_angles)
    subsequent_as_direction = WrapAroundAngleArray(subsequent_angles)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        np.abs(initial_angles - subsequent_angles),
        initial_as_direction - subsequent_as_direction,
    )
