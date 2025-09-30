from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from xarray import testing as xrt

from rojak.core.calculations import (
    _icao_constants,
    altitude_to_pressure_troposphere,
    bilinear_interpolation,
    pressure_to_altitude_icao,
    pressure_to_altitude_stratosphere,
    pressure_to_altitude_troposphere,
    pressure_to_altitude_us_std_atm,
)
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from rojak.utilities.types import NumpyOrDataArray


def test_pressure_to_altitude_standard_atmosphere() -> None:
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, pressure_to_altitude_us_std_atm(pressures), rtol=1e-3)


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
@pytest.mark.parametrize(
    ("pressures", "converter_method", "descriptor"),
    [
        pytest.param(np.asarray([220]), pressure_to_altitude_troposphere, "less than", id="single value troposphere"),
        pytest.param(
            np.asarray([_icao_constants.tropopause_pressure]),
            pressure_to_altitude_troposphere,
            "less than",
            id="troposphere boundary value",
        ),
        pytest.param(
            np.asarray([220, 230]),
            pressure_to_altitude_troposphere,
            "less than",
            id="one value is above troposphere",
        ),
        pytest.param(
            np.linspace(_icao_constants.tropopause_pressure, 100, 10),
            pressure_to_altitude_troposphere,
            "less than",
            id="multiple value is above troposphere",
        ),
        pytest.param(
            np.linspace(1013, _icao_constants.tropopause_pressure, 20),
            pressure_to_altitude_troposphere,
            "less than",
            id="final value is tropopause boundary",
        ),
        pytest.param(
            np.asarray([230]), pressure_to_altitude_stratosphere, "greater than", id="single value stratosphere"
        ),
        pytest.param(
            np.asarray([220, 230]),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="one value is below stratosphere",
        ),
        pytest.param(
            np.linspace(250, 500, 10),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="multiple value is below stratosphere",
        ),
        pytest.param(
            np.linspace(_icao_constants.tropopause_pressure, 500, 10),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="multiple value is below stratosphere (with boundary)",
        ),
    ],
)
def test_pressure_to_altitude_fails_checks(
    pressures: "NumpyOrDataArray", converter_method: Callable, descriptor: str, wrap_in_data_array: bool, is_2d: bool
) -> None:
    matches: str = (
        f"Attempting to convert pressure to altitude for troposphere with pressure {descriptor} tropopause pressure"
    )
    if is_2d:
        pressures = pressures * np.ones((pressures.size, pressures.size))
    if wrap_in_data_array:
        pressures = xr.DataArray(pressures)
    with pytest.raises(ValueError, match=matches):
        converter_method(pressures)


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_troposphere_and_vice_versa(wrap_in_data_array: bool, is_2d: bool) -> None:
    # Values from Metric Table I in section 5 of NACA3182
    pressure = np.asarray(
        [1013.25, 794.95, 701.08, 616.40, 577.28, 471.81, 452.72, 410.61, 330.99, 350.88, 300.62, 250.50]
    )
    altitude_from_table = np.asarray([0, 2000, 3000, 4000, 4500, 6000, 6300, 7000, 8500, 8100, 9150, 10350])

    if is_2d:
        pressure = pressure.reshape((2, 6))
        altitude_from_table = altitude_from_table.reshape((2, 6))

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    computed_altitude = pressure_to_altitude_troposphere(pressure)
    computed_pressure = altitude_to_pressure_troposphere(altitude_from_table)
    np.testing.assert_allclose(computed_pressure, pressure, rtol=1e-4)
    # rtol based on altitude -> pressure as table is from altitude -> pressure
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)
    if not is_2d:
        if wrap_in_data_array:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure).values, computed_altitude)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure), computed_altitude)

    # Test that we get back approximately the same thing when passed through inverses
    np.testing.assert_allclose(altitude_to_pressure_troposphere(computed_altitude), pressure)
    np.testing.assert_allclose(pressure_to_altitude_troposphere(computed_pressure), altitude_from_table)


def test_pressure_to_altitude_troposphere_equiv_to_wallace() -> None:
    pressure = np.asarray([1013.25, 794.95, 701.08, 616.40, 577.28, 478.81, 410.61, 330.99, 350.88, 300.62, 250.50])
    np.testing.assert_allclose(
        pressure_to_altitude_troposphere(pressure), pressure_to_altitude_us_std_atm(pressure), rtol=1e-3
    )


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_stratosphere(is_2d: bool, wrap_in_data_array: bool) -> None:
    pressure = np.asarray([226.32, 199.50, 175.85, 150.20, 124.30, 99.68])
    altitude_from_table = np.asarray([11000, 11800, 12600, 13600, 14800, 16200])

    if is_2d:
        pressure = pressure.reshape((2, 3))
        altitude_from_table = altitude_from_table.reshape((2, 3))

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    computed_altitude = pressure_to_altitude_stratosphere(pressure)
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)

    if not is_2d:
        if wrap_in_data_array:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure).values, computed_altitude)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure), computed_altitude)


@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_icao(wrap_in_data_array) -> None:
    pressure = np.asarray(
        [
            1013.25,
            794.95,
            701.08,
            616.40,
            577.28,
            471.81,
            452.72,
            410.61,
            330.99,
            350.88,
            300.62,
            250.50,
            226.32,
            199.50,
            175.85,
            150.20,
            124.30,
            99.68,
        ]
    )
    altitude_from_table = np.asarray(
        [0, 2000, 3000, 4000, 4500, 6000, 6300, 7000, 8500, 8100, 9150, 10350, 11000, 11800, 12600, 13600, 14800, 16200]
    )

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    converted_pressure = pressure_to_altitude_icao(pressure)
    np.testing.assert_equal(np.round(converted_pressure, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(converted_pressure, altitude_from_table, rtol=1e-4)


def test_pressure_to_altitude_icao_era5_data(make_dummy_cat_data) -> None:
    dummy_data = make_dummy_cat_data({"pressure_level": [300.62, 250.50, 226.32, 199.50]})
    altitude_from_table = np.asarray([9150, 10350, 11000, 11800])
    computed_altitude = pressure_to_altitude_icao(dummy_data["pressure_level"])
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)


def linear_function(x_vals, y_vals):
    return x_vals + y_vals


def quadratic_function(x_vals, y_vals):
    return x_vals * x_vals + y_vals * y_vals


@pytest.mark.parametrize(("ff", "is_ff_linear"), [(linear_function, True), (quadratic_function, False)])
@pytest.mark.parametrize(
    ("x_slice", "y_slice", "interpolation_point"),
    [
        (slice(2), slice(2), (-1, -1)),
        (slice(2), slice(1, None), (1, -1)),
        (slice(1, None), slice(2), (-1, 1)),
        (slice(1, None), slice(1, None), (1, 1)),
    ],
)
def test_bilinear_interpolation(x_slice, y_slice, interpolation_point, ff, is_ff_linear) -> None:
    x = np.asarray([-2, 0, 2])
    y = np.asarray([-2, 0, 2])
    xx, yy = np.meshgrid(x, y)

    rgi = RegularGridInterpolator((x, y), ff(xx, yy))
    interpolated = bilinear_interpolation(
        x[x_slice], y[y_slice], ff(*np.meshgrid(x[x_slice], y[y_slice])), Coordinate(*interpolation_point)
    )
    assert rgi(interpolation_point) == interpolated
    if is_ff_linear:
        assert interpolated == ff(interpolation_point[0], interpolation_point[1])
