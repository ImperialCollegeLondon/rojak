from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from xarray import testing as xrt

from rojak.core.calculations import (
    bilinear_interpolation,
    icao_constants,
    pressure_to_altitude_std_atm,
    pressure_to_altitude_stratosphere,
    pressure_to_altitude_troposphere,
)
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from rojak.utilities.types import NumpyOrDataArray


def test_pressure_to_altitude_standard_atmosphere() -> None:
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, pressure_to_altitude_std_atm(pressures), rtol=1e-3)


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
@pytest.mark.parametrize(
    ("pressures", "converter_method", "descriptor"),
    [
        pytest.param(np.asarray([220]), pressure_to_altitude_troposphere, "less than", id="single value troposphere"),
        pytest.param(
            np.asarray([icao_constants.tropopause_pressure]),
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
            np.linspace(icao_constants.tropopause_pressure, 100, 10),
            pressure_to_altitude_troposphere,
            "less than",
            id="multiple value is above troposphere",
        ),
        pytest.param(
            np.linspace(1013, icao_constants.tropopause_pressure, 20),
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
            np.linspace(icao_constants.tropopause_pressure, 500, 10),
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


def linear_function(x_vals, y_vals):
    return x_vals + y_vals


def quadratic_function(x_vals, y_vals):
    return x_vals + y_vals


@pytest.mark.parametrize("ff", [linear_function, quadratic_function])
@pytest.mark.parametrize(
    ("x_slice", "y_slice", "interpolation_point"),
    [
        (slice(2), slice(2), (-1, -1)),
        (slice(2), slice(1, None), (1, -1)),
        (slice(1, None), slice(2), (-1, 1)),
        (slice(1, None), slice(1, None), (1, 1)),
    ],
)
def test_bilinear_interpolation(x_slice, y_slice, interpolation_point, ff) -> None:
    x = np.asarray([-2, 0, 2])
    y = np.asarray([-2, 0, 2])
    xx, yy = np.meshgrid(x, y)

    rgi = RegularGridInterpolator((x, y), ff(xx, yy))
    interpolated = bilinear_interpolation(
        x[x_slice], y[y_slice], ff(*np.meshgrid(x[x_slice], y[y_slice])), Coordinate(*interpolation_point)
    )
    assert rgi(interpolation_point) == interpolated
    assert interpolated == ff(interpolation_point[0], interpolation_point[1])
