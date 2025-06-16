import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from xarray import testing as xrt

from rojak.core.calculations import Coordinate, bilinear_interpolation, pressure_to_altitude_std_atm


def test_pressure_to_altitude_standard_atmosphere() -> None:
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, pressure_to_altitude_std_atm(pressures), rtol=1e-3)


def linear_function(x_vals, y_vals):
    return x_vals + y_vals


def quadratic_function(x_vals, y_vals):
    return x_vals + y_vals


@pytest.mark.parametrize(
    ("x_slice", "y_slice", "interpolation_point", "ff"),
    [
        (slice(2), slice(2), (-1, -1), linear_function),
        (slice(2), slice(1, None), (1, -1), linear_function),
        (slice(1, None), slice(2), (-1, 1), linear_function),
        (slice(1, None), slice(1, None), (1, 1), linear_function),
        (slice(2), slice(2), (-1, -1), quadratic_function),
        (slice(2), slice(1, None), (1, -1), quadratic_function),
        (slice(1, None), slice(2), (-1, 1), quadratic_function),
        (slice(1, None), slice(1, None), (1, 1), quadratic_function),
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
