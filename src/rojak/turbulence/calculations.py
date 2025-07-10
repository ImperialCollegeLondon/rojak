#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import numpy as np
import xarray as xr

from rojak.core import derivatives
from rojak.core.constants import GRAVITATIONAL_ACCELERATION

# https://physics.nist.gov/cgi-bin/cuu/Value?gn
EARTH_AVG_RADIUS: float = 6371008.7714  # m
EARTH_ANGULAR_VELOCITY: float = 7292115e-11  # rad/s


def shearing_deformation(dv_dx: xr.DataArray, du_dy: xr.DataArray) -> xr.DataArray:
    return dv_dx + du_dy


def stretching_deformation(du_dx: xr.DataArray, dv_dy: xr.DataArray) -> xr.DataArray:
    return du_dx - dv_dy


def total_deformation(
    du_dx: xr.DataArray, du_dy: xr.DataArray, dv_dx: xr.DataArray, dv_dy: xr.DataArray, is_squared: bool
) -> xr.DataArray:
    return magnitude_of_vector(
        shearing_deformation(dv_dx, du_dy), stretching_deformation(du_dx, dv_dy), is_squared=is_squared
    )


def magnitude_of_vector(
    x_component: xr.DataArray, y_component: xr.DataArray, is_abs: bool = False, is_squared: bool = False
) -> xr.DataArray:
    x_component = np.abs(x_component) if is_abs else x_component  # pyright: ignore[reportAssignmentType]
    y_component = np.abs(y_component) if is_abs else y_component  # pyright: ignore[reportAssignmentType]

    return (x_component * x_component + y_component * y_component) if is_squared else np.hypot(x_component, y_component)  # pyright: ignore[reportReturnType]


def vertical_component_vorticity(dvdx: xr.DataArray, dudy: xr.DataArray) -> xr.DataArray:
    return dvdx - dudy


# TODO: TEST
def altitude_derivative_on_pressure_level(
    function: xr.DataArray, geopotential: xr.DataArray, level_coord_name: str = "pressure_level"
) -> xr.DataArray:
    return GRAVITATIONAL_ACCELERATION * (
        function.differentiate(level_coord_name) / geopotential.differentiate(level_coord_name)
    )


# Note: numpy dtype can't be overloaded as it's immutable. Therefore, go via array subclassing
class WrapAroundAngleArray(np.ndarray):
    def __new__(cls, input_array: np.ndarray) -> np.ndarray:
        # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
        return np.asarray(input_array).view(cls)

    def __sub__(self, other):  # noqa: ANN001, ANN204 # pyright:ignore [reportIncompatibleMethodOverride]
        if isinstance(other, self.__class__):
            abs_diff: np.ndarray = np.abs(np.asarray(self) - np.asarray(other))
            remaining_angle: np.ndarray = (2 * np.pi) - abs_diff
            return np.where(abs_diff < remaining_angle, abs_diff, remaining_angle).view(WrapAroundAngleArray)
        raise TypeError(f"unsupported operand types(s) for: '{self.__class__} and '{other.__class__}'")


def angles_gradient(array: np.ndarray, target_axis: int, coord_values: np.ndarray | None = None) -> np.ndarray:
    if coord_values is None:
        return np.gradient(WrapAroundAngleArray(array), axis=target_axis)
    return np.gradient(WrapAroundAngleArray(array), coord_values, axis=target_axis)


def wind_direction(u_wind: xr.DataArray, v_wind: xr.DataArray) -> xr.DataArray:
    # See https://github.com/Unidata/MetPy/blob/34bfda1deaead3fed9070f3a766f7d842373c6d9/src/metpy/calc/basic.py#L106
    # np.arctan2 returns angle between hypotenuse and x-axis but wind direction is w.r.t y-axis
    # This means we need to do remove angle from pi/2
    # To get the **from** wind direction, we multiply values by -1 to change the quadrant
    direction: xr.DataArray = np.pi / 2 - np.arctan2(-v_wind, -u_wind)  # pyright: ignore [reportAssignmentType]
    # Direction in radians in range of [-pi, pi]
    # Meteorological wind direction angle must be [0, 2pi]
    met_wind_direction: xr.DataArray = xr.where(direction <= 0.0, direction + 2 * np.pi, direction)
    # If u and v are zero, np.arctan2([0.0], [0.0]) = -3.14159265
    return xr.where((u_wind == 0) & (v_wind == 0), 0, met_wind_direction)


def potential_temperature(temperature: xr.DataArray, pressure: xr.DataArray) -> xr.DataArray:
    r"""
    .. math:: \Theta = T (P_0 / P)^\kappa
    """
    reference_pressure: int = 1000  # hPa
    kappa: float = 0.28571428571428564  # R / cp (dimensionless)
    if "units" in pressure.attrs and pressure.attrs["units"] != "hPa":
        raise NotImplementedError("Only case where pressure hPa units has been implemented")
    return temperature / ((pressure / reference_pressure) ** kappa)


def coriolis_parameter(latitude: xr.DataArray) -> xr.DataArray:
    if latitude.max() > np.pi:
        latitude = np.deg2rad(latitude)  # pyright: ignore[reportAssignmentType]
    return 2 * EARTH_ANGULAR_VELOCITY * np.sin(latitude)  # pyright: ignore[reportReturnType]


def latitudinal_derivative(coriolis_param: xr.DataArray) -> xr.DataArray:
    return coriolis_param / EARTH_AVG_RADIUS


# TODO: TEST
def absolute_vorticity(vorticity: xr.DataArray) -> xr.DataArray:
    return vorticity + coriolis_parameter(vorticity["latitude"])


# TODO: TEST
# theta is potential temperature
def potential_vorticity(vorticity: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    return -GRAVITATIONAL_ACCELERATION * absolute_vorticity(vorticity) * theta.differentiate("pressure_level")


def magnitude_of_geospatial_gradient(
    array: xr.DataArray,
    is_squared: bool = False,
) -> xr.DataArray:
    grad = derivatives.spatial_gradient(array, "deg", derivatives.GradientMode.GEOSPATIAL)
    return magnitude_of_vector(grad["dfdx"], grad["dfdy"], is_squared=is_squared)


# TODO: TEST
def vertical_wind_shear(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    geopotential: xr.DataArray | None = None,
    is_abs_velocities: bool = False,
    is_vws_squared: bool = False,
) -> xr.DataArray:
    if geopotential is None:
        du_dz: xr.DataArray = u_wind.differentiate("pressure_level")
        dv_dz: xr.DataArray = v_wind.differentiate("pressure_level")
    else:
        du_dz: xr.DataArray = altitude_derivative_on_pressure_level(u_wind, geopotential)
        dv_dz: xr.DataArray = altitude_derivative_on_pressure_level(v_wind, geopotential)

    return magnitude_of_vector(du_dz, dv_dz, is_abs=is_abs_velocities, is_squared=is_vws_squared)


def wind_speed(
    u_wind: xr.DataArray, v_wind: xr.DataArray, is_abs: bool = False, is_squared: bool = False
) -> xr.DataArray:
    return magnitude_of_vector(u_wind, v_wind, is_abs=is_abs, is_squared=is_squared)
