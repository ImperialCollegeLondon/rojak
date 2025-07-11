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
    """
    Shear deformation

    .. math:: D_{\\text{sh}} = \\frac{ \\partial v }{ \\partial x }  + \\frac{ \\partial u }{ \\partial y }

    Args:
        dv_dx: Array containing values for :math:`\\frac{ \\partial v }{ \\partial x }`
        du_dy: Array containing values for :math:`\\frac{ \\partial u }{ \\partial y }`
    """
    return dv_dx + du_dy


def stretching_deformation(du_dx: xr.DataArray, dv_dy: xr.DataArray) -> xr.DataArray:
    """
    Stretch deformation

    .. math:: D_{\\text{st}} = \\frac{ \\partial u }{ \\partial x }  - \\frac{ \\partial v }{ \\partial y }

    Args:
        du_dx: Array containing values for :math:`\\frac{ \\partial u }{ \\partial x }`
        dv_dy: Array containing values for :math:`\\frac{ \\partial v }{ \\partial y }`
    """
    return du_dx - dv_dy


def total_deformation(
    du_dx: xr.DataArray, du_dy: xr.DataArray, dv_dx: xr.DataArray, dv_dy: xr.DataArray, is_squared: bool
) -> xr.DataArray:
    """
    Total deformation

    .. math:: \\text{DEF} = \\sqrt{ D_{\\text{sh}}^{2} + D_{\\text{st}}^{2} }

    where :math:`D_{\\text{sh}}` is the shear deformation and :math:`D_{\\text{st}}` is the stretch deformation.
    See :py:func:`shearing_deformation` and :py:func:`stretching_deformation` for more details.

    Args:
        du_dx: Array containing values for :math:`\\frac{ \\partial u }{ \\partial x }`
        du_dy: Array containing values for :math:`\\frac{ \\partial u }{ \\partial y }`
        dv_dx: Array containing values for :math:`\\frac{ \\partial v }{ \\partial x }`
        dv_dy: Array containing values for :math:`\\frac{ \\partial v }{ \\partial y }`
        is_squared (bool): Controls whether deformation is squared, i.e.
            :math:`D_{\\text{sh}}^{2} + D_{\\text{st}}^{2}`
    """
    return magnitude_of_vector(
        shearing_deformation(dv_dx, du_dy), stretching_deformation(du_dx, dv_dy), is_squared=is_squared
    )


def magnitude_of_vector(
    x_component: xr.DataArray, y_component: xr.DataArray, is_abs: bool = False, is_squared: bool = False
) -> xr.DataArray:
    """
    Magnitude of vector

    Convenience method to calculate magnitude of a 2D vector, i.e. :math:`\\sqrt{x^2 + y^2}`

    Args:
        x_component: x-values of vector
        y_component: y-values of vector
        is_abs: If ``True``, the components of vector is absolute (i.e. :math:`\\sqrt{|x|^2 + |y|^2}`.
            Defaults to ``False``.
        is_squared: If ``True``, the components of vector is squared (i.e. :math:`x^2 + y^2`).
            Defaults to ``False``.
    """
    x_component = np.abs(x_component) if is_abs else x_component  # pyright: ignore[reportAssignmentType]
    y_component = np.abs(y_component) if is_abs else y_component  # pyright: ignore[reportAssignmentType]

    return (x_component * x_component + y_component * y_component) if is_squared else np.hypot(x_component, y_component)  # pyright: ignore[reportReturnType]


def vertical_component_vorticity(dvdx: xr.DataArray, dudy: xr.DataArray) -> xr.DataArray:
    """
    Vertical component of vorticity

    .. math:: \\frac{ \\partial v }{ \\partial x } - \\frac{ \\partial u }{ \\partial y }

    Args:
        dvdx: Array containing values for :math:`\\frac{ \\partial v }{ \\partial x }`
        dudy: Array containing values for :math:`\\frac{ \\partial u }{ \\partial y }`
    ..
    """
    return dvdx - dudy


# TODO: TEST
def altitude_derivative_on_pressure_level(
    function: xr.DataArray, geopotential: xr.DataArray, level_coord_name: str = "pressure_level"
) -> xr.DataArray:
    """
    Derivative w.r.t. altitude for data on pressure level

    Using the definition of geopotential,

    .. math:: \\Phi = gz \\implies \\frac{ \\partial \\Phi }{ \\partial z } = g

    Express derivative w.r.t. altitude in terms of pressure. For example, for the function :math:`f`,

    .. math::
        \\frac{ \\partial f }{ \\partial z } = \\frac{ \\partial f }{ \\partial p }
            \\frac{ \\partial p }{ \\partial \\Phi } \\frac{ \\partial \\Phi }{ \\partial z }
            = g \\frac{ \\partial f }{ \\partial p } \\left( \\frac{ \\partial \\Phi }{ \\partial p } \\right)^{-1}

    Args:
        function: Function to perform derivative on that varies with pressure level
        geopotential: Geopotential data
        level_coord_name: Name of pressure level coordinate
    """
    return GRAVITATIONAL_ACCELERATION * (
        function.differentiate(level_coord_name) / geopotential.differentiate(level_coord_name)
    )


# Note: numpy dtype can't be overloaded as it's immutable. Therefore, go via array subclassing
class _WrapAroundAngleArray(np.ndarray):
    """
    Class extends numpy ndarray through array subclassing to overload the subtract (`__sub__`) method such that
    differences on the angles wraps around to avoid discontinuities
    """

    def __new__(cls, input_array: np.ndarray) -> np.ndarray:
        # https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
        return np.asarray(input_array).view(cls)

    def __sub__(self, other):  # noqa: ANN001, ANN204 # pyright:ignore [reportIncompatibleMethodOverride]
        if isinstance(other, self.__class__):
            abs_diff: np.ndarray = np.abs(np.asarray(self) - np.asarray(other))
            remaining_angle: np.ndarray = (2 * np.pi) - abs_diff
            return np.where(abs_diff < remaining_angle, abs_diff, remaining_angle).view(_WrapAroundAngleArray)
        raise TypeError(f"unsupported operand types(s) for: '{self.__class__} and '{other.__class__}'")


def angles_gradient(array: np.ndarray, target_axis: int, coord_values: np.ndarray | None = None) -> np.ndarray:
    """
    Gradient of array of angles

    Args:
        array: Array of angles
        target_axis: Target axis
        coord_values: Coordinate values
    """
    if coord_values is None:
        return np.gradient(_WrapAroundAngleArray(array), axis=target_axis)
    return np.gradient(_WrapAroundAngleArray(array), coord_values, axis=target_axis)


def wind_direction(u_wind: xr.DataArray, v_wind: xr.DataArray) -> xr.DataArray:
    """
    Wind direction

    Compute meteorological wind direction angle from wind components, :math:`u` and :math:`v`. This follows the
    convention and description in the `ECMWF docs`_

    Args:
        u_wind: :math:`u` wind component
        v_wind: :math:`v` wind component

    .. _ECMWF docs: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
    """
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
    """
    Potential temperature

    Computes potential temperature Poisson's Equation [Wallace2006]_ (Eqn. 3.54 on pg. 78),

    .. math:: \\Theta = T (P_0 / P)^\\kappa

    where :math:`P_0` is the reference pressure in hPa and :math:`\\kappa` is the dimensionless quantity
    :math:`R/C_p`.

    Args:
        temperature: Temperature in Kelvin
        pressure: Pressure in hPa
    """
    reference_pressure: int = 1000  # hPa
    kappa: float = 0.28571428571428564  # R / cp (dimensionless)
    if "units" in pressure.attrs and pressure.attrs["units"] != "hPa":
        raise NotImplementedError("Only case where pressure hPa units has been implemented")
    return temperature / ((pressure / reference_pressure) ** kappa)


def coriolis_parameter(latitude: xr.DataArray) -> xr.DataArray:
    """
    Coriolis parameter

    Computes coriolis parameter for latitudes using [Wallace2006]_ (pg. 277),

    .. math:: f = 2 \\Omega \\sin( \\phi )

    where :math:`\\Omega` is the earth's angular velocity and :math:`\\phi` is the latitude.

    Args:
        latitude: Array of latitude to compute the coriolis parameter
    """
    if latitude.max() > np.pi:
        latitude = np.deg2rad(latitude)  # pyright: ignore[reportAssignmentType]
    return 2 * EARTH_ANGULAR_VELOCITY * np.sin(latitude)  # pyright: ignore[reportReturnType]


def latitudinal_derivative(coriolis_param: xr.DataArray) -> xr.DataArray:
    """
    Latitudinal derivative or Rossby Parameter

    Computes latitudinal derivative of the coriolis parameter using [Wallace2006]_ (Eqn. 7.25 on pg. 288),

    .. math:: \\beta = \\frac{\\partial f}{\\partial y} = \\frac{2 \\Omega \\cos \\phi}{R_E}

    where :math:`R_E` is the radius of the earth

    Args:
        coriolis_param: Array of coriolis parameter values
    """
    return coriolis_param / EARTH_AVG_RADIUS


# TODO: TEST
def absolute_vorticity(vorticity: xr.DataArray) -> xr.DataArray:
    """
    Vertical component of absolute vorticity

    .. math:: \\zeta_{a} = \\zeta + f

    where :math:`\\zeta` is the vertical component of vorticity and :math:`f` is the coriolis parameter.

    Args:
        vorticity: Array of vorticity values
    """
    assert "latitude" in vorticity.coords, "latitude must be a coordinate of vorticity array"
    return vorticity + coriolis_parameter(vorticity["latitude"])


# TODO: TEST
# theta is potential temperature
def potential_vorticity(vorticity: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """
    Ertel's potential vorticity

    Isentropic potential vorticity is defined in [Wallace2006]_ (pg. 290) as,

    .. math:: \\text{PV} = -g \\zeta_{a} \\frac{ \\partial \\theta }{ \\partial p }

    where :math:`\\zeta_{a}` is the absolute vorticity (see :py:func:`absolute_vorticity`), :math:`\\theta` is the
    potential temperature, :math:`p` is the pressure, :math:`g` is the acceleration due to gravity.

    Args:
        vorticity: Array of vorticity values
        theta: Array of potential temperature values
    """
    return -GRAVITATIONAL_ACCELERATION * absolute_vorticity(vorticity) * theta.differentiate("pressure_level")


def magnitude_of_geospatial_gradient(
    array: xr.DataArray,
    is_squared: bool = False,
) -> xr.DataArray:
    """
    Magnitude of geospatial gradient

    Convenience method to compute magnitude of the geospatial gradient

    Args:
        array: Array to compute geospatial gradient on
        is_squared (optional): If `True`, returns square of the magnitude. Default is `False`.
    """
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
    """
    Vertical wind shear

    Vertical wind shear as defined in [Sharman2006]_ is,

    .. math::

        \\begin{align}
        S_{v} &= \\left| \\frac{ \\partial \\mathbf{u} }{ \\partial z }  \\right| \\\\
            &= \\sqrt{ \\left| \\frac{ \\partial u }{ \\partial z }  \\right|^{2} +
                \\left|  \\frac{ \\partial v }{ \\partial z } \\right| ^{2} }
        \\end{align}

    Args:
        u_wind: :math:`u` wind component
        v_wind: :math:`v` wind component
        geopotential (optional): Array of geopotential values. If provided, derivative in vertical coordinate will
            be on altitude (see py:func:`altitude_derivative_on_pressure_level` instead of on pressure level
        is_abs_velocities (optional): If `True`, uses absolute velocities. Default is `False`.
        is_vws_squared (optional): If `True`, returns square of the wind shear. Default is `False`.
    """
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
    """
    Wind speed

    .. math:: | \\mathbf{u} | = \\sqrt{u^2 + v^2}

    Args:
        u_wind: :math:`u` wind component
        v_wind: :math:`v` wind component
        is_abs (optional): If `True`, use absolute value of wind components. Defaults to `False`.
        is_squared (optional): If `True`, returns wind speed squared, i.e. :math:`u^2 + v^2`. Defaults to `False`.
    """
    return magnitude_of_vector(u_wind, v_wind, is_abs=is_abs, is_squared=is_squared)
