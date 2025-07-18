import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rojak.core.constants import GAS_CONSTANT_DRY_AIR, GRAVITATIONAL_ACCELERATION
from rojak.utilities.types import is_np_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.utilities.types import Coordinate, NumpyOrDataArray


@dataclasses.dataclass(frozen=True)
class _PressureToAltitudeConstantsICAO:
    """
    Class to store constants used to convert pressure to altitude based on ICAO manual [NACA3182]_

    Attributes:
        reference_pressure (float): Pressure, :math:`P_0`, in hPa as defined in Equation 9
        reference_temperature (float): Temperature, :math:`T_0`, in K as defined in Equation 9
        tropopause_pressure (float): Pressure, :math:`P^*`, in hPa as defined in Equation 29
        tropopause_temperature (float): Temperature, :math:`T^*`, in K as defined in Equation 14
        lapse_rate (float): Lapse rate, :math:`a=-\\frac{dT}{dH}`, in K/m as defined in Equation 12
        n_value (float): Dimensionless constant value, :math:`n=\\frac{G}{aR}`, as defined in Equation 17
        inverse_n (float): :math:`\\frac{1}{n}`
        geopotential_dimensional_constant (float):  Constant :math:`G` determines magnitude of :math:`H` in terms of
            length and time, as defined in Equation 6
    """

    reference_pressure: float = 1013.25  # P_0 in hPa
    reference_temperature: float = 288.16  # T_0 in K
    tropopause_pressure: float = 226.32  # P^* in hPa
    tropopause_temperature: float = 216.66  # T^* in K
    lapse_rate: float = 0.0065  # a in K / m
    n_value: float = 5.2561  # n = G/aR dimensionless
    inverse_n: float = 1 / 5.2621  # 1/n (dimensionless) where n = G / aR
    geopotential_dimensional_constant: float = 9.80665  # G

    @property
    def b_value(self) -> float:
        """
        Constant value named B in [NACA3182]_

        .. math ::
            \\text{B} = \\frac{G \\log_{10}(e)}{R T^*}

        Returns:

        """
        return (self.geopotential_dimensional_constant * np.log10(np.e)) / (
            GAS_CONSTANT_DRY_AIR * self.tropopause_temperature
        )

    @property
    def tropopause_height(self) -> float:
        """
        Tropopause height

        By rearranging equation 14 in [NACA3182]_,

        .. math ::
            \\text{H}^* = \\frac{T_0 - T^*}{a}

        Returns:

        """
        return (self.reference_temperature - self.tropopause_temperature) / self.lapse_rate


_icao_constants = _PressureToAltitudeConstantsICAO()


def pressure_to_altitude_us_std_atm(pressure: "NumpyOrDataArray") -> "NumpyOrDataArray":
    """
    Convert pressure to altitude for a standard atmosphere

    An implementation of Equation 3.106 on page 104 in [Wallace2006]_,

    .. math:: z = \\frac{T_0}{\\Gamma} \\left[ 1 - \\left( \\frac{p}{p_0} \\right)^{\\frac{R\\Gamma}{g}} \\right]

    Args:
        pressure (NumpyOrDataArray): Pressure in hPa
    """
    reference_temperature: float = 288.0  # kelvin
    gamma: float = 0.0065  # 6.5 K/km => 0.0065 K/m
    reference_pressure: float = 1013.25  # hPa
    return (reference_temperature / gamma) * (
        1 - ((pressure / reference_pressure) ** ((GAS_CONSTANT_DRY_AIR * gamma) / GRAVITATIONAL_ACCELERATION))
    )


def _check_if_pressures_are_valid(pressure: "NumpyOrDataArray", is_below_tropopause: bool) -> None:
    # Closer to ground => larger pressure
    # Thus, below tropopause => values > tropopause pressure. Condition will be opposite
    condition = (
        pressure <= _icao_constants.tropopause_pressure
        if is_below_tropopause
        else pressure > _icao_constants.tropopause_pressure
    )
    descriptive_comparator: str = "less than" if is_below_tropopause else "greater than"

    if pressure.ndim == 1 or is_np_array(pressure):
        if pressure[condition].size != 0:
            raise ValueError(
                f"Attempting to convert pressure to altitude for troposphere with pressure {descriptive_comparator} "
                "tropopause pressure"
            )
    elif pressure.where(condition, drop=True).size != 0:
        raise ValueError(
            f"Attempting to convert pressure to altitude for troposphere with pressure {descriptive_comparator} "
            f"tropopause pressure"
        )


def pressure_to_altitude_troposphere(pressure: "NumpyOrDataArray") -> "NumpyOrDataArray":
    """
    Convert pressure to altitude for the troposphere

    Please use :py:func:`pressure_to_altitude_icao` for pressures in both the troposphere and stratosphere.

    An implementation of equation 40 from ICAO manual [NACA3182]_

    .. math::
        \\text{H} = \\frac{T_0}{a} \\left[ 1 - \\left( \\frac{P}{P_0} \\right)^{1/n} \\right]

    where :math:`H` is the geopotential height and is treated as equivalent to altitude for engineering purposes
    (see [NACA3182]_ for details).

    Args:
        pressure (NumpyOrDataArray): Pressure in hPa

    Returns:

    """
    _check_if_pressures_are_valid(pressure, True)

    return (_icao_constants.reference_temperature / _icao_constants.lapse_rate) * (
        1 - (pressure / _icao_constants.reference_pressure) ** _icao_constants.inverse_n
    )


def altitude_to_pressure_troposphere(altitude: "NumpyOrDataArray") -> "NumpyOrDataArray":
    """
    Convert altitude to pressure for the troposphere

    An implementation of equation 18 from ICAO manual [NACA3182]_

    .. math::
        P = P_0 \\left( \\frac{T_0 - aH}{T_0} \\right) ^ n

    where :math:`a` is the lapse rate, :math:`H` is the altitude, and :math:`n = \\frac{G}{aR} = 5.2561` is a
    dimensionless constant.

    Args:
        altitude: Altitude in m

    Returns:
        NumpyOrDataArray: Pressure in hPa

    """
    return (
        _icao_constants.reference_pressure
        * (
            (_icao_constants.reference_temperature - _icao_constants.lapse_rate * altitude)
            / _icao_constants.reference_temperature
        )
        ** _icao_constants.n_value
    )


def pressure_to_altitude_stratosphere(pressure: "NumpyOrDataArray") -> "NumpyOrDataArray":
    """
    Convert pressure to altitude for the stratosphere

    Please use :py:func:`pressure_to_altitude_icao` for pressures in both the troposphere and stratosphere.

    An implementation of equation 41 from ICAO manual [NACA3182]_

    .. math::
        \\text{H} = \\text{H}^* + \\frac{1}{\\text{B}} \\left[ n \\log_{10} \\left( \\frac{T^*}{T_0} \\right) \\right]
        - \\frac{1}{\\text{B}} \\log_{10} \\left( \\frac{P}{P_0} \\right)

    where :math:`^*` indicates values at the tropopause, :math:`H` is the geopotential height and is treated as
    equivalent to altitude for engineering purposes (see [NACA3182]_ for details).

    Args:
        pressure (NumpyOrDataArray): Pressure in hPa

    Returns:

    """
    _check_if_pressures_are_valid(pressure, False)

    b_inverse: float = 1 / _icao_constants.b_value
    middle_constant_term = (
        b_inverse
        * _icao_constants.n_value
        * np.log10(_icao_constants.tropopause_temperature / _icao_constants.reference_temperature)
    )
    return (
        _icao_constants.tropopause_height
        + middle_constant_term
        - b_inverse * np.log10(pressure / _icao_constants.reference_pressure)
    )


def pressure_to_altitude_icao(pressure: "NumpyOrDataArray") -> "NumpyOrDataArray":
    """
    Convert pressure to altitude for ICAO standard atmosphere up to 80 km [NACA3182]_

    Args:
        pressure: One dimensional array of pressure in hPa

    Returns:
        One dimensional array of altitude in m

    """
    if pressure.ndim > 1:
        raise NotImplementedError("Multidimensional pressure not yet supported")

    mask = pressure > _icao_constants.tropopause_pressure
    pressures_within_troposphere = pressure[mask]
    pressures_within_stratosphere = pressure[~mask]

    if pressures_within_troposphere.size == pressure.size:
        return pressure_to_altitude_troposphere(pressure)
    if pressures_within_stratosphere.size == pressure.size:
        return pressure_to_altitude_stratosphere(pressure)

    if is_np_array(pressure):
        pressure[mask] = pressure_to_altitude_troposphere(pressures_within_troposphere)
        pressure[~mask] = pressure_to_altitude_stratosphere(pressures_within_stratosphere)
    else:
        pressure.loc[mask] = pressure_to_altitude_troposphere(pressures_within_troposphere)
        pressure.loc[~mask] = pressure_to_altitude_stratosphere(pressures_within_stratosphere)

    return pressure


def bilinear_interpolation(
    longitude: "NDArray", latitude: "NDArray", function_value: "NDArray", target_coordinate: "Coordinate"
) -> "NDArray":
    """
    Bilinear interpolation using 4 points

    Args:
        longitude:
        latitude:
        function_value:
        target_coordinate:

    Returns:

    """
    assert len(longitude) == len(latitude)
    assert len(longitude) > 1
    squeezed_values = np.squeeze(function_value)
    assert squeezed_values.ndim == 2, (  # noqa: PLR2004
        f"Function value ({function_value}, shape={function_value.shape}) must have two dimensions "
        f"instead of {function_value.ndim}"
    )

    return RegularGridInterpolator((longitude, latitude), squeezed_values, method="linear")(
        (target_coordinate.longitude, target_coordinate.latitude)
    )
