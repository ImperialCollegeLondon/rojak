from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rojak.core.constants import GAS_CONSTANT_DRY_AIR, GRAVITATIONAL_ACCELERATION

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.utilities.types import ArrayLike, Coordinate


def pressure_to_altitude_std_atm(pressure: "ArrayLike") -> "ArrayLike":
    """
    Equation 3.106 on page 104 in Wallace, J. M., and Hobbs, P. V., “Atmospheric Science: An Introductory Survey,”
    Elsevier Science & Technology, San Diego, UNITED STATES, 2006.

    ..math:: z = \\frac{T_0}{\\Gamma} \\left[ 1 - \\left( \\frac{p}{p_0} \\right)^{\\frac{R\\Gamma}{g}} \\right]
    """
    reference_temperature: float = 288.0  # kelvin
    gamma: float = 0.0065  # 6.5 K/km => 0.0065 K/m
    reference_pressure: float = 1013.25  # hPa
    return (reference_temperature / gamma) * (
        1 - ((pressure / reference_pressure) ** ((GAS_CONSTANT_DRY_AIR * gamma) / GRAVITATIONAL_ACCELERATION))
    )


def bilinear_interpolation(
    longitude: "NDArray", latitude: "NDArray", function_value: "NDArray", target_coordinate: "Coordinate"
) -> "NDArray":
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
