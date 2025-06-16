from typing import TYPE_CHECKING, NamedTuple, Sequence

from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.utilities.types import ArrayLike


def pressure_to_altitude_std_atm(pressure: "ArrayLike") -> "ArrayLike":
    """
    Equation 3.106 on page 104 in Wallace, J. M., and Hobbs, P. V., “Atmospheric Science: An Introductory Survey,”
    Elsevier Science & Technology, San Diego, UNITED STATES, 2006.
    ..math:: z = \frac{T_0}{\\Gamma} \\left[ 1 - \\left( \frac{p}{p_0} \right)^{\frac{R\\Gamma}{g}} \right]
    """
    reference_temperature: float = 288.0  # kelvin
    gamma: float = 0.0065  # 6.5 K/km => 0.0065 K/m
    reference_pressure: float = 1013.25  # hPa
    gas_constant_dry_air: float = 287  # J / (K kg)
    gravitational_acceleration: float = 9.80665  # m / s^2
    return (reference_temperature / gamma) * (
        1 - ((pressure / reference_pressure) ** ((gas_constant_dry_air * gamma) / gravitational_acceleration))
    )


Coordinate = NamedTuple("Coordinate", [("latitude", float), ("longitude", float)])


def bilinear_interpolation(
    longitude: Sequence[float], latitude: Sequence[float], function_value: "NDArray", target_coordinate: Coordinate
) -> float:
    assert len(longitude) == len(latitude)
    assert len(longitude) > 1
    assert function_value.ndim == 2  # noqa: PLR2004

    return RegularGridInterpolator((longitude, latitude), function_value.T, method="linear")(
        (target_coordinate.longitude, target_coordinate.latitude)
    )[0]
