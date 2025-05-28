import warnings
from typing import NamedTuple, Literal, Tuple

import numpy as np
from pyproj import Geod
from rojak.utilities.types import ArrayLike, GoHomeYouAreDrunk


GridSpacing = NamedTuple("GridSpacing", [("dx", ArrayLike), ("dy", ArrayLike)])


def is_in_degrees(
    array: ArrayLike,
    coordinate: Literal["latitude", "longitude"] | None = None,
    axis: int | None = None,
) -> bool:
    """
    Checks if array could be in degrees based whether it exceeds the maximum radian value for the specified coordinate.

    >>> np.set_printoptions(legacy="1.25")
    >>> is_in_degrees(np.asarray([0, 180, 360, -180]), coordinate="longitude")
    True
    >>> is_in_degrees(np.asarray([-90, 0, 90]), coordinate="latitude")
    True
    >>> is_in_degrees(np.asarray([0, 2 *np.pi]))
    False
    >>> is_in_degrees(np.asarray([0, np.pi / 4]), coordinate="latitude")
    False
    >>> is_in_degrees(np.asarray([np.pi/2 + 0.01]), coordinate="latitude")
    True
    >>> is_in_degrees(np.asarray([0]), coordinate="longitude")
    False
    """
    if coordinate is None:
        pi_factor = 2
    elif coordinate == "longitude":
        pi_factor = 1
    else:  # Latitude
        pi_factor = 1 / 2
    return np.any(
        (array > (pi_factor * np.pi)) | (array < (-pi_factor * np.pi)), axis=axis
    )


def is_lat_lon_in_degrees(
    latitude: ArrayLike,
    longitude: ArrayLike,
) -> bool:
    is_lat_in_degrees: bool = is_in_degrees(latitude, coordinate="latitude")
    is_lon_in_degrees: bool = is_in_degrees(longitude, coordinate="longitude")

    if is_lat_in_degrees and not is_lon_in_degrees:
        raise ValueError("Latitude is in degrees, but longitude is not")
    elif not is_lat_in_degrees and is_lon_in_degrees:
        raise ValueError("Longitude is in degrees, but latitude is not")
    else:
        # Both should be true or false
        return is_lat_in_degrees


type LatLonUnits = Literal["deg", "rad"]


def check_lat_lon_units(
    latitude: "ArrayLike", longitude: "ArrayLike", units: LatLonUnits
) -> Tuple["ArrayLike", "ArrayLike"]:
    are_in_degrees: bool = is_lat_lon_in_degrees(latitude, longitude)
    if units == "deg" and not are_in_degrees:
        warnings.warn(
            "Latitude and longitude specified to be in degrees, but are smaller than pi values"
        )
    elif units == "rad" and are_in_degrees:
        raise ValueError(
            "Latitude and longitude specified to be in radians, but are too large to be in radians"
        )
    elif units == "rad" and not are_in_degrees:
        latitude = np.rad2deg(latitude)
        longitude = np.rad2deg(longitude)

    return latitude, longitude


# Modified from https://github.com/Unidata/MetPy/blob/b9a9dbd88524e1d9600e353318ee9d9f25b05f57/src/metpy/calc/tools.py#L789
def grid_spacing(
    latitude: ArrayLike,
    longitude: ArrayLike,
    units: LatLonUnits,
    geod: Geod | None = None,
) -> GridSpacing:
    if geod is None:
        geod = Geod(ellps="WGS84")

    if latitude.ndim != longitude.ndim:
        raise ValueError("latitude and longitude must have same number of dimensions")

    latitude, longitude = check_lat_lon_units(latitude, longitude, units)

    lat_grid: ArrayLike
    lon_grid: ArrayLike
    if latitude.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    elif latitude.ndim == 2:
        # lat_grid = latitude
        # lon_grid = longitude
        raise NotImplementedError(
            "Function doesn't support 2D latitude and longitude inputs"
        )
    else:
        raise GoHomeYouAreDrunk(
            "What are you doing? How do lat and lon have >2 dimensions?"
        )

    forward_azimuth, _, dy = geod.inv(
        lon_grid[:-1, :], lat_grid[:-1, :], lon_grid[1:, :], lat_grid[1:, :]
    )
    # I don't understand why this lines is here... Copied from metpy
    dy[(forward_azimuth < -90.0) | (forward_azimuth > 90.0)] *= -1
    forward_azimuth, _, dx = geod.inv(
        lon_grid[:, :-1], lat_grid[:, :-1], lon_grid[:, 1:], lat_grid[:, 1:]
    )
    dx[(forward_azimuth < 0.0) | (forward_azimuth > 180.0)] *= -1

    return GridSpacing(dx, dy)


# Modified from: https://github.com/Unidata/MetPy/blob/6df0cde7893c0f55e44946137263cb322d59aae4/src/metpy/calc/tools.py#L868
def nominal_grid_spacing(
    latitude: ArrayLike,
    longitude: ArrayLike,
    units: LatLonUnits,
    geod: Geod | None = None,
) -> GridSpacing:
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Latitude and longitude must have 1 dimension")
    if geod is None:
        # In metpy, geod = CRS('+proj=latlon').get_geod()
        geod = Geod(ellps="WGS84")

    latitude, longitude = check_lat_lon_units(latitude, longitude, units)

    lat_equator = np.zeros_like(longitude)
    _, _, dx = geod.inv(
        longitude[:-1], lat_equator[:-1], longitude[1:], lat_equator[1:]
    )
    lon_meridian = np.zeros_like(latitude)
    forward_azimuth, _, dy = geod.inv(
        lon_meridian[:-1], latitude[:-1], lon_meridian[1:], lat_equator[1:]
    )
    dy[(forward_azimuth < -90.0) | (forward_azimuth > 90.0)] *= -1

    return GridSpacing(dx, dy)
