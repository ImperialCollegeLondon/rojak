from typing import NamedTuple, TYPE_CHECKING, Literal

import numpy as np
from pyproj import Geod
from rojak.utilities.types import GoHomeYouAreDrunk

if TYPE_CHECKING:
    from rojak.utilities.types import ArrayLike

GridSpacing = NamedTuple("GridSpacing", [("dx", ArrayLike), ("dy", ArrayLike)])


def is_in_degrees(
    array: ArrayLike,
    coordinate: Literal["latitude", "longitude"] | None = None,
    axis: int | None = None,
) -> bool:
    if coordinate is None or coordinate == "longitude":
        pi_factor = 2
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


# Modified from https://github.com/Unidata/MetPy/blob/b9a9dbd88524e1d9600e353318ee9d9f25b05f57/src/metpy/calc/tools.py#L789
def grid_spacing(
    latitude: ArrayLike, longitude: ArrayLike, geod: Geod | None = None
) -> GridSpacing:
    if geod is None:
        geod = Geod(ellps="WGS84")

    if latitude.ndim != longitude.ndim:
        raise ValueError("latitude and longitude must have same number of dimensions")

    if not is_lat_lon_in_degrees(latitude, longitude):
        latitude = np.rad2deg(latitude)
        longitude = np.rad2deg(longitude)

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
