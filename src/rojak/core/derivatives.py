import warnings
from enum import StrEnum, Enum, auto
from typing import NamedTuple, Literal, Tuple

import dask.array as da
import xarray as xr
from dask.base import is_dask_collection
import numpy as np
from pyproj import Geod, CRS, Proj
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
    if coordinate is None or coordinate == "longitude":
        positive_factor, negative_factor = 2, 2
    elif coordinate == "longitude":
        positive_factor, negative_factor = 2, 1
    else:  # Latitude
        positive_factor, negative_factor = 1 / 2, 1 / 2
    return np.any(
        (array > (positive_factor * np.pi)) | (array < (-negative_factor * np.pi)),
        axis=axis,
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


def ensure_lat_lon_in_deg(
    latitude: "ArrayLike", longitude: "ArrayLike", units: LatLonUnits
) -> Tuple["ArrayLike", "ArrayLike"]:
    """
    >>> ensure_lat_lon_in_deg(np.asarray([90, 0, -90]), np.asarray([360, 180, 0]), "deg")
    (array([ 90,   0, -90]), array([360, 180,   0]))
    >>> ensure_lat_lon_in_deg(np.asarray([np.pi/2, 0, -np.pi/2]), np.asarray([2*np.pi, np.pi, 0]), "rad")
    (array([ 90.,   0., -90.]), array([360., 180.,   0.]))
    """
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


# TODO: TEST
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

    latitude, longitude = ensure_lat_lon_in_deg(latitude, longitude, units)

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

    latitude, longitude = ensure_lat_lon_in_deg(latitude, longitude, units)

    lat_equator = np.zeros_like(longitude)
    _, _, dx = geod.inv(
        longitude[:-1], lat_equator[:-1], longitude[1:], lat_equator[1:]
    )
    lon_meridian = np.zeros_like(latitude)
    forward_azimuth, _, dy = geod.inv(
        lon_meridian[:-1], latitude[:-1], lon_meridian[1:], latitude[1:]
    )
    dy[(forward_azimuth < -90.0) | (forward_azimuth > 90.0)] *= -1

    return GridSpacing(dx, dy)


class ProjectionCorrectionFactors(NamedTuple):
    # add | float as type checker thinks it should be a float ¯\_(ツ)_/¯
    parallel_scale: xr.DataArray
    meridional_scale: xr.DataArray


# TODO: TEST
# Modified from https://github.com/Unidata/MetPy/blob/6df0cde7893c0f55e44946137263cb322d59aae4/src/metpy/calc/tools.py#L1124
def get_projection_correction_factors(
    latitude: "xr.DataArray",
    longitude: "xr.DataArray",
    is_radians=False,
    crs: CRS | None = None,
) -> ProjectionCorrectionFactors:
    if latitude.ndim != longitude.ndim:
        raise ValueError("Latitude and longitude must have same number of dimensions")
    if latitude.ndim != 1:
        raise ValueError("Latitude and longitude must have 1 dimension")

    if crs is None:
        crs = CRS("+proj=latlon")

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    factors = Proj(crs).get_factors(lon_grid, lat_grid, radians=is_radians)

    return ProjectionCorrectionFactors(factors.parallel_scale, factors.meridional_scale)  # type: ignore


def get_dimension_number(name: str, data_array: "xr.DataArray") -> int:
    if name not in data_array.dims:
        raise ValueError(
            f"Attempting to retrieve inexistent dimension ({name}) from data array"
        )
    return data_array.dims.index(name)


# TODO: TEST
def first_derivative(
    array: "xr.DataArray", grid_spacing_in_meters: ArrayLike, axis: int
) -> "xr.DataArray":
    coordinate_of_values: np.ndarray = np.cumsum(
        np.insert(grid_spacing_in_meters, 0, [0])
    )
    if is_dask_collection(array):
        computed_gradient = da.gradient(array, coordinate_of_values, axis=axis)
    else:
        computed_gradient = np.gradient(array, coordinate_of_values, axis=axis)
    return computed_gradient


class CartesianDimension(StrEnum):
    X = "x"
    Y = "y"

    def get_geographic_coordinate(self) -> str | None:
        match self:
            case CartesianDimension.X:
                return "longitude"
            case CartesianDimension.Y:
                return "latitude"
        return None

    def get_grid_spacing(self, grid_deltas: GridSpacing) -> ArrayLike:
        match self:
            case CartesianDimension.X:
                grid_delta = grid_deltas.dx
            case CartesianDimension.Y:
                grid_delta = grid_deltas.dy
        return grid_delta

    def get_correction_factor(
        self, factors: ProjectionCorrectionFactors | None
    ) -> xr.DataArray:
        if factors is None:
            raise ValueError("Factors cannot be None")

        match self:
            case CartesianDimension.X:
                factor = factors.parallel_scale
            case CartesianDimension.Y:
                factor = factors.meridional_scale
        return factor


class GradientMode(Enum):
    GEOSPATIAL = auto()
    CARTESIAN = auto()


SpatialGradient = NamedTuple(
    "SpatialGradient", [("dfdx", xr.DataArray | None), ("dfdy", xr.DataArray | None)]
)


def check_lat_lon_dimensions_in_array(array: "xr.DataArray") -> None:
    if "longitude" not in array.dims:
        raise ValueError(f"Longitude not in dimension of array - {array.dims}")
    if "latitude" not in array.dims:
        raise ValueError(f"Latitude not in dimension of array - {array.dims}")


type SpatialGradientKeys = Literal["dfdx", "dfdy"]


# TODO: TEST
# Combines implementation from metpy and the existing derivatives methods in prototype lib
def spatial_gradient(
    array: "xr.DataArray",
    units: LatLonUnits,
    gradient_mode: GradientMode,
    dimension: CartesianDimension | None = None,
    geod: Geod | None = None,
    crs: CRS | None = None,
) -> dict[SpatialGradientKeys, xr.DataArray]:
    check_lat_lon_dimensions_in_array(array)

    gradients: dict[SpatialGradientKeys, xr.DataArray] = {}
    grid_deltas = nominal_grid_spacing(
        array["latitude"], array["longitude"], units, geod=geod
    )
    if gradient_mode == GradientMode.GEOSPATIAL:
        correction_factors = get_projection_correction_factors(
            array["latitude"], array["longitude"], is_radians=(units == "rad"), crs=crs
        )
    else:
        correction_factors = None

    target_dimensions: list[CartesianDimension] = (
        [dimension]
        if dimension is not None
        else [CartesianDimension.X, CartesianDimension.Y]
    )
    for dim in target_dimensions:
        dim_name: str | None = dim.get_geographic_coordinate()
        assert dim_name is not None
        axis: int = get_dimension_number(dim_name, array)
        grid_delta = dim.get_grid_spacing(grid_deltas)
        computed_gradient: "xr.DataArray" = first_derivative(array, grid_delta, axis)
        if gradient_mode == GradientMode.GEOSPATIAL:
            correction = dim.get_correction_factor(correction_factors)
            computed_gradient = computed_gradient * correction
        match dim:
            case CartesianDimension.X:
                gradients["dfdx"] = computed_gradient
            case CartesianDimension.Y:
                gradients["dfdy"] = computed_gradient

    return gradients


def divergence(du_dx: ArrayLike, dv_dy: ArrayLike) -> ArrayLike:
    return du_dx + dv_dy


# TODO: TEST
def spatial_laplacian(
    array: "xr.DataArray",
    units: LatLonUnits,
    gradient_mode: GradientMode,
    geod: Geod | None = None,
    crs: CRS | None = None,
) -> ArrayLike:
    gradients = spatial_gradient(array, units, gradient_mode, geod=geod, crs=crs)
    return divergence(gradients["dfdx"], gradients["dfdy"])


class VelocityDerivative(StrEnum):
    DU_DX = "du_dx"
    DU_DY = "du_dy"
    DV_DX = "dv_dx"
    DV_DY = "dv_dy"


# TODO: TEST
def vector_derivatives(
    u: xr.DataArray,
    v: xr.DataArray,
    units: LatLonUnits,
    components: list[VelocityDerivative] | None = None,
    geod: Geod | None = None,
    crs: CRS | None = None,
) -> dict[VelocityDerivative, xr.DataArray]:
    check_lat_lon_dimensions_in_array(u)
    check_lat_lon_dimensions_in_array(v)

    if components is None:
        components = [
            VelocityDerivative.DU_DX,
            VelocityDerivative.DU_DY,
            VelocityDerivative.DV_DX,
            VelocityDerivative.DV_DY,
        ]

    correction_factors = get_projection_correction_factors(
        u["latitude"], u["longitude"], is_radians=(units == "rad"), crs=crs
    )

    dp_dy: xr.DataArray = spatial_gradient(
        correction_factors.parallel_scale,
        units,
        GradientMode.CARTESIAN,
        dimension=CartesianDimension.Y,
        geod=geod,
        crs=crs,
    )["dfdy"]
    dm_dx: xr.DataArray = spatial_gradient(
        correction_factors.meridional_scale,
        units,
        GradientMode.CARTESIAN,
        dimension=CartesianDimension.X,
        geod=geod,
        crs=crs,
    )["dfdx"]
    dx_correction: xr.DataArray = (
        correction_factors.meridional_scale / correction_factors.parallel_scale
    ) * dp_dy
    dy_correction: xr.DataArray = (
        correction_factors.parallel_scale / correction_factors.meridional_scale
    ) * dm_dx

    derivatives: dict[VelocityDerivative, xr.DataArray] = {}
    for component in components:
        match component:
            case VelocityDerivative.DU_DX:
                derivatives[VelocityDerivative.DU_DX] = (
                    correction_factors.parallel_scale
                    * spatial_gradient(
                        u,
                        units,
                        GradientMode.CARTESIAN,
                        dimension=CartesianDimension.X,
                        geod=geod,
                        crs=crs,
                    )["dfdx"]
                    - v * dx_correction
                )
            case VelocityDerivative.DU_DY:
                derivatives[VelocityDerivative.DU_DY] = (
                    correction_factors.meridional_scale
                    * spatial_gradient(
                        u,
                        units,
                        GradientMode.CARTESIAN,
                        dimension=CartesianDimension.Y,
                        geod=geod,
                        crs=crs,
                    )["dfdy"]
                    + v * dy_correction
                )
            case VelocityDerivative.DV_DX:
                derivatives[VelocityDerivative.DV_DX] = (
                    correction_factors.parallel_scale
                    * spatial_gradient(
                        v,
                        units,
                        GradientMode.CARTESIAN,
                        dimension=CartesianDimension.X,
                        geod=geod,
                        crs=crs,
                    )["dfdx"]
                    + u * dx_correction
                )
            case VelocityDerivative.DV_DY:
                derivatives[VelocityDerivative.DV_DY] = (
                    correction_factors.meridional_scale
                    * spatial_gradient(
                        v,
                        units,
                        GradientMode.CARTESIAN,
                        dimension=CartesianDimension.Y,
                        geod=geod,
                        crs=crs,
                    )["dfdy"]
                    - u * dy_correction
                )

    return derivatives
