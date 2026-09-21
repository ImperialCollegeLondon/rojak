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
"""
Coordinate-aware spatial derivatives on a latitude/longitude grid

This module computes gradients, divergence, and the Laplacian of fields defined on a geographic (latitude/
longitude) grid, accounting for the physical (geodesic) distance between grid points and, where relevant, the
distortion introduced by a map projection. It underlies the derivative-based turbulence diagnostics in
:mod:`rojak.turbulence.calculations` and :mod:`rojak.turbulence.diagnostic`.

Grid spacing between adjacent points can be computed either from the true geodesic distance at every grid point
(:func:`grid_spacing`) or, more cheaply, from a nominal spacing measured along the equator and prime meridian
(:func:`nominal_grid_spacing`). :func:`get_projection_correction_factors` computes the map projection scale factors
used to correct raw Cartesian derivatives (:func:`spatial_gradient`, in :attr:`GradientMode.GEOSPATIAL` mode) for
projection distortion, and :func:`vector_derivatives` computes the full set of horizontal velocity derivatives
(du/dx, du/dy, dv/dx, dv/dy) with this correction applied. :func:`divergence` and :func:`spatial_laplacian` build on
:func:`spatial_gradient`.
"""

import warnings
from enum import Enum, StrEnum, auto
from typing import Literal, NamedTuple, assert_never

import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from pyproj import CRS, Geod, Proj

from rojak.core.constants import MAX_LATITUDE, MAX_LONGITUDE
from rojak.utilities.types import GoHomeYouAreDrunkError, NumpyOrDataArray


class GridSpacing(NamedTuple):
    """Grid spacing (in meters) along the x and y Cartesian directions, e.g. as computed by :func:`grid_spacing`"""

    dx: NumpyOrDataArray
    dy: NumpyOrDataArray


def _is_in_degrees(
    array: NumpyOrDataArray,
    coordinate: Literal["latitude", "longitude"] | None = None,
    axis: int | None = None,
) -> bool:
    """
    Checks if array could be in degrees based whether it exceeds the maximum radian value for the specified coordinate.

    >>> np.set_printoptions(legacy="1.25")
    >>> _is_in_degrees(np.asarray([0, 180, 360, -180]), coordinate="longitude")
    True
    >>> _is_in_degrees(np.asarray([-90, 0, 90]), coordinate="latitude")
    True
    >>> _is_in_degrees(np.asarray([0, 2 *np.pi]))
    False
    >>> _is_in_degrees(np.asarray([0, np.pi / 4]), coordinate="latitude")
    False
    >>> _is_in_degrees(np.asarray([np.pi/2 + 0.01]), coordinate="latitude")
    True
    >>> _is_in_degrees(np.asarray([0]), coordinate="longitude")
    False
    """
    if coordinate is None or coordinate == "longitude":
        positive_factor, negative_factor = 2, 2
    elif coordinate == "longitude":
        positive_factor, negative_factor = 2, 1
    else:  # Latitude
        positive_factor, negative_factor = 1 / 2, 1 / 2
    return bool(
        np.any(
            (array > (positive_factor * np.pi)) | (array < (-negative_factor * np.pi)),
            axis=axis,
        )
    )


def _is_lat_lon_in_degrees(latitude: NumpyOrDataArray, longitude: NumpyOrDataArray) -> bool:
    """
    Check that latitude and longitude are consistently either both in degrees or both in radians

    Args:
        latitude: Array of latitude values
        longitude: Array of longitude values

    Returns:
        ``True`` if both are in degrees, ``False`` if both are in radians

    Raises:
        ValueError: If only one of ``latitude``/``longitude`` appears to be in degrees
    """
    is_lat_in_degrees: bool = _is_in_degrees(latitude, coordinate="latitude")
    is_lon_in_degrees: bool = _is_in_degrees(longitude, coordinate="longitude")

    if is_lat_in_degrees and not is_lon_in_degrees:
        raise ValueError("Latitude is in degrees, but longitude is not")
    if not is_lat_in_degrees and is_lon_in_degrees:
        raise ValueError("Longitude is in degrees, but latitude is not")
    # Both should be true or false
    return is_lat_in_degrees


# type LatLonUnits = Literal["deg", "rad"]


class LatLonUnits(StrEnum):
    """Units latitude/longitude coordinates are expressed in"""

    DEG = "deg"
    RAD = "rad"


def _ensure_lat_lon_in_deg(
    latitude: "NumpyOrDataArray",
    longitude: "NumpyOrDataArray",
    units: LatLonUnits,
) -> tuple["NumpyOrDataArray", "NumpyOrDataArray"]:
    """
    >>> _ensure_lat_lon_in_deg(np.asarray([90, 0, -90]), np.asarray([360, 180, 0]), "deg")
    (array([ 90,   0, -90]), array([360, 180,   0]))
    >>> _ensure_lat_lon_in_deg(np.asarray([np.pi/2, 0, -np.pi/2]), np.asarray([2*np.pi, np.pi, 0]), "rad")
    (array([ 90.,   0., -90.]), array([360., 180.,   0.]))
    """
    are_in_degrees: bool = _is_lat_lon_in_degrees(latitude, longitude)
    if units == "deg" and not are_in_degrees:
        warnings.warn("Latitude and longitude specified to be in degrees, but are smaller than pi values", stacklevel=2)
    elif units == "rad" and are_in_degrees:
        raise ValueError("Latitude and longitude specified to be in radians, but are too large to be in radians")
    elif units == "rad" and not are_in_degrees:
        latitude = np.rad2deg(latitude)
        longitude = np.rad2deg(longitude)

    return latitude, longitude


# TODO: TEST
# Modified from https://github.com/Unidata/MetPy/blob/b9a9dbd88524e1d9600e353318ee9d9f25b05f57/src/metpy/calc/tools.py#L789
def grid_spacing(
    latitude: NumpyOrDataArray,
    longitude: NumpyOrDataArray,
    units: LatLonUnits,
    geod: Geod | None = None,
) -> GridSpacing:
    """
    Compute the geodesic distance between adjacent points of a 2D latitude/longitude grid

    Unlike :func:`nominal_grid_spacing`, the distance is computed at every grid point (rather than approximated
    from a single row/column), so this accounts for the grid spacing varying with latitude.

    Args:
        latitude: 1D array of latitude coordinates
        longitude: 1D array of longitude coordinates
        units: Units ``latitude``/``longitude`` are expressed in
        geod: Geodesic to compute distances with. Defaults to a WGS84 ellipsoid.

    Returns:
        Grid spacing (in meters) between adjacent grid points along each Cartesian direction

    Raises:
        ValueError: If ``latitude`` and ``longitude`` do not have the same number of dimensions
        NotImplementedError: If ``latitude``/``longitude`` are 2D (not yet supported)
        GoHomeYouAreDrunkError: If ``latitude``/``longitude`` have more than 2 dimensions
    """
    if geod is None:
        geod = Geod(ellps="WGS84")

    if latitude.ndim != longitude.ndim:
        raise ValueError("latitude and longitude must have same number of dimensions")

    latitude, longitude = _ensure_lat_lon_in_deg(latitude, longitude, units)

    lat_grid: NumpyOrDataArray
    lon_grid: NumpyOrDataArray
    if latitude.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    elif latitude.ndim == 2:  # noqa: PLR2004
        # lat_grid = latitude
        # lon_grid = longitude
        raise NotImplementedError("Function doesn't support 2D latitude and longitude inputs")
    else:
        raise GoHomeYouAreDrunkError("What are you doing? How do lat and lon have >2 dimensions?")

    forward_azimuth, _, dy = geod.inv(lon_grid[:-1, :], lat_grid[:-1, :], lon_grid[1:, :], lat_grid[1:, :])
    # I don't understand why this lines is here... Copied from metpy
    dy[(forward_azimuth < -MAX_LATITUDE) | (forward_azimuth > MAX_LATITUDE)] *= -1
    forward_azimuth, _, dx = geod.inv(lon_grid[:, :-1], lat_grid[:, :-1], lon_grid[:, 1:], lat_grid[:, 1:])
    dx[(forward_azimuth < 0.0) | (forward_azimuth > MAX_LONGITUDE)] *= -1

    return GridSpacing(dx, dy)


# Modified from: https://github.com/Unidata/MetPy/blob/6df0cde7893c0f55e44946137263cb322d59aae4/src/metpy/calc/tools.py#L868
def nominal_grid_spacing(
    latitude: NumpyOrDataArray,
    longitude: NumpyOrDataArray,
    units: LatLonUnits,
    geod: Geod | None = None,
) -> GridSpacing:
    """
    Estimate the grid spacing of a regular latitude/longitude grid from a single row and column

    Unlike :func:`grid_spacing`, this does not compute the distance at every grid point: ``dx`` is the geodesic
    distance between adjacent longitude values along the equator, and ``dy`` is the geodesic distance between
    adjacent latitude values along the prime meridian. This is cheaper than :func:`grid_spacing` but only an
    approximation away from the equator/meridian.

    Args:
        latitude: 1D array of latitude coordinates
        longitude: 1D array of longitude coordinates
        units: Units ``latitude``/``longitude`` are expressed in
        geod: Geodesic to compute distances with. Defaults to a WGS84 ellipsoid.

    Returns:
        Nominal grid spacing (in meters) along each Cartesian direction

    Raises:
        ValueError: If ``latitude`` or ``longitude`` is not 1D
    """
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Latitude and longitude must have 1 dimension")
    if geod is None:
        # In metpy, geod = CRS('+proj=latlon').get_geod()
        geod = Geod(ellps="WGS84")

    latitude, longitude = _ensure_lat_lon_in_deg(latitude, longitude, units)

    lat_equator = np.zeros_like(longitude)
    _, _, dx = geod.inv(longitude[:-1], lat_equator[:-1], longitude[1:], lat_equator[1:])
    lon_meridian = np.zeros_like(latitude)
    forward_azimuth, _, dy = geod.inv(lon_meridian[:-1], latitude[:-1], lon_meridian[1:], latitude[1:])
    dy[(forward_azimuth < -MAX_LATITUDE) | (forward_azimuth > MAX_LATITUDE)] *= -1

    return GridSpacing(dx, dy)


class ProjectionCorrectionFactors(NamedTuple):
    """
    Map projection scale factors, as computed by :func:`get_projection_correction_factors`

    These scale a distance measured on the map projection to the corresponding true distance on the ground, along
    the parallels (``parallel_scale``, i.e. the x/longitude direction) and meridians (``meridional_scale``, i.e.
    the y/latitude direction).
    """

    # add | float as type checker thinks it should be a float ¯\_(ツ)_/¯
    parallel_scale: xr.DataArray
    meridional_scale: xr.DataArray


# Heavily modified from https://github.com/Unidata/MetPy/blob/6df0cde7893c0f55e44946137263cb322d59aae4/src/metpy/calc/tools.py#L1124
def get_projection_correction_factors(
    latitude: "xr.DataArray",
    longitude: "xr.DataArray",
    use_dask: bool,
    is_radians: bool = False,
    crs: CRS | None = None,
) -> ProjectionCorrectionFactors:
    """
    Compute the map projection scale factors at each point of a latitude/longitude grid

    These factors (see :class:`ProjectionCorrectionFactors`) are used to correct derivatives computed on the raw
    latitude/longitude grid (see :attr:`GradientMode.GEOSPATIAL` in :func:`spatial_gradient`, and
    :func:`vector_derivatives`) for the distortion introduced by the map projection.

    Args:
        latitude: 1D array of latitude coordinates
        longitude: 1D array of longitude coordinates
        use_dask: If ``True``, compute the factors lazily with dask
        is_radians: Whether ``latitude``/``longitude`` are in radians. Defaults to ``False`` (degrees).
        crs: Coordinate reference system to compute the scale factors for. Defaults to plain latitude/longitude
            (``"+proj=latlon"``).

    Returns:
        Scale factors, with dimensions matching ``latitude`` and ``longitude``

    Raises:
        ValueError: If ``latitude`` and ``longitude`` do not both have exactly 1 dimension
    """
    if latitude.ndim != longitude.ndim:
        raise ValueError("Latitude and longitude must have same number of dimensions")
    if latitude.ndim != 1:
        raise ValueError("Latitude and longitude must have 1 dimension")

    if crs is None:
        crs = CRS("+proj=latlon")

    if use_dask:
        # pyright is drunk. It thinks that the return type is NoReturn so it is not iterable... ¯\_(ツ)_/¯
        lon_grid, lat_grid = da.meshgrid(longitude, latitude)  # pyright: ignore[reportGeneralTypeIssues]
        parallel_scale = da.map_blocks(
            lambda lon, lat: Proj(crs).get_factors(lon, lat, radians=is_radians).parallel_scale,
            lon_grid,
            lat_grid,
        ).persist()
        meridional_scale = da.map_blocks(
            lambda lon, lat: Proj(crs).get_factors(lon, lat, radians=is_radians).meridional_scale,
            lon_grid,
            lat_grid,
        ).persist()
    else:
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)
        factors = Proj(crs).get_factors(lon_grid, lat_grid, radians=is_radians)
        parallel_scale = factors.parallel_scale
        meridional_scale = factors.meridional_scale

    return ProjectionCorrectionFactors(
        xr.DataArray(
            parallel_scale,
            dims=(latitude.dims[0], longitude.dims[-1]),
            coords={**latitude.coords, **longitude.coords},
            # coords={"longitude": longitude, "latitude": latitude},
        ),
        xr.DataArray(
            meridional_scale,
            dims=(latitude.dims[0], longitude.dims[-1]),
            coords={**latitude.coords, **longitude.coords},
            # coords={"longitude": longitude, "latitude": latitude},
        ),
    )


def get_dimension_number(name: str, data_array: "xr.DataArray") -> int:
    """
    Get the axis position of a named dimension in a DataArray

    Args:
        name: Name of the dimension to look up
        data_array: Array to look up the dimension's position in

    Returns:
        Position of ``name`` in ``data_array.dims``

    Raises:
        ValueError: If ``name`` is not a dimension of ``data_array``
    """
    if name not in data_array.dims:
        raise ValueError(f"Attempting to retrieve inexistent dimension ({name}) from data array")
    return data_array.dims.index(name)


def first_derivative(array: "xr.DataArray", grid_spacing_in_meters: NumpyOrDataArray, axis: int) -> "xr.DataArray":
    """
    First derivative of ``array`` along ``axis``, with respect to physical distance

    The (possibly irregular) grid spacing along ``axis`` is used to build the coordinate values that the central
    difference derivative (:func:`numpy.gradient`/:func:`dask.array.gradient`) is taken with respect to, so the
    result is a derivative with respect to true distance (in meters) rather than grid index.

    Args:
        array: Array to differentiate
        grid_spacing_in_meters: Spacing (in meters) between adjacent points along ``axis``, of length
            ``array.shape[axis] - 1``
        axis: Axis of ``array`` to differentiate along

    Returns:
        Derivative of ``array`` along ``axis``, with the same shape as ``array``
    """
    coordinate_of_values: np.ndarray = np.cumsum(np.insert(grid_spacing_in_meters, 0, [0]))
    if is_dask_collection(array):
        computed_gradient = da.gradient(array, coordinate_of_values, axis=axis)
    else:
        computed_gradient = np.gradient(array, coordinate_of_values, axis=axis)
    return array.copy(data=computed_gradient)


class CartesianDimension(StrEnum):
    """Cartesian x/y direction, and the corresponding geographic coordinate/grid spacing/scale factor"""

    X = "x"
    Y = "y"

    def get_geographic_coordinate(self) -> str | None:
        """The geographic coordinate name corresponding to this Cartesian dimension (longitude for X, latitude for Y)"""
        match self:
            case CartesianDimension.X:
                return "longitude"
            case CartesianDimension.Y:
                return "latitude"
            case _ as unreachable:
                assert_never(unreachable)
        return None

    def get_grid_spacing(self, grid_deltas: GridSpacing) -> NumpyOrDataArray:
        """
        Select this dimension's component (``dx`` for X, ``dy`` for Y) from a :class:`GridSpacing`

        Args:
            grid_deltas: Grid spacing to select from

        Returns:
            ``grid_deltas.dx`` if this is :attr:`X`, or ``grid_deltas.dy`` if this is :attr:`Y`
        """
        match self:
            case CartesianDimension.X:
                grid_delta = grid_deltas.dx
            case CartesianDimension.Y:
                grid_delta = grid_deltas.dy
            case _ as unreachable:
                assert_never(unreachable)
        return grid_delta

    def get_correction_factor(self, factors: ProjectionCorrectionFactors | None) -> xr.DataArray:
        """
        Select this dimension's component (``parallel_scale`` for X, ``meridional_scale`` for Y) from
        :class:`ProjectionCorrectionFactors`

        Args:
            factors: Projection correction factors to select from

        Returns:
            ``factors.parallel_scale`` if this is :attr:`X`, or ``factors.meridional_scale`` if this is :attr:`Y`

        Raises:
            ValueError: If ``factors`` is ``None``
        """
        if factors is None:
            raise ValueError("Factors cannot be None")

        match self:
            case CartesianDimension.X:
                factor = factors.parallel_scale
            case CartesianDimension.Y:
                factor = factors.meridional_scale
            case _ as unreachable:
                assert_never(unreachable)
        return factor


class GradientMode(Enum):
    """
    How :func:`spatial_gradient` should compute the gradient

    ``GEOSPATIAL`` corrects the raw Cartesian gradient by the map projection scale factors (see
    :func:`get_projection_correction_factors`) so that it is with respect to true geographic distance.
    ``CARTESIAN`` returns the raw, uncorrected gradient with respect to the (nominal) grid spacing.
    """

    GEOSPATIAL = auto()
    CARTESIAN = auto()


class SpatialGradient(NamedTuple):
    """x and y components of a spatial gradient, as computed by :func:`spatial_gradient`"""

    dfdx: xr.DataArray | None
    dfdy: xr.DataArray | None


def _check_lat_lon_dimensions_in_array(array: "xr.DataArray") -> None:
    """
    Check that ``array`` has ``"longitude"`` and ``"latitude"`` dimensions

    Args:
        array: Array to check

    Raises:
        ValueError: If ``array`` is missing either dimension
    """
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
    """
    Spatial gradient of a scalar field on a latitude/longitude grid

    Args:
        array: Field to compute the gradient of. Must have ``"longitude"`` and ``"latitude"`` dimensions.
        units: Units ``array``'s latitude/longitude coordinates are expressed in
        gradient_mode: Whether to correct for map projection distortion (:attr:`GradientMode.GEOSPATIAL`) or not
            (:attr:`GradientMode.CARTESIAN`)
        dimension: If provided, only compute the gradient along this dimension. Defaults to both x and y.
        geod: Geodesic used to compute the nominal grid spacing (see :func:`nominal_grid_spacing`). Defaults to a
            WGS84 ellipsoid.
        crs: Coordinate reference system used to compute projection correction factors when
            ``gradient_mode`` is :attr:`GradientMode.GEOSPATIAL`. Defaults to plain latitude/longitude.

    Returns:
        Mapping with ``"dfdx"`` and/or ``"dfdy"`` keys (depending on ``dimension``) to the corresponding component
        of the gradient
    """
    _check_lat_lon_dimensions_in_array(array)

    gradients: dict[SpatialGradientKeys, xr.DataArray] = {}
    grid_deltas = nominal_grid_spacing(array["latitude"], array["longitude"], units, geod=geod)
    if gradient_mode == GradientMode.GEOSPATIAL:
        correction_factors = get_projection_correction_factors(
            array["latitude"],
            array["longitude"],
            is_dask_collection(array),
            is_radians=(units == "rad"),
            crs=crs,
        )
    else:
        correction_factors = None

    target_dimensions: list[CartesianDimension] = (
        [dimension] if dimension is not None else [CartesianDimension.X, CartesianDimension.Y]
    )
    for dim in target_dimensions:
        dim_name: str | None = dim.get_geographic_coordinate()
        assert dim_name is not None
        axis: int = get_dimension_number(dim_name, array)
        grid_delta = dim.get_grid_spacing(grid_deltas)
        computed_gradient: xr.DataArray = first_derivative(array, grid_delta, axis)
        if gradient_mode == GradientMode.GEOSPATIAL:
            correction = dim.get_correction_factor(correction_factors)
            computed_gradient = computed_gradient * correction
        match dim:
            case CartesianDimension.X:
                gradients["dfdx"] = computed_gradient
            case CartesianDimension.Y:
                gradients["dfdy"] = computed_gradient
            case _ as unreachable:
                assert_never(unreachable)

    return gradients


def divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    units: LatLonUnits,
    geod: Geod | None = None,
    crs: CRS | None = None,
) -> xr.DataArray:
    """
    Horizontal divergence of a 2D vector field

    .. math:: \\delta = \\frac{ \\partial u }{ \\partial x } + \\frac{ \\partial v }{ \\partial y }

    Computed using :func:`vector_derivatives` so that the derivatives are corrected for map projection distortion.

    Args:
        u: x-component of the vector field
        v: y-component of the vector field
        units: Units ``u``/``v``'s latitude/longitude coordinates are expressed in
        geod: Geodesic used to compute the nominal grid spacing. Defaults to a WGS84 ellipsoid.
        crs: Coordinate reference system used to compute projection correction factors. Defaults to plain
            latitude/longitude.

    Returns:
        Horizontal divergence of ``u`` and ``v``
    """
    gradients = vector_derivatives(
        u, v, units, [VelocityDerivative.DU_DX, VelocityDerivative.DV_DY], geod=geod, crs=crs
    )
    return gradients[VelocityDerivative.DU_DX] + gradients[VelocityDerivative.DV_DY]


def spatial_laplacian(
    array: "xr.DataArray",
    units: LatLonUnits,
    gradient_mode: GradientMode,
    geod: Geod | None = None,
    crs: CRS | None = None,
) -> xr.DataArray:
    """
    Laplacian of a scalar field on a latitude/longitude grid

    .. math::
        \\nabla^{2} f = \\frac{ \\partial^{2} f }{ \\partial x^{2} } + \\frac{ \\partial^{2} f }{ \\partial y^{2} }

    Computed as the divergence (see :func:`divergence`) of the field's gradient (see :func:`spatial_gradient`).

    Args:
        array: Field to compute the Laplacian of. Must have ``"longitude"`` and ``"latitude"`` dimensions.
        units: Units ``array``'s latitude/longitude coordinates are expressed in
        gradient_mode: Whether to correct the gradient for map projection distortion, see :func:`spatial_gradient`
        geod: Geodesic used to compute the nominal grid spacing. Defaults to a WGS84 ellipsoid.
        crs: Coordinate reference system used to compute projection correction factors. Defaults to plain
            latitude/longitude.

    Returns:
        Laplacian of ``array``
    """
    gradients = spatial_gradient(array, units, gradient_mode, geod=geod, crs=crs)
    return divergence(gradients["dfdx"], gradients["dfdy"], units=units, geod=geod, crs=crs)


class VelocityDerivative(StrEnum):
    """The four horizontal derivatives of a 2D velocity field, as computed by :func:`vector_derivatives`"""

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
    """
    Horizontal derivatives of a 2D velocity field, corrected for map projection distortion

    Rather than directly correcting each raw Cartesian derivative by the corresponding
    :class:`ProjectionCorrectionFactors` component (as :attr:`GradientMode.GEOSPATIAL` does in
    :func:`spatial_gradient`), this also accounts for the spatial variation of the projection's scale factors
    themselves, coupling each derivative to the velocity component perpendicular to it.

    Args:
        u: x-component of the vector field. Must have ``"longitude"`` and ``"latitude"`` dimensions.
        v: y-component of the vector field. Must have ``"longitude"`` and ``"latitude"`` dimensions.
        units: Units ``u``/``v``'s latitude/longitude coordinates are expressed in
        components: Derivatives to compute. Defaults to all four of :class:`VelocityDerivative`.
        geod: Geodesic used to compute the nominal grid spacing. Defaults to a WGS84 ellipsoid.
        crs: Coordinate reference system used to compute projection correction factors. Defaults to plain
            latitude/longitude.

    Returns:
        Mapping from each requested :class:`VelocityDerivative` to its computed value
    """
    _check_lat_lon_dimensions_in_array(u)
    _check_lat_lon_dimensions_in_array(v)

    if components is None:
        components = [
            VelocityDerivative.DU_DX,
            VelocityDerivative.DU_DY,
            VelocityDerivative.DV_DX,
            VelocityDerivative.DV_DY,
        ]

    correction_factors = get_projection_correction_factors(
        u["latitude"],
        u["longitude"],
        is_dask_collection(u),
        is_radians=(units == "rad"),
        crs=crs,
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
    dx_correction: xr.DataArray = (correction_factors.meridional_scale / correction_factors.parallel_scale) * dp_dy
    dy_correction: xr.DataArray = (correction_factors.parallel_scale / correction_factors.meridional_scale) * dm_dx

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
            case _ as unreachable:
                assert_never(unreachable)

    return derivatives
