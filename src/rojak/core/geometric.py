"""
Geometric utilities for spatial gridding, aggregation, and geodesic calculations

This module provides standalone functions for building spatial grids over a domain and aggregating data onto them
(:func:`create_grid_data_frame`, :func:`create_rectangular_spatial_grid_buckets`,
:func:`create_polygon_spatial_grid_buckets`, :func:`spatial_aggregation`), computing waypoints along the great
circle between two coordinates and interpolating gridded data along them (:func:`geodesic_waypoints_between`,
:func:`interpolate_to_geodesic_waypoints`), and computing the great-circle distance between two points
(:func:`haversine_distance`).
"""

import itertools
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import dask_geopandas as dgpd
import geopandas as gpd
import numba
import numpy as np
import pyproj
import xarray as xr
from shapely import geometry
from shapely.prepared import prep

from rojak.core.constants import EARTH_AVG_RADIUS
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.orchestrator.configuration import SpatialDomain


def _create_grid_boxes(bounding_box: geometry.Polygon, step_size: float) -> list[geometry.Polygon]:
    """
    Tile a polygon's bounding box into a regular grid of rectangular polygons

    Args:
        bounding_box: Polygon whose bounding box is tiled
        step_size: Approximate width/height of each grid cell, in the units of ``bounding_box``'s CRS

    Returns:
        List of rectangular polygons tiling ``bounding_box``'s bounding box. The grid spacing is adjusted slightly
        so that the bounding box's extent is tiled exactly, so the actual cell size may differ slightly from
        ``step_size``.
    """
    # Modified from
    # https://www.matecdev.com/posts/shapely-polygon-gridding.html
    min_x, min_y, max_x, max_y = bounding_box.bounds
    nx: int = int(np.ceil((max_x - min_x) / step_size))
    ny: int = int(np.ceil((max_y - min_y) / step_size))

    x_loc: NDArray = np.linspace(min_x, max_x, nx + 1)
    y_loc: NDArray = np.linspace(min_y, max_y, ny + 1)
    return [
        geometry.box(x_loc[x_index], y_loc[y_index], x_loc[x_index + 1], y_loc[y_index + 1])
        for x_index, y_index in itertools.product(range(nx), range(ny))
    ]


def create_rectangular_spatial_grid_buckets(domain: "SpatialDomain", step_size: float) -> list[geometry.Polygon]:
    """
    Tile a rectangular spatial domain into a regular grid of polygons

    Args:
        domain: Rectangular spatial domain to tile
        step_size: Approximate width/height of each grid cell, in degrees

    Returns:
        List of rectangular polygons tiling ``domain``
    """
    bounding_box: geometry.Polygon = geometry.box(
        domain.minimum_longitude,
        domain.minimum_latitude,
        domain.maximum_longitude,
        domain.maximum_latitude,
    )
    return _create_grid_boxes(bounding_box, step_size)


def create_polygon_spatial_grid_buckets(domain: geometry.Polygon, step_size: float) -> list[geometry.Polygon]:
    """
    Tile an arbitrary polygon's bounding box into a regular grid, keeping only cells that intersect it

    Args:
        domain: Polygon to tile
        step_size: Approximate width/height of each grid cell, in the units of ``domain``'s CRS

    Returns:
        List of rectangular polygons tiling ``domain``'s bounding box, filtered to those that intersect ``domain``
    """
    prepared_geometry = prep(domain)
    return list(filter(prepared_geometry.intersects, _create_grid_boxes(domain, step_size)))


def create_grid_data_frame(
    domain: "SpatialDomain | geometry.Polygon",
    step_size: float,
    crs: str = "epsg:4326",
) -> dgpd.GeoDataFrame:
    """
    Build a (dask) GeoDataFrame of grid cells tiling a spatial domain

    Args:
        domain: Spatial domain to tile. If a :class:`~rojak.orchestrator.configuration.SpatialDomain`, its
            rectangular bounding box is tiled (see :func:`create_rectangular_spatial_grid_buckets`). If a
            :class:`shapely.geometry.Polygon`, only cells intersecting it are kept (see
            :func:`create_polygon_spatial_grid_buckets`).
        step_size: Approximate width/height of each grid cell, in the units of ``crs``
        crs: Coordinate reference system of the grid. Defaults to ``"epsg:4326"``.

    Returns:
        Dask GeoDataFrame with one row per grid cell
    """
    grid = gpd.GeoDataFrame(
        geometry=create_polygon_spatial_grid_buckets(domain, step_size)
        if isinstance(domain, geometry.Polygon)
        else create_rectangular_spatial_grid_buckets(domain, step_size),
        crs=crs,
    )
    return dgpd.from_geopandas(grid)


def spatial_aggregation(
    grid: "dgpd.GeoDataFrame",
    data_to_aggregate: "dgpd.GeoDataFrame",
    columns_to_aggregate: list[str],
    agg_func: Callable | str | dict,
    by: str = "index_right",
    drop_na: bool = True,
) -> "dgpd.GeoDataFrame":
    """
    Aggregate data onto a grid, based on a prior spatial join between the data and the grid

    Args:
        grid: GeoDataFrame of grid cells (see :func:`create_grid_data_frame`) to aggregate onto
        data_to_aggregate: GeoDataFrame of data to aggregate. Expected to already contain a column (``by``)
            identifying which row of ``grid`` each row belongs to, e.g. as added by spatially joining
            ``data_to_aggregate`` onto ``grid`` with :meth:`geopandas.GeoDataFrame.sjoin`.
        columns_to_aggregate: Columns of ``data_to_aggregate`` to aggregate. ``"geometry"`` is appended
            automatically if not already present, since it is required by ``dissolve``.
        agg_func: Aggregation function(s), passed as the ``aggfunc`` of :meth:`geopandas.GeoDataFrame.dissolve`
        by: Column of ``data_to_aggregate`` identifying which grid cell each row belongs to. Defaults to
            ``"index_right"``, the column added by :meth:`geopandas.GeoDataFrame.sjoin`.
        drop_na: If ``True`` (default), drop grid cells with no aggregated data (i.e. no rows joined to them)

    Returns:
        ``grid`` joined with the aggregated values of ``columns_to_aggregate``, one row per grid cell
    """
    if not {"geometry"}.issubset(columns_to_aggregate):
        columns_to_aggregate.append("geometry")

    relevant_data = data_to_aggregate[columns_to_aggregate]
    aggregated_data = grid.join(relevant_data.dissolve(by=by, aggfunc=agg_func))

    return aggregated_data.dropna() if drop_na else aggregated_data


# For now, assume symmetric grid
def _estimate_num_waypoints(start: Coordinate, end: Coordinate, grid_size: float, n_points_safety_factor: float) -> int:
    """
    Estimate the number of waypoints between two points based on the grid_size

    Args:
        start: Starting coordinate
        end: Ending coordinate
        grid_size: Grid spacing in degrees. For Era5, this is 0.25
        n_points_safety_factor: Safety factor applied for the estimation of number of waypoints. Larger => more points

    Returns:
        Estimated number of waypoints

    Examples
    ---------

    >>> lhr = Coordinate(51.47138888, -0.45277777)
    >>> jfk = Coordinate(40.641766, -73.780968)
    >>> _estimate_num_waypoints(lhr, jfk, 0.25, 2)
    330
    >>> _estimate_num_waypoints(jfk, lhr, 0.25, 2)
    330
    >>> _estimate_num_waypoints(lhr, jfk, 0.25, 4)
    658
    >>> _estimate_num_waypoints(lhr, jfk, 0.25, 0.5)
    84
    >>> _estimate_num_waypoints(lhr, jfk, -0.1, 2)
    Traceback (most recent call last):
    ValueError: grid_size must be non-negative

    """
    if grid_size < 0:
        raise ValueError("grid_size must be non-negative")

    mid_point: Coordinate = start.mid_point_from(end)
    geod = pyproj.Geod(ellps="WGS84")

    _, _, approx_cell_distance = geod.inv(
        mid_point.longitude,
        mid_point.latitude,
        mid_point.longitude + grid_size,
        mid_point.latitude + grid_size,
    )
    _, _, total_distance = geod.inv(start.longitude, start.latitude, end.longitude, end.latitude)

    # mathematically equiv to total_distance / (approx_cell_distance / n_points_safety_factor)
    return max(1, int(np.ceil(total_distance * n_points_safety_factor / approx_cell_distance))) + 1


def geodesic_waypoints_between(
    start: Coordinate,
    end: Coordinate,
    grid_size: float,
    n_points_safety_factor: float = 2,
    n_points: int | None = None,
) -> np.ndarray:
    """
    Find the coordinates (i.e. waypoints) on the great circle between the two points.

    Args:
        start: Starting coordinate
        end: Ending coordinate
        grid_size: Grid spacing in degrees. For Era5, this is 0.25
        n_points_safety_factor: Safety factor applied for the estimation of number of waypoints. Larger => more points.
        If n_points is specified, this value is ignored.
        n_points: If None, then number of points is estimated. Else, this value is used to compute the waypoints

    Returns:
        2D numpy array of points with shape (num_waypoints, 2). The first column is the longitude and the second column
        is the latitude.

    Examples
    --------

    >>> lhr = Coordinate(51.47138888, -0.45277777)
    >>> jfk = Coordinate(40.641766, -73.780968)
    >>> way_points_lhr_jfk = geodesic_waypoints_between(lhr, jfk, 0.25)
    >>> way_points_lhr_jfk.shape
    (330, 2)

    The first and last rows correspond to the longitude and latitude of the starting and end point, respectively.

    >>> way_points_lhr_jfk[0, :]
    array([-0.45277777, 51.47138888])
    >>> way_points_lhr_jfk[-1, :]
    array([-73.780968,  40.641766])

    Flipping the star and end points reverses the order of the array of way points.

    >>> way_points_jfk_lhr = geodesic_waypoints_between(jfk, lhr, 0.25)
    >>> np.testing.assert_array_almost_equal(way_points_lhr_jfk, np.flipud(way_points_jfk_lhr))

    Specifying the number of points

    >>> min_points = geodesic_waypoints_between(lhr, jfk, 0.25, n_points=2)
    >>> min_points.shape
    (2, 2)
    >>> min_points
    array([[ -0.45277777,  51.47138888],
           [-73.780968  ,  40.641766  ]])
    >>> geodesic_waypoints_between(lhr, jfk, 0.25, n_points=1)
    Traceback (most recent call last):
    ValueError: Number of points cannot be less than 2 as it must include start and end points.

    """
    if n_points is None:
        num_points = _estimate_num_waypoints(start, end, grid_size, n_points_safety_factor)
    else:
        if n_points < 2:  # noqa: PLR2004
            raise ValueError("Number of points cannot be less than 2 as it must include start and end points.")
        num_points = n_points

    geod = pyproj.Geod(ellps="WGS84")
    return np.asarray(
        geod.npts(
            start.longitude,
            start.latitude,
            end.longitude,
            end.latitude,
            num_points,
            initial_idx=0,
            terminus_idx=0,
        ),
    )


def interpolate_to_geodesic_waypoints[T: (xr.Dataset, xr.DataArray)](
    start: Coordinate,
    end: Coordinate,
    grid_size: float,
    target_data: T,
    n_points_safety_factor: float = 2,
    n_points: int | None = None,
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    waypoints_dim_name: str = "waypoints",
    **interpolation_kwargs: Any,  # noqa: ANN401
) -> T:
    """
    Interpolate gridded data onto the geodesic waypoints between two coordinates

    See :func:`geodesic_waypoints_between` for how the waypoints themselves are computed.

    Args:
        start: Starting coordinate
        end: Ending coordinate
        grid_size: Grid spacing in degrees, used to estimate the number of waypoints if ``n_points`` is not given
        target_data: Gridded data to interpolate, with ``lat_dim_name``/``lon_dim_name`` coordinates
        n_points_safety_factor: Safety factor applied to the estimated number of waypoints. Ignored if ``n_points``
            is given.
        n_points: Number of waypoints to use. If ``None`` (default), the number is estimated from ``grid_size``.
        lat_dim_name: Name of the latitude coordinate in ``target_data``. Defaults to ``"latitude"``.
        lon_dim_name: Name of the longitude coordinate in ``target_data``. Defaults to ``"longitude"``.
        waypoints_dim_name: Name to give the new dimension along the waypoints. Defaults to ``"waypoints"``.
        **interpolation_kwargs: Additional keyword arguments passed to :meth:`xarray.DataArray.interp`/
            :meth:`xarray.Dataset.interp`

    Returns:
        ``target_data`` interpolated onto the waypoints between ``start`` and ``end``, with a new
        ``waypoints_dim_name`` dimension
    """
    waypoints = geodesic_waypoints_between(
        start, end, grid_size, n_points_safety_factor=n_points_safety_factor, n_points=n_points
    )
    return target_data.interp(
        coords={
            lon_dim_name: xr.DataArray(data=waypoints[:, 0], dims=waypoints_dim_name),
            lat_dim_name: xr.DataArray(data=waypoints[:, 1], dims=waypoints_dim_name),
        },
        **interpolation_kwargs,
    )


class DistanceUnits(StrEnum):
    """Units for a computed distance"""

    METERS = "meters"
    KILOMETERS = "kilometers"


@numba.njit
def haversine_distance(
    lon_1: np.ndarray,
    lat_1: np.ndarray,
    lon_2: np.ndarray,
    lat_2: np.ndarray,
    # distance_units: DistanceUnits = DistanceUnits.KILOMETERS,
    /,
) -> np.ndarray:
    """
    Haversine distance in m between two points as pairs of longitude and latitude

    Args:
        lon_1: Array of longitude in degrees for first pair
        lat_1: Array of latitudes in degrees for first pair
        lon_2: Array of longitude in degrees for second pair
        lat_2: Array of latitudes in degrees for second pair

    Returns:
        haversine distance in meters

    References:
        `Wikipedia on haversine distance <https://en.wikipedia.org/wiki/Haversine_formula#Formulation>`__
        Equation 19 in `this paper <https://doi.org/10.1017%2FS0373463309990415>`__

    Examples:
    >>> float(haversine_distance(np.array(77.037),np.array(38.898), np.array(2.294), np.array(48.858)))
    5846821.4

    """

    # List it out makes it numba friendly
    lon_1 = np.radians(lon_1)
    lon_2 = np.radians(lon_2)
    lat_1 = np.radians(lat_1)
    lat_2 = np.radians(lat_2)

    delta_lat: np.ndarray = lat_2 - lat_1
    delta_lon: np.ndarray = lon_2 - lon_1
    sine_half_delta_lat: np.ndarray = np.sin(delta_lat / 2)
    sine_half_delta_lon: np.ndarray = np.sin(delta_lon / 2)

    haversine_theta: np.ndarray = (
        sine_half_delta_lat * sine_half_delta_lat
        + np.cos(lat_1) * np.cos(lat_2) * sine_half_delta_lon * sine_half_delta_lon
    )

    theta: np.ndarray = 2 * np.arcsin(np.sqrt(haversine_theta))
    # earth_radius: float = EARTH_AVG_RADIUS if distance_units == DistanceUnits.METERS else EARTH_AVG_RADIUS * 1e-3
    return theta * EARTH_AVG_RADIUS
