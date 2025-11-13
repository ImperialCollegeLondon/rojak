import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pyproj
from shapely import geometry
from shapely.prepared import prep

from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.orchestrator.configuration import SpatialDomain


def _create_grid_boxes(bounding_box: geometry.Polygon, step_size: float) -> list[geometry.Polygon]:
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
    bounding_box: geometry.Polygon = geometry.box(
        domain.minimum_longitude, domain.minimum_latitude, domain.maximum_longitude, domain.maximum_latitude
    )
    return _create_grid_boxes(bounding_box, step_size)


def create_polygon_spatial_grid_buckets(domain: geometry.Polygon, step_size: float) -> list[geometry.Polygon]:
    prepared_geometry = prep(domain)
    return list(filter(prepared_geometry.intersects, _create_grid_boxes(domain, step_size)))


def create_grid_data_frame(
    domain: "SpatialDomain | geometry.Polygon", step_size: float, crs: str = "epsg:4326"
) -> dgpd.GeoDataFrame:
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

    """
    mid_point: Coordinate = start.mid_point_from(end)
    geod = pyproj.Geod(ellps="WGS84")

    _, _, approx_cell_distance = geod.inv(
        mid_point.longitude, mid_point.latitude, mid_point.longitude + grid_size, mid_point.latitude + grid_size
    )
    _, _, total_distance = geod.inv(start.longitude, start.latitude, end.longitude, end.latitude)

    # mathematically equiv to total_distance / (approx_cell_distance / n_points_safety_factor)
    return max(1, np.ceil(total_distance * n_points_safety_factor / approx_cell_distance, dtype=np.int_)) + 1


def geodesic_waypoints_between(
    start: Coordinate, end: Coordinate, grid_size: float, n_points_safety_factor: float = 2, n_points: int | None = None
) -> np.ndarray:
    """
    Find the coordinates (i.e. waypoints) on the great circle between the two points.

    Args:
        start: Starting coordinate
        end: Ending coordinate
        grid_size: Grid spacing in degrees. For Era5, this is 0.25
        n_points_safety_factor: Safety factor applied for the estimation of number of waypoints. Larger => more points
        n_points: If None, then number of points is estimated. Else, this value is used to compute the waypoints

    Returns:
        2D numpy array of points with shape (num_waypoints, 2). The first column is the latitude and the second column
        is the longitude

    """
    if n_points is None:
        num_points = _estimate_num_waypoints(start, end, grid_size, n_points_safety_factor)
    else:
        num_points = n_points
    geod = pyproj.Geod(ellps="WGS84")
    return np.asarray(
        geod.npts(
            start.longitude, start.latitude, end.longitude, end.latitude, num_points, initial_idx=0, terminus_idx=0
        )
    )
