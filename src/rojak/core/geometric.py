import itertools
from typing import TYPE_CHECKING

import numpy as np
from shapely import geometry
from shapely.prepared import prep

if TYPE_CHECKING:
    from rojak.orchestrator.configuration import SpatialDomain


def _create_grid_boxes(bounding_box: geometry.Polygon, step_size: float) -> list[geometry.Polygon]:
    # Modified from
    # https://www.matecdev.com/posts/shapely-polygon-gridding.html
    min_x, min_y, max_x, max_y = bounding_box.bounds
    nx = int(np.ceil((max_x - min_x) / step_size))
    ny = int(np.ceil((max_y - min_y) / step_size))

    x_loc = np.linspace(min_x, max_x, nx + 1)
    y_loc = np.linspace(min_y, max_y, ny + 1)
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
