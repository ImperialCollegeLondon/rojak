import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import Geod
from shapely import geometry

from rojak.core.geometric import (
    create_grid_data_frame,
    create_polygon_spatial_grid_buckets,
    create_rectangular_spatial_grid_buckets,
    haversine_distance,
)
from rojak.orchestrator.configuration import SpatialDomain


@pytest.mark.parametrize(
    ("step_size", "num_buckets", "first_box"),
    [(1, 25, geometry.box(0, 0, 1, 1)), (5, 1, geometry.box(0, 0, 5, 5)), (2.5, 4, geometry.box(0, 0, 2.5, 2.5))],
)
def test_rectangular_spatial_grid_buckets(step_size, num_buckets, first_box):
    domain = SpatialDomain(minimum_latitude=0, maximum_latitude=5, minimum_longitude=0, maximum_longitude=5)
    boxes = create_rectangular_spatial_grid_buckets(domain, step_size)
    assert len(boxes) == num_buckets
    assert boxes[0] == first_box


@pytest.mark.parametrize(
    ("step_size", "num_buckets", "first_box", "last_box"),
    [
        (0.5, 22, geometry.box(0, -1, 0.5, -0.5), geometry.box(1.5, 1, 2, 1.5)),
        (1, 6, geometry.box(0, -1, 1, 0), geometry.box(1, 1, 2, 2)),
    ],
)
def test_create_polygon_spatial_grid_buckets(step_size, num_buckets, first_box, last_box):
    geom = geometry.Polygon([[0, 0], [0, 2], [2, 1], [1, -1], [0, 0]])
    boxes = create_polygon_spatial_grid_buckets(geom, step_size)
    assert len(boxes) == num_buckets
    assert boxes[0] == first_box
    assert boxes[-1] == last_box


def test_create_grid_data_frame_rectangular():
    domain = SpatialDomain(minimum_latitude=0, maximum_latitude=5, minimum_longitude=0, maximum_longitude=5)
    df_from_create_grid = create_grid_data_frame(domain, 1).compute()
    df_it_should_have = create_rectangular_spatial_grid_buckets(domain, 1)

    assert_geodataframe_equal(df_from_create_grid, gpd.GeoDataFrame(geometry=df_it_should_have, crs="epsg:4326"))


def test_create_grid_data_frame_polygon():
    geom = geometry.Polygon([[0, 0], [0, 2], [2, 1], [1, -1], [0, 0]])
    df_from_create_grid = create_grid_data_frame(geom, 0.25).compute()
    df_it_should_have = create_polygon_spatial_grid_buckets(geom, 0.25)

    assert_geodataframe_equal(df_from_create_grid, gpd.GeoDataFrame(geometry=df_it_should_have, crs="epsg:4326"))


def test_haversine_distance():
    target_shape = (40, 50)
    rng = np.random.default_rng()
    lon_1 = (rng.random(size=target_shape, dtype=float) * 360.0) - 180.0
    lon_2 = (rng.random(size=target_shape, dtype=float) * 360.0) - 180.0
    lat_1 = (rng.random(size=target_shape, dtype=float) * 180.0) - 90.0
    lat_2 = (rng.random(size=target_shape, dtype=float) * 180.0) - 90.0

    distances = haversine_distance(lon_1, lat_1, lon_2, lat_2)
    geod = Geod(ellps="WGS84")
    _, _, geod_distances = geod.inv(lon_1, lat_1, lon_2, lat_2)
    np.testing.assert_allclose(distances, geod_distances, rtol=1e-2)
