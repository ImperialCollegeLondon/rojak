import pytest
from shapely import geometry

from rojak.core.geometric import create_polygon_spatial_grid_buckets, create_rectangular_spatial_grid_buckets
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
