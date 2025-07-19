import pytest

from rojak.orchestrator.configuration import SpatialDomain
from rojak.plot.turbulence_plotter import _calculate_extent


@pytest.mark.parametrize(
    ("min_lat", "max_lat", "min_lon", "max_lon"), [(0, 90, 0, 90), (-90, 90, -180, 180), (30, 45, -10, 20)]
)
def test_calculate_extent_spatial_domain(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> None:
    spatial_domain = SpatialDomain(
        minimum_latitude=min_lat, maximum_latitude=max_lat, minimum_longitude=min_lon, maximum_longitude=max_lon
    )
    assert _calculate_extent(spatial_domain) == (min_lon, max_lon, min_lat, max_lat)


def test_calculate_extent_polygon(valid_region_for_flat_data) -> None:
    assert _calculate_extent(valid_region_for_flat_data) == (-130, 28, 25, 60)
