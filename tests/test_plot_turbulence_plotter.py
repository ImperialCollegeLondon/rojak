import numpy as np
import pytest
import xarray as xr

from rojak.orchestrator.configuration import SpatialDomain
from rojak.plot.turbulence_plotter import _calculate_extent, _cluster_2d_correlations


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


def test_cluster_2d_correlation_identity_matrix() -> None:
    correlation_array: xr.DataArray = xr.DataArray(data=np.identity(15))
    clustered: xr.DataArray = _cluster_2d_correlations(correlation_array)
    np.testing.assert_equal(clustered.values, correlation_array.values)


@pytest.mark.parametrize(
    "symmetric_array",
    [
        np.asarray([[1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1]]),
        np.asarray([[1, 0.5, -0.5], [-0.5, 1, 0.5], [0.5, -0.5, 1]]),
        np.asarray([[1, 0.5, 0], [0, 1, 0.4], [0.3, 0, 1]]),
        np.asarray([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]),
        np.asarray([[1, 0.5, 0.4], [0.5, 1, 0.3], [0.4, 0.3, 1]]),
        np.asarray([[1, 0.75, 0.6], [0.75, 1, 0.2], [0.6, 0.2, 1]]),
    ],
)
def test_cluster_2d_correlation_no_reordering(symmetric_array: np.ndarray) -> None:
    np.fill_diagonal(symmetric_array, 1)
    np.testing.assert_equal(_cluster_2d_correlations(xr.DataArray(data=symmetric_array)), symmetric_array)
