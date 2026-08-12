from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import Geod
from shapely import geometry

from rojak.core.geometric import (
    create_grid_data_frame,
    create_polygon_spatial_grid_buckets,
    create_rectangular_spatial_grid_buckets,
    haversine_distance,
    interpolate_to_geodesic_waypoints,
)
from rojak.orchestrator.configuration import SpatialDomain
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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


class TestInterpolateToWaypoints:
    START = Coordinate(51.47138888, -0.45277777)
    END = Coordinate(40.641766, -73.780968)
    PATCH_TARGET: str = "rojak.core.geometric.geodesic_waypoints_between"

    def dummy_waypoints(self) -> np.ndarray:
        return np.stack(
            (
                np.linspace(self.START.longitude, self.END.longitude, 20),
                np.linspace(self.START.latitude, self.END.latitude, 20),
            ),
            axis=-1,
        )

    @staticmethod
    def dummy_multi_dim_dataset(is_constant: bool = False) -> xr.Dataset:
        lat = np.linspace(30, 60, 60)
        lon = np.linspace(-90, 0, 180)
        time = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64")
        level = np.array([500.0, 850.0])
        rng = np.random.default_rng(42)
        data = np.ones((2, 2, 60, 180)) if is_constant else rng.random((2, 2, 60, 180))
        return xr.Dataset(
            {"temperature": (["time", "level", "latitude", "longitude"], data)},
            coords={"latitude": lat, "longitude": lon, "time": time, "level": level},
        )

    def test_return_dataset(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()

        result = interpolate_to_geodesic_waypoints(self.START, self.END, 1, self.dummy_multi_dim_dataset())

        waypoints_compute_mock.assert_called_once()
        assert isinstance(result, xr.Dataset)

    def test_non_interp_behaviour_dataarray(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()
        result = interpolate_to_geodesic_waypoints(
            self.START, self.END, 1, self.dummy_multi_dim_dataset()["temperature"]
        )

        waypoints_compute_mock.assert_called_once()
        assert isinstance(result, xr.DataArray)
        assert "waypoints" in result.dims
        assert "latitude" not in result.dims
        assert "longitude" not in result.dims
        assert "time" in result.dims
        assert "level" in result.dims

        assert len(result["waypoints"]) == len(self.dummy_waypoints())

    def test_call_geodesic_waypoints_with_default_kwargs(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()
        _ = interpolate_to_geodesic_waypoints(self.START, self.END, 1, self.dummy_multi_dim_dataset()["temperature"])
        waypoints_compute_mock.assert_called_once_with(
            self.START,
            self.END,
            1,
            n_points_safety_factor=2,
            n_points=None,
        )

    def test_call_geodesic_waypoints_with_new_kwargs(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()
        _ = interpolate_to_geodesic_waypoints(
            self.START,
            self.END,
            1,
            self.dummy_multi_dim_dataset()["temperature"],
            n_points_safety_factor=10,
            n_points=3,
        )
        waypoints_compute_mock.assert_called_once_with(
            self.START,
            self.END,
            1,
            n_points_safety_factor=10,
            n_points=3,
        )

    def test_custom_waypoints_dim_name(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()

        custom_name: str = "potato"
        result = interpolate_to_geodesic_waypoints(
            self.START, self.END, 1, self.dummy_multi_dim_dataset()["temperature"], waypoints_dim_name=custom_name
        )

        waypoints_compute_mock.assert_called_once()
        assert custom_name in result.dims
        assert len(result[custom_name]) == len(self.dummy_waypoints())

    def test_interpolation_kwarg_forwarded(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()
        _ = interpolate_to_geodesic_waypoints(
            self.START,
            self.END,
            1,
            self.dummy_multi_dim_dataset()["temperature"],
            method="nearest",
            method_non_numeric="pad",
        )

        waypoints_compute_mock.assert_called_once()

    def test_interpolation_kwarg_forwarded_raises(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()

        with pytest.raises(ValueError, match="fake_method"):
            _ = interpolate_to_geodesic_waypoints(
                self.START, self.END, 1, self.dummy_multi_dim_dataset()["temperature"], method="fake_method"
            )

    def test_constant_value_interpolation(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        waypoints_compute_mock.return_value = self.dummy_waypoints()

        result = interpolate_to_geodesic_waypoints(
            self.START, self.END, 1, self.dummy_multi_dim_dataset(is_constant=True)["temperature"]
        )
        np.testing.assert_array_almost_equal(result, np.ones_like(result))

    def test_waypoints_output_flipped(self, mocker: "MockerFixture") -> None:
        waypoints_compute_mock = mocker.patch(self.PATCH_TARGET)
        lat_lon_opposite_order = np.stack(
            (
                np.linspace(self.START.latitude, self.END.latitude, 20),
                np.linspace(self.START.longitude, self.END.longitude, 20),
            ),
            axis=-1,
        )
        waypoints_compute_mock.return_value = lat_lon_opposite_order

        result_da = interpolate_to_geodesic_waypoints(
            self.START, self.END, 1, self.dummy_multi_dim_dataset()["temperature"]
        )
        waypoints_compute_mock.assert_called_once()
        assert result_da.isnull().all()

        result_ds = interpolate_to_geodesic_waypoints(self.START, self.END, 1, self.dummy_multi_dim_dataset())
        assert result_ds.isnull().all()
