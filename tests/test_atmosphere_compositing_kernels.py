from typing import TYPE_CHECKING

import numpy as np
import pyproj
import pytest
import scipy.interpolate as si

from rojak.atmosphere._compositing_kernels import project_about_centre, wrap_longitude_offset

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def smooth_field_on_sphere(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # Single-valued at the poles and periodic in longitude, so it is well-defined everywhere on the sphere
    return np.cos(np.radians(lat)) * np.cos(np.radians(lon)) + np.sin(np.radians(lat))


def expected_on_km_grid(centre_lat: float, centre_lon: float, km_coords: np.ndarray) -> np.ndarray:
    x_km, y_km = np.meshgrid(km_coords, km_coords)
    ortho = pyproj.Proj(proj="ortho", lat_0=centre_lat, lon_0=centre_lon, units="km", ellps="WGS84")
    lons, lats = ortho(x_km, y_km, inverse=True)
    return smooth_field_on_sphere(lats, lons)


class TestWrapLongitudeOffset:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (0.0, 0.0),
            (179.5, 179.5),
            (180.0, -180.0),
            (-180.0, -180.0),
            (-190.0, 170.0),
            (350.0, -10.0),
            (-350.0, 10.0),
            (720.0, 0.0),
        ],
    )
    def test_wrapped_values_relative_to_zero(self, offset: float, expected: float) -> None:
        np.testing.assert_allclose(wrap_longitude_offset(np.array([offset]), 0.0), [expected])

    @pytest.mark.parametrize(
        ("lons", "relative_to", "expected"),
        [
            pytest.param([170.0, 179.5, -180.0, -175.0], 175.0, [-5.0, 4.5, 5.0, 10.0], id="across_antimeridian"),
            pytest.param([175.0, -175.0], -178.0, [-7.0, 3.0], id="across_antimeridian_negative_reference"),
            pytest.param([355.0, 0.0, 5.0], 2.0, [-7.0, -2.0, 3.0], id="across_0_360_seam"),
            pytest.param([358.0, 0.0, 2.0], 0.0, [-2.0, 0.0, 2.0], id="reference_on_0_360_seam"),
            pytest.param([170.0, 190.0], -175.0, [-15.0, 5.0], id="lons_0_to_360_negative_reference"),
            pytest.param([-10.0, 10.0], 355.0, [-5.0, 15.0], id="lons_-180_to_180_reference_above_180"),
        ],
    )
    def test_wrapped_values(self, lons: list[float], relative_to: float, expected: list[float]) -> None:
        np.testing.assert_allclose(wrap_longitude_offset(np.array(lons), relative_to), expected)

    @pytest.mark.parametrize("relative_to", [0.0, 175.0, -178.0, 359.0])
    def test_output_in_range(self, relative_to: float) -> None:
        half_circle: float = 180
        wrapped = wrap_longitude_offset(np.linspace(-1000, 1000, 2001), relative_to)
        assert (wrapped >= -half_circle).all()
        assert (wrapped < half_circle).all()

    @pytest.mark.parametrize("relative_to", [0.0, 175.0, -178.0, 359.0])
    def test_independent_of_longitude_convention(self, relative_to: float) -> None:
        lons_180 = np.arange(-180, 180, 0.5)
        lons_360 = np.where(lons_180 < 0, lons_180 + 360, lons_180)
        np.testing.assert_allclose(
            wrap_longitude_offset(lons_180, relative_to), wrap_longitude_offset(lons_360, relative_to)
        )


class TestProjectAboutCentre:
    lats: np.ndarray = np.arange(-90, 90.1, 0.5)
    km_coords: np.ndarray = np.arange(-2000, 2001, 50.0)
    tolerance: float = 1e-3

    @staticmethod
    def _field(lons: np.ndarray) -> np.ndarray:
        lon_grid, lat_grid = np.meshgrid(lons, TestProjectAboutCentre.lats)
        return smooth_field_on_sphere(lat_grid, lon_grid)

    def _project(self, lons: np.ndarray, centre_lat: float, centre_lon: float) -> np.ndarray:
        return project_about_centre(
            self._field(lons), self.km_coords, centre_lat, centre_lon, self.lats, lons, 0.5, self.km_coords.size
        )

    def test_output_shape(self) -> None:
        result = self._project(np.arange(-180, 180, 0.5), 50.0, 0.0)
        assert result.shape == (self.km_coords.size, self.km_coords.size)

    @pytest.mark.parametrize(
        "lons",
        [np.arange(-180, 180, 0.5), np.arange(0, 360, 0.5)],
        ids=["lon_-180_to_180", "lon_0_to_360"],
    )
    @pytest.mark.parametrize(
        ("centre_lat", "centre_lon"),
        [
            pytest.param(50.0, -30.0, id="away_from_any_seam"),
            pytest.param(50.0, 175.0, id="crosses_antimeridian_east"),
            pytest.param(-40.0, -178.0, id="crosses_antimeridian_west"),
            pytest.param(10.0, 180.0, id="on_antimeridian"),
            pytest.param(50.0, 0.0, id="on_prime_meridian"),
            pytest.param(50.0, 359.0, id="centre_lon_above_180"),
            pytest.param(85.0, 30.0, id="covers_north_pole"),
            pytest.param(-88.0, -100.0, id="covers_south_pole"),
        ],
    )
    def test_matches_analytic_field(
        self, lons: np.ndarray, centre_lat: float, centre_lon: float, mocker: "MockerFixture"
    ) -> None:
        griddata_spy = mocker.spy(si, "griddata")
        result = self._project(lons, centre_lat, centre_lon)

        # Points projected to inf (i.e. not visible) must not be passed on to griddata, as qhull fails on them
        points = griddata_spy.call_args.kwargs["points"]
        assert np.isfinite(points[0]).all()
        assert np.isfinite(points[1]).all()

        assert not np.isnan(result).any()
        np.testing.assert_allclose(
            result, expected_on_km_grid(centre_lat, centre_lon, self.km_coords), atol=self.tolerance
        )

    def test_equivalent_for_both_longitude_conventions(self) -> None:
        np.testing.assert_allclose(
            self._project(np.arange(-180, 180, 0.5), 30.0, 179.0),
            self._project(np.arange(0, 360, 0.5), 30.0, 179.0),
        )

    def test_only_selects_longitudes_near_centre_when_crossing_antimeridian(self, mocker: "MockerFixture") -> None:
        griddata_spy = mocker.spy(si, "griddata")
        lons = np.arange(-180, 180, 0.5)
        _ = self._project(lons, 50.0, 175.0)

        num_points_passed = griddata_spy.call_args.kwargs["points"][0].size
        assert num_points_passed < self.lats.size * lons.size / 4

    def test_returns_nan_when_centre_outside_of_domain(self) -> None:
        regional_lats = np.linspace(65.0, 25.0, 81)
        regional_lons = np.linspace(-100.0, 10.0, 221)
        result = project_about_centre(
            np.ones((regional_lats.size, regional_lons.size)),
            self.km_coords,
            -50.0,
            120.0,
            regional_lats,
            regional_lons,
            0.5,
            self.km_coords.size,
        )
        assert result.shape == (self.km_coords.size, self.km_coords.size)
        assert np.isnan(result).all()
