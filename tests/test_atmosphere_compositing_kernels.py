from typing import TYPE_CHECKING

import numpy as np
import pyproj
import pytest
import scipy.interpolate as si
import xarray as xr

from rojak.atmosphere._compositing_kernels import (
    bilinear_about_point,
    bilinear_project_about_centre,
    is_periodic_longitude,
    project_about_centre,
    wrap_longitude_offset,
)

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


class TestIsPeriodicLongitude:
    @pytest.mark.parametrize(
        ("lons", "expected"),
        [
            pytest.param(np.arange(0, 360, 0.25), True, id="global_0_to_360"),
            pytest.param(np.arange(-180, 180, 0.25), True, id="global_-180_to_180"),
            pytest.param(np.arange(359.75, -0.1, -0.25), True, id="global_descending"),
            pytest.param(np.arange(-80, 20.1, 0.25), False, id="regional"),
            pytest.param(np.arange(0, 360.1, 0.25), False, id="global_with_duplicated_seam"),
            pytest.param(np.array([10.0]), False, id="single_point"),
            pytest.param(
                np.array([0.0, 0.5, *np.arange(1.0, 360.0, 1.0)]), True, id="global_irregular_spacing_at_start"
            ),
            pytest.param(
                np.setdiff1d(np.arange(0.0, 360.0, 1.0), np.arange(100.0, 150.0, 1.0)),
                True,
                id="global_with_hole_away_from_seam",
            ),
            pytest.param(np.arange(0.0, 350.0, 1.0), False, id="nearly_global"),
            pytest.param(np.linspace(0, 360, 1441)[:-1], True, id="global_linspace"),
            pytest.param(np.linspace(0, 360, 1441), False, id="global_linspace_with_duplicated_seam"),
        ],
    )
    def test_is_periodic_longitude(self, lons: np.ndarray, expected: bool) -> None:
        assert is_periodic_longitude(lons) is expected


class TestBilinearAboutPoint:
    # Descending latitudes, as used by ERA5
    lats: np.ndarray = np.arange(70, 29.9, -0.5)
    lons: np.ndarray = np.arange(-80, 20.1, 0.5)

    def _interpolate(
        self, data: np.ndarray, target_lats: np.ndarray, target_lons: np.ndarray, lons: np.ndarray | None = None
    ) -> np.ndarray:
        lons = self.lons if lons is None else lons
        return bilinear_about_point(
            data,
            target_lats=target_lats,
            target_lons=target_lons,
            lats=self.lats,
            lons=lons,
            is_lon_periodic=is_periodic_longitude(lons),
        )

    def test_exact_for_bilinear_field(self) -> None:
        # Bilinear interpolation reproduces any field of the form a + b * lat + c * lon + d * lat * lon exactly
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        data = 3.0 + 0.5 * lat_grid - 0.2 * lon_grid + 0.01 * lat_grid * lon_grid
        rng = np.random.default_rng(0)
        target_lats = rng.uniform(30, 70, (20, 30))
        target_lons = rng.uniform(-80, 20, (20, 30))

        np.testing.assert_allclose(
            self._interpolate(data, target_lats, target_lons),
            3.0 + 0.5 * target_lats - 0.2 * target_lons + 0.01 * target_lats * target_lons,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_matches_xarray_interp(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.random((self.lats.size, self.lons.size))
        target_lats = rng.uniform(25, 75, (40, 40))
        target_lons = rng.uniform(-85, 25, (40, 40))

        expected = (
            xr.DataArray(data, dims=("latitude", "longitude"), coords={"latitude": self.lats, "longitude": self.lons})
            .interp(
                latitude=xr.DataArray(target_lats, dims=("y", "x")),
                longitude=xr.DataArray(target_lons, dims=("y", "x")),
            )
            .to_numpy()
        )
        result = self._interpolate(data, target_lats, target_lons)

        # Targets outside of the domain are NaN in both
        np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_regional_grid_does_not_wrap(self) -> None:
        data = np.ones((self.lats.size, self.lons.size))
        result = self._interpolate(data, np.array([50.0, 50.0, 50.0]), np.array([20.25, -80.25, 10.0]))
        np.testing.assert_array_equal(result, [np.nan, np.nan, 1.0])

    def test_longitudes_normalised_to_convention_of_data(self) -> None:
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        data = lat_grid + lon_grid
        # -30.25 degrees is the same as 329.75 degrees
        result = self._interpolate(data, np.array([50.0, 50.0]), np.array([-30.25, 329.75]))
        np.testing.assert_allclose(result, [50.0 - 30.25, 50.0 - 30.25])

    @pytest.mark.parametrize(
        "lons",
        [
            np.arange(0, 360, 1.0),
            np.arange(-180, 180, 1.0),
            np.arange(359.0, -0.1, -1.0),
            np.array([0.0, 0.5, *np.arange(1.0, 360.0, 1.0)]),
        ],
        ids=["lon_0_to_360", "lon_-180_to_180", "lon_descending", "lon_irregular"],
    )
    def test_wraps_between_last_and_first_longitude(self, lons: np.ndarray) -> None:
        rng = np.random.default_rng(2)
        data = rng.random((self.lats.size, lons.size))
        # Find the two columns either side of the seam in the data, i.e. the most easterly and most westerly columns
        east_col, west_col = int(np.argmax(lons)), int(np.argmin(lons))
        # A quarter of the way from the most easterly longitude, across the seam, to the most westerly longitude
        target_lon = lons[east_col] + 0.25
        lat_index = 10

        result = self._interpolate(data, np.array([self.lats[lat_index]]), np.array([target_lon]), lons=lons)

        np.testing.assert_allclose(result, [0.75 * data[lat_index, east_col] + 0.25 * data[lat_index, west_col]])


class TestBilinearProjectAboutCentre:
    lats: np.ndarray = np.arange(-90, 90.1, 0.5)
    km_coords: np.ndarray = np.arange(-2000, 2001, 50.0)
    tolerance: float = 1e-3

    def _project(self, lons: np.ndarray, centre_lat: float, centre_lon: float) -> np.ndarray:
        lon_grid, lat_grid = np.meshgrid(lons, self.lats)
        return bilinear_project_about_centre(
            smooth_field_on_sphere(lat_grid, lon_grid),
            self.km_coords,
            centre_lat,
            centre_lon,
            self.lats,
            lons,
            is_lon_periodic=is_periodic_longitude(lons),
        )

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
        ],
    )
    def test_matches_analytic_field(self, lons: np.ndarray, centre_lat: float, centre_lon: float) -> None:
        result = self._project(lons, centre_lat, centre_lon)
        assert not np.isnan(result).any()
        np.testing.assert_allclose(
            result, expected_on_km_grid(centre_lat, centre_lon, self.km_coords), atol=self.tolerance
        )

    def test_equivalent_for_both_longitude_conventions(self) -> None:
        np.testing.assert_allclose(
            self._project(np.arange(-180, 180, 0.5), 30.0, 179.0),
            self._project(np.arange(0, 360, 0.5), 30.0, 179.0),
        )

    def test_close_to_griddata(self) -> None:
        lons = np.arange(-180, 180, 0.5)
        lon_grid, lat_grid = np.meshgrid(lons, self.lats)
        griddata = project_about_centre(
            smooth_field_on_sphere(lat_grid, lon_grid),
            self.km_coords,
            45.0,
            170.0,
            self.lats,
            lons,
            0.5,
            self.km_coords.size,
        )
        np.testing.assert_allclose(self._project(lons, 45.0, 170.0), griddata, atol=self.tolerance)

    def test_nan_outside_of_regional_domain(self) -> None:
        regional_lats = np.linspace(65.0, 25.0, 81)
        regional_lons = np.linspace(-100.0, 10.0, 221)
        result = bilinear_project_about_centre(
            np.ones((regional_lats.size, regional_lons.size)),
            self.km_coords,
            -50.0,
            120.0,
            regional_lats,
            regional_lons,
            is_lon_periodic=False,
        )
        assert np.isnan(result).all()
