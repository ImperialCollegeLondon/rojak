import functools
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr

from rojak.atmosphere.compositing import (
    CompositeAboutPoints,
    CompositeInterpolation,
    _project_data_about_extrema,
    composite_about_extrema,
    composite_about_points,
)
from rojak.atmosphere.regions import ExtremaKind, identify_and_stack_circular_extrema

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestProjectDataAboutExtrema:
    target_shape: tuple[int, int] = (161, 441)
    lats: np.ndarray = np.linspace(65.0, 25.0, 161)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 441)

    def test_output_shape_is_n_km_by_n_km(self) -> None:
        mask = np.zeros(self.target_shape, dtype=np.int8)
        mask[10, 20] = 2
        mask[8:13, 18:23] = 1
        field = np.random.default_rng(42).random(self.target_shape)

        result = _project_data_about_extrema(
            mask,
            field,
            lats=self.lats,
            lons=self.lons,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        n_km = len(np.arange(-500, 501, 50))
        assert result.shape == (n_km, n_km)

    def test_returns_nan_when_no_centre(self) -> None:
        result = _project_data_about_extrema(
            np.zeros(self.target_shape, dtype=np.int8),
            np.ones(self.target_shape),
            lats=self.lats,
            lons=self.lons,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        assert np.all(np.isnan(result))

    def test_raises_if_multiple_centres(self) -> None:
        mask = np.zeros(self.target_shape, dtype=np.int8)
        mask[10, 20] = 2
        mask[50, 100] = 2  # second centre — invalid
        field = np.ones(self.target_shape)

        with pytest.raises(ValueError, match="There should only be one extrema"):
            _project_data_about_extrema(
                mask,
                field,
                lats=self.lats,
                lons=self.lons,
                max_km_extent=500,
                grid_km_spacing=50,
            )

    def test_centre_of_output_is_not_nan(self, load_mslp_data: xr.DataArray) -> None:
        msl_shape = load_mslp_data.isel(time=0).shape
        mask = np.zeros(msl_shape, dtype=np.int8)
        mask[80, 200] = 2  # centre of the domain
        mask[78:83, 218:223] = 1

        result = _project_data_about_extrema(
            mask,
            load_mslp_data.isel(time=0).values,
            lats=load_mslp_data["latitude"].values,
            lons=load_mslp_data["longitude"].values,
            max_km_extent=1000,
            grid_km_spacing=25,
        )
        centre_idx = result.shape[0] // 2
        assert not np.isnan(result[centre_idx, centre_idx])

    def test_custom_int_for_center(self, load_mslp_data: xr.DataArray) -> None:
        msl_shape = load_mslp_data.isel(time=0).shape
        mask = np.zeros(msl_shape, dtype=np.int8)
        mask[80, 200] = 9  # centre of the domain
        mask[78:83, 218:223] = 1

        result = _project_data_about_extrema(
            mask,
            load_mslp_data.isel(time=0).values,
            lats=load_mslp_data["latitude"].values,
            lons=load_mslp_data["longitude"].values,
            max_km_extent=1000,
            grid_km_spacing=25,
            int_for_center=9,
        )
        assert result.shape[0] > 0
        assert not np.all(np.isnan(result))

    def test_extrema_near_antimeridian(self) -> None:
        lats = np.arange(-90, 90.1, 0.5)
        lons = np.arange(-180, 180, 0.5)
        mask = np.zeros((lats.size, lons.size), dtype=np.int8)
        mask[np.argmin(np.abs(lats - 50)), np.argmin(np.abs(lons - 179.5))] = 2

        # This used to raise a QhullError as every longitude was selected, including ones not visible in the projection
        result = _project_data_about_extrema(
            mask,
            np.ones(mask.shape),
            lats=lats,
            lons=lons,
            max_km_extent=1000,
            grid_km_spacing=50,
        )
        np.testing.assert_allclose(result, 1.0)


class TestCompositeAboutExtrema:
    target_shape: tuple[int, int] = (161, 441)
    lats: np.ndarray = np.linspace(65.0, 25.0, 161)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 441)
    time: np.ndarray = np.array(
        ["2025-12-29T00:00", "2025-12-29T06:00", "2025-12-29T12:00", "2025-12-29T18:00"], dtype="datetime64[h]"
    )

    @pytest.fixture
    def dummy_target_data(self) -> xr.DataArray:
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        data = np.stack([lat_grid + lon_grid] * len(self.time), axis=0)
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": self.time,
                "latitude": self.lats,
                "longitude": self.lons,
            },
        )

    @pytest.fixture
    def dummy_extrema_data(self) -> xr.DataArray:
        n_extrema = len(self.time)
        data = np.zeros((n_extrema, self.target_shape[0], self.target_shape[1]), dtype=np.int8)

        for i in range(n_extrema):
            data[i, 8:13, 18:23] = 1  # region
            data[i, 10, 20] = 2  # centre

        return xr.DataArray(
            data,
            dims=["n_extrema", "latitude", "longitude"],
            coords={
                "n_extrema": np.arange(n_extrema),
                "time": ("n_extrema", self.time),
                "latitude": self.lats,
                "longitude": self.lons,
            },
        )

    def test_raises_if_lat_lon_missing_from_target(self, dummy_extrema_data: xr.DataArray) -> None:
        bad_target = xr.DataArray(
            np.ones((4, 10)),
            dims=["time", "x"],
            coords={"time": self.time},
        )
        with pytest.raises(ValueError, match="do not contain latitude and longitude"):
            composite_about_extrema(bad_target, dummy_extrema_data)

    def test_raises_if_vert_dim_not_in_target(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray
    ) -> None:
        with pytest.raises(ValueError, match="Expected 'pressure_level' to be present in target_data"):
            composite_about_extrema(dummy_target_data, dummy_extrema_data, vert_dim_name="pressure_level")

    def test_raises_if_n_extrema_not_in_extrema_data(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray
    ) -> None:
        bad_extrema = dummy_extrema_data.rename({"n_extrema": "wrong_dim"})
        with pytest.raises(ValueError, match="Expected 'n_extrema' to be in"):
            composite_about_extrema(dummy_target_data, bad_extrema)

    def test_raises_if_no_extrema(self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray) -> None:
        empty_extrema = dummy_extrema_data.isel(n_extrema=slice(0, 0))
        with pytest.raises(ValueError, match="at least one extrema along 'n_extrema'"):
            composite_about_extrema(dummy_target_data, empty_extrema)

    def test_raises_if_time_not_in_target(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray
    ) -> None:
        bad_target = dummy_target_data.rename({"time": "wrong_time"})
        with pytest.raises(ValueError, match="Expected 'time' to be present in target_data"):
            composite_about_extrema(bad_target, dummy_extrema_data)

    def test_raises_if_time_not_in_extrema_coords(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray
    ) -> None:
        bad_extrema = dummy_extrema_data.drop_vars("time")
        with pytest.raises(ValueError, match="Expected 'time' to be present in extrema_data coords"):
            composite_about_extrema(dummy_target_data, bad_extrema)

    def test_on_mslp_data(self, load_mslp_data: xr.DataArray) -> None:
        extrema_data = identify_and_stack_circular_extrema(
            load_mslp_data, ExtremaKind.MINIMA, extrema_threshold_value=100000
        )
        result = composite_about_extrema(load_mslp_data, extrema_data)

        # Check basic properties of the result
        assert isinstance(result, xr.DataArray)
        assert "y_km" in result.dims
        assert "x_km" in result.dims
        assert "time" in result.coords
        assert "n_extrema" in result["time"].dims

        # Check km_coords match the defaults for the composite function
        expected = np.arange(-2000, 2001, 25)
        np.testing.assert_array_equal(result["y_km"].values, expected)
        np.testing.assert_array_equal(result["x_km"].values, expected)

        # Obtained from manually inspecting the data
        num_minima_in_mslp: int = 22
        assert result["n_extrema"].size == num_minima_in_mslp

        # Taking the mean over `n_extrema` should not result in any NaNs
        msl_composite_mean = result.mean(dim="n_extrema", skipna=True) / 100
        assert not np.isnan(msl_composite_mean).any()

    def test_is_only_extrema_region_applies_where(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray, mocker: "MockerFixture"
    ) -> None:
        mock_where = mocker.patch.object(xr.DataArray, "where")
        composite_about_extrema(dummy_target_data, dummy_extrema_data, is_only_extrema_region=True)

        mock_where.assert_called_once()

    def test_is_only_extrema_region_false_does_not_apply_where(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray, mocker: "MockerFixture"
    ) -> None:
        mock_where = mocker.patch.object(xr.DataArray, "where", wraps=xr.DataArray.where)
        composite_about_extrema(dummy_target_data, dummy_extrema_data, is_only_extrema_region=False)
        mock_where.assert_not_called()

    def test_with_vertical_dimension(
        self, dummy_target_data: xr.DataArray, dummy_extrema_data: xr.DataArray, mocker: "MockerFixture"
    ) -> None:
        pressure = np.array([500.0, 850.0])
        target_4d = dummy_target_data.expand_dims({"pressure_level": pressure})
        result = composite_about_extrema(
            target_4d,
            dummy_extrema_data,
            vert_dim_name="pressure_level",
        )
        assert "pressure_level" in result.coords
        np.testing.assert_array_equal(result["pressure_level"], pressure)


class TestCompositeAboutPoints:
    @pytest.fixture
    def extrema_and_centres(self, load_mslp_data: xr.DataArray) -> tuple[xr.DataArray, pd.DataFrame]:
        extrema = identify_and_stack_circular_extrema(
            load_mslp_data, ExtremaKind.MINIMA, extrema_threshold_value=100000
        )
        indices = np.argwhere(extrema.transpose("n_extrema", "latitude", "longitude").to_numpy() == 2)  # noqa: PLR2004
        centres = pd.DataFrame(
            {
                "time": extrema["time"].to_numpy()[indices[:, 0]],
                "latitude": load_mslp_data["latitude"].to_numpy()[indices[:, 1]],
                "longitude": load_mslp_data["longitude"].to_numpy()[indices[:, 2]],
                "extrema_id": indices[:, 0],
            }
        ).sample(frac=1, random_state=0)
        return extrema, centres

    @pytest.mark.parametrize(
        ("time_chunk", "num_partitions", "max_centres_per_task"), [(8, 1, 500), (3, 3, 500), (3, 2, 4)]
    )
    def test_griddata_equiv_composite_about_extrema(
        self,
        load_mslp_data: xr.DataArray,
        extrema_and_centres: tuple[xr.DataArray, pd.DataFrame],
        time_chunk: int,
        num_partitions: int,
        max_centres_per_task: int,
    ) -> None:
        extrema, centres = extrema_and_centres
        expected = composite_about_extrema(load_mslp_data, extrema).compute()

        result = composite_about_points(
            load_mslp_data.chunk(time=time_chunk),
            dd.from_pandas(centres, npartitions=num_partitions),
            carry_columns=["extrema_id"],
            interpolation=CompositeInterpolation.GRIDDATA,
            max_centres_per_task=max_centres_per_task,
        ).compute()

        expected = expected.sel(n_extrema=result["extrema_id"].to_numpy())
        np.testing.assert_array_equal(result["time"].to_numpy(), expected["time"].to_numpy())
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

    def test_bilinear_close_to_griddata(
        self, load_mslp_data: xr.DataArray, extrema_and_centres: tuple[xr.DataArray, pd.DataFrame]
    ) -> None:
        _, centres = extrema_and_centres
        composites = {
            interpolation: composite_about_points(
                load_mslp_data.chunk(time=3),
                dd.from_pandas(centres, npartitions=3),
                interpolation=interpolation,
            ).compute()
            for interpolation in CompositeInterpolation
        }
        bilinear = composites[CompositeInterpolation.BILINEAR].to_numpy()
        griddata = composites[CompositeInterpolation.GRIDDATA].to_numpy()
        # griddata can return values just outside the domain (within the convex hull of the projected grid)
        assert not (np.isfinite(bilinear) & np.isnan(griddata)).any()
        both_finite = np.isfinite(bilinear) & np.isfinite(griddata)
        np.testing.assert_allclose(bilinear[both_finite], griddata[both_finite], atol=50)  # Pa

    def test_nearest_vertical_level_and_time_tolerance(self) -> None:
        times = pd.date_range("2025-01-01", periods=4, freq="6h").to_numpy()
        levels = np.array([300.0, 250.0, 200.0])
        lats = np.arange(70, 29.9, -0.5)
        lons = np.arange(-60, 0.1, 0.5)
        data = np.broadcast_to(
            (np.arange(4)[:, None, None, None] * 10 + np.arange(3)[None, :, None, None]).astype(float),
            (4, 3, lats.size, lons.size),
        )
        target = xr.DataArray(
            data,
            dims=("time", "pressure_level", "latitude", "longitude"),
            coords={"time": times, "pressure_level": levels, "latitude": lats, "longitude": lons},
        ).chunk(time=2)
        centres = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2025-01-01T05:00", "2025-01-01T13:00", "2025-01-01T19:00", "2025-01-02T03:00"]
                ).tz_localize("UTC"),
                "latitude": [50.0, 45.0, 55.0, 50.0],
                "longitude": [-30.0, -20.0, -40.0, -30.0],
                "level": [290.0, 210.0, 240.0, 300.0],
                "row": [0, 1, 2, 3],
            }
        )
        result = composite_about_points(
            target,
            dd.from_pandas(centres, npartitions=2),
            vert_dim_name="pressure_level",
            vert_column="level",
            carry_columns=["row"],
            time_tolerance=np.timedelta64(1, "h"),
            max_km_extent=200,
            grid_km_spacing=50,
        ).compute()

        # Last row is 3h away from the nearest time step
        np.testing.assert_array_equal(result["row"].to_numpy(), [0, 1, 2])
        np.testing.assert_array_equal(result["pressure_level"].to_numpy(), [300.0, 200.0, 250.0])
        np.testing.assert_array_equal(result["time"].to_numpy(), times[[1, 2, 3]])
        np.testing.assert_allclose(result.isel(y_km=4, x_km=4).to_numpy(), [10.0, 22.0, 31.0])

    def test_raises_if_vert_column_missing(self, load_mslp_data: xr.DataArray) -> None:
        centres = dd.from_pandas(pd.DataFrame({"time": [], "latitude": [], "longitude": []}), npartitions=1)
        with pytest.raises(ValueError, match="vert_dim_name and vert_column"):
            composite_about_points(load_mslp_data.expand_dims(level=[1.0]), centres, vert_dim_name="level")

    def test_raises_if_column_missing(self, load_mslp_data: xr.DataArray) -> None:
        centres = dd.from_pandas(pd.DataFrame({"time": [], "latitude": []}), npartitions=1)
        with pytest.raises(ValueError, match="missing from centres"):
            composite_about_points(load_mslp_data, centres)

    @staticmethod
    def _time_index_field(times: np.ndarray) -> xr.DataArray:
        # Value of the field is the index of the time step, so the composite reveals which time step was selected
        lats = np.arange(70, 29.9, -0.5)
        lons = np.arange(-60, 0.1, 0.5)
        return xr.DataArray(
            np.broadcast_to(np.arange(times.size, dtype=float)[:, None, None], (times.size, lats.size, lons.size)),
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

    def test_bilinear_matches_xarray_interp_at_off_grid_centres(self, load_mslp_data: xr.DataArray) -> None:
        rng = np.random.default_rng(3)
        num_centres = 25
        model_times = load_mslp_data["time"].to_numpy()
        centres = pd.DataFrame(
            {
                # Times between the time steps, so the nearest time step has to be selected
                "time": model_times[0] + pd.to_timedelta(rng.uniform(0, 42, num_centres), unit="h").to_numpy(),
                "latitude": rng.uniform(35, 65, num_centres),
                "longitude": rng.uniform(-75, 15, num_centres),
                "row": np.arange(num_centres),
            }
        )
        km_coords = np.arange(-500, 501, 50.0)
        result = composite_about_points(
            load_mslp_data.chunk(time=3),
            dd.from_pandas(centres, npartitions=3),
            carry_columns=["row"],
            max_km_extent=500,
            grid_km_spacing=50,
        ).compute()

        x_km, y_km = np.meshgrid(km_coords, km_coords)
        for row in result["row"].to_numpy():
            centre = centres.iloc[row]
            ortho = pyproj.Proj(
                proj="ortho", lat_0=centre["latitude"], lon_0=centre["longitude"], units="km", ellps="WGS84"
            )
            target_lons, target_lats = ortho(x_km, y_km, inverse=True)
            expected = (
                load_mslp_data.sel(time=centre["time"], method="nearest")
                .interp(
                    latitude=xr.DataArray(target_lats, dims=("y_km", "x_km")),
                    longitude=xr.DataArray(target_lons, dims=("y_km", "x_km")),
                )
                .to_numpy()
            )
            composited = result.isel(n_centre=int(np.flatnonzero(result["row"].to_numpy() == row)[0]))

            assert composited["time"].to_numpy() == load_mslp_data["time"].sel(time=centre["time"], method="nearest")
            np.testing.assert_array_equal(np.isnan(composited.to_numpy()), np.isnan(expected))
            np.testing.assert_allclose(composited.to_numpy(), expected, rtol=1e-6)

    def test_coordinates_align_with_data_when_computed_separately(self) -> None:
        times = pd.date_range("2025-01-01", periods=6, freq="6h").to_numpy()
        lats = np.arange(70, 29.9, -0.5)
        lons = np.arange(-60, 0.1, 0.5)
        # Linear in latitude, so the centre of each composite is exactly the latitude of its centre
        target = xr.DataArray(
            np.broadcast_to(lats[None, :, None], (times.size, lats.size, lons.size)),
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": lats, "longitude": lons},
        ).chunk(time=2)
        rng = np.random.default_rng(4)
        num_centres = 60
        centres = pd.DataFrame(
            {
                # Many centres share each time step and are spread across partitions. Rows with the same time step
                # are the ones that could be reordered between separate computes
                "time": rng.choice(times, num_centres),
                "latitude": rng.uniform(40, 60, num_centres),
                "longitude": rng.uniform(-50, -10, num_centres),
                "row": np.arange(num_centres),
            }
        )
        result = composite_about_points(
            target,
            dd.from_pandas(centres, npartitions=4),
            carry_columns=["row"],
            max_km_extent=200,
            grid_km_spacing=50,
            max_centres_per_task=7,
        )
        centre_index = result.sizes["y_km"] // 2

        # Each of these is a separate compute, so the shuffle is evaluated separately for each of them
        data_alone = result.data.compute()
        latitude_alone = result["centre_latitude"].data.compute()
        row_alone = result["row"].data.compute()

        np.testing.assert_allclose(data_alone[:, centre_index, centre_index], latitude_alone, atol=1e-9)
        np.testing.assert_array_equal(latitude_alone, centres["latitude"].to_numpy()[row_alone])
        np.testing.assert_array_equal(np.sort(row_alone), np.arange(num_centres))

        # Computing again gives the same result in the same order
        recomputed = result.compute()
        np.testing.assert_array_equal(recomputed.to_numpy(), data_alone)
        np.testing.assert_array_equal(recomputed["row"].to_numpy(), row_alone)

    @pytest.mark.parametrize(
        "centre_times",
        [
            pytest.param(pd.Series(pd.to_datetime(["2025-01-01T06:00"])), id="naive_assumed_utc"),
            pytest.param(pd.Series(pd.to_datetime(["2025-01-01T06:00"]).tz_localize("UTC")), id="utc"),
            pytest.param(pd.Series(pd.to_datetime(["2025-01-01T01:00"]).tz_localize("America/New_York")), id="non_utc"),
            pytest.param(pd.Series(["2025-01-01T07:00+01:00"]), id="string_with_offset"),
            pytest.param(pd.Series(["2025-01-01 06:00"]), id="naive_string"),
        ],
    )
    def test_centre_times_normalised_to_utc(self, centre_times: pd.Series) -> None:
        times = pd.date_range("2025-01-01", periods=4, freq="6h").to_numpy()
        centres = pd.DataFrame({"time": centre_times, "latitude": [50.0], "longitude": [-30.0]})
        result = composite_about_points(
            self._time_index_field(times).chunk(time=2),
            dd.from_pandas(centres, npartitions=1),
            # Only an exact match with 2025-01-01T06:00 UTC is kept
            time_tolerance=np.timedelta64(0, "s"),
            max_km_extent=100,
            grid_km_spacing=50,
        ).compute()

        np.testing.assert_array_equal(result["time"].to_numpy(), times[[1]])
        np.testing.assert_allclose(result.to_numpy(), 1.0)

    def test_time_chunks_without_centres(self, load_mslp_data: xr.DataArray) -> None:
        rng = np.random.default_rng(5)
        num_centres = 12
        model_times = load_mslp_data["time"].to_numpy()
        centres = pd.DataFrame(
            {
                # Only time steps 2 and 5, so 6 of the 8 time chunks are empty
                "time": rng.choice(model_times[[2, 5]], num_centres),
                "latitude": rng.uniform(40, 60, num_centres),
                "longitude": rng.uniform(-60, 0, num_centres),
                "row": np.arange(num_centres),
            }
        )
        composite = functools.partial(
            composite_about_points, carry_columns=["row"], max_km_extent=300, grid_km_spacing=50
        )
        result = composite(load_mslp_data.chunk(time=1), dd.from_pandas(centres, npartitions=3))
        # Only the non-empty time chunks produce blocks
        assert len(result.data.chunks[0]) == 2  # noqa: PLR2004

        result = result.compute()
        expected = composite(load_mslp_data.chunk(time=-1), dd.from_pandas(centres, npartitions=1)).compute()
        assert result.sizes["n_centre"] == num_centres
        result = result.isel(n_centre=np.argsort(result["row"].to_numpy()))
        expected = expected.isel(n_centre=np.argsort(expected["row"].to_numpy()))
        np.testing.assert_array_equal(result["time"].to_numpy(), expected["time"].to_numpy())
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

    def test_raises_if_target_has_unexpected_dims(self, load_mslp_data: xr.DataArray) -> None:
        centres = dd.from_pandas(pd.DataFrame({"time": [], "latitude": [], "longitude": []}), npartitions=1)
        with pytest.raises(ValueError, match="Expected target_data to have dimensions"):
            composite_about_points(load_mslp_data.expand_dims(level=[1.0]), centres)

    @pytest.mark.parametrize("max_centres_per_task", [0, -1])
    def test_raises_if_max_centres_per_task_not_positive(
        self, load_mslp_data: xr.DataArray, max_centres_per_task: int
    ) -> None:
        centres = dd.from_pandas(pd.DataFrame({"time": [], "latitude": [], "longitude": []}), npartitions=1)
        with pytest.raises(ValueError, match="max_centres_per_task must be positive"):
            composite_about_points(load_mslp_data, centres, max_centres_per_task=max_centres_per_task)

    def test_raises_if_no_centres_within_time_range(self, load_mslp_data: xr.DataArray) -> None:
        centres = pd.DataFrame({"time": pd.to_datetime(["2000-01-01T00:00"]), "latitude": [50.0], "longitude": [-30.0]})
        with pytest.raises(ValueError, match="no centres within the time range"):
            composite_about_points(
                load_mslp_data, dd.from_pandas(centres, npartitions=1), time_tolerance=np.timedelta64(1, "h")
            )

    @staticmethod
    def _time_and_level_field() -> xr.DataArray:
        times = pd.date_range("2025-01-01", periods=6, freq="6h").to_numpy()
        levels = np.array([300.0, 250.0, 200.0, 150.0])
        lats = np.arange(70, 29.9, -0.5)
        lons = np.arange(-60, 0.1, 0.5)
        # Depends on every dimension, so selecting the wrong time step or level gives a different value
        data = (
            np.arange(times.size)[:, None, None, None] * 1000.0
            + np.arange(levels.size)[None, :, None, None] * 100.0
            + lats[None, None, :, None]
            + lons[None, None, None, :] / 10
        )
        return xr.DataArray(
            data,
            dims=("time", "pressure_level", "latitude", "longitude"),
            coords={"time": times, "pressure_level": levels, "latitude": lats, "longitude": lons},
        )

    @staticmethod
    def _random_centres_with_level(target: xr.DataArray, num_centres: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        times = target["time"].to_numpy()
        return pd.DataFrame(
            {
                "time": rng.choice(times, num_centres),
                "latitude": rng.uniform(40, 60, num_centres),
                "longitude": rng.uniform(-50, -10, num_centres),
                "level": rng.uniform(140, 310, num_centres),
                "row": np.arange(num_centres),
            }
        )

    def test_group_by_vertical_matches_group_by_time(self) -> None:
        target = self._time_and_level_field()
        centres = self._random_centres_with_level(target, num_centres=40, seed=9)
        composite = functools.partial(
            composite_about_points,
            vert_dim_name="pressure_level",
            vert_column="level",
            carry_columns=["row"],
            max_km_extent=200,
            grid_km_spacing=50,
            max_centres_per_task=7,
        )

        by_vertical = composite(
            target.chunk(pressure_level=1), dd.from_pandas(centres, npartitions=3), group_dim_name="pressure_level"
        ).compute()
        by_time = composite(
            target.chunk(time=2), dd.from_pandas(centres, npartitions=3), group_dim_name="time"
        ).compute()

        assert by_vertical.sizes["n_centre"] == by_time.sizes["n_centre"] == len(centres)
        by_vertical = by_vertical.isel(n_centre=np.argsort(by_vertical["row"].to_numpy()))
        by_time = by_time.isel(n_centre=np.argsort(by_time["row"].to_numpy()))
        for coord in ("time", "pressure_level", "centre_latitude", "centre_longitude"):
            np.testing.assert_array_equal(by_vertical[coord].to_numpy(), by_time[coord].to_numpy())
        np.testing.assert_array_equal(by_vertical.to_numpy(), by_time.to_numpy())

        # The selected time step and level are encoded in the value at the centre of each composite
        centre_index = by_vertical.sizes["y_km"] // 2
        time_index = np.searchsorted(target["time"].to_numpy(), by_vertical["time"].to_numpy())
        level_index = np.argmax(
            target["pressure_level"].to_numpy() == by_vertical["pressure_level"].to_numpy()[:, None], axis=1
        )
        np.testing.assert_allclose(
            by_vertical.isel(y_km=centre_index, x_km=centre_index).to_numpy(),
            time_index * 1000.0
            + level_index * 100.0
            + by_vertical["centre_latitude"].to_numpy()
            + by_vertical["centre_longitude"].to_numpy() / 10,
        )

    def test_group_by_vertical_has_one_block_per_non_empty_vertical_chunk(self) -> None:
        target = self._time_and_level_field()
        centres = self._random_centres_with_level(target, num_centres=30, seed=10)
        # Only levels 250 and 150 are nearest to these centres
        centres["level"] = np.where(np.arange(len(centres)) % 2 == 0, 255.0, 160.0)
        result = composite_about_points(
            target.chunk(pressure_level=1),
            dd.from_pandas(centres, npartitions=2),
            vert_dim_name="pressure_level",
            vert_column="level",
            group_dim_name="pressure_level",
            max_km_extent=100,
            grid_km_spacing=50,
        )
        assert len(result.data.chunks[0]) == 2  # noqa: PLR2004

    def test_group_dim_defaults_to_time(self, load_mslp_data: xr.DataArray) -> None:
        centres = pd.DataFrame(
            {"time": load_mslp_data["time"].to_numpy()[[1, 5]], "latitude": [50.0, 45.0], "longitude": [-30.0, 0.0]}
        )
        composite = functools.partial(composite_about_points, max_km_extent=100, grid_km_spacing=50)
        default = composite(load_mslp_data.chunk(time=2), dd.from_pandas(centres, npartitions=1))
        explicit = composite(
            load_mslp_data.chunk(time=2), dd.from_pandas(centres, npartitions=1), group_dim_name="time"
        )
        assert default.data.chunks == explicit.data.chunks
        xr.testing.assert_identical(default.compute(), explicit.compute())

    @pytest.mark.parametrize(
        ("group_dim_name", "add_ensemble_dim", "vert_dim_name", "match"),
        [
            pytest.param("pressure_level", False, None, "to be a dimension of target_data", id="not_a_dimension"),
            pytest.param("latitude", False, None, "Cannot group by the horizontal dimension", id="latitude"),
            pytest.param("longitude", False, None, "Cannot group by the horizontal dimension", id="longitude"),
            pytest.param("ensemble", True, None, "to be either time_dim_name", id="neither_time_nor_vertical"),
            pytest.param(
                "pressure_level", True, None, "to be either time_dim_name", id="vertical_without_vert_dim_name"
            ),
        ],
    )
    def test_raises_if_group_dim_invalid(
        self,
        load_mslp_data: xr.DataArray,
        group_dim_name: str,
        add_ensemble_dim: bool,
        vert_dim_name: str | None,
        match: str,
    ) -> None:
        target = load_mslp_data
        if add_ensemble_dim:
            target = target.expand_dims({"ensemble": [0], "pressure_level": [500.0]})
        centres = dd.from_pandas(pd.DataFrame({"time": [], "latitude": [], "longitude": []}), npartitions=1)
        with pytest.raises(ValueError, match=match):
            composite_about_points(target, centres, group_dim_name=group_dim_name, vert_dim_name=vert_dim_name)

    def test_class_on_dataset_matches_composite_about_points(self, load_mslp_data: xr.DataArray) -> None:
        centres = pd.DataFrame(
            {
                "time": load_mslp_data["time"].to_numpy()[[1, 5, 6]],
                "latitude": [50.0, 45.0, 55.0],
                "longitude": [-30.0, 0.0, -10.0],
            }
        )
        dataset = xr.Dataset({"mslp": load_mslp_data, "mslp_doubled": 2 * load_mslp_data}).chunk(time=3)
        kwargs: dict[str, Any] = {"max_km_extent": 100, "grid_km_spacing": 50, "max_centres_per_task": 2}
        composite = CompositeAboutPoints.from_centres(dataset, dd.from_pandas(centres, npartitions=2), **kwargs)

        result = dataset.map(composite).compute()
        for name, field in dataset.data_vars.items():
            expected = composite_about_points(field, dd.from_pandas(centres, npartitions=2), **kwargs).compute()
            # Dataset.map() copies the attributes of the dataset's coordinates, e.g. those of time
            xr.testing.assert_equal(result[name], expected.rename(name))

        # A field that is not chunked like the template is rechunked to match it
        xr.testing.assert_equal(composite(load_mslp_data).compute(), result["mslp"])

    def test_class_raises_if_coordinates_differ(self, load_mslp_data: xr.DataArray) -> None:
        centres = pd.DataFrame(
            {"time": load_mslp_data["time"].to_numpy()[[1]], "latitude": [50.0], "longitude": [-30.0]}
        )
        composite = CompositeAboutPoints.from_centres(
            load_mslp_data, dd.from_pandas(centres, npartitions=1), max_km_extent=100, grid_km_spacing=50
        )
        with pytest.raises(ValueError, match="Coordinate 'time' of target_data differs"):
            composite(load_mslp_data.isel(time=slice(1, None)))
        with pytest.raises(ValueError, match="Expected target_data to have dimensions"):
            composite(load_mslp_data.expand_dims(level=[1.0]))
