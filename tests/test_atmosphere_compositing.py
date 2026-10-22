from typing import TYPE_CHECKING

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.base import is_dask_collection

from rojak.atmosphere._compositing_kernels import project_about_centre
from rojak.atmosphere.compositing import (
    _project_data_about_centre,
    _project_data_about_extrema,
    centres_from_dataframe,
    composite_about_centre,
    composite_about_extrema,
)
from rojak.atmosphere.regions import ExtremaKind, identify_and_stack_circular_extrema
from tests.test_atmosphere_compositing_kernels import expected_on_km_grid, smooth_field_on_sphere

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


class TestProjectDataAboutCentre:
    lats: np.ndarray = np.linspace(65.0, 25.0, 81)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 221)

    def test_equivalent_to_project_about_centre(self) -> None:
        field = np.random.default_rng(42).random((self.lats.size, self.lons.size))
        km_coords = np.arange(-500, 501, 50)
        result = _project_data_about_centre(
            field, 45.0, -40.0, lats=self.lats, lons=self.lons, max_km_extent=500, grid_km_spacing=50
        )
        expected = project_about_centre(field, km_coords, 45.0, -40.0, self.lats, self.lons, 0.5, km_coords.size)
        np.testing.assert_array_equal(result, expected)

    def test_accepts_zero_dim_arrays_as_centre(self) -> None:
        # When vectorized by apply_ufunc, the centres are passed in as 0D arrays
        field = np.ones((self.lats.size, self.lons.size))
        result = _project_data_about_centre(
            field,
            np.array(45.0),  # pyright: ignore[reportArgumentType]
            np.array(-40.0),  # pyright: ignore[reportArgumentType]
            lats=self.lats,
            lons=self.lons,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        n_km = len(np.arange(-500, 501, 50))
        assert result.shape == (n_km, n_km)
        np.testing.assert_allclose(result, 1.0)


@pytest.fixture
def centres_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-12-29T01:00", "2025-12-29T05:00", "2025-12-29T06:00"]),
            "latitude": [50.0, 45.0, 40.0],
            "longitude": [-40.0, -30.0, -20.0],
            "altitude": [10000.0, 11000.0, 9000.0],
        }
    )


class TestCentresFromDataframe:
    def test_dims_values_and_coords(self, centres_dataframe: pd.DataFrame) -> None:
        centres = centres_from_dataframe(centres_dataframe)

        assert isinstance(centres, xr.DataArray)
        assert centres.dims == ("n_centre",)
        assert centres.name == "n_centre"
        np.testing.assert_array_equal(centres.to_numpy(), np.arange(len(centres_dataframe)))
        np.testing.assert_array_equal(centres["n_centre"].to_numpy(), np.arange(len(centres_dataframe)))
        for col_name in ("time", "latitude", "longitude", "altitude"):
            assert centres[col_name].dims == ("n_centre",)
            np.testing.assert_array_equal(centres[col_name].to_numpy(), centres_dataframe[col_name].to_numpy())

    def test_extra_columns_are_dropped(self, centres_dataframe: pd.DataFrame) -> None:
        centres = centres_from_dataframe(centres_dataframe.assign(other=[1, 2, 3]))
        assert "other" not in centres.coords

    def test_custom_names(self, centres_dataframe: pd.DataFrame) -> None:
        renamed = centres_dataframe.rename(
            columns={"time": "t", "latitude": "lat", "longitude": "lon", "altitude": "alt"}
        )
        centres = centres_from_dataframe(
            renamed,
            time_col_name="t",
            lat_col_name="lat",
            lon_col_name="lon",
            alt_col_name="alt",
            num_centre_dim="obs",
        )
        assert centres.dims == ("obs",)
        assert {"obs", "t", "lat", "lon", "alt"} == set(centres.coords)

    @pytest.mark.parametrize("missing_column", ["time", "latitude", "longitude", "altitude"])
    def test_raises_if_column_missing(self, centres_dataframe: pd.DataFrame, missing_column: str) -> None:
        with pytest.raises(ValueError, match=f"Expected columns \\['{missing_column}'\\] to be present"):
            centres_from_dataframe(centres_dataframe.drop(columns=missing_column))

    def test_empty_dataframe(self, centres_dataframe: pd.DataFrame) -> None:
        centres = centres_from_dataframe(centres_dataframe.iloc[:0])
        assert centres.sizes == {"n_centre": 0}

    def test_dask_dataframe_stays_lazy(self, centres_dataframe: pd.DataFrame) -> None:
        centres = centres_from_dataframe(dd.from_pandas(centres_dataframe, npartitions=2))

        assert centres.sizes == {"n_centre": len(centres_dataframe)}
        assert is_dask_collection(centres.variable)
        for col_name in ("time", "latitude", "longitude", "altitude"):
            assert is_dask_collection(centres[col_name].variable)

    def test_dask_dataframe_equivalent_to_pandas(self, centres_dataframe: pd.DataFrame) -> None:
        from_dask = centres_from_dataframe(dd.from_pandas(centres_dataframe, npartitions=2))
        from_pandas = centres_from_dataframe(centres_dataframe)
        xr.testing.assert_identical(from_dask.compute(), from_pandas)


class TestCompositeAboutCentre:
    lats: np.ndarray = np.linspace(65.0, 25.0, 81)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 221)
    time: np.ndarray = np.array(["2025-12-29T00:00", "2025-12-29T06:00"], dtype="datetime64[ns]")
    # Offset between each time step so that it is possible to identify which time step has been selected
    time_offset: float = 1000.0

    @pytest.fixture
    def dummy_target_data(self) -> xr.DataArray:
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        data = np.stack([lat_grid + lon_grid + self.time_offset * i for i in range(len(self.time))], axis=0)
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={"time": self.time, "latitude": self.lats, "longitude": self.lons},
        )

    @pytest.fixture
    def dummy_centres(self, centres_dataframe: pd.DataFrame) -> xr.DataArray:
        return centres_from_dataframe(centres_dataframe)

    def test_raises_if_lat_lon_time_missing_from_target(
        self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray
    ) -> None:
        for dim in ("time", "latitude", "longitude"):
            with pytest.raises(ValueError, match="do not contain the time, latitude and longitude dims"):
                composite_about_centre(dummy_target_data.rename({dim: "wrong"}), dummy_centres)

    def test_raises_if_vert_dim_not_in_target(
        self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray
    ) -> None:
        with pytest.raises(ValueError, match="Expected 'pressure_level' to be present in to_composite"):
            composite_about_centre(dummy_target_data, dummy_centres, vert_dim_name="pressure_level")

    def test_raises_if_centre_dim_wrong(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="Expected centres to only have the dimension 'n_centre'"):
            composite_about_centre(dummy_target_data, dummy_centres.rename({"n_centre": "wrong"}))

    def test_raises_if_centres_not_1d(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="Expected centres to only have the dimension 'n_centre'"):
            composite_about_centre(dummy_target_data, dummy_centres.expand_dims("extra"))

    @pytest.mark.parametrize("coord_name", ["time", "latitude", "longitude"])
    def test_raises_if_coord_missing_from_centres(
        self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray, coord_name: str
    ) -> None:
        with pytest.raises(ValueError, match=f"Expected '{coord_name}' to be a coordinate of centres"):
            composite_about_centre(dummy_target_data, dummy_centres.drop_vars(coord_name))

    def test_output_dims_and_coords(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        result = composite_about_centre(dummy_target_data, dummy_centres, max_km_extent=500, grid_km_spacing=50)

        expected_km = np.arange(-500, 501, 50)
        assert result.dims == ("n_centre", "y_km", "x_km")
        assert result.shape == (dummy_centres.size, expected_km.size, expected_km.size)
        np.testing.assert_array_equal(result["n_centre"].values, dummy_centres["n_centre"].values)
        np.testing.assert_array_equal(result["y_km"].values, expected_km)
        np.testing.assert_array_equal(result["x_km"].values, expected_km)
        # Coordinates of the centres are retained so that the source of each composite can be identified
        for coord_name in ("time", "altitude"):
            assert result[coord_name].dims == ("n_centre",)
            np.testing.assert_array_equal(result[coord_name].values, dummy_centres[coord_name].values)

    def test_default_km_grid(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        result = composite_about_centre(dummy_target_data, dummy_centres)
        expected_km = np.arange(-2000, 2001, 25)
        np.testing.assert_array_equal(result["y_km"].values, expected_km)
        np.testing.assert_array_equal(result["x_km"].values, expected_km)

    def test_centre_value_from_nearest_time(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        result = composite_about_centre(dummy_target_data, dummy_centres, max_km_extent=500, grid_km_spacing=50)

        # Centres are on grid points, so the value at the centre of the km grid is exactly the value at the centre
        # 01:00 -> 00:00, 05:00 -> 06:00, 06:00 -> 06:00
        nearest_time_index = np.array([0, 1, 1])
        expected = (
            dummy_centres["latitude"].to_numpy()
            + dummy_centres["longitude"].to_numpy()
            + self.time_offset * nearest_time_index
        )
        np.testing.assert_allclose(result.sel(x_km=0, y_km=0).values, expected)

    def test_exact_time_selection_raises_if_no_match(
        self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray
    ) -> None:
        with pytest.raises(KeyError):
            composite_about_centre(dummy_target_data, dummy_centres, time_sel_method=None)

    def test_exact_time_selection(self, dummy_target_data: xr.DataArray, centres_dataframe: pd.DataFrame) -> None:
        on_time_steps = centres_dataframe.assign(time=pd.to_datetime(["2025-12-29T00:00"] * 2 + ["2025-12-29T06:00"]))
        result = composite_about_centre(
            dummy_target_data,
            centres_from_dataframe(on_time_steps),
            time_sel_method=None,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        expected = on_time_steps["latitude"] + on_time_steps["longitude"] + self.time_offset * np.array([0, 0, 1])
        np.testing.assert_allclose(result.sel(x_km=0, y_km=0).values, expected)

    def test_with_vertical_dimension(self, dummy_target_data: xr.DataArray, dummy_centres: xr.DataArray) -> None:
        pressure = np.array([500.0, 850.0])
        # Put vertical dim last to check that it is transposed correctly
        target_4d = xr.concat(
            [dummy_target_data, dummy_target_data * 2], dim=xr.DataArray(pressure, dims="pressure_level")
        ).transpose("time", "latitude", "longitude", "pressure_level")
        result = composite_about_centre(
            target_4d, dummy_centres, vert_dim_name="pressure_level", max_km_extent=500, grid_km_spacing=50
        )

        assert result.dims == ("n_centre", "pressure_level", "y_km", "x_km")
        np.testing.assert_array_equal(result["pressure_level"].values, pressure)
        np.testing.assert_allclose(result.isel(pressure_level=1).values, result.isel(pressure_level=0).to_numpy() * 2)

    def test_matches_composite_about_extrema(self, dummy_target_data: xr.DataArray) -> None:
        # Compositing about a point should be the same as compositing about an extremum at the same point
        lat_index, lon_index = 30, 120
        extrema = np.zeros((1, self.lats.size, self.lons.size), dtype=np.int8)
        extrema[0, lat_index, lon_index] = 2
        extrema_data = xr.DataArray(
            extrema,
            dims=["n_extrema", "latitude", "longitude"],
            coords={
                "n_extrema": [0],
                "time": ("n_extrema", self.time[:1]),
                "latitude": self.lats,
                "longitude": self.lons,
            },
        )
        centres = centres_from_dataframe(
            pd.DataFrame(
                {
                    "time": self.time[:1],
                    "latitude": self.lats[lat_index : lat_index + 1],
                    "longitude": self.lons[lon_index : lon_index + 1],
                    "altitude": [10000.0],
                }
            )
        )
        about_extrema = composite_about_extrema(dummy_target_data, extrema_data, max_km_extent=500, grid_km_spacing=50)
        about_centre = composite_about_centre(dummy_target_data, centres, max_km_extent=500, grid_km_spacing=50)
        np.testing.assert_allclose(about_centre.values, about_extrema.values)

    def test_across_antimeridian(self) -> None:
        lats = np.arange(-90, 90.1, 0.5)
        lons = np.arange(-180, 180, 0.5)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        target = xr.DataArray(
            smooth_field_on_sphere(lat_grid, lon_grid)[np.newaxis, ...],
            dims=["time", "latitude", "longitude"],
            coords={"time": self.time[:1], "latitude": lats, "longitude": lons},
        )
        centres = centres_from_dataframe(
            pd.DataFrame({"time": self.time[:1], "latitude": [50.0], "longitude": [179.0], "altitude": [10000.0]})
        )
        result = composite_about_centre(target, centres, max_km_extent=1000, grid_km_spacing=50)

        np.testing.assert_allclose(
            result.isel(n_centre=0).values, expected_on_km_grid(50.0, 179.0, result["x_km"].values), atol=1e-3
        )

    def test_dask_inputs_stay_lazy_and_match_eager(
        self, dummy_target_data: xr.DataArray, centres_dataframe: pd.DataFrame
    ) -> None:
        eager = composite_about_centre(
            dummy_target_data, centres_from_dataframe(centres_dataframe), max_km_extent=500, grid_km_spacing=50
        )
        lazy = composite_about_centre(
            dummy_target_data.chunk({"time": 1}),
            centres_from_dataframe(dd.from_pandas(centres_dataframe, npartitions=2)),
            max_km_extent=500,
            grid_km_spacing=50,
        )

        assert is_dask_collection(lazy.variable)
        xr.testing.assert_allclose(lazy.compute(), eager)


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
