import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, assert_never

import numpy as np
import pytest
import xarray as xr
from dask.base import is_dask_collection
from scipy.interpolate import RegularGridInterpolator
from xarray import testing as xrt

from rojak.core.calculations import (
    XrAggregationMethod,
    _icao_constants,
    altitude_to_pressure_icao,
    altitude_to_pressure_stratosphere,
    altitude_to_pressure_troposphere,
    apply_data_var_reduction,
    bilinear_interpolation,
    interpolation_on_lat_lon,
    pressure_to_altitude_icao,
    pressure_to_altitude_stratosphere,
    pressure_to_altitude_troposphere,
    pressure_to_altitude_us_std_atm,
)
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from rojak.core.data import CATData
    from rojak.utilities.types import NumpyOrDataArray


def test_pressure_to_altitude_standard_atmosphere() -> None:
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, pressure_to_altitude_us_std_atm(pressures), rtol=1e-3)


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
@pytest.mark.parametrize(
    ("pressures", "converter_method", "descriptor"),
    [
        pytest.param(np.asarray([220]), pressure_to_altitude_troposphere, "less than", id="single value troposphere"),
        pytest.param(
            np.asarray([_icao_constants.tropopause_pressure]),
            pressure_to_altitude_troposphere,
            "less than",
            id="troposphere boundary value",
        ),
        pytest.param(
            np.asarray([220, 230]),
            pressure_to_altitude_troposphere,
            "less than",
            id="one value is above troposphere",
        ),
        pytest.param(
            np.linspace(_icao_constants.tropopause_pressure, 100, 10),
            pressure_to_altitude_troposphere,
            "less than",
            id="multiple value is above troposphere",
        ),
        pytest.param(
            np.linspace(1013, _icao_constants.tropopause_pressure, 20),
            pressure_to_altitude_troposphere,
            "less than",
            id="final value is tropopause boundary",
        ),
        pytest.param(
            np.asarray([230]),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="single value stratosphere",
        ),
        pytest.param(
            np.asarray([220, 230]),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="one value is below stratosphere",
        ),
        pytest.param(
            np.linspace(250, 500, 10),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="multiple value is below stratosphere",
        ),
        pytest.param(
            np.linspace(_icao_constants.tropopause_pressure, 500, 10),
            pressure_to_altitude_stratosphere,
            "greater than",
            id="multiple value is below stratosphere (with boundary)",
        ),
    ],
)
def test_pressure_to_altitude_fails_checks(
    pressures: "NumpyOrDataArray",
    converter_method: Callable,
    descriptor: str,
    wrap_in_data_array: bool,
    is_2d: bool,
) -> None:
    matches: str = (
        f"Attempting to convert pressure to altitude for troposphere with pressure {descriptor} tropopause pressure"
    )
    if is_2d:
        pressures = pressures * np.ones((pressures.size, pressures.size))
    if wrap_in_data_array:
        pressures = xr.DataArray(pressures)
    with pytest.raises(ValueError, match=matches):
        converter_method(pressures)


# Values from Metric Table I in section 5 of NACA3182
_TROPOSPHERE_ALTITUDES_FROM_TABLE = np.asarray([0, 2000, 3000, 4000, 4500, 6000, 6300, 7000, 8500, 8100, 9150, 10350])
_TROPOSPHERE_PRESSURES_FROM_TABLE = np.asarray(
    [1013.25, 794.95, 701.08, 616.40, 577.28, 471.81, 452.72, 410.61, 330.99, 350.88, 300.62, 250.50]
)
_STRATOSPHERE_ALTITUDES_FROM_TABLE = np.asarray([11000, 11800, 12600, 13600, 14800, 16200])
_STRATOSPHERE_PRESSURES_FROM_TABLE = np.asarray([226.32, 199.50, 175.85, 150.20, 124.30, 99.68])
_ICAO_ALTITUDES_FROM_TABLE = np.concatenate([_TROPOSPHERE_ALTITUDES_FROM_TABLE, _STRATOSPHERE_ALTITUDES_FROM_TABLE])
_ICAO_PRESSURES_FROM_TABLE = np.concatenate([_TROPOSPHERE_PRESSURES_FROM_TABLE, _STRATOSPHERE_PRESSURES_FROM_TABLE])
# Table pressures are given to 2 d.p. and the ICAO constants are rounded, so an exact match is not expected
_TABLE_PRESSURE_RTOL: float = 2e-4


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_troposphere_and_vice_versa(wrap_in_data_array: bool, is_2d: bool) -> None:
    pressure = _TROPOSPHERE_PRESSURES_FROM_TABLE
    altitude_from_table = _TROPOSPHERE_ALTITUDES_FROM_TABLE

    if is_2d:
        pressure = pressure.reshape((2, 6))
        altitude_from_table = altitude_from_table.reshape((2, 6))

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    computed_altitude = pressure_to_altitude_troposphere(pressure)
    computed_pressure = altitude_to_pressure_troposphere(altitude_from_table)
    np.testing.assert_allclose(computed_pressure, pressure, rtol=1e-4)
    # rtol based on altitude -> pressure as table is from altitude -> pressure
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)
    if not is_2d:
        if wrap_in_data_array:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure).values, computed_altitude)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure), computed_altitude)

    # Test that we get back approximately the same thing when passed through inverses
    np.testing.assert_allclose(altitude_to_pressure_troposphere(computed_altitude), pressure)
    np.testing.assert_allclose(pressure_to_altitude_troposphere(computed_pressure), altitude_from_table)


def test_pressure_to_altitude_troposphere_equiv_to_wallace() -> None:
    pressure = np.asarray([1013.25, 794.95, 701.08, 616.40, 577.28, 478.81, 410.61, 330.99, 350.88, 300.62, 250.50])
    np.testing.assert_allclose(
        pressure_to_altitude_troposphere(pressure),
        pressure_to_altitude_us_std_atm(pressure),
        rtol=1e-3,
    )


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_stratosphere(is_2d: bool, wrap_in_data_array: bool) -> None:
    pressure = _STRATOSPHERE_PRESSURES_FROM_TABLE
    altitude_from_table = _STRATOSPHERE_ALTITUDES_FROM_TABLE

    if is_2d:
        pressure = pressure.reshape((2, 3))
        altitude_from_table = altitude_from_table.reshape((2, 3))

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    computed_altitude = pressure_to_altitude_stratosphere(pressure)
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)

    if not is_2d:
        if wrap_in_data_array:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure).values, computed_altitude)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            np.testing.assert_equal(pressure_to_altitude_icao(pressure), computed_altitude)


@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_pressure_to_altitude_icao(wrap_in_data_array) -> None:
    pressure = _ICAO_PRESSURES_FROM_TABLE
    altitude_from_table = _ICAO_ALTITUDES_FROM_TABLE

    if wrap_in_data_array:
        pressure = xr.DataArray(pressure)
        altitude_from_table = xr.DataArray(altitude_from_table)

    converted_pressure = pressure_to_altitude_icao(pressure)
    np.testing.assert_equal(np.round(converted_pressure, decimals=-1), altitude_from_table.data)
    np.testing.assert_allclose(converted_pressure, altitude_from_table, rtol=1e-4)


def test_pressure_to_altitude_icao_era5_data(make_dummy_cat_data) -> None:
    dummy_data = make_dummy_cat_data({"pressure_level": [300.62, 250.50, 226.32, 199.50]})
    altitude_from_table = np.asarray([9150, 10350, 11000, 11800])
    computed_altitude = pressure_to_altitude_icao(dummy_data["pressure_level"])
    np.testing.assert_equal(np.round(computed_altitude, decimals=-1), altitude_from_table)
    np.testing.assert_allclose(computed_altitude, altitude_from_table, rtol=1e-4)


@pytest.mark.parametrize("is_2d", [True, False])
@pytest.mark.parametrize("wrap_in_data_array", [True, False])
def test_altitude_to_pressure_stratosphere(is_2d: bool, wrap_in_data_array: bool) -> None:
    altitude: NumpyOrDataArray = _STRATOSPHERE_ALTITUDES_FROM_TABLE
    pressure_from_table: NumpyOrDataArray = _STRATOSPHERE_PRESSURES_FROM_TABLE

    if is_2d:
        altitude = altitude.reshape((2, 3))
        pressure_from_table = pressure_from_table.reshape((2, 3))

    if wrap_in_data_array:
        altitude = xr.DataArray(altitude)
        pressure_from_table = xr.DataArray(pressure_from_table)

    computed_pressure = altitude_to_pressure_stratosphere(altitude)
    np.testing.assert_allclose(computed_pressure, pressure_from_table, rtol=_TABLE_PRESSURE_RTOL)

    # Test that we get back the same thing when passed through the inverse
    np.testing.assert_allclose(pressure_to_altitude_stratosphere(computed_pressure), altitude)
    np.testing.assert_allclose(
        altitude_to_pressure_stratosphere(pressure_to_altitude_stratosphere(pressure_from_table)), pressure_from_table
    )


def test_altitude_to_pressure_is_continuous_at_tropopause() -> None:
    np.testing.assert_allclose(
        altitude_to_pressure_troposphere(np.asarray([_icao_constants.tropopause_height])),
        altitude_to_pressure_stratosphere(np.asarray([_icao_constants.tropopause_height])),
    )


class TestAltitudeToPressureIcao:
    altitude: np.ndarray = _ICAO_ALTITUDES_FROM_TABLE
    pressure_from_table: np.ndarray = _ICAO_PRESSURES_FROM_TABLE

    @pytest.mark.parametrize("wrap_in_data_array", [True, False])
    def test_matches_table_values(self, wrap_in_data_array: bool) -> None:
        altitude = xr.DataArray(self.altitude) if wrap_in_data_array else self.altitude
        computed_pressure = altitude_to_pressure_icao(altitude)
        assert type(computed_pressure) is type(altitude)
        np.testing.assert_allclose(computed_pressure, self.pressure_from_table, rtol=_TABLE_PRESSURE_RTOL)

    @pytest.mark.parametrize("wrap_in_data_array", [True, False])
    def test_inverse_of_pressure_to_altitude_icao(self, wrap_in_data_array: bool) -> None:
        pressure = xr.DataArray(self.pressure_from_table) if wrap_in_data_array else self.pressure_from_table
        np.testing.assert_allclose(altitude_to_pressure_icao(pressure_to_altitude_icao(pressure)), pressure)

        altitude = np.linspace(0, 20000, 101)
        np.testing.assert_allclose(pressure_to_altitude_icao(altitude_to_pressure_icao(altitude)), altitude)

    @pytest.mark.parametrize(
        ("altitude", "layer_converter"),
        [
            pytest.param(_TROPOSPHERE_ALTITUDES_FROM_TABLE, altitude_to_pressure_troposphere, id="only_troposphere"),
            pytest.param(_STRATOSPHERE_ALTITUDES_FROM_TABLE, altitude_to_pressure_stratosphere, id="only_stratosphere"),
        ],
    )
    def test_equivalent_to_layer_converter(self, altitude: np.ndarray, layer_converter: Callable) -> None:
        np.testing.assert_allclose(altitude_to_pressure_icao(altitude), layer_converter(altitude))

    def test_uses_correct_layer_either_side_of_tropopause(self) -> None:
        below = _icao_constants.tropopause_height - 1
        altitude = np.asarray([below, _icao_constants.tropopause_height])
        computed_pressure = altitude_to_pressure_icao(altitude)
        np.testing.assert_allclose(computed_pressure[0], altitude_to_pressure_troposphere(np.asarray([below])))
        np.testing.assert_allclose(
            computed_pressure[1], altitude_to_pressure_stratosphere(np.asarray([_icao_constants.tropopause_height]))
        )

    def test_pressure_decreases_with_altitude(self) -> None:
        computed_pressure = altitude_to_pressure_icao(np.linspace(0, 30000, 301))
        assert (np.diff(computed_pressure) < 0).all()

    def test_no_invalid_values_high_in_stratosphere(self) -> None:
        # The troposphere formula is invalid above ~44 km, so it must not be evaluated for these altitudes
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            computed_pressure = altitude_to_pressure_icao(np.asarray([1000.0, 50000.0, 80000.0]))
        assert np.isfinite(computed_pressure).all()
        np.testing.assert_allclose(
            computed_pressure[1:], altitude_to_pressure_stratosphere(np.asarray([50000.0, 80000.0]))
        )

    def test_integer_altitudes_give_float_pressures(self) -> None:
        integer_altitude = np.asarray([0, 5000, 11000, 15000])
        computed_pressure = altitude_to_pressure_icao(integer_altitude)
        assert np.issubdtype(computed_pressure.dtype, np.floating)
        np.testing.assert_allclose(computed_pressure, altitude_to_pressure_icao(integer_altitude.astype(float)))

    def test_does_not_modify_input(self) -> None:
        altitude = np.asarray([0.0, 5000.0, 11000.0, 15000.0])
        original = altitude.copy()
        altitude_to_pressure_icao(altitude)
        np.testing.assert_array_equal(altitude, original)

    def test_nan_propagates(self) -> None:
        computed_pressure = altitude_to_pressure_icao(np.asarray([0.0, np.nan, 15000.0]))
        assert np.isnan(computed_pressure[1])
        assert np.isfinite(computed_pressure[[0, 2]]).all()

    def test_multidimensional(self) -> None:
        altitude = self.altitude.reshape((3, 6))
        computed_pressure = altitude_to_pressure_icao(altitude)
        assert computed_pressure.shape == altitude.shape
        np.testing.assert_allclose(computed_pressure, altitude_to_pressure_icao(self.altitude).reshape((3, 6)))

    def test_data_array_retains_coords_attrs_and_name(self) -> None:
        altitude = xr.DataArray(
            self.altitude,
            dims=["n_centre"],
            coords={"n_centre": np.arange(self.altitude.size), "time": ("n_centre", np.arange(self.altitude.size))},
            attrs={"description": "altitude of each centre"},
            name="altitude",
        )
        computed_pressure = altitude_to_pressure_icao(altitude)
        assert isinstance(computed_pressure, xr.DataArray)
        assert computed_pressure.dims == altitude.dims
        assert computed_pressure.name == altitude.name
        assert computed_pressure.attrs == altitude.attrs
        xrt.assert_identical(computed_pressure.coords.to_dataset(), altitude.coords.to_dataset())

    @pytest.mark.parametrize("wrap_in_data_array", [True, False])
    def test_dask_stays_lazy(self, wrap_in_data_array: bool) -> None:
        chunked_altitude = xr.DataArray(self.altitude).chunk({"dim_0": 5})
        altitude = chunked_altitude if wrap_in_data_array else chunked_altitude.data
        computed_pressure = altitude_to_pressure_icao(altitude)

        assert is_dask_collection(computed_pressure)
        np.testing.assert_allclose(np.asarray(computed_pressure), altitude_to_pressure_icao(self.altitude), rtol=1e-12)


def linear_function(x_vals, y_vals):
    return x_vals + y_vals


def quadratic_function(x_vals, y_vals):
    return x_vals * x_vals + y_vals * y_vals


@pytest.mark.parametrize(("ff", "is_ff_linear"), [(linear_function, True), (quadratic_function, False)])
@pytest.mark.parametrize(
    ("x_slice", "y_slice", "interpolation_point"),
    [
        (slice(2), slice(2), (-1, -1)),
        (slice(2), slice(1, None), (1, -1)),
        (slice(1, None), slice(2), (-1, 1)),
        (slice(1, None), slice(1, None), (1, 1)),
    ],
)
def test_bilinear_interpolation(x_slice, y_slice, interpolation_point, ff, is_ff_linear) -> None:
    x = np.asarray([-2, 0, 2])
    y = np.asarray([-2, 0, 2])
    xx, yy = np.meshgrid(x, y)

    rgi = RegularGridInterpolator((x, y), ff(xx, yy))
    with pytest.deprecated_call():
        interpolated = bilinear_interpolation(
            x[x_slice],
            y[y_slice],
            ff(*np.meshgrid(x[x_slice], y[y_slice])),
            Coordinate(*interpolation_point),
        )
    assert rgi(interpolation_point) == interpolated
    if is_ff_linear:
        assert interpolated == ff(interpolation_point[0], interpolation_point[1])


@pytest.mark.skip(reason="interpolation_on_lat_lon() method has been deprecated")
def test_interpolation_on_lat_lon_apply_ufunc_equiv_to_slice(load_cat_data, client) -> None:
    cat_data: CATData = load_cat_data(None, with_chunks=True)

    num_interp_points: int = 400
    rng = np.random.default_rng()
    longitude_point: np.ndarray = rng.uniform(-180, 179.75, num_interp_points)
    latitude_points: np.ndarray = rng.uniform(-90, 90, num_interp_points)
    interpolation_points: xr.DataArray = xr.DataArray(
        data=np.stack((longitude_point, latitude_points), axis=1),
        dims=["points", "xy"],
    )

    apply_on = cat_data.v_wind()
    interpolated_to: xr.DataArray = interpolation_on_lat_lon(apply_on, interpolation_points)
    lat_coord = apply_on["latitude"]
    lon_coord = apply_on["longitude"]

    for time_index in range(apply_on["time"].size):
        for pressure_index in range(apply_on["pressure_level"].size):
            lat_lon_slice = apply_on.isel(time=time_index, pressure_level=pressure_index)
            interp_on_slice = RegularGridInterpolator(
                (lon_coord, lat_coord),
                lat_lon_slice.to_numpy().T,
                method="linear",
            )(interpolation_points.values)
            np.testing.assert_allclose(
                interpolated_to.isel(time=time_index, pressure_level=pressure_index),
                interp_on_slice,
            )


def test_interpolation_on_lat_lon_to_grid_points(load_cat_data, client) -> None:
    cat_data: CATData = load_cat_data(None, with_chunks=True)
    apply_on = cat_data.v_wind()
    lat_coord = apply_on["latitude"]
    lon_coord = apply_on["longitude"]
    llon, llat = np.meshgrid(lon_coord, lat_coord)

    interpolation_points: xr.DataArray = xr.DataArray(
        data=np.stack((llon.ravel(), llat.ravel()), axis=1),
        dims=["points", "xy"],
    )

    with pytest.deprecated_call():
        interpolated_to: xr.DataArray = interpolation_on_lat_lon(apply_on, interpolation_points)
        np.testing.assert_array_equal(
            apply_on.stack(z=["latitude", "longitude"]),
            interpolated_to,
        )


class TestApplyDataVarsReduction:
    default_dim_name: str = "ensemble"
    default_var_name: str = "ensemble"

    @pytest.mark.parametrize("agg_method", [item.value for item in XrAggregationMethod])
    def test_retrives_correct_method(  # noqa: PLR0912
        self, make_dummy_cat_data, agg_method: XrAggregationMethod
    ) -> None:
        dummy_data: xr.Dataset = make_dummy_cat_data({})
        new_dim_name: str = "new_dim_name"
        reduced = apply_data_var_reduction(dummy_data, agg_method, append=False, new_dim=new_dim_name)
        match agg_method:
            case XrAggregationMethod.ALL:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).all(dim=new_dim_name), reduced)
            case XrAggregationMethod.ANY:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).any(dim=new_dim_name), reduced)
            case XrAggregationMethod.ARGMAX:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).argmax(dim=new_dim_name), reduced)
            case XrAggregationMethod.ARGMIN:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).argmin(dim=new_dim_name), reduced)
            case XrAggregationMethod.COUNT:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).count(dim=new_dim_name), reduced)
            case XrAggregationMethod.IDX_MAX:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).idxmax(dim=new_dim_name), reduced)
            case XrAggregationMethod.IDX_MIN:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).idxmin(dim=new_dim_name), reduced)
            case XrAggregationMethod.MAX:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).max(dim=new_dim_name), reduced)
            case XrAggregationMethod.MIN:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).min(dim=new_dim_name), reduced)
            case XrAggregationMethod.MEAN:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).mean(dim=new_dim_name), reduced)
            case XrAggregationMethod.MEDIAN:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).median(dim=new_dim_name), reduced)
            case XrAggregationMethod.PROD:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).prod(dim=new_dim_name), reduced)
            case XrAggregationMethod.SUM:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).sum(dim=new_dim_name), reduced)
            case XrAggregationMethod.STDDEV:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).std(dim=new_dim_name), reduced)
            case XrAggregationMethod.VARIANCE:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).var(dim=new_dim_name), reduced)
            case XrAggregationMethod.CUM_PROD:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).cumprod(dim=new_dim_name), reduced)
            case XrAggregationMethod.CUM_SUM:
                np.testing.assert_array_equal(dummy_data.to_array(dim=new_dim_name).cumsum(dim=new_dim_name), reduced)
            case _ as unreachable:
                assert_never(unreachable)

    def test_append_true_dataset(self, make_dummy_cat_data) -> None:
        dummy_data: xr.Dataset = make_dummy_cat_data({})
        reduced = apply_data_var_reduction(dummy_data, XrAggregationMethod.MEAN, append=True)

        assert isinstance(reduced, xr.Dataset)
        assert self.default_var_name in reduced.data_vars

        np.testing.assert_array_equal(
            dummy_data.to_array(dim=self.default_var_name).mean(dim=self.default_var_name),
            reduced[self.default_var_name],
        )

        # Check existing coords, dims and data vars are still present in data set
        assert set(dummy_data.coords) == set(reduced.coords)
        assert set(dummy_data.dims) == set(reduced.dims)
        assert set(dummy_data.data_vars.keys()).issubset(reduced.data_vars.keys())

    def test_append_false_var_name_change_name(self, make_dummy_cat_data) -> None:
        dummy_data: xr.Dataset = make_dummy_cat_data({})
        new_var_name: str = "blah"
        reduced: xr.DataArray = apply_data_var_reduction(
            dummy_data, XrAggregationMethod.PROD, append=False, var_name=new_var_name
        )

        assert reduced.name == new_var_name
        assert self.default_dim_name not in reduced.dims

        # Check existing coords and dims are still present in data set
        assert set(dummy_data.coords) == set(reduced.coords)
        assert set(dummy_data.dims) == set(reduced.dims)

    @pytest.mark.parametrize("is_append", [True, False])
    def test_append_true_custom_names(self, make_dummy_cat_data, is_append: bool) -> None:
        dummy_data: xr.Dataset = make_dummy_cat_data({})
        new_dim_name: str = "custom_dim_name"
        new_var_name: str = "custom_var_name"
        reduced = apply_data_var_reduction(
            dummy_data, XrAggregationMethod.COUNT, append=is_append, new_dim=new_dim_name, var_name=new_var_name
        )

        if is_append:
            assert isinstance(reduced, xr.Dataset)
            assert new_dim_name not in reduced.data_vars
            assert new_dim_name not in reduced.dims

            assert new_var_name in reduced.data_vars

        else:
            assert isinstance(reduced, xr.DataArray)
            assert reduced.name == new_var_name

    def test_skipna_kwarg(self, make_dummy_cat_data):
        # Add NaN values
        data_with_nan = make_dummy_cat_data({}).copy()
        data_with_nan["temperature"].values[0, 0] = np.nan

        result_skipna_true = apply_data_var_reduction(
            data_with_nan,
            XrAggregationMethod.MEAN,
            append=False,
            skipna=True,
        )
        result_skipna_false = apply_data_var_reduction(
            data_with_nan,
            XrAggregationMethod.MEAN,
            skipna=False,
            append=False,
        )

        assert not np.allclose(
            result_skipna_true.values,
            result_skipna_false.values,
            equal_nan=True,
        )

    def test_single_variable_dataset(self, make_dummy_cat_data) -> None:
        temperature = make_dummy_cat_data({})["temperature"]
        single_variable_dataset = xr.Dataset({"temp": temperature})
        result = apply_data_var_reduction(
            single_variable_dataset,
            XrAggregationMethod.MEDIAN,
            append=False,
        )
        assert isinstance(result, xr.DataArray)

    def test_empty_dataset(self) -> None:
        with pytest.raises(IndexError):
            apply_data_var_reduction(xr.Dataset(), XrAggregationMethod.MEAN)
