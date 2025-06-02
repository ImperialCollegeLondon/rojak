from contextlib import nullcontext
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from pyproj import Geod

from rojak.core import derivatives

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from rojak.utilities.types import ArrayLike


@pytest.mark.parametrize(
    "values, coordinate, expected",
    [
        pytest.param([8], None, True, id="No coordinate in degrees"),
        pytest.param([np.pi * 2, -np.pi * 2], None, False, id="No coordinate on boundary"),
        pytest.param([2 * np.pi, 0, -2 * np.pi], None, False, id="Longitude boundary"),
        pytest.param([np.pi * (1 / 2), 0, -np.pi * (1 / 2)], None, False, id="Latitude boundary"),
        pytest.param(
            [2 * np.pi + 0.01, 0, -2 * np.pi + 0.01],
            "longitude",
            True,
            id="Longitude just above boundary",
        ),
        pytest.param(
            [np.pi * (1 / 2 + 0.01), 0, -np.pi * (1 / 2) + 0.01],
            "latitude",
            True,
            id="Latitude just above boundary",
        ),
        pytest.param(
            xr.DataArray([np.pi * (1 / 2 + 0.01), 0, -np.pi * (1 / 2) + 0.01]),
            "latitude",
            True,
            id="Latitude just above boundary (xarray)",
        ),
        pytest.param(da.asarray([1, 2, 3, 4]), "latitude", True, id="Latitude dask"),
        pytest.param(da.asarray([1, 2, 3, 7]), "longitude", True, id="Longitude dask"),
        pytest.param(da.asarray([1, 2, 3, 4]), None, False, id="No coord dask too small"),
        pytest.param(da.asarray([1, 2, 3, 4, 9]), None, True, id="No coord dask"),
    ],
)
def test_is_in_degrees(values, coordinate, expected):
    outcome = derivatives.is_in_degrees(np.asarray(values), coordinate)
    assert outcome == expected


@pytest.mark.parametrize(
    "expected, lat, lon",
    [
        (True, np.asarray([1, 2, 3]), np.asarray([3, 4, 5])),
        (True, xr.DataArray([1, 2, 3]), xr.DataArray([3, 4, 5])),
        (False, np.asarray([1, 2, 3]), np.asarray([3, 4, 5])),
        (False, xr.DataArray([1, 2, 3]), xr.DataArray([3, 4, 5])),
    ],
)
def test_is_lat_lon_in_degrees(expected: bool, mocker: "MockerFixture", lat, lon) -> None:
    is_in_deg_mock = mocker.patch("rojak.core.derivatives.is_in_degrees", return_value=expected)
    outcome = derivatives.is_lat_lon_in_degrees(lon, lon)
    assert outcome == expected
    assert is_in_deg_mock.call_count == 2


@pytest.mark.parametrize(
    "lat, lon",
    [
        (np.asarray([0.0, np.pi / 2]), np.asarray([180.0, 360.0])),
        (np.asarray([0.0, 90.0]), np.asarray([0, np.pi])),
        (xr.DataArray([0.0, np.pi / 2]), xr.DataArray([180.0, 360.0])),
        (xr.DataArray([0.0, 90.0]), xr.DataArray([0, np.pi])),
    ],
)
def test_is_lat_lon_in_degrees_error(lat: "ArrayLike", lon: "ArrayLike") -> None:
    with pytest.raises(ValueError) as excinfo:
        derivatives.is_lat_lon_in_degrees(lat, lon)
    assert excinfo.type is ValueError


def test_check_lat_lon_units_warning(mocker: "MockerFixture") -> None:
    is_in_deg_mock = mocker.patch("rojak.core.derivatives.is_lat_lon_in_degrees", return_value=False)
    with pytest.warns(UserWarning) as record:
        derivatives.ensure_lat_lon_in_deg(np.asarray([0, 1]), np.asarray([0, 1]), "deg")

    assert len(record) == 1
    assert (
        record[0].message.args[0]  # type: ignore
        == "Latitude and longitude specified to be in degrees, but are smaller than pi values"
    )
    assert is_in_deg_mock.call_count == 1


def test_check_lat_lon_units_error(mocker: "MockerFixture") -> None:
    is_in_deg_mock = mocker.patch("rojak.core.derivatives.is_lat_lon_in_degrees", return_value=True)
    with pytest.raises(ValueError) as excinfo:
        derivatives.ensure_lat_lon_in_deg(np.asarray([0, 90]), np.asarray([180, 360]), "rad")

    assert excinfo.type is ValueError
    assert is_in_deg_mock.call_count == 1


latitude_in_deg: list[float] = [90, 45, 0, 45 - 90]
longitude_in_deg: list[float] = [-180, 360, 180, 0]
latitude_in_rad: np.ndarray = np.deg2rad(latitude_in_deg)
longitude_in_rad: np.ndarray = np.deg2rad(longitude_in_deg)


@pytest.mark.parametrize(
    "lat, lon, is_deg",
    [
        pytest.param(
            xr.DataArray(latitude_in_deg),
            xr.DataArray(longitude_in_deg),
            True,
            id="xarray deg",
        ),
        pytest.param(
            xr.DataArray(latitude_in_deg),
            xr.DataArray(longitude_in_deg),
            True,
            id="xarray rad",
        ),
    ],
)
def test_check_lat_lon_in_degree(mocker: "MockerFixture", lat, lon, is_deg: bool) -> None:
    is_in_deg_mock = mocker.patch("rojak.core.derivatives.is_lat_lon_in_degrees", return_value=is_deg)
    new_lat, new_lon = derivatives.ensure_lat_lon_in_deg(lat, lon, "deg" if is_deg else "rad")
    if is_deg:
        npt.assert_array_almost_equal(np.asarray(new_lat), np.asarray(latitude_in_deg))
        npt.assert_array_almost_equal(np.asarray(new_lon), np.asarray(longitude_in_deg))
    else:
        npt.assert_array_almost_equal(np.asarray(new_lat), latitude_in_rad)
        npt.assert_array_almost_equal(np.asarray(new_lon), longitude_in_rad)

    assert is_in_deg_mock.call_count == 1


def test_nominal_grid_spacing(mocker: "MockerFixture") -> None:
    lat = np.array([25.0, 35.0, 45.0])
    lon = np.array([-105, -100, -95, -90])

    ensure_is_deg_mock = mocker.patch("rojak.core.derivatives.ensure_lat_lon_in_deg", return_value=(lat, lon))

    # Subcase 1: Specify geod
    grid_deltas: derivatives.GridSpacing = derivatives.nominal_grid_spacing(lat, lon, "deg", geod=Geod(a=4370997))
    npt.assert_array_almost_equal(grid_deltas.dx, np.asarray([381441.44622397, 381441.44622397, 381441.44622397]))
    npt.assert_array_almost_equal(grid_deltas.dy, np.asarray([762882.8924479455, 762882.8924479454]))
    assert ensure_is_deg_mock.call_count == 1

    # Subcase 2: default geod
    default_geod_deltas: derivatives.GridSpacing = derivatives.nominal_grid_spacing(lat, lon, "deg")
    npt.assert_array_almost_equal(
        default_geod_deltas.dx,
        np.asarray([556597.45396637, 556597.45396637, 556597.45396637]),
    )
    npt.assert_array_almost_equal(default_geod_deltas.dy, np.asarray([1108538.7325489155, 1110351.4762828045]))
    assert ensure_is_deg_mock.call_count == 2


def test_nominal_grid_spacing_error(mocker: "MockerFixture") -> None:
    lat = np.arange(-10, 10, 0.5)
    lon = np.arange(-30, 30, 2)

    with pytest.raises(ValueError) as excinfo:
        derivatives.nominal_grid_spacing(lat.reshape((2, -1)), lon, "deg")

    assert excinfo.type is ValueError
    assert excinfo.match("Latitude and longitude must have 1 dimension")

    with pytest.raises(ValueError) as excinfo:
        derivatives.nominal_grid_spacing(lat, lon.reshape((2, -1)), "deg")

    assert excinfo.type is ValueError
    assert excinfo.match("Latitude and longitude must have 1 dimension")


@pytest.fixture()
def create_random_lat_lon_dataarray():
    return xr.DataArray(np.random.default_rng().random((20, 30, 4)), dims=["latitude", "time", "longitude"])


@pytest.mark.parametrize(
    "dim_name, expectation",
    [
        pytest.param("latitude", nullcontext(0), id="latitude"),
        pytest.param("longitude", nullcontext(2), id="longitude"),
        pytest.param("lol", pytest.raises(ValueError), id="inexistent dim"),
    ],
)
def test_get_dimension_number(create_random_lat_lon_dataarray, dim_name, expectation):
    with expectation as e:
        dim_num: int = derivatives.get_dimension_number(dim_name, create_random_lat_lon_dataarray)
        assert dim_num == e
    if not isinstance(e, int):
        assert e.type is ValueError


@pytest.mark.parametrize(
    "dimension, expected_name",
    [
        (derivatives.CartesianDimension.X, "longitude"),
        (derivatives.CartesianDimension.Y, "latitude"),
    ],
)
def test_cartesian_dimension_get_geographic_coord_name(
    dimension: derivatives.CartesianDimension, expected_name: str
) -> None:
    name: str | None = dimension.get_geographic_coordinate()
    assert name == expected_name


@pytest.mark.parametrize(
    "dim, grid_deltas, expected",
    [
        (
            derivatives.CartesianDimension.X,
            derivatives.GridSpacing(np.arange(5), np.arange(6, 10)),
            np.arange(5),
        ),
        (
            derivatives.CartesianDimension.Y,
            derivatives.GridSpacing(np.arange(5), np.arange(6, 10)),
            np.arange(6, 10),
        ),
    ],
)
def test_cartesian_dimension_get_grid_spacing(
    dim: derivatives.CartesianDimension, grid_deltas: derivatives.GridSpacing, expected
) -> None:
    delta = dim.get_grid_spacing(grid_deltas)
    npt.assert_array_equal(delta, expected)


@pytest.mark.parametrize(
    "dim, factors, expected",
    [
        (
            derivatives.CartesianDimension.X,
            derivatives.ProjectionCorrectionFactors(xr.DataArray(np.arange(5)), xr.DataArray(np.arange(6, 10))),
            xr.DataArray(np.arange(5)),
        ),
        (
            derivatives.CartesianDimension.Y,
            derivatives.ProjectionCorrectionFactors(xr.DataArray(np.arange(5)), xr.DataArray(np.arange(6, 10))),
            xr.DataArray(np.arange(6, 10)),
        ),
    ],
)
def test_get_correction_factor(
    dim: derivatives.CartesianDimension,
    factors: derivatives.ProjectionCorrectionFactors,
    expected,
) -> None:
    scaling_factor: xr.DataArray = dim.get_correction_factor(factors)
    xr.testing.assert_equal(scaling_factor, expected)


def test_divergence(create_random_lat_lon_dataarray) -> None:
    du_dx = create_random_lat_lon_dataarray
    dv_dy = create_random_lat_lon_dataarray
    xr.testing.assert_allclose(du_dx + dv_dy, derivatives.divergence(du_dx, dv_dy))


@pytest.mark.parametrize(
    "lat, lon",
    [
        (xr.DataArray(np.arange(6).reshape((2, 3))), xr.DataArray(np.arange(6))),
        (xr.DataArray(np.arange(6).reshape((2, 3))), xr.DataArray(np.arange(6).reshape((2, 3)))),
    ],
)
def test_get_projection_correction_factors_error(lat, lon):
    with pytest.raises(ValueError) as excinfo:
        derivatives.get_projection_correction_factors(lat, lon)
    assert excinfo.type is ValueError


def test_get_projection_correction_factos() -> None:
    parallel = np.asarray(
        [
            [1.15373388, 1.15373388, 1.15373388, 1.15373388],
            [1.19569521, 1.19569521, 1.19569521, 1.19569521],
            [1.24520235, 1.24520235, 1.24520235, 1.24520235],
            [1.30360069, 1.30360069, 1.30360069, 1.30360069],
        ]
    )
    meridional = np.asarray(
        [
            [1.00421324, 1.00421324, 1.00421324, 1.00421324],
            [1.00368845, 1.00368845, 1.00368845, 1.00368845],
            [1.00313671, 1.00313671, 1.00313671, 1.00313671],
            [1.00256549, 1.00256549, 1.00256549, 1.00256549],
        ]
    )
    factors = derivatives.get_projection_correction_factors(
        xr.DataArray(np.linspace(30, 40, 4)), xr.DataArray(np.linspace(260, 270, 4))
    )
    npt.assert_array_almost_equal(factors.parallel_scale.data, parallel)
    npt.assert_array_almost_equal(factors.meridional_scale.data, meridional)


@pytest.mark.parametrize("package", [np, da])
def test_first_derivative_x_squared_equal_spacing(package) -> None:
    x_squared = package.arange(10) * package.arange(10)
    npt.assert_array_equal(
        package.gradient(x_squared, axis=0), derivatives.first_derivative(xr.DataArray(x_squared), np.ones(9), 0)
    )
    two_x = package.arange(10) * 2
    npt.assert_array_equal(derivatives.first_derivative(xr.DataArray(x_squared), np.ones(9), 0)[1:-1], two_x[1:-1])

    y = package.arange(15).reshape(3, 5)
    npt.assert_array_equal(
        derivatives.first_derivative(xr.DataArray(y), np.ones(2), axis=0), package.gradient(y, axis=0)
    )
    npt.assert_array_equal(derivatives.first_derivative(xr.DataArray(y), np.ones(2), axis=0), np.ones_like(y) * 5)
    npt.assert_array_equal(
        derivatives.first_derivative(xr.DataArray(y), np.ones(4), axis=1), package.gradient(y, axis=1)
    )
    npt.assert_array_equal(derivatives.first_derivative(xr.DataArray(y), np.ones(4), axis=1), np.ones_like(y))
