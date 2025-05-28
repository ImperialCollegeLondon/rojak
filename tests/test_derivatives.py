from typing import TYPE_CHECKING
import numpy as np
import numpy.testing as npt
import xarray as xr
import dask.array as da
import pytest

from rojak.core import derivatives
from rojak.utilities.types import ArrayLike

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "values, coordinate, expected",
    [
        pytest.param([8], None, True, id="No coordinate in degrees"),
        pytest.param(
            [np.pi * 2, -np.pi * 2], None, False, id="No coordinate on boundary"
        ),
        pytest.param([2 * np.pi, 0, -2 * np.pi], None, False, id="Longitude boundary"),
        pytest.param(
            [np.pi * (1 / 2), 0, -np.pi * (1 / 2)], None, False, id="Latitude boundary"
        ),
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
        pytest.param(
            da.asarray([1, 2, 3, 4]), None, False, id="No coord dask too small"
        ),
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
def test_is_lat_lon_in_degrees(
    expected: bool, mocker: "MockerFixture", lat, lon
) -> None:
    is_in_deg_mock = mocker.patch(
        "rojak.core.derivatives.is_in_degrees", return_value=expected
    )
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
    is_in_deg_mock = mocker.patch(
        "rojak.core.derivatives.is_lat_lon_in_degrees", return_value=False
    )
    with pytest.warns(UserWarning) as record:
        derivatives.ensure_lat_lon_in_deg(np.asarray([0, 1]), np.asarray([0, 1]), "deg")

    assert len(record) == 1
    assert (
        record[0].message.args[0]  # type: ignore
        == "Latitude and longitude specified to be in degrees, but are smaller than pi values"
    )
    assert is_in_deg_mock.call_count == 1


def test_check_lat_lon_units_error(mocker: "MockerFixture") -> None:
    is_in_deg_mock = mocker.patch(
        "rojak.core.derivatives.is_lat_lon_in_degrees", return_value=True
    )
    with pytest.raises(ValueError) as excinfo:
        derivatives.ensure_lat_lon_in_deg(
            np.asarray([0, 90]), np.asarray([180, 360]), "rad"
        )

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
def test_check_lat_lon_in_degree(
    mocker: "MockerFixture", lat, lon, is_deg: bool
) -> None:
    is_in_deg_mock = mocker.patch(
        "rojak.core.derivatives.is_lat_lon_in_degrees", return_value=is_deg
    )
    new_lat, new_lon = derivatives.ensure_lat_lon_in_deg(
        lat, lon, "deg" if is_deg else "rad"
    )
    if is_deg:
        npt.assert_array_almost_equal(np.asarray(new_lat), np.asarray(latitude_in_deg))
        npt.assert_array_almost_equal(np.asarray(new_lon), np.asarray(longitude_in_deg))
    else:
        npt.assert_array_almost_equal(np.asarray(new_lat), latitude_in_rad)
        npt.assert_array_almost_equal(np.asarray(new_lon), longitude_in_rad)

    assert is_in_deg_mock.call_count == 1
