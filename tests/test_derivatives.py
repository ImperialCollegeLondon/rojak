import numpy as np
import xarray as xr
import dask.array as da
import pytest

from rojak.core import derivatives


@pytest.mark.parametrize(
    "values, coordinate, expected",
    [
        pytest.param([8], None, True, id="No coordinate in degrees"),
        pytest.param(
            [np.pi * 2, -np.pi * 2], None, False, id="No coordinate on boundary"
        ),
        pytest.param([np.pi, 0, -np.pi], None, False, id="Longitude boundary"),
        pytest.param(
            [np.pi * (1 / 2), 0, -np.pi * (1 / 2)], None, False, id="Latitude boundary"
        ),
        pytest.param(
            [np.pi + 0.01, 0, -np.pi + 0.01],
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
        pytest.param(da.asarray([1, 2, 3, 4]), "longitude", True, id="Longitude dask"),
        pytest.param(
            da.asarray([1, 2, 3, 4]), None, False, id="No coord dask too small"
        ),
        pytest.param(da.asarray([1, 2, 3, 4, 9]), None, True, id="No coord dask"),
    ],
)
def test_is_in_degrees(values, coordinate, expected):
    outcome = derivatives.is_in_degrees(np.asarray(values), coordinate)
    assert outcome == expected
