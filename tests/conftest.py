from typing import Tuple

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from rojak.core.data import CATPrognosticData


def time_coordinate():
    return np.arange("2005-02-01T00", "2005-02-02T00", dtype="datetime64[h]")


def generate_array_data(shape: Tuple, use_numpy: bool, rng_seed=None):
    data = np.random.default_rng(rng_seed).random(shape)
    return data if use_numpy else da.from_array(data)


@pytest.fixture
def make_dummy_cat_data():
    def _make_dummy_cat_data(to_replace: dict, use_numpy: bool = True, rng_seed=None) -> xr.Dataset:
        default_coords = {
            "longitude": np.arange(10),
            "latitude": np.arange(10),
            "time": time_coordinate(),
            "pressure_level": np.arange(4),
        }
        if to_replace:
            default_coords.update(to_replace)
        data_vars = {
            var: xr.DataArray(
                data=generate_array_data((10, 10, 24, 4), use_numpy, rng_seed),
                dims=["longitude", "latitude", "time", "pressure_level"],
            )
            for var in CATPrognosticData.required_variables
        }
        ds = xr.Dataset(data_vars=data_vars, coords=default_coords)
        return ds.assign_coords(altitude=("pressure_level", np.arange(4)))

    return _make_dummy_cat_data
