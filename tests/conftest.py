from typing import TYPE_CHECKING, Callable, Tuple

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import box

from rojak.core.data import CATPrognosticData

if TYPE_CHECKING:
    from shapely.geometry.polygon import Polygon


def time_coordinate():
    return np.arange("2005-02-01T00", "2005-02-02T00", dtype="datetime64[h]")


def generate_array_data(shape: Tuple, use_numpy: bool, rng_seed=None):
    data = np.random.default_rng(rng_seed).random(shape)
    return data if use_numpy else da.from_array(data)


@pytest.fixture
def make_dummy_cat_data() -> Callable:
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


@pytest.fixture
def load_flat_data() -> pd.DataFrame:
    return pd.read_csv("tests/_static/flat_data.csv")


@pytest.fixture
def load_geo_data(load_flat_data: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        load_flat_data,
        geometry=gpd.points_from_xy(load_flat_data["longitude"], load_flat_data["latitude"]),
        crs="EPSG:4326",
    )


@pytest.fixture
def valid_region_for_flat_data() -> "Polygon":
    return box(-130, 25, 28, 60)
