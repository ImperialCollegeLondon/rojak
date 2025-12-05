from collections.abc import Callable
from typing import TYPE_CHECKING

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import box

from rojak.core.data import CATPrognosticData
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import SpatialDomain
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from shapely.geometry.polygon import Polygon

    from rojak.core.data import CATData

# Add fixtures from dask.distributed
pytest_plugins = ["distributed.utils_test"]


def time_window_dummy_coordinate() -> Limits["np.datetime64"]:
    return Limits(np.datetime64("2005-02-01T00"), np.datetime64("2005-02-01T23"))


def time_coordinate():
    return np.arange("2005-02-01T00", "2005-02-02T00", dtype="datetime64[h]")


def generate_array_data(shape: tuple, use_numpy: bool, rng_seed=None):
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


@pytest.fixture
def load_era5_data() -> Callable:
    def _load_era5_data(with_chunks: bool = False) -> Era5Data:
        dataset: xr.Dataset = xr.open_dataset("tests/_static/test_era5_data.nc", engine="h5netcdf")
        if with_chunks:
            dataset = dataset.chunk(chunks={"valid_time": 2})
        return Era5Data(dataset)

    return _load_era5_data


@pytest.fixture
def load_cat_data(load_era5_data) -> Callable:
    def _load_cat_data(domain: SpatialDomain | None, with_chunks: bool = False) -> "CATData":
        data: Era5Data = load_era5_data(with_chunks=with_chunks)
        if domain is None:
            domain = SpatialDomain(
                minimum_latitude=-90,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
            )
        return data.to_clear_air_turbulence_data(domain)

    return _load_cat_data
