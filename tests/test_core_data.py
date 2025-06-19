from typing import TYPE_CHECKING, Tuple

import dask.array as da
import dask_geopandas
import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from rojak.core.data import CATPrognosticData, MetData, as_geo_dataframe
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import SpatialDomain

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from rojak.utilities.types import Limits


@pytest.fixture
def create_met_data_mock(mocker: "MockerFixture"):
    mocker.patch.multiple(MetData, __abstractmethods__=set())
    return MetData()  # pyright: ignore[reportAbstractUsage]


def time_coordinate():
    return np.arange("2005-02-01T00", "2005-02-02T00", dtype="datetime64[h]")


def generate_array_data(shape: Tuple, use_numpy: bool, rng_seed=None):
    data = np.random.default_rng(rng_seed).random(shape)
    return data if use_numpy else da.from_array(data)


@pytest.fixture
def make_select_domain_dummy_data():
    # Factory as fixtures
    # https://docs.pytest.org/en/latest/how-to/fixtures.html#factories-as-fixtures
    def _make_select_domain_dummy_data(to_replace: dict, use_numpy: bool = True) -> xr.Dataset:
        default_coords = {
            "longitude": np.arange(10),
            "latitude": np.arange(10),
            "time": time_coordinate(),
            "level": np.arange(4),
        }
        if to_replace:
            default_coords.update(to_replace)
        return xr.Dataset(
            data_vars={
                "a": xr.DataArray(
                    data=generate_array_data((10, 10, 24, 4), use_numpy),
                    dims=["longitude", "latitude", "time", "level"],
                )
            },
            coords=default_coords,
        )

    return _make_select_domain_dummy_data


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


def test_select_spatial_domain_no_slicing(make_select_domain_dummy_data):
    domain = SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180)
    dummy_data = make_select_domain_dummy_data({})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(dummy_data, down_selected)


def test_select_domain_emtpy_slice(make_select_domain_dummy_data):
    domain = SpatialDomain(
        minimum_latitude=0,
        maximum_latitude=90,
        minimum_longitude=0,
        maximum_longitude=180,
        maximum_level=0,
        minimum_level=-1,
    )
    dummy_data = make_select_domain_dummy_data(
        {"longitude": np.linspace(-90, 0, 10), "latitude": np.linspace(-90, 0, 10)}
    )
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(
        dummy_data.sel(time=slice(None), longitude=slice(0, 180), latitude=slice(0, 90), level=slice(-1, 0)),
        down_selected,
    )
    assert down_selected["a"].shape == (1, 1, 24, 1)


@pytest.mark.parametrize(
    ("domain", "new_longitude_array", "longitude_slice"),
    [
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=0, maximum_longitude=180),
            np.linspace(-180, 180, 10),
            slice(0, None),
            id="slice_above_min",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=0, maximum_longitude=180),
            np.flip(np.linspace(-180, 180, 10)),
            slice(180, 0),
            id="slice_above_min_reversed",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=-90),
            np.linspace(-180, 0, 10),
            slice(-180, -90),
            id="slice_below_max",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=-90),
            np.flip(np.linspace(-180, 0, 10)),
            slice(-90, -180),
            id="slice_below_max_reversed",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-100, maximum_longitude=90),
            np.linspace(-180, 180, 10),
            slice(-100, 90),
            id="center_of_domain",
        ),
    ],
)
def test_select_domain_slice_longitude(make_select_domain_dummy_data, domain, new_longitude_array, longitude_slice):
    dummy_data = make_select_domain_dummy_data({"longitude": new_longitude_array})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(
        dummy_data.sel(longitude=longitude_slice, latitude=slice(None), time=slice(None), level=slice(None)),
        down_selected,
    )
    assert down_selected["a"].shape == (5, 10, 24, 4)


@pytest.mark.parametrize(
    ("domain", "new_latitude_array", "latitude_slice"),
    [
        pytest.param(
            SpatialDomain(minimum_latitude=0, maximum_latitude=90, minimum_longitude=0, maximum_longitude=180),
            np.linspace(-90, 90, 10),
            slice(0, None),
            id="slice_above_min",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=0, maximum_latitude=90, minimum_longitude=0, maximum_longitude=180),
            np.flip(np.linspace(-90, 90, 10)),
            slice(90, 0),
            id="slice_above_min_reversed",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=0, minimum_longitude=-180, maximum_longitude=180),
            np.linspace(-90, 90, 10),
            slice(-90, 0),
            id="slice_below_max_negative",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=0, maximum_latitude=45, minimum_longitude=-180, maximum_longitude=180),
            np.linspace(0, 90, 10),
            slice(0, 45),
            id="slice_below_max_positive",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=0, minimum_longitude=-180, maximum_longitude=180),
            np.flip(np.linspace(-90, 90, 10)),
            slice(0, -90),
            id="slice_below_max_negative_reversed",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=0, maximum_latitude=45, minimum_longitude=-180, maximum_longitude=180),
            np.flip(np.linspace(0, 90, 10)),
            slice(45, 0),
            id="slice_below_max_positive_reversed",
        ),
        pytest.param(
            SpatialDomain(minimum_latitude=-45, maximum_latitude=45, minimum_longitude=-180, maximum_longitude=180),
            np.linspace(-80, 90, 10),
            slice(-45, 45),
            id="center_of_domain",
        ),
    ],
)
def test_select_domain_slice_latitude(make_select_domain_dummy_data, domain, new_latitude_array, latitude_slice):
    dummy_data = make_select_domain_dummy_data({"latitude": new_latitude_array})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(
        dummy_data.sel(longitude=slice(None), latitude=latitude_slice, time=slice(None), level=slice(None)),
        down_selected,
    )
    assert down_selected["a"].shape == (10, 5, 24, 4)


@pytest.mark.parametrize(
    ("domain", "new_level", "level_slice"),
    [
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=0,
                maximum_longitude=180,
                minimum_level=5,
                maximum_level=10,
            ),
            np.linspace(0, 9, 4),
            slice(5, 10),
            id="slice_above_min",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=0,
                maximum_longitude=180,
                minimum_level=5,
                maximum_level=10,
            ),
            np.flip(np.linspace(0, 9, 4)),
            slice(10, 5),
            id="slice_above_min_reversed",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_level=-10,
                maximum_level=-5,
            ),
            np.linspace(-9, 0, 4),
            slice(-10, -5),
            id="slice_below_max_negative",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_level=0,
                maximum_level=5,
            ),
            np.linspace(0, 9, 4),
            slice(0, 5),
            id="slice_below_max_positive",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_level=-10,
                maximum_level=-5,
            ),
            np.flip(np.linspace(-9, 0, 4)),
            slice(-5, -10),
            id="slice_below_max_negative_reversed",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_level=0,
                maximum_level=5,
            ),
            np.flip(np.linspace(0, 9, 4)),
            slice(5, 0),
            id="slice_below_max_positive_reversed",
        ),
        pytest.param(
            SpatialDomain(
                minimum_latitude=0,
                maximum_latitude=90,
                minimum_longitude=-180,
                maximum_longitude=180,
                minimum_level=2,
                maximum_level=7,
            ),
            np.linspace(0, 9, 4),
            slice(2, 7),
            id="center_of_domain",
        ),
    ],
)
def test_select_domain_slice_level(make_select_domain_dummy_data, domain, new_level, level_slice):
    dummy_data = make_select_domain_dummy_data({"level": new_level})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(
        dummy_data.sel(longitude=slice(None), latitude=slice(None), time=slice(None), level=level_slice),
        down_selected,
    )
    assert down_selected["a"].shape == (10, 10, 24, 2)


def test_select_domain_shift_longitude(make_select_domain_dummy_data):
    dummy_data = make_select_domain_dummy_data({"longitude": np.linspace(0, 350, 10)})
    assert dummy_data["a"]["longitude"].min() == 0
    assert dummy_data["a"]["longitude"].max() == 350  # noqa: PLR2004
    domain = SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180)
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    assert down_selected["a"]["longitude"].min() < -165  # noqa: PLR2004
    assert down_selected["a"]["longitude"].max() > 155  # noqa: PLR2004
    assert down_selected["a"].shape == (10, 10, 24, 4)


def test_as_geo_dataframe(make_select_domain_dummy_data):
    dummy_data = make_select_domain_dummy_data({}, use_numpy=False)
    gdf = as_geo_dataframe(dummy_data.to_dask_dataframe())
    assert "geometry" in gdf.columns
    assert isinstance(gdf, dask_geopandas.GeoDataFrame)
    assert gdf.crs == "epsg:4326"
    gdf.head()  # Call head() to check it can be computed


def test_instantiate_cat_prognostic_fail_on_variables(make_select_domain_dummy_data):
    with pytest.raises(
        ValueError, match="Attempting to instantiate CATPrognosticData with missing data variables"
    ) as excinfo:
        CATPrognosticData(make_select_domain_dummy_data({}))

    assert excinfo.type is ValueError


def test_instantiate_cat_prognostic_fail_on_coords(make_dummy_cat_data):
    dummy_data = make_dummy_cat_data({})
    dummy_data = dummy_data.drop_vars("altitude")
    with pytest.raises(ValueError, match="Attempting to instantiate CATPrognosticData with missing coords") as excinfo:
        CATPrognosticData(dummy_data)

    assert excinfo.type is ValueError


def test_instantiate_cat_prognostic_successfully(make_dummy_cat_data):
    dummy_data = CATPrognosticData(make_dummy_cat_data({}))
    assert isinstance(dummy_data, CATPrognosticData)


@pytest.mark.parametrize(
    ("attr_name", "dataset_name"),
    [
        ("temperature", "temperature"),
        ("divergence", "divergence_of_wind"),
        ("geopotential", "geopotential"),
        ("specific_humidity", "specific_humidity"),
        ("u_wind", "eastward_wind"),
        ("v_wind", "northward_wind"),
        ("potential_vorticity", "potential_vorticity"),
        ("vorticity", "vorticity"),
        ("altitude", "altitude"),
    ],
)
def test_getters_on_cat_prognostic_dataset(make_dummy_cat_data, attr_name: str, dataset_name: str):
    dummy_data = make_dummy_cat_data({})
    dataset = CATPrognosticData(dummy_data)

    xrt.assert_equal(getattr(dataset, attr_name)(), dummy_data[dataset_name])


def test_time_window_on_cat_prognostic_dataset(make_dummy_cat_data):
    dataset = CATPrognosticData(make_dummy_cat_data({}))
    min_time = time_coordinate().min()
    max_time = time_coordinate().max()
    window: "Limits" = dataset.time_window()
    assert window.lower == min_time
    assert window.upper == max_time
