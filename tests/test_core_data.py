import threading
from collections import Counter
from typing import TYPE_CHECKING

import dask.dataframe as dd
import dask_geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.testing as xrt
from dask import config as dask_config
from dask.core import flatten
from dask.local import get_sync

from rojak.atmosphere.contrails import issr
from rojak.core.data import CATData, CATPrognosticData, MetData, as_geo_dataframe
from rojak.core.derivatives import VelocityDerivative
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.datalib.madis.amdar import AcarsAmdarRepository, AcarsAmdarTurbulenceData
from rojak.datalib.ukmo.amdar import UkmoAmdarRepository
from rojak.orchestrator.configuration import SpatialDomain
from tests.conftest import generate_array_data, time_coordinate

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from rojak.utilities.types import Limits


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
                ),
            },
            coords=default_coords,
        )

    return _make_select_domain_dummy_data


def test_select_spatial_domain_no_slicing(make_select_domain_dummy_data) -> None:
    domain = SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180)
    dummy_data = make_select_domain_dummy_data({})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(dummy_data, down_selected)


def test_select_domain_emtpy_slice(make_select_domain_dummy_data) -> None:
    domain = SpatialDomain(
        minimum_latitude=0,
        maximum_latitude=90,
        minimum_longitude=0,
        maximum_longitude=180,
        maximum_level=0,
        minimum_level=-1,
    )
    dummy_data = make_select_domain_dummy_data(
        {"longitude": np.linspace(-90, 0, 10), "latitude": np.linspace(-90, 0, 10)},
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
def test_select_domain_slice_longitude(
    make_select_domain_dummy_data,
    domain,
    new_longitude_array,
    longitude_slice,
) -> None:
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
def test_select_domain_slice_latitude(
    make_select_domain_dummy_data,
    domain,
    new_latitude_array,
    latitude_slice,
) -> None:
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
def test_select_domain_slice_level(make_select_domain_dummy_data, domain, new_level, level_slice) -> None:
    dummy_data = make_select_domain_dummy_data({"level": new_level})
    down_selected = Era5Data(dummy_data).select_domain(domain, dummy_data)
    xrt.assert_equal(
        dummy_data.sel(longitude=slice(None), latitude=slice(None), time=slice(None), level=level_slice),
        down_selected,
    )
    assert down_selected["a"].shape == (10, 10, 24, 2)


def test_select_domain_shift_longitude(make_select_domain_dummy_data) -> None:
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
        ValueError,
        match="Attempting to instantiate CATPrognosticData with missing data variables",
    ) as excinfo:
        CATPrognosticData(make_select_domain_dummy_data({}), 100)

    assert excinfo.type is ValueError


def test_instantiate_cat_prognostic_fail_on_coords(make_dummy_cat_data):
    dummy_data = make_dummy_cat_data({})
    dummy_data = dummy_data.drop_vars("altitude")
    with pytest.raises(ValueError, match="Attempting to instantiate CATPrognosticData with missing coords") as excinfo:
        CATPrognosticData(dummy_data, 100)

    assert excinfo.type is ValueError


def test_instantiate_cat_prognostic_successfully(make_dummy_cat_data):
    dummy_data = CATPrognosticData(make_dummy_cat_data({}), 100)
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
@pytest.mark.parametrize("persist", [True, False])
def test_getters_on_cat_prognostic_dataset(make_dummy_cat_data, attr_name: str, dataset_name: str, persist: bool):
    dummy_data = make_dummy_cat_data({})
    dataset = CATPrognosticData(dummy_data, 100, persist=persist)

    xrt.assert_equal(getattr(dataset, attr_name)(), dummy_data[dataset_name])


def fail_if_computed(*args, **kwargs) -> None:
    raise AssertionError("Dask computation was triggered")


def test_cat_prognostic_data_only_persists_accessed_variable(make_dummy_cat_data) -> None:
    requested_keys: list[set] = []

    def recording_scheduler(dsk, keys, **kwargs):
        requested_keys.append(set(flatten(keys)))
        return get_sync(dsk, keys, **kwargs)

    with dask_config.set(scheduler=recording_scheduler):
        data = CATPrognosticData(make_dummy_cat_data({}, use_numpy=False), 100)
        assert requested_keys == [], "Nothing should be loaded on construction"

        u_wind = data.u_wind()
        assert requested_keys == [set(flatten(u_wind.data.__dask_keys__()))], "Only u wind should be loaded"

        assert data.u_wind() is u_wind
        assert len(requested_keys) == 1, "u wind should only be loaded once"


def test_cat_data_persists_nothing_when_disabled(make_dummy_cat_data) -> None:
    with dask_config.set(scheduler=fail_if_computed):
        data = CATData(make_dummy_cat_data({}, use_numpy=False), 100, persist=False)
        for accessor in [
            "temperature",
            "divergence",
            "geopotential",
            "specific_humidity",
            "u_wind",
            "v_wind",
            "potential_vorticity",
            "vorticity",
            "potential_temperature",
        ]:
            getattr(data, accessor)()


@pytest.mark.parametrize("persist", [True, False])
def test_met_data_creates_cat_data_respecting_persist(make_dummy_cat_data, persist: bool) -> None:
    class DummyMetData(MetData):
        def __init__(self, dataset: xr.Dataset) -> None:
            super().__init__(persist=persist)
            self._dataset = dataset

        def to_clear_air_turbulence_data(self, domain: "SpatialDomain") -> CATData:
            return self._create_cat_data(self._dataset, 100)

    met_data = DummyMetData(make_dummy_cat_data({}, use_numpy=False))
    assert met_data.persist == persist

    requested: list = []

    def recording_scheduler(dsk, keys, **kwargs):
        requested.append(keys)
        return get_sync(dsk, keys, **kwargs)

    with dask_config.set(scheduler=recording_scheduler):
        met_data.to_clear_air_turbulence_data(
            SpatialDomain(minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180)
        ).u_wind()
    assert bool(requested) == persist


def test_time_window_on_cat_prognostic_dataset(make_dummy_cat_data):
    dataset = CATPrognosticData(make_dummy_cat_data({}), 100)
    min_time = time_coordinate().min()
    max_time = time_coordinate().max()
    window: Limits = dataset.time_window()
    assert window.lower == min_time
    assert window.upper == max_time


def test_cat_data_potential_temperature(mocker: "MockerFixture", make_dummy_cat_data) -> None:
    dummy_data = make_dummy_cat_data({})
    temp_to_potential_temp = mocker.patch("rojak.turbulence.calculations.potential_temperature")
    temp_to_potential_temp.return_value = dummy_data["temperature"]

    data = CATData(dummy_data, 100)
    theta = data.potential_temperature()
    temp_to_potential_temp.assert_called_once()
    xrt.assert_equal(theta, dummy_data["temperature"])
    xrt.assert_equal(temp_to_potential_temp.call_args.args[0], data.temperature())
    xrt.assert_equal(temp_to_potential_temp.call_args.args[0], dummy_data["temperature"])
    xrt.assert_equal(temp_to_potential_temp.call_args.args[1], data.temperature()["pressure_level"])
    xrt.assert_equal(temp_to_potential_temp.call_args.args[1], dummy_data["pressure_level"])


def test_cat_data_velocity_derivatives(mocker: "MockerFixture", make_dummy_cat_data) -> None:
    dummy_data = make_dummy_cat_data({})
    velocity_derivatives = mocker.patch("rojak.core.derivatives.vector_derivatives")
    ret_val = {VelocityDerivative.DV_DX: None}
    velocity_derivatives.return_value = ret_val

    data = CATData(dummy_data, 100)
    computed_derivs = data.velocity_derivatives()
    velocity_derivatives.assert_called_once()
    assert computed_derivs == ret_val
    xrt.assert_equal(velocity_derivatives.call_args.args[0], data.u_wind())
    xrt.assert_equal(velocity_derivatives.call_args.args[0], dummy_data["eastward_wind"])
    xrt.assert_equal(velocity_derivatives.call_args.args[1], data.v_wind())
    xrt.assert_equal(velocity_derivatives.call_args.args[1], dummy_data["northward_wind"])
    assert velocity_derivatives.call_args.args[2] == "deg"

    dv_dx = data.specific_velocity_derivative(VelocityDerivative.DV_DX)
    assert dv_dx is None


@pytest.mark.parametrize(
    ("deformation_type", "method_name"),
    [("shearing_deformation", "shear_deformation"), ("stretching_deformation", "stretching_deformation")],
)
def test_shear_and_stretch_deformation(
    mocker: "MockerFixture",
    make_dummy_cat_data,
    deformation_type,
    method_name,
) -> None:
    dummy_data = make_dummy_cat_data({})
    data = CATData(dummy_data, 100)

    vel_deriv_mock = mocker.patch.object(data, "specific_velocity_derivative", return_value=dummy_data["eastward_wind"])
    deformation = mocker.patch(f"rojak.turbulence.calculations.{deformation_type}")
    deformation.return_value = dummy_data["northward_wind"]

    computed_deformation = getattr(data, method_name)()

    deformation.assert_called_once()
    xrt.assert_equal(deformation.call_args.args[0], dummy_data["eastward_wind"])
    xrt.assert_equal(deformation.call_args.args[1], dummy_data["eastward_wind"])
    vel_deriv_mock.assert_called()
    assert vel_deriv_mock.call_count == 2  # noqa: PLR2004

    xrt.assert_equal(computed_deformation, dummy_data["northward_wind"])

    stored_deformation = getattr(data, method_name)()
    # Verify call count did not increase
    assert vel_deriv_mock.call_count == 2  # noqa: PLR2004
    xrt.assert_equal(stored_deformation, dummy_data["northward_wind"])


def test_total_deformation(make_dummy_cat_data) -> None:
    dummy_data = make_dummy_cat_data({})
    data = CATData(dummy_data, 100)

    deformation_from_class = data.total_deformation()
    total_deformation = np.sqrt(
        data.shear_deformation() * data.shear_deformation()
        + data.stretching_deformation() * data.stretching_deformation()
    )
    xrt.assert_allclose(deformation_from_class, total_deformation)


def test_jacobian_horizontal_velocity(mocker: "MockerFixture", make_dummy_cat_data) -> None:
    # Analytical Solution for
    # x = u^2 - v^3 => dx_du = 2u dx_dv = -3v^2
    # y = u^2 + v^3 => dy_du = 2u dy_dv = 3v^2
    # determinant = 12 u v^2

    dummy_data = make_dummy_cat_data({})
    data = CATData(dummy_data, 100)
    derivatives = {
        VelocityDerivative.DU_DX: 2 * dummy_data["eastward_wind"],
        VelocityDerivative.DV_DX: -3 * dummy_data["northward_wind"] * dummy_data["northward_wind"],
        VelocityDerivative.DU_DY: 2 * dummy_data["eastward_wind"],
        VelocityDerivative.DV_DY: 3 * dummy_data["northward_wind"] * dummy_data["northward_wind"],
    }
    mocker.patch.object(data, "velocity_derivatives", return_value=derivatives)
    analytical_det_of_jacobian = (
        12 * dummy_data["eastward_wind"] * dummy_data["northward_wind"] * dummy_data["northward_wind"]
    )
    xrt.assert_allclose(analytical_det_of_jacobian, data.jacobian_horizontal_velocity())


@pytest.fixture
def get_standard_atmosphere_pressure_and_altitude():
    pressure = np.asarray([1013.25, 843.07, 329.32, 238.42, 187.54])
    # MSL, 5000ft, 28 000ft, 35 000ft, 40 000ft
    altitude = np.asarray([0, 1524.0, 8534.4, 10668.0, 12192.0])
    return pressure, altitude


@pytest.mark.parametrize("concrete_class", [AcarsAmdarRepository, UkmoAmdarRepository])
def test_compute_closest_pressure_level(
    mocker: "MockerFixture",
    get_standard_atmosphere_pressure_and_altitude,
    concrete_class,
):
    pressure, altitude = get_standard_atmosphere_pressure_and_altitude
    # See docs on why it is core.data instead of where it is imported from, i.e. core.calculations
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    converter_mock = mocker.patch("rojak.core.data.pressure_to_altitude_icao")
    converter_mock.return_value = altitude
    ddf = dd.from_pandas(pd.DataFrame({"altitude": [500, 1000, 2000, 9000, 15000]}))
    closest = pd.Series([1013.24, 843.07, 843.07, 329.32, 187.54], name="level")

    instance = concrete_class("")
    ddf["level"] = instance._compute_closest_pressure_level(ddf, pressure, "altitude")
    converter_mock.assert_called_once_with(pressure)
    assert isinstance(ddf["level"], dd.Series)
    pd.testing.assert_series_equal(ddf["level"].compute(), closest)


@pytest.mark.parametrize("concrete_class", [AcarsAmdarRepository, UkmoAmdarRepository])
def test_convert_to_amdar_turbulence_data(
    mocker: "MockerFixture",
    load_flat_data: pd.DataFrame,
    concrete_class,
    valid_region_for_flat_data,
):
    instance = concrete_class("")
    load_mock = mocker.patch.object(instance, "load", return_value=dd.from_pandas(load_flat_data))

    turbulence_data = instance.to_amdar_turbulence_data(valid_region_for_flat_data, 0.25, [150, 200, 250])
    load_mock.assert_called_once()
    computed_df = turbulence_data.data_frame.compute()
    assert isinstance(computed_df, gpd.GeoDataFrame)
    assert set(computed_df.columns) == {
        "recNum",
        "altitude",
        "level",
        "index_right",
        "datetime",
        "geometry",
        "latitude",
        "longitude",
    }


# Number of times each partition has been read. Module level (rather than a closure) so that dask can deterministically
# tokenise the function which counts the reads
_PARTITION_READS: Counter[int] = Counter()
_PARTITION_READS_LOCK = threading.Lock()


def _count_partition_reads(partition: pd.DataFrame, partition_info: dict | None = None) -> pd.DataFrame:
    if partition_info is not None:
        with _PARTITION_READS_LOCK:
            _PARTITION_READS[partition_info["number"]] += 1
    return partition


@pytest.mark.parametrize("concrete_class", [AcarsAmdarRepository, UkmoAmdarRepository])
def test_convert_to_amdar_turbulence_data_loads_data_once(
    mocker: "MockerFixture",
    load_flat_data: pd.DataFrame,
    concrete_class,
    valid_region_for_flat_data,
) -> None:
    _PARTITION_READS.clear()
    num_partitions: int = 3
    loaded = dd.from_pandas(load_flat_data, npartitions=num_partitions).map_partitions(
        _count_partition_reads, meta=load_flat_data.iloc[:0]
    )
    instance = concrete_class("")
    mocker.patch.object(instance, "load", return_value=loaded)

    instance.to_amdar_turbulence_data(valid_region_for_flat_data, 0.25, [150, 200, 250]).data_frame.compute()

    assert dict(_PARTITION_READS) == dict.fromkeys(range(num_partitions), 1)


def test_turbulence_frequency_statistics_in_single_pass() -> None:
    rng = np.random.default_rng(6)
    values = rng.lognormal(-3, 1, size=500)
    values[::7] = np.nan
    data = pd.DataFrame(dict.fromkeys(AcarsAmdarTurbulenceData.turbulence_column_names(), values))

    # Bypass __init__ as only the data frame is used and the quality control requires geospatial columns
    acars_data = object.__new__(AcarsAmdarTurbulenceData)
    acars_data._data_frame = dd.from_pandas(data, npartitions=4)

    num_computes: list[int] = []

    def counting_scheduler(dsk, keys, **kwargs):
        num_computes.append(1)
        return get_sync(dsk, keys, **kwargs)

    with dask_config.set(scheduler=counting_scheduler):
        statistics = acars_data.turbulence_frequency_statistics("maxEDR", 0.1)

    assert len(num_computes) == 1
    num_observations = int(np.count_nonzero(~np.isnan(values)))
    num_turb_obs = int(np.sum(values > 0.1))  # noqa: PLR2004
    assert statistics == {
        "num_turb_obs": num_turb_obs,
        "num_observations": num_observations,
        "frequency": num_turb_obs / num_observations,
    }


def test_ice_super_saturated_regions(load_cat_data) -> None:
    instantiated: CATData = load_cat_data(None)
    np.testing.assert_allclose(
        instantiated.ice_supersaturated_regions(),
        issr(
            air_temperature=instantiated.temperature(),
            specific_humidity=instantiated.specific_humidity(),
            air_pressure=instantiated.temperature()["pressure_level"] * 100,
            rhi_threshold=0.9,
        ),
    )
    np.testing.assert_allclose(
        instantiated.ice_supersaturated_regions(),
        issr(
            air_temperature=instantiated.temperature(),
            specific_humidity=instantiated.specific_humidity(),
            air_pressure=instantiated.pressure_level(convert_to_pascals=False) * 100,
            rhi_threshold=0.9,
        ),
    )

    assert instantiated.ice_supersaturated_regions().name == "issr"


def test_acars_edr_distribution_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    num_rows: int = 1000
    max_edr = rng.lognormal(mean=-3, sigma=0.5, size=num_rows)
    # Non-positive and missing values must be excluded
    max_edr[::10] = 0
    max_edr[1::10] = -0.1
    max_edr[2::10] = np.nan
    data = pd.DataFrame(
        {column: rng.random(num_rows) for column in AcarsAmdarTurbulenceData.turbulence_column_names()}
    ).assign(maxEDR=max_edr)

    # Bypass __init__ as only the data frame is used and the quality control requires geospatial columns
    acars_data = object.__new__(AcarsAmdarTurbulenceData)
    acars_data._data_frame = dd.from_pandas(data, npartitions=7)
    distribution = acars_data.edr_distribution()

    expected = np.log(max_edr[max_edr > 0])
    np.testing.assert_allclose(distribution.mean, np.mean(expected))
    np.testing.assert_allclose(distribution.variance, np.var(expected))
