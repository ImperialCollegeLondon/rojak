from collections import namedtuple
from typing import TYPE_CHECKING

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from rojak.core.data import AmdarTurbulenceData
from rojak.turbulence.diagnostic import DiagnosticSuite
from rojak.turbulence.verification import DiagnosticsAmdarDataHarmoniser

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from rojak.core.data import CATData

DataHarmoniserAndMocks = namedtuple(
    "DataHarmoniserAndMocks", ["harmoniser", "suite_mock", "diagnostic_names_mock", "amdar_data_mock"]
)


@pytest.fixture
def instantiate_diagnostic_amdar_data_harmoniser(mocker: "MockerFixture") -> DataHarmoniserAndMocks:
    suite_mock = mocker.Mock(spec=DiagnosticSuite)
    diagnostic_names_mock = mocker.patch.object(suite_mock, "diagnostic_names", return_value=["f3d", "def"])
    amdar_data_mock = mocker.Mock(spec=AmdarTurbulenceData)
    harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data_mock, suite_mock)
    return DataHarmoniserAndMocks(harmoniser, suite_mock, diagnostic_names_mock, amdar_data_mock)


def test_create_nearest_diagnostic_value_series_dummy_data(
    mocker: "MockerFixture", make_dummy_cat_data, instantiate_diagnostic_amdar_data_harmoniser
) -> None:
    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    diagnostic_names_mock = instantiate_diagnostic_amdar_data_harmoniser.diagnostic_names_mock
    harmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_dataset = make_dummy_cat_data({}, use_numpy=False)
    diagnostic_computed_values_mock = mocker.patch.object(
        suite_mock,
        "computed_values",
        return_value=iter([("f3d", cat_dataset["geopotential"]), ("def", cat_dataset["vorticity"])]),
    )

    index_into_dataset = [
        {"latitude": 0, "longitude": 2, "pressure_level": 2, "time": 0},
        {"latitude": 1, "longitude": 5, "pressure_level": 1, "time": 0},
        {"latitude": 2, "longitude": 8, "pressure_level": 3, "time": 1},
    ]
    observational_data = dd.from_pandas(
        pd.DataFrame(
            {"lat_index": list(range(3)), "lon_index": [2, 5, 8], "level_index": [2, 1, 3], "time_index": [0, 0, 1]},
        ),
        npartitions=2,
    )
    coords_of_obs_mock = mocker.patch.object(
        harmoniser,
        "_coordinates_of_observations",
        return_value={
            "lat_index": observational_data["lat_index"].to_dask_array(lengths=True),
            "lon_index": observational_data["lon_index"].to_dask_array(lengths=True),
            "level_index": observational_data["level_index"].to_dask_array(lengths=True),
            "time_index": observational_data["time_index"].to_dask_array(lengths=True),
        },
    )

    values: dict = harmoniser._create_diagnostic_value_series(cat_dataset["temperature"], observational_data, {})
    diagnostic_names_mock.assert_called_once()
    diagnostic_computed_values_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()
    f3d_values = pd.Series(
        [cat_dataset["geopotential"][indexer].data.compute() for indexer in index_into_dataset],
        name="f3d",
    )
    def_values = pd.Series(
        [cat_dataset["vorticity"][indexer].data.compute() for indexer in index_into_dataset],
        name="def",
    )

    pd.testing.assert_series_equal(values["f3d"].compute(), f3d_values)
    pd.testing.assert_series_equal(values["def"].compute(), def_values)


@pytest.mark.parametrize("stack_on_axis", [0, 1])
def test_get_gridded_coordinates(
    mocker: "MockerFixture", load_cat_data, client, instantiate_diagnostic_amdar_data_harmoniser, stack_on_axis: int
) -> None:
    harmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_data: CATData = load_cat_data(None, with_chunks=True)

    rand_generator = np.random.default_rng()
    num_points: int = 30
    indices = {
        "lat_index": rand_generator.integers(0, high=cat_data.u_wind()["latitude"].size, size=num_points),
        "lon_index": rand_generator.integers(0, high=cat_data.u_wind()["longitude"].size, size=num_points),
        "level_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
    }

    observational_data = dd.from_pandas(pd.DataFrame(indices), npartitions=10)
    coords_of_obs_mock = mocker.patch.object(
        harmoniser,
        "_coordinates_of_observations",
        return_value={
            "lat_index": observational_data["lat_index"].to_dask_array(lengths=True),
            "lon_index": observational_data["lon_index"].to_dask_array(lengths=True),
            "level_index": observational_data["level_index"].to_dask_array(lengths=True),
            "time_index": observational_data["time_index"].to_dask_array(lengths=True),
        },
    )

    desired = np.stack(
        [indices[harmoniser.coordinate_axes_to_index_col_name()[target_dim]] for target_dim in cat_data.u_wind().dims],
        axis=stack_on_axis,
    )
    computed = harmoniser._get_gridded_coordinates(observational_data, cat_data.u_wind(), stack_on_axis).compute()

    coords_of_obs_mock.assert_called_once()

    np.testing.assert_array_equal(computed, desired)


def test_create_nearest_diagnostic_value_series_era5_data(
    mocker: "MockerFixture", load_cat_data, client, instantiate_diagnostic_amdar_data_harmoniser
) -> None:
    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    diagnostic_names_mock = instantiate_diagnostic_amdar_data_harmoniser.diagnostic_names_mock
    harmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_data: CATData = load_cat_data(None, with_chunks=True)
    diagnostic_computed_values_mock = mocker.patch.object(
        suite_mock,
        "computed_values",
        return_value=iter([("f3d", cat_data.u_wind()), ("def", cat_data.total_deformation())]),
    )

    rand_generator = np.random.default_rng()
    num_points: int = 30
    indices = {
        "lat_index": rand_generator.integers(0, high=cat_data.u_wind()["latitude"].size, size=num_points),
        "lon_index": rand_generator.integers(0, high=cat_data.u_wind()["longitude"].size, size=num_points),
        "level_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
    }
    index_into_dataset = [
        {
            "latitude": indices["lat_index"][index],
            "longitude": indices["lon_index"][index],
            "pressure_level": indices["level_index"][index],
            "time": indices["time_index"][index],
        }
        for index in range(num_points)
    ]

    observational_data = dd.from_pandas(pd.DataFrame(indices), npartitions=10)
    coords_of_obs_mock = mocker.patch.object(
        harmoniser,
        "_coordinates_of_observations",
        return_value={
            "lat_index": observational_data["lat_index"].to_dask_array(lengths=True),
            "lon_index": observational_data["lon_index"].to_dask_array(lengths=True),
            "level_index": observational_data["level_index"].to_dask_array(lengths=True),
            "time_index": observational_data["time_index"].to_dask_array(lengths=True),
        },
    )

    values: dict = harmoniser._create_diagnostic_value_series(cat_data.v_wind(), observational_data, {})
    diagnostic_names_mock.assert_called_once()
    diagnostic_computed_values_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()
    f3d_values = pd.Series([cat_data.u_wind()[indexer].data.compute() for indexer in index_into_dataset], name="f3d")
    def_values = pd.Series(
        [cat_data.total_deformation()[indexer].data.compute() for indexer in index_into_dataset],
        name="def",
    )

    pd.testing.assert_series_equal(values["f3d"].compute(), f3d_values)
    pd.testing.assert_series_equal(values["def"].compute(), def_values)
