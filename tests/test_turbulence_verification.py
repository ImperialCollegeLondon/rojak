import warnings
from collections import namedtuple
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, assert_never

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from rojak.core.data import AmdarTurbulenceData
from rojak.orchestrator.configuration import DiagnosticValidationCondition
from rojak.turbulence.diagnostic import DiagnosticSuite
from rojak.turbulence.verification import (
    AmdarDataHarmoniser,
    DiagnosticsAmdarDataHarmoniser,
    IndexingFormat,
    ObservationCoordinates,
)
from tests.conftest import time_window_dummy_coordinate, time_window_for_cat_data

if TYPE_CHECKING:
    import dask.array as da
    from pytest_mock import MockerFixture

    from rojak.core.data import CATData

DataHarmoniserAndMocks = namedtuple(
    "DataHarmoniserAndMocks", ["harmoniser", "suite_mock", "diagnostic_names_mock", "amdar_data_mock"]
)

TestCaseValues = namedtuple("TestCaseValues", ["index_into_dataset", "observational_data"])

AmdarTurbDataMock = namedtuple("AmdarTurbDataMock", ["amdar_data_mock", "observational_data_mock"])


@pytest.fixture
def instantiate_diagnostic_amdar_data_harmoniser(mocker: "MockerFixture") -> DataHarmoniserAndMocks:
    suite_mock = mocker.Mock(spec=DiagnosticSuite)
    diagnostic_names_mock = mocker.patch.object(suite_mock, "diagnostic_names", return_value=["f3d", "def"])
    amdar_data_mock = mocker.Mock(spec=AmdarTurbulenceData)
    harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data_mock, suite_mock)
    return DataHarmoniserAndMocks(harmoniser, suite_mock, diagnostic_names_mock, amdar_data_mock)


class TestCaseSize(Enum):
    SMALL = auto()
    LARGE = auto()


possible_test_case_sizes: list[TestCaseSize] = [TestCaseSize.SMALL, TestCaseSize.LARGE]


@pytest.fixture
def get_test_case(load_cat_data) -> Callable[[TestCaseSize], TestCaseValues]:
    def _get_test_case(size: TestCaseSize) -> TestCaseValues:
        rand_generator = np.random.default_rng()
        match size:
            case TestCaseSize.SMALL:
                index_into_dataset = [
                    {"latitude": 0, "longitude": 2, "pressure_level": 2, "time": 0},
                    {"latitude": 1, "longitude": 5, "pressure_level": 1, "time": 0},
                    {"latitude": 2, "longitude": 8, "pressure_level": 0, "time": 1},
                ]
                edr_values = rand_generator.lognormal(-3, 1, size=3)
                observational_data: dd.DataFrame = dd.from_pandas(
                    pd.DataFrame(
                        {
                            "lat_index": list(range(3)),
                            "lon_index": [2, 5, 8],
                            "level_index": [2, 1, 0],
                            "time_index": [0, 0, 1],
                            "maxEDR": edr_values,
                            "medEDR": edr_values,
                        },
                    ),
                    npartitions=2,
                )
                return TestCaseValues(index_into_dataset, observational_data)
            case TestCaseSize.LARGE:
                cat_data: CATData = load_cat_data(None, with_chunks=True)
                num_points: int = 30
                indices: dict[str, np.ndarray] = {
                    "lat_index": rand_generator.integers(0, high=cat_data.u_wind()["latitude"].size, size=num_points),
                    "lon_index": rand_generator.integers(0, high=cat_data.u_wind()["longitude"].size, size=num_points),
                    "level_index": rand_generator.integers(
                        0, high=cat_data.u_wind()["pressure_level"].size, size=num_points
                    ),
                    "time_index": rand_generator.integers(0, high=cat_data.u_wind()["time"].size, size=num_points),
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

                mean: float = -3
                std_dev: float = 1
                indices["maxEDR"] = rand_generator.lognormal(mean=mean, sigma=std_dev, size=num_points)
                indices["medEDR"] = rand_generator.lognormal(mean=mean, sigma=std_dev, size=num_points)

                observational_data: dd.DataFrame = dd.from_pandas(pd.DataFrame(indices), npartitions=10)
                return TestCaseValues(index_into_dataset, observational_data)
            case _ as unreachable:
                assert_never(unreachable)

    return _get_test_case


@pytest.mark.parametrize("case_size", possible_test_case_sizes)
def test_observation_coordinates_as_arrays(get_test_case, case_size: TestCaseSize) -> None:
    case = get_test_case(case_size)
    coords: ObservationCoordinates = ObservationCoordinates(
        case.observational_data["level_index"].to_dask_array(lengths=True),
        case.observational_data["lat_index"],
        case.observational_data["lon_index"],
        case.observational_data["time_index"],
    )

    index_col_names = [
        AmdarDataHarmoniser.level_index_column,
        AmdarDataHarmoniser.time_index_column,
        AmdarDataHarmoniser.lat_index_column,
        AmdarDataHarmoniser.lon_index_column,
    ]
    as_arrays: dict[str, da.Array] = coords.as_arrays()
    for col_name in index_col_names:
        np.testing.assert_array_equal(
            as_arrays[col_name], case.observational_data[col_name].to_dask_array(lengths=True)
        )


class TestAmdarDataHarmoniser:
    @staticmethod
    def amdar_data_mock(mocker: "MockerFixture", obs_data: "dd.DataFrame") -> AmdarTurbDataMock:
        amdar_data_mock = mocker.Mock(spec=AmdarTurbulenceData)
        df_mock = mocker.patch.object(amdar_data_mock, "clip_to_time_window", return_value=obs_data)
        return AmdarTurbDataMock(amdar_data_mock, df_mock)

    @staticmethod
    def coords_of_obs_return_value(obs_data: dd.DataFrame) -> ObservationCoordinates:
        return ObservationCoordinates(
            obs_data["level_index"].to_dask_array(lengths=True),
            obs_data["lat_index"],
            obs_data["lon_index"],
            obs_data["time_index"],
        )

    @pytest.mark.parametrize("indexing_format", [IndexingFormat.COORDINATES, IndexingFormat.FLAT])
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_observations_index_to_grid_fails(
        self,
        mocker: "MockerFixture",
        case_size: TestCaseSize,
        indexing_format: IndexingFormat,
        load_cat_data,
        get_test_case,
        client,
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = self.amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=self.coords_of_obs_return_value(case.observational_data),
        )

        phrase_to_match: str = (
            "stack_on_axis cannot be None"
            if indexing_format == IndexingFormat.COORDINATES
            else "stack_on_axis must be None"
        )

        with pytest.raises(TypeError, match=phrase_to_match):
            data_harmoniser.observations_index_to_grid(
                indexing_format, stack_on_axis=None if indexing_format == IndexingFormat.COORDINATES else 0
            )

        get_coords_mock.assert_called_once()

    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    @pytest.mark.parametrize("stack_on_axis", [0, 1])
    def test_observations_index_to_grid_coords(
        self, mocker: "MockerFixture", stack_on_axis: int, case_size: TestCaseSize, load_cat_data, get_test_case, client
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = self.amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=self.coords_of_obs_return_value(case.observational_data),
        )

        assert cat_data.u_wind().get_axis_num(["latitude", "longitude", "time", "pressure_level"]) == (0, 1, 2, 3)

        coords: list[tuple[int, int, int, int]] = [
            (item["latitude"], item["longitude"], item["time"], item["pressure_level"])
            for item in case.index_into_dataset
        ]

        mapped_to_grid_rows = data_harmoniser.observations_index_to_grid(
            IndexingFormat.COORDINATES, stack_on_axis=stack_on_axis
        )
        get_coords_mock.assert_called_once()

        desired: np.ndarray = np.asarray(coords)
        if stack_on_axis == 0:
            desired = desired.T
        np.testing.assert_array_equal(mapped_to_grid_rows, desired)

    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_observations_index_to_grid_ravel_equiv_coords(
        self, mocker: "MockerFixture", case_size: TestCaseSize, load_cat_data, get_test_case, client
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = self.amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=self.coords_of_obs_return_value(case.observational_data),
        )

        flattened_idx = data_harmoniser.observations_index_to_grid(IndexingFormat.FLAT)
        get_coords_mock.assert_called_once()
        coords_idx = data_harmoniser.observations_index_to_grid(IndexingFormat.COORDINATES, stack_on_axis=1)
        np.testing.assert_array_equal(flattened_idx, np.ravel_multi_index(coords_idx.T, cat_data.u_wind().shape))

    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_has_observation(
        self, mocker: "MockerFixture", case_size: TestCaseSize, load_cat_data, get_test_case, client
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = self.amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=self.coords_of_obs_return_value(case.observational_data),
        )

        desired: xr.DataArray = xr.zeros_like(cat_data.u_wind(), dtype=int)
        num_points: int = len(case.index_into_dataset)
        for index in case.index_into_dataset:
            desired[index] = 1

        was_observed: xr.DataArray = data_harmoniser.has_observation()
        get_coords_mock.assert_called_once()
        np.testing.assert_array_equal(was_observed, desired)

        assert was_observed.sum().compute() == num_points


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
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["time"].size, size=num_points),
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
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["time"].size, size=num_points),
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


@pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
def test_grid_turbulence_observation_frequency_simple(
    mocker: "MockerFixture", make_dummy_cat_data, instantiate_diagnostic_amdar_data_harmoniser, min_edr: float
) -> None:
    harmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_dataset: xr.Dataset = make_dummy_cat_data({}, use_numpy=False)

    index_into_dataset = [
        {"latitude": 0, "longitude": 2, "pressure_level": 2, "time": 0},
        {"latitude": 1, "longitude": 5, "pressure_level": 1, "time": 0},
        {"latitude": 2, "longitude": 8, "pressure_level": 3, "time": 1},
    ]
    rand_generator = np.random.default_rng()
    edr_values = rand_generator.lognormal(-3, 1, size=3)
    observational_data = dd.from_pandas(
        pd.DataFrame(
            {
                "lat_index": list(range(3)),
                "lon_index": [2, 5, 8],
                "level_index": [2, 1, 3],
                "time_index": [0, 0, 1],
                "maxEDR": edr_values,
                "medEDR": edr_values,
            },
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

    desired: xr.DataArray = xr.zeros_like(cat_dataset["geopotential"], dtype=int)
    for index_into_array, index_to_dataset in enumerate(index_into_dataset):
        desired[index_to_dataset] = int(edr_values[index_into_array] > min_edr)

    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    get_prototype_mock = mocker.patch.object(
        suite_mock, "get_prototype_computed_diagnostic", return_value=cat_dataset["geopotential"]
    )
    get_observational_data_mock = mocker.patch.object(
        harmoniser, "_get_observational_data", return_value=observational_data
    )

    conditions: list[DiagnosticValidationCondition] = [
        DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
        DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
    ]
    computed = harmoniser.grid_turbulence_observation_frequency(time_window_dummy_coordinate(), conditions)
    get_prototype_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()
    get_observational_data_mock.assert_called_once()

    assert sparse.any(computed["maxEDR"].data) == np.any(edr_values > min_edr)
    if min_edr == 1:
        assert not sparse.all(computed["maxEDR"])

    np.testing.assert_array_equal(
        computed["maxEDR"].data.map_blocks(lambda block: block.todense(), dtype=int).compute(), desired
    )
    np.testing.assert_array_equal(
        computed["medEDR"].data.map_blocks(lambda block: block.todense(), dtype=int).compute(), desired
    )


@pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
def test_grid_turbulence_observation_frequency_multiple_chunks(
    mocker: "MockerFixture", load_cat_data, client, instantiate_diagnostic_amdar_data_harmoniser, min_edr: float
) -> None:
    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    harmoniser: DiagnosticsAmdarDataHarmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_data: CATData = load_cat_data(None, with_chunks=True)

    rand_generator = np.random.default_rng()
    num_points: int = 30
    indices: dict[str, np.ndarray] = {
        "lat_index": rand_generator.integers(0, high=cat_data.u_wind()["latitude"].size, size=num_points),
        "lon_index": rand_generator.integers(0, high=cat_data.u_wind()["longitude"].size, size=num_points),
        "level_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["time"].size, size=num_points),
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

    mean: float = -3
    std_dev: float = 1
    indices["maxEDR"] = rand_generator.lognormal(mean=mean, sigma=std_dev, size=num_points)
    indices["medEDR"] = rand_generator.lognormal(mean=mean, sigma=std_dev, size=num_points)

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

    get_prototype_mock = mocker.patch.object(
        suite_mock, "get_prototype_computed_diagnostic", return_value=cat_data.u_wind()
    )
    get_observational_data_mock = mocker.patch.object(
        harmoniser, "_get_observational_data", return_value=observational_data
    )

    conditions: list[DiagnosticValidationCondition] = [
        DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
        DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
    ]

    computed: xr.Dataset = harmoniser.grid_turbulence_observation_frequency(time_window_for_cat_data(), conditions)
    get_prototype_mock.assert_called_once()
    get_observational_data_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()

    desired_data_vars: dict[str, xr.DataArray] = {
        cond.observed_turbulence_column_name: xr.zeros_like(cat_data.u_wind()) for cond in conditions
    }
    for i, index in enumerate(index_into_dataset):
        for cond in conditions:
            desired_data_vars[cond.observed_turbulence_column_name][index] = (
                indices[cond.observed_turbulence_column_name][i] > min_edr
            )

    for condition in conditions:
        col_name: str = condition.observed_turbulence_column_name
        np.testing.assert_array_equal(
            computed[col_name].sum(dim="time").compute().data.todense(),
            desired_data_vars[col_name].sum(dim="time"),
        )
        np.testing.assert_array_equal(
            computed[col_name].data.map_blocks(lambda block: block.todense()), desired_data_vars[col_name]
        )
        computed[col_name].data = computed[col_name].data.map_blocks(lambda block: block.todense())
        xr.testing.assert_equal(computed[col_name], desired_data_vars[col_name])


def test_grid_total_observation_count(
    mocker: "MockerFixture", make_dummy_cat_data, instantiate_diagnostic_amdar_data_harmoniser
) -> None:
    harmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_dataset: xr.Dataset = make_dummy_cat_data({}, use_numpy=False)

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

    desired: xr.DataArray = xr.zeros_like(cat_dataset["geopotential"], dtype=int)
    for index in index_into_dataset:
        desired[index] = 1

    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    get_prototype_mock = mocker.patch.object(
        suite_mock, "get_prototype_computed_diagnostic", return_value=cat_dataset["geopotential"]
    )

    computed = harmoniser.grid_total_observations_count(time_window_dummy_coordinate())
    get_prototype_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()

    computed = computed.data.map_blocks(lambda block: block.todense()).compute()
    np.testing.assert_array_equal(computed, desired)


def test_grid_total_observation_count_with_chunks(
    mocker: "MockerFixture", load_cat_data, client, instantiate_diagnostic_amdar_data_harmoniser
) -> None:
    suite_mock = instantiate_diagnostic_amdar_data_harmoniser.suite_mock
    harmoniser: DiagnosticsAmdarDataHarmoniser = instantiate_diagnostic_amdar_data_harmoniser.harmoniser
    cat_data: CATData = load_cat_data(None, with_chunks=True)

    rand_generator = np.random.default_rng()
    num_points: int = 30
    indices: dict[str, np.ndarray] = {
        "lat_index": rand_generator.integers(0, high=cat_data.u_wind()["latitude"].size, size=num_points),
        "lon_index": rand_generator.integers(0, high=cat_data.u_wind()["longitude"].size, size=num_points),
        "level_index": rand_generator.integers(0, high=cat_data.u_wind()["pressure_level"].size, size=num_points),
        "time_index": rand_generator.integers(0, high=cat_data.u_wind()["time"].size, size=num_points),
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

    get_prototype_mock = mocker.patch.object(
        suite_mock, "get_prototype_computed_diagnostic", return_value=cat_data.u_wind()
    )

    computed = harmoniser.grid_total_observations_count(time_window_for_cat_data())
    get_prototype_mock.assert_called_once()
    coords_of_obs_mock.assert_called_once()

    desired: xr.DataArray = xr.zeros_like(cat_data.u_wind(), dtype=int)
    for index in range(num_points):
        desired[
            {
                "latitude": indices["lat_index"][index],
                "longitude": indices["lon_index"][index],
                "pressure_level": indices["level_index"][index],
                "time": indices["time_index"][index],
            }
        ] = 1

    # Ignore sending large graph warning
    warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
    assert sparse.all(computed.data == desired.to_numpy())

    np.testing.assert_array_equal(computed.data.map_blocks(lambda block: block.todense()), desired)
