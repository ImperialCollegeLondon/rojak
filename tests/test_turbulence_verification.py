from collections import namedtuple
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, assert_never

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rojak.core.data import AmdarTurbulenceData
from rojak.orchestrator.configuration import (
    AggregationMetricOption,
    DiagnosticValidationCondition,
    SpatialGroupByStrategy,
)
from rojak.turbulence.verification import (
    AmdarDataHarmoniser,
    DiagnosticsAmdarVerification,
    IndexingFormat,
    ObservationCoordinates,
)
from tests.conftest import time_window_for_cat_data

if TYPE_CHECKING:
    import dask.array as da
    from pytest_mock import MockerFixture

    from rojak.core.data import CATData
    from rojak.turbulence.verification import RocVerificationResult


TestCaseValues = namedtuple("TestCaseValues", ["index_into_dataset", "observational_data", "indices"])

AmdarTurbDataMock = namedtuple("AmdarTurbDataMock", ["amdar_data_mock", "observational_data_mock"])


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
                indices: dict[str, np.ndarray] = {
                    "lat_index": np.arange(3),
                    "lon_index": np.asarray([2, 5, 8]),
                    "level_index": np.asarray([2, 1, 0]),
                    "time_index": np.asarray([0, 0, 1]),
                    "maxEDR": edr_values,
                    "medEDR": edr_values,
                    "index_right": np.arange(3),
                }
                observational_data: dd.DataFrame = dd.from_pandas(pd.DataFrame(indices), npartitions=2)
                return TestCaseValues(index_into_dataset, observational_data, indices)
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
                indices["index_right"] = np.arange(num_points)

                observational_data: dd.DataFrame = dd.from_pandas(pd.DataFrame(indices), npartitions=10)
                return TestCaseValues(index_into_dataset, observational_data, indices)
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


def coords_of_obs_return_value(obs_data: dd.DataFrame) -> ObservationCoordinates:
    return ObservationCoordinates(
        obs_data["level_index"].to_dask_array(lengths=True),
        obs_data["lat_index"],
        obs_data["lon_index"],
        obs_data["time_index"],
    )


def amdar_data_mock(mocker: "MockerFixture", obs_data: "dd.DataFrame") -> AmdarTurbDataMock:
    amdar_turb_data_mock = mocker.Mock(spec=AmdarTurbulenceData)
    df_mock = mocker.patch.object(amdar_turb_data_mock, "clip_to_time_window", return_value=obs_data)
    return AmdarTurbDataMock(amdar_turb_data_mock, df_mock)


class TestAmdarDataHarmoniser:
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
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
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
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
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
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
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
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        desired: xr.DataArray = xr.zeros_like(cat_data.u_wind(), dtype=int)
        num_points: int = len(case.index_into_dataset)
        for index in case.index_into_dataset:
            desired[index] = 1

        was_observed: xr.DataArray = data_harmoniser.has_observation()
        get_coords_mock.assert_called_once()
        np.testing.assert_array_equal(was_observed, desired)

        assert was_observed.sum().compute() == num_points

    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    @pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
    def test_has_positive_turbulence_observation(
        self, mocker: "MockerFixture", case_size: TestCaseSize, min_edr: float, load_cat_data, get_test_case, client
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )

        conditions: list[DiagnosticValidationCondition] = [
            DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
            DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
        ]

        desired_data_vars: dict[str, xr.DataArray] = {
            cond.observed_turbulence_column_name: xr.zeros_like(cat_data.u_wind()) for cond in conditions
        }
        for i, indexer in enumerate(case.index_into_dataset):
            for cond in conditions:
                desired_data_vars[cond.observed_turbulence_column_name][indexer] = (
                    case.indices[cond.observed_turbulence_column_name][i] > min_edr
                )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )
        computed: xr.Dataset = data_harmoniser.grid_has_positive_turbulence_observation(conditions)
        get_coords_mock.assert_called_once()

        for condition in conditions:
            col_name: str = condition.observed_turbulence_column_name
            np.testing.assert_array_equal(
                computed[col_name].sum(dim="time"),
                desired_data_vars[col_name].sum(dim="time"),
            )
            np.testing.assert_array_equal(computed[col_name], desired_data_vars[col_name])
            assert computed[col_name].sum().compute() == np.sum(case.indices[col_name] > min_edr)


class TestDiagnosticAmdarVerification:
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_data_with_diagnostics(
        self, mocker: "MockerFixture", case_size: TestCaseSize, load_cat_data, get_test_case, client
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = xr.Dataset(
            data_vars={"f3d": cat_data.u_wind(), "def": cat_data.total_deformation()}
        )
        verifier: DiagnosticsAmdarVerification = DiagnosticsAmdarVerification(data_harmoniser, diagnostics_mock)

        with_diagnostics: dd.DataFrame = verifier.data
        get_coords_mock.assert_called()

        f3d_values = pd.Series(
            [cat_data.u_wind()[indexer].data.compute() for indexer in case.index_into_dataset], name="f3d"
        )
        def_values = pd.Series(
            [cat_data.total_deformation()[indexer].data.compute() for indexer in case.index_into_dataset],
            name="def",
        )

        pd.testing.assert_series_equal(with_diagnostics["f3d"].compute(), f3d_values)
        pd.testing.assert_series_equal(with_diagnostics["def"].compute(), def_values)

    # NOTE: Only checks that this method runs and the total is correct. It does NOT establish whether the grouping is
    # correct
    @pytest.mark.parametrize("groupby_strategy", [item.value for item in SpatialGroupByStrategy])
    @pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_num_obs_per_has_correct_total(
        self,
        mocker: "MockerFixture",
        case_size: TestCaseSize,
        min_edr: float,
        groupby_strategy: SpatialGroupByStrategy,
        load_cat_data,
        get_test_case,
        client,
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = xr.Dataset(
            data_vars={"f3d": cat_data.u_wind(), "def": cat_data.total_deformation()}
        )
        verifier: DiagnosticsAmdarVerification = DiagnosticsAmdarVerification(data_harmoniser, diagnostics_mock)

        conditions: list[DiagnosticValidationCondition] = [
            DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
            DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
        ]

        number_of_observations: dd.DataFrame = verifier.num_obs_per(conditions, groupby_strategy)
        get_coords_mock.assert_called()

        assert number_of_observations["num_obs"].sum().compute() == len(case.index_into_dataset)

    # For some reason the SpatialGroupByStrategy.GRID_POINT results in very flaky tests
    # TODO: Debug why this case is so flaky
    @pytest.mark.parametrize(
        "groupby_strategy",
        [
            SpatialGroupByStrategy.GRID_BOX,
            SpatialGroupByStrategy.HORIZONTAL_BOX,
            SpatialGroupByStrategy.HORIZONTAL_POINT,
        ],
    )
    @pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_aggregate_by_auc_runs(
        self,
        mocker: "MockerFixture",
        case_size: TestCaseSize,
        min_edr: float,
        groupby_strategy: SpatialGroupByStrategy,
        load_cat_data,
        get_test_case,
        client,
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = xr.Dataset(
            data_vars={"f3d": cat_data.u_wind(), "def": cat_data.total_deformation()}
        )
        verifier: DiagnosticsAmdarVerification = DiagnosticsAmdarVerification(data_harmoniser, diagnostics_mock)

        conditions: list[DiagnosticValidationCondition] = [
            DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
            DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
        ]

        auc_for_diagnostic: dict[str, dd.DataFrame] = verifier.aggregate_by_auc(
            conditions, groupby_strategy, 5, AggregationMetricOption.AUC
        )
        get_coords_mock.assert_called()

        print(auc_for_diagnostic)
        for computed_auc in auc_for_diagnostic.values():
            print(computed_auc.compute())

    @pytest.mark.parametrize("min_edr", [0, 0.1, 0.22, 0.5, 1])
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_nearst_value_roc_runs(
        self,
        mocker: "MockerFixture",
        case_size: TestCaseSize,
        min_edr: float,
        load_cat_data,
        get_test_case,
        client,
    ) -> None:
        case: TestCaseValues = get_test_case(case_size)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        cat_data: CATData = load_cat_data(None, with_chunks=True)
        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, cat_data.u_wind(), time_window_for_cat_data()
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = xr.Dataset(
            data_vars={"f3d": cat_data.u_wind(), "def": cat_data.total_deformation()}
        )
        verifier: DiagnosticsAmdarVerification = DiagnosticsAmdarVerification(data_harmoniser, diagnostics_mock)

        conditions: list[DiagnosticValidationCondition] = [
            DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
            DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
        ]
        roc_verification_result: RocVerificationResult = verifier.nearest_value_roc(conditions)
        get_coords_mock.assert_called()

        for _, by_diagnostic_roc in roc_verification_result.iterate_by_amdar_column():
            ##### Can't test AUC as there might not be enough true positives for it not to fail
            # auc_for_col: dict[str, float] = roc_verification_result.auc_for_amdar_column(amdar_verification_col)
            # for diagnostic, auc_for_diagnostic in auc_for_col.items():
            #     print(f"AUC for {diagnostic}: {auc_for_diagnostic}")

            for classification_result in by_diagnostic_roc.values():
                print(
                    f"tp: {classification_result.true_positives.compute()}, "
                    f"fp: {classification_result.false_positives.compute()}"
                )
