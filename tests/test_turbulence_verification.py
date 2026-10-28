from collections import namedtuple
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, assert_never

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import integrate

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
from rojak.utilities.types import Limits
from tests.conftest import time_window_for_cat_data

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

    from rojak.core.data import CATData
    from rojak.turbulence.verification import RocVerificationResult


ThisTestCaseValues = namedtuple("TestCaseValues", ["index_into_dataset", "observational_data", "indices"])

AmdarTurbDataMock = namedtuple("AmdarTurbDataMock", ["amdar_data_mock", "observational_data_mock"])


class ThisTestCaseSize(Enum):
    SMALL = auto()
    LARGE = auto()


possible_test_case_sizes: list[ThisTestCaseSize] = [ThisTestCaseSize.SMALL, ThisTestCaseSize.LARGE]


GridSetup = namedtuple("GridSetup", ["grid_prototype", "diagnostics", "time_window"])


class GridBackend(StrEnum):
    # Small synthetic grid on the default (local) scheduler
    SYNTHETIC = "synthetic"
    # ERA5 test data on a distributed cluster
    ERA5 = "era5"


def synthetic_grid_setup() -> GridSetup:
    rng = np.random.default_rng(0)
    coords = {
        "latitude": np.linspace(-60, 60, 5),
        "longitude": np.arange(-180, 180, 36, dtype=float),
        "time": np.arange("2018-08-01T00", "2018-08-02T00", 6, dtype="datetime64[h]").astype("datetime64[ns]"),
        "pressure_level": np.asarray([200.0, 250.0, 300.0]),
    }
    dims = ("latitude", "longitude", "time", "pressure_level")
    shape = tuple(coords[dim].size for dim in dims)

    def make_array() -> xr.DataArray:
        return xr.DataArray(da.asarray(rng.random(shape), chunks=(2, 5, 2, 3)), dims=dims, coords=coords)

    grid_prototype = make_array()
    return GridSetup(
        grid_prototype,
        xr.Dataset(data_vars={"f3d": grid_prototype, "def": make_array()}),
        Limits(np.datetime64("2018-08-01T00"), np.datetime64("2018-08-01T18")),
    )


@pytest.fixture(params=[GridBackend.SYNTHETIC, pytest.param(GridBackend.ERA5, marks=pytest.mark.slow)])
def grid_setup(request: pytest.FixtureRequest) -> GridSetup:
    """
    Grid, diagnostics, and time window for the tests to run against

    The synthetic backend checks the same behaviour as the ERA5 one in a fraction of the time. The ERA5 backend is
    retained (marked as slow) to check the behaviour on real data with a distributed scheduler.
    """
    if request.param == GridBackend.SYNTHETIC:
        return synthetic_grid_setup()

    request.getfixturevalue("client")
    cat_data: CATData = request.getfixturevalue("load_cat_data")(None, with_chunks=True)
    return GridSetup(
        cat_data.u_wind(),
        xr.Dataset(data_vars={"f3d": cat_data.u_wind(), "def": cat_data.total_deformation()}),
        time_window_for_cat_data(),
    )


def get_test_case(size: ThisTestCaseSize, grid_prototype: xr.DataArray) -> ThisTestCaseValues:
    rand_generator = np.random.default_rng()
    match size:
        case ThisTestCaseSize.SMALL:
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
            return ThisTestCaseValues(index_into_dataset, observational_data, indices)
        case ThisTestCaseSize.LARGE:
            num_points: int = 30
            dims = ["latitude", "longitude", "pressure_level", "time"]
            # Sample without replacement so that each observation is in a distinct grid cell
            flat_cells = rand_generator.choice(grid_prototype.size, size=num_points, replace=False)
            unravelled = np.unravel_index(flat_cells, tuple(grid_prototype[dim].size for dim in dims))
            indices: dict[str, np.ndarray] = dict(
                zip(["lat_index", "lon_index", "level_index", "time_index"], unravelled, strict=True)
            )
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
            return ThisTestCaseValues(index_into_dataset, observational_data, indices)
        case _ as unreachable:
            assert_never(unreachable)


@pytest.mark.parametrize("case_size", possible_test_case_sizes)
def test_observation_coordinates_as_arrays(grid_setup: GridSetup, case_size: ThisTestCaseSize) -> None:
    case = get_test_case(case_size, grid_setup.grid_prototype)
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


def make_harmoniser(
    mocker: "MockerFixture", observational_data: dd.DataFrame, grid_setup: GridSetup
) -> tuple[AmdarDataHarmoniser, "Mock"]:
    amdar_turb_data_mock, _ = amdar_data_mock(mocker, observational_data)
    data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
        amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
    )
    get_coords_mock = mocker.patch.object(
        data_harmoniser,
        "coordinates_of_observations",
        return_value=coords_of_obs_return_value(observational_data),
    )
    return data_harmoniser, get_coords_mock


class TestAmdarDataHarmoniser:
    @pytest.mark.parametrize("indexing_format", [IndexingFormat.COORDINATES, IndexingFormat.FLAT])
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_observations_index_to_grid_fails(
        self,
        mocker: "MockerFixture",
        case_size: ThisTestCaseSize,
        indexing_format: IndexingFormat,
        grid_setup: GridSetup,
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
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
        self,
        mocker: "MockerFixture",
        stack_on_axis: int,
        case_size: ThisTestCaseSize,
        grid_setup: GridSetup,
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        assert grid_setup.grid_prototype.get_axis_num(["latitude", "longitude", "time", "pressure_level"]) == (
            0,
            1,
            2,
            3,
        )

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
        self, mocker: "MockerFixture", case_size: ThisTestCaseSize, grid_setup: GridSetup
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        flattened_idx = data_harmoniser.observations_index_to_grid(IndexingFormat.FLAT)
        get_coords_mock.assert_called_once()
        coords_idx = data_harmoniser.observations_index_to_grid(IndexingFormat.COORDINATES, stack_on_axis=1)
        np.testing.assert_array_equal(
            flattened_idx, np.ravel_multi_index(coords_idx.T, grid_setup.grid_prototype.shape)
        )

    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_has_observation(self, mocker: "MockerFixture", case_size: ThisTestCaseSize, grid_setup: GridSetup) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )

        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        desired: xr.DataArray = xr.zeros_like(grid_setup.grid_prototype, dtype=int)
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
        self, mocker: "MockerFixture", case_size: ThisTestCaseSize, min_edr: float, grid_setup: GridSetup
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )

        conditions: list[DiagnosticValidationCondition] = [
            DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
            DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
        ]

        desired_data_vars: dict[str, xr.DataArray] = {
            cond.observed_turbulence_column_name: xr.zeros_like(grid_setup.grid_prototype) for cond in conditions
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
        self, mocker: "MockerFixture", case_size: ThisTestCaseSize, grid_setup: GridSetup
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = grid_setup.diagnostics
        verifier: DiagnosticsAmdarVerification = DiagnosticsAmdarVerification(data_harmoniser, diagnostics_mock)

        with_diagnostics: dd.DataFrame = verifier.data
        get_coords_mock.assert_called()

        f3d_values = pd.Series(
            [grid_setup.diagnostics["f3d"][indexer].data.compute() for indexer in case.index_into_dataset], name="f3d"
        )
        def_values = pd.Series(
            [grid_setup.diagnostics["def"][indexer].data.compute() for indexer in case.index_into_dataset],
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
        case_size: ThisTestCaseSize,
        min_edr: float,
        groupby_strategy: SpatialGroupByStrategy,
        grid_setup: GridSetup,
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = grid_setup.diagnostics
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
        case_size: ThisTestCaseSize,
        min_edr: float,
        groupby_strategy: SpatialGroupByStrategy,
        grid_setup: GridSetup,
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = grid_setup.diagnostics
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
        case_size: ThisTestCaseSize,
        min_edr: float,
        grid_setup: GridSetup,
    ) -> None:
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, case.observational_data)

        data_harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_turb_data_mock, grid_setup.grid_prototype, grid_setup.time_window
        )
        get_coords_mock = mocker.patch.object(
            data_harmoniser,
            "coordinates_of_observations",
            return_value=coords_of_obs_return_value(case.observational_data),
        )

        diagnostics_mock: xr.Dataset = grid_setup.diagnostics
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


# Horizontal positions (lat_index, lon_index) used by the correctness tests. Each one is observed in every
# combination of time and pressure level (i.e. in distinct grid cells), apart from the last position which only has
# two observations so that it falls below the minimum group size
HORIZONTAL_POSITIONS: list[tuple[int, int]] = [(1, 0), (1, 4), (2, 2), (2, 7), (3, 5), (3, 9)]
MINIMUM_GROUP_SIZE: int = 3

SPACE_INDEX_COLUMNS: dict[SpatialGroupByStrategy, list[str]] = {
    SpatialGroupByStrategy.GRID_BOX: ["index_right", AmdarDataHarmoniser.level_index_column],
    SpatialGroupByStrategy.GRID_POINT: [
        AmdarDataHarmoniser.lat_index_column,
        AmdarDataHarmoniser.lon_index_column,
        AmdarDataHarmoniser.level_index_column,
    ],
    SpatialGroupByStrategy.HORIZONTAL_BOX: ["index_right"],
    SpatialGroupByStrategy.HORIZONTAL_POINT: [
        AmdarDataHarmoniser.lat_index_column,
        AmdarDataHarmoniser.lon_index_column,
    ],
}
INDEX_TO_COORDINATE: dict[str, str] = {
    AmdarDataHarmoniser.lat_index_column: "latitude",
    AmdarDataHarmoniser.lon_index_column: "longitude",
    AmdarDataHarmoniser.level_index_column: "pressure_level",
}


def observations_at_positions(grid_prototype: xr.DataArray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        (lat, lon, level, time)
        for lat, lon in HORIZONTAL_POSITIONS[:-1]
        for level in range(grid_prototype["pressure_level"].size)
        for time in range(grid_prototype["time"].size)
    ]
    lat, lon = HORIZONTAL_POSITIONS[-1]
    rows.extend([(lat, lon, 0, 0), (lat, lon, 0, 1)])
    observations = pd.DataFrame(rows, columns=["lat_index", "lon_index", "level_index", "time_index"])
    return observations.assign(
        # Boxes contain all the positions which share a latitude
        index_right=observations["lat_index"] * 10,
        maxEDR=rng.lognormal(-3, 1, size=len(observations)),
        medEDR=rng.lognormal(-3, 1, size=len(observations)),
    )


def diagnostic_values_at(diagnostic: xr.DataArray, observations: pd.DataFrame) -> np.ndarray:
    return (
        diagnostic.isel(
            latitude=xr.DataArray(observations["lat_index"].to_numpy()),
            longitude=xr.DataArray(observations["lon_index"].to_numpy()),
            pressure_level=xr.DataArray(observations["level_index"].to_numpy()),
            time=xr.DataArray(observations["time_index"].to_numpy()),
        )
        .compute()
        .to_numpy()
    )


def rates_from_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True and false positive rates at each threshold for labels sorted by descending diagnostic value"""
    num_positives: int = int(labels.sum())
    num_negatives: int = labels.size - num_positives
    true_positive_rate = np.concatenate(([0], np.cumsum(labels))) / num_positives
    false_positive_rate = (
        np.concatenate(([0], np.cumsum(~labels))) / num_negatives if num_negatives else np.zeros(labels.size + 1)
    )
    return true_positive_rate, false_positive_rate


def expected_group_metric(
    labels: np.ndarray, metric: AggregationMetricOption, integration_scheme: str = "simpson"
) -> float:
    # Minimum number of positive observations for the rates to be computed
    min_true_positives: int = 2
    if labels.size < MINIMUM_GROUP_SIZE:
        return np.nan
    if metric == AggregationMetricOption.PREVALENCE:
        return labels.sum() / labels.size
    if labels.sum() < min_true_positives:
        return -1
    true_positive_rate, false_positive_rate = rates_from_labels(labels)
    if metric == AggregationMetricOption.TSS:
        return float(np.max(true_positive_rate - false_positive_rate))
    if integration_scheme == "trapz":
        return float(np.trapezoid(true_positive_rate, x=false_positive_rate))
    return float(integrate.simpson(true_positive_rate, x=false_positive_rate))


def keyed_by(data_frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Move the grid box index into a column (if it is the index) and index by ``key_columns`` for comparison"""
    if data_frame.index.name == "index_right":
        data_frame = data_frame.reset_index(drop="index_right" in data_frame.columns)
    return data_frame.reset_index(drop=True).set_index(key_columns).sort_index()


def conditions_for(min_edr: float) -> list[DiagnosticValidationCondition]:
    return [
        DiagnosticValidationCondition(observed_turbulence_column_name="maxEDR", value_greater_than=min_edr),
        DiagnosticValidationCondition(observed_turbulence_column_name="medEDR", value_greater_than=min_edr),
    ]


class TestVerificationCorrectness:
    @pytest.mark.parametrize("case_size", possible_test_case_sizes)
    def test_observation_data_with_indices_matches_coordinates(
        self, mocker: "MockerFixture", case_size: ThisTestCaseSize, grid_setup: GridSetup
    ) -> None:
        # Unlike the other tests, coordinates_of_observations is NOT mocked so that the mapping of the observations'
        # positions onto the grid is checked
        case: ThisTestCaseValues = get_test_case(case_size, grid_setup.grid_prototype)
        grid = grid_setup.grid_prototype
        observations = pd.DataFrame(
            {
                AmdarDataHarmoniser.latitude_column: grid["latitude"].to_numpy()[case.indices["lat_index"]],
                AmdarDataHarmoniser.longitude_column: grid["longitude"].to_numpy()[case.indices["lon_index"]],
                AmdarDataHarmoniser.time_column: grid["time"].to_numpy()[case.indices["time_index"]],
                AmdarDataHarmoniser.level_column: grid["pressure_level"].to_numpy()[case.indices["level_index"]],
                "index_right": case.indices["index_right"],
            }
        )
        amdar_turb_data_mock, _ = amdar_data_mock(mocker, dd.from_pandas(observations, npartitions=2))
        data_harmoniser = AmdarDataHarmoniser(amdar_turb_data_mock, grid, grid_setup.time_window)

        with_indices: pd.DataFrame = data_harmoniser.observation_data_with_indices().compute()
        as_arrays: dict[str, da.Array] = data_harmoniser.coordinates_of_observations().as_arrays()
        for index_column in ["lat_index", "lon_index", "level_index", "time_index"]:
            np.testing.assert_array_equal(with_indices[index_column].to_numpy(), case.indices[index_column])
            np.testing.assert_array_equal(as_arrays[index_column].compute(), case.indices[index_column])

    @pytest.mark.parametrize("groupby_strategy", list(SpatialGroupByStrategy))
    def test_num_obs_per_matches_pandas(
        self, mocker: "MockerFixture", groupby_strategy: SpatialGroupByStrategy, grid_setup: GridSetup
    ) -> None:
        observations = observations_at_positions(grid_setup.grid_prototype, seed=1)
        data_harmoniser, _ = make_harmoniser(mocker, dd.from_pandas(observations, npartitions=3), grid_setup)
        verifier = DiagnosticsAmdarVerification(data_harmoniser, grid_setup.diagnostics)

        space_columns = SPACE_INDEX_COLUMNS[groupby_strategy]
        computed = keyed_by(verifier.num_obs_per(conditions_for(0.1), groupby_strategy).compute(), space_columns)
        expected = observations.groupby(space_columns).size().rename("num_obs").to_frame().sort_index()

        pd.testing.assert_frame_equal(computed[["num_obs"]], expected, check_dtype=False)

    @pytest.mark.parametrize("groupby_strategy", list(SpatialGroupByStrategy))
    @pytest.mark.parametrize("min_edr", [0.05, 0.22])
    def test_aggregate_by_auc_matches_pandas(
        self,
        mocker: "MockerFixture",
        groupby_strategy: SpatialGroupByStrategy,
        min_edr: float,
        grid_setup: GridSetup,
    ) -> None:
        observations = observations_at_positions(grid_setup.grid_prototype, seed=2)
        data_harmoniser, _ = make_harmoniser(mocker, dd.from_pandas(observations, npartitions=3), grid_setup)
        verifier = DiagnosticsAmdarVerification(data_harmoniser, grid_setup.diagnostics)
        conditions = conditions_for(min_edr)

        computed_by_diagnostic = verifier.aggregate_by_auc(
            conditions, groupby_strategy, MINIMUM_GROUP_SIZE, AggregationMetricOption.AUC
        )

        space_columns = SPACE_INDEX_COLUMNS[groupby_strategy]
        coordinate_columns = [INDEX_TO_COORDINATE.get(column, column) for column in space_columns]
        validation_columns = [condition.observed_turbulence_column_name for condition in conditions]
        for diagnostic_name, diagnostic in grid_setup.diagnostics.data_vars.items():
            in_descending_order = observations.assign(
                diagnostic=diagnostic_values_at(diagnostic, observations)
            ).sort_values("diagnostic", ascending=False)
            expected = pd.DataFrame(
                {
                    condition.observed_turbulence_column_name: in_descending_order.groupby(space_columns)[
                        condition.observed_turbulence_column_name
                    ].apply(
                        lambda values, threshold=condition.value_greater_than: expected_group_metric(
                            values.to_numpy() > threshold, AggregationMetricOption.AUC
                        )
                    )
                    for condition in conditions
                }
            )
            # Mirror aggregate_by_auc: drop groups with too few observations, then invalid values become NaN
            expected = expected.dropna()
            expected = expected.where(expected >= 0)
            # Grid indices are returned as their coordinate values
            expected = expected.reset_index()
            for index_column, coordinate in INDEX_TO_COORDINATE.items():
                if index_column in expected.columns:
                    expected[coordinate] = grid_setup.grid_prototype[coordinate].to_numpy()[expected[index_column]]
            expected = expected.set_index(coordinate_columns)[validation_columns].sort_index()

            computed = keyed_by(computed_by_diagnostic[str(diagnostic_name)].compute(), coordinate_columns)
            pd.testing.assert_frame_equal(computed[validation_columns], expected, check_dtype=False)

    @pytest.mark.parametrize("min_edr", [0.05, 0.1])
    def test_nearest_value_roc_matches_numpy(
        self, mocker: "MockerFixture", min_edr: float, grid_setup: GridSetup
    ) -> None:
        rng = np.random.default_rng(3)
        # Several observations in the same grid cell should be aggregated: any for the observed turbulence and mean
        # for the (identical) diagnostic values
        cells = observations_at_positions(grid_setup.grid_prototype, seed=3)[
            ["lat_index", "lon_index", "level_index", "time_index", "index_right"]
        ]
        observations = cells.loc[cells.index.repeat(rng.integers(1, 4, size=len(cells)))].reset_index(drop=True)
        observations = observations.assign(
            maxEDR=rng.lognormal(-3, 1, size=len(observations)),
            medEDR=rng.lognormal(-3, 1, size=len(observations)),
        )
        data_harmoniser, _ = make_harmoniser(mocker, dd.from_pandas(observations, npartitions=4), grid_setup)
        verifier = DiagnosticsAmdarVerification(data_harmoniser, grid_setup.diagnostics)
        conditions = conditions_for(min_edr)

        roc = verifier.nearest_value_roc(conditions)

        cell_columns = ["lat_index", "lon_index", "level_index", "time_index"]
        for diagnostic_name, diagnostic in grid_setup.diagnostics.data_vars.items():
            for condition in conditions:
                column = condition.observed_turbulence_column_name
                by_cell = (
                    observations.assign(is_turbulent=observations[column] > condition.value_greater_than)
                    .groupby(cell_columns)["is_turbulent"]
                    .any()
                    .reset_index()
                )
                by_cell = by_cell.assign(diagnostic=diagnostic_values_at(diagnostic, by_cell))
                labels = by_cell.sort_values("diagnostic", ascending=False)["is_turbulent"].to_numpy()
                expected_tpr, expected_fpr = rates_from_labels(labels)

                computed = roc.by_amdar_col_then_diagnostic[column][str(diagnostic_name)]
                np.testing.assert_allclose(computed.true_positives.compute(), expected_tpr)
                np.testing.assert_allclose(computed.false_positives.compute(), expected_fpr)
