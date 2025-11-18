import datetime
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import dask.dataframe as dd
import pytest

from rojak.datalib.madis.amdar import AcarsAmdarTurbulenceData
from rojak.orchestrator.configuration import (
    AggregationMetricOption,
    AmdarConfig,
    AmdarDataSource,
    DataConfig,
    DiagnosticValidationCondition,
    DiagnosticValidationConfig,
    SpatialDomain,
    SpatialGroupByStrategy,
    TurbulenceCalibrationConfig,
    TurbulenceCalibrationPhaseOption,
    TurbulenceCalibrationPhases,
    TurbulenceConfig,
    TurbulenceDiagnostics,
    TurbulenceEvaluationConfig,
    TurbulenceEvaluationPhaseOption,
    TurbulenceEvaluationPhases,
    TurbulencePhases,
    TurbulenceThresholds,
)
from rojak.orchestrator.configuration import (
    Context as ConfigContext,
)
from rojak.orchestrator.turbulence import DISTRIBUTION_PARAMS_TYPE_ADAPTER, THRESHOLDS_TYPE_ADAPTER, TurbulenceLauncher
from rojak.utilities.types import Limits
from tests.integration.conftest import randomly_select_diagnostics

if TYPE_CHECKING:
    from rojak.turbulence.analysis import HistogramData


@pytest.fixture
def create_calibration_only_config(create_config_context) -> Callable:
    def _calibration_only_config(num_diagnostics: int) -> ConfigContext:
        diagnostics: list[TurbulenceDiagnostics] = randomly_select_diagnostics(num_diagnostics)
        turbulence_config = TurbulenceConfig(
            chunks={"pressure_level": 3, "latitude": 721, "longitude": 1440, "valid_time": 2},
            diagnostics=diagnostics,
            phases=TurbulencePhases(
                calibration_phases=TurbulenceCalibrationPhases(
                    phases=[
                        TurbulenceCalibrationPhaseOption.THRESHOLDS,
                        TurbulenceCalibrationPhaseOption.HISTOGRAM,
                    ],
                    calibration_config=TurbulenceCalibrationConfig(
                        calibration_data_dir=Path("tests/_static/"),
                        percentile_thresholds=TurbulenceThresholds(light=0.97, light_to_moderate=98.0, moderate=99.0),
                    ),
                )
            ),
        )
        return create_config_context("calibration_only", turb_config=turbulence_config)

    return _calibration_only_config


@pytest.fixture
def create_evaluation_config_restore_from_outputs(create_config_context) -> Callable:
    def _evaluation_config_restore_from_outputs(
        diagnostics: list[TurbulenceDiagnostics], path_to_thresholds: Path, path_to_distribution_params: Path
    ) -> ConfigContext:
        turbulence_config = TurbulenceConfig(
            chunks={"pressure_level": 3, "latitude": 721, "longitude": 1440, "valid_time": 2},
            diagnostics=diagnostics,
            phases=TurbulencePhases(
                calibration_phases=TurbulenceCalibrationPhases(
                    phases=[
                        TurbulenceCalibrationPhaseOption.THRESHOLDS,
                        TurbulenceCalibrationPhaseOption.HISTOGRAM,
                    ],
                    calibration_config=TurbulenceCalibrationConfig(
                        thresholds_file_path=path_to_thresholds,
                        diagnostic_distribution_file_path=path_to_distribution_params,
                    ),
                ),
                evaluation_phases=TurbulenceEvaluationPhases(
                    phases=[
                        TurbulenceEvaluationPhaseOption.PROBABILITIES,
                        TurbulenceEvaluationPhaseOption.EDR,
                        TurbulenceEvaluationPhaseOption.TURBULENT_REGIONS,
                        TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES,
                        # Transforming to EDR on junk data results in NaNs breaking the clustered correlations
                        # TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR,
                    ],
                    evaluation_config=TurbulenceEvaluationConfig(
                        evaluation_data_dir=Path("tests/_static/"),
                    ),
                ),
            ),
        )
        return create_config_context("evaluation_restore_from_outputs", turb_config=turbulence_config)

    return _evaluation_config_restore_from_outputs


class CalibrationOutputFiles(NamedTuple):
    thresholds: Path | None
    distribution: Path | None


def marry_up_output_files(files: list[Path]) -> CalibrationOutputFiles:
    match files:
        case [file_path] if file_path.stem.startswith("thresholds"):
            return CalibrationOutputFiles(thresholds=files[0], distribution=None)
        case [file_path] if file_path.stem.startswith("distribution_params"):
            return CalibrationOutputFiles(thresholds=None, distribution=files[0])
        case [file_path1, file_path2] if file_path1.stem.startswith("thresholds") and file_path2.stem.startswith(
            "distribution_params"
        ):
            return CalibrationOutputFiles(thresholds=files[0], distribution=files[1])
        case [file_path1, file_path2] if file_path2.stem.startswith("thresholds") and file_path1.stem.startswith(
            "distribution_params"
        ):
            return CalibrationOutputFiles(thresholds=files[1], distribution=files[0])
        case _ as unreachable:
            raise ValueError(f"File path name combination ({unreachable}) should be unreachable")


def test_turbulence_calibration_only(create_calibration_only_config: Callable, client):
    calibration_config: ConfigContext = create_calibration_only_config(6)
    assert calibration_config.turbulence_config is not None

    TurbulenceLauncher(calibration_config).launch()

    output_dir: Path = calibration_config.output_dir / calibration_config.name
    assert output_dir.exists()
    assert output_dir.is_dir()

    output_json_files = list(output_dir.glob("*.json"))
    assert len(output_json_files) == len(calibration_config.turbulence_config.phases.calibration_phases.phases)
    output_files = marry_up_output_files(output_json_files)
    assert output_files.thresholds is not None
    assert output_files.distribution is not None

    distribution_params: dict[str, HistogramData] = DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(
        output_files.distribution.read_text()
    )
    thresholds: dict[str, TurbulenceThresholds] = THRESHOLDS_TYPE_ADAPTER.validate_json(
        output_files.thresholds.read_text()
    )
    calibrated_diagnostics = set(calibration_config.turbulence_config.diagnostics)
    assert calibrated_diagnostics.intersection(thresholds.keys()) == calibrated_diagnostics.union(thresholds.keys())
    assert calibrated_diagnostics.intersection(distribution_params.keys()) == calibrated_diagnostics.union(
        distribution_params.keys()
    )


def test_turbulence_evaluation_restore_from_file(
    create_calibration_only_config, client, create_evaluation_config_restore_from_outputs
):
    calibration_config: ConfigContext = create_calibration_only_config(4)
    assert calibration_config.turbulence_config is not None

    TurbulenceLauncher(calibration_config).launch()
    output_dir: Path = calibration_config.output_dir / calibration_config.name
    output_json_files = list(output_dir.glob("*.json"))
    assert len(output_json_files) == len(calibration_config.turbulence_config.phases.calibration_phases.phases)
    output_files = marry_up_output_files(output_json_files)
    assert output_files.thresholds is not None
    assert output_files.distribution is not None

    evaluation_config: ConfigContext = create_evaluation_config_restore_from_outputs(
        calibration_config.turbulence_config.diagnostics, output_files.thresholds, output_files.distribution
    )
    assert evaluation_config.turbulence_config is not None
    assert evaluation_config.turbulence_config.phases is not None
    assert evaluation_config.turbulence_config.phases.evaluation_phases is not None

    result = TurbulenceLauncher(evaluation_config).launch()
    assert result is not None
    assert set(evaluation_config.turbulence_config.phases.evaluation_phases.phases) == set(result.phase_outcomes.keys())


@pytest.mark.cdsapi
@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test is so slow, runners time out")
def test_turbulence_calibration_and_evaluation(create_config_context, client, retrieve_era5_cat_data) -> None:
    diagnostics: list[TurbulenceDiagnostics] = randomly_select_diagnostics(4)
    eval_phases = [
        TurbulenceEvaluationPhaseOption.PROBABILITIES,
        TurbulenceEvaluationPhaseOption.EDR,
        TurbulenceEvaluationPhaseOption.TURBULENT_REGIONS,
        TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES,
        # TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR,
        # TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_PROBABILITIES,
        # TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_EDR,
    ]
    turbulence_config = TurbulenceConfig(
        chunks={"pressure_level": 6, "latitude": 721, "longitude": 1440, "valid_time": 3},
        diagnostics=diagnostics,
        phases=TurbulencePhases(
            calibration_phases=TurbulenceCalibrationPhases(
                phases=[
                    TurbulenceCalibrationPhaseOption.THRESHOLDS,
                    TurbulenceCalibrationPhaseOption.HISTOGRAM,
                ],
                calibration_config=TurbulenceCalibrationConfig(
                    # calibration_data_dir=Path("tests/_static/"),
                    calibration_data_dir=retrieve_era5_cat_data,
                    percentile_thresholds=TurbulenceThresholds(light=0.97, light_to_moderate=98.0, moderate=99.0),
                ),
            ),
            evaluation_phases=TurbulenceEvaluationPhases(
                phases=eval_phases,
                evaluation_config=TurbulenceEvaluationConfig(
                    # evaluation_data_dir=Path("tests/_static/"),
                    evaluation_data_dir=retrieve_era5_cat_data,
                ),
            ),
        ),
    )
    config = create_config_context("calibration_and_evaluation", turb_config=turbulence_config)
    launcher_result = TurbulenceLauncher(config).launch()
    assert launcher_result is not None
    assert set(eval_phases) == set(launcher_result.phase_outcomes.keys())


@pytest.mark.cdsapi
@pytest.mark.xfail(reason="The code path within the launcher to invoke the harmonisation has been removed")
def test_turbulence_amdar_acars_harmonisation(
    create_config_context, client, retrieve_era5_cat_data, retrieve_single_day_madis_data
) -> None:
    diagnostics: list[TurbulenceDiagnostics] = randomly_select_diagnostics(2)
    turbulence_config = TurbulenceConfig(
        chunks={"pressure_level": 6, "latitude": 721, "longitude": 1440, "valid_time": 3},
        diagnostics=diagnostics,
        phases=TurbulencePhases(
            calibration_phases=TurbulenceCalibrationPhases(
                phases=[
                    TurbulenceCalibrationPhaseOption.THRESHOLDS,
                    TurbulenceCalibrationPhaseOption.HISTOGRAM,
                ],
                calibration_config=TurbulenceCalibrationConfig(
                    calibration_data_dir=retrieve_era5_cat_data,
                    percentile_thresholds=TurbulenceThresholds(light=0.97, light_to_moderate=98.0, moderate=99.0),
                ),
            ),
            evaluation_phases=TurbulenceEvaluationPhases(
                phases=[],  # Don't specify EDR as it should automatically calculate due to how code is structured
                evaluation_config=TurbulenceEvaluationConfig(
                    evaluation_data_dir=retrieve_era5_cat_data,
                ),
            ),
        ),
    )
    data_config = DataConfig(
        spatial_domain=SpatialDomain(
            minimum_latitude=25, maximum_latitude=54, minimum_longitude=-125, maximum_longitude=2, grid_size=0.25
        ),
        amdar_config=AmdarConfig(
            data_dir=retrieve_single_day_madis_data,
            data_source=AmdarDataSource.MADIS,
            glob_pattern="**/*.parquet",
            time_window=Limits(
                datetime.datetime(2024, month=1, day=1), datetime.datetime(2024, month=1, day=1, hour=18)
            ),
            save_harmonised_data=True,
        ),
    )
    config: ConfigContext = create_config_context(
        "diagnostic_amdar_harmonisation_acars", turb_config=turbulence_config, data_config=data_config
    )
    launcher_result = TurbulenceLauncher(config).launch()
    assert launcher_result is not None

    harmonisation_output_dir = config.output_dir / config.name / "data_harmonisation"
    assert harmonisation_output_dir.exists()
    assert harmonisation_output_dir.is_dir()
    parquet_files = harmonisation_output_dir.glob("*.parquet")
    assert list(parquet_files)  # check list is not empty
    loaded_output_data = dd.read_parquet(f"{str(harmonisation_output_dir)}/**/*.parquet")
    loaded_output_data.head()  # evaluate the first few rows to check it is valid


@pytest.mark.parametrize(
    "groupby_strategy",
    [
        SpatialGroupByStrategy.GRID_BOX,
        pytest.param(
            SpatialGroupByStrategy.GRID_POINT,
            marks=pytest.mark.xfail(raises=ValueError, reason="Dataframe for num_obs missing geometry column"),
        ),
        pytest.param(
            SpatialGroupByStrategy.HORIZONTAL_POINT,
            marks=pytest.mark.xfail(raises=ValueError, reason="Dataframe for num_obs missing geometry column"),
        ),
        SpatialGroupByStrategy.HORIZONTAL_BOX,
        None,
    ],
)
@pytest.mark.parametrize(
    "agg_metric", [AggregationMetricOption.TSS, AggregationMetricOption.AUC, AggregationMetricOption.PREVALENCE, None]
)
@pytest.mark.cdsapi
@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test is so slow, runners time out")
def test_turbulence_amdar_roc(
    create_config_context,
    client,
    retrieve_era5_cat_data,
    retrieve_single_day_madis_data,
    groupby_strategy: SpatialGroupByStrategy | None,
    agg_metric: AggregationMetricOption | None,
) -> None:
    if (groupby_strategy is not None and agg_metric is None) or (groupby_strategy is None and agg_metric is not None):
        pytest.skip("Invalid combination from test parametrisation")

    diagnostics: list[TurbulenceDiagnostics] = randomly_select_diagnostics(5)
    turbulence_config = TurbulenceConfig(
        chunks={"pressure_level": 6, "latitude": 721, "longitude": 1440, "valid_time": 3},
        diagnostics=diagnostics,
        phases=TurbulencePhases(
            calibration_phases=TurbulenceCalibrationPhases(
                phases=[],
                calibration_config=TurbulenceCalibrationConfig(
                    calibration_data_dir=retrieve_era5_cat_data,
                    percentile_thresholds=TurbulenceThresholds(light=0.97),
                ),
            ),
            evaluation_phases=TurbulenceEvaluationPhases(
                phases=[],
                evaluation_config=TurbulenceEvaluationConfig(
                    evaluation_data_dir=retrieve_era5_cat_data,
                ),
            ),
        ),
    )
    validation_conditions = [
        DiagnosticValidationCondition(observed_turbulence_column_name=col_name, value_greater_than=0.1)
        for col_name in AcarsAmdarTurbulenceData.turbulence_column_names()
    ]
    data_config = DataConfig(
        spatial_domain=SpatialDomain(
            minimum_latitude=25, maximum_latitude=54, minimum_longitude=-125, maximum_longitude=2, grid_size=0.25
        ),
        amdar_config=AmdarConfig(
            data_dir=retrieve_single_day_madis_data,
            data_source=AmdarDataSource.MADIS,
            glob_pattern="**/*.parquet",
            time_window=Limits(
                datetime.datetime(2024, month=1, day=1), datetime.datetime(2024, month=1, day=1, hour=18)
            ),
            diagnostic_validation=DiagnosticValidationConfig(
                validation_conditions=validation_conditions,
                spatial_group_by_strategy=groupby_strategy,
                aggregation_metric=agg_metric,
            ),
        ),
    )
    config: ConfigContext = create_config_context(
        "diagnostic_amdar_acars_roc", turb_config=turbulence_config, data_config=data_config
    )
    _ = TurbulenceLauncher(config).launch()

    roc_plots_dir = config.plots_dir / config.name / "madis"
    assert roc_plots_dir.exists()
    assert roc_plots_dir.is_dir()

    for condition in validation_conditions:
        matching_files = list(roc_plots_dir.glob(f"roc_{condition.observed_turbulence_column_name}_*.png"))
        assert len(matching_files) == 1
