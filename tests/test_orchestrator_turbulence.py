from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rojak.orchestrator.configuration import (
    SpatialDomain,
    TurbulenceCalibrationConfig,
    TurbulenceCalibrationPhaseOption,
    TurbulenceCalibrationPhases,
    TurbulenceDiagnostics,
    TurbulenceThresholds,
)
from rojak.orchestrator.turbulence import DISTRIBUTION_PARAMS_TYPE_ADAPTER, THRESHOLDS_TYPE_ADAPTER, CalibrationStage
from rojak.turbulence.analysis import HistogramData

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# BUILDER CLASS CAUSED ISSUES WITH VARIABLES BEING SEEN ACROSS TESTS


def with_calibration_data_dir(data_dir, fields) -> dict:
    fields["calibration_data_dir"] = data_dir
    return fields


def with_percentile_threshold(fields) -> dict:
    fields["percentile_thresholds"] = TurbulenceThresholds(light=90, moderate=95, severe=99)
    return fields


def with_dummy_thresholds_file_path(root_path, fields) -> dict:
    fields["thresholds_file_path"] = root_path / "thresholds.json"
    return fields


def with_dummy_diagnostic_distribution(root_path, fields) -> dict:
    fields["diagnostic_distribution_file_path"] = root_path / "diagnostic_distribution.json"
    return fields


def build(fields) -> TurbulenceCalibrationConfig:
    return TurbulenceCalibrationConfig(**fields)


@pytest.fixture
def calibration_config_thresholds_only(tmp_path) -> TurbulenceCalibrationConfig:
    base = {}
    base = with_dummy_thresholds_file_path(tmp_path, base)
    return build(base)


@pytest.fixture
def calibration_config_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    base = {}
    base = with_calibration_data_dir(data_dir, base)
    base = with_percentile_threshold(base)
    return build(base)


def no_phases_calibration(calibration_config: TurbulenceCalibrationConfig):
    return TurbulenceCalibrationPhases(phases=[], calibration_config=calibration_config)


def threshold_phases_calibration(calibration_config: TurbulenceCalibrationConfig):
    return TurbulenceCalibrationPhases(
        phases=[TurbulenceCalibrationPhaseOption.THRESHOLDS], calibration_config=calibration_config
    )


def dist_param_phases_calibration(calibration_config: TurbulenceCalibrationConfig):
    return TurbulenceCalibrationPhases(
        phases=[TurbulenceCalibrationPhaseOption.HISTOGRAM], calibration_config=calibration_config
    )


@pytest.fixture
def output_thresholds() -> dict:
    return {"def": TurbulenceThresholds(light=1, moderate=2, severe=3)}


@pytest.fixture
def output_dist_params() -> dict:
    return {
        "def": HistogramData(
            hist_values=np.asarray([1.0, 2.0, 3.0, 4.0]), bins=np.asarray([4.0, 5.0, 6.0, 7.0]), mean=99.0, variance=0.4
        )
    }


def test_calibration_stage_launch_no_calibration_data(
    mocker: "MockerFixture", tmp_path, calibration_config_thresholds_only
) -> None:
    calibration = CalibrationStage(
        no_phases_calibration(calibration_config_thresholds_only),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path / "output",
        "test",
        datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
    )
    spy_on_run_phase = mocker.spy(calibration, "run_phase")
    calibration.launch([TurbulenceDiagnostics.DEF], {})
    spy_on_run_phase.assert_not_called()


def test_calibration_stage_launch_calibration_data(
    mocker: "MockerFixture", tmp_path_factory, calibration_config_data_dir
):
    calibration = CalibrationStage(
        no_phases_calibration(calibration_config_data_dir),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path_factory.getbasetemp() / "output",
        "test",
        datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
    )
    mock_suite_creation = mocker.patch.object(calibration, "create_diagnostic_suite", return_value=None)
    calibration.launch([TurbulenceDiagnostics.DEF], {})
    mock_suite_creation.assert_called_once_with([TurbulenceDiagnostics.DEF], {})


@pytest.fixture
def dump_to_file(tmp_path_factory, calibration_config_data_dir, mocker: "MockerFixture", output_thresholds):
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    calibration = CalibrationStage(
        threshold_phases_calibration(calibration_config_data_dir),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path_factory.getbasetemp() / "output",
        "test",
        start_time,
    )

    # Mock the CalibrationDiagnosticSuite.compute_thresholds method
    suite_mock = mocker.Mock()
    mocker.patch.object(suite_mock, "compute_thresholds", return_value=output_thresholds)
    mock_suite_creation = mocker.patch.object(calibration, "create_diagnostic_suite", return_value=suite_mock)

    # Call will test the logic in perform_calibration method - includes exporting of data
    calibration.launch([TurbulenceDiagnostics.DEF], {})
    mock_suite_creation.assert_called_once_with([TurbulenceDiagnostics.DEF], {})

    return tmp_path_factory.getbasetemp() / "output" / "test" / f"thresholds_{start_time}.json"


def test_calibration_stage_perform_calibration(
    mocker: "MockerFixture", tmp_path_factory, calibration_config_data_dir, output_thresholds
):
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    calibration = CalibrationStage(
        threshold_phases_calibration(calibration_config_data_dir),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path_factory.getbasetemp() / "output",
        "test",
        start_time,
    )

    # Mock the CalibrationDiagnosticSuite.compute_thresholds method
    suite_mock = mocker.Mock()
    compute_threshold_mock = mocker.patch.object(suite_mock, "compute_thresholds", return_value=output_thresholds)
    mock_suite_creation = mocker.patch.object(calibration, "create_diagnostic_suite", return_value=suite_mock)

    # Call will test the logic in perform_calibration method - includes exporting of data
    calibration.launch([TurbulenceDiagnostics.DEF], {})
    mock_suite_creation.assert_called_once_with([TurbulenceDiagnostics.DEF], {})
    # Verify that mocked method was called with correct values
    compute_threshold_mock.assert_called()
    compute_threshold_mock.assert_called_with(TurbulenceThresholds(light=90, moderate=95, severe=99))

    # Verify that exported threshold file exists
    generated_threshold_file = tmp_path_factory.getbasetemp() / "output" / "test" / f"thresholds_{start_time}.json"
    assert generated_threshold_file.exists()
    assert generated_threshold_file.is_file()

    # Verify the serialisation and deserialisation of the thresholds worked
    instantiated_from_generated = THRESHOLDS_TYPE_ADAPTER.validate_json(generated_threshold_file.read_text())
    assert instantiated_from_generated == output_thresholds


def test_calibration_stage_load_thresholds_from_file(tmp_path_factory, dump_to_file, output_thresholds):
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    calibration = CalibrationStage(
        threshold_phases_calibration(
            TurbulenceCalibrationConfig(thresholds_file_path=dump_to_file),
        ),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path_factory.getbasetemp() / "output",
        "test",
        start_time,
    )
    assert calibration.load_thresholds_from_file().result == output_thresholds


def test_calibration_stage_compute_distribution_params(
    mocker: "MockerFixture",
    tmp_path_factory,
    calibration_config_data_dir,
    output_dist_params,
):
    start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    calibration = CalibrationStage(
        dist_param_phases_calibration(calibration_config_data_dir),
        SpatialDomain(maximum_latitude=90, maximum_longitude=90, minimum_longitude=0, minimum_latitude=0),
        tmp_path_factory.getbasetemp() / "output",
        "test",
        start_time,
    )

    suite_mock = mocker.Mock()
    compute_dist_params_mock = mocker.patch.object(
        suite_mock, "compute_distribution_parameters", return_value=output_dist_params
    )
    mock_suite_creation = mocker.patch.object(calibration, "create_diagnostic_suite", return_value=suite_mock)

    # Call will test the logic in perform_calibration method - includes exporting of data
    calibration.launch([TurbulenceDiagnostics.DEF], {})
    mock_suite_creation.assert_called_once_with([TurbulenceDiagnostics.DEF], {})

    compute_dist_params_mock.assert_called_once()

    generated_dist_params_file = (
        tmp_path_factory.getbasetemp() / "output" / "test" / f"distribution_params_{start_time}.json"
    )
    assert generated_dist_params_file.exists()
    assert generated_dist_params_file.is_file()
    instantiated_from_generated = DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(generated_dist_params_file.read_text())
    assert instantiated_from_generated == output_dist_params
