from datetime import datetime
from typing import TYPE_CHECKING, Mapping, assert_never

from pydantic import TypeAdapter

from rojak.core import data
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import TurbulenceCalibrationPhaseOption, TurbulenceThresholds
from rojak.turbulence.analysis import HistogramData
from rojak.turbulence.diagnostic import CalibrationDiagnosticSuite, DiagnosticFactory

if TYPE_CHECKING:
    from pathlib import Path

    from rojak.core.data import CATData
    from rojak.orchestrator.configuration import Context as ConfigContext
    from rojak.orchestrator.configuration import (
        SpatialDomain,
        TurbulenceCalibrationConfig,
        TurbulenceCalibrationPhases,
        TurbulenceConfig,
        TurbulenceDiagnostics,
    )
    from rojak.utilities.types import DiagnosticName

type RunName = str
type TimeStr = str


class Result[T]:
    _result: T

    def __init__(self, result: T) -> None:
        self._result = result

    @property
    def result(self) -> T:
        return self._result


class CalibrationStage:
    _phases: "TurbulenceCalibrationPhases"
    _config: "TurbulenceCalibrationConfig"
    _domain: "SpatialDomain"
    _output_dir: "Path"
    _name: RunName
    _start_time: TimeStr

    def __init__(
        self,
        phases: "TurbulenceCalibrationPhases",
        spatial_domain: "SpatialDomain",
        output_dir: "Path",
        name: RunName,
        start_time: TimeStr,
    ) -> None:
        self._phases = phases
        self._config = phases.calibration_config
        self._spatial_domain = spatial_domain
        self._output_dir = output_dir
        self._name = name
        self._start_time = start_time

    def launch(self, diagnostics: list["TurbulenceDiagnostics"]) -> Mapping[TurbulenceCalibrationPhaseOption, Result]:
        suite: CalibrationDiagnosticSuite | None = (
            self.create_diagnostic_suite(diagnostics) if self._config.calibration_data_dir is not None else None
        )

        return {phase: self.run_phase(phase, suite) for phase in self._phases.phases}

    def create_diagnostic_suite(self, diagnostics: list["TurbulenceDiagnostics"]) -> "CalibrationDiagnosticSuite":
        assert self._config.calibration_data_dir is not None
        calibration_data: "CATData" = Era5Data(
            data.load_from_folder(self._config.calibration_data_dir),
        ).to_clear_air_turbulence_data(self._spatial_domain)
        return CalibrationDiagnosticSuite(DiagnosticFactory(calibration_data), diagnostics)

    def run_phase(
        self, current_phase: TurbulenceCalibrationPhaseOption, suite: CalibrationDiagnosticSuite | None
    ) -> Result:
        match current_phase:
            case TurbulenceCalibrationPhaseOption.THRESHOLDS:
                if self._config.thresholds_file_path is not None:
                    return self.load_thresholds_from_file()
                return self.perform_calibration(suite)
            case TurbulenceCalibrationPhaseOption.HISTOGRAM:
                if self._config.diagnostic_distribution_file_path is not None:
                    return self.load_distribution_parameters_from_file()
                return self.compute_distribution_parameters(suite)
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def thresholds_type_adapter() -> TypeAdapter:
        # str is DiagnosticName
        return TypeAdapter(dict[str, TurbulenceThresholds])

    @staticmethod
    def distribution_parameters_type_adapter() -> TypeAdapter:
        # str is DiagnosticName
        return TypeAdapter(dict[str, HistogramData])

    def load_thresholds_from_file(self) -> Result[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        assert self._config.thresholds_file_path is not None
        json_str: str = self._config.thresholds_file_path.read_text()
        thresholds = self.thresholds_type_adapter().validate_json(json_str)
        return Result(thresholds)

    def perform_calibration(
        self, suite: CalibrationDiagnosticSuite | None
    ) -> Result[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        assert suite is not None
        assert self._config.percentile_thresholds is not None
        thresholds = suite.compute_thresholds(self._config.percentile_thresholds)
        self.export_thresholds(thresholds)
        return Result(thresholds)

    def export_thresholds(self, diagnostic_thresholds: Mapping["DiagnosticName", "TurbulenceThresholds"]) -> None:
        target_dir: "Path" = (self._output_dir / self._name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file: "Path" = target_dir / f"thresholds_{self._start_time}.json"

        with output_file.open("wb") as output_json:
            output_json.write(self.thresholds_type_adapter().dump_json(diagnostic_thresholds, indent=4))

    def load_distribution_parameters_from_file(self) -> Result:
        assert self._config.diagnostic_distribution_file_path is not None
        json_str: str = self._config.diagnostic_distribution_file_path.read_text()
        distribution_parameters = self.distribution_parameters_type_adapter().validate_json(json_str)
        return Result(distribution_parameters)

    def export_distribution_parameters(self, diagnostic_thresholds: Mapping["DiagnosticName", "HistogramData"]) -> None:
        target_dir: "Path" = (self._output_dir / self._name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file: "Path" = target_dir / f"distribution_params_{self._start_time}.json"

        with output_file.open("wb") as output_json:
            output_json.write(self.distribution_parameters_type_adapter().dump_json(diagnostic_thresholds, indent=4))

    def compute_distribution_parameters(self, suite: CalibrationDiagnosticSuite | None) -> Result:
        assert suite is not None
        distribution_parameters = suite.compute_distribution_parameters()
        self.export_distribution_parameters(distribution_parameters)
        return Result(distribution_parameters)


class EvaluationStage:
    _calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result]

    def __init__(self, calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result]) -> None:
        self._calibration_result = calibration_result


class TurbulenceLauncher:
    _config: "TurbulenceConfig"
    _context: "ConfigContext"

    def __init__(self, context: "ConfigContext") -> None:
        self._context = context
        assert context.turbulence_config is not None
        self._config = context.turbulence_config
        self._start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    def launch(self) -> None:
        calibration_result = CalibrationStage(
            self._config.phases.calibration_phases,
            self._context.data_config.spatial_domain,
            self._context.output_dir,
            self._context.name,
            self._start_time,
        ).launch(self._config.diagnostics)
        if self._config.phases.evaluation_phases is not None:
            EvaluationStage(calibration_result)
