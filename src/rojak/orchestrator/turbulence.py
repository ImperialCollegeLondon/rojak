from datetime import datetime
from typing import TYPE_CHECKING, Mapping, assert_never

from pydantic import TypeAdapter

from rojak.core import data
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import (
    TurbulenceCalibrationPhaseOption,
    TurbulenceEvaluationConfig,
    TurbulenceEvaluationPhaseOption,
    TurbulenceEvaluationPhases,
    TurbulenceThresholds,
)
from rojak.turbulence.analysis import (
    CorrelationBetweenDiagnostics,
    HistogramData,
    LatitudinalCorrelationBetweenDiagnostics,
)
from rojak.turbulence.diagnostic import CalibrationDiagnosticSuite, DiagnosticFactory, EvaluationDiagnosticSuite
from rojak.utilities.types import DistributionParameters

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

import logging

logger = logging.getLogger(__name__)

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

    def launch(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: Mapping
    ) -> Mapping[TurbulenceCalibrationPhaseOption, Result]:
        suite: CalibrationDiagnosticSuite | None = (
            self.create_diagnostic_suite(diagnostics, chunks) if self._config.calibration_data_dir is not None else None
        )

        return {phase: self.run_phase(phase, suite) for phase in self._phases.phases}

    def create_diagnostic_suite(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: Mapping
    ) -> "CalibrationDiagnosticSuite":
        assert self._config.calibration_data_dir is not None
        logger.debug("Loading CATData")
        calibration_data: "CATData" = Era5Data(
            data.load_from_folder(self._config.calibration_data_dir, chunks=chunks),
        ).to_clear_air_turbulence_data(self._spatial_domain)
        logger.debug("Instantiating CalibrationDiagnosticSuite")
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
    _phases: list[TurbulenceEvaluationPhaseOption]
    _config: "TurbulenceEvaluationConfig"
    _spatial_domain: "SpatialDomain"
    _plots_dir: "Path"
    _name: RunName
    _start_time: TimeStr

    def __init__(
        self,
        calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result],
        phases_config: "TurbulenceEvaluationPhases",
        domain: "SpatialDomain",
        plots_dir: "Path",
        name: RunName,
        start_time: TimeStr,
    ) -> None:
        self._calibration_result = calibration_result
        self._phases = phases_config.phases
        self._config = phases_config.evaluation_config
        self._spatial_domain = domain
        self._name = name
        self._start_time = start_time
        self._plots_dir = plots_dir

    def launch(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: dict
    ) -> Mapping[TurbulenceEvaluationPhaseOption, Result]:
        suite: EvaluationDiagnosticSuite = self.create_diagnostic_suite(diagnostics, chunks)
        return {phase: self.run_phase(phase, suite) for phase in self._phases}

    def create_diagnostic_suite(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: Mapping
    ) -> EvaluationDiagnosticSuite:
        assert self._config.evaluation_data_dir is not None
        logger.debug("Loading CATData")
        evaluation_data: "CATData" = Era5Data(
            data.load_from_folder(self._config.evaluation_data_dir, chunks=chunks),
        ).to_clear_air_turbulence_data(self._spatial_domain)
        if TurbulenceCalibrationPhaseOption.HISTOGRAM in self._calibration_result:
            dist_params = {
                name: DistributionParameters(histogram_data.mean, histogram_data.variance)
                for name, histogram_data in self._calibration_result[
                    TurbulenceCalibrationPhaseOption.HISTOGRAM
                ].result.items()  # DiagnosticName, HistogramData
            }
        else:
            dist_params = None
        logger.debug("Instantiating EvaluationDiagnosticSuite")
        return EvaluationDiagnosticSuite(
            DiagnosticFactory(evaluation_data),
            diagnostics,
            severities=self._config.severities,
            pressure_levels=self._config.pressure_levels,
            probability_thresholds=self._calibration_result[TurbulenceCalibrationPhaseOption.THRESHOLDS].result,
            threshold_mode=self._config.threshold_mode,
            distribution_parameters=dist_params,
        )

    def run_phase(self, phase: TurbulenceEvaluationPhaseOption, suite: EvaluationDiagnosticSuite) -> Result:  # noqa: PLR0911
        match phase:
            case TurbulenceEvaluationPhaseOption.PROBABILITIES:
                return Result(suite.probabilities)
            case TurbulenceEvaluationPhaseOption.EDR:
                return Result(suite.edr)
            case TurbulenceEvaluationPhaseOption.TURBULENT_REGIONS:
                return Result(suite.compute_turbulent_regions())
            case TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES:
                return Result(
                    CorrelationBetweenDiagnostics(
                        dict(suite.probabilities),
                        {"pressure_level": self._config.pressure_levels, "threshold": self._config.severities},
                    )
                )
            case TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR:
                return Result(
                    CorrelationBetweenDiagnostics(
                        dict(suite.edr),
                        {"pressure_level": self._config.pressure_levels, "threshold": self._config.severities},
                    )
                )
            case TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_PROBABILITIES:
                sel_condition: dict = {"pressure_level": self._config.pressure_levels}
                if not self._config.severities:  # Add a check that "threshold" is an axis
                    sel_condition["threshold"] = self._config.severities
                return Result(
                    # Add in config to specify hemisphere and regions
                    LatitudinalCorrelationBetweenDiagnostics(dict(suite.probabilities), sel_condition)
                )
            case TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_EDR:
                return Result(
                    # Add in config to specify hemisphere and regions
                    LatitudinalCorrelationBetweenDiagnostics(
                        dict(suite.edr), {"pressure_level": self._config.pressure_levels}
                    )
                )
            case _ as unreachable:
                assert_never(unreachable)


class TurbulenceLauncher:
    _config: "TurbulenceConfig"
    _context: "ConfigContext"

    def __init__(self, context: "ConfigContext") -> None:
        self._context = context
        assert context.turbulence_config is not None
        self._config = context.turbulence_config
        self._start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    def launch(self) -> None:
        logger.info("Launching Turbulence Calibration")
        calibration_result = CalibrationStage(
            self._config.phases.calibration_phases,
            self._context.data_config.spatial_domain,
            self._context.output_dir,
            self._context.name,
            self._start_time,
        ).launch(self._config.diagnostics, self._config.chunks)
        logger.info("Finished Turbulence")
        if self._config.phases.evaluation_phases is not None:
            EvaluationStage(
                calibration_result,
                self._config.phases.evaluation_phases,
                self._context.data_config.spatial_domain,
                self._context.plots_dir,
                self._context.name,
                self._start_time,
            ).launch(self._config.diagnostics, self._config.chunks)
