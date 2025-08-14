#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import itertools
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final, NamedTuple, assert_never

import dask_geopandas as dgpd
import numpy as np
import xarray as xr
from pydantic import TypeAdapter

from rojak.core import data
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.datalib.madis.amdar import AcarsAmdarRepository
from rojak.datalib.ukmo.amdar import UkmoAmdarRepository
from rojak.orchestrator.configuration import (
    AmdarDataSource,
    AmdarDiagnosticCmpSource,
    TurbulenceCalibrationPhaseOption,
    TurbulenceEvaluationPhaseOption,
    TurbulenceThresholds,
)
from rojak.plot.turbulence_plotter import (
    create_diagnostic_correlation_plot,
    create_interactive_aggregated_auc_plots,
    create_interactive_heatmap_plot,
    create_interactive_roc_curve_plot,
    create_multi_region_correlation_plot,
    create_multi_turbulence_diagnotics_probability_plot,
    save_hv_plot,
)
from rojak.turbulence.analysis import (
    CorrelationBetweenDiagnostics,
    HistogramData,
    LatitudinalCorrelationBetweenDiagnostics,
)
from rojak.turbulence.diagnostic import (
    CalibrationDiagnosticSuite,
    DiagnosticFactory,
    DiagnosticSuite,
    EvaluationDiagnosticSuite,
)
from rojak.turbulence.verification import (
    DiagnosticsAmdarDataHarmoniser,
    DiagnosticsAmdarVerification,
    SpatialGroupByStrategy,
)
from rojak.utilities.types import DistributionParameters, Limits

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd

    from rojak.core.data import AmdarDataRepository, CATData
    from rojak.orchestrator.configuration import Context as ConfigContext
    from rojak.orchestrator.configuration import (
        DataConfig,
        DiagnosticsAmdarHarmonisationStrategyOptions,
        DiagnosticValidationCondition,
        SpatialDomain,
        TurbulenceCalibrationConfig,
        TurbulenceCalibrationPhases,
        TurbulenceConfig,
        TurbulenceDiagnostics,
        TurbulenceEvaluationConfig,
        TurbulenceEvaluationPhases,
    )
    from rojak.turbulence.verification import (
        RocVerificationResult,
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


# See pydantic docs about only instantiating the type adapter once
# https://docs.pydantic.dev/latest/concepts/performance/#typeadapter-instantiated-once
# str is DiagnosticName
THRESHOLDS_TYPE_ADAPTER: TypeAdapter = TypeAdapter(dict[str, TurbulenceThresholds])
DISTRIBUTION_PARAMS_TYPE_ADAPTER: TypeAdapter = TypeAdapter(dict[str, HistogramData])


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

        result = {phase: self.run_phase(phase, suite) for phase in self._phases.phases}

        del suite  # "teardown" try to get python to release memory related to calibration data
        return result

    def create_diagnostic_suite(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: Mapping
    ) -> "CalibrationDiagnosticSuite":
        assert self._config.calibration_data_dir is not None
        logger.debug("Loading CATData")
        calibration_data: CATData = Era5Data(
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

    def load_thresholds_from_file(self) -> Result[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        assert self._config.thresholds_file_path is not None
        json_str: str = self._config.thresholds_file_path.read_text()
        thresholds = THRESHOLDS_TYPE_ADAPTER.validate_json(json_str)
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
        target_dir: Path = (self._output_dir / self._name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file: Path = target_dir / f"thresholds_{self._start_time}.json"

        with output_file.open("wb") as output_json:
            output_json.write(THRESHOLDS_TYPE_ADAPTER.dump_json(diagnostic_thresholds, indent=4))

    def load_distribution_parameters_from_file(self) -> Result:
        assert self._config.diagnostic_distribution_file_path is not None
        json_str: str = self._config.diagnostic_distribution_file_path.read_text()
        distribution_parameters = DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(json_str)
        return Result(distribution_parameters)

    def export_distribution_parameters(self, diagnostic_thresholds: Mapping["DiagnosticName", "HistogramData"]) -> None:
        target_dir: Path = (self._output_dir / self._name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file: Path = target_dir / f"distribution_params_{self._start_time}.json"

        with output_file.open("wb") as output_json:
            output_json.write(DISTRIBUTION_PARAMS_TYPE_ADAPTER.dump_json(diagnostic_thresholds, indent=4))

    def compute_distribution_parameters(self, suite: CalibrationDiagnosticSuite | None) -> Result:
        assert suite is not None
        distribution_parameters = suite.compute_distribution_parameters()
        self.export_distribution_parameters(distribution_parameters)
        return Result(distribution_parameters)


def _abbreviate_diagnostic_name(diagnostic_name: str) -> str:
    if len(diagnostic_name) > 3:  # noqa: PLR2004
        if diagnostic_name == "ncsu1" or diagnostic_name[:3] == "ngm":
            return diagnostic_name
        if "-" in diagnostic_name:
            return "".join([name[0] for name in diagnostic_name.split("-")])
        return diagnostic_name[:3]
    return diagnostic_name


def _chain_diagnostic_names(diagnostic_names: Iterable[str]) -> str:
    joined: str = "_".join(diagnostic_names)
    # max filename length is 255 chars or bytes depending on file system
    # Remove 55 chars as a safety factor (might stuff before this)
    max_file_chars: Final[int] = 200
    if len(joined) >= max_file_chars or sys.getsizeof(joined) >= max_file_chars:
        abbreviated_names: list[str] = [_abbreviate_diagnostic_name(name) for name in diagnostic_names]
        abbrev_joined: str = "_".join(abbreviated_names)
        # Did a quick test with all available diagnostics and that totals to 110 so this should always pass
        assert len(abbrev_joined) < max_file_chars and sys.getsizeof(abbrev_joined) < max_file_chars, (  # noqa: PT018
            "Name of joined abbreviated diagnostics should be shorter than 255 bytes"
        )
        return abbrev_joined
    return joined


class EvaluationStageResult(NamedTuple):
    suite: EvaluationDiagnosticSuite
    phase_outcomes: Mapping[TurbulenceEvaluationPhaseOption, Result]


class EvaluationStage:
    _calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result]
    _phases: list[TurbulenceEvaluationPhaseOption]
    _config: "TurbulenceEvaluationConfig"
    _spatial_domain: "SpatialDomain"
    _plots_dir: "Path"
    _start_time: TimeStr
    _image_format: str

    def __init__(
        self,
        calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result],
        phases_config: "TurbulenceEvaluationPhases",
        domain: "SpatialDomain",
        plots_dir: "Path",
        name: RunName,
        start_time: TimeStr,
        image_format: str,
    ) -> None:
        self._calibration_result = calibration_result
        self._phases = phases_config.phases
        self._config = phases_config.evaluation_config
        self._spatial_domain = domain
        self._start_time = start_time
        self._plots_dir = plots_dir / name
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._image_format = image_format

    def launch(self, diagnostics: list["TurbulenceDiagnostics"], chunks: dict) -> EvaluationStageResult:
        suite: EvaluationDiagnosticSuite = self.create_diagnostic_suite(diagnostics, chunks)
        return EvaluationStageResult(suite, {phase: self.run_phase(phase, suite) for phase in self._phases})

    def create_diagnostic_suite(
        self, diagnostics: list["TurbulenceDiagnostics"], chunks: Mapping
    ) -> EvaluationDiagnosticSuite:
        assert self._config.evaluation_data_dir is not None
        logger.debug("Loading CATData")
        evaluation_data: CATData = Era5Data(
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
            probability_thresholds=self._calibration_result[TurbulenceCalibrationPhaseOption.THRESHOLDS].result
            if TurbulenceCalibrationPhaseOption.THRESHOLDS in self._calibration_result
            else None,
            threshold_mode=self._config.threshold_mode,
            distribution_parameters=dist_params,
        )

    def run_phase(self, phase: TurbulenceEvaluationPhaseOption, suite: EvaluationDiagnosticSuite) -> Result:  # noqa: PLR0911
        match phase:
            case TurbulenceEvaluationPhaseOption.PROBABILITIES:
                result = suite.probabilities
                for pressure_level, severity in itertools.product(
                    self._config.pressure_levels, self._config.severities
                ):
                    chained_names: str = _chain_diagnostic_names(result.keys())
                    create_multi_turbulence_diagnotics_probability_plot(
                        xr.Dataset(
                            data_vars={
                                name: diagnostic.sel(pressure_level=pressure_level, severity=severity)
                                for name, diagnostic in result.items()
                            }
                        ),
                        suite.diagnostic_names(),
                        str(
                            self._plots_dir / f"multi_diagnostic_{chained_names}_on_{pressure_level:.0f}_{severity}"
                            f".{self._image_format}"
                        ),
                    )
                return Result(result)
            case TurbulenceEvaluationPhaseOption.EDR:
                return Result(suite.edr)
            case TurbulenceEvaluationPhaseOption.TURBULENT_REGIONS:
                return Result(suite.compute_turbulent_regions())
            case (
                TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES
                | TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR
            ):
                correlation_on = (
                    suite.probabilities
                    if phase == TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES
                    else suite.edr
                )
                condition: dict[str, list] = {"pressure_level": self._config.pressure_levels}
                if phase == TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES:
                    condition["severity"] = [str(sev) for sev in self._config.severities]
                    corr_on_what = "probability"
                else:
                    corr_on_what = "edr"
                correlation = CorrelationBetweenDiagnostics(dict(correlation_on), condition).execute()
                chained_names: str = _chain_diagnostic_names(correlation_on.keys())
                create_diagnostic_correlation_plot(
                    correlation,
                    str(self._plots_dir / f"corr_{corr_on_what}_btw_{chained_names}.{self._image_format}"),
                    "diagnostic1",
                    "diagnostic2",
                )
                return Result(correlation)
            case (
                TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_PROBABILITIES
                | TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_EDR
            ):
                sel_condition: dict = {"pressure_level": self._config.pressure_levels}
                if phase == TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_PROBABILITIES:
                    # Add a check that "threshold" is an axis
                    if not self._config.severities:
                        sel_condition["threshold"] = self._config.severities
                    corr_on_what = "probability"
                else:
                    corr_on_what = "edr"

                correlation_on = (
                    suite.probabilities
                    if phase == TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES
                    else suite.edr
                )
                # TODO: Add in config to specify hemisphere and regions
                correlation = LatitudinalCorrelationBetweenDiagnostics(dict(correlation_on), sel_condition).execute()
                chained_names: str = _chain_diagnostic_names(correlation_on.keys())
                create_multi_region_correlation_plot(
                    correlation,
                    str(self._plots_dir / f"regional_{corr_on_what}_corr_btw_{chained_names}.{self._image_format}"),
                    "diagnostic1",
                    "diagnostic2",
                )
                return Result(correlation)
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

    def launch(self) -> None | EvaluationStageResult:
        logger.info("Launching Turbulence Calibration")
        calibration_result = CalibrationStage(
            self._config.phases.calibration_phases,
            self._context.data_config.spatial_domain,
            self._context.output_dir,
            self._context.name,
            self._start_time,
        ).launch(self._config.diagnostics, self._config.chunks)
        logger.info("Finished Turbulence")
        result: EvaluationStageResult | None = (
            EvaluationStage(
                calibration_result,
                self._config.phases.evaluation_phases,
                self._context.data_config.spatial_domain,
                self._context.plots_dir,
                self._context.name,
                self._start_time,
                self._context.image_format,
            ).launch(self._config.diagnostics, self._config.chunks)
            if self._config.phases.evaluation_phases is not None
            else None
        )
        if result is not None:
            logger.info("Finished Turbulence Evaluation")

        if self._context.data_config.amdar_config is not None:
            if self._context.data_config.amdar_config.diagnostics_from == AmdarDiagnosticCmpSource.CALIBRATION:
                # I will figure out the plumbing for this later
                raise NotImplementedError("Comparing amdar data with calibration data not yet supported")

            assert self._context.turbulence_config is not None
            assert self._context.turbulence_config.phases.evaluation_phases is not None, (
                "Code path should not be possible"
            )

            assert result is not None, "Pydantic checks on config should prevent this assert from failing"
            # if evaluation phases are empty, trigger pre-compute
            DiagnosticsAmdarLauncher(
                self._context.data_config,
                self._context.output_dir,
                self._context.plots_dir,
                self._context.name,
                not self._context.turbulence_config.phases.evaluation_phases.phases,
            ).launch(result.suite)

        return result


# PUT THIS IN THIS FILE FOR NOW
class DiagnosticsAmdarLauncher:
    _path_to_files: str
    _data_source: AmdarDataSource
    _spatial_domain: "SpatialDomain"
    _strategies: list["DiagnosticsAmdarHarmonisationStrategyOptions"]
    _time_window: "Limits[datetime]"
    _output_filepath: "Path"
    _plots_dir: "Path"
    _save_output: bool
    _validation_conditions: list["DiagnosticValidationCondition"]
    _trigger_diagnostics_compute: bool

    def __init__(
        self,
        data_config: "DataConfig",
        output_dir: "Path",
        plots_dir: "Path",
        run_name: "RunName",
        trigger_diagnostic_compute: bool,
    ) -> None:
        assert data_config.amdar_config is not None
        self._data_source = data_config.amdar_config.data_source
        self._path_to_files = str(data_config.amdar_config.data_dir.resolve() / data_config.amdar_config.glob_pattern)
        self._spatial_domain = data_config.spatial_domain
        self._strategies = (
            data_config.amdar_config.harmonisation_strategies
            if data_config.amdar_config.harmonisation_strategies is not None
            else []
        )
        self._time_window = data_config.amdar_config.time_window
        self._validation_conditions = (
            []
            if data_config.amdar_config.diagnostic_validation is None
            else data_config.amdar_config.diagnostic_validation.validation_conditions
        )
        self._trigger_diagnostics_compute = trigger_diagnostic_compute

        self._save_output = data_config.amdar_config.save_harmonised_data
        base_dir = output_dir / run_name / "data_harmonisation"
        if self._save_output:
            base_dir.mkdir(parents=True, exist_ok=True)
        time_format: str = "%Y-%m-%dT%H%M"  # e.g. 2025-02-31T1200
        self._output_filepath = (
            base_dir / f"{self._data_source}_{self._time_window.lower.strftime(time_format)}"
            f"_{self._time_window.upper.strftime(time_format)}.parquet"
        )
        self._plots_dir = plots_dir / run_name / str(self._data_source)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

    def create_amdar_data_repository(self) -> "AmdarDataRepository":
        match self._data_source:
            case AmdarDataSource.MADIS:
                return AcarsAmdarRepository(self._path_to_files)
            case AmdarDataSource.UKMO:
                return UkmoAmdarRepository(self._path_to_files)
            case _ as unreachable:
                assert_never(unreachable)

    def launch(self, diagnostic_suite: DiagnosticSuite) -> None:
        if self._spatial_domain.grid_size is None:
            raise ValueError("Grid size for spatial domain must be specified for diagnostics amdar data harmonisation")

        logger.info("Started Turbulence Amdar Harmonisation")
        amdar_data = self.create_amdar_data_repository().to_amdar_turbulence_data(
            self._spatial_domain,
            self._spatial_domain.grid_size,
            diagnostic_suite.get_prototype_computed_diagnostic()["pressure_level"].data,
        )
        harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data, diagnostic_suite)
        time_window_as_np_datetime: Limits[np.datetime64] = Limits(
            np.datetime64(self._time_window.lower), np.datetime64(self._time_window.upper)
        )

        if self._trigger_diagnostics_compute:
            for diagnostic_name, _ in diagnostic_suite.computed_values("Trigger computation of turbulence diagnostics"):
                logger.debug("Computed CAT diagnostic %s", diagnostic_name)

        if self._strategies:
            result: dd.DataFrame = harmoniser.execute_harmonisation(
                self._strategies,
                time_window_as_np_datetime,
            ).persist()
            logger.info("Finished Turbulence Amdar Harmonisation")
            if self._save_output:
                logger.info("Saving results to %s", self._output_filepath)
                result.drop(columns="geometry").set_index(harmoniser.grid_box_column_name).to_parquet(
                    self._output_filepath
                )

        if self._validation_conditions:
            logger.info("Starting validation of diagnostics with amdar data")
            verifier = DiagnosticsAmdarVerification(harmoniser, time_window_as_np_datetime)
            chained_names: str = _chain_diagnostic_names(harmoniser.harmonised_diagnostics)
            group_by_strategy = SpatialGroupByStrategy.HORIZONTAL_BOX
            grid_auc = verifier.aggregate_by_auc(
                self._validation_conditions,
                diagnostic_suite.get_prototype_computed_diagnostic(),
                group_by_strategy,
            )
            num_observations = verifier.num_obs_per(self._validation_conditions, group_by_strategy)
            is_agg_by_point: bool = group_by_strategy in {
                SpatialGroupByStrategy.GRID_POINT,
                SpatialGroupByStrategy.HORIZONTAL_POINT,
            }
            if not is_agg_by_point:
                for diagnostic_name in grid_auc:
                    grid_auc[diagnostic_name] = (
                        dgpd.from_dask_dataframe(
                            grid_auc[diagnostic_name].join(amdar_data.grid, on=harmoniser.grid_box_column_name)
                        )
                        .set_crs(4326)
                        .drop(columns=[harmoniser.grid_box_column_name])
                    )
                num_observations = dgpd.from_dask_dataframe(
                    num_observations.join(amdar_data.grid, on=harmoniser.grid_box_column_name)
                ).set_crs(4326)[["num_obs", "geometry"]]
            auc_plots = create_interactive_aggregated_auc_plots(
                grid_auc,
                self._validation_conditions,
                is_agg_by_point,
            )
            save_hv_plot(
                auc_plots,
                str(self._plots_dir / f"regional_auc_{chained_names}_on_{str(group_by_strategy)}"),
                "png",
                savefig_kwargs={"dpi": 400},
            )

            save_hv_plot(
                create_interactive_heatmap_plot(
                    num_observations if is_agg_by_point else num_observations.compute(),
                    "num_obs",
                    opts_kwargs={"fig_size": 400, "title": "Total Number of Observations", "lw": 0},
                ),
                str(self._plots_dir / f"num_observations_for_{str(group_by_strategy)}"),
                "png",
                savefig_kwargs={"dpi": 400},
            )

            roc: RocVerificationResult = verifier.nearest_value_roc(self._validation_conditions)
            roc_curve_plots: dict = create_interactive_roc_curve_plot(roc)
            chained_names: str = _chain_diagnostic_names(harmoniser.harmonised_diagnostics)
            for amdar_col, plot_for_col in roc_curve_plots.items():
                save_hv_plot(plot_for_col, str(self._plots_dir / f"roc_{amdar_col}_on_{chained_names}"), "png")
                logger.debug("Saved roc plot for %s AMDAR turbulence measure", amdar_col)

            logger.info("Finished validation of diagnostics with amdar data")
