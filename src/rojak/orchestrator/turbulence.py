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
"""
Orchestrates a full turbulence analysis run (``rojak run``): calibration, evaluation, and AMDAR comparison

:class:`TurbulenceLauncher` is the entry point, used by :func:`rojak.cli.main.run`. It runs the calibration stage
(:class:`CalibrationStage`, deriving thresholds and/or EDR distribution parameters from a calibration dataset, or
loading them from file), then, if configured, the evaluation stage (:class:`EvaluationStage`, applying the
calibration results to an evaluation dataset to compute probabilities, EDR, turbulent regions, and diagnostic
correlations, with plots), and finally, if AMDAR observations are configured, compares the computed diagnostics
against them (:class:`DiagnosticsAmdarLauncher`). :class:`Result` wraps each phase's output.
"""

import itertools
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple, assert_never

import numpy as np
import xarray as xr

from rojak.core import data
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.datalib.madis.amdar import AcarsAmdarRepository
from rojak.datalib.ukmo.amdar import UkmoAmdarRepository
from rojak.orchestrator.configuration import (
    AmdarDataSource,
    AmdarDiagnosticCmpSource,
    SpatialGroupByStrategy,
    TurbulenceCalibrationPhaseOption,
    TurbulenceEvaluationPhaseOption,
    TurbulenceThresholds,
)
from rojak.orchestrator.lite_controller import (
    HISTOGRAM_DATA_TYPE_ADAPTER,
    THRESHOLDS_TYPE_ADAPTER,
    export_json,
    load_thresholds_from_file,
)
from rojak.plot.turbulence_plotter import (
    GREY_HEX_CODE,
    chain_diagnostic_names,
    create_diagnostic_correlation_plot,
    create_histogram_n_obs,
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
    MatthewsCorrelationOnThresholdedDiagnostics,
)
from rojak.turbulence.diagnostic import (
    CalibrationDiagnosticSuite,
    DiagnosticFactory,
    DiagnosticSuite,
    EvaluationDiagnosticSuite,
)
from rojak.turbulence.verification import (
    AmdarDataHarmoniser,
    DiagnosticsAmdarVerification,
)
from rojak.utilities.types import DistributionParameters, Limits

if TYPE_CHECKING:
    from pathlib import Path

    from rojak.core.data import AmdarDataRepository, CATData
    from rojak.orchestrator.configuration import (
        AggregationMetricOption,
        DataConfig,
        DiagnosticValidationCondition,
        SpatialDomain,
        TurbulenceCalibrationConfig,
        TurbulenceCalibrationPhases,
        TurbulenceConfig,
        TurbulenceDiagnostics,
        TurbulenceEvaluationConfig,
        TurbulenceEvaluationPhases,
    )
    from rojak.orchestrator.configuration import Context as ConfigContext
    from rojak.turbulence.verification import (
        RocVerificationResult,
    )
    from rojak.utilities.types import DiagnosticName

import logging

logger = logging.getLogger(__name__)

type RunName = str
type TimeStr = str


class Result[T]:
    """Wraps the outcome of running a single calibration/evaluation phase"""

    _result: T

    def __init__(self, result: T) -> None:
        """
        Args:
            result: Outcome to wrap
        """
        self._result = result

    @property
    def result(self) -> T:
        """The wrapped outcome"""
        return self._result


class CalibrationStage:
    """
    Runs the turbulence calibration stage

    For each configured :class:`~rojak.orchestrator.configuration.TurbulenceCalibrationPhaseOption`, either
    computes the result from a calibration dataset (thresholds via
    :meth:`~rojak.turbulence.diagnostic.CalibrationDiagnosticSuite.compute_thresholds`, or EDR distribution
    parameters via :meth:`~rojak.turbulence.diagnostic.CalibrationDiagnosticSuite.compute_distribution_parameters`)
    or loads a previously computed result from file, per
    :class:`~rojak.orchestrator.configuration.TurbulenceCalibrationConfig`.
    """

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
        """
        Args:
            phases: Which calibration phases to run, and their configuration
            spatial_domain: Spatial (and optionally vertical) domain to restrict calibration data to
            output_dir: Base directory to export phase results into
            name: Name of this run, used to namespace exported files under ``output_dir``
            start_time: Timestamp identifying this run, used to name exported files
        """
        self._phases = phases
        self._config = phases.calibration_config
        self._spatial_domain = spatial_domain
        self._output_dir = output_dir
        self._name = name
        self._start_time = start_time

    def launch(
        self,
        diagnostics: list["TurbulenceDiagnostics"],
        chunks: Mapping,
    ) -> Mapping[TurbulenceCalibrationPhaseOption, Result]:
        """
        Run every configured calibration phase

        A :class:`~rojak.turbulence.diagnostic.CalibrationDiagnosticSuite` is built once (if
        ``calibration_data_dir`` is configured) and shared across phases, then released once every phase has run.

        Args:
            diagnostics: Turbulence diagnostics to calibrate
            chunks: Dask chunking to load the calibration data with

        Returns:
            Mapping from each configured phase to its :class:`Result`
        """
        suite: CalibrationDiagnosticSuite | None = (
            self.create_diagnostic_suite(diagnostics, chunks) if self._config.calibration_data_dir is not None else None
        )

        result = {phase: self.run_phase(phase, suite) for phase in self._phases.phases}

        del suite  # "teardown" try to get python to release memory related to calibration data
        return result

    def create_diagnostic_suite(
        self,
        diagnostics: list["TurbulenceDiagnostics"],
        chunks: Mapping,
    ) -> "CalibrationDiagnosticSuite":
        """
        Load the calibration data and compute ``diagnostics`` from it

        Args:
            diagnostics: Turbulence diagnostics to compute
            chunks: Dask chunking to load the calibration data with

        Returns:
            :class:`~rojak.turbulence.diagnostic.CalibrationDiagnosticSuite` of the computed diagnostics

        Raises:
            AssertionError: If ``self._config.calibration_data_dir`` is ``None``
        """
        assert self._config.calibration_data_dir is not None
        logger.debug("Loading CATData")
        calibration_data: CATData = Era5Data(
            data.load_from_folder(self._config.calibration_data_dir, chunks=chunks),
        ).to_clear_air_turbulence_data(self._spatial_domain)
        logger.debug("Instantiating CalibrationDiagnosticSuite")
        return CalibrationDiagnosticSuite(DiagnosticFactory(calibration_data), diagnostics)

    def run_phase(
        self,
        current_phase: TurbulenceCalibrationPhaseOption,
        suite: CalibrationDiagnosticSuite | None,
    ) -> Result:
        """
        Run a single calibration phase, either loading its result from file or computing it from ``suite``

        Args:
            current_phase: Calibration phase to run
            suite: Diagnostic suite to compute the result from, if a corresponding file path is not configured
                (see :attr:`~rojak.orchestrator.configuration.TurbulenceCalibrationConfig.thresholds_file_path`/
                :attr:`~rojak.orchestrator.configuration.TurbulenceCalibrationConfig.diagnostic_distribution_file_path`)

        Returns:
            :class:`Result` of running ``current_phase``
        """
        match current_phase:
            case TurbulenceCalibrationPhaseOption.THRESHOLDS:
                if self._config.thresholds_file_path is not None:
                    return self.load_thresholds_file()
                return self.perform_calibration(suite)
            case TurbulenceCalibrationPhaseOption.HISTOGRAM:
                if self._config.diagnostic_distribution_file_path is not None:
                    return self.load_distribution_parameters_from_file()
                return self.compute_distribution_parameters(suite)
            case _ as unreachable:
                assert_never(unreachable)

    def load_thresholds_file(self) -> Result[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        """
        Load previously computed severity thresholds from :attr:`~TurbulenceCalibrationConfig.thresholds_file_path`

        Returns:
            :class:`Result` of the loaded thresholds, see
            :func:`~rojak.orchestrator.lite_controller.load_thresholds_from_file`

        Raises:
            AssertionError: If ``self._config.thresholds_file_path`` is ``None``
        """
        assert self._config.thresholds_file_path is not None
        thresholds = load_thresholds_from_file(self._config.thresholds_file_path)
        return Result(thresholds)

    def perform_calibration(
        self,
        suite: CalibrationDiagnosticSuite | None,
    ) -> Result[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        """
        Compute severity thresholds for every diagnostic in ``suite`` and export them

        Args:
            suite: Diagnostic suite to compute thresholds from

        Returns:
            :class:`Result` of the computed thresholds

        Raises:
            AssertionError: If ``suite`` or ``self._config.percentile_thresholds`` is ``None``
        """
        assert suite is not None
        assert self._config.percentile_thresholds is not None
        thresholds = suite.compute_thresholds(self._config.percentile_thresholds)
        self.export_thresholds(thresholds)
        return Result(thresholds)

    def export_thresholds(self, diagnostic_thresholds: Mapping["DiagnosticName", "TurbulenceThresholds"]) -> None:
        """
        Export computed thresholds to a JSON file under ``self._output_dir / self._name``

        Args:
            diagnostic_thresholds: Mapping from diagnostic name to its computed thresholds
        """
        export_json(
            dict(diagnostic_thresholds),
            (self._output_dir / self._name),
            self._start_time,
            THRESHOLDS_TYPE_ADAPTER,
            "thresholds",
        )

    def load_distribution_parameters_from_file(self) -> Result:
        """
        Load previously computed distribution histograms from
        :attr:`~TurbulenceCalibrationConfig.diagnostic_distribution_file_path`

        Returns:
            :class:`Result` of the loaded mapping from diagnostic name to its
            :class:`~rojak.turbulence.analysis.HistogramData`

        Raises:
            AssertionError: If ``self._config.diagnostic_distribution_file_path`` is ``None``
        """
        assert self._config.diagnostic_distribution_file_path is not None
        json_str: str = self._config.diagnostic_distribution_file_path.read_text()
        distribution_parameters = HISTOGRAM_DATA_TYPE_ADAPTER.validate_json(json_str)
        return Result(distribution_parameters)

    def export_distribution_parameters(self, diagnostic_thresholds: Mapping["DiagnosticName", "HistogramData"]) -> None:
        """
        Export computed distribution histograms to a JSON file under ``self._output_dir / self._name``

        Args:
            diagnostic_thresholds: Mapping from diagnostic name to its computed
                :class:`~rojak.turbulence.analysis.HistogramData`
        """
        export_json(
            dict(diagnostic_thresholds),
            self._output_dir / self._name,
            self._start_time,
            HISTOGRAM_DATA_TYPE_ADAPTER,
            "distribution_params",
        )

    def compute_distribution_parameters(self, suite: CalibrationDiagnosticSuite | None) -> Result:
        """
        Compute the log-normal distribution histogram for every diagnostic in ``suite`` and export it

        Args:
            suite: Diagnostic suite to compute distribution histograms from

        Returns:
            :class:`Result` of the computed mapping from diagnostic name to its
            :class:`~rojak.turbulence.analysis.HistogramData`

        Raises:
            AssertionError: If ``suite`` is ``None``
        """
        assert suite is not None
        distribution_parameters = suite.compute_distribution_parameters()
        self.export_distribution_parameters(distribution_parameters)
        return Result(distribution_parameters)


class EvaluationStageResult(NamedTuple):
    """Result of running the turbulence evaluation stage: the diagnostic suite used, and each phase's outcome"""

    suite: EvaluationDiagnosticSuite
    phase_outcomes: Mapping[TurbulenceEvaluationPhaseOption, Result]


class EvaluationStage:
    """
    Runs the turbulence evaluation stage

    For each configured :class:`~rojak.orchestrator.configuration.TurbulenceEvaluationPhaseOption`, computes the
    corresponding result (probabilities, EDR, turbulent regions, or diagnostic correlation) from an evaluation
    dataset, using the thresholds/distribution parameters produced by the calibration stage where needed, and
    produces the corresponding plot(s). See :meth:`run_phase`.
    """

    _calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result]
    _phases: list[TurbulenceEvaluationPhaseOption]
    _config: "TurbulenceEvaluationConfig"
    _spatial_domain: "SpatialDomain"
    _output_dir: "Path"
    _plots_dir: "Path"
    _start_time: TimeStr
    _image_format: str

    def __init__(
        self,
        calibration_result: Mapping[TurbulenceCalibrationPhaseOption, Result],
        phases_config: "TurbulenceEvaluationPhases",
        domain: "SpatialDomain",
        output_dir: "Path",
        plots_dir: "Path",
        name: RunName,
        start_time: TimeStr,
        image_format: str,
    ) -> None:
        """
        Args:
            calibration_result: Outcome of :meth:`CalibrationStage.launch`, providing thresholds/distribution
                parameters to evaluate with
            phases_config: Which evaluation phases to run, and their configuration
            domain: Spatial (and optionally vertical) domain to restrict evaluation data to
            output_dir: Base directory to export zarr results into (a ``name`` subdirectory is created)
            plots_dir: Base directory to save plots into (a ``name`` subdirectory is created)
            name: Name of this run, used to namespace exported files
            start_time: Timestamp identifying this run
            image_format: File format (e.g. ``"png"``) to save plots as
        """
        self._calibration_result = calibration_result
        self._phases = phases_config.phases
        self._config = phases_config.evaluation_config
        self._spatial_domain = domain
        self._start_time = start_time
        self._plots_dir = plots_dir / name
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir / name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._image_format = image_format

    def launch(self, diagnostics: list["TurbulenceDiagnostics"], chunks: dict) -> EvaluationStageResult:
        """
        Run every configured evaluation phase

        Args:
            diagnostics: Turbulence diagnostics to evaluate
            chunks: Dask chunking to load the evaluation data with

        Returns:
            The diagnostic suite used, and each configured phase's :class:`Result`
        """
        suite: EvaluationDiagnosticSuite = self.create_diagnostic_suite(diagnostics, chunks)
        return EvaluationStageResult(suite, {phase: self.run_phase(phase, suite) for phase in self._phases})

    def create_diagnostic_suite(
        self,
        diagnostics: list["TurbulenceDiagnostics"],
        chunks: Mapping,
    ) -> EvaluationDiagnosticSuite:
        """
        Load the evaluation data, compute ``diagnostics`` from it, and attach calibration results

        Args:
            diagnostics: Turbulence diagnostics to compute
            chunks: Dask chunking to load the evaluation data with

        Returns:
            :class:`~rojak.turbulence.diagnostic.EvaluationDiagnosticSuite` of the computed diagnostics, with
            probability thresholds and/or EDR distribution parameters attached from
            ``self._calibration_result`` where those phases were run

        Raises:
            AssertionError: If ``self._config.evaluation_data_dir`` is ``None``
        """
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

    def run_phase(self, phase: TurbulenceEvaluationPhaseOption, suite: EvaluationDiagnosticSuite) -> Result:  # noqa: PLR0912
        """
        Run a single evaluation phase against ``suite``, producing the corresponding plot(s) as a side effect

        Args:
            phase: Evaluation phase to run: probability of each severity, EDR, boolean turbulent regions, global
                or latitudinally-stratified correlation between probabilities/EDR, or Matthews correlation between
                thresholded diagnostics (which is additionally exported to zarr, both globally and per pressure
                level)
            suite: Diagnostic suite to compute the result from

        Returns:
            :class:`Result` of running ``phase``

        Raises:
            ValueError: If ``phase`` is :attr:`~TurbulenceEvaluationPhaseOption.MATTHEWS_CORRELATION` and ``suite``
                has no thresholds attached
        """
        match phase:
            case TurbulenceEvaluationPhaseOption.PROBABILITIES:
                result = suite.probabilities
                for pressure_level, severity in itertools.product(
                    self._config.pressure_levels,
                    self._config.severities,
                ):
                    chained_names: str = chain_diagnostic_names(result.keys())
                    create_multi_turbulence_diagnotics_probability_plot(
                        xr.Dataset(
                            data_vars={
                                name: diagnostic.sel(pressure_level=pressure_level, severity=severity)
                                for name, diagnostic in result.items()
                            },
                        ),
                        suite.diagnostic_names(),
                        str(
                            self._plots_dir / f"multi_diagnostic_{chained_names}_on_{pressure_level:.0f}_{severity}"
                            f".{self._image_format}",
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
                correlation = CorrelationBetweenDiagnostics(dict(correlation_on), sel_condition=condition).execute()
                chained_names: str = chain_diagnostic_names(correlation_on.keys())
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
                chained_names: str = chain_diagnostic_names(correlation_on.keys())
                create_multi_region_correlation_plot(
                    correlation,
                    str(self._plots_dir / f"regional_{corr_on_what}_corr_btw_{chained_names}.{self._image_format}"),
                    "diagnostic1",
                    "diagnostic2",
                )
                return Result(correlation)
            case TurbulenceEvaluationPhaseOption.MATTHEWS_CORRELATION:
                thresholds = suite.thresholds()
                if thresholds is None:
                    raise ValueError("Thresholds must be present to compute Matthews correlation")
                matthews_correlation: xr.DataArray = MatthewsCorrelationOnThresholdedDiagnostics(
                    suite.as_dataset(), self._config.severities, thresholds, self._config.threshold_mode
                ).execute()
                chained_names: str = chain_diagnostic_names(suite.diagnostic_names())
                # False positive by pyright - StoreLike inlcudes Path
                # See https://zarr.readthedocs.io/en/v3.1.5/api/zarr/storage/#zarr.storage.StoreLike
                _ = matthews_correlation.to_zarr(
                    self._output_dir / f"matthews_corr_{chained_names}.zarr",  # pyright: ignore[reportArgumentType]
                    mode="w",
                    zarr_format=2,
                )
                for level in self._config.pressure_levels:
                    matthews_correlation_on_level: xr.DataArray = MatthewsCorrelationOnThresholdedDiagnostics(
                        suite.as_dataset(),
                        self._config.severities,
                        thresholds,
                        self._config.threshold_mode,
                        pressure_level=level,
                    ).execute()
                    # From the zarr docs, Path is part of the StoreLike type alias. However, pyright is not
                    # picking this up
                    #   see: https://zarr.readthedocs.io/en/v3.1.5/api/zarr/storage/
                    _ = matthews_correlation_on_level.to_zarr(
                        self._output_dir / f"matthews_corr_{level:.0f}_{chained_names}.zarr",  # pyright: ignore[reportArgumentType]
                        mode="w",
                        zarr_format=2,
                    )

                return Result(matthews_correlation)
            case _ as unreachable:
                assert_never(unreachable)


class TurbulenceLauncher:
    """
    Top-level orchestrator for a full turbulence analysis run, used by :func:`rojak.cli.main.run`

    Runs the calibration stage, then (if configured) the evaluation stage, then (if AMDAR observations are
    configured) compares the computed diagnostics against them.
    """

    _config: "TurbulenceConfig"
    _context: "ConfigContext"

    def __init__(self, context: "ConfigContext") -> None:
        """
        Args:
            context: Root run configuration. Must have ``turbulence_config`` set.

        Raises:
            AssertionError: If ``context.turbulence_config`` is ``None``
        """
        self._context = context
        assert context.turbulence_config is not None
        self._config = context.turbulence_config
        self._start_time = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H_%M_%S")

    def launch(self) -> EvaluationStageResult | None:
        """
        Run calibration, then evaluation (if configured), then AMDAR comparison (if configured)

        Returns:
            The evaluation stage's result, or ``None`` if no evaluation phases were configured

        Raises:
            NotImplementedError: If AMDAR data is configured to be compared against the calibration stage (not
                yet supported)
            AssertionError: If the configuration is in a state that should have been prevented by
                :class:`~rojak.orchestrator.configuration.Context`'s validators
        """
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
                self._context.output_dir,
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
            ).launch(result.suite)

        return result


# PUT THIS IN THIS FILE FOR NOW
class DiagnosticsAmdarLauncher:
    """
    Compares computed turbulence diagnostics against observed AMDAR turbulence

    Spatiotemporally harmonises the AMDAR observations with the diagnostic suite's data (see
    :class:`~rojak.turbulence.verification.AmdarDataHarmoniser`), and, if validation conditions are configured,
    validates the diagnostics against the harmonised observations (see
    :class:`~rojak.turbulence.verification.DiagnosticsAmdarVerification`), producing ROC curve plots and, if a
    spatial group-by strategy is configured, AUC and observation-count plots aggregated by spatial group.
    """

    _path_to_files: str
    _data_source: AmdarDataSource
    _spatial_domain: "SpatialDomain"
    _time_window: "Limits[datetime]"
    _output_filepath: "Path"
    _plots_dir: "Path"
    _save_output: bool
    _validation_conditions: list["DiagnosticValidationCondition"]
    _min_group_size: int
    _group_by_strategy: SpatialGroupByStrategy | None
    _aggregation_metric: "AggregationMetricOption | None"

    def __init__(
        self,
        data_config: "DataConfig",
        output_dir: "Path",
        plots_dir: "Path",
        run_name: "RunName",
    ) -> None:
        """
        Args:
            data_config: Configuration for the input data. Must have ``amdar_config`` set.
            output_dir: Base directory to save harmonised data into (if configured to do so)
            plots_dir: Base directory to save plots into
            run_name: Name of this run, used to namespace exported files

        Raises:
            AssertionError: If ``data_config.amdar_config`` is ``None``
        """
        assert data_config.amdar_config is not None
        self._data_source = data_config.amdar_config.data_source
        self._path_to_files = str(data_config.amdar_config.data_dir.resolve() / data_config.amdar_config.glob_pattern)
        self._spatial_domain = data_config.spatial_domain
        self._time_window = data_config.amdar_config.time_window
        if data_config.amdar_config.diagnostic_validation is None:
            self._validation_conditions = []
            self._min_group_size = -1
            self._group_by_strategy = None
            self._aggregation_metric = None
        else:
            self._validation_conditions = data_config.amdar_config.diagnostic_validation.validation_conditions
            self._min_group_size = data_config.amdar_config.diagnostic_validation.min_group_size
            self._group_by_strategy = data_config.amdar_config.diagnostic_validation.spatial_group_by_strategy
            self._aggregation_metric = data_config.amdar_config.diagnostic_validation.aggregation_metric

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
        """The :class:`~rojak.core.data.AmdarDataRepository` matching :attr:`_data_source`"""
        match self._data_source:
            case AmdarDataSource.MADIS:
                return AcarsAmdarRepository(self._path_to_files)
            case AmdarDataSource.UKMO:
                return UkmoAmdarRepository(self._path_to_files)
            case _ as unreachable:
                assert_never(unreachable)

    def launch(self, diagnostic_suite: DiagnosticSuite) -> None:
        """
        Harmonise AMDAR observations with ``diagnostic_suite``, then validate diagnostics against them if configured

        Args:
            diagnostic_suite: Computed turbulence diagnostics to compare AMDAR observations against

        Raises:
            ValueError: If the spatial domain has no grid size configured (needed to spatially bucket the AMDAR
                observations)
        """
        if self._spatial_domain.grid_size is None:
            raise ValueError("Grid size for spatial domain must be specified for diagnostics amdar data harmonisation")

        logger.info("Started Turbulence Amdar Harmonisation")
        amdar_data = self.create_amdar_data_repository().to_amdar_turbulence_data(
            self._spatial_domain,
            self._spatial_domain.grid_size,
            diagnostic_suite.get_prototype_computed_diagnostic()["pressure_level"].to_numpy().tolist(),
        )
        time_window_as_np_datetime: Limits[np.datetime64] = Limits(
            np.datetime64(self._time_window.lower),
            np.datetime64(self._time_window.upper),
        )
        harmoniser: AmdarDataHarmoniser = AmdarDataHarmoniser(
            amdar_data, diagnostic_suite.get_prototype_computed_diagnostic(), time_window_as_np_datetime
        )

        if self._validation_conditions:
            logger.info("Starting validation of diagnostics with amdar data")
            verifier = DiagnosticsAmdarVerification(harmoniser, diagnostic_suite.as_dataset())
            chained_names: str = chain_diagnostic_names(diagnostic_suite.diagnostic_names())

            if self._group_by_strategy is not None:
                assert self._aggregation_metric is not None
                grid_auc = verifier.aggregate_by_auc(
                    self._validation_conditions,
                    self._group_by_strategy,
                    self._min_group_size,
                    self._aggregation_metric,
                )
                logger.debug("Finished aggregating on groups")
                num_observations = verifier.num_obs_per(self._validation_conditions, self._group_by_strategy)
                logger.debug("Finished computing number of observations per group")
                is_agg_by_point: bool = self._group_by_strategy in {
                    SpatialGroupByStrategy.GRID_POINT,
                    SpatialGroupByStrategy.HORIZONTAL_POINT,
                }
                if not is_agg_by_point:
                    for diagnostic_name in grid_auc:
                        grid_auc[diagnostic_name] = amdar_data.grid.join(grid_auc[diagnostic_name], how="right").drop(
                            columns=[verifier.grid_box_column],
                        )
                    num_observations = amdar_data.grid.join(num_observations, how="right")
                auc_plots = create_interactive_aggregated_auc_plots(
                    grid_auc,
                    self._validation_conditions,
                    is_agg_by_point,
                )
                save_hv_plot(
                    auc_plots,
                    str(
                        self._plots_dir
                        / f"regional_{self._aggregation_metric}_{chained_names}_on_{self._group_by_strategy!s}",
                    ),
                    "png",
                    savefig_kwargs={"dpi": 400},
                )

                save_hv_plot(
                    create_interactive_heatmap_plot(
                        num_observations if is_agg_by_point else num_observations.compute(),
                        "num_obs",
                        opts_kwargs={
                            "fig_size": 400,
                            "title": f"Total Number of Observations (min = {self._min_group_size})",
                            "lw": 0,
                            "clim": (self._min_group_size, None),
                            "clipping_colors": {"min": GREY_HEX_CODE},
                        },
                    ),
                    str(self._plots_dir / f"num_observations_for_{self._group_by_strategy!s}"),
                    "png",
                    savefig_kwargs={"dpi": 400},
                )

                save_hv_plot(
                    create_histogram_n_obs(num_observations, hist_kwargs={"normed": True, "ylabel": "Density"}),
                    str(self._plots_dir / f"num_obs_histogram_for_{self._group_by_strategy!s}"),
                    "png",
                    savefig_kwargs={"dpi": 400},
                )

                bottom_counts: float = num_observations["num_obs"].quantile(q=0.25, method="tdigest").compute()
                save_hv_plot(
                    create_histogram_n_obs(num_observations.loc[num_observations["num_obs"] <= bottom_counts]),
                    str(self._plots_dir / f"num_obs_histogram_for_{self._group_by_strategy!s}"),
                    "png",
                    savefig_kwargs={"dpi": 400},
                )

            roc: RocVerificationResult = verifier.nearest_value_roc(self._validation_conditions)
            roc_curve_plots: dict = create_interactive_roc_curve_plot(roc)
            for amdar_col, plot_for_col in roc_curve_plots.items():
                save_hv_plot(plot_for_col, str(self._plots_dir / f"roc_{amdar_col}_on_{chained_names}"), "png")
                logger.debug("Saved roc plot for %s AMDAR turbulence measure", amdar_col)

            logger.info("Finished validation of diagnostics with amdar data")
