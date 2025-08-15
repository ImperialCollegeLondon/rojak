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

import datetime
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self, assert_never

import numpy as np
import yaml
from pydantic import AfterValidator, BaseModel, Field, ValidationError, model_validator
from pydantic.types import PositiveInt

from rojak.datalib.madis.amdar import AcarsAmdarTurbulenceData
from rojak.datalib.ukmo.amdar import UkmoAmdarTurbulenceData
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from rojak.core.data import AmdarTurbulenceData


class InvalidConfigurationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def _dir_must_exist(path: Path) -> Path:
    if not path.exists():
        raise InvalidConfigurationError(f"{path} does not exist")
    if not path.is_dir():
        raise InvalidConfigurationError(f"{path} is not a directory")
    return path


def _make_dir_if_not_present(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class TurbulenceSeverity(StrEnum):
    LIGHT = "light"
    LIGHT_TO_MODERATE = "light_to_moderate"
    MODERATE = "moderate"
    MODERATE_TO_SEVERE = "moderate_to_severe"
    SEVERE = "severe"
    # MODERATE_OR_GREATER = "moderate_or_greater"

    @classmethod
    def get_in_ascending_order(cls) -> list["TurbulenceSeverity"]:
        return [
            TurbulenceSeverity.LIGHT,
            TurbulenceSeverity.LIGHT_TO_MODERATE,
            TurbulenceSeverity.MODERATE,
            TurbulenceSeverity.MODERATE_TO_SEVERE,
            TurbulenceSeverity.SEVERE,
        ]

    def get_index(self) -> int:
        match self:
            case TurbulenceSeverity.LIGHT:
                return 0
            case TurbulenceSeverity.LIGHT_TO_MODERATE:
                return 1
            case TurbulenceSeverity.MODERATE:
                return 2
            case TurbulenceSeverity.MODERATE_TO_SEVERE:
                return 3
            case TurbulenceSeverity.SEVERE:
                return 4
            case _ as unreachable:
                assert_never(unreachable)


class TurbulenceThresholdMode(StrEnum):
    BOUNDED = "bounded"
    GEQ = "geq"


class TurbulenceDiagnostics(StrEnum):
    RICHARDSON = "richardson"
    F2D = "f2d"
    F3D = "f3d"
    UBF = "ubf"
    TI1 = "ti1"
    TI2 = "ti2"
    NCSU1 = "ncsu1"
    ENDLICH = "endlich"
    COLSON_PANOFSKY = "colson_panofsky"
    WIND_SPEED = "wind_speed"
    WIND_DIRECTION = "wind_direction"
    BRUNT_VAISALA = "bunt_vaisala"
    VWS = "vertical_wind_shear"
    DEF = "deformation"
    DIRECTIONAL_SHEAR = "directional_shear"
    TEMPERATURE_GRADIENT = "temperature_gradient"
    HORIZONTAL_DIVERGENCE = "horizontal_divergence"
    NGM1 = "ngm1"
    NGM2 = "ngm2"
    BROWN1 = "brown1"
    BROWN2 = "brown2"
    MAGNITUDE_PV = "magnitude_pv"
    PV_GRADIENT = "gradient_pv"
    NVA = "nva"
    DUTTON = "dutton"
    EDR_LUNNON = "edr_lunnon"
    VORTICITY_SQUARED = "vorticity_squared"


class BaseConfigModel(BaseModel):
    @classmethod
    def from_yaml(cls, path: "Path") -> Self:
        if path.is_file():
            data: dict = {}
            with path.open(mode="r") as f:
                data = yaml.safe_load(f)

            try:
                instance = cls.model_validate(data)
            except ValidationError as e:
                raise InvalidConfigurationError(str(e)) from e
            return instance
        raise InvalidConfigurationError("Configuration file not found or is not a file.")


class TurbulenceThresholds(BaseConfigModel):
    light: Annotated[
        float | None,
        Field(description="Light turbulence intensity (percentile or diagnostic value)", repr=True, frozen=True),
    ] = None
    light_to_moderate: Annotated[
        float | None,
        Field(
            description="Light to moderate turbulence intensity (percentile or diagnostic value)",
            repr=True,
            frozen=True,
        ),
    ] = None
    moderate: Annotated[
        float | None,
        Field(description="Moderate turbulence intensity (percentile or diagnostic value)", repr=True, frozen=True),
    ] = None
    moderate_to_severe: Annotated[
        float | None,
        Field(
            description="Moderate to severe turbulence intensity (percentile or diagnostic value)",
            repr=True,
            frozen=True,
        ),
    ] = None
    severe: Annotated[
        float | None,
        Field(description="Severe turbulence intensity (percentile or diagnostic value)", repr=True, frozen=True),
    ] = None
    _all_severities: list[float | None] = []

    @model_validator(mode="after")
    def check_values_increasing(self) -> Self:
        if all(severity is None for severity in self._all_severities):
            raise InvalidConfigurationError("Threshold values cannot all be None")
        without_nones = [item for item in self._all_severities if item is not None]
        sorted_severities = sorted(without_nones)
        if without_nones != sorted_severities:
            raise InvalidConfigurationError("Values must be in ascending order")
        return self

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        self._all_severities = [self.light, self.light_to_moderate, self.moderate, self.moderate_to_severe, self.severe]

    @property
    def all_severities(self) -> list[float | None]:
        return self._all_severities

    def get_by_severity(self, severity: TurbulenceSeverity) -> float | None:
        match severity:
            case TurbulenceSeverity.LIGHT:
                return self.light
            case TurbulenceSeverity.LIGHT_TO_MODERATE:
                return self.light_to_moderate
            case TurbulenceSeverity.MODERATE:
                return self.moderate
            case TurbulenceSeverity.MODERATE_TO_SEVERE:
                return self.moderate_to_severe
            case TurbulenceSeverity.SEVERE:
                return self.severe
            case _ as unreachable:
                assert_never(unreachable)

    def get_bounds(self, severity: TurbulenceSeverity, mode: TurbulenceThresholdMode) -> Limits:
        lower_bound: float | None = self.get_by_severity(severity)
        if lower_bound is None:
            raise ValueError("Attempting to retrieve threshold value for a severity that is None")

        if mode == TurbulenceThresholdMode.GEQ or severity == TurbulenceSeverity.SEVERE:
            return Limits(lower_bound, np.inf)

        try:
            next_severity: TurbulenceSeverity = next(
                higher_severity
                for higher_severity in TurbulenceSeverity.get_in_ascending_order()[severity.get_index() + 1 :]
                if self.get_by_severity(higher_severity) is not None
            )
        except StopIteration as exception:
            raise StopIteration(f"Failed to get upper bound for severity {severity}") from exception
        upper_bound = self.get_by_severity(next_severity)
        assert upper_bound is not None
        return Limits(lower_bound, upper_bound)


class TurbulenceCalibrationPhaseOption(StrEnum):
    THRESHOLDS = "thresholds"
    HISTOGRAM = "histogram"


class TurbulenceCalibrationConfig(BaseConfigModel):
    calibration_data_dir: Annotated[
        Path | None,
        Field(
            description="Path to directory containing calibration data",
            repr=True,
            frozen=True,
        ),
    ] = None
    percentile_thresholds: Annotated[
        TurbulenceThresholds | None, Field(description="Percentile thresholds", frozen=True, repr=True)
    ] = None
    thresholds_file_path: Annotated[
        Path | None,
        Field(
            description="Path to directory containing thresholds data",
            repr=True,
            frozen=True,
        ),
    ] = None
    diagnostic_distribution_file_path: Annotated[
        Path | None, Field(description="Path to directory containing distribution of diagnostic indices")
    ] = None


class TurbulenceCalibrationPhases(BaseConfigModel):
    phases: Annotated[
        list[TurbulenceCalibrationPhaseOption],
        Field(description="Turbulence calibration phases", repr=True, frozen=True),
    ]
    calibration_config: TurbulenceCalibrationConfig

    @model_validator(mode="after")
    def check_necessary_config_for_phases(self) -> Self:
        for phase in self.phases:
            match phase:
                case TurbulenceCalibrationPhaseOption.THRESHOLDS:
                    # Check that either calibration or thresholds file specified
                    if (
                        self.calibration_config.calibration_data_dir is None
                        and self.calibration_config.thresholds_file_path is None
                    ):
                        raise InvalidConfigurationError(
                            "Either calibration data directory or thresholds file must be provided."
                        )
                    if (
                        self.calibration_config.calibration_data_dir is not None
                        and self.calibration_config.thresholds_file_path is not None
                    ):
                        raise InvalidConfigurationError(
                            "Either calibration data directory or thresholds file must be provided, NOT both"
                        )

                    # Check that the paths are to the correct type of file system object
                    if (
                        self.calibration_config.calibration_data_dir is not None
                        and not self.calibration_config.calibration_data_dir.is_dir()
                    ):
                        raise InvalidConfigurationError("Calibration data directory is not a directory.")
                    if (
                        self.calibration_config.thresholds_file_path is not None
                        and not self.calibration_config.thresholds_file_path.is_file()
                    ):
                        raise InvalidConfigurationError("Thresholds file provided is not a file.")

                    # If performing calibration, percentiles must be specified
                    if (
                        self.calibration_config.calibration_data_dir is not None
                        and self.calibration_config.percentile_thresholds is None
                    ):
                        raise InvalidConfigurationError(
                            "If calibration data directory is provided, threshold configuration must be provided."
                        )
                case TurbulenceCalibrationPhaseOption.HISTOGRAM:
                    if (
                        self.calibration_config.diagnostic_distribution_file_path is not None
                        and not self.calibration_config.diagnostic_distribution_file_path.is_file()
                    ):
                        raise InvalidConfigurationError("Diagnostic index distribution file provided is not a file.")

        return self

    @model_validator(mode="after")
    def check_no_repeated_phases(self) -> Self:
        if self.phases and len(set(self.phases)) != len(self.phases):
            raise InvalidConfigurationError("Duplicate phases detected.")
        return self


class TurbulenceEvaluationPhaseOption(StrEnum):
    PROBABILITIES = "probabilities"
    EDR = "edr"
    TURBULENT_REGIONS = "turbulent_regions"
    CORRELATION_BTW_PROBABILITIES = "correlation_between_probabilities"
    CORRELATION_BTW_EDR = "correlation_between_edr"
    REGIONAL_CORRELATION_PROBABILITIES = "regional_correlation_between_probabilities"
    REGIONAL_CORRELATION_EDR = "regional_correlation_between_edr"


class TurbulenceEvaluationConfig(BaseConfigModel):
    evaluation_data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing evaluation data",
            repr=True,
            frozen=True,
        ),
        AfterValidator(_dir_must_exist),
    ]  # Remove this?????
    threshold_mode: Annotated[
        TurbulenceThresholdMode,
        Field(
            default=TurbulenceThresholdMode.BOUNDED,
            description="How thresholds are used to determine if turbulent",
            repr=True,
            frozen=True,
        ),
    ] = TurbulenceThresholdMode.BOUNDED
    severities: Annotated[
        list[TurbulenceSeverity],
        Field(description="Target turbulence severity", repr=True),
    ] = [TurbulenceSeverity.LIGHT]
    pressure_levels: Annotated[
        list[float], Field(description="Pressure levels to evaluate on", repr=True, frozen=True)
    ] = [200.0]


class TurbulenceEvaluationPhases(BaseConfigModel):
    phases: Annotated[
        list[TurbulenceEvaluationPhaseOption], Field(description="Turbulence evaluation phases", repr=True, frozen=True)
    ]
    evaluation_config: TurbulenceEvaluationConfig

    @model_validator(mode="after")
    def check_dependent_phase_is_present(self) -> Self:
        for index, phase in enumerate(self.phases):
            match phase:
                case (
                    TurbulenceEvaluationPhaseOption.CORRELATION_BTW_PROBABILITIES
                    | TurbulenceEvaluationPhaseOption.REGIONAL_CORRELATION_PROBABILITIES
                ):
                    if TurbulenceEvaluationPhaseOption.PROBABILITIES not in self.phases[:index]:
                        raise InvalidConfigurationError(
                            "To compute correlation between probabilities, probabilities phase must be occur before "
                            "correlations phase"
                        )
                case (
                    TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR
                    | TurbulenceEvaluationPhaseOption.CORRELATION_BTW_EDR
                ):
                    if TurbulenceEvaluationPhaseOption.EDR not in self.phases[:index]:
                        raise InvalidConfigurationError(
                            "To compute correlation between edr probabilities, edr phase must be occur before "
                            "correlations phase"
                        )
        return self


class TurbulencePhases(BaseConfigModel):
    calibration_phases: TurbulenceCalibrationPhases
    evaluation_phases: TurbulenceEvaluationPhases | None = None  # Makes it possible to just run calibration

    @model_validator(mode="after")
    def check_dependent_phase_is_present(self) -> Self:
        if self.evaluation_phases is not None:
            eval_set: set[TurbulenceEvaluationPhaseOption] = set(self.evaluation_phases.phases)
            calibration_set: set[TurbulenceCalibrationPhaseOption] = set(self.calibration_phases.phases)

            if (
                TurbulenceEvaluationPhaseOption.PROBABILITIES in eval_set
                and TurbulenceCalibrationPhaseOption.THRESHOLDS not in calibration_set
            ):
                raise InvalidConfigurationError(
                    "To evaluate probabilities, thresholding phase must occur at calibration stage"
                )
            if (
                TurbulenceEvaluationPhaseOption.EDR in eval_set
                and TurbulenceCalibrationPhaseOption.HISTOGRAM not in calibration_set
            ):
                raise InvalidConfigurationError("To evaluate EDR, histogram phase must occur at calibration stage")

        return self

    @model_validator(mode="after")
    def check_valid_severities(self) -> Self:
        if (
            self.evaluation_phases is not None
            and self.calibration_phases.calibration_config.percentile_thresholds is not None
            and any(
                self.calibration_phases.calibration_config.percentile_thresholds.get_by_severity(severity) is None
                for severity in self.evaluation_phases.evaluation_config.severities
            )
        ):
            raise InvalidConfigurationError(
                "Attempting to evaluate for a severity that has not been computed for int the calibration phase"
            )
        return self


class TurbulenceConfig(BaseConfigModel):
    chunks: Annotated[
        dict,
        Field(
            description="How data should be chunked (dask)",
            frozen=True,
            repr=True,
            strict=True,
        ),
    ]
    diagnostics: Annotated[
        list[TurbulenceDiagnostics],
        Field(
            description="List of turbulence diagnostics to evaluate",
            repr=True,
            frozen=True,
        ),
    ]
    phases: TurbulencePhases


class ContrailModel(StrEnum):
    ISSR = "issr"
    SAC = "sac"
    PCR = "pcr"


class ContrailsConfig(BaseConfigModel):
    # Config for contrail analysis
    contrail_model: Annotated[
        ContrailModel,
        Field(description="Contrail model from pycontrails to use", repr=True, frozen=True),
    ]
    ...


class DiagnosticValidationCondition(BaseConfigModel):
    observed_turbulence_column_name: Annotated[
        str, Field(description="Observed turbulence column name", frozen=True, repr=True, strict=True)
    ]
    value_greater_than: Annotated[
        float, Field(description="Value greater than", repr=True, strict=True, ge=0.0, frozen=True)
    ]

    def __hash__(self) -> int:
        return hash((self.observed_turbulence_column_name, self.value_greater_than))


class DiagnosticValidationConfig(BaseConfigModel):
    validation_conditions: list[DiagnosticValidationCondition]
    min_group_size: Annotated[
        PositiveInt, Field(description="Minimum group size for aggregation", repr=True, strict=True, default=20)
    ] = 20

    @model_validator(mode="after")
    def check_conditions_are_unique(self) -> Self:
        assert len(set(self.validation_conditions)) == len(self.validation_conditions), (
            "Validation conditions must be unique"
        )
        return self


class SpatialDomain(BaseConfigModel):
    minimum_latitude: Annotated[float, Field(default=-90, description="Minimum latitude", ge=-90, le=90)]
    maximum_latitude: Annotated[float, Field(default=90, description="Maximum latitude", ge=-90, le=90)]
    minimum_longitude: Annotated[float, Field(default=-180, description="Minimum longitude", ge=-180, le=180)]
    maximum_longitude: Annotated[float, Field(default=180, description="Minimum longitude", ge=-180, le=180)]
    minimum_level: Annotated[float | None, Field(description="Minimum level", default=None)] = None
    maximum_level: Annotated[float | None, Field(description="Maximum level", default=None)] = None
    grid_size: Annotated[float | None, Field(description="Grid size", default=None)] = None

    @model_validator(mode="after")
    def check_valid_ranges(self) -> Self:
        if self.minimum_latitude > self.maximum_latitude:
            raise ValueError("Maximum latitude must be greater than minimum latitude")
        # TODO: Handle ranges that cross the anti-meridian (i.e. 180 degrees)
        if self.minimum_longitude > self.maximum_longitude:
            raise ValueError("Maximum longitude must be greater than minimum longitude")
        if self.minimum_latitude == self.maximum_latitude:
            raise ValueError("Minimum latitude must NOT be equal to maximum latitude")
        if self.minimum_longitude == self.maximum_longitude:
            raise ValueError("Minimum longitude must NOT be equal to maximum longitude")
        if (
            self.minimum_level is not None
            and self.maximum_level is not None
            and self.maximum_level < self.minimum_level
        ):
            raise ValueError("Minimum level must be less than maximum level")
        return self

    def central_latitude(self, use_int_division: bool = False) -> float | int:
        latitude_range: float = self.minimum_latitude + self.maximum_latitude
        return latitude_range // 2 if use_int_division else latitude_range / 2

    def central_longitude(self, use_int_division: bool = False) -> float | int:
        longitude_range: float = self.minimum_longitude + self.maximum_longitude
        return longitude_range // 2 if use_int_division else longitude_range / 2

    def use_hemisphere_projection(self) -> bool:
        is_start_or_end_at_equator = self.minimum_latitude == 0 or self.maximum_latitude == 0
        return is_start_or_end_at_equator and abs(self.minimum_latitude + self.maximum_latitude) >= 45  # noqa: PLR2004


class MetDataSource(StrEnum):
    ERA5 = "era5"


class AmdarDataSource(StrEnum):
    MADIS = "madis"
    UKMO = "ukmo"


class BaseInputDataConfig[T: StrEnum](BaseConfigModel):
    data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing the data",
            repr=True,
            frozen=True,
        ),
        AfterValidator(_dir_must_exist),
    ]
    data_source: Annotated[T, Field(description="Where data comes from", repr=True, frozen=True)]


class MeteorologyConfig(BaseInputDataConfig[MetDataSource]): ...


class DiagnosticsAmdarHarmonisationStrategyOptions(StrEnum):
    RAW_INDEX_VALUES = "raw"
    INDEX_TURBULENCE_INTENSITY = "index_severity"
    EDR = "edr"
    EDR_TURBULENCE_INTENSITY = "edr_severity"

    def column_name_method(self, severity: "TurbulenceSeverity | None" = None) -> Callable:
        match self:
            case (
                DiagnosticsAmdarHarmonisationStrategyOptions.EDR
                | DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES
            ):
                assert severity is None
                return lambda name: f"{name}_{str(self)}"
            case (
                DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY
                | DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY
            ):
                assert severity is not None
                return lambda name: f"{name}_{str(self)}_{str(severity)}"
            case _ as unreachable:
                assert_never(unreachable)


class AmdarDiagnosticCmpSource(StrEnum):
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class AmdarConfig(BaseInputDataConfig[AmdarDataSource]):
    glob_pattern: Annotated[
        str, Field(description="Glob pattern to match to get the data files", repr=True, frozen=True)
    ]
    time_window: Annotated[
        Limits[datetime.datetime], Field(description="Time window to extract data for", repr=True, frozen=True)
    ]
    # ASSUME FOR NOW ONLY USE FOR THIS IS DATA HARMONISATION
    harmonisation_strategies: Annotated[
        list[DiagnosticsAmdarHarmonisationStrategyOptions] | None,
        Field(description="List of harmonisation strategies", repr=True, default=None),
    ] = None
    diagnostics_from: Annotated[
        AmdarDiagnosticCmpSource,
        Field(
            description="Which data (thus diagnostics) should be amdar data be compared against",
            repr=True,
            frozen=True,
            default=AmdarDiagnosticCmpSource.EVALUATION,
        ),
    ] = AmdarDiagnosticCmpSource.EVALUATION
    save_harmonised_data: Annotated[
        bool, Field(description="Save harmonised data", repr=True, frozen=True, default=True)
    ] = True
    diagnostic_validation: Annotated[
        DiagnosticValidationConfig | None,
        Field(description="Diagnostic validation configuration", repr=True, frozen=True, default=None),
    ] = None

    def model_post_init(self, context: Any, /) -> None:  # noqa: ANN401
        if self.harmonisation_strategies is None:
            self.harmonisation_strategies = []

    @model_validator(mode="after")
    def check_valid_glob_pattern(self) -> Self:
        if "*" not in self.glob_pattern:
            raise InvalidConfigurationError("Asterisk not found in glob pattern")
        match self.data_source:
            case AmdarDataSource.UKMO:
                if not self.glob_pattern.endswith("csv"):
                    raise InvalidConfigurationError("UKMO files must end with .csv")
            case AmdarDataSource.MADIS:
                if not (
                    self.glob_pattern.endswith("parquet")
                    or self.glob_pattern.endswith("pqt")
                    or self.glob_pattern.endswith("parq")
                    or self.glob_pattern.endswith("pq")
                ):
                    raise InvalidConfigurationError("Madis AMDAR files must be in parquet format")
        return self

    @model_validator(mode="after")
    def check_time_window_increasing(self) -> Self:
        if self.time_window.lower > self.time_window.upper:
            raise InvalidConfigurationError("Time must be increasing from lower to upp")
        return self

    @model_validator(mode="after")
    def check_valid_diagnostic_validation_conditions(self) -> Self:
        if self.diagnostic_validation is not None:
            # Update this once there is more than two classes
            data_source_class: AmdarTurbulenceData = (
                UkmoAmdarTurbulenceData if self.data_source == AmdarDataSource.UKMO else AcarsAmdarTurbulenceData
            )  # pyright: ignore [reportAssignmentType]
            print(data_source_class)
            if not {
                condition.observed_turbulence_column_name
                for condition in self.diagnostic_validation.validation_conditions
            }.issubset(data_source_class.turbulence_column_names()):
                raise InvalidConfigurationError(
                    "Diagnostic validation conditions must be one of the turbulence columns"
                )
        return self


class DataConfig(BaseConfigModel):
    # Config for data, this would cover both observational data and weather data
    spatial_domain: SpatialDomain
    meteorology_config: MeteorologyConfig | None = None
    # FOR NOW! Assume that if amdar data is provided, it is for comparing with the turbulence diagnostics
    amdar_config: AmdarConfig | None = None

    @model_validator(mode="after")
    def check_grid_size_specified(self) -> Self:
        if self.amdar_config is not None and self.spatial_domain.grid_size is None:
            raise InvalidConfigurationError("Grid size must be specified if processing AMDAR data")
        return self


class Context(BaseConfigModel):
    # Root configuration
    name: Annotated[str, Field(description="Name of run", repr=True, strict=True, frozen=True)]
    image_format: Annotated[str, Field(description="Format of output plots", strict=True, frozen=True)]
    output_dir: Annotated[
        Path,
        Field(description="Output directory", repr=True, frozen=True),
        AfterValidator(_make_dir_if_not_present),
    ]
    plots_dir: Annotated[
        Path,
        Field(description="Plots directory", repr=True, frozen=True),
        AfterValidator(_make_dir_if_not_present),
    ]
    turbulence_config: TurbulenceConfig | None = None
    contrails_config: ContrailsConfig | None = None
    data_config: DataConfig

    @model_validator(mode="after")
    def check_amdar_data_comparison_diagnostics_exists(self) -> Self:
        if self.data_config.amdar_config is not None:
            if self.turbulence_config is None:
                raise InvalidConfigurationError("Amdar data has been specified but turbulence config is missing")

            match self.data_config.amdar_config.diagnostics_from:
                case AmdarDiagnosticCmpSource.CALIBRATION:
                    if self.turbulence_config.phases.calibration_phases.calibration_config.calibration_data_dir is None:
                        raise InvalidConfigurationError(
                            "To compare amdar data against calibration, calibration data must be specified and "
                            "cannot be restored from an earlier run"
                        )
                case AmdarDiagnosticCmpSource.EVALUATION:
                    if self.turbulence_config.phases.evaluation_phases is None:
                        raise InvalidConfigurationError(
                            "To compare amdar data against evaluation, evaluation phases must be specified"
                        )
                    #  No need to check if evaluation data dir is specified as it is a required argument once
                    #  evaluation_phases is specified
                case _ as unreachable:
                    assert_never(unreachable)

        return self
