from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self, assert_never

import numpy as np
import yaml
from pydantic import AfterValidator, BaseModel, Field, ValidationError, model_validator

from rojak.utilities.types import Limits


class InvalidConfigurationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def dir_must_exist(path: Path) -> Path:
    if not path.exists():
        raise InvalidConfigurationError(f"{path} does not exist")
    if not path.is_dir():
        raise InvalidConfigurationError(f"{path} is not a directory")
    return path


def make_dir_if_not_present(path: Path) -> Path:
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


# class TurbulenceSeverityPercentileConfig(BaseConfigModel):
#     name: Annotated[str, Field(min_length=1, description="Name of intensity", repr=True, frozen=True, strict=True)]
#     lower_bound: Annotated[
#         float, Field(ge=0, lt=100.0, description="Lower bound of percentile", repr=True, frozen=True)
#     ]
#     upper_bound: Annotated[float, Field(ge=0, description="Upper bound of percentile", repr=True, frozen=True)]
#
#     @model_validator(mode="after")
#     def check_reasonable_bounds(self) -> Self:
#         if self.lower_bound > self.upper_bound:
#             raise InvalidConfigurationError(
#                 f"Lower bound ({self.lower_bound}) must be greater than upper bound ({self.upper_bound})"
#             )
#         if np.inf > self.upper_bound > 100:
#             raise InvalidConfigurationError(
#                 f"Upper bound ({self.upper_bound}) must be infinite or less than or equal to 100"
#             )
#         return self


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

        next_severity: TurbulenceSeverity = next(
            higher_severity
            for higher_severity in TurbulenceSeverity.get_in_ascending_order()[severity.get_index() + 1 :]
            if self.get_by_severity(higher_severity) is not None
        )
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
        AfterValidator(dir_must_exist),
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


class SpatialDomain(BaseConfigModel):
    minimum_latitude: Annotated[float, Field(default=-90, description="Minimum latitude", ge=-90, le=90)]
    maximum_latitude: Annotated[float, Field(default=90, description="Maximum latitude", ge=-90, le=90)]
    minimum_longitude: Annotated[float, Field(default=-180, description="Minimum longitude", ge=-180, le=180)]
    maximum_longitude: Annotated[float, Field(default=180, description="Minimum longitude", ge=-180, le=180)]

    @model_validator(mode="after")
    def check_valid_ranges(self) -> Self:
        if self.minimum_latitude > self.maximum_latitude:
            raise ValueError("Maximum latitude must be greater than minimum latitude")
        if self.minimum_longitude > self.maximum_longitude:
            raise ValueError("Maximum longitude must be greater than minimum longitude")
        if self.minimum_latitude == self.maximum_latitude:
            raise ValueError("Minimum latitude must NOT be equal to maximum latitude")
        if self.minimum_longitude == self.maximum_longitude:
            raise ValueError("Minimum longitude must NOT be equal to maximum longitude")
        return self


class MetDataSource(StrEnum):
    ERA5 = "era5"


class MeteorologyConfig(BaseConfigModel):
    data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing the data from a NWP/GCM",
            repr=True,
            frozen=True,
        ),
        AfterValidator(dir_must_exist),
    ]
    data_source: Annotated[
        MetDataSource, Field(default=MetDataSource.ERA5, description="Source of Met data", repr=True, frozen=True)
    ]


class DataConfig(BaseConfigModel):
    # Config for data, this would cover both observational data and weather data
    spatial_domain: SpatialDomain
    meteorology_config: MeteorologyConfig | None = None
    ...


class Context(BaseConfigModel):
    # Root configuration
    name: Annotated[str, Field(description="Name of run", repr=True, strict=True, frozen=True)]
    image_format: Annotated[str, Field(description="Format of output plots", strict=True, frozen=True)]
    output_dir: Annotated[
        Path,
        Field(description="Output directory", repr=True, frozen=True),
        AfterValidator(make_dir_if_not_present),
    ]
    plots_dir: Annotated[
        Path,
        Field(description="Plots directory", repr=True, frozen=True),
        AfterValidator(make_dir_if_not_present),
    ]
    turbulence_config: TurbulenceConfig | None = None
    contrails_config: ContrailsConfig | None = None
    data_config: DataConfig
