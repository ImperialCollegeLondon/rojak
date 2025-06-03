from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import AfterValidator, BaseModel, Field, ValidationError, model_validator


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
    MODERATE_OR_GREATER = "moderate_or_greater"

    @classmethod
    def get_in_ascending_order(cls) -> list["TurbulenceSeverity"]:
        return [
            TurbulenceSeverity.LIGHT,
            TurbulenceSeverity.LIGHT_TO_MODERATE,
            TurbulenceSeverity.MODERATE,
            TurbulenceSeverity.MODERATE_TO_SEVERE,
            TurbulenceSeverity.SEVERE,
        ]


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
    VORTICITY_SQUARED = "vortical_squared"


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


class TurbulenceConfig(BaseConfigModel):
    # Remove this and have it only in data config?????
    evaluation_data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing evaluation data",
            repr=True,
            frozen=True,
        ),
        AfterValidator(dir_must_exist),
    ]  # Remove this?????
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
    threshold_mode: Annotated[
        TurbulenceThresholdMode,
        Field(
            default=TurbulenceThresholdMode.BOUNDED,
            description="How thresholds are used to determine if turbulent",
            repr=True,
            frozen=True,
        ),
    ] = TurbulenceThresholdMode.BOUNDED
    calibration_data_dir: Annotated[
        Path | None,
        Field(
            description="Path to directory containing calibration data",
            repr=True,
            frozen=True,
        ),
    ] = None
    thresholds_file_path: Annotated[
        Path | None,
        Field(
            description="Path to directory containing thresholds data",
            repr=True,
            frozen=True,
        ),
    ] = None
    severities: Annotated[
        list[TurbulenceSeverity],
        Field(description="Target turbulence severity", repr=True),
    ] = [TurbulenceSeverity.LIGHT]
    percentile_thresholds: Annotated[
        TurbulenceThresholds | None, Field(description="Percentile thresholds", frozen=True, repr=True)
    ] = None
    # threshold_config: Annotated[
    #     list[TurbulenceSeverityPercentileConfig] | None,
    #     Field(description="Configuration for threshold computation", frozen=True, repr=True),
    # ] = None

    @model_validator(mode="after")
    def check_either_calibration_data_or_thresholds_present(self) -> Self:
        if self.calibration_data_dir is None and self.thresholds_file_path is None:
            raise InvalidConfigurationError("Either calibration data directory or thresholds file must be provided.")
        if self.calibration_data_dir is not None and self.thresholds_file_path is not None:
            raise InvalidConfigurationError(
                "Either calibration data directory or thresholds file must be provided, NOT both"
            )
        if self.calibration_data_dir is not None and not self.calibration_data_dir.is_dir():
            raise InvalidConfigurationError("Calibration data directory is not a directory.")
        if self.thresholds_file_path is not None and not self.thresholds_file_path.is_file():
            raise InvalidConfigurationError("Thresholds file provided is not a file.")
        return self

    # TODO: Add comprehensive tests
    @model_validator(mode="after")
    def check_calibration_fully_configured(self) -> Self:
        if self.calibration_data_dir is not None and self.percentile_thresholds is None:
            raise InvalidConfigurationError(
                "If calibration data directory is provided, threshold configuration must be provided."
            )
        # if (
        #     self.threshold_config is not None
        #     and self.threshold_mode == TurbulenceThresholdMode.GEQ
        #     and any(config.upper_bound != np.inf for config in self.threshold_config)
        # ):
        #     raise InvalidConfigurationError("If thresholding is GEQ, upper bound must be infinite")
        # if self.threshold_config is not None and len(self.threshold_config) > 1:
        #     previous_lower: float = 0.0
        #     previous_upper: float = 0.0
        #     for config in self.threshold_config:
        #         if config.lower_bound < previous_lower:
        #             raise InvalidConfigurationError(
        #                 "Threshold config must be specified in ascending order of lower bound"
        #             )
        #         if config.upper_bound != np.inf and previous_upper != 0.0 and config.lower_bound != previous_upper:
        #             raise InvalidConfigurationError(
        #                 "If multiple thresholds are specified and are bounded, they must be consecutive"
        #             )
        #         previous_lower = config.lower_bound

        return self


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


class MeteorologyConfig(BaseConfigModel):
    evaluation_data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing evaluation data",
            repr=True,
            frozen=True,
        ),
        AfterValidator(dir_must_exist),
    ]


class DataConfig(BaseConfigModel):
    # Config for data, this would cover both observational data and weather data
    name: str
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
