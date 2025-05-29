from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

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
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            try:
                instance = cls.model_validate(data)
            except ValidationError as e:
                raise InvalidConfigurationError(str(e)) from e
            return instance
        raise InvalidConfigurationError("Configuration file not found or is not a file.")


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
    ]
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
