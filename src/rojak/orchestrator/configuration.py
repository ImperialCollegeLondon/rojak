from enum import StrEnum
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field, ValidationError, AfterValidator
from pathlib import Path


class InvalidConfiguration(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


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


class TurbulenceConfig(BaseModel):
    # Config for turbulence analysis
    evaluation_data_dir: Annotated[
        Path,
        Field(
            description="Path to directory containing evaluation data",
            repr=True,
            strict=True,
            frozen=True,
        ),
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
    ]
    severities: Annotated[
        list[TurbulenceSeverity],
        Field(description="Target turbulence severity", repr=True),
    ] = [TurbulenceSeverity.LIGHT]
    ...


class ContrailsConfig(BaseModel):
    # Config for contrail analysis
    ...


class DataConfig(BaseModel):
    # Config for data, this would cover both observational data and weather data
    name: str
    ...


class Context(BaseModel):
    # Root configuration
    name: Annotated[
        str, Field(description="Name of run", repr=True, strict=True, frozen=True)
    ]
    image_format: Annotated[
        str, Field(description="Format of output plots", strict=True, frozen=True)
    ]
    output_dir: Annotated[
        Path,
        Field(description="Output directory", repr=True, strict=True, frozen=True),
        AfterValidator(make_dir_if_not_present),
    ]
    plots_dir: Annotated[
        Path,
        Field(description="Plots directory", repr=True, strict=True, frozen=True),
        AfterValidator(make_dir_if_not_present),
    ]
    turbulence_config: TurbulenceConfig | None = None
    contrails_config: ContrailsConfig | None = None
    data_config: DataConfig

    @classmethod
    def from_yaml(cls, path: "Path") -> Self:
        if path.is_file():
            data: dict = {}
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            try:
                instance = cls.model_validate(data)
            except ValidationError as e:
                raise InvalidConfiguration(str(e)) from e
            return instance
        else:
            raise InvalidConfiguration("Configuration file not found or is not a file.")
