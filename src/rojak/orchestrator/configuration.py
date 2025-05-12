from typing import Annotated, Self, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from pathlib import Path


class InvalidConfiguration(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class TurbulenceConfig(BaseModel):
    # Config for turbulence analysis
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
