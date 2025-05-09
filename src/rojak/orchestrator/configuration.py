from typing import Annotated

from pydantic import BaseModel, Field


class TurbulenceConfig(BaseModel):
    # Config for turbulence analysis
    ...


class ContrailsConfig(BaseModel):
    # Config for contrail analysis
    ...


class DataConfig(BaseModel):
    # Config for data, this would cover both observational data and weather data
    ...


class Context(BaseModel):
    # Root configuration
    name: Annotated[
        str, Field(description="Name of run", repr=True, strict=True, frozen=True)
    ]
    turbulence_config: TurbulenceConfig | None = None
    contrails_config: ContrailsConfig | None = None
    data_config: DataConfig
