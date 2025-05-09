from pydantic import BaseModel


class Context(BaseModel):
    # Root configuration
    ...


class TurbulenceConfig(BaseModel):
    # Config for turbulence analysis
    ...


class ContrailsConfig(BaseModel):
    # Config for contrail analysis
    ...


class DataConfig(BaseModel):
    # Config for data, this would cover both observational data and weather data
    ...
