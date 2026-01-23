from pydantic import Field

from rojak.orchestrator.configuration import (
    BaseConfigModel,
    CreateDirectoryPath,
    MetDataSource,
    SpatialDomain,
    TurbulenceDiagnostics,
)


class BaseTurbulenceContext(BaseConfigModel):
    name: str = Field(description="Identifier for this configuration", repr=True, frozen=True)
    spatial_domain: SpatialDomain
    data_dir: CreateDirectoryPath = Field(
        description="Path to directory containing calibration data", repr=True, frozen=True
    )
    chunks: dict[str, float | str] = Field(
        description="How data should be chunked (dask)",
        frozen=True,
        repr=True,
        strict=True,
    )
    diagnostics: list[TurbulenceDiagnostics] = Field(
        description="List of turbulence diagnostics to evaluate",
        repr=True,
        frozen=True,
    )
    glob_pattern: str = Field(
        default="*.nc",
        description="Glob pattern to match to get the data files",
        repr=True,
        frozen=True,
        validate_default=True,
    )
    data_source: MetDataSource = Field(
        default=MetDataSource.ERA5,
        description="Source of meterological data",
        repr=True,
        frozen=True,
        validate_default=True,
    )


class TurbulenceContextWithOutput(BaseTurbulenceContext):
    output_dir: CreateDirectoryPath = Field(description="Output directory", repr=True, frozen=True)
