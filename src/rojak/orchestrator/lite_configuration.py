from pydantic import Field

from rojak.orchestrator.configuration import (
    BaseConfigModel,
    CreateDirectoryPath,
    SpatialDomain,
    TurbulenceDiagnostics,
)


class DistributionParametersContext(BaseConfigModel):
    spatial_domain: SpatialDomain
    data_dir: CreateDirectoryPath = Field(
        description="Path to directory containing calibration data", repr=True, frozen=True
    )
    glob_pattern: str = Field(description="Glob pattern to match to get the data files", repr=True, frozen=True)
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
    output_dir: CreateDirectoryPath = Field(description="Output directory", repr=True, frozen=True)
