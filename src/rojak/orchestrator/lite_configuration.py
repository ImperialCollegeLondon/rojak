"""
Pydantic schema for the YAML configuration consumed by ``rojak lite turbulence`` commands

Unlike :class:`~rojak.orchestrator.configuration.Context`, which describes an entire ``rojak run`` invocation,
each class here configures a single step of the lower-memory ``rojak lite turbulence`` workflow (see
:mod:`rojak.cli.lite_interface`). :class:`BaseTurbulenceContext` is the common base every step shares.
:class:`TurbulenceContextWithOutput` adds an output directory, for steps that only produce output
(``export-diagnostic``, ``distribution-parameters``). :class:`TurbulenceContextWithAdditionalPath` further adds a
path to load a previous step's output from, for steps that also consume one (``export-ensemble-edr``,
``diagnostic-correlation``). :class:`DiagnosticThresholdsContext` adds the percentile thresholds needed by
``turbulence-thresholds``.
"""

from enum import StrEnum

from pydantic import DirectoryPath, Field, FilePath

from rojak.orchestrator.configuration import (
    BaseConfigModel,
    CreateDirectoryPath,
    MetDataSource,
    SpatialDomain,
    TurbulenceDiagnostics,
    TurbulenceThresholds,
)


class DiagnosticsFormat(StrEnum):
    """Where a ``rojak lite turbulence`` command should obtain diagnostic values from"""

    FROM_MET_DATA = "from_met_data"
    PRECOMPUTED_FROM_ZARR = "precomputed_from_zarr"


class BaseTurbulenceContext(BaseConfigModel):
    """Configuration shared by every ``rojak lite turbulence`` step"""

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
    """:class:`BaseTurbulenceContext` plus an output directory, for steps that write results to disk"""

    output_dir: CreateDirectoryPath = Field(description="Output directory", repr=True, frozen=True)


class TurbulenceContextWithAdditionalPath(TurbulenceContextWithOutput):
    """:class:`TurbulenceContextWithOutput` plus a path to a previous step's output to load from"""

    load_from: FilePath | DirectoryPath = Field(
        description="Additional file or directory path to load from", repr=True, frozen=True
    )


class DiagnosticThresholdsContext(TurbulenceContextWithOutput):
    """:class:`TurbulenceContextWithOutput` plus the percentile thresholds used by ``turbulence-thresholds``"""

    percentile_thresholds: TurbulenceThresholds = Field(description="Percentile thresholds", repr=True, frozen=True)
