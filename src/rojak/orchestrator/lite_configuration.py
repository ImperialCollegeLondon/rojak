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

The meteorological data for each step is described by :attr:`BaseTurbulenceContext.met_data`. Configurations that
only give ``data_dir`` (and optionally ``glob_pattern``/``data_source``) are still accepted and are rewritten into
that form.
"""

from enum import StrEnum
from typing import Any

from pydantic import DirectoryPath, Field, FilePath, model_validator

from rojak.orchestrator.configuration import (
    BaseConfigModel,
    CreateDirectoryPath,
    InvalidConfigurationError,
    MeteorologyConfig,
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
        description="Path to directory containing calibration data (or precomputed diagnostics)",
        repr=True,
        frozen=True,
    )
    met_data: MeteorologyConfig = Field(
        description="Meteorological data to compute diagnostics from. Defaults to the files in data_dir",
        repr=True,
        frozen=True,
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

    @model_validator(mode="before")
    @classmethod
    def default_met_data_from_data_dir(cls, data: Any) -> Any:  # noqa: ANN401
        """
        If :attr:`met_data` is not given, build it from ``data_dir`` and the legacy ``glob_pattern`` and
        ``data_source`` fields (which are only accepted when :attr:`met_data` is not given)
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        glob_pattern = data.pop("glob_pattern", None)
        data_source = data.pop("data_source", None)
        if data.get("met_data") is not None:
            if glob_pattern is not None or data_source is not None:
                raise InvalidConfigurationError("glob_pattern and data_source must be given within met_data")
            return data
        if "data_dir" in data:
            location: dict[str, Any] = {"type": "files", "root": str(data["data_dir"])}
            if glob_pattern is not None:
                location["glob_pattern"] = glob_pattern
            data["met_data"] = {"location": location} | ({"data_source": data_source} if data_source else {})
        return data


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
