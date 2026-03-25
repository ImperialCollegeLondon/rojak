from datetime import datetime
from typing import TYPE_CHECKING

import distributed
import xarray as xr
from dask.base import is_dask_collection
from pydantic import TypeAdapter
from rich.progress import track

from rojak.core.data import load_from_folder
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import MetDataSource, TurbulenceThresholds
from rojak.plot.turbulence_plotter import chain_diagnostic_names
from rojak.turbulence.analysis import (
    ComputeDistributionParametersForEDR,
    HistogramData,
    MatthewsCorrelationOnThresholdedDiagnostics,
    TurbulenceIntensityThresholds,
)
from rojak.turbulence.diagnostic import DiagnosticFactory
from rojak.utilities.types import DistributionParameters

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from rojak.core.data import CATData
    from rojak.orchestrator.configuration import TurbulenceSeverity, TurbulenceThresholdMode
    from rojak.orchestrator.lite_configuration import (
        BaseTurbulenceContext,
        DiagnosticThresholdsContext,
        TurbulenceContextWithAdditionalPath,
        TurbulenceContextWithOutput,
    )
    from rojak.utilities.types import DiagnosticName


# See pydantic docs about only instantiating the type adapter once
# https://docs.pydantic.dev/latest/concepts/performance/#typeadapter-instantiated-once
# str is DiagnosticName
THRESHOLDS_TYPE_ADAPTER: TypeAdapter[dict[str, TurbulenceThresholds]] = TypeAdapter(dict[str, TurbulenceThresholds])
HISTOGRAM_DATA_TYPE_ADAPTER: TypeAdapter[dict[str, HistogramData]] = TypeAdapter(dict[str, HistogramData])
DISTRIBUTION_PARAMS_TYPE_ADAPTER: TypeAdapter[dict[str, DistributionParameters]] = TypeAdapter(
    dict[str, DistributionParameters]
)


def _load_era5_data(context: "BaseTurbulenceContext", /) -> "CATData":
    if context.data_source != MetDataSource.ERA5:
        raise NotImplementedError("Only ERA5 data is currently supported")
    return Era5Data(
        load_from_folder(context.data_dir, glob_pattern=context.glob_pattern, chunks=context.chunks, engine="h5netcdf"),
    ).to_clear_air_turbulence_data(context.spatial_domain)


def _instantiate_diagnostic_factory(context: "BaseTurbulenceContext", /) -> DiagnosticFactory:
    if context.data_source != MetDataSource.ERA5:
        raise NotImplementedError("Only ERA5 data is currently supported")
    source_data: CATData = _load_era5_data(context)
    return DiagnosticFactory(source_data)


def export_json[T](
    obj_to_dump: T,
    output_dir: "Path",
    start_time: str,
    type_adpater: TypeAdapter[T],
    fname_identifier: str,
    /,
    *,
    indent: int = 4,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{fname_identifier}_{start_time}.json"

    with output_file.open("wb") as f:
        _ = f.write(type_adpater.dump_json(obj_to_dump, indent=indent))


def load_thresholds_from_file(file_path: "Path") -> "Mapping[DiagnosticName, TurbulenceThresholds]":
    assert file_path.exists()
    assert file_path.is_file()

    json_str: str = file_path.read_text()
    return THRESHOLDS_TYPE_ADAPTER.validate_json(json_str)


def blocking_wait_futures(dask_collection: object) -> None:
    if is_dask_collection(dask_collection):
        _ = distributed.wait(distributed.futures_of(dask_collection))


def compute_distribution_parameters(context: "TurbulenceContextWithOutput", /) -> None:
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    diagnostic_factory: DiagnosticFactory = _instantiate_diagnostic_factory(context)

    distribution_parameters: dict[str, DistributionParameters] = {}
    for diagnostic in track(context.diagnostics):
        computed_diagnostic: xr.DataArray = diagnostic_factory.create(diagnostic).computed_value
        distribution_parameters[diagnostic] = ComputeDistributionParametersForEDR(computed_diagnostic).execute()
        del computed_diagnostic

    export_json(
        distribution_parameters,
        (context.output_dir / context.name),
        start_time,
        DISTRIBUTION_PARAMS_TYPE_ADAPTER,
        "distribution_params",
    )


def compute_thresholds(context: "DiagnosticThresholdsContext", /) -> None:
    start_time: str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    diagnostic_factory: DiagnosticFactory = _instantiate_diagnostic_factory(context)

    diagnostic_thresholds: Mapping[str, TurbulenceThresholds] = {}
    for diagnostic in track(context.diagnostics):
        computed_diagnostic: xr.DataArray = diagnostic_factory.create(diagnostic).computed_value.persist()
        diagnostic_thresholds[diagnostic] = TurbulenceIntensityThresholds(
            context.percentile_thresholds, computed_diagnostic
        ).execute()
        del computed_diagnostic

    export_json(
        diagnostic_thresholds, (context.output_dir / context.name), start_time, THRESHOLDS_TYPE_ADAPTER, "thresholds"
    )


def export_turbulence_diagnostics(
    context: "TurbulenceContextWithOutput",
    *,
    start_time: str,
) -> None:
    diagnostic_factory: DiagnosticFactory = _instantiate_diagnostic_factory(context)

    output_to: Path = context.output_dir / context.name / start_time

    for diagnostic in track(context.diagnostics):
        instantiated_diagnostic = diagnostic_factory.create(diagnostic)
        instantiated_diagnostic.to_zarr(output_to, file_name=str(diagnostic))

        # Force all the futures to be completed to ensure diagnostic is deleted
        blocking_wait_futures(instantiated_diagnostic)
        del instantiated_diagnostic


def correlation_between_diagnostics(
    context: "TurbulenceContextWithAdditionalPath",
    *,
    severities: list["TurbulenceSeverity"],
    threshold_mode: "TurbulenceThresholdMode",
) -> None:
    thresholds: Mapping[DiagnosticName, TurbulenceThresholds] = load_thresholds_from_file(context.load_from)
    assert set(context.diagnostics).issubset(thresholds.keys()), "Diagnostics must be a subset of diagnostic thresholds"

    # Version 1: Load everything into a dataset and hope that it doesn't load it until compute
    all_diagnostics = xr.Dataset(
        data_vars={
            # pyright takes issue with the chunks kwarg. This is due to it being untyped
            # See https://github.com/pydata/xarray/issues/11221
            diagnostic_name: xr.open_zarr(
                context.data_dir / f"{diagnostic_name}.zarr",
                chunks="auto",  # pyright: ignore[reportArgumentType]
                drop_variables=["altitude", "expver", "number"],
                consolidated=False,
            )[diagnostic_name]
            for diagnostic_name in context.diagnostics
        }
    )
    matthews_correlation: xr.DataArray = MatthewsCorrelationOnThresholdedDiagnostics(
        all_diagnostics, severities, thresholds, threshold_mode
    ).execute()

    chained_names: str = chain_diagnostic_names(context.diagnostics)
    # False positive by pyright - StoreLike includes Path
    # See https://zarr.readthedocs.io/en/v3.1.5/api/zarr/storage/#zarr.storage.StoreLike
    _ = matthews_correlation.to_zarr(
        context.output_dir / f"matthews_correlation_{chained_names}.zarr",  # pyright: ignore[reportArgumentType]
        mode="w",
        zarr_format=2,
    )
