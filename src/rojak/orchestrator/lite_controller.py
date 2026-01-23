from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from rich.progress import track

from rojak.core.data import load_from_folder
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import MetDataSource
from rojak.turbulence.analysis import ComputeDistributionParametersForEDR
from rojak.turbulence.diagnostic import DiagnosticFactory
from rojak.utilities.types import DistributionParameters

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from rojak.core.data import CATData
    from rojak.orchestrator.lite_configuration import BaseTurbulenceContext, TurbulenceContextWithOutput


DISTRIBUTION_PARAMS_TYPE_ADAPTER: TypeAdapter[dict[str, DistributionParameters]] = TypeAdapter(
    dict[str, DistributionParameters]
)


def _instantiate_diagnostic_factory(context: "BaseTurbulenceContext", /) -> DiagnosticFactory:
    if context.data_source != MetDataSource.ERA5:
        raise NotImplementedError("Only ERA5 data is currently supported")
    source_data: CATData = Era5Data(
        load_from_folder(context.data_dir, glob_pattern=context.glob_pattern, chunks=context.chunks, engine="h5netcdf"),
    ).to_clear_air_turbulence_data(context.spatial_domain)
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
