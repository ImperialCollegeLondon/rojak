"""
Business logic behind the ``rojak lite turbulence`` commands

Each public function here implements one step of the low-memory, step-by-step turbulence workflow driven by
:mod:`rojak.cli.lite_interface`: computing diagnostic distribution parameters
(:func:`compute_distribution_parameters`), severity thresholds (:func:`compute_thresholds`), exporting raw
diagnostic values (:func:`export_turbulence_diagnostics`), converting diagnostics into an ensemble EDR
(:func:`compute_and_export_ensemble_edr`), and computing correlation between thresholded diagnostics
(:func:`correlation_between_diagnostics`). Diagnostic values are obtained either by computing them fresh from
meteorological data, or by loading them back from a previous :func:`export_turbulence_diagnostics` run (see
:class:`~rojak.orchestrator.lite_configuration.DiagnosticsFormat` and :func:`get_computed_diagnostic_helper`).
"""

import functools
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, assert_never

import distributed
import xarray as xr
from dask.base import is_dask_collection
from pydantic import TypeAdapter
from rich.progress import track

from rojak.core.calculations import XrAggregationMethod, apply_data_var_reduction
from rojak.core.constants import TWENTIES_CLIMATOLOGICAL_PARAMETER
from rojak.core.data import load_from_folder
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import MetDataSource, TurbulenceThresholds
from rojak.orchestrator.lite_configuration import DiagnosticsFormat
from rojak.plot.turbulence_plotter import chain_diagnostic_names
from rojak.turbulence.analysis import (
    ComputeDistributionParametersForEDR,
    HistogramData,
    MatthewsCorrelationOnThresholdedDiagnostics,
    TransformToEDR,
    TurbulenceIntensityThresholds,
)
from rojak.turbulence.diagnostic import DiagnosticFactory
from rojak.utilities.types import DistributionParameters

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from rojak.core.data import CATData
    from rojak.orchestrator.configuration import TurbulenceDiagnostics, TurbulenceSeverity, TurbulenceThresholdMode
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
    """
    Load the ERA5 data described by ``context`` and adapt it into a :class:`~rojak.core.data.CATData`

    Args:
        context: Configuration describing where the data is and how to load it

    Returns:
        :class:`~rojak.core.data.CATData` restricted to ``context.spatial_domain``

    Raises:
        NotImplementedError: If ``context.data_source`` is not ERA5
    """
    if context.data_source != MetDataSource.ERA5:
        raise NotImplementedError("Only ERA5 data is currently supported")
    return Era5Data(
        load_from_folder(context.data_dir, glob_pattern=context.glob_pattern, chunks=context.chunks, engine="h5netcdf"),
    ).to_clear_air_turbulence_data(context.spatial_domain)


def _instantiate_diagnostic_factory(context: "BaseTurbulenceContext", /) -> DiagnosticFactory:
    """
    Build a :class:`~rojak.turbulence.diagnostic.DiagnosticFactory` from the meteorological data in ``context``

    Args:
        context: Configuration describing where the data is and how to load it

    Returns:
        :class:`~rojak.turbulence.diagnostic.DiagnosticFactory` backed by ``context``'s data

    Raises:
        NotImplementedError: If ``context.data_source`` is not ERA5
    """
    if context.data_source != MetDataSource.ERA5:
        raise NotImplementedError("Only ERA5 data is currently supported")
    source_data: CATData = _load_era5_data(context)
    return DiagnosticFactory(source_data)


def _open_diagnostic_zarr(path: "Path", name: "TurbulenceDiagnostics") -> xr.DataArray:
    """
    Open a single diagnostic previously exported by :func:`export_turbulence_diagnostics`

    Args:
        path: Path to the diagnostic's zarr store
        name: Name of the diagnostic (i.e. the data variable to select from the store)

    Returns:
        The diagnostic's computed values

    Raises:
        AssertionError: If ``path`` does not have a ``.zarr`` suffix
    """
    assert path.suffix == ".zarr"
    return xr.open_zarr(
        # pyright takes issue with the chunks kwarg. This is due to it being untyped
        # See https://github.com/pydata/xarray/issues/11221
        path,
        chunks="auto",  # pyright: ignore[reportArgumentType]
        drop_variables=["altitude", "expver", "number"],
        consolidated=False,
    )[name]


def create_diagnostic_dataset(context: "BaseTurbulenceContext", /, diagnostics_from: DiagnosticsFormat) -> xr.Dataset:
    """
    Combine every diagnostic configured in ``context`` into a single Dataset

    Args:
        context: Configuration listing the diagnostics to include
        diagnostics_from: Where to obtain each diagnostic's value from, see :func:`get_computed_diagnostic_helper`

    Returns:
        Dataset with one data variable per diagnostic in ``context.diagnostics``
    """
    get_computed_diagnostic: Callable[[str], xr.DataArray] = get_computed_diagnostic_helper(context, diagnostics_from)
    return xr.Dataset(
        data_vars={diagnostic_name: get_computed_diagnostic(diagnostic_name) for diagnostic_name in context.diagnostics}
    )


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
    """
    Serialise ``obj_to_dump`` to a timestamped JSON file under ``output_dir``

    Args:
        obj_to_dump: Object to serialise. Must match the type ``type_adpater`` was built for.
        output_dir: Directory to write the JSON file into, created (along with any missing parents) if needed
        start_time: Timestamp identifying this run, included in the output filename
        type_adpater: Pydantic :class:`~pydantic.TypeAdapter` used to serialise ``obj_to_dump``
        fname_identifier: Prefix for the output filename, which is ``f"{fname_identifier}_{start_time}.json"``
        indent: Indentation (in spaces) for the written JSON. Defaults to ``4``.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{fname_identifier}_{start_time}.json"

    with output_file.open("wb") as f:
        _ = f.write(type_adpater.dump_json(obj_to_dump, indent=indent))


def load_thresholds_from_file(file_path: "Path") -> "Mapping[DiagnosticName, TurbulenceThresholds]":
    """
    Load a mapping of diagnostic name to :class:`~rojak.orchestrator.configuration.TurbulenceThresholds` from JSON

    Loads the file written by :func:`compute_thresholds` (via :func:`export_json`).

    Args:
        file_path: Path to the thresholds JSON file

    Returns:
        Mapping from diagnostic name to its thresholds

    Raises:
        AssertionError: If ``file_path`` does not exist or is not a file
    """
    assert file_path.exists()
    assert file_path.is_file()

    json_str: str = file_path.read_text()
    return THRESHOLDS_TYPE_ADAPTER.validate_json(json_str)


def load_distribution_parameters_from_file(file_path: "Path") -> "Mapping[DiagnosticName, DistributionParameters]":
    """
    Load a mapping of diagnostic name to :class:`~rojak.utilities.types.DistributionParameters` from JSON

    Loads the file written by :func:`compute_distribution_parameters` (via :func:`export_json`).

    Args:
        file_path: Path to the distribution parameters JSON file

    Returns:
        Mapping from diagnostic name to its distribution parameters

    Raises:
        FileNotFoundError: If ``file_path`` does not exist
        ValueError: If ``file_path`` is not a file
    """
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if not file_path.is_file():
        raise ValueError(f"File in  {file_path} is not a file")

    json_str: str = file_path.read_text()
    return DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(json_str)


def blocking_wait_futures(dask_collection: object) -> None:
    """
    Block until every dask future backing ``dask_collection`` has completed

    A no-op if ``dask_collection`` is not a dask collection.

    Args:
        dask_collection: Object to wait on the futures of
    """
    if is_dask_collection(dask_collection):
        _ = distributed.wait(distributed.futures_of(dask_collection))


def get_computed_diagnostic_helper(
    context: "BaseTurbulenceContext", diagnostics_from: DiagnosticsFormat
) -> Callable[[str], xr.DataArray]:
    """
    Build a function computing/loading a single diagnostic's value by name

    Args:
        context: Configuration describing the data to compute diagnostics from, or the directory to load
            precomputed diagnostics from
        diagnostics_from: If :attr:`~DiagnosticsFormat.FROM_MET_DATA`, the returned function computes the
            diagnostic fresh from ``context``'s meteorological data. If
            :attr:`~DiagnosticsFormat.PRECOMPUTED_FROM_ZARR`, it loads the diagnostic's previously exported values
            from ``context.data_dir`` (see :func:`export_turbulence_diagnostics`).

    Returns:
        Function mapping a diagnostic name to its computed values
    """
    match diagnostics_from:
        case DiagnosticsFormat.FROM_MET_DATA:
            diagnostic_factory: DiagnosticFactory = _instantiate_diagnostic_factory(context)
            return functools.partial(lambda f, diagnostic: f.create(diagnostic).computed_value, diagnostic_factory)
        case DiagnosticsFormat.PRECOMPUTED_FROM_ZARR:
            return functools.partial(
                lambda data_dir, diagnostic: _open_diagnostic_zarr(data_dir / f"{diagnostic!s}.zarr", diagnostic),
                context.data_dir,
            )
        case _ as unreachable:
            assert_never(unreachable)


def compute_distribution_parameters(
    context: "TurbulenceContextWithOutput", /, diagnostics_from: DiagnosticsFormat
) -> None:
    """
    Compute and export the log-normal distribution parameters of every diagnostic configured in ``context``

    Implements ``rojak lite turbulence distribution-parameters``
    (:func:`rojak.cli.lite_interface.distribution_parameters`). Writes a JSON file (see :func:`export_json`) under
    ``context.output_dir / context.name``, loadable via :func:`load_distribution_parameters_from_file`.

    Args:
        context: Configuration listing the diagnostics to compute and where to write the result
        diagnostics_from: Where to obtain each diagnostic's value from
    """
    start_time: str = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H_%M_%S")
    distribution_parameters: dict[str, DistributionParameters] = {}

    get_computed_diagnostic: Callable[[str], xr.DataArray] = get_computed_diagnostic_helper(context, diagnostics_from)
    for diagnostic in track(context.diagnostics):
        computed_diagnostic: xr.DataArray = get_computed_diagnostic(diagnostic)
        distribution_parameters[diagnostic] = ComputeDistributionParametersForEDR(computed_diagnostic).execute()
        del computed_diagnostic

    export_json(
        distribution_parameters,
        (context.output_dir / context.name),
        start_time,
        DISTRIBUTION_PARAMS_TYPE_ADAPTER,
        "distribution_params",
    )


def compute_thresholds(context: "DiagnosticThresholdsContext", /, diagnostics_from: DiagnosticsFormat) -> None:
    """
    Compute and export percentile-based severity thresholds for every diagnostic configured in ``context``

    Implements ``rojak lite turbulence turbulence-thresholds``
    (:func:`rojak.cli.lite_interface.turbulence_thresholds`). Writes a JSON file (see :func:`export_json`) under
    ``context.output_dir / context.name``, loadable via :func:`load_thresholds_from_file`.

    Args:
        context: Configuration listing the diagnostics to compute, the percentile thresholds to use, and where to
            write the result
        diagnostics_from: Where to obtain each diagnostic's value from
    """
    start_time: str = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H_%M_%S")
    get_computed_diagnostic: Callable[[str], xr.DataArray] = get_computed_diagnostic_helper(context, diagnostics_from)

    diagnostic_thresholds: Mapping[str, TurbulenceThresholds] = {}
    for diagnostic in track(context.diagnostics):
        computed_diagnostic: xr.DataArray = get_computed_diagnostic(diagnostic)
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
    """
    Compute every diagnostic configured in ``context`` from meteorological data and export each to its own zarr store

    Implements ``rojak lite turbulence export-diagnostic`` (:func:`rojak.cli.lite_interface.export_diagnostic`).
    Each diagnostic is written to ``context.output_dir / context.name / start_time / {diagnostic}.zarr``, and can
    be reloaded via :func:`get_computed_diagnostic_helper`/:func:`_open_diagnostic_zarr` by pointing a later
    step's ``data_dir`` at that ``start_time`` directory.

    Args:
        context: Configuration listing the diagnostics to compute, the meteorological data to compute them from,
            and where to write the result
        start_time: Timestamp identifying this run, used to name the output subdirectory
    """
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
    """
    Compute the Matthew's Correlation Coefficient between every pair of diagnostics, thresholded at ``severities``

    Implements ``rojak lite turbulence diagnostic-correlation``
    (:func:`rojak.cli.lite_interface.diagnostic_correlation`). Loads thresholds from ``context.load_from`` (see
    :func:`load_thresholds_from_file`, produced by :func:`compute_thresholds`) and diagnostics previously exported
    to zarr (see :func:`export_turbulence_diagnostics`) from ``context.data_dir``, then exports the result to
    ``context.output_dir``.

    Args:
        context: Configuration listing the diagnostics to correlate, the thresholds file to load, and where to
            read diagnostics from/write the result to
        severities: Turbulence severities to threshold diagnostics at before computing their correlation
        threshold_mode: Whether severity thresholds are bounded intervals or lower bounds

    Raises:
        AssertionError: If ``context.diagnostics`` is not a subset of the diagnostics with thresholds loaded from
            ``context.load_from``
    """
    thresholds: Mapping[DiagnosticName, TurbulenceThresholds] = load_thresholds_from_file(context.load_from)
    assert set(context.diagnostics).issubset(thresholds.keys()), "Diagnostics must be a subset of diagnostic thresholds"

    # Version 1: Load everything into a dataset and hope that it doesn't load it until compute
    # all_diagnostics = load_diagnostics_from_individual_zarrs(context.diagnostics, context.data_dir)
    all_diagnostics = create_diagnostic_dataset(context, DiagnosticsFormat.PRECOMPUTED_FROM_ZARR)
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


def compute_and_export_ensemble_edr(
    context: "TurbulenceContextWithAdditionalPath", /, diagnostics_from: DiagnosticsFormat
) -> None:
    """
    Convert every diagnostic configured in ``context`` into EDR and export the ensemble-mean result

    Implements ``rojak lite turbulence export-ensemble-edr``
    (:func:`rojak.cli.lite_interface.export_ensemble_edr`). Distribution parameters are loaded from
    ``context.load_from`` (see :func:`load_distribution_parameters_from_file`, produced by
    :func:`compute_distribution_parameters`); each diagnostic is converted to EDR using those parameters together
    with the ``TWENTIES_CLIMATOLOGICAL_PARAMETER`` climatological scaling constants, then averaged across
    diagnostics (:data:`~rojak.core.calculations.XrAggregationMethod.MEAN`) to give a single ensemble EDR, written
    to ``context.output_dir``.

    Args:
        context: Configuration listing the diagnostics to convert, the distribution parameters file to load, and
            where to read diagnostics from/write the result to
        diagnostics_from: Where to obtain each diagnostic's value from
    """

    def __transform_to_edr(diagnostic: xr.DataArray) -> xr.DataArray:
        """Convert a single diagnostic's values into EDR using its precomputed distribution parameters"""
        return TransformToEDR(
            diagnostic,
            c1=TWENTIES_CLIMATOLOGICAL_PARAMETER.c1,
            c2=TWENTIES_CLIMATOLOGICAL_PARAMETER.c2,
            mean=distribution_params[str(diagnostic.name)].mean,
            variance=distribution_params[str(diagnostic.name)].variance,
        ).execute()

    distribution_params = load_distribution_parameters_from_file(context.load_from)
    all_diagnostics = create_diagnostic_dataset(context, diagnostics_from)
    diagnostics_as_edr = all_diagnostics.map(__transform_to_edr, keep_attrs=False)
    # Explicitly set append=False to return a DataArray
    ensemble_edr: xr.DataArray = apply_data_var_reduction(diagnostics_as_edr, XrAggregationMethod.MEAN, append=False)

    chained_names: str = chain_diagnostic_names(context.diagnostics)
    # False positive by pyright - StoreLike includes Path
    # See https://zarr.readthedocs.io/en/v3.1.5/api/zarr/storage/#zarr.storage.StoreLike
    _ = ensemble_edr.to_zarr(
        context.output_dir / f"ensemble_edr_{chained_names}.zarr",  # pyright: ignore[reportArgumentType]
        mode="w",
        zarr_format=2,
    )
