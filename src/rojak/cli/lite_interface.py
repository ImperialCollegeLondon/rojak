"""
``rojak lite`` command group: lower-memory, step-by-step turbulence workflows

Unlike ``rojak run`` (:func:`rojak.cli.main.run`), which computes and evaluates every configured turbulence
diagnostic in a single pass, the ``rojak lite turbulence ...`` commands (:data:`turbulence_app`) split a full
turbulence evaluation into separate steps that can each be run (and re-run) independently, keeping peak memory
usage lower for large datasets:

1. :func:`export_diagnostic` -- compute the configured diagnostics from meteorological data and save them to zarr.
2. :func:`distribution_parameters` -- compute each diagnostic's log-normal distribution parameters (for EDR).
3. :func:`turbulence_thresholds` -- compute each diagnostic's percentile-based severity thresholds.
4. :func:`export_ensemble_edr` -- convert diagnostics into Eddy Dissipation Rate (EDR) using the output of step 2.
5. :func:`diagnostic_correlation` -- compute correlation between diagnostics using the output of step 3.

Each command loads its own YAML configuration file (a subclass of
:class:`~rojak.orchestrator.lite_configuration.BaseTurbulenceContext`) and starts its own local Dask
:class:`~distributed.Client` for the duration of the command.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from distributed import Client

from rojak.orchestrator.configuration import TurbulenceSeverity, TurbulenceThresholdMode
from rojak.orchestrator.lite_configuration import (
    DiagnosticsFormat,
    DiagnosticThresholdsContext,
    TurbulenceContextWithAdditionalPath,
    TurbulenceContextWithOutput,
)
from rojak.orchestrator.lite_controller import (
    compute_and_export_ensemble_edr,
    compute_distribution_parameters,
    compute_thresholds,
    correlation_between_diagnostics,
    export_turbulence_diagnostics,
)

# Root application for this interface
lite_app = typer.Typer(
    help=(
        "Lower-memory, step-by-step turbulence workflows for datasets too large to comfortably run with "
        "`rojak run` in a single pass."
    ),
    no_args_is_help=True,
)

# Turbulence Functionality
turbulence_app = typer.Typer(
    help=(
        "Compute turbulence diagnostics, distribution parameters, thresholds, EDR, and correlations one step at "
        "a time. Each command takes its own YAML configuration file; run them in the order shown in "
        "`rojak lite --help` for a full evaluation."
    ),
    no_args_is_help=True,
)

# Add applications related to lite app here
lite_app.add_typer(turbulence_app, name="turbulence")

# Shared Argument definitions, reused by every command below so their help text stays consistent.
ConfigFileArgument = Annotated[
    Path,
    typer.Argument(
        help=(
            "Path to the YAML configuration file for this step (input data, spatial domain, diagnostics, and "
            "output location)."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]
DiagnosticsFromArgument = Annotated[
    DiagnosticsFormat,
    typer.Argument(
        help=(
            "Where to get diagnostic values from: 'from_met_data' computes them from the meteorological data in "
            "the config file, 'precomputed_from_zarr' loads values previously saved by 'export-diagnostic'."
        ),
    ),
]


@turbulence_app.command(
    help=(
        "Compute and export the log-normal distribution parameters (mean, variance) of each configured "
        "diagnostic's values. These are needed later by 'export-ensemble-edr' to convert diagnostic values into "
        "EDR; the config file's 'output_dir' controls where the result is written."
    )
)
def distribution_parameters(config_file: ConfigFileArgument, diagnostics_from: DiagnosticsFromArgument) -> None:
    """
    Compute and export the log-normal distribution parameters of each diagnostic configured in ``config_file``

    Args:
        config_file: Path to the YAML configuration file
        diagnostics_from: Where to obtain diagnostic values from
    """
    client = Client()
    context: TurbulenceContextWithOutput = TurbulenceContextWithOutput.from_yaml(config_file)
    compute_distribution_parameters(context, diagnostics_from)
    _ = client.close()


@turbulence_app.command(
    help=(
        "Compute and export percentile-based turbulence severity thresholds for each configured diagnostic. "
        "These are needed later by 'diagnostic-correlation'; the config file's 'output_dir' controls where the "
        "result is written."
    )
)
def turbulence_thresholds(config_file: ConfigFileArgument, diagnostics_from: DiagnosticsFromArgument) -> None:
    """
    Compute and export percentile-based severity thresholds for each diagnostic configured in ``config_file``

    Args:
        config_file: Path to the YAML configuration file
        diagnostics_from: Where to obtain diagnostic values from
    """
    client = Client()
    context: DiagnosticThresholdsContext = DiagnosticThresholdsContext.from_yaml(config_file)
    compute_thresholds(context, diagnostics_from)
    _ = client.close()


@turbulence_app.command(
    help=(
        "Compute each diagnostic configured in the config file from meteorological data and export the raw "
        "values to zarr, under the config file's 'output_dir'. A copy of the config file is saved alongside the "
        "output for provenance. This is usually the first step of a lite turbulence workflow."
    )
)
def export_diagnostic(config_file: ConfigFileArgument) -> None:
    """
    Compute and export the raw diagnostic values configured in ``config_file``

    Args:
        config_file: Path to the YAML configuration file
    """
    client = Client()

    context: TurbulenceContextWithOutput = TurbulenceContextWithOutput.from_yaml(config_file)

    start_time: str = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H_%M_%S")
    output_to: Path = context.output_dir / context.name / start_time
    output_to.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy(config_file, output_to / config_file.name)

    export_turbulence_diagnostics(context, start_time=start_time)

    _ = client.close()


@turbulence_app.command(
    help=(
        "Convert each configured diagnostic into Eddy Dissipation Rate (EDR) and export the ensemble-mean result "
        "to zarr. Requires the config file's 'load_from' field to point at the distribution parameters JSON "
        "produced by 'distribution-parameters'."
    )
)
def export_ensemble_edr(config_file: ConfigFileArgument, diagnostics_from: DiagnosticsFromArgument) -> None:
    """
    Convert diagnostics configured in ``config_file`` into EDR and export the ensemble-mean result

    Args:
        config_file: Path to the YAML configuration file. Its ``load_from`` field must point at the distribution
            parameters JSON produced by :func:`distribution_parameters`.
        diagnostics_from: Where to obtain diagnostic values from
    """
    client = Client()

    context: TurbulenceContextWithAdditionalPath = TurbulenceContextWithAdditionalPath.from_yaml(config_file)
    compute_and_export_ensemble_edr(context, diagnostics_from)

    _ = client.close()


@turbulence_app.command(
    help=(
        "Compute the Matthew's Correlation Coefficient between every pair of configured diagnostics, thresholded "
        "at the given severities, and export the result to zarr. Requires the diagnostics to have already been "
        "exported (see 'export-diagnostic') and the config file's 'load_from' field to point at the thresholds "
        "JSON produced by 'turbulence-thresholds'."
    )
)
def diagnostic_correlation(
    config_file: ConfigFileArgument,
    severity: Annotated[
        list[TurbulenceSeverity],
        typer.Option(
            help=(
                "Turbulence severity/severities to threshold diagnostics at before computing their correlation. "
                "Repeat to pass multiple, e.g. --severity light --severity moderate."
            ),
        ),
    ] = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE],  # noqa: B006
    threshold_mode: Annotated[
        TurbulenceThresholdMode,
        typer.Option(
            help=(
                "How each severity's threshold is interpreted: 'bounded' treats it as an interval between two "
                "severities, 'geq' treats it as a lower bound (value >= threshold)."
            ),
        ),
    ] = TurbulenceThresholdMode.GEQ,
) -> None:
    """
    Compute correlation between diagnostics configured in ``config_file``, thresholded at ``severity``

    Args:
        config_file: Path to the YAML configuration file. Its ``load_from`` field must point at the thresholds
            JSON produced by :func:`turbulence_thresholds`.
        severity: Turbulence severities to threshold diagnostics at
        threshold_mode: Whether severity thresholds are bounded intervals or lower bounds
    """
    client = Client()

    context: TurbulenceContextWithOutput = TurbulenceContextWithAdditionalPath.from_yaml(config_file)
    correlation_between_diagnostics(context, severities=severity, threshold_mode=threshold_mode)

    _ = client.close()
