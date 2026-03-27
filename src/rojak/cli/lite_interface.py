import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from distributed import Client

from rojak.orchestrator.configuration import TurbulenceSeverity, TurbulenceThresholdMode
from rojak.orchestrator.lite_configuration import (
    DiagnosticThresholdsContext,
    TurbulenceContextWithAdditionalPath,
    TurbulenceContextWithOutput,
)
from rojak.orchestrator.lite_controller import (
    compute_distribution_parameters,
    compute_thresholds,
    correlation_between_diagnostics,
    export_turbulence_diagnostics,
)

# Root application for this interface
lite_app = typer.Typer(help="Lite run of rojak for lower memory usage")

# Turbulence Functionality
turbulence_app = typer.Typer(help="Computations related to turbulence")

# Add applications related to lite app here
lite_app.add_typer(turbulence_app, name="turbulence")


@turbulence_app.command()
def distribution_parameters(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    client = Client()
    context: TurbulenceContextWithOutput = TurbulenceContextWithOutput.from_yaml(config_file)
    compute_distribution_parameters(context)
    _ = client.close()


@turbulence_app.command()
def turbulence_thresholds(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    client = Client()
    context: DiagnosticThresholdsContext = DiagnosticThresholdsContext.from_yaml(config_file)
    compute_thresholds(context)
    _ = client.close()


@turbulence_app.command()
def export_diagnostic(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    client = Client()

    context: TurbulenceContextWithOutput = TurbulenceContextWithOutput.from_yaml(config_file)

    start_time: str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_to: Path = context.output_dir / context.name / start_time
    output_to.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy(config_file, output_to / config_file.name)

    export_turbulence_diagnostics(context, start_time=start_time)

    _ = client.close()


@turbulence_app.command()
def diagnostic_correlation(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Path to configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    severity: Annotated[
        list[TurbulenceSeverity],
        # typer.Option(default=[TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE]),
        typer.Option(),
    ] = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE],  # noqa: B006
    threshold_mode: Annotated[
        TurbulenceThresholdMode,
        # typer.Option(default=TurbulenceThresholdMode.GEQ)
        typer.Option(),
    ] = TurbulenceThresholdMode.GEQ,
) -> None:
    client = Client()

    context: TurbulenceContextWithOutput = TurbulenceContextWithAdditionalPath.from_yaml(config_file)
    correlation_between_diagnostics(context, severities=severity, threshold_mode=threshold_mode)

    _ = client.close()
