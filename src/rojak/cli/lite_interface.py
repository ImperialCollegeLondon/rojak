from pathlib import Path
from typing import Annotated

import typer
from distributed import Client

from rojak.orchestrator.lite_configuration import DiagnosticThresholdsContext, TurbulenceContextWithOutput
from rojak.orchestrator.lite_controller import compute_distribution_parameters, compute_thresholds

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
