import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from rojak.cli import data_interface
from rojak.orchestrator.configuration import Context as ConfigContext
from rojak.orchestrator.turbulence import TurbulenceLauncher

app = typer.Typer()
app.add_typer(data_interface.data_app, name="data")


@app.command()
def turbulence() -> None:
    print("HELLO from the other side")


@app.command()
def run(
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
    logging.basicConfig(level="NOTSET", handlers=[RichHandler(rich_tracebacks=True)])
    context = ConfigContext.from_yaml(config_file)
    if context.turbulence_config is not None:
        TurbulenceLauncher(context).launch()


@app.command()
def get_data() -> None:
    print("potatoes")
