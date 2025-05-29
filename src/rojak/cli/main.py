from pathlib import Path
from typing import Annotated

import typer

from rojak.cli import data_interface

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
) -> None: ...


@app.command()
def get_data() -> None:
    print("potatoes")
