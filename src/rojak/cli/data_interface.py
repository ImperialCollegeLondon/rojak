from enum import StrEnum
from typing import Annotated

import typer

data_app = typer.Typer()


class DataSource(StrEnum):
    MADIS = "madis-amdar"
    UKMO_AMDAR = "ukmo-amdar"


@data_app.command()
def retrieve(
    source: Annotated[
        DataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="Select where data should be retreived from",
        ),
    ],
): ...

@data_app.command()
def preprocess(
        source: Annotated[
            DataSource,
            typer.Option(
                "-s",
                "--source",
                case_sensitive=False,
                help="Select where data should be retreived from",
            ),
        ],
): ...


if __name__ == "__main__":
    data_app()
