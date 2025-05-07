from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

import typer

from rojak.datalib.madis.amdar import MadisAmdarPreprocessor

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
            help="Select where data should be retrieved from",
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
            help="Select where data should be retrieved from",
        ),
    ],
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input_dir",
            help="Directory containing files to preprocess",
            exists=True,
            dir_okay=True,
            file_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output_dir",
            help="Directory to save preprocessed files. If unspecified, it will be the input directory",
        ),
    ] = None,
    glob_pattern: Annotated[
        Optional[str], typer.Option(help="Glob pattern to select files")
    ] = None,
):
    match source:
        case "madis-amdar":
            preprocess_madis_amdar_data(input_dir, output_dir, glob_pattern)
        case "ukmo-amdar":
            raise NotImplementedError("Not implemented UKMO AMDAR data preprocessing")


def preprocess_madis_amdar_data(
    input_dir: Path, output_dir: Path | None, glob_pattern: str | None
):
    preprocessor: MadisAmdarPreprocessor = MadisAmdarPreprocessor(input_dir, glob_pattern=glob_pattern)
    if output_dir is None:
        output_dir = input_dir
    preprocessor.filter_and_export_as_parquet(output_dir)


if __name__ == "__main__":
    data_app()
