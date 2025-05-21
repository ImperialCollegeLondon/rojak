from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional, TYPE_CHECKING

import typer

from rojak.datalib.madis.amdar import MadisAmdarPreprocessor, AcarsRetriever

if TYPE_CHECKING:
    from rojak.core.data import DataPreprocessor

data_app = typer.Typer(help="Perform operations on data")
amdar_app = typer.Typer(help="Operations for AMDAR data")
meteorology_app = typer.Typer(help="Operations for Meteorology data")
data_app.add_typer(amdar_app, name="amdar")
data_app.add_typer(meteorology_app, name="meteorology")


class AmdarDataSource(StrEnum):
    MADIS = "madis"
    UKMO_AMDAR = "ukmo"


def create_output_dir(
    output_dir: Path | None, source: StrEnum, intermediate_folder_name: str
) -> Path:
    if output_dir is None:
        output_dir = Path.cwd() / intermediate_folder_name / source
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@amdar_app.command()
def retrieve(
    source: Annotated[
        AmdarDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="Select where data should be retrieved from",
        ),
    ],
    years: Annotated[
        list[int], typer.Option("-y", "--years", help="Year(s) to retrieve data for")
    ],
    months: Annotated[
        list[int],
        typer.Option(
            "-m",
            "--months",
            help="Months(s) to retrieve data for. Use -1 to specify all months in year",
        ),
    ],
    days: Annotated[
        list[int],
        typer.Option(
            "-d",
            "--days",
            help="Day(s) to retrieve data for. Use -1 to specify all days in month",
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output_dir",
            help="Directory to save retrieved files. If unspecified, it will be in the 'data' folder in current directory",
        ),
    ] = None,
    glob_pattern: Annotated[
        Optional[str],
        typer.Option(help="Glob pattern to select files ONLY APPLICABLE for MADIS"),
    ] = None,
):
    output_dir = create_output_dir(output_dir, source, "data")

    match source:
        case "madis":
            retriever = AcarsRetriever(glob_pattern)
            retriever.download_files(years, months, days, output_dir)
        case "ukmo":
            raise NotImplementedError("Not implemented UKMO AMDAR data retrieval")


@amdar_app.command()
def preprocess(
    source: Annotated[
        AmdarDataSource,
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
        case "madis":
            preprocess_madis_amdar_data(input_dir, output_dir, glob_pattern)
        case "ukmo":
            raise NotImplementedError("Not implemented UKMO AMDAR data preprocessing")


def preprocess_madis_amdar_data(
    input_dir: Path, output_dir: Path | None, glob_pattern: str | None
):
    preprocessor: "DataPreprocessor" = MadisAmdarPreprocessor(
        input_dir, glob_pattern=glob_pattern
    )
    if output_dir is None:
        output_dir = input_dir
    preprocessor.apply_preprocessor(output_dir)


if __name__ == "__main__":
    data_app()
