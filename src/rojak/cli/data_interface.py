from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, assert_never

import typer

from rojak.datalib.ecmwf.era5 import (
    Era5DatasetName,
    Era5DefaultsName,
    Era5Retriever,
    InvalidEra5RequestConfigurationError,
)
from rojak.datalib.madis.amdar import AcarsRetriever, MadisAmdarPreprocessor

if TYPE_CHECKING:
    from rojak.core.data import DataPreprocessor

data_app = typer.Typer(help="Perform operations on data")
amdar_app = typer.Typer(help="Operations for AMDAR data")
meteorology_app = typer.Typer(help="Operations for Meteorology data")
data_app.add_typer(amdar_app, name="amdar")
data_app.add_typer(meteorology_app, name="meteorology")


class AmdarDataSource(StrEnum):
    MADIS = "madis"
    UKMO = "ukmo"


def create_output_dir(output_dir: Path | None, source: StrEnum, intermediate_folder_name: str) -> Path:
    if output_dir is None:
        output_dir = Path.cwd() / intermediate_folder_name / source
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@amdar_app.command()
def retrieve(  # noqa: PLR0913
    source: Annotated[
        AmdarDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="Select where data should be retrieved from",
        ),
    ],
    years: Annotated[list[int], typer.Option("-y", "--years", help="Year(s) to retrieve data for")],
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
            help="Directory to save retrieved files. "
            "If unspecified, it will be in the 'data' folder in current directory",
        ),
    ] = None,
    glob_pattern: Annotated[
        Optional[str],
        typer.Option(
            "-g",
            "--glob-pattern",
            help="Glob pattern to select files ONLY APPLICABLE for MADIS",
        ),
    ] = None,
) -> None:
    output_dir = create_output_dir(output_dir, source, "data")

    match source:
        case AmdarDataSource.MADIS:
            retriever = AcarsRetriever(glob_pattern)
            retriever.download_files(years, months, days, output_dir)
        case AmdarDataSource.UKMO:
            raise NotImplementedError("Not implemented UKMO AMDAR data retrieval")
        case _ as unreachable:
            assert_never(unreachable)


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
    glob_pattern: Annotated[Optional[str], typer.Option(help="Glob pattern to select files")] = None,
) -> None:
    match source:
        case AmdarDataSource.MADIS:
            preprocess_madis_amdar_data(input_dir, output_dir, glob_pattern)
        case AmdarDataSource.UKMO:
            raise NotImplementedError("Not implemented UKMO AMDAR data preprocessing")
        case _ as unreachable:
            assert_never(unreachable)


def preprocess_madis_amdar_data(input_dir: Path, output_dir: Path | None, glob_pattern: str | None) -> None:
    preprocessor: "DataPreprocessor" = MadisAmdarPreprocessor(input_dir, glob_pattern=glob_pattern)
    if output_dir is None:
        output_dir = input_dir
    preprocessor.apply_preprocessor(output_dir)


class MeteorologyDataSource(StrEnum):
    ERA5 = "era5"


def validate_era5_default_name(default_name_input: str | None) -> Era5DefaultsName:
    if default_name_input is None:
        return None
    if default_name_input == "cat":
        return "cat"
    if default_name_input == "surface":
        return "surface"
    if default_name_input == "contrail":
        return "contrail"
    raise InvalidEra5RequestConfigurationError("Invalid default name")


def validate_era5_dataset_name(dataset_name_input: str) -> Era5DatasetName:
    if dataset_name_input == "pressure-level":
        return "pressure-level"
    if dataset_name_input == "single-level":
        return "single-level"
    raise InvalidEra5RequestConfigurationError("Invalid dataset name")


@meteorology_app.command("retrieve")
def retrieve_meteorology(  # noqa: PLR0913
    source: Annotated[
        MeteorologyDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="Select where data should be retrieved from",
        ),
    ],
    years: Annotated[list[int], typer.Option("-y", "--years", help="Year(s) to retrieve data for")],
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
    data_set_name: Annotated[str, typer.Option("-n", "--data-set-name", help="Name of data set to retrieve")],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output-dir",
            help="Directory to save retrieved files. If unspecified, it will be the input directory",
        ),
    ] = None,
    default_name: Annotated[
        Optional[str],
        typer.Option(
            "--default-name",
            help="Default request to use, options vary based on data source",
        ),
    ] = None,
    pressure_levels: Annotated[
        list[int] | None,
        typer.Option("-p", "--pressure-levels", help="Pressure levels to request data on"),
    ] = None,
    variables: Annotated[
        list[str] | None,
        typer.Option("-v", "--variables", help="Variables to retrieve in data request"),
    ] = None,
    times: Annotated[
        list[str] | None,
        typer.Option("-t", "--times", help="Times to retrieve in data request"),
    ] = None,
) -> None:
    output_dir = create_output_dir(output_dir, source, "met_data")

    match source:
        case MeteorologyDataSource.ERA5:
            retriever = Era5Retriever(
                validate_era5_dataset_name(data_set_name),
                output_dir.stem,
                default_name=validate_era5_default_name(default_name),
                pressure_levels=pressure_levels,
                variables=variables,
                times=times,
            )
            retriever.download_files(years, months, days, output_dir.parent)
        case _ as unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    data_app()
