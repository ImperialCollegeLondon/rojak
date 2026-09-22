#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
``rojak data`` command group: retrieving and preprocessing the data rojak analyses consume

This exposes three subcommand groups:

- ``rojak data amdar ...`` (:data:`amdar_app`): download (:func:`retrieve`) and preprocess (:func:`preprocess`)
  AMDAR turbulence observations.
- ``rojak data meteorology ...`` (:data:`meteorology_app`): download meteorological reanalysis data
  (:func:`retrieve_meteorology`).
- ``rojak data utils ...`` (:data:`utils_app`): miscellaneous data file utilities (:func:`repartition_parquet`).
"""

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, assert_never

import dask.dataframe as dd
import typer
from rich.progress import track

from rojak.datalib.ecmwf.era5 import (
    Era5DatasetName,
    Era5DefaultsName,
    Era5Retriever,
)
from rojak.datalib.madis.amdar import AcarsRetriever, MadisAmdarPreprocessor
from rojak.orchestrator.configuration import AmdarDataSource

if TYPE_CHECKING:
    from rojak.core.data import DataPreprocessor

data_app = typer.Typer(
    help="Retrieve, preprocess, and manage the input data used by rojak analyses",
    no_args_is_help=True,
)
amdar_app = typer.Typer(
    help=(
        "Download and preprocess AMDAR (Aircraft Meteorological Data Relay) turbulence observations. "
        "Currently supports MADIS"
    ),
    no_args_is_help=True,
)
meteorology_app = typer.Typer(
    help="Download meteorological reanalysis data (currently: ECMWF ERA5) used to compute turbulence diagnostics",
    no_args_is_help=True,
)
utils_app = typer.Typer(help="Miscellaneous utilities for working with data files", no_args_is_help=True)
data_app.add_typer(amdar_app, name="amdar")
data_app.add_typer(meteorology_app, name="meteorology")
data_app.add_typer(utils_app, name="utils")


def create_output_dir(output_dir: Path | None, source: StrEnum, intermediate_folder_name: str) -> Path:
    """
    Resolve (and create) the directory to save downloaded/processed files into

    If ``output_dir`` is ``None``, defaults to ``./{intermediate_folder_name}/{source}``.

    Args:
        output_dir: Directory to use, if explicitly given by the user
        source: Data source name, used to build a default output directory if ``output_dir`` is not given
        intermediate_folder_name: Parent folder name (relative to the current directory) to use in the default
            output directory, e.g. ``"data"`` or ``"met_data"``

    Returns:
        ``output_dir``, created (along with any missing parent directories) if it did not already exist
    """
    if output_dir is None:
        output_dir = Path.cwd() / intermediate_folder_name / source
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@amdar_app.command(help=("Download raw AMDAR turbulence observation files for the given date(s)."))
def retrieve(
    source: Annotated[
        AmdarDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="AMDAR data provider to download from. UKMO retrieval is not yet implemented.",
        ),
    ],
    years: Annotated[
        list[int],
        typer.Option(
            "-y",
            "--years",
            help="Year(s) to retrieve data for. Repeat to pass multiple, e.g. `-y 2020 -y 2021`.",
        ),
    ],
    months: Annotated[
        list[int],
        typer.Option(
            "-m",
            "--months",
            help=(
                "Month(s) to retrieve data for, as integers 1-12. Repeat to pass multiple, e.g. `-m 1 -m 2`. "
                "Use `-m -1` to retrieve all months in each year."
            ),
        ),
    ],
    days: Annotated[
        list[int],
        typer.Option(
            "-d",
            "--days",
            help=(
                "Day(s) of the month to retrieve data for. Repeat to pass multiple, e.g. `-d 1 -d 15`. "
                "Use `-d -1` to retrieve every day of each month."
            ),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output_dir",
            help="Directory to save the downloaded files into. Defaults to `./data/<source>` if not given.",
        ),
    ] = None,
    glob_pattern: Annotated[
        str | None,
        typer.Option(
            "-g",
            "--glob-pattern",
            help=(
                "Glob pattern used to select which files to download for each date, e.g. `'*.gz'`. "
                "Only used for the MADIS source; ignored otherwise."
            ),
        ),
    ] = None,
) -> None:
    """
    Download AMDAR turbulence observations for the given years/months/days

    Args:
        source: AMDAR data provider to download from
        years: Years to download data for
        months: Months to download data for. ``[-1]`` means every month.
        days: Days to download data for. ``[-1]`` means every day of the month.
        output_dir: Directory to save the downloaded files into
        glob_pattern: Glob pattern selecting which files to download each date (MADIS only)

    Raises:
        NotImplementedError: If ``source`` is :attr:`~rojak.orchestrator.configuration.AmdarDataSource.UKMO`
    """
    output_dir = create_output_dir(output_dir, source, "data")

    match source:
        case AmdarDataSource.MADIS:
            retriever = AcarsRetriever(glob_pattern)
            retriever.download_files(years, months, days, output_dir)
        case AmdarDataSource.UKMO:
            raise NotImplementedError("Not implemented UKMO AMDAR data retrieval")
        case _ as unreachable:
            assert_never(unreachable)


@amdar_app.command(
    help=(
        "Filter and convert raw downloaded AMDAR files (see `retrieve`) into parquet files containing only the "
        "variables and quality-controlled observations needed for turbulence analysis."
    )
)
def preprocess(
    source: Annotated[
        AmdarDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="AMDAR data provider the raw files came from. UKMO preprocessing is not yet implemented.",
        ),
    ],
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input_dir",
            help=(
                "Directory containing the raw files to preprocess, or a path to a single raw file. If this is a "
                "directory, `--glob-pattern` must also be given to select which files within it to preprocess."
            ),
            exists=True,
            dir_okay=True,
            file_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output_dir",
            help="Directory to save the preprocessed parquet files into. Defaults to `--input_dir` if not given.",
        ),
    ] = None,
    glob_pattern: Annotated[
        str | None,
        typer.Option(
            "-g",
            "--glob-pattern",
            help=(
                "Glob pattern (relative to `--input_dir`) selecting which raw files to preprocess, e.g. `'*.gz'`. "
                "Required when `--input_dir` is a directory; not used when it is a single file."
            ),
        ),
    ] = None,
) -> None:
    """
    Preprocess raw AMDAR files from ``source`` into filtered parquet files

    Args:
        source: AMDAR data provider the raw files came from
        input_dir: Directory (or single file) of raw files to preprocess
        output_dir: Directory to save the preprocessed parquet files into
        glob_pattern: Glob pattern selecting which files within ``input_dir`` to preprocess

    Raises:
        NotImplementedError: If ``source`` is :attr:`~rojak.orchestrator.configuration.AmdarDataSource.UKMO`
    """
    match source:
        case AmdarDataSource.MADIS:
            preprocess_madis_amdar_data(input_dir, output_dir, glob_pattern)
        case AmdarDataSource.UKMO:
            raise NotImplementedError("Not implemented UKMO AMDAR data preprocessing")
        case _ as unreachable:
            assert_never(unreachable)


def preprocess_madis_amdar_data(input_dir: Path, output_dir: Path | None, glob_pattern: str | None) -> None:
    """
    Preprocess MADIS AMDAR files via :class:`~rojak.datalib.madis.amdar.MadisAmdarPreprocessor`

    Args:
        input_dir: Directory (or single file) of raw MADIS files to preprocess
        output_dir: Directory to save the preprocessed parquet files into. Defaults to ``input_dir`` if ``None``.
        glob_pattern: Glob pattern selecting which files within ``input_dir`` to preprocess
    """
    preprocessor: DataPreprocessor = MadisAmdarPreprocessor(input_dir, glob_pattern=glob_pattern)
    if output_dir is None:
        output_dir = input_dir
    preprocessor.apply_preprocessor(output_dir)


class MeteorologyDataSource(StrEnum):
    """Meteorological data provider accepted by the ``--source`` option of :func:`retrieve_meteorology`"""

    ERA5 = "era5"


@meteorology_app.command(
    "retrieve",
    help=(
        "Download meteorological reanalysis data for the given date(s). Currently only the ERA5 source is "
        "implemented; this requires a Copernicus Climate Data Store (CDS) API key configured, see "
        "[https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)."
    ),
)
def retrieve_meteorology(
    source: Annotated[
        MeteorologyDataSource,
        typer.Option(
            "-s",
            "--source",
            case_sensitive=False,
            help="Meteorological data provider to download from",
        ),
    ],
    years: Annotated[
        list[int],
        typer.Option(
            "-y",
            "--years",
            help="Year(s) to retrieve data for. Repeat to pass multiple, e.g. `-y 2020 -y 2021`.",
        ),
    ],
    months: Annotated[
        list[int],
        typer.Option(
            "-m",
            "--months",
            help=(
                "Month(s) to retrieve data for, as integers 1-12. Repeat to pass multiple, e.g. `-m 1 -m 2`. "
                "Use `-m -1` to retrieve all months in each year."
            ),
        ),
    ],
    days: Annotated[
        list[int],
        typer.Option(
            "-d",
            "--days",
            help=(
                "Day(s) of the month to retrieve data for. Repeat to pass multiple, e.g. `-d 1 -d 15`. "
                "Use `-d -1` to retrieve every day of each month."
            ),
        ),
    ],
    data_set_name: Annotated[
        Era5DatasetName,
        typer.Option(
            "-n",
            "--data-set-name",
            help="ERA5 dataset to request from",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output-dir",
            help="Directory to save the downloaded files into. Defaults to `./met_data/<source>` if not given.",
        ),
    ] = None,
    default_name: Annotated[
        Era5DefaultsName | None,
        typer.Option(
            "--default-name",
            help=(
                "Name of a built-in CDS request template controlling which variables/pressure levels are "
                "requested by default."
            ),
        ),
    ] = None,
    pressure_levels: Annotated[
        list[int] | None,
        typer.Option(
            "-p",
            "--pressure-levels",
            help=(
                "Pressure levels (hPa) to request, e.g. `-p 200 -p 250 -p 300`. Only used for the 'pressure-level' "
                "dataset; overrides the levels from `--default-name` if both are given. Required if `--default-name` "
                "is not given and `--data-set-name pressure-level`."
            ),
        ),
    ] = None,
    variables: Annotated[
        list[str] | None,
        typer.Option(
            "-v",
            "--variables",
            help=(
                "CDS variable name(s) to request, e.g. `-v temperature -v u_component_of_wind`. Overrides the "
                "variables from `--default-name` if both are given. Required if `--default-name` is not given. See "
                "the [CDS ERA5 documentation for valid variable names](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation#heading-Parameterlistings)."
            ),
        ),
    ] = None,
    times: Annotated[
        list[str] | None,
        typer.Option(
            "-t",
            "--times",
            help=(
                "Time(s) of day (UTC, 'HH:MM') to request, e.g. `-t 00:00 -t 12:00`. Defaults to every 6 hours "
                "(00:00, 06:00, 12:00, 18:00) if not given."
            ),
        ),
    ] = None,
) -> None:
    """
    Download meteorological reanalysis data for the given years/months/days

    Args:
        source: Meteorological data provider to download from
        years: Years to download data for
        months: Months to download data for. ``[-1]`` means every month.
        days: Days to download data for. ``[-1]`` means every day of the month.
        data_set_name: ERA5 dataset to request from
        output_dir: Directory to save the downloaded files into
        default_name: Name of a built-in CDS request template
        pressure_levels: Pressure levels (hPa) to request, for the pressure-level dataset
        variables: CDS variable names to request
        times: Times of day (UTC) to request

    Raises:
        InvalidEra5RequestConfigurationError: If ``data_set_name``/``default_name`` is invalid, or a value
            required in the absence of ``default_name`` was not provided
    """
    output_dir = create_output_dir(output_dir, source, "met_data")

    match source:
        case MeteorologyDataSource.ERA5:
            retriever = Era5Retriever(
                data_set_name,
                output_dir.stem,
                default_name=default_name,
                pressure_levels=pressure_levels,
                variables=variables,
                times=times,
            )
            retriever.download_files(years, months, days, output_dir.parent)
        case _ as unreachable:
            assert_never(unreachable)


@utils_app.command(
    "repartition",
    help=(
        "Repartition parquet dataset(s) on disk into a different number of partitions. Useful after preprocessing "
        "produces many small partitions (e.g. one per input file), which can be slow to work with."
    ),
)
def repartition_parquet(
    root_dir: Annotated[
        Path,
        typer.Option(
            "-d",
            "--root-dir",
            help=(
                "Directory containing the parquet dataset to repartition. If `-r` is set, this is instead the "
                "parent directory containing one subdirectory per dataset to repartition."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dir",
            help=(
                "Directory to write the repartitioned dataset(s) into. Created if it does not already exist; one "
                "subdirectory per dataset is created within it (matching each input directory's name)."
            ),
            file_okay=False,
            dir_okay=True,
            writable=True,
        ),
    ],
    glob_pattern: Annotated[
        str,
        typer.Option(
            "-p",
            "--glob-pattern",
            help=(
                "Glob pattern (relative to each dataset's directory) matching the parquet files to read, e.g. "
                "`'*.parquet'`."
            ),
        ),
    ],
    num_partitions: Annotated[
        int,
        typer.Option("-n", "--num-partitions", help="Number of partitions to repartition each dataset into", min=0),
    ],
    is_nested: Annotated[
        bool,
        typer.Option(
            "-r",
            help=(
                "If passed, treat every immediate subdirectory of `--root-dir` as a separate dataset to repartition "
                "(non-recursively beyond that level). If omitted, `--root-dir` itself is treated as the single "
                "dataset to repartition."
            ),
        ),
    ],
) -> None:
    """
    Repartition parquet dataset(s) under ``root_dir`` into ``output_dir``, with ``num_partitions`` partitions each

    Args:
        root_dir: Directory of the dataset to repartition, or (if ``is_nested``) its parent directory
        output_dir: Directory to write the repartitioned dataset(s) into
        glob_pattern: Glob pattern selecting which parquet files to read from each dataset's directory
        num_partitions: Number of partitions to repartition each dataset into
        is_nested: If ``True``, repartition every immediate subdirectory of ``root_dir`` separately

    Raises:
        AssertionError: If a dataset's directory contains no files matching ``glob_pattern``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    directories_to_process: list[Path] = (
        [item for item in root_dir.iterdir() if item.is_dir()] if is_nested else [root_dir]
    )

    for directory in track(directories_to_process, "Processing directories..."):
        assert list(directory.glob(glob_pattern)), "Directory does not contains matching glob pattern"
        output_location = output_dir / directory.stem
        output_location.mkdir(parents=True, exist_ok=True)
        dd.read_parquet(str(directory / glob_pattern)).repartition(npartitions=num_partitions).to_parquet(
            str(output_location),
        )


if __name__ == "__main__":
    data_app()
