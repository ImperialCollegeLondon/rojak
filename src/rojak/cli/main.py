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
Entry point for the ``rojak`` command line interface

Running ``rojak`` (see :data:`app`) exposes the top-level :func:`run` command, which launches a full analysis
(currently, computing turbulence diagnostics) from a single YAML configuration file, plus two command groups for
more specific workflows:

- ``rojak data ...`` (:data:`~rojak.cli.data_interface.data_app`): retrieving and preprocessing the input data
  (AMDAR observations and meteorological reanalysis) that an analysis needs.
- ``rojak lite ...`` (:data:`~rojak.cli.lite_interface.lite_app`): lower-memory, step-by-step turbulence workflows
  for large datasets that can't comfortably be run as a single :func:`run` invocation.
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dask.distributed import Client
from rich.logging import RichHandler

from rojak.cli import data_interface, lite_interface
from rojak.orchestrator.configuration import Context as ConfigContext
from rojak.orchestrator.turbulence import TurbulenceLauncher

app = typer.Typer(
    pretty_exceptions_show_locals=True,
    rich_markup_mode="markdown",
    no_args_is_help=True,
    help=(
        "rojak: compute clear-air turbulence (CAT) diagnostics from meteorological data, retrieve and preprocess "
        "AMDAR turbulence observations, and evaluate diagnostics against those observations.\n\n"
        "Run `rojak run CONFIG_FILE` for a full analysis from a single YAML configuration file, or see the `data` "
        "and `lite` command groups below for more specific workflows."
    ),
)
app.add_typer(data_interface.data_app, name="data")
app.add_typer(lite_interface.lite_app, name="lite")


class LogLevel(StrEnum):
    """Logging verbosity accepted by the ``--log`` option of :func:`run`"""

    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@app.command(
    help=(
        "Run a full rojak analysis as described by a single YAML configuration file."
        "This starts a local Dask cluster for the duration of the run"
    )
)
def run(
    config_file: Annotated[
        Path,
        typer.Argument(
            help=(
                "Path to the YAML configuration file describing what to run (input data, spatial domain, "
                "diagnostics to compute, and where to write output)."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    log_level: Annotated[
        LogLevel | None,
        typer.Option(
            "--log",
            case_sensitive=False,
            help="Logging verbosity to print to the console while running. If omitted, logging is left off.",
        ),
    ] = None,
) -> None:
    """
    Run a full analysis (currently: turbulence diagnostics) from ``config_file``

    Starts a local Dask :class:`~dask.distributed.Client`, loads ``config_file`` into a
    :class:`~rojak.orchestrator.configuration.Context`, and launches
    :class:`~rojak.orchestrator.turbulence.TurbulenceLauncher` if turbulence analysis is configured.

    Args:
        config_file: Path to the YAML configuration file
        log_level: Logging verbosity. If ``None`` (default), logging is left unconfigured.
    """
    if log_level is not None:
        logging.basicConfig(level=log_level.upper(), handlers=[RichHandler(rich_tracebacks=True)])

    client = Client()
    context = ConfigContext.from_yaml(config_file)

    if context.turbulence_config is not None:
        TurbulenceLauncher(context).launch()

    client.close()
