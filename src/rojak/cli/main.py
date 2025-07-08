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

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dask.distributed import Client
from rich.logging import RichHandler

from rojak.cli import data_interface
from rojak.orchestrator.configuration import Context as ConfigContext
from rojak.orchestrator.turbulence import DiagnosticsAmdarLauncher, TurbulenceLauncher

app = typer.Typer()
app.add_typer(data_interface.data_app, name="data")


class LogLevel(StrEnum):
    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@app.command(help="Performs analysis on data according to configuration file")
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
    log_level: Annotated[
        LogLevel | None,
        typer.Option("--log", case_sensitive=False, help="Logging level"),
    ] = None,
) -> None:
    if log_level is not None:
        logging.basicConfig(level=log_level.upper(), handlers=[RichHandler(rich_tracebacks=True)])

    client = Client()
    context = ConfigContext.from_yaml(config_file)

    turbulence_result = None
    if context.turbulence_config is not None:
        turbulence_result = TurbulenceLauncher(context).launch()

    if turbulence_result is None and context.data_config.amdar_config is not None:
        raise ValueError("Failed to launch diagnostic amdar harmonisation as evaluation phase was not run")
    if turbulence_result is not None and context.data_config.amdar_config is not None:
        DiagnosticsAmdarLauncher(context.data_config, context.output_dir, context.name).launch(turbulence_result.suite)
    client.close()
