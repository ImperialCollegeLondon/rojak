import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rojak.cli.main import app
from rojak.orchestrator.configuration import (
    Context as ConfigContext,
)
from rojak.orchestrator.configuration import (
    DataConfig,
    SpatialDomain,
    TurbulenceDiagnostics,
)
from tests.test_cli import runner

if TYPE_CHECKING:
    from click.testing import Result

    from rojak.orchestrator.configuration import ContrailsConfig, TurbulenceConfig


@pytest.fixture(scope="module")
def retrieve_era5_cat_data(tmp_path_factory) -> Path:
    data_dir = tmp_path_factory.mktemp("met_data")
    get_through_cli: Result = runner.invoke(
        app,
        [
            "data",
            "meteorology",
            "retrieve",
            "-s",
            "era5",
            "-y",
            "2024",
            "-m",
            "1",
            "-d",
            "1",
            "-d",
            "2",
            "-d",
            "3",
            "-n",
            "pressure-level",
            "--default-name",
            "cat-o",
            str(data_dir),
        ],
    )
    assert get_through_cli.exit_code == 0
    netcdf_files: set[Path] = set(data_dir.glob("*.nc"))
    assert len(netcdf_files) == 3  # noqa: PLR2004
    assert Path(data_dir / "2024-1-1.nc") in netcdf_files
    assert Path(data_dir / "2024-1-2.nc") in netcdf_files
    assert Path(data_dir / "2024-1-3.nc") in netcdf_files

    return data_dir


@pytest.fixture(scope="module")
def retrieve_single_day_madis_data(tmp_path_factory) -> Path:
    amdar_dir = tmp_path_factory.mktemp("amdar_data")
    get_through_cli = runner.invoke(
        app,
        ["data", "amdar", "retrieve", "-s", "madis", "-y", "2024", "-m", "1", "-d", "1", "-o", amdar_dir],
    )
    assert get_through_cli.exit_code == 0
    pre_process_madis = runner.invoke(
        app,
        [
            "data",
            "amdar",
            "preprocess",
            "-s",
            "madis",
            "-i",
            str(amdar_dir),
            "--glob-pattern",
            "**/*.gz",
        ],
    )
    assert pre_process_madis.exit_code == 0
    return amdar_dir


def randomly_select_diagnostics(num_diagnostics: int) -> list[TurbulenceDiagnostics]:
    return random.sample(list(TurbulenceDiagnostics), k=num_diagnostics)


@pytest.fixture
def create_config_context(tmp_path_factory) -> Callable:
    def _create_config_context(
        name: str,
        turb_config: "TurbulenceConfig | None" = None,
        data_config: DataConfig | None = None,
        contrails_config: "ContrailsConfig | None" = None,
    ) -> ConfigContext:
        plots_dir: Path = tmp_path_factory.mktemp("plots")
        output_dir: Path = tmp_path_factory.mktemp("output")
        if data_config is None:
            data_config = DataConfig(
                spatial_domain=SpatialDomain(
                    minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180
                )
            )
        return ConfigContext(
            name=name,
            image_format="png",
            output_dir=output_dir,
            plots_dir=plots_dir,
            turbulence_config=turb_config,
            data_config=data_config,
            contrails_config=contrails_config,
        )

    return _create_config_context
