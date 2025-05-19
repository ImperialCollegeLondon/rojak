import pytest
import yaml
from typing import TYPE_CHECKING

from rojak.orchestrator import configuration
from rojak.orchestrator.configuration import DataConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def basic_context_yaml_file(tmp_path_factory) -> "Path":
    tmp_output: Path = tmp_path_factory.mktemp("output")
    tmp_plots: Path = tmp_path_factory.mktemp("plots")
    content: dict = {
        "data_config": {"name": "test"},
        "name": "test",
        "image_format": "png",
        "output_dir": str(tmp_output),
        "plots_dir": str(tmp_plots),
    }
    output_file: "Path" = tmp_path_factory.mktemp("config") / "simplest_config.yml"
    with open(output_file, "w") as file:
        yaml.safe_dump(content, file, encoding="utf-8")

    return output_file


@pytest.fixture
def basic_context_yaml_file_create_output_plots_on_validation(
    tmp_path_factory,
) -> "Path":
    # Prevent creating the folders in the folder where tests are run
    tmp_folder: Path = tmp_path_factory.mktemp("empty_folder")
    content: dict = {
        "data_config": {"name": "test"},
        "name": "test",
        "image_format": "png",
        "output_dir": str(tmp_folder / "output"),
        "plots_dir": str(tmp_folder / "plots"),
    }
    output_file: "Path" = tmp_path_factory.mktemp("config") / "simplest_config.yml"
    with open(output_file, "w") as file:
        yaml.safe_dump(content, file, encoding="utf-8")

    return output_file


@pytest.mark.parametrize("is_created", [True, False])
def test_context_from_yaml_basic(
    basic_context_yaml_file,
    basic_context_yaml_file_create_output_plots_on_validation,
    is_created,
):
    if is_created:
        context = configuration.Context.from_yaml(basic_context_yaml_file)
    else:
        context = configuration.Context.from_yaml(
            basic_context_yaml_file_create_output_plots_on_validation
        )

    assert isinstance(context, configuration.Context)
    assert context.name == "test"
    assert isinstance(context.data_config, DataConfig)
    assert context.turbulence_config is None
    assert context.contrails_config is None
    assert context.plots_dir.exists() and context.plots_dir.is_dir()
    assert context.output_dir.exists() and context.output_dir.is_dir()
    assert context.image_format == "png"

    if is_created:
        assert context.plots_dir.name.startswith("plots")
        assert context.output_dir.name.startswith("output")
    else:
        assert context.plots_dir.name == "plots"
        assert context.output_dir.name == "output"
