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


def test_context_from_yaml_basic(basic_context_yaml_file) -> None:
    context = configuration.Context.from_yaml(basic_context_yaml_file)

    assert isinstance(context, configuration.Context)
    assert context.name == "test"
    assert isinstance(context.data_config, DataConfig)
    assert context.turbulence_config is None
    assert context.contrails_config is None
