import copy
from contextlib import nullcontext
from enum import StrEnum
from typing import TYPE_CHECKING

import pytest
import yaml

from rojak.orchestrator import configuration
from rojak.orchestrator.configuration import (
    DataConfig,
    InvalidConfigurationError,
    SpatialDomain,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def basic_context_yaml_file(tmp_path_factory) -> "Path":
    tmp_output: "Path" = tmp_path_factory.mktemp("output")
    tmp_plots: "Path" = tmp_path_factory.mktemp("plots")
    content: dict = {
        "data_config": {"name": "test", "spatial_domain": {}},
        "name": "test",
        "image_format": "png",
        "output_dir": str(tmp_output),
        "plots_dir": str(tmp_plots),
    }
    output_file: "Path" = tmp_path_factory.mktemp("config") / "simplest_config.yml"
    with output_file.open(mode="w") as file:
        yaml.safe_dump(content, file, encoding="utf-8")

    return output_file


@pytest.fixture
def basic_context_yaml_file_create_output_plots_on_validation(
    tmp_path_factory,
) -> "Path":
    # Prevent creating the folders in the folder where tests are run
    tmp_folder: "Path" = tmp_path_factory.mktemp("empty_folder")
    content: dict = {
        "data_config": {"name": "test", "spatial_domain": {}},
        "name": "test",
        "image_format": "png",
        "output_dir": str(tmp_folder / "output"),
        "plots_dir": str(tmp_folder / "plots"),
    }
    output_file: "Path" = tmp_path_factory.mktemp("config") / "simplest_config.yml"
    with output_file.open("w") as file:
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
        context = configuration.Context.from_yaml(basic_context_yaml_file_create_output_plots_on_validation)

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

    assert context.data_config.spatial_domain.minimum_latitude == -90
    assert context.data_config.spatial_domain.maximum_latitude == 90
    assert context.data_config.spatial_domain.minimum_longitude == -180
    assert context.data_config.spatial_domain.maximum_longitude == 180


def dump_dict_to_file(target_path: "Path", content: dict) -> "Path":
    output_file: "Path" = target_path / "config.yml"
    # Allows for StrEnums to be dumped to yaml
    # See: https://github.com/yaml/pyyaml/issues/722
    # noinspection PyTypeChecker
    yaml.SafeDumper.add_multi_representer(StrEnum, yaml.representer.SafeRepresenter.represent_str)
    with output_file.open("w") as file:
        yaml.safe_dump(content, file, encoding="utf-8")
    return output_file


@pytest.fixture
def dict_to_file(request, tmp_path) -> "Path":
    return dump_dict_to_file(tmp_path, request.param)


@pytest.mark.parametrize(
    "dict_to_file",
    [
        {
            "minimum_latitude": 90,
            "maximum_latitude": -90,
            "minimum_longitude": 0,
            "maximum_longitude": 180,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 90,
            "maximum_longitude": 0,
        },
        {
            "minimum_latitude": -91,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 90,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 91,
            "minimum_longitude": 0,
            "maximum_longitude": 90,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": -181,
            "maximum_longitude": 0,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 181,
        },
        {
            "minimum_latitude": 90,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 180,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 0,
        },
    ],
    indirect=True,
)
def test_spatial_domain_invalid_config(dict_to_file) -> None:
    with pytest.raises(InvalidConfigurationError) as excinfo:
        configuration.SpatialDomain.from_yaml(dict_to_file)
    assert excinfo.type is InvalidConfigurationError


@pytest.mark.parametrize(
    "dict_to_file",
    [
        {
            "minimum_latitude": -90,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 180,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 90,
        },
        {
            "minimum_latitude": -90,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 90,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 90,
        },
        {
            "minimum_latitude": 0,
            "maximum_latitude": 90,
            "minimum_longitude": 0,
            "maximum_longitude": 180,
        },
    ],
    indirect=True,
)
def test_spatial_domain_valid_config(dict_to_file) -> None:
    spatial_domain = SpatialDomain.from_yaml(dict_to_file)
    assert spatial_domain.minimum_longitude == 0


@pytest.fixture()
def make_empty_temp_dir(tmp_path_factory) -> "Path":
    return tmp_path_factory.mktemp("temp")


@pytest.fixture()
def make_empty_temp_text_file(tmp_path_factory) -> "Path":
    output_file = tmp_path_factory.getbasetemp() / "file.txt"
    output_file.touch()
    return output_file


@pytest.mark.parametrize("dict_to_file", [{}], indirect=True)
def test_turbulence_config_invalid_config_default(dict_to_file) -> None:
    with pytest.raises(InvalidConfigurationError) as excinfo:
        configuration.TurbulenceConfig.from_yaml(dict_to_file)
    assert excinfo.type is InvalidConfigurationError


@pytest.fixture
def make_turbulence_config_with_calibration_dir(make_empty_temp_dir, tmp_path_factory, request) -> "Path":
    content = copy.deepcopy(request.param)
    content["calibration_data_dir"] = str(make_empty_temp_dir)
    content["evaluation_data_dir"] = str(tmp_path_factory.mktemp("data"))
    return dump_dict_to_file(tmp_path_factory.getbasetemp(), content)


turbulence_config_field_permutations = [
    {"chunks": {}, "diagnostics": [configuration.TurbulenceDiagnostics.DEF]},
    {
        "chunks": {"something": 1},
        "diagnostics": [
            configuration.TurbulenceDiagnostics.DEF,
            configuration.TurbulenceDiagnostics.BROWN1,
        ],
        "threshold_mode": configuration.TurbulenceThresholdMode.GEQ,
    },
    {
        "chunks": {"something": 1},
        "diagnostics": [
            configuration.TurbulenceDiagnostics.DEF,
            configuration.TurbulenceDiagnostics.BROWN1,
        ],
        "threshold_mode": configuration.TurbulenceThresholdMode.GEQ,
        "severities": [
            configuration.TurbulenceSeverity.LIGHT,
            configuration.TurbulenceSeverity.SEVERE,
        ],
    },
]
turbulence_config_parametrisation = (
    [
        pytest.param({}, pytest.raises(InvalidConfigurationError), id="only_has_dirs"),
        pytest.param({"chunks": {}}, pytest.raises(InvalidConfigurationError), id="only_has_chunks"),
        pytest.param(
            turbulence_config_field_permutations[0],
            nullcontext(turbulence_config_field_permutations[0]),
            id="succeeds_only_required",
        ),
        pytest.param(
            turbulence_config_field_permutations[1],
            nullcontext(turbulence_config_field_permutations[1]),
            id="succeeds_with_threshold_mode",
        ),
        pytest.param(
            turbulence_config_field_permutations[2],
            nullcontext(turbulence_config_field_permutations[2]),
            id="succeeds_with_multiple_severities",
        ),
    ],
)


@pytest.mark.parametrize(
    "make_turbulence_config_with_calibration_dir, expectation",
    *turbulence_config_parametrisation,
    indirect=["make_turbulence_config_with_calibration_dir"],
)
def test_turbulence_config_with_calibration_dir(make_turbulence_config_with_calibration_dir, expectation) -> None:
    with expectation as e:
        config = configuration.TurbulenceConfig.from_yaml(make_turbulence_config_with_calibration_dir)
        # assert isinstance(config, configuration.TurbulenceConfig)
        assert config.chunks == e["chunks"]
        assert config.diagnostics == e["diagnostics"]
        assert config.thresholds_file_path is None
        if "threshold_mode" in e:
            assert config.threshold_mode == e["threshold_mode"]
        else:
            assert config.threshold_mode == configuration.TurbulenceThresholdMode.BOUNDED
        if "severities" in e:
            assert config.severities == e["severities"]
        else:
            assert config.severities == [configuration.TurbulenceSeverity.LIGHT]

    if not isinstance(e, dict):
        assert e.type is InvalidConfigurationError


@pytest.fixture
def make_turbulence_config_with_threshold_file(make_empty_temp_text_file, tmp_path_factory, request) -> "Path":
    content = copy.deepcopy(request.param)
    content["thresholds_file_path"] = str(make_empty_temp_text_file)
    content["evaluation_data_dir"] = str(tmp_path_factory.mktemp("data"))
    return dump_dict_to_file(tmp_path_factory.getbasetemp(), content)


@pytest.mark.parametrize(
    "make_turbulence_config_with_threshold_file, expectation",
    *turbulence_config_parametrisation,
    indirect=["make_turbulence_config_with_threshold_file"],
)
def test_turbulence_config_with_threshold_file(make_turbulence_config_with_threshold_file, expectation) -> None:
    with expectation as e:
        config = configuration.TurbulenceConfig.from_yaml(make_turbulence_config_with_threshold_file)
        assert config.chunks == e["chunks"]
        assert config.diagnostics == e["diagnostics"]
        if "threshold_mode" in e:
            assert config.threshold_mode == e["threshold_mode"]
        else:
            assert config.threshold_mode == configuration.TurbulenceThresholdMode.BOUNDED
        if "severities" in e:
            assert config.severities == e["severities"]
        else:
            assert config.severities == [configuration.TurbulenceSeverity.LIGHT]
        assert config.calibration_data_dir is None

    if not isinstance(e, dict):
        assert e.type is InvalidConfigurationError


@pytest.mark.parametrize(
    "dict_to_file, expectation",
    [
        pytest.param({}, pytest.raises(InvalidConfigurationError)),
        pytest.param({"contrail_model": "invalid_option"}, pytest.raises(InvalidConfigurationError)),
        pytest.param({"contrail_model": "issr"}, nullcontext("issr")),
        pytest.param({"contrail_model": "sac"}, nullcontext("sac")),
        pytest.param({"contrail_model": "pcr"}, nullcontext("pcr")),
    ],
    indirect=["dict_to_file"],
)
def test_contrails_config(dict_to_file, expectation) -> None:
    with expectation as e:
        config = configuration.ContrailsConfig.from_yaml(dict_to_file)
        assert config.contrail_model == e
    if not isinstance(e, str):
        assert e.type is InvalidConfigurationError


@pytest.mark.parametrize(
    "dict_to_file, expectation",
    [
        pytest.param({}, pytest.raises(InvalidConfigurationError), id="empty_config"),
        pytest.param(
            {"evaluation_data_dir": "random/nonsense/dir"},
            pytest.raises(InvalidConfigurationError),
            id="dir_does_not_exist",
        ),
    ],
    indirect=["dict_to_file"],
)
def test_meteorology_config(dict_to_file, expectation) -> None:
    with expectation as e:
        configuration.MeteorologyConfig.from_yaml(dict_to_file)
    assert e.type is InvalidConfigurationError
