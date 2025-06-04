import copy
from contextlib import nullcontext
from enum import StrEnum
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

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
    assert context.plots_dir.exists()
    assert context.plots_dir.is_dir()
    assert context.output_dir.exists()
    assert context.output_dir.is_dir()
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
            "maximum_longitude": 180,
        },
    ],
    indirect=True,
)
def test_spatial_domain_valid_config(dict_to_file) -> None:
    spatial_domain = SpatialDomain.from_yaml(dict_to_file)
    assert spatial_domain.minimum_longitude == 0


@pytest.fixture
def make_empty_temp_dir(tmp_path_factory) -> "Path":
    return tmp_path_factory.mktemp("temp")


@pytest.fixture
def make_empty_temp_text_file(tmp_path_factory) -> "Path":
    output_file = tmp_path_factory.getbasetemp() / "file.txt"
    output_file.touch()
    return output_file


@pytest.mark.parametrize("dict_to_file", [{}], indirect=True)
def test_turbulence_config_invalid_config_default(dict_to_file) -> None:
    with pytest.raises(InvalidConfigurationError) as excinfo:
        configuration.TurbulenceConfig.from_yaml(dict_to_file)
    assert excinfo.type is InvalidConfigurationError


def dummy_threshold_config():
    return [{"name": "test", "lower_bound": 97, "upper_bound": 100}]


@pytest.fixture
def make_turbulence_config_with_calibration_dir(make_empty_temp_dir, tmp_path_factory, request) -> "Path":
    content = copy.deepcopy(request.param)
    content["phases"]["calibration_phases"]["calibration_config"]["calibration_data_dir"] = str(make_empty_temp_dir)
    content["phases"]["calibration_phases"]["calibration_config"]["threshold_config"] = dummy_threshold_config()
    content["phases"]["evaluation_phases"]["evaluation_config"]["evaluation_data_dir"] = str(
        tmp_path_factory.mktemp("data")
    )
    return dump_dict_to_file(tmp_path_factory.getbasetemp(), content)


turbulence_config_field_permutations = [
    {
        "chunks": {},
        "diagnostics": [configuration.TurbulenceDiagnostics.DEF],
        "phases": {
            "calibration_phases": {"calibration_config": {}, "phases": []},
            "evaluation_phases": {"evaluation_config": {}, "phases": []},
        },
    },
    {
        "chunks": {"something": 1},
        "diagnostics": [
            configuration.TurbulenceDiagnostics.DEF,
            configuration.TurbulenceDiagnostics.BROWN1,
        ],
        "phases": {
            "calibration_phases": {"calibration_config": {}, "phases": []},
            "evaluation_phases": {
                "evaluation_config": {
                    "threshold_mode": configuration.TurbulenceThresholdMode.BOUNDED,
                },
                "phases": [],
            },
        },
    },
    {
        "chunks": {"something": 1},
        "diagnostics": [
            configuration.TurbulenceDiagnostics.DEF,
            configuration.TurbulenceDiagnostics.BROWN1,
        ],
        "phases": {
            "calibration_phases": {"calibration_config": {}, "phases": []},
            "evaluation_phases": {
                "evaluation_config": {
                    "threshold_mode": configuration.TurbulenceThresholdMode.BOUNDED,
                    "severities": [
                        configuration.TurbulenceSeverity.LIGHT,
                        configuration.TurbulenceSeverity.SEVERE,
                    ],
                },
                "phases": [],
            },
        },
    },
]
turbulence_config_parametrisation = (
    [
        pytest.param(
            {
                "phases": {
                    "calibration_phases": {"calibration_config": {}, "phases": []},
                    "evaluation_phases": {"evaluation_config": {}, "phases": []},
                },
            },
            pytest.raises(InvalidConfigurationError),
            id="only_has_dirs",
        ),
        pytest.param(
            {
                "chunks": {},
                "phases": {
                    "calibration_phases": {"calibration_config": {}, "phases": []},
                    "evaluation_phases": {"evaluation_config": {}, "phases": []},
                },
            },
            pytest.raises(InvalidConfigurationError),
            id="only_has_chunks",
        ),
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
        assert config.phases.calibration_phases.calibration_config.thresholds_file_path is None
        if config.phases.evaluation_phases is not None:
            if "threshold_mode" in e["phases"]["evaluation_phases"]["evaluation_config"]:
                assert (
                    config.phases.evaluation_phases.evaluation_config.threshold_mode
                    == e["phases"]["evaluation_phases"]["evaluation_config"]["threshold_mode"]
                )
            else:
                assert (
                    config.phases.evaluation_phases.evaluation_config.threshold_mode
                    == configuration.TurbulenceThresholdMode.BOUNDED
                )
            if "severities" in e["phases"]["evaluation_phases"]["evaluation_config"]:
                assert (
                    config.phases.evaluation_phases.evaluation_config.severities
                    == e["phases"]["evaluation_phases"]["evaluation_config"]["severities"]
                )
            else:
                assert config.phases.evaluation_phases.evaluation_config.severities == [
                    configuration.TurbulenceSeverity.LIGHT
                ]

    if not isinstance(e, dict):
        assert e.type is InvalidConfigurationError


@pytest.fixture
def make_turbulence_config_with_threshold_file(make_empty_temp_text_file, tmp_path_factory, request) -> "Path":
    content = copy.deepcopy(request.param)
    content["phases"]["calibration_phases"]["calibration_config"]["thresholds_file_path"] = str(
        make_empty_temp_text_file
    )
    content["phases"]["evaluation_phases"]["evaluation_config"]["evaluation_data_dir"] = str(
        tmp_path_factory.mktemp("data")
    )
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
        if config.phases.evaluation_phases is not None:
            if "threshold_mode" in e["phases"]["evaluation_phases"]["evaluation_config"]:
                assert (
                    config.phases.evaluation_phases.evaluation_config.threshold_mode
                    == e["phases"]["evaluation_phases"]["evaluation_config"]["threshold_mode"]
                )
            else:
                assert (
                    config.phases.evaluation_phases.evaluation_config.threshold_mode
                    == configuration.TurbulenceThresholdMode.BOUNDED
                )
        # if "severities" in e:
        #     assert config.severities == e["severities"]
        # else:
        #     assert config.severities == [configuration.TurbulenceSeverity.LIGHT]
        # assert config.calibration_data_dir is None

    if not isinstance(e, dict):
        assert e.type is InvalidConfigurationError


@pytest.mark.parametrize(
    ("dict_to_file", "expectation"),
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
    ("dict_to_file", "expectation"),
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


# @pytest.mark.parametrize(
#     "upper, lower, expectation",
#     [
#         pytest.param(np.inf, 0.0, nullcontext(0), id="inf_upper"),
#         pytest.param(99.0, 0.0, nullcontext(0), id="zero_lower"),
#         pytest.param(99.9, 99.8, nullcontext(0), id="close_to_hundred"),
#         pytest.param(90.0, 95.0, pytest.raises(InvalidConfigurationError), id="lower_greater_than_upper"),
#         pytest.param(101.0, 90.0, pytest.raises(InvalidConfigurationError), id="upper_greater_than_100"),
#         pytest.param(99.0, -1.0, pytest.raises(ValidationError), id="negative_lower"),
#     ],
# )
# def test_turbulence_severity_percentile_config(upper: float, lower: float, expectation):
#     with expectation as e:
#         config = configuration.TurbulenceSeverityPercentileConfig(name="test", lower_bound=lower, upper_bound=upper)
#         assert config.lower_bound == lower
#         assert config.upper_bound == upper
#
#     if e != 0:
#         assert e.type is expectation.expected_exception


def test_turbulence_thresholds_all_none():
    with pytest.raises(InvalidConfigurationError) as e:
        configuration.TurbulenceThresholds(
            light=None, light_to_moderate=None, moderate=None, moderate_to_severe=None, severe=None
        )
    assert e.type is InvalidConfigurationError


@pytest.mark.parametrize(
    ("phases", "expectation"),
    [
        pytest.param([], nullcontext(), id="empty_list_phases"),
        pytest.param(None, pytest.raises(ValidationError), id="fail as phases is none"),
        pytest.param(
            [configuration.TurbulenceCalibrationPhaseOption.THRESHOLDS], nullcontext(), id="single value thresholds"
        ),
        pytest.param(
            [configuration.TurbulenceCalibrationPhaseOption.HISTOGRAM], nullcontext(), id="single value histogram"
        ),
        pytest.param(
            [
                configuration.TurbulenceCalibrationPhaseOption.HISTOGRAM,
                configuration.TurbulenceCalibrationPhaseOption.THRESHOLDS,
            ],
            nullcontext(),
            id="both values",
        ),
        pytest.param(
            [
                configuration.TurbulenceCalibrationPhaseOption.HISTOGRAM,
                configuration.TurbulenceCalibrationPhaseOption.THRESHOLDS,
                configuration.TurbulenceCalibrationPhaseOption.THRESHOLDS,
            ],
            pytest.raises(InvalidConfigurationError),
            id="fail duplicate values",
        ),
    ],
)
def test_turbulence_calibration_phases(phases, expectation, tmp_path) -> None:
    tmp_threshold_file = tmp_path / "test.json"
    tmp_threshold_file.touch()
    with expectation as e:
        config = configuration.TurbulenceCalibrationPhases(
            phases=phases,
            calibration_config=configuration.TurbulenceCalibrationConfig(thresholds_file_path=tmp_threshold_file),
        )
        assert config.phases == phases
    if phases is None:
        assert e.type is expectation.expected_exception
