from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from rojak.orchestrator.configuration import TurbulenceDiagnostics, TurbulenceSeverity
from rojak.orchestrator.mediators import (
    DiagnosticsAmdarDataHarmoniser,
    DiagnosticsAmdarHarmonisationStrategy,
    DiagnosticsAmdarHarmonisationStrategyFactory,
    DiagnosticsAmdarHarmonisationStrategyOptions,
    DiagnosticsSeveritiesStrategy,
    EdrSeveritiesStrategy,
    NotWithinTimeFrameError,
    ValuesStrategy,
)
from rojak.utilities.types import Limits
from tests.conftest import generate_array_data

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize(
    "time_window",
    [
        pytest.param(Limits(np.datetime64("1970-01-01"), np.datetime64("1980-01-01")), id="below_min"),
        pytest.param(Limits(np.datetime64("2005-02-01T03:00"), np.datetime64("2025-02-01T00:00")), id="above_max"),
    ],
)
def test_diagnostic_amdar_data_harmoniser_execute_harmonisation_fail_time_window(
    mocker: "MockerFixture", make_dummy_cat_data, time_window: Limits
) -> None:
    amdar_data_mock = mocker.Mock()
    suite_mock = mocker.Mock()
    computed_vals_mock = mocker.patch.object(
        suite_mock, "computed_values_as_dict", return_value={"unused_key": make_dummy_cat_data({})}
    )

    harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data_mock, suite_mock)

    with pytest.raises(NotWithinTimeFrameError) as excinfo:
        harmoniser.execute_harmonisation([DiagnosticsAmdarHarmonisationStrategyOptions.EDR], time_window)

    assert excinfo.type is NotWithinTimeFrameError
    computed_vals_mock.assert_called_once()


# @pytest.mark.parametrize(
#     ("grid_shape", "grid_spacing"),
#     [
#         pytest.param(box(-130, 25, 28, 60), 0.1, id="grid_spacing_too_small"),
#         pytest.param(box(-180, 25, 28, 60), 0.25, id="lower_x_limit_too_low"),
#         pytest.param(box(-130, 0, 28, 60), 0.25, id="lower_y_limit_too_low"),
#         pytest.param(box(-130, 25, 80, 60), 0.25, id="upper_x_limit_too_high"),
#         pytest.param(box(-130, 25, 28, 180), 0.25, id="upper_y_limit_too_high"),
#     ],
# )
# def test_diagnostic_amdar_data_harmoniser_execute_harmonisation_fail_on_grid(
#     grid_shape: "Polygon",
#     grid_spacing: float,
#     mocker: "MockerFixture",
# ) -> None:
#     x_axis = np.arange(-130, 28, 0.25)
#     y_axis = np.arange(25, 60, 0.25)
#     representative_array = xr.DataArray(
#         data=np.random.default_rng().random((len(x_axis), len(y_axis))),
#         coords={"longitude": x_axis, "latitude": y_axis},
#     )
#     suite_mock = mocker.Mock()
#     computed_vals_mock = mocker.patch.object(
#         suite_mock, "computed_values_as_dict", return_value={"unused_key": representative_array}
#     )
#     amdar_data_mock = mocker.Mock()
#     mocker.patch.object(
#         amdar_data_mock, "data_frame", new=expand_grid_bounds(create_grid_data_frame(box(-130, 25, 28, 60), 0.25))
#     )
#
#     harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data_mock, suite_mock)
#     time_window_check_mock = mocker.patch.object(harmoniser, "_check_time_window_within_met_data")
#
#     with pytest.raises(AssertionError) as excinfo:
#         harmoniser.execute_harmonisation(
#             [DiagnosticsAmdarHarmonisationStrategyOptions.EDR],
#             Limits(np.datetime64("1970-01-01"), np.datetime64("1980-01-01")),
#         )
#
#     assert excinfo.type is AssertionError
#     time_window_check_mock.assert_called_once()
#     computed_vals_mock.assert_called_once()


@pytest.mark.parametrize(
    ("method_to_mock", "option"),
    [
        ("computed_values_as_dict", DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES),
        ("computed_values_as_dict", DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY),
        ("edr", DiagnosticsAmdarHarmonisationStrategyOptions.EDR),
        ("edr", DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY),
    ],
)
def test_diagnostic_amdar_harmonisation_strategy_factory_get_met_values(
    mocker: "MockerFixture", method_to_mock: str, option: DiagnosticsAmdarHarmonisationStrategyOptions
) -> None:
    suite_mock = mocker.Mock()
    return_val: dict = {
        "stallion-boat-tingly": xr.DataArray(
            data=generate_array_data((10, 10), True), coords={"dim0": np.arange(10), "dim1": np.arange(10)}
        )
    }
    method_mock = mocker.patch.object(suite_mock, method_to_mock, return_value=return_val)

    factory = DiagnosticsAmdarHarmonisationStrategyFactory(suite_mock)
    retrieved_values = factory.get_met_values(option)
    if method_to_mock == "edr":
        assert retrieved_values == {}
    else:
        method_mock.assert_called_once()
        xr.testing.assert_equal(retrieved_values["stallion-boat-tingly"], return_val["stallion-boat-tingly"])


def test_diagnostic_amdar_harmonisation_strategy_factory_create_strategies_values_strategy(
    mocker: "MockerFixture",
) -> None:
    suite_mock = mocker.Mock()
    factory = DiagnosticsAmdarHarmonisationStrategyFactory(suite_mock)
    get_met_mock = mocker.patch.object(factory, "get_met_values", return_value={"key": "value"})

    options = [
        DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
        DiagnosticsAmdarHarmonisationStrategyOptions.EDR,
    ]
    strategies: list[DiagnosticsAmdarHarmonisationStrategy] = factory.create_strategies(options)

    assert len(strategies) == len(options)
    get_met_mock.assert_called()
    assert get_met_mock.call_count == len(options)
    assert get_met_mock.call_args_list[0][0][0] == options[0]
    assert get_met_mock.call_args_list[1][0][0] == options[1]

    for strategy, option in zip(strategies, options, strict=False):
        assert isinstance(strategy, DiagnosticsAmdarHarmonisationStrategy)
        assert isinstance(strategy, ValuesStrategy)
        assert option == strategy.name_suffix


@pytest.mark.parametrize(
    ("method_to_mock", "option"),
    [
        ("get_limits_for_severities", DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY),
        ("get_edr_bounds", DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY),
    ],
)
def test_diagnostic_amdar_harmonisation_strategy_factory_create_strategies_non_values_trivial(
    mocker: "MockerFixture", method_to_mock: str, option: DiagnosticsAmdarHarmonisationStrategyOptions
) -> None:
    suite_mock = mocker.Mock()
    factory = DiagnosticsAmdarHarmonisationStrategyFactory(suite_mock)
    get_met_mock = mocker.patch.object(factory, "get_met_values", return_value={"key": "value"})
    method_mock = mocker.patch.object(suite_mock, method_to_mock, return_value={})

    strategy = factory.create_strategies([option])
    assert len(strategy) == 0
    get_met_mock.assert_called_once_with(option)
    method_mock.assert_called_once()


@pytest.mark.parametrize(
    ("limits_data", "suite_method_mock", "option", "child_strategy_class"),
    [
        pytest.param(
            {
                TurbulenceSeverity.LIGHT: {
                    TurbulenceDiagnostics.DEF: Limits(5, 10),
                    TurbulenceDiagnostics.BROWN1: Limits(90, 99),
                },
                TurbulenceSeverity.MODERATE: {
                    TurbulenceDiagnostics.DEF: Limits(10.0, np.inf),
                    TurbulenceDiagnostics.BROWN1: Limits(99.0, np.inf),
                },
            },
            "get_limits_for_severities",
            DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY,
            DiagnosticsSeveritiesStrategy,
            id="diagnostic_severity",
        ),
        pytest.param(
            {TurbulenceSeverity.LIGHT: Limits(5, 10), TurbulenceSeverity.MODERATE: Limits(10.0, np.inf)},
            "get_edr_bounds",
            DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY,
            EdrSeveritiesStrategy,
            id="edr_severity",
        ),
    ],
)
def test_diagnostic_amdar_harmonisation_strategy_factory_create_strategies_non_values_strategy(
    mocker: "MockerFixture",
    limits_data: dict,
    suite_method_mock: str,
    option: DiagnosticsAmdarHarmonisationStrategyOptions,
    child_strategy_class: type,
) -> None:
    suite_mock = mocker.Mock()
    factory = DiagnosticsAmdarHarmonisationStrategyFactory(suite_mock)
    get_met_mock = mocker.patch.object(factory, "get_met_values", return_value={"key": "value"})

    def limits_method():
        for key, value in limits_data.items():
            yield key, value

    mocker.patch.object(suite_mock, suite_method_mock, new=limits_method)

    strategies = factory.create_strategies([option])
    assert len(strategies) == len(limits_data)
    get_met_mock.assert_called_once_with(option)

    for strategy, severity in zip(strategies, limits_data, strict=False):
        assert isinstance(strategy, DiagnosticsAmdarHarmonisationStrategy)
        assert strategy.name_suffix == f"{str(option)}_{str(severity)}"
        assert isinstance(strategy, child_strategy_class)
