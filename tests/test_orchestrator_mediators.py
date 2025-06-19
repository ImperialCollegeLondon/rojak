from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from shapely import box

from rojak.core.geometric import create_grid_data_frame
from rojak.orchestrator.mediators import (
    DiagnosticsAmdarDataHarmoniser,
    DiagnosticsAmdarHarmonisationStrategyOptions,
    NotWithinTimeFrameError,
)
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from shapely.geometry import Polygon


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


@pytest.mark.parametrize(
    ("grid_shape", "grid_spacing"),
    [
        pytest.param(box(-130, 25, 28, 60), 0.1, id="grid_spacing_too_small"),
        pytest.param(box(-180, 25, 28, 60), 0.25, id="lower_x_limit_too_low"),
        pytest.param(box(-130, 0, 28, 60), 0.25, id="lower_y_limit_too_low"),
        pytest.param(box(-130, 25, 80, 60), 0.25, id="upper_x_limit_too_high"),
        pytest.param(box(-130, 25, 28, 180), 0.25, id="upper_y_limit_too_high"),
    ],
)
def test_diagnostic_amdar_data_harmoniser_execute_harmonisation_fail_on_grid(
    grid_shape: "Polygon",
    grid_spacing: float,
    mocker: "MockerFixture",
) -> None:
    x_axis = np.arange(-130, 28, 0.25)
    y_axis = np.arange(25, 60, 0.25)
    representative_array = xr.DataArray(
        data=np.random.default_rng().random((len(x_axis), len(y_axis))),
        coords={"longitude": x_axis, "latitude": y_axis},
    )
    suite_mock = mocker.Mock()
    computed_vals_mock = mocker.patch.object(
        suite_mock, "computed_values_as_dict", return_value={"unused_key": representative_array}
    )
    amdar_data_mock = mocker.Mock()
    mocker.patch.object(amdar_data_mock, "grid", new=create_grid_data_frame(box(-130, 25, 28, 60), 0.25))

    harmoniser = DiagnosticsAmdarDataHarmoniser(amdar_data_mock, suite_mock)
    time_window_check_mock = mocker.patch.object(harmoniser, "_check_time_window_within_met_data")

    with pytest.raises(ValueError, match="Grid points are not coordinates of met data") as excinfo:
        harmoniser.execute_harmonisation(
            [DiagnosticsAmdarHarmonisationStrategyOptions.EDR],
            Limits(np.datetime64("1970-01-01"), np.datetime64("1980-01-01")),
        )

    assert excinfo.type is ValueError
    time_window_check_mock.assert_called_once()
    computed_vals_mock.assert_called_once()
