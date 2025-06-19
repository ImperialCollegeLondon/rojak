from typing import TYPE_CHECKING

import numpy as np
import pytest

from rojak.orchestrator.mediators import (
    DiagnosticsAmdarDataHarmoniser,
    DiagnosticsAmdarHarmonisationStrategyOptions,
    NotWithinTimeFrameError,
)
from rojak.utilities.types import Limits

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
