from typing import TYPE_CHECKING

from rojak.core.distributed_tools import blocking_wait_futures
from rojak.orchestrator.mediators import (
    DiagnosticsAmdarHarmonisationStrategyOptions,
)

if TYPE_CHECKING:
    import dask.dataframe as dd
    import numpy as np

    from rojak.orchestrator.mediators import (
        DiagnosticsAmdarDataHarmoniser,
    )
    from rojak.utilities.types import Limits


# Keep this extendable for verification against other forms of data??
class DiagnosticAmdarVerification:
    _data_harmoniser: "DiagnosticsAmdarDataHarmoniser"
    _harmonised_data: "dd.DataFrame | None"
    _time_window: "Limits[np.datetime64]"

    def __init__(self, data_harmoniser: "DiagnosticsAmdarDataHarmoniser", time_window: "Limits[np.datetime64]") -> None:
        self._data_harmoniser = data_harmoniser
        self._harmonised_data = None
        self._time_window = time_window

    @property
    def data(self) -> "dd.DataFrame":
        if self._harmonised_data is None:
            data: "dd.DataFrame" = self._data_harmoniser.execute_harmonisation(
                [DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES], self._time_window
            ).persist()  # Need to do this assignment to make pyright happy
            self._harmonised_data = data
            blocking_wait_futures(self._harmonised_data)
            return self._harmonised_data
        return self._harmonised_data

    def execute(
        self,
    ) -> None: ...
