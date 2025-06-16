from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import dask.dataframe as dd

    from rojak.core.data import AmdarTurbulenceData
    from rojak.turbulence.diagnostic import DiagnosticSuite
    from rojak.utilities.types import Limits


class DiagnosticsAmdarComparator:
    """
    Class handles the comparison of diagnostics computed from meteorological data and AMDAR observational data
    """

    _amdar_data: "AmdarTurbulenceData"
    _diagnostics_suite: "DiagnosticSuite"

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "DiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite

    def _process_amdar_row(self, row: "dd.Series", action: str) -> "dd.Series":
        return row

    def compute_action(self, action: str, time_window: "Limits[np.datetime64]") -> "dd.DataFrame":
        observational_data: "dd.DataFrame" = self._amdar_data.clip_to_time_window(time_window)
        meta_for_new_dataframe = {
            "datetime": np.datetime64,
            "level": float,
            "geometry": object,
            "grid_box": object,
            "index_right": int,
        }
        for item in self._diagnostics_suite.diagnostic_names():
            meta_for_new_dataframe[item] = float

        return observational_data.apply(
            self._process_amdar_row,
            axis=1,
            meta=meta_for_new_dataframe,
            args=action,
        )
