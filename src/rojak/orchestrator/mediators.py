from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Iterator, Mapping, NamedTuple, Tuple

import dask.dataframe as dd
import numpy as np

from rojak.core.calculations import bilinear_interpolation
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.turbulence.diagnostic import EvaluationDiagnosticSuite
    from rojak.utilities.types import DiagnosticName, Limits


class NotWithinTimeFrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DiagnosticsAmdarHarmonisationStrategyOptions(StrEnum):
    RAW_INDEX_VALUES = "raw"
    INDEX_TURBULENCE_INTENSITY = "index_severity"
    EDR = "edr"
    EDR_TURBULENCE_INTENSITY = "edr_severity"


class SpatialTemporalIndex(NamedTuple):
    longitudes: list[float]
    latitudes: list[float]
    level: float
    obs_time: np.datetime64


class DiagnosticsAmdarHarmonisationStrategy(ABC):
    _name_suffix: str
    _met_values: Iterator[Tuple["DiagnosticName", "xr.DataArray"]]

    def __init__(self, name_suffix: str, met_values: Iterator[Tuple["DiagnosticName", "xr.DataArray"]]) -> None:
        self._name_suffix = name_suffix
        self._met_values = met_values

    @staticmethod
    def get_nearest_values(indexer: SpatialTemporalIndex, values_array: "xr.DataArray") -> "xr.DataArray":
        return values_array.sel(longitude=indexer.longitudes, latitude=indexer.latitudes, level=indexer.level).sel(
            time=indexer.obs_time, method="nearest"
        )

    @staticmethod
    def check_time_within_window(surrounding_values: "xr.DataArray", observation_time: np.datetime64) -> None:
        closest_time_stamp = surrounding_values["time"].values[0]
        if not (
            closest_time_stamp - np.timedelta64(3, "h")
            <= observation_time
            <= closest_time_stamp + np.timedelta64(3, "h")
        ):
            raise NotWithinTimeFrameError("Observation time is not within +/- 3 hrs of the closest time stamp")

    def harmonise(self, indexer: SpatialTemporalIndex, observation_coord: Coordinate) -> dict:
        output = {}
        for name, diagnostic in self._met_values:  # DiagnosticName, xr.DataArray
            surrounding_values: "xr.DataArray" = self.get_nearest_values(indexer, diagnostic)
            self.check_time_within_window(surrounding_values, indexer.obs_time)

            output[f"{name}_{self._name_suffix}"] = self.interpolate(
                observation_coord, indexer, surrounding_values.values, name
            )

        return output

    # Re-usable method that's used to interpolated from the selected data points to the target point
    @abstractmethod
    def interpolate(
        self,
        observation_coord: Coordinate,
        indexer: SpatialTemporalIndex,
        diagnostic_values: "NDArray",
        diagnostic_name: "DiagnosticName",
    ) -> float: ...


class ValuesStrategy(DiagnosticsAmdarHarmonisationStrategy):
    def __init__(self, name_suffix: str, met_values: Iterator[Tuple["DiagnosticName", "xr.DataArray"]]) -> None:
        super().__init__(name_suffix, met_values)

    def interpolate(
        self,
        observation_coord: Coordinate,
        indexer: SpatialTemporalIndex,
        diagnostic_values: "NDArray",
        diagnostic_name: "DiagnosticName",
    ) -> float:
        return bilinear_interpolation(
            np.asarray(indexer.longitudes),
            np.asarray(indexer.latitudes),
            diagnostic_values,
            observation_coord,
        )[0]


class DiagnosticsSeveritiesStrategy(DiagnosticsAmdarHarmonisationStrategy):
    _severity_bounds: Mapping["DiagnosticName", "Limits"]

    def __init__(
        self,
        name_suffix: str,
        met_values: Iterator[Tuple["DiagnosticName", "xr.DataArray"]],
        thresholds: Mapping["DiagnosticName", "Limits"],
    ) -> None:
        super().__init__(name_suffix, met_values)
        self._thresholds = thresholds

    def interpolate(
        self,
        observation_coord: Coordinate,
        indexer: SpatialTemporalIndex,
        diagnostic_values: "NDArray",
        diagnostic_name: "DiagnosticName | None",
    ) -> float:
        assert diagnostic_name is not None
        interpolated_value: float = bilinear_interpolation(
            np.asarray(indexer.longitudes),
            np.asarray(indexer.latitudes),
            diagnostic_values,
            observation_coord,
        )[0]
        threshold = self._thresholds[diagnostic_name]
        return threshold.lower <= interpolated_value < threshold.upper


class EdrSeveritiesStrategy(DiagnosticsAmdarHarmonisationStrategy):
    _edr_bounds: "Limits"

    def __init__(
        self, name_suffix: str, met_values: Iterator[Tuple["DiagnosticName", "xr.DataArray"]], edr_bounds: "Limits"
    ) -> None:
        super().__init__(name_suffix, met_values)
        self._edr_bounds = edr_bounds

    def interpolate(
        self,
        observation_coord: Coordinate,
        indexer: SpatialTemporalIndex,
        diagnostic_values: "NDArray",
        diagnostic_name: "DiagnosticName",
    ) -> float:
        interpolated_value: float = bilinear_interpolation(
            np.asarray(indexer.longitudes),
            np.asarray(indexer.latitudes),
            diagnostic_values,
            observation_coord,
        )[0]
        return self._edr_bounds.lower <= interpolated_value < self._edr_bounds.upper


class DiagnosticsAmdarHarmonisationStrategyFactory:
    _diagnostics_suite: "EvaluationDiagnosticSuite"

    def __init__(self, diagnostics_suite: "EvaluationDiagnosticSuite") -> None:
        self._diagnostics_suite = diagnostics_suite

    def get_met_values(
        self, option: DiagnosticsAmdarHarmonisationStrategyOptions
    ) -> Iterator[Tuple["DiagnosticName", "xr.DataArray"]]:
        if option in {
            DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
            DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY,
        }:
            return iter(self._diagnostics_suite.computed_values(""))
        return iter(self._diagnostics_suite.edr.items())

    def get_strategy(
        self, options: list[DiagnosticsAmdarHarmonisationStrategyOptions]
    ) -> list[DiagnosticsAmdarHarmonisationStrategy]:
        return []


class DiagnosticsAmdarDataHarmoniser:
    """
    Class brings together the turbulence diagnostics computed from meteorological data and AMDAR observational data
    to form a unified data set
    """

    _amdar_data: "AmdarTurbulenceData"
    _diagnostics_suite: "EvaluationDiagnosticSuite"

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "EvaluationDiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite

    def _process_amdar_row(
        self, row: "dd.Series", comparison_method: DiagnosticsAmdarHarmonisationStrategy, use_edr: bool
    ) -> "dd.Series":
        coords = row["grid_box"].exterior.coords
        longitudes = [coord[0] for coord in coords]
        latitudes = [coord[1] for coord in coords]
        level: float = row["level"]
        this_time: np.datetime64 = row["datetime"]
        target_locations: Coordinate = Coordinate(row["latitude"], row["longitude"])  # noqa: F841

        for name, diagnostic in self._diagnostics_suite.computed_values(""):  # DiagnosticName, Diagnostic
            diagnostic.sel(longitude=longitudes, latitude=latitudes, level=level).sel(time=this_time, method="nearest")
            print(name)

        return dd.Series([this_time, row["level"], row["geometry"], row["grid_box"], row["index_right"]])

    def execute_harmonisation(
        self,
        methods: list[DiagnosticsAmdarHarmonisationStrategyOptions],
        time_window: "Limits[np.datetime64]",
        use_edr: bool = False,
    ) -> "dd.DataFrame":
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

        if use_edr:
            try:
                _ = self._diagnostics_suite.edr
            except ValueError as exception:
                raise ValueError(
                    "Attempting to assimilate EDR data when missing values needed to compute EDR"
                ) from exception

        return observational_data.apply(
            self._process_amdar_row,
            axis=1,
            meta=meta_for_new_dataframe,
            args=(methods, use_edr),
        )
