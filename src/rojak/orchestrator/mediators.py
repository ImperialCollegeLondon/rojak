from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Mapping, NamedTuple

import numpy as np
import pandas as pd

from rojak.core.calculations import bilinear_interpolation
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    import dask.dataframe as dd
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
    _met_values: dict["DiagnosticName", "xr.DataArray"]

    TIME_WINDOW_DELTA: ClassVar[np.timedelta64] = np.timedelta64(3, "h")

    def __init__(self, name_suffix: str, met_values: dict["DiagnosticName", "xr.DataArray"]) -> None:
        self._name_suffix = name_suffix
        self._met_values = met_values

    @property
    def name_suffix(self) -> str:
        return self._name_suffix

    def get_nearest_values(self, indexer: SpatialTemporalIndex, values_array: "xr.DataArray") -> "xr.DataArray":
        closest_values: "xr.DataArray" = values_array.sel(
            longitude=indexer.longitudes, latitude=indexer.latitudes, pressure_level=indexer.level
        ).sel(
            time=[indexer.obs_time - self.TIME_WINDOW_DELTA, indexer.obs_time + self.TIME_WINDOW_DELTA],
            method="nearest",
        )
        if len(closest_values["time"]) != 1:
            closest_time = np.abs(closest_values["time"].to_numpy() - indexer.obs_time).argmin()
            closest_values = closest_values.isel(time=closest_time)
        return closest_values

    def harmonise(self, indexer: SpatialTemporalIndex, observation_coord: Coordinate) -> dict:
        output = {}
        for name, diagnostic in self._met_values.items():  # DiagnosticName, xr.DataArray
            surrounding_values: "xr.DataArray" = self.get_nearest_values(indexer, diagnostic)

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
    def __init__(self, name_suffix: str, met_values: dict["DiagnosticName", "xr.DataArray"]) -> None:
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
        ).item()


class DiagnosticsSeveritiesStrategy(DiagnosticsAmdarHarmonisationStrategy):
    _severity_bounds: Mapping["DiagnosticName", "Limits"]

    def __init__(
        self,
        name_suffix: str,
        met_values: dict["DiagnosticName", "xr.DataArray"],
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
        ).item()
        threshold = self._thresholds[diagnostic_name]
        return threshold.lower <= interpolated_value < threshold.upper


class EdrSeveritiesStrategy(DiagnosticsAmdarHarmonisationStrategy):
    _edr_bounds: "Limits"

    def __init__(
        self, name_suffix: str, met_values: dict["DiagnosticName", "xr.DataArray"], edr_bounds: "Limits"
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
        ).item()
        return self._edr_bounds.lower <= interpolated_value < self._edr_bounds.upper


class DiagnosticsAmdarHarmonisationStrategyFactory:
    _diagnostics_suite: "EvaluationDiagnosticSuite"

    def __init__(self, diagnostics_suite: "EvaluationDiagnosticSuite") -> None:
        self._diagnostics_suite = diagnostics_suite

    def get_met_values(
        self, option: DiagnosticsAmdarHarmonisationStrategyOptions
    ) -> dict["DiagnosticName", "xr.DataArray"]:
        if option in {
            DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
            DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY,
        }:
            return self._diagnostics_suite.computed_values_as_dict()
        return dict(self._diagnostics_suite.edr)

    def create_strategies(
        self, options: list[DiagnosticsAmdarHarmonisationStrategyOptions]
    ) -> list[DiagnosticsAmdarHarmonisationStrategy]:
        strategies = []
        for option in options:
            met_values = self.get_met_values(option)
            match option:
                case DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES:
                    strategies.append(ValuesStrategy(str(option), met_values))
                case DiagnosticsAmdarHarmonisationStrategyOptions.EDR:
                    strategies.append(ValuesStrategy(str(option), met_values))
                case DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY:
                    for severity, threshold_mapping in self._diagnostics_suite.get_limits_for_severities():
                        strategies.append(
                            DiagnosticsSeveritiesStrategy(
                                f"{str(option)}_{str(severity)}",
                                met_values,
                                threshold_mapping,
                            )
                        )
                case DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY:
                    for severity, edr_limits in self._diagnostics_suite.get_edr_bounds():
                        strategies.append(
                            EdrSeveritiesStrategy(f"{str(option)}_{str(severity)}", met_values, edr_limits)
                        )
        return strategies


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

    @staticmethod
    def _process_amdar_row(
        row: "dd.Series", methods: list[DiagnosticsAmdarHarmonisationStrategy], amdar_turblence_columns: list[str]
    ) -> pd.Series:
        # Use bounds as the DataArray.sel will get the (2, 2) data
        min_lon, min_lat, max_lon, max_lat = row["grid_box"].bounds
        longitudes = [min_lon, max_lon]
        latitudes = [min_lat, max_lat]

        level: float = row["level"]
        this_time: np.datetime64 = np.datetime64(row["datetime"])
        target_coord: Coordinate = Coordinate(row["latitude"], row["longitude"])

        new_row = {
            "datetime": this_time,
            "level": row["level"],
            "geometry": row["geometry"],
            "grid_box": row["grid_box"],
            "index_right": row["index_right"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        }
        for column in amdar_turblence_columns:
            new_row[column] = row[column]

        indexer: SpatialTemporalIndex = SpatialTemporalIndex(longitudes, latitudes, level, this_time)
        for method in methods:
            new_row = new_row | method.harmonise(indexer, target_coord)

        return pd.Series(new_row)

    def _check_time_window_within_met_data(self, time_window: "Limits[np.datetime64]") -> None:
        time_coordinate: "xr.DataArray" = next(iter(self._diagnostics_suite.computed_values_as_dict().values()))["time"]
        if time_window.lower < time_coordinate.min() or time_window.upper > time_coordinate.max():
            raise NotWithinTimeFrameError("Time window is not within time coordinate of met data")

    def execute_harmonisation(
        self,
        methods: list[DiagnosticsAmdarHarmonisationStrategyOptions],
        time_window: "Limits[np.datetime64]",
    ) -> "dd.DataFrame":
        self._check_time_window_within_met_data(time_window)

        observational_data: "dd.DataFrame" = self._amdar_data.clip_to_time_window(time_window)

        strategies: list[DiagnosticsAmdarHarmonisationStrategy] = DiagnosticsAmdarHarmonisationStrategyFactory(
            self._diagnostics_suite
        ).create_strategies(methods)

        meta_for_new_dataframe = {
            "datetime": "datetime64[s]",
            "level": float,
            "geometry": object,
            "grid_box": object,
            "index_right": int,
            "latitude": float,
            "longitude": float,
        }
        column_names = self._amdar_data.turbulence_column_names()
        for name in column_names:
            meta_for_new_dataframe[name] = float
        for strategy in strategies:
            for name in self._diagnostics_suite.diagnostic_names():
                meta_for_new_dataframe[f"{name}_{strategy.name_suffix}"] = float

        return observational_data.apply(
            self._process_amdar_row,
            axis=1,
            meta=meta_for_new_dataframe,
            args=(strategies, self._amdar_data.turbulence_column_names()),
        )
