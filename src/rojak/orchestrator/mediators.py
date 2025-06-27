from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Mapping, NamedTuple

import dask.dataframe as dd
import numpy as np
import pandas as pd

from rojak.core.calculations import bilinear_interpolation
from rojak.core.indexing import map_values_to_nearest_coordinate_index
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray
    from shapely.geometry import Polygon

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
    obs_time_index: int


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

    @staticmethod
    def get_nearest_values(indexer: SpatialTemporalIndex, values_array: "xr.DataArray") -> "xr.DataArray":
        return values_array.isel(time=indexer.obs_time_index).sel(
            longitude=indexer.longitudes, latitude=indexer.latitudes, pressure_level=indexer.level
        )

    def harmonise(self, indexer: SpatialTemporalIndex, observation_coord: Coordinate) -> dict:
        output = {}
        for name, diagnostic in self._met_values.items():  # DiagnosticName, xr.DataArray
            surrounding_values: "xr.DataArray" = self.get_nearest_values(indexer, diagnostic)

            output[f"{name}_{self._name_suffix}"] = self.interpolate(
                observation_coord, indexer, surrounding_values.to_numpy(), name
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

    TIME_WINDOW_DELTA: ClassVar[np.timedelta64] = np.timedelta64(3, "h")

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "EvaluationDiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite

    @staticmethod
    def _process_amdar_row(
        row: "dd.Series",
        methods: list[DiagnosticsAmdarHarmonisationStrategy],
        amdar_turblence_columns: list[str],
    ) -> pd.Series:
        longitudes = [row["min_lon"], row["max_lon"]]
        latitudes = [row["min_lat"], row["max_lat"]]

        level: float = row["level"]
        this_time: np.datetime64 = np.datetime64(row["datetime"])
        target_coord: Coordinate = Coordinate(row["latitude"], row["longitude"])

        new_row = {
            "datetime": this_time,
            "level": row["level"],
            "geometry": row["geometry"],
            "index_right": row["index_right"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        }
        for column in amdar_turblence_columns:
            new_row[column] = row[column]

        indexer: SpatialTemporalIndex = SpatialTemporalIndex(longitudes, latitudes, level, row["time_index"])
        for method in methods:
            new_row = new_row | method.harmonise(indexer, target_coord)

        return pd.Series(new_row)

    @staticmethod
    def _check_time_window_within_met_data(
        time_window: "Limits[np.datetime64]", first_diagnostic: "xr.DataArray"
    ) -> None:
        time_coordinate: "xr.DataArray" = first_diagnostic["time"]
        if time_window.lower < time_coordinate.min() or time_window.upper > time_coordinate.max():
            raise NotWithinTimeFrameError("Time window is not within time coordinate of met data")

    def _check_grids_match(self, representative_array: "xr.DataArray") -> None:
        latitudes = set(representative_array["latitude"].values)
        longitudes = set(representative_array["longitude"].values)

        def check_row(geom: "Polygon") -> bool:
            min_lon, min_lat, max_lon, max_lat = geom.bounds
            if (
                min_lon not in longitudes
                or min_lat not in latitudes
                or max_lon not in longitudes
                or max_lat not in latitudes
            ):
                raise ValueError("Grid points are not coordinates of met data")
            return True

        self._amdar_data.grid["geometry"].apply(check_row, meta=("is_valid", bool)).compute()

    def _construct_meta_dataframe(
        self, strategies: list[DiagnosticsAmdarHarmonisationStrategy], num_partitions: int
    ) -> dd.DataFrame:
        meta_for_new_dataframe = {
            "datetime": "datetime64[s]",
            "level": float,
            "geometry": object,
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
        return dd.from_pandas(
            pd.DataFrame(
                {column_name: pd.Series(dtype=data_type) for column_name, data_type in meta_for_new_dataframe.items()}
            ),
            npartitions=num_partitions,
        )

    def _add_time_index_column(
        self, dataframe: "dd.DataFrame", datetime_series: "dd.Series", time_coordinate: "NDArray"
    ) -> "dd.DataFrame":
        return dataframe.assign(
            time_index=map_values_to_nearest_coordinate_index(
                datetime_series,
                time_coordinate,
                valid_window=self.TIME_WINDOW_DELTA,
            )
        )

    def execute_harmonisation(
        self,
        methods: list[DiagnosticsAmdarHarmonisationStrategyOptions],
        time_window: "Limits[np.datetime64]",
    ) -> "dd.DataFrame":
        a_computed_diagnostic: "xr.DataArray" = next(iter(self._diagnostics_suite.computed_values_as_dict().values()))
        self._check_time_window_within_met_data(time_window, a_computed_diagnostic)
        self._check_grids_match(a_computed_diagnostic)

        observational_data: "dd.DataFrame" = self._amdar_data.clip_to_time_window(time_window)
        current_dtypes = observational_data.dtypes.to_dict()
        current_dtypes["time_index"] = int
        observational_data = observational_data.map_partitions(
            self._add_time_index_column,
            observational_data["datetime"],
            a_computed_diagnostic["time"].values,
            meta=current_dtypes,
        ).persist()

        strategies: list[DiagnosticsAmdarHarmonisationStrategy] = DiagnosticsAmdarHarmonisationStrategyFactory(
            self._diagnostics_suite
        ).create_strategies(methods)

        return observational_data.apply(
            self._process_amdar_row,
            axis=1,
            meta=self._construct_meta_dataframe(strategies, observational_data.npartitions),
            args=(strategies, self._amdar_data.turbulence_column_names()),
        )
