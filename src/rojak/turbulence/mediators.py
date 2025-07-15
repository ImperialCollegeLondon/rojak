import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, ClassVar, Generator, Mapping, NamedTuple, assert_never

import dask.dataframe as dd
import numpy as np
import pandas as pd

from rojak.core.calculations import bilinear_interpolation
from rojak.core.distributed_tools import blocking_wait_futures
from rojak.core.indexing import map_values_to_nearest_coordinate_index
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.orchestrator.configuration import TurbulenceSeverity
    from rojak.turbulence.diagnostic import EvaluationDiagnosticSuite
    from rojak.utilities.types import DiagnosticName, Limits

logger = logging.getLogger(__name__)


class NotWithinTimeFrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DiagnosticsAmdarHarmonisationStrategyOptions(StrEnum):
    RAW_INDEX_VALUES = "raw"
    INDEX_TURBULENCE_INTENSITY = "index_severity"
    EDR = "edr"
    EDR_TURBULENCE_INTENSITY = "edr_severity"

    def column_name_method(self, severity: "TurbulenceSeverity | None" = None) -> Callable:
        match self:
            case (
                DiagnosticsAmdarHarmonisationStrategyOptions.EDR
                | DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES
            ):
                assert severity is None
                return lambda name: f"{name}_{str(self)}"
            case (
                DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY
                | DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY
            ):
                assert severity is not None
                return lambda name: f"{name}_{str(self)}_{str(severity)}"
            case _ as unreachable:
                assert_never(unreachable)


class SpatialTemporalIndex(NamedTuple):
    longitudes: list[float]
    latitudes: list[float]
    level: float
    obs_time_index: int


class DiagnosticsAmdarHarmonisationStrategy(ABC):
    _column_namer: Callable
    _met_values: dict["DiagnosticName", "xr.DataArray"]

    def __init__(self, column_namer: Callable, met_values: dict["DiagnosticName", "xr.DataArray"]) -> None:
        self._column_namer = column_namer
        self._met_values = met_values

    def column_name(self, diagnostic_name: "DiagnosticName") -> str:
        return self._column_namer(diagnostic_name)

    @staticmethod
    def get_nearest_values(indexer: SpatialTemporalIndex, values_array: "xr.DataArray") -> "xr.DataArray":
        return values_array.isel(time=indexer.obs_time_index).sel(
            longitude=indexer.longitudes, latitude=indexer.latitudes, pressure_level=indexer.level
        )

    def harmonise(self, indexer: SpatialTemporalIndex, observation_coord: Coordinate) -> dict:
        output = {}
        for name, diagnostic in self._met_values.items():  # DiagnosticName, xr.DataArray
            surrounding_values: "xr.DataArray" = self.get_nearest_values(indexer, diagnostic)

            output[self.column_name(name)] = self.interpolate(
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
    def __init__(self, column_namer: Callable, met_values: dict["DiagnosticName", "xr.DataArray"]) -> None:
        super().__init__(column_namer, met_values)

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
        column_namer: Callable,
        met_values: dict["DiagnosticName", "xr.DataArray"],
        thresholds: Mapping["DiagnosticName", "Limits"],
    ) -> None:
        super().__init__(column_namer, met_values)
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
        self, column_namer: Callable, met_values: dict["DiagnosticName", "xr.DataArray"], edr_bounds: "Limits"
    ) -> None:
        super().__init__(column_namer, met_values)
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
                    strategies.append(ValuesStrategy(option.column_name_method(), met_values))
                case DiagnosticsAmdarHarmonisationStrategyOptions.EDR:
                    strategies.append(ValuesStrategy(option.column_name_method(), met_values))
                case DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY:
                    for severity, threshold_mapping in self._diagnostics_suite.get_limits_for_severities():
                        strategies.append(
                            DiagnosticsSeveritiesStrategy(
                                option.column_name_method(severity=severity),
                                met_values,
                                threshold_mapping,
                            )
                        )
                case DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY:
                    for severity, edr_limits in self._diagnostics_suite.get_edr_bounds():
                        strategies.append(
                            EdrSeveritiesStrategy(option.column_name_method(severity=severity), met_values, edr_limits)
                        )
        logger.debug("Finished instantiating the harmonisation strategies")
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

    def _process_amdar_row(
        self,
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
            self.grid_box_column_name: row[self.grid_box_column_name],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            self.common_time_column_name: row[self.common_time_column_name],
        }
        for column in amdar_turblence_columns:
            new_row[column] = row[column]

        indexer: SpatialTemporalIndex = SpatialTemporalIndex(
            longitudes, latitudes, level, row[self.common_time_column_name]
        )
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
        """
        Method checks that grid of dataframe matches that of the representative array. As the coordinates of the
        array are numpy arrays, this is a potentially expensive as a large dask graph is sent

        Args:
            representative_array: A data array that has the coordinates longitude and latitude.

        Returns:
            None
        """
        assert {"min_lon", "max_lon", "min_lat", "max_lat"}.issubset(self._amdar_data.data_frame.columns)
        assert {"latitude", "longitude"}.issubset(representative_array.coords)
        latitudes = representative_array["latitude"].to_series()
        longitudes = representative_array["longitude"].to_series()

        for column in ["min_lon", "max_lon", "min_lat", "max_lat"]:
            if column.endswith("lon"):
                assert self._amdar_data.data_frame[column].unique().isin(longitudes).all().compute()
            else:  # endswith("lat")
                assert self._amdar_data.data_frame[column].unique().isin(latitudes).all().compute()

    def strategy_values_columns(
        self, strategies: list[DiagnosticsAmdarHarmonisationStrategyOptions]
    ) -> Generator[str, None, None]:
        for strategy in filter(
            lambda strat: strat
            in {
                DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
                DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
            },
            strategies,
        ):
            for name in self._diagnostics_suite.diagnostic_names():
                yield strategy.column_name_method()(name)

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
            "time_index": int,
        }
        column_names = self._amdar_data.turbulence_column_names()
        for name in column_names:
            meta_for_new_dataframe[name] = float
        for strategy in strategies:
            for name in self._diagnostics_suite.diagnostic_names():
                meta_for_new_dataframe[strategy.column_name(name)] = float
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

    @property
    def common_time_column_name(self) -> str:
        return "time_index"

    @property
    def grid_box_column_name(self) -> str:
        return "index_right"

    def execute_harmonisation(
        self,
        methods: list[DiagnosticsAmdarHarmonisationStrategyOptions],
        time_window: "Limits[np.datetime64]",
    ) -> "dd.DataFrame":
        a_computed_diagnostic: "xr.DataArray" = next(iter(self._diagnostics_suite.computed_values_as_dict().values()))
        self._check_time_window_within_met_data(time_window, a_computed_diagnostic)
        # self._check_grids_match(a_computed_diagnostic)
        logger.debug("Verified met data and AMDAR data can be harmonised")

        observational_data: "dd.DataFrame" = self._amdar_data.clip_to_time_window(time_window)
        current_dtypes = observational_data.dtypes.to_dict()
        current_dtypes[self.common_time_column_name] = int
        observational_data = observational_data.map_partitions(
            self._add_time_index_column,
            observational_data["datetime"],
            a_computed_diagnostic["time"].values,
            meta=current_dtypes,
        ).persist()
        logger.debug("Observational data successfully prepared to be harmonised")
        # https://docs.dask.org/en/stable/user-interfaces.html#combining-interfaces
        blocking_wait_futures(observational_data)
        logger.debug("Futures from persisting observational data have completed successfully")

        strategies: list[DiagnosticsAmdarHarmonisationStrategy] = DiagnosticsAmdarHarmonisationStrategyFactory(
            self._diagnostics_suite
        ).create_strategies(methods)

        logger.info("Starting harmonisation with %s strategies", len(strategies))
        return observational_data.apply(
            self._process_amdar_row,
            axis=1,
            meta=self._construct_meta_dataframe(strategies, observational_data.npartitions),
            args=(strategies, self._amdar_data.turbulence_column_names()),
        )
