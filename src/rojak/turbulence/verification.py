#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import dask.dataframe as dd
import numpy as np
import pandas as pd

from rojak.core.calculations import bilinear_interpolation
from rojak.core.distributed_tools import blocking_wait_futures
from rojak.core.indexing import map_values_to_nearest_coordinate_index
from rojak.orchestrator.configuration import DiagnosticsAmdarHarmonisationStrategyOptions
from rojak.turbulence.diagnostic import EvaluationDiagnosticSuite
from rojak.turbulence.metrics import BinaryClassificationResult, area_under_curve, received_operating_characteristic
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.orchestrator.configuration import DiagnosticValidationCondition
    from rojak.turbulence.diagnostic import DiagnosticSuite
    from rojak.utilities.types import DiagnosticName, Limits


logger = logging.getLogger(__name__)


class NotWithinTimeFrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


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
            surrounding_values: xr.DataArray = self.get_nearest_values(indexer, diagnostic)

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
    _diagnostics_suite: "DiagnosticSuite"

    def __init__(self, diagnostics_suite: "DiagnosticSuite") -> None:
        self._diagnostics_suite = diagnostics_suite

    def get_met_values(
        self, option: DiagnosticsAmdarHarmonisationStrategyOptions
    ) -> dict["DiagnosticName", "xr.DataArray"]:
        if option in {
            DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES,
            DiagnosticsAmdarHarmonisationStrategyOptions.INDEX_TURBULENCE_INTENSITY,
        }:
            return self._diagnostics_suite.computed_values_as_dict()
        assert isinstance(self._diagnostics_suite, EvaluationDiagnosticSuite)
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
                    assert isinstance(self._diagnostics_suite, EvaluationDiagnosticSuite)
                    for severity, threshold_mapping in self._diagnostics_suite.get_limits_for_severities():
                        strategies.append(
                            DiagnosticsSeveritiesStrategy(
                                option.column_name_method(severity=severity),
                                met_values,
                                threshold_mapping,
                            )
                        )
                case DiagnosticsAmdarHarmonisationStrategyOptions.EDR_TURBULENCE_INTENSITY:
                    assert isinstance(self._diagnostics_suite, EvaluationDiagnosticSuite)
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
    _diagnostics_suite: "DiagnosticSuite"

    TIME_WINDOW_DELTA: ClassVar[np.timedelta64] = np.timedelta64(3, "h")

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "DiagnosticSuite") -> None:
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
        time_coordinate: xr.DataArray = first_diagnostic["time"]
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
        a_computed_diagnostic: xr.DataArray = self._diagnostics_suite.get_prototype_computed_diagnostic()
        self._check_time_window_within_met_data(time_window, a_computed_diagnostic)
        # self._check_grids_match(a_computed_diagnostic)
        logger.debug("Verified met data and AMDAR data can be harmonised")

        observational_data: dd.DataFrame = self._amdar_data.clip_to_time_window(time_window)
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


def _observed_turbulence_aggregation(condition: "DiagnosticValidationCondition") -> dd.Aggregation:
    # See https://docs.dask.org/en/latest/dataframe-groupby.html#dataframe-groupby-aggregate
    def on_chunk(within_partition: "pd.Series") -> float:
        return within_partition.max()

    def aggregate_chunks(chunk_maxes: "pd.Series") -> float:
        return chunk_maxes.max()

    def apply_condition(maxima: float) -> float:
        return maxima > condition.value_greater_than

    return dd.Aggregation(
        name=f"has_turbulence_{condition.observed_turbulence_column_name}_{condition.value_greater_than:0.2f}",
        chunk=on_chunk,
        agg=aggregate_chunks,
        finalize=apply_condition,
    )


@dataclasses.dataclass
class RocVerificationResult:
    # first key: amdar column name
    # second key: diagnostic column name
    by_amdar_col_then_diagnostic: dict[str, dict[str, BinaryClassificationResult]]
    _auc_by_amdar_then_diagnostic: dict[str, dict[str, float]] | None = None

    # define iterators and getters on this class
    def iterate_by_amdar_column(self) -> Generator[tuple[str, dict[str, BinaryClassificationResult]], None, None]:
        yield from self.by_amdar_col_then_diagnostic.items()

    def _area_under_curve(self) -> dict[str, dict[str, float]]:
        if self._auc_by_amdar_then_diagnostic is None:
            auc = defaultdict(dict)
            for column, by_diagnostic in self.by_amdar_col_then_diagnostic.items():
                for diagnostic_name, roc_results in by_diagnostic.items():
                    auc[column][diagnostic_name] = area_under_curve(
                        roc_results.false_positives, roc_results.true_positives
                    )
            self._auc_by_amdar_then_diagnostic = dict(auc)
        return self._auc_by_amdar_then_diagnostic

    def auc_for_amdar_column(self, amdar_column: str) -> dict[str, float]:
        return self._area_under_curve()[amdar_column]


# Keep this extendable for verification against other forms of data??
class DiagnosticsAmdarVerification:
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
            data: dd.DataFrame = self._data_harmoniser.execute_harmonisation(
                [DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES], self._time_window
            ).persist()  # Need to do this assignment to make pyright happy
            self._harmonised_data = data
            blocking_wait_futures(self._harmonised_data)
            return self._harmonised_data
        return self._harmonised_data

    def _add_nearest_grid_indices(
        self,
        validation_columns: list[str],
        grid_prototype: "xr.DataArray",
    ) -> "dd.DataFrame":
        if not set(validation_columns).issubset(self.data.columns):
            raise ValueError("Validation columns must be present in the data")

        space_time_columns: list[str] = [
            self._data_harmoniser.common_time_column_name,
            "level",
            "longitude",
            "latitude",
        ]
        target_columns = space_time_columns + validation_columns
        target_data: dd.DataFrame = self.data[target_columns]
        dataframe_meta: dict[str, pd.Series] = {
            col_name: pd.Series(dtype=col_dtype) for col_name, col_dtype in target_data.dtypes.to_dict().items()
        }
        dataframe_meta["level_index"] = pd.Series(dtype=int)
        target_data = target_data.map_partitions(
            lambda df: df.assign(
                level_index=df.apply(
                    lambda row, pressure_level=grid_prototype["pressure_level"].values: np.abs(  # noqa: PD011
                        row.level - pressure_level
                    ).argmin(),
                    axis=1,
                )
            ),
            meta=dataframe_meta,
        )
        dataframe_meta["lat_index"] = pd.Series(dtype=int)
        dataframe_meta["lon_index"] = pd.Series(dtype=int)
        return target_data.map_partitions(
            lambda df: df.assign(
                lat_index=map_values_to_nearest_coordinate_index(df.latitude, grid_prototype["latitude"].values),
                lon_index=map_values_to_nearest_coordinate_index(df.longitude, grid_prototype["longitude"].values),
            ),
            meta=dd.from_pandas(pd.DataFrame(dataframe_meta)),
        ).optimize()

    def _spatio_temporal_data_aggregation(
        self,
        target_data: "dd.DataFrame",
        strategy_columns: list[str],
        validation_conditions: "list[DiagnosticValidationCondition]",
    ) -> "dd.DataFrame":
        group_by_columns: list[str] = [
            "lat_index",
            "lon_index",
            "level_index",
            self._data_harmoniser.common_time_column_name,
        ]
        target_data = target_data.drop(columns=["level", "longitude", "latitude"])
        grouped_by_space_time = target_data.groupby(group_by_columns)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _observed_turbulence_aggregation(condition)
            for condition in validation_conditions
        }
        for strategy_column in strategy_columns:
            aggregation_spec[strategy_column] = "mean"
        return grouped_by_space_time.aggregate(aggregation_spec)

    def compute_roc_curve(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        prototype_diagnostic_array: "xr.DataArray",
    ) -> RocVerificationResult:
        assert {"pressure_level", "longitude", "latitude", "time"}.issubset(prototype_diagnostic_array.coords)
        diagnostic_value_columns: list[str] = list(
            self._data_harmoniser.strategy_values_columns(
                [DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES]
            )
        )
        validation_columns: list[str] = [
            condition.observed_turbulence_column_name for condition in validation_conditions
        ]
        target_data = self._add_nearest_grid_indices(
            validation_columns + diagnostic_value_columns,
            prototype_diagnostic_array,
        )
        logger.debug("Added required columns for spatio temporal aggregating")
        aggregated_data = self._spatio_temporal_data_aggregation(
            target_data, diagnostic_value_columns, validation_conditions
        ).persist()
        logger.debug("Triggered spatio temporal aggregation")
        blocking_wait_futures(aggregated_data)
        logger.debug("Finished spatio temporal aggregation")
        result: defaultdict[str, dict[str, BinaryClassificationResult]] = defaultdict(dict)
        for diagnostic_val_col in diagnostic_value_columns:
            # descending values
            subset_df = aggregated_data[[*validation_columns, diagnostic_val_col]].sort_values(
                diagnostic_val_col, ascending=False
            )
            for amdar_turbulence_col in validation_columns:
                result[amdar_turbulence_col][diagnostic_val_col] = received_operating_characteristic(
                    subset_df[amdar_turbulence_col].values.compute_chunk_sizes(),  # noqa: PD011
                    subset_df[diagnostic_val_col].values.compute_chunk_sizes(),  # noqa: PD011
                )
        logger.debug("Finished computing ROC curves")

        return RocVerificationResult(dict(result))
