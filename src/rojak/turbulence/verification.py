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
import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, NamedTuple, assert_never

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from rojak.core.calculations import bilinear_interpolation
from rojak.core.indexing import map_index_to_coordinate_value, map_order, map_values_to_nearest_coordinate_index
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


def get_dataframe_dtypes(data_frame: "dd.DataFrame") -> dict[str, pd.Series]:
    return {col_name: pd.Series(dtype=col_dtype) for col_name, col_dtype in data_frame.dtypes.to_dict().items()}


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
                DiagnosticsAmdarHarmonisationStrategyOptions.EDR,
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

    @property
    def latitude_index_column(self) -> str:
        return "lat_index"

    @property
    def longitude_index_column(self) -> str:
        return "lon_index"

    @property
    def vertical_coordinate_index_column(self) -> str:
        return "level_index"

    @property
    def harmonised_diagnostics(self) -> list[str]:
        return self._diagnostics_suite.diagnostic_names()

    def _create_diagnostic_value_series(
        self, grid_prototype: "xr.DataArray", observational_data: dd.DataFrame, dataframe_meta: dict[str, pd.Series]
    ) -> dict["DiagnosticName", dd.Series]:
        coordinate_axes: list[str] = ["longitude", "latitude", "pressure_level", "time"]
        assert set(coordinate_axes).issubset(grid_prototype.coords)
        axis_order: tuple[int, ...] = grid_prototype.get_axis_num(coordinate_axes)
        name_of_index_columns: list[str] = [
            self.longitude_index_column,
            self.latitude_index_column,
            self.vertical_coordinate_index_column,
            self.common_time_column_name,
        ]
        assert set(name_of_index_columns).issubset(observational_data)

        # Retrieves the index for each row of data and stores them as dask arrays
        indexing_columns: list[da.Array] = [
            # Linter thinks this is a pandas array. Using .values on a dask dataframe converts it to
            # a dask array. Therefore, ignore linter warning
            observational_data[col_name].values.compute_chunk_sizes().persist()  # noqa: PD011
            for col_name in name_of_index_columns
        ]
        # Order of coordinate axes are not known beforehand. Therefore, use the axis order so that the index
        # values matches the dimension of the array
        in_order_of_array_coords: list[da.Array] = map_order(indexing_columns, list(axis_order))
        # Combine them such that coordinates contains [(x1, y1, z1, t1), ..., (xn, yn, zn, tn)] which are the
        # values to slices from the computed diagnostic
        coordinates: da.Array = da.stack(in_order_of_array_coords, axis=1)  # shape = (4, n)
        # da.Array doesn't support np advanced indexing such that coordinates can be used directly
        #   => index into it as a contiguous array
        flattened_index: da.Array = da.ravel_multi_index(coordinates.T, grid_prototype.shape).persist()
        for name in self.harmonised_diagnostics:
            dataframe_meta[name] = pd.Series(dtype=float)

        return {
            diagnostic_name: da.ravel(computed_diagnostic.data)[flattened_index]
            # .compute_chunk_sizes()
            .to_dask_dataframe(
                meta=pd.DataFrame({diagnostic_name: pd.Series(dtype=float)}),
                index=observational_data.index,  # ensures it matches observational_data.index
                columns=diagnostic_name,
            )
            .persist()
            for diagnostic_name, computed_diagnostic in self._diagnostics_suite.computed_values(
                "Vectorised indexing to get diagnostic value for observation"
            )
        }

    def nearest_diagnostic_value(self, time_window: "Limits[np.datetime64]") -> "dd.DataFrame":
        grid_prototype: xr.DataArray = self._diagnostics_suite.get_prototype_computed_diagnostic()
        self._check_time_window_within_met_data(time_window, grid_prototype)

        observational_data: dd.DataFrame = self._amdar_data.clip_to_time_window(time_window).persist()
        dataframe_meta: dict[str, pd.Series] = get_dataframe_dtypes(observational_data)
        dataframe_meta["level_index"] = pd.Series(dtype=int)
        observational_data = observational_data.assign(
            level_index=observational_data["level"].apply(
                lambda row, pressure_level=grid_prototype["pressure_level"].values: np.abs(  # noqa: PD011
                    row - pressure_level
                ).argmin(),
                meta=("level_index", int),
            ),
        ).persist()
        # observational_data = observational_data.map_partitions(
        #     lambda df: df.assign(
        #         level_index=df["level"].apply(
        #             lambda row, pressure_level=grid_prototype["pressure_level"].values: np.abs(  # noqa: PD011
        #                 row - pressure_level
        #             ).argmin(),
        #         ),
        #         meta=pd.Series(dtype=int),
        #     ),
        #     meta=pd.DataFrame(dataframe_meta),
        # ).persist()
        dataframe_meta[self.latitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.longitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.common_time_column_name] = pd.Series(dtype=int)
        observational_data = observational_data.map_partitions(
            lambda df: df.assign(
                **{
                    self.latitude_index_column: map_values_to_nearest_coordinate_index(
                        df.latitude, grid_prototype["latitude"].values
                    ),
                    self.longitude_index_column: map_values_to_nearest_coordinate_index(
                        df.longitude, grid_prototype["longitude"].values
                    ),
                    self.common_time_column_name: map_values_to_nearest_coordinate_index(
                        df.datetime,
                        grid_prototype["time"].values,
                        valid_window=self.TIME_WINDOW_DELTA,
                    ),
                }
            ),
            meta=dd.from_pandas(pd.DataFrame(dataframe_meta), npartitions=observational_data.npartitions),
        ).persist()

        diagnostics_columns = self._create_diagnostic_value_series(grid_prototype, observational_data, dataframe_meta)
        for diagnostic_col_name, as_column in diagnostics_columns.items():
            observational_data = dd.map_partitions(
                lambda base_df, new_col, col_name=diagnostic_col_name: base_df.assign(**{col_name: new_col}),
                observational_data,
                as_column,
            ).persist()
        return observational_data

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

    def apply_condition(maxima: float) -> bool:
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


class SpatialGroupByStrategy(StrEnum):
    GRID_BOX = "grid_box"
    GRID_POINT = "grid_point"
    HORIZONTAL_BOX = "horizontal_box"
    HORIZONTAL_POINT = "horizontal_point"


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
            data: dd.DataFrame = self._data_harmoniser.nearest_diagnostic_value(self._time_window).persist()
            self._harmonised_data = data
            return self._harmonised_data
        return self._harmonised_data

    def _grid_spatial_columns(self, group_by_strategy: SpatialGroupByStrategy) -> list[str]:
        match group_by_strategy:
            case SpatialGroupByStrategy.GRID_BOX:
                return [
                    self._data_harmoniser.grid_box_column_name,
                    self._data_harmoniser.vertical_coordinate_index_column,
                ]
            case SpatialGroupByStrategy.GRID_POINT:
                return [
                    self._data_harmoniser.latitude_index_column,
                    self._data_harmoniser.longitude_index_column,
                    self._data_harmoniser.vertical_coordinate_index_column,
                ]
            case SpatialGroupByStrategy.HORIZONTAL_BOX:
                return [self._data_harmoniser.grid_box_column_name]
            case SpatialGroupByStrategy.HORIZONTAL_POINT:
                return [self._data_harmoniser.latitude_index_column, self._data_harmoniser.longitude_index_column]
            case _ as unreachable:
                assert_never(unreachable)

    # def _add_nearest_grid_indices(
    #     self,
    #     validation_columns: list[str],
    #     grid_prototype: "xr.DataArray",
    # ) -> "dd.DataFrame":
    #     if not set(validation_columns).issubset(self.data.columns):
    #         raise ValueError("Validation columns must be present in the data")
    #
    #     space_time_columns: list[str] = [
    #         self._data_harmoniser.common_time_column_name,
    #         "level",
    #         "longitude",
    #         "latitude",
    #     ]
    #     target_columns = space_time_columns + validation_columns
    #     target_data: dd.DataFrame = self.data[target_columns]
    #     dataframe_meta: dict[str, pd.Series] = get_dataframe_dtypes(target_data)
    #     dataframe_meta["level_index"] = pd.Series(dtype=int)
    #     target_data = target_data.map_partitions(
    #         lambda df: df.assign(
    #             level_index=df.apply(
    #                 lambda row, pressure_level=grid_prototype["pressure_level"].values: np.abs(  # noqa: PD011
    #                     row.level - pressure_level
    #                 ).argmin(),
    #                 axis=1,
    #             )
    #         ),
    #         meta=dataframe_meta,
    #     )
    #     dataframe_meta["lat_index"] = pd.Series(dtype=int)
    #     dataframe_meta["lon_index"] = pd.Series(dtype=int)
    #     return target_data.map_partitions(
    #         lambda df: df.assign(
    #             lat_index=map_values_to_nearest_coordinate_index(df.latitude, grid_prototype["latitude"].values),
    #             lon_index=map_values_to_nearest_coordinate_index(df.longitude, grid_prototype["longitude"].values),
    #         ),
    #         meta=dd.from_pandas(pd.DataFrame(dataframe_meta)),
    #     ).optimize()

    # def _spatio_temporal_data_aggregation(
    #     self,
    #     target_data: "dd.DataFrame",
    #     strategy_columns: list[str],
    #     validation_conditions: "list[DiagnosticValidationCondition]",
    # ) -> "dd.DataFrame":
    #     group_by_columns: list[str] = [
    #         "lat_index",
    #         "lon_index",
    #         "level_index",
    #         self._data_harmoniser.common_time_column_name,
    #     ]
    #     assert set(group_by_columns).issubset(target_data.columns)
    #     columns_to_drop: list[str] = ["level", "longitude", "latitude"]
    #     assert set(columns_to_drop).issubset(target_data.columns)
    #     target_data = target_data.drop(columns=columns_to_drop)
    #     grouped_by_space_time = target_data.groupby(group_by_columns)
    #
    #     aggregation_spec: dict = {
    #         condition.observed_turbulence_column_name: _observed_turbulence_aggregation(condition)
    #         for condition in validation_conditions
    #     }
    #     for strategy_column in strategy_columns:
    #         aggregation_spec[strategy_column] = "mean"
    #     return grouped_by_space_time.aggregate(aggregation_spec)

    @staticmethod
    def _get_partition_level_values(partition: pd.Index, level_name: str) -> pd.Index:
        return partition.get_level_values(level_name)

    @staticmethod
    def _concat_columns_as_str(data_frame: dd.DataFrame, column_names: list[str], separator: str = "_") -> dd.Series:
        multi_index_columns = [data_frame[col_name].astype(str) for col_name in column_names]
        return functools.reduce(lambda left, right: left + separator + right, multi_index_columns)

    def _retrieve_grouped_columns(
        self, data_frame: dd.DataFrame, grouped_columns: list[str], trigger_reset_index: bool
    ) -> dd.DataFrame:
        for grouped_column in grouped_columns:
            data_frame[grouped_column] = data_frame.index.map_partitions(
                self._get_partition_level_values, grouped_column
            )

        if self._data_harmoniser.grid_box_column_name in set(grouped_columns) and trigger_reset_index:
            # set_index is an expensive operation due to the shuffles it triggers
            #   However, this cost has been undertaken as it means that the data can be joined INTO a GeoPandas
            #   grid turning the entire thing into a GeoDataFrame without having to invoke dgpd.from_dask_dataframe.
            #   It also means that the crs will be inherited and not require manual intervention
            data_frame = data_frame.set_index(data_frame[self._data_harmoniser.grid_box_column_name], drop=True)
        elif trigger_reset_index:
            data_frame = data_frame.reset_index(drop=True)

        return data_frame

    def _retrieve_index_column_values(
        self,
        data_frame: dd.DataFrame,
        grouped_columns: list[str],
        prototype_array: "xr.DataArray",
        drop_index_cols: bool,
    ) -> dd.DataFrame:
        assert {"pressure_level", "longitude", "latitude"}.issubset(prototype_array.coords)
        index_columns: list[str] = list(
            {
                self._data_harmoniser.longitude_index_column,
                self._data_harmoniser.latitude_index_column,
                self._data_harmoniser.vertical_coordinate_index_column,
            }.intersection(grouped_columns)
        )
        if not index_columns:  # set is empty
            return data_frame

        col_name_mapping: dict[str, str] = {
            self._data_harmoniser.longitude_index_column: "longitude",
            self._data_harmoniser.latitude_index_column: "latitude",
            self._data_harmoniser.vertical_coordinate_index_column: "pressure_level",
        }

        data_frame_dtypes = get_dataframe_dtypes(data_frame)
        for index_col_name in index_columns:
            value_col_name = col_name_mapping[index_col_name]
            data_frame_dtypes[value_col_name] = pd.Series(dtype=float)

        data_frame = data_frame.map_partitions(
            lambda df: df.assign(
                **{
                    col_name_mapping[col_name]: map_index_to_coordinate_value(
                        df[col_name], prototype_array[col_name_mapping[col_name]].to_numpy()
                    )
                    for col_name in index_columns
                }
            ),
            meta=pd.DataFrame(data_frame_dtypes),
        )

        if drop_index_cols:
            return data_frame.drop(columns=index_columns)

        return data_frame

    def num_obs_per(
        self, validation_conditions: "list[DiagnosticValidationCondition]", group_by_strategy: SpatialGroupByStrategy
    ) -> dd.DataFrame:
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)
        turbulence_col: str = validation_conditions[0].observed_turbulence_column_name
        groupby_columns: list[str] = self._grid_spatial_columns(group_by_strategy)
        minimum_columns: list[str] = groupby_columns + [turbulence_col]
        num_obs = target_data[minimum_columns].groupby(groupby_columns).count()
        num_obs = self._retrieve_grouped_columns(num_obs, groupby_columns, True)
        return num_obs.rename(columns={turbulence_col: "num_obs"})

    def _spatial_data_grouping(
        self, validation_conditions: "list[DiagnosticValidationCondition]", group_by_strategy: SpatialGroupByStrategy
    ) -> dd.DataFrame:
        target_data: dd.DataFrame = self.data
        space_columns = self._grid_spatial_columns(group_by_strategy)
        validation_columns = self._get_validation_column_names(validation_conditions)
        target_cols = space_columns + validation_columns + self._data_harmoniser.harmonised_diagnostics
        target_data = target_data[target_cols]

        # 1) Use the columns that will be used to spatially group the data as the index of the dataframe
        #    Without a column as the index, the aggregation to compute AUC fails with,
        #       ValueError: If using all scalar values, you must pass an index
        target_data["multi_index"] = self._concat_columns_as_str(target_data, space_columns)

        # Dask doesn't support multi-index so we've got to use a "poor man's" multi-index
        return target_data.set_index("multi_index").persist()

    def aggregate_by_auc(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        prototype_diagnostic: "xr.DataArray",
        group_by_strategy: SpatialGroupByStrategy,
    ) -> dict[str, dd.DataFrame]:
        space_columns = self._grid_spatial_columns(group_by_strategy)
        validation_columns = self._get_validation_column_names(validation_conditions)
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: condition.grid_point_auc_agg()
            for condition in validation_conditions
        }
        auc_by_diagnostic: dict[str, dd.DataFrame] = {}
        for diagnostic_name in self._data_harmoniser.harmonised_diagnostics:
            columns_for_diagnostic: list[str] = space_columns + [diagnostic_name] + validation_columns
            data_for_diagnostic: dd.DataFrame = target_data[columns_for_diagnostic]
            # 2) Sort values so that diagnostic is in descending order as required by the ROC calculation
            #    The set_index() operation performs a sort => need to do sort after and for each diagnostic
            data_for_diagnostic = data_for_diagnostic.sort_values(by=diagnostic_name, ascending=False)
            data_for_diagnostic = data_for_diagnostic.drop(columns=[diagnostic_name])

            # Specify sort=False to get better performance
            # See https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.groupby.html#dask-dataframe-dataframe-groupby
            grid_point_auc: dd.DataFrame = data_for_diagnostic.groupby(space_columns, sort=False).aggregate(
                aggregation_spec, meta={col: pd.Series(dtype=float) for col in validation_columns}
            )

            grid_point_auc = self._retrieve_grouped_columns(grid_point_auc, space_columns, True)
            grid_point_auc = self._retrieve_index_column_values(
                grid_point_auc, space_columns, prototype_diagnostic, True
            )
            auc_by_diagnostic[diagnostic_name] = grid_point_auc.persist()

        return auc_by_diagnostic

    def nearest_value_roc(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
    ) -> RocVerificationResult:
        turbulence_diagnostics = self._data_harmoniser.harmonised_diagnostics
        target_data = self.data
        validation_columns = self._get_validation_column_names(validation_conditions)
        space_time_columns: list[str] = [
            # self._data_harmoniser.grid_box_column_name,
            "lat_index",
            "lon_index",
            "level_index",
            self._data_harmoniser.common_time_column_name,
        ]
        target_columns: list[str] = space_time_columns + turbulence_diagnostics + validation_columns
        target_data = target_data[target_columns]
        grouped_by_space_time = target_data.groupby(space_time_columns)
        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _observed_turbulence_aggregation(condition)
            for condition in validation_conditions
        }
        for diagnostic_column in turbulence_diagnostics:
            aggregation_spec[diagnostic_column] = "mean"
        aggregated_data = grouped_by_space_time.aggregate(aggregation_spec).persist()

        result: defaultdict[str, dict[str, BinaryClassificationResult]] = defaultdict(dict)
        for diagnostic_val_col in turbulence_diagnostics:
            # descending values
            subset_df = (
                aggregated_data[[*validation_columns, diagnostic_val_col]]
                .sort_values(diagnostic_val_col, ascending=False)
                .persist()
            )
            for amdar_turbulence_col in validation_columns:
                result[amdar_turbulence_col][diagnostic_val_col] = received_operating_characteristic(
                    subset_df[amdar_turbulence_col].values.compute_chunk_sizes().persist(),  # noqa: PD011
                    subset_df[diagnostic_val_col].values.compute_chunk_sizes().persist(),  # noqa: PD011
                    num_intervals=-1,
                )
        return RocVerificationResult(dict(result))

    @staticmethod
    def _get_validation_column_names(validation_conditions: "list[DiagnosticValidationCondition]") -> list[str]:
        return [condition.observed_turbulence_column_name for condition in validation_conditions]

    # def compute_roc_curve(
    #     self,
    #     validation_conditions: "list[DiagnosticValidationCondition]",
    #     prototype_diagnostic_array: "xr.DataArray",
    # ) -> RocVerificationResult:
    #     assert {"pressure_level", "longitude", "latitude", "time"}.issubset(prototype_diagnostic_array.coords)
    #     diagnostic_value_columns: list[str] = list(
    #         self._data_harmoniser.strategy_values_columns(
    #             [DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES]
    #         )
    #     )
    #     validation_columns = self._get_validation_column_names(validation_conditions)
    #     target_data = self._add_nearest_grid_indices(
    #         validation_columns + diagnostic_value_columns,
    #         prototype_diagnostic_array,
    #     )
    #     logger.debug("Added required columns for spatio temporal aggregating")
    #     aggregated_data = self._spatio_temporal_data_aggregation(
    #         target_data, diagnostic_value_columns, validation_conditions
    #     ).persist()
    #     logger.debug("Triggered spatio temporal aggregation")
    #     logger.debug("Finished spatio temporal aggregation")
    #     result: defaultdict[str, dict[str, BinaryClassificationResult]] = defaultdict(dict)
    #     for diagnostic_val_col in diagnostic_value_columns:
    #         # descending values
    #         subset_df = aggregated_data[[*validation_columns, diagnostic_val_col]].sort_values(
    #             diagnostic_val_col, ascending=False
    #         )
    #         for amdar_turbulence_col in validation_columns:
    #             result[amdar_turbulence_col][diagnostic_val_col] = received_operating_characteristic(
    #                 subset_df[amdar_turbulence_col].values.compute_chunk_sizes(),  # noqa: PD011
    #                 subset_df[diagnostic_val_col].values.compute_chunk_sizes(),  # noqa: PD011
    #                 num_intervals=-1,
    #             )
    #     logger.debug("Finished computing ROC curves")
    #
    #     return RocVerificationResult(dict(result))
