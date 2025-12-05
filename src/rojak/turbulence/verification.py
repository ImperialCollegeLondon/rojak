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
from collections import defaultdict
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, ClassVar, Literal, assert_never, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from scipy import integrate

from rojak.core.indexing import (
    map_index_to_coordinate_value,
    map_order,
    map_values_to_nearest_coordinate_index,
    map_values_to_nearest_index_irregular_grid,
)
from rojak.orchestrator.configuration import (
    AggregationMetricOption,
    SpatialGroupByStrategy,
)
from rojak.turbulence.metrics import (
    BinaryClassificationResult,
    area_under_curve,
    binary_classification_rate_from_cumsum,
    received_operating_characteristic,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.orchestrator.configuration import DiagnosticValidationCondition
    from rojak.turbulence.diagnostic import DiagnosticSuite
    from rojak.turbulence.metrics import BinaryClassificationRateFromLabels
    from rojak.utilities.types import DiagnosticName, Limits


logger = logging.getLogger(__name__)


class NotWithinTimeFrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


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

    @staticmethod
    def _check_time_window_within_met_data(
        time_window: "Limits[np.datetime64]",
        first_diagnostic: "xr.DataArray",
    ) -> None:
        time_coordinate: xr.DataArray = first_diagnostic["time"]
        if time_window.lower < time_coordinate.min() or time_window.upper > time_coordinate.max():
            raise NotWithinTimeFrameError("Time window is not within time coordinate of met data")

    def _get_grid_prototype(self, time_window: "Limits[np.datetime64]") -> "xr.DataArray":
        grid_prototype: xr.DataArray = self._diagnostics_suite.get_prototype_computed_diagnostic()
        self._check_time_window_within_met_data(time_window, grid_prototype)
        return grid_prototype

    def _get_observational_data(self, time_window: "Limits[np.datetime64]") -> dd.DataFrame:
        return self._amdar_data.clip_to_time_window(time_window).persist()

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
    def coordinate_axes(self) -> list[str]:
        return ["longitude", "latitude", "pressure_level", "time"]

    @property
    def index_column_names(self) -> list[str]:
        return [
            self.longitude_index_column,
            self.latitude_index_column,
            self.vertical_coordinate_index_column,
            self.common_time_column_name,
        ]

    def coordinate_axes_to_index_col_name(self) -> dict[str, str]:
        return dict(zip(self.coordinate_axes, self.index_column_names, strict=True))

    @property
    def harmonised_diagnostics(self) -> list[str]:
        return self._diagnostics_suite.diagnostic_names()

    def _get_gridded_coordinates(
        self, observational_data: dd.DataFrame, grid_prototype: xr.DataArray, stack_on_axis: int
    ) -> da.Array:
        assert grid_prototype.ndim == len(self.coordinate_axes), (
            "Grid prototype must have same number of dimensions as defined coordinates"
        )
        # Retrieves the index for each row of data and stores them as dask arrays
        indexing_columns: list[da.Array] = [
            self._coordinates_of_observations(observational_data, grid_prototype)[
                self.coordinate_axes_to_index_col_name()[coord_name]
            ]
            for coord_name in self.coordinate_axes
        ]

        # Get order of the coordinate axes of the prototype diagnostic
        assert set(self.coordinate_axes).issubset(grid_prototype.coords)
        axis_order: list[int] = list(grid_prototype.get_axis_num(self.coordinate_axes))

        # Order of coordinate axes are not known beforehand. Therefore, use the axis order so that the index
        # values matches the dimension of the array
        in_order_of_array_coords: list[da.Array] = map_order(indexing_columns, axis_order)

        # Combine the coordinates on the specified axis
        return da.stack(in_order_of_array_coords, axis=stack_on_axis)

    def _create_diagnostic_value_series(
        self,
        grid_prototype: "xr.DataArray",
        observational_data: dd.DataFrame,
        dataframe_meta: dict[str, pd.Series],
    ) -> dict["DiagnosticName", dd.Series]:
        # Combine them such that coordinates contains [(x1, y1, z1, t1), ..., (xn, yn, zn, tn)] which are the
        # values to slices from the computed diagnostic
        coordinates: da.Array = self._get_gridded_coordinates(observational_data, grid_prototype, stack_on_axis=1)
        assert coordinates.shape[1] == grid_prototype.ndim, "Shape of coordinates should be (nnz, ndim)"

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
                "Vectorised indexing to get diagnostic value for observation",
            )
        }

    def nearest_diagnostic_value(self, time_window: "Limits[np.datetime64]") -> "dd.DataFrame":
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        dataframe_meta: dict[str, pd.Series] = get_dataframe_dtypes(observational_data)
        dataframe_meta["level_index"] = pd.Series(dtype=int)
        dataframe_meta[self.latitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.longitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.common_time_column_name] = pd.Series(dtype=int)

        observational_data = observational_data.assign(
            level_index=map_values_to_nearest_index_irregular_grid(
                observational_data["level"], grid_prototype["pressure_level"].values
            )
        ).persist()
        observational_data = observational_data.map_partitions(
            lambda df: df.assign(
                **{
                    self.latitude_index_column: map_values_to_nearest_coordinate_index(
                        df.latitude,
                        grid_prototype["latitude"].values,
                    ),
                    self.longitude_index_column: map_values_to_nearest_coordinate_index(
                        df.longitude,
                        grid_prototype["longitude"].values,
                    ),
                    self.common_time_column_name: map_values_to_nearest_coordinate_index(
                        df.datetime,
                        grid_prototype["time"].values,
                        valid_window=self.TIME_WINDOW_DELTA,
                    ),
                },
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

    def _coordinates_of_observations(
        self, observational_data: dd.DataFrame, grid_prototype: "xr.DataArray"
    ) -> dict[str, da.Array]:
        level_index: da.Array = cast(
            "da.Array",
            map_values_to_nearest_index_irregular_grid(
                observational_data["level"], grid_prototype["pressure_level"].values
            ),
        )
        latitude_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.latitude,
                grid_prototype["latitude"].values,
            ),
            # Specifying meta this way resolves this error,
            #   AttributeError: 'DataFrame' object has no attribute 'name'
            meta=(self.latitude_index_column, int),
        ).to_dask_array(lengths=True)
        longitude_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.longitude,
                grid_prototype["longitude"].values,
            ),
            meta=(self.longitude_index_column, int),
        ).to_dask_array(lengths=True)
        time_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.datetime,
                grid_prototype["time"].values,
                valid_window=self.TIME_WINDOW_DELTA,
            ),
            meta=(self.common_time_column_name, int),
        ).to_dask_array(lengths=True)
        return {
            self.vertical_coordinate_index_column: level_index,
            self.latitude_index_column: latitude_index,
            self.longitude_index_column: longitude_index,
            self.common_time_column_name: time_index,
        }

    def grid_turbulence_observation_frequency(
        self, time_window: "Limits[np.datetime64]", positive_obs_condition: "list[DiagnosticValidationCondition]"
    ) -> "xr.Dataset":
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        # Coords for sparse COO must be in (ndim, nnz)
        gridded_coordinates: da.Array = self._get_gridded_coordinates(
            observational_data, grid_prototype, stack_on_axis=0
        )  # shape = (4, n)
        assert gridded_coordinates.shape[0] == grid_prototype.ndim, (
            "Shape of gridded coordinates must be in (ndim, nnz) for sparse COO matrix"
        )

        as_coo: dict[str, da.Array] = {
            condition.observed_turbulence_column_name: da.map_blocks(
                lambda coords_, was_observed, output_shape: sparse.COO(
                    coords=coords_, data=was_observed, shape=output_shape
                ),
                gridded_coordinates,
                (
                    observational_data[condition.observed_turbulence_column_name] > condition.value_greater_than
                ).to_dask_array(lengths=True),
                grid_prototype.shape,
                # Returns a count of the number of times turbulence was observed at each grid point
                dtype=int,
                # Arrays being mapped over have shape (4, n), both axes need to be dropped as the shape of the
                # resulting array needs to match (n_lon, n_lat, n_level, n_time)
                drop_axis=[0, 1],
                # Make the chunking of the resulting array must be the same as the diagnostic
                chunks=grid_prototype.chunks,
            )
            for condition in positive_obs_condition
        }

        return xr.Dataset(
            data_vars={
                condition_name: xr.DataArray(
                    data=regridded,
                    coords=grid_prototype.coords,
                    dims=grid_prototype.dims,
                    name=f"{condition_name}_count",
                )
                for condition_name, regridded in as_coo.items()
            },
            coords=grid_prototype.coords,
        )

    def grid_total_observations_count(self, time_window: "Limits[np.datetime64]") -> xr.DataArray:
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        # Coords for sparse COO must be in (ndim, nnz)
        gridded_coordinates: da.Array = self._get_gridded_coordinates(
            observational_data, grid_prototype, stack_on_axis=0
        )  # shape = (4, n)
        assert gridded_coordinates.shape[0] == grid_prototype.ndim, (
            "Shape of gridded coordinates must be in (ndim, nnz) for sparse COO matrix"
        )

        return xr.DataArray(
            data=da.map_blocks(
                lambda coords_, count, output_shape: sparse.COO(coords=coords_, data=count, shape=output_shape),
                gridded_coordinates,
                # Chunks of gridded coords is in the shape of ((chunks_for_ndim), (chunks_for_length))
                # So, for the ones array, use chunks of length dimension
                da.ones(gridded_coordinates.shape[1], dtype=int, chunks=gridded_coordinates.chunks[1]),
                grid_prototype.shape,
                dtype=int,
                drop_axis=[0, 1],
                chunks=grid_prototype.chunks,
            ),
            coords=grid_prototype.coords,
            dims=grid_prototype.dims,
            name="number_of_observations",
        )


def _any_aggregation() -> dd.Aggregation:
    return dd.Aggregation(
        name="aggregate_by_any",
        chunk=lambda within_partition: within_partition.any(),
        agg=lambda across_partitions: across_partitions.any(),
    )


def metric_based_aggregation(
    condition: "DiagnosticValidationCondition",
    minimum_group_size: int,
    aggregation_function: AggregationMetricOption,
    integration_scheme: Literal["trapz", "simpson"] = "simpson",
) -> dd.Aggregation:
    assert minimum_group_size > 0, "Minimum group size must be positive"

    def apply_condition(on_chunk: "pd.api.typing.SeriesGroupBy") -> "pd.Series":
        return on_chunk.apply(lambda row: row > condition.value_greater_than)

    def aggregate_with_cumsum(on_partition: "pd.api.typing.SeriesGroupBy") -> "pd.Series":
        return on_partition.cumsum()

    def prevalence_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        return float(on_group.to_numpy()[-1] / group_size)

    def tss_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        binary_classification: BinaryClassificationRateFromLabels | None = binary_classification_rate_from_cumsum(
            on_group,
        )
        if binary_classification is None:
            return -1

        tss: NDArray = binary_classification.true_positives_rate + binary_classification.false_positives_rate - 1
        return tss.max()

    def auc_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        binary_classification: BinaryClassificationRateFromLabels | None = binary_classification_rate_from_cumsum(
            on_group,
        )
        if binary_classification is None:
            return -1

        if integration_scheme == "trapz":
            return float(
                np.trapezoid(binary_classification.true_positives_rate, x=binary_classification.false_positives_rate),
            )
        return float(
            integrate.simpson(binary_classification.true_positives_rate, x=binary_classification.false_positives_rate),
        )

    finalisers: dict[AggregationMetricOption, Callable[[pd.Series], float]] = {
        AggregationMetricOption.AUC: auc_from_cumsum,
        AggregationMetricOption.TSS: tss_from_cumsum,
        AggregationMetricOption.PREVALENCE: prevalence_from_cumsum,
    }

    return dd.Aggregation(
        name=f"{aggregation_function}_on_{condition.observed_turbulence_column_name}_{condition.value_greater_than:0.2f}",
        chunk=apply_condition,
        agg=aggregate_with_cumsum,
        # See https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
        # The example for nunique has this groupby
        # My understanding is that it forces the partition into the original groups, on which, I am applying
        # the cumsum => result is still separated based on the groups
        finalize=lambda on_aggregation_result: on_aggregation_result.groupby(
            level=list(range(on_aggregation_result.index.nlevels)),
        ).apply(finalisers[aggregation_function]),
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
                        roc_results.false_positives,
                        roc_results.true_positives,
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

    @staticmethod
    def _get_partition_level_values(partition: pd.Index, level_name: str) -> pd.Index:
        return partition.get_level_values(level_name)

    @staticmethod
    def _concat_columns_as_str(data_frame: dd.DataFrame, column_names: list[str], separator: str = "_") -> dd.Series:
        multi_index_columns = [data_frame[col_name].astype(str) for col_name in column_names]
        return functools.reduce(lambda left, right: left + separator + right, multi_index_columns)

    def _retrieve_grouped_columns(
        self,
        data_frame: dd.DataFrame,
        grouped_columns: list[str],
        trigger_reset_index: bool,
    ) -> dd.DataFrame:
        data_frame = data_frame.map_partitions(
            lambda df: df.assign(
                **{
                    grouped_column: self._get_partition_level_values(df.index, grouped_column)
                    for grouped_column in grouped_columns
                },
            ),
        )

        if self._data_harmoniser.grid_box_column_name in set(grouped_columns) and trigger_reset_index:
            # set_index is an expensive operation due to the shuffles it triggers
            #   However, this cost has been undertaken as it means that the data can be joined INTO a GeoPandas
            #   grid turning the entire thing into a GeoDataFrame without having to invoke dgpd.from_dask_dataframe.
            #   It also means that the crs will be inherited and not require manual intervention
            data_frame = data_frame.set_index(data_frame[self._data_harmoniser.grid_box_column_name], drop=True)
        elif trigger_reset_index:
            data_frame = data_frame.reset_index(drop=True)

        return data_frame.persist()

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
            }.intersection(grouped_columns),
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
                        df[col_name],
                        prototype_array[col_name_mapping[col_name]].to_numpy(),
                    )
                    for col_name in index_columns
                },
            ),
            meta=pd.DataFrame(data_frame_dtypes),
        )

        if drop_index_cols:
            return data_frame.drop(columns=index_columns)

        return data_frame

    def num_obs_per(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        group_by_strategy: SpatialGroupByStrategy,
    ) -> dd.DataFrame:
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)
        turbulence_col: str = validation_conditions[0].observed_turbulence_column_name
        groupby_columns: list[str] = self._grid_spatial_columns(group_by_strategy)
        minimum_columns: list[str] = [*groupby_columns, turbulence_col]
        num_obs = target_data[minimum_columns].groupby(groupby_columns).count()
        num_obs = self._retrieve_grouped_columns(num_obs, groupby_columns, True)
        return num_obs.rename(columns={turbulence_col: "num_obs"})

    def _spatial_data_grouping(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        group_by_strategy: SpatialGroupByStrategy,
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
        minimum_group_size: int,
        aggregation_metric: AggregationMetricOption,
    ) -> dict[str, dd.DataFrame]:
        space_columns = self._grid_spatial_columns(group_by_strategy)
        validation_columns = self._get_validation_column_names(validation_conditions)
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: metric_based_aggregation(
                condition,
                minimum_group_size,
                aggregation_metric,
            )
            for condition in validation_conditions
        }
        auc_by_diagnostic: dict[str, dd.DataFrame] = {}
        for diagnostic_name in self._data_harmoniser.harmonised_diagnostics:
            columns_for_diagnostic: list[str] = [*space_columns, diagnostic_name, *validation_columns]
            data_for_diagnostic: dd.DataFrame = target_data[columns_for_diagnostic]
            # 2) Sort values so that diagnostic is in descending order as required by the ROC calculation
            #    The set_index() operation performs a sort => need to do sort after and for each diagnostic
            data_for_diagnostic = data_for_diagnostic.sort_values(by=diagnostic_name, ascending=False)
            data_for_diagnostic = data_for_diagnostic.drop(columns=[diagnostic_name])

            # Specify sort=False to get better performance
            # See https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.groupby.html#dask-dataframe-dataframe-groupby
            grid_point_auc: dd.DataFrame = data_for_diagnostic.groupby(space_columns, sort=False).aggregate(
                aggregation_spec,
                meta={col: pd.Series(dtype=float) for col in validation_columns},
            )
            # 3) Drop values which do not meet the minimum number of observations
            grid_point_auc = grid_point_auc.dropna()
            # 4) Make values < 0 (another form of invalid values, e.g. no positive observations) into NaNs
            grid_point_auc = grid_point_auc.where(grid_point_auc >= 0, other=np.nan).persist()  # pyright: ignore [reportOperatorIssue]

            grid_point_auc = self._retrieve_grouped_columns(grid_point_auc, space_columns, True)
            grid_point_auc = self._retrieve_index_column_values(
                grid_point_auc,
                space_columns,
                prototype_diagnostic,
                True,
            )
            auc_by_diagnostic[diagnostic_name] = grid_point_auc.persist()

        return auc_by_diagnostic

    def nearest_value_roc(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
    ) -> RocVerificationResult:
        turbulence_diagnostics = self._data_harmoniser.harmonised_diagnostics
        validation_columns = self._get_validation_column_names(validation_conditions)
        space_time_columns: list[str] = [
            # self._data_harmoniser.grid_box_column_name,
            "lat_index",
            "lon_index",
            "level_index",
            self._data_harmoniser.common_time_column_name,
        ]
        target_columns: list[str] = space_time_columns + turbulence_diagnostics + validation_columns
        target_data = self.data[target_columns]

        # Apply condition so that any can be used at the groupby stage
        for condition in validation_conditions:
            target_data[condition.observed_turbulence_column_name] = (
                target_data[condition.observed_turbulence_column_name] > condition.value_greater_than
            )
        grouped_by_space_time = target_data.groupby(space_time_columns)
        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _any_aggregation() for condition in validation_conditions
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
            diagnostic_col_as_array: da.Array = subset_df[diagnostic_val_col].to_dask_array(lengths=True).persist()
            for amdar_turbulence_col in validation_columns:
                result[amdar_turbulence_col][diagnostic_val_col] = received_operating_characteristic(
                    subset_df[amdar_turbulence_col].to_dask_array(lengths=diagnostic_col_as_array.chunks[0]).persist(),
                    diagnostic_col_as_array,
                    num_intervals=-1,
                )
        return RocVerificationResult(dict(result))

    @staticmethod
    def _get_validation_column_names(validation_conditions: "list[DiagnosticValidationCondition]") -> list[str]:
        return [condition.observed_turbulence_column_name for condition in validation_conditions]
