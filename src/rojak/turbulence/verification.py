import functools
from typing import TYPE_CHECKING

import dask.array as da
import dask.dataframe as dd
import numpy as np
import sparse

from rojak.core.distributed_tools import blocking_wait_futures
from rojak.core.indexing import map_values_to_nearest_coordinate_index
from rojak.orchestrator.mediators import (
    DiagnosticsAmdarHarmonisationStrategyOptions,
)

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from rojak.orchestrator.configuration import DiagnosticValidationCondition
    from rojak.orchestrator.mediators import (
        DiagnosticsAmdarDataHarmoniser,
    )
    from rojak.utilities.types import Limits


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

    def _add_nearest_grid_indices(
        self,
        validation_columns: list[str],
        grid_prototype: "xr.DataArray",
    ) -> "dd.DataFrame":
        space_time_columns: list[str] = [
            self._data_harmoniser.common_time_column_name,
            "level",
            "longitude",
            "latitude",
        ]
        target_columns = space_time_columns + validation_columns
        target_data: "dd.DataFrame" = self.data[target_columns]
        target_data = target_data.map_partitions(
            lambda df: df.assign(
                level_index=df.apply(
                    lambda row, pressure_level=grid_prototype["pressure_level"].values: np.abs(  # noqa: PD011
                        row.level - pressure_level
                    ).argmin(),
                    axis=1,
                )
            )
        )
        return target_data.map_partitions(
            lambda df: df.assign(
                lat_index=map_values_to_nearest_coordinate_index(df.latitude, grid_prototype["latitude"].values),
                lon_index=map_values_to_nearest_coordinate_index(df.longitude, grid_prototype["longitude"].values),
            )
        )

    def _spatio_temporal_data_aggregation(
        self,
        target_data: "dd.DataFrame",
        validation_columns: list[str],
        validation_conditions: "list[DiagnosticValidationCondition]",
    ) -> "dd.DataFrame":
        group_by_columns: list[str] = [
            "lat_index",
            "lon_index",
            "level_index",
            self._data_harmoniser.common_time_column_name,
        ]
        target_columns = group_by_columns + validation_columns
        grouped_by_space_time = target_data[target_columns].groupby(group_by_columns)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _observed_turbulence_aggregation(condition)
            for condition in validation_conditions
        }
        return grouped_by_space_time.aggregate(aggregation_spec)

    def use_sparse(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        prototype_diagnostic_array: "xr.DataArray",
        do_persist: bool = True,
    ) -> list[da.Array]:
        # assert len(set(validation_conditions)) == len(validation_conditions), (
        #     "Validation conditions must be unique to ensure name of dd.Aggregation is unique to prevent
        #     data corruption"
        # )
        assert {"pressure_level", "longitude", "latitude", "time"}.issubset(prototype_diagnostic_array.coords)
        validation_columns: list[str] = [
            condition.observed_turbulence_column_name for condition in validation_conditions
        ]
        target_data = self._add_nearest_grid_indices(validation_columns, prototype_diagnostic_array)
        aggregated_data = self._spatio_temporal_data_aggregation(target_data, validation_columns, validation_conditions)

        # Cannot implement as values.map_blocks as the shape changes so it cannot be broadcasted to the new shape
        # coordinate_values: "da.Array" = aggregated_data.index.values.map_blocks(lambda x: np.asarray(list(x)))
        coordinate_values: "da.Array" = da.map_blocks(lambda x: np.stack(x), aggregated_data.index.values)

        def combine_function(args: list, **kwargs: dict) -> da.Array:
            if len(args) == 2:  # noqa: PLR2004
                return sparse.elemwise(np.logical_or, args[0], args[1])

            out = args[0]
            for item in args[1:]:
                out = sparse.elemwise(np.logical_or, out, item)
            return out

        shape_of_output = (
            prototype_diagnostic_array["latitude"].size,
            prototype_diagnostic_array["longitude"].size,
            prototype_diagnostic_array["pressure_level"].size,
            prototype_diagnostic_array["time"].size,
        )
        result = []
        for condition in validation_conditions:
            was_observed: "da.Array" = aggregated_data[condition.observed_turbulence_column_name].values  # noqa: PD011
            as_sparse_array: "da.Array" = da.map_blocks(
                functools.partial(
                    lambda coord, data, output_shape: sparse.COO(coord.T, data=data, shape=output_shape),
                    output_shape=shape_of_output,
                ),
                coordinate_values,  # coords must be (ndim, nnz)
                was_observed,
                dtype=bool,
                meta=sparse.COO((), shape=shape_of_output),
            )
            # as_sparse_array: "da.Array" = da.map_blocks(
            #     lambda coord, data: sparse.COO(coord.T, data=data, shape=shape_of_output, fill_value=np.nan),
            #     coordinate_values,
            #     was_observed,
            #     dtype=bool,
            # )
            as_sparse_array = da.reduction(
                as_sparse_array,
                lambda x, axis, keepdims: x,
                lambda y, axis, keepdims: sparse.elemwise(np.logical_or, *y),
                # combine=lambda y, axis, keepdims: sparse.elemwise(np.logical_or, *y),
                combine=combine_function,
                # axis=0,
                concatenate=False,
                dtype=bool,
                meta=sparse.COO((), shape=shape_of_output, fill_value=np.nan),
            )
            if do_persist:
                as_sparse_array = as_sparse_array.persist()
                blocking_wait_futures(as_sparse_array)
            result.append(as_sparse_array)

        return result
