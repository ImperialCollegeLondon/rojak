from typing import TYPE_CHECKING

import dask.dataframe as dd
import numpy as np

from rojak.core.distributed_tools import blocking_wait_futures
from rojak.core.indexing import map_values_to_nearest_coordinate_index
from rojak.orchestrator.mediators import (
    DiagnosticsAmdarHarmonisationStrategyOptions,
)
from rojak.turbulence.metrics import BinaryClassificationResult, received_operating_characteristic

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

    def compute_roc_curve(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        prototype_diagnostic_array: "xr.DataArray",
    ) -> dict[str, dict[str, BinaryClassificationResult]]:
        assert {"pressure_level", "longitude", "latitude", "time"}.issubset(prototype_diagnostic_array.coords)
        strategy_columns: list[str] = list(
            self._data_harmoniser.strategy_values_columns(
                [DiagnosticsAmdarHarmonisationStrategyOptions.RAW_INDEX_VALUES]
            )
        )
        validation_columns: list[str] = [
            condition.observed_turbulence_column_name for condition in validation_conditions
        ]
        target_columns = validation_columns + strategy_columns
        target_data = self._add_nearest_grid_indices(
            target_columns,
            prototype_diagnostic_array,
        )
        group_by_columns: list[str] = [
            "lat_index",
            "lon_index",
            "level_index",
            self._data_harmoniser.common_time_column_name,
        ]
        target_columns = group_by_columns + validation_columns + strategy_columns
        grouped_by_space_time = target_data[target_columns].groupby(group_by_columns)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _observed_turbulence_aggregation(condition)
            for condition in validation_conditions
        }
        for strategy_column in strategy_columns:
            aggregation_spec[strategy_column] = "mean"

        aggregated_data = grouped_by_space_time.aggregate(aggregation_spec).persist()
        blocking_wait_futures(aggregated_data)

        result: dict[str, dict[str, BinaryClassificationResult]] = {}
        for strategy_column in strategy_columns:
            # descending values
            result[strategy_column] = {}
            subset_df = aggregated_data[[*validation_columns, strategy_column]].sort_values(
                strategy_column, ascending=False
            )
            for column in validation_columns:
                result[strategy_column][column] = received_operating_characteristic(
                    subset_df[column].values.compute_chunk_sizes(),  # noqa: PD011
                    subset_df[strategy_column].values.compute_chunk_sizes(),  # noqa: PD011
                )

        return result
