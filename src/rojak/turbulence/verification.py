import functools
from typing import TYPE_CHECKING, Callable, NamedTuple

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


class BinaryClassificationResult(NamedTuple):
    false_positives: "da.Array"
    true_positives: "da.Array"
    thresholds: "da.Array"


# Modified from scikit-learn metric roc's invocation of binary classification method
# https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L826
def binary_classification_curve(
    sorted_truth: "da.Array",
    sorted_values: "da.Array",
    num_intervals: int = 100,
    positive_classification_label: int | float | bool | str | None = None,
) -> BinaryClassificationResult:
    """
    Received operating characteristic or ROC curve

    Args:
        sorted_truth:
        sorted_values:
        num_intervals:
        positive_classification_label:

    Returns:
        BinaryClassificationResult: A named tuple with the false positive, true positive, and thresholds

    Modifying the example in the scikit-learn documentation:
    >>> import dask.array as da
    >>> y = np.asarray([1, 1, 2, 2])
    >>> scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    >>> binary_classification_curve(da.asarray(y), da.asarray(scores), positive_classification_label=2)
    Traceback (most recent call last):
    ValueError: values must be strictly decreasing
    >>> decrease_idx = np.argsort(scores)[::-1]
    >>> scores = da.asarray(scores[decrease_idx])
    >>> y = da.asarray(y[decrease_idx])
    >>> classification = binary_classification_curve(y, scores, positive_classification_label=2)
    >>> classification
    RocResults(false_positives=dask.array<sub, shape=(nan,), dtype=int64, chunksize=(nan,), chunktype=numpy.ndarray>,
    true_positives=dask.array<slice_with_int_dask_array_aggregate, shape=(nan,), dtype=int64, chunksize=(nan,),
    chunktype=numpy.ndarray>, thresholds=dask.array<slice_with_int_dask_array_aggregate, shape=(nan,), dtype=float64,
    chunksize=(nan,), chunktype=numpy.ndarray>)
    >>> classification.false_positives.compute()
    array([0, 1, 1, 2])
    >>> classification.true_positives.compute()
    array([1, 1, 2, 2])
    >>> classification.thresholds.compute()
    array([0.8 , 0.4 , 0.35, 0.1 ])
    """
    if sorted_truth.ndim != 1 or sorted_values.ndim != 1:
        raise ValueError("sorted_truth and sorted_values must be 1D")
    if sorted_truth.size != sorted_values.size:
        raise ValueError("sorted_truth and sorted_values must have same size")

    if sorted_truth.dtype != bool:
        if positive_classification_label is None:
            raise ValueError(
                f"positive_classification_label must be specified if truth array is not bool ({sorted_truth.dtype})"
            )
        sorted_truth = sorted_truth == positive_classification_label

    diff_values: da.Array = da.diff(sorted_values)
    if not da.all(diff_values < 0).compute():
        raise ValueError("values must be strictly decreasing")
    diff_values = da.abs(diff_values)

    values_min: float = da.nanmin(sorted_values).compute()
    values_max: float = da.nanmax(sorted_values).compute()
    minimum_step_size: float = np.abs(values_max - values_min) / num_intervals

    # As the values would be from the turbulence diagnostics, they are continuous
    # To reduce the data, use step_size to determine the minimum difference between two data points
    bucketed_value_indices = da.nonzero(diff_values > minimum_step_size)[0]
    threshold_indices = da.hstack((bucketed_value_indices, da.asarray([sorted_truth.size - 1])))

    true_positive = da.cumsum(sorted_truth)[threshold_indices]
    # Magical equation from scikit-learn which means another cumsum is avoided
    # https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L907
    false_positive = 1 + threshold_indices - true_positive

    return BinaryClassificationResult(
        false_positives=false_positive, true_positives=true_positive, thresholds=sorted_values[threshold_indices]
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


def _combine_sparse_arrays(
    block_concatenated_coo_arrays: "da.Array", shape_of_output: tuple, binary_numpy_function: Callable
) -> "da.Array":
    def combine_function(args: list, **kwargs: dict) -> da.Array:
        if len(args) == 2:  # noqa: PLR2004
            return sparse.elemwise(binary_numpy_function, args[0], args[1])

        out = args[0]
        for item in args[1:]:
            out = sparse.elemwise(binary_numpy_function, out, item)
        return out

    def on_chunk(x_chunk: "sparse.COO", axis: int, keepdims: bool) -> "sparse.COO":
        return x_chunk

    def finalise_aggregation(from_combine: "list[sparse.COO]", axis: int, keepdims: bool) -> "sparse.COO":
        return sparse.elemwise(binary_numpy_function, *from_combine)

    reduced: "da.Array" = da.reduction(
        block_concatenated_coo_arrays,
        on_chunk,
        finalise_aggregation,
        combine=combine_function,
        concatenate=False,
        dtype=bool,
        meta=sparse.COO((), shape=shape_of_output, fill_value=np.nan),
    )
    return reduced


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

    @staticmethod
    def _transform_aggregated_data_into_sparse_array(
        aggregated_data: "dd.DataFrame",
        coordinate_values: "da.Array",
        shape_of_output: tuple[int, int, int, int],
        condition: "DiagnosticValidationCondition",
    ) -> "da.Array":
        was_observed: "da.Array" = aggregated_data[condition.observed_turbulence_column_name].values  # noqa: PD011
        as_sparse_coo = functools.partial(
            lambda coord, data, output_shape: sparse.COO(coord.T, data=data, shape=output_shape),
            output_shape=shape_of_output,
        )
        return da.map_blocks(
            as_sparse_coo,
            coordinate_values,  # coords must be (ndim, nnz)
            was_observed,
            dtype=bool,
            meta=sparse.COO((), shape=shape_of_output),
        )

    def get_verification_ground_truth(
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

        shape_of_output = (
            prototype_diagnostic_array["latitude"].size,
            prototype_diagnostic_array["longitude"].size,
            prototype_diagnostic_array["pressure_level"].size,
            prototype_diagnostic_array["time"].size,
        )
        result = []
        for condition in validation_conditions:
            # Sparse arrays have been concatenated together => shape will be nchunks * latitude.size
            block_concatenated = self._transform_aggregated_data_into_sparse_array(
                aggregated_data, coordinate_values, shape_of_output, condition
            )
            as_sparse_array = _combine_sparse_arrays(block_concatenated, shape_of_output, np.logical_or)
            if do_persist:
                as_sparse_array = as_sparse_array.persist()
                blocking_wait_futures(as_sparse_array)
            result.append(as_sparse_array)

        return result
