from typing import NamedTuple

import dask
import dask.array as da
import numpy as np


class BinaryClassificationResult(NamedTuple):
    false_positives: "da.Array"
    true_positives: "da.Array"
    thresholds: "da.Array"


# Modified from scikit-learn.metrics roc_curve() method
# https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L1069
def received_operating_characteristic(
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

    Modifying the example in the scikit-learn `documentation on roc_curves`_:

    >>> import dask.array as da
    >>> y = np.asarray([1, 1, 2, 2])
    >>> scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    >>> received_operating_characteristic(da.asarray(y), da.asarray(scores), positive_classification_label=2)
    Traceback (most recent call last):
    ValueError: values must be strictly decreasing
    >>> decrease_idx = np.argsort(scores)[::-1]
    >>> scores = da.asarray(scores[decrease_idx])
    >>> y = da.asarray(y[decrease_idx])
    >>> roc = received_operating_characteristic(y, scores, positive_classification_label=2)
    >>> roc
    BinaryClassificationResult(false_positives=dask.array<truediv, shape=(5,), dtype=float64, chunksize=(3,),
    chunktype=numpy.ndarray>, true_positives=dask.array<truediv, shape=(5,), dtype=float64, chunksize=(3,),
    chunktype=numpy.ndarray>, thresholds=dask.array<concatenate, shape=(5,), dtype=float64, chunksize=(3,),
    chunktype=numpy.ndarray>)
    >>> roc.false_positives.compute()
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> roc.true_positives.compute()
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> roc.thresholds.compute()
    array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])

    see :py:func:`binary_classification_curve` for a more detailed explanation of doc test

    .. _documentation on roc_curves: https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
    """
    classification_result = binary_classification_curve(
        sorted_truth,
        sorted_values,
        num_intervals=num_intervals,
        positive_classification_label=positive_classification_label,
    )

    # Make curve start at 0
    zero_array = da.zeros(1)
    true_positive = da.hstack((zero_array, classification_result.true_positives.compute_chunk_sizes()))  # pyright: ignore[reportAttributeAccessIssue]
    false_positive = da.hstack((zero_array, classification_result.false_positives.compute_chunk_sizes()))  # pyright: ignore[reportAttributeAccessIssue]
    thresholds = da.hstack((da.asarray([np.inf]), classification_result.thresholds.compute_chunk_sizes()))  # pyright: ignore[reportAttributeAccessIssue]

    if false_positive[-1] < 0:
        raise ValueError("false positives cannot be negative")
    false_positive = false_positive / false_positive[-1]

    if true_positive[-1] < 0:
        raise ValueError("true positives cannot be negative")
    true_positive = true_positive / true_positive[-1]

    return BinaryClassificationResult(
        false_positives=false_positive, true_positives=true_positive, thresholds=thresholds
    )


# Modified from scikit-learn metric binary classification method
# https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L826
def binary_classification_curve(
    sorted_truth: "da.Array",
    sorted_values: "da.Array",
    num_intervals: int = 100,
    positive_classification_label: int | float | bool | str | None = None,
) -> BinaryClassificationResult:
    """
    Binary classification curve

    Args:
        sorted_truth:
        sorted_values:
        num_intervals:
        positive_classification_label:

    Returns:
        BinaryClassificationResult: A named tuple with the false positive, true positive, and thresholds

    Modifying the example in the scikit-learn `documentation on roc_curves`_:

    >>> import dask.array as da
    >>> y = np.asarray([1, 1, 2, 2])
    >>> scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    >>> binary_classification_curve(da.asarray(y), da.asarray(scores), positive_classification_label=2)
    Traceback (most recent call last):
    ValueError: values must be strictly decreasing

    Scikit-learn does not require the arrays to be sorted. However, as this implementation uses dask arrays, there
    is no built-in way to sort a dask array. Thus, the arrays passed into these methods must already be sorted

    >>> decrease_idx = np.argsort(scores)[::-1]
    >>> scores = da.asarray(scores[decrease_idx])
    >>> y = da.asarray(y[decrease_idx])

    Once the values are sorted, they can be passed into the method.

    >>> classification = binary_classification_curve(y, scores, positive_classification_label=2)
    >>> classification
    BinaryClassificationResult(false_positives=dask.array<sub, shape=(nan,), dtype=int64, chunksize=(nan,),
    chunktype=numpy.ndarray>, true_positives=dask.array<slice_with_int_dask_array_aggregate, shape=(nan,), dtype=int64,
    chunksize=(nan,), chunktype=numpy.ndarray>, thresholds=dask.array<slice_with_int_dask_array_aggregate, shape=(nan,),
    dtype=float64, chunksize=(nan,), chunktype=numpy.ndarray>)

    The method returns a named tuple :py:class:`BinaryClassificationResult` containing `dask.array.Array`. To get the
    values, `compute()` must be invoked on them evaluate the lazy collection.

    >>> classification.false_positives.compute()
    array([0, 1, 1, 2])
    >>> classification.true_positives.compute()
    array([1, 1, 2, 2])
    >>> classification.thresholds.compute()
    array([0.8 , 0.4 , 0.35, 0.1 ])

    .. _documentation on roc_curves: https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc
    """
    if sorted_truth.ndim != 1 or sorted_values.ndim != 1:
        raise ValueError("sorted_truth and sorted_values must be 1D")
    sizes = dask.compute(sorted_truth.size, sorted_values.size)  # pyright: ignore[reportPrivateImportUsage]
    if sizes[0] != sizes[1]:
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
