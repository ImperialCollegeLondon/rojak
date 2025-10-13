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
from functools import singledispatch
from typing import TYPE_CHECKING, NamedTuple

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.base import is_dask_collection
from scipy import integrate
from sparse import COO

from rojak.utilities.types import is_dask_array, is_np_array, is_xr_data_array

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.utilities.types import NumpyOrDataArray


class BinaryClassificationResult(NamedTuple):
    false_positives: "da.Array"
    true_positives: "da.Array"
    thresholds: "da.Array"


class BinaryClassificationRateFromLabels(NamedTuple):
    true_positives_rate: "NDArray"
    false_positives_rate: "NDArray"


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

    This method is a dask-friendly implementation of :func:`scikit-learn:sklearn.metrics.roc_curve`

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
    true_positive = da.hstack((zero_array, classification_result.true_positives)).persist()  # pyright: ignore[reportAttributeAccessIssue]
    false_positive = da.hstack((zero_array, classification_result.false_positives)).persist()  # pyright: ignore[reportAttributeAccessIssue]
    thresholds = da.hstack((da.asarray([np.inf]), classification_result.thresholds)).persist()  # pyright: ignore[reportAttributeAccessIssue]

    if false_positive[-1] < 0:
        raise ValueError("false positives cannot be negative")
    false_positive = false_positive / false_positive[-1]

    if true_positive[-1] < 0:
        raise ValueError("true positives cannot be negative")
    true_positive = true_positive / true_positive[-1]

    return BinaryClassificationResult(
        false_positives=false_positive, true_positives=true_positive, thresholds=thresholds
    )


def _check_lazy_sizes_equal[T: xr.DataArray | da.Array](first_array: T, second_array: T) -> int:
    assert is_dask_collection(first_array)
    assert is_dask_collection(second_array)
    sizes = dask.compute(first_array.size, second_array.size)  # pyright: ignore[reportPrivateImportUsage]

    if sizes[0] != sizes[1]:
        raise ValueError("Both arrays should have the same size")

    return sizes[0]


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
    BinaryClassificationResult(false_positives=dask.array<sub, shape=(4,), dtype=int64, chunksize=(3,),
    chunktype=numpy.ndarray>, true_positives=dask.array<slice_with_int_dask_array_aggregate, shape=(4,), dtype=int64,
    chunksize=(3,), chunktype=numpy.ndarray>, thresholds=dask.array<slice_with_int_dask_array_aggregate, shape=(4,),
    dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>)


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
    _check_lazy_sizes_equal(sorted_truth, sorted_values)

    if sorted_truth.dtype != bool:
        if positive_classification_label is None:
            raise ValueError(
                f"positive_classification_label must be specified if truth array is not bool ({sorted_truth.dtype})"
            )
        sorted_truth = sorted_truth == positive_classification_label

    diff_values: da.Array = da.diff(sorted_values)
    if not da.all(diff_values <= 0).compute():
        raise ValueError("values must be strictly decreasing")
    diff_values = da.abs(diff_values)

    if num_intervals == -1:
        minimum_step_size: float = 0.0
    else:
        values_min: float = da.nanmin(sorted_values).compute()
        values_max: float = da.nanmax(sorted_values).compute()
        minimum_step_size: float = np.abs(values_max - values_min) / num_intervals

    # As the values would be from the turbulence diagnostics, they are continuous
    # To reduce the data, use step_size to determine the minimum difference between two data points
    bucketed_value_indices = da.nonzero(diff_values > minimum_step_size)[0].compute_chunk_sizes()
    # ^ computing chunks here means that the subsequent dask arrays have a known number of chunks
    threshold_indices = da.hstack((bucketed_value_indices, da.asarray([sorted_truth.size - 1]))).persist()

    true_positive = da.cumsum(sorted_truth)[threshold_indices].persist()
    # Magical equation from scikit-learn which means another cumsum is avoided
    # https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L907
    false_positive = 1 + threshold_indices - true_positive

    return BinaryClassificationResult(
        false_positives=false_positive, true_positives=true_positive, thresholds=sorted_values[threshold_indices]
    )


@singledispatch
def binary_classification_rate_from_cumsum(
    cumsum_for_group: pd.Series | np.ndarray,
) -> BinaryClassificationRateFromLabels | None:
    """
    Binary classification curve with cumulative sum on labels

    Assumes that labels have already been sorted such that the values are increasing. Moreover, the cumulative sum
    has been performed on boolean truth labels such that it represents the number of true positives (or positive
    observations). The main use case for this function is on a given group from a Pandas GroupBy object.

    Args:
        cumsum_for_group: Cumulative sum on boolean truth labels

    Returns:
        True positive and false positive rate. If there are no true positives, returns None
    """


@binary_classification_rate_from_cumsum.register
def _(
    cumsum_for_group: pd.Series,
) -> BinaryClassificationRateFromLabels | None:
    return _binary_classification_from_cumsum(cumsum_for_group.to_numpy())


@binary_classification_rate_from_cumsum.register
def _binary_classification_from_cumsum(
    cumsum_for_group: np.ndarray, min_true_positives: int = 2
) -> BinaryClassificationRateFromLabels | None:
    group_size: int = cumsum_for_group.size
    true_positive_rate = cumsum_for_group
    false_positive_rate = 1 + np.arange(group_size) - true_positive_rate

    num_true_positives: int = true_positive_rate[-1]
    num_false_positives: int = false_positive_rate[-1]

    if num_true_positives < 0:
        raise ValueError("num_true_positives must not be negative")
    if num_false_positives < 0:
        raise ValueError("num_false_positives must not be negative")

    if num_true_positives < min_true_positives:
        return None

    true_positive_rate = np.hstack((np.zeros(1), true_positive_rate)) / num_true_positives
    # Prevent divisions by zero
    false_positive_rate = (
        np.hstack((np.zeros(1), false_positive_rate)) / num_false_positives
        if num_false_positives != 0
        else np.zeros_like(true_positive_rate)
    )

    return BinaryClassificationRateFromLabels(
        true_positives_rate=true_positive_rate, false_positives_rate=false_positive_rate
    )


def _serial_area_under_curve(x_values: "NDArray", y_values: "NDArray") -> float:
    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have same size")
    if x_values.size < 2:  # noqa: PLR2004
        raise ValueError("x_value and y_values must have at least 2 points to compute area under the curve")

    dx: NDArray = np.diff(x_values)
    is_decreasing: np.bool = np.all(dx <= 0)
    if not (is_decreasing or np.all(dx >= 0)):
        raise ValueError("x_values must be increasing or decreasing")

    area: np.floating | NDArray = integrate.trapezoid(y_values, x_values)

    return -float(area) if is_decreasing else float(area)


def _parallel_area_under_curve(x_values: da.Array | xr.DataArray, y_values: da.Array | xr.DataArray) -> float:
    if is_xr_data_array(x_values):
        assert is_dask_array(x_values.values)
        # Import pandas into is throwing up incorrect linting
        x_vals: da.Array = x_values.values  # noqa: PD011
    else:
        assert is_dask_array(x_values)
        x_vals: da.Array = x_values

    if is_xr_data_array(y_values):
        assert is_dask_array(y_values.values)
        y_vals: da.Array = y_values.values  # noqa: PD011
    else:
        assert is_dask_array(y_values)
        y_vals: da.Array = y_values

    array_size = _check_lazy_sizes_equal(x_vals, y_vals)
    if array_size < 2:  # noqa: PLR2004
        raise ValueError("x_value and y_values must have at least 2 points to compute area under the curve")

    delta_x: da.Array = da.diff(x_vals)
    is_decreasing = da.all(delta_x <= 0).compute()
    is_increasing = da.all(delta_x >= 0).compute()
    if not (is_increasing or is_decreasing):
        raise ValueError("x_value must be increasing or decreasing")

    area: float | NDArray = np.add.reduce(
        da.map_overlap(lambda x, y: integrate.trapezoid(y[:-1], x[:-1]), x_vals, y_vals, depth=1).compute()
    ) + integrate.trapezoid(np.asarray(y_vals[-2:]), np.asarray(x_vals[-2:]))

    # If dx is decreasing, then area is negative
    return -float(area) if is_decreasing else float(area)


# Modified implementation of scikit-learn.metrics.auc
# https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_ranking.py#L43
def area_under_curve(
    x_values: "da.Array | NumpyOrDataArray",
    y_values: "da.Array | NumpyOrDataArray",
) -> float:
    """
    Area under the curve

    Integrates using the :func:`scipy:scipy.integrate.trapezoid` method. Integrals over :class:`dask:dask.array.Array`
    collections are evaluated for each chunk and combined accordingly

    Args:
        x_values: 1D array of points corresponding to the y values
        y_values: 1D array to integrate

    Returns:
        Area under the curve

    Modifying the examples in the documentation for :func:`scipy:scipy.integrate.trapezoid` method,

    >>> area_under_curve(np.asarray([4, 5, 6]), np.asarray([1, 2, 3]))
    4.0
    >>> area_under_curve(da.asarray([4, 5, 6]), da.asarray([1, 2, 3]))
    4.0

    As this is to be used for computing AUC for ROC, the area under the curve should always be positive if
    the values to integrate over are decreasing in the x-axis. Thus, this method will return + 8.0 and not -8.0
    as seen in :func:`scipy:scipy.integrate.trapezoid`

    >>> area_under_curve(np.asarray([8, 6, 4]), np.asarray([1, 2, 3]))
    8.0
    >>> area_under_curve(da.asarray([8, 6, 4]), da.asarray([1, 2, 3]))
    8.0

    This behaviour is consistent with :func:`scikit-learn:sklearn.metrics.auc`. Modifying the example in the
    scikit-learn docs,

    >>> y = np.asarray([1, 1, 2, 2])
    >>> scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    >>> decrease_idx = np.argsort(scores)[::-1]
    >>> scores = da.asarray(scores[decrease_idx], chunks=2)
    >>> y = da.asarray(y[decrease_idx], chunks=2)
    >>> roc = received_operating_characteristic(y, scores, positive_classification_label=2)
    >>> area_under_curve(roc.false_positives, roc.true_positives)
    0.75
    """
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be 1D")

    if (
        is_dask_collection(x_values)
        and is_dask_collection(y_values)
        and not is_np_array(x_values)
        and not is_np_array(y_values)
    ):
        return _parallel_area_under_curve(x_values, y_values)

    if (is_dask_collection(x_values) and not is_dask_collection(y_values)) or (
        not is_dask_collection(x_values) and is_dask_collection(y_values)
    ):
        raise ValueError("x_values and y_values must either be both dask collections or both not dask collections")

    assert is_np_array(x_values)
    assert is_np_array(y_values)

    return _serial_area_under_curve(x_values, y_values)


def _check_roc_curve_input_validity(roc_curve: BinaryClassificationResult) -> None:
    if roc_curve.true_positives[0] != 0 or roc_curve.false_positives[0] != 0:
        raise ValueError("First value of POD and POFD must be 0")
    if da.max(roc_curve.true_positives).compute() > 1:
        raise ValueError("Maximum value of POD must be less than or equal to 1")
    if da.min(roc_curve.true_positives).compute() < 0:
        raise ValueError("Minimum value of POD must be greater than or equal to 0")


def true_skill_score(roc_curve: BinaryClassificationResult) -> da.Array:
    """
    True Skill Score (TSS) statistic

    The TSS is defined in `Wikipedia <https://en.wikipedia.org/wiki/Youden%27s_J_statistic#Definition>`__ as:

    .. math::
        \\begin{align}
            TSS &= \\text{sensitivity} + \\text{specificity} - 1 \\
            &= \\frac{\\text{TP}}{\\text{TP} + \\text{FP}} + \\frac{\\text{TN}}{\\text{TN} + \\text{FP}} - 1
        \\end{align}

    where :math:`TP` is the number of true positives, :math:`TN` the number of true negatives,
    :math:`FP` the number of false positives, :math:`FN` the number of false negatives.

    This is also the definition used in [Sharman2006]_

    """
    _check_roc_curve_input_validity(roc_curve)

    sensitivity = roc_curve.true_positives
    specificity = roc_curve.false_positives
    return sensitivity + specificity - 1


def _check_array_is_boolean(array: da.Array) -> None:
    assert is_dask_collection(array)
    if array.dtype != bool and not da.isin(array, [0, 1]).all().compute():
        raise ValueError("Array must be boolean")


# Modified from: https://github.com/scikit-learn/scikit-learn/blob/c60dae20604f8b9e585fc18a8fa0e0fb50712179/sklearn/metrics/_classification.py#L371
def confusion_matrix(truth: da.Array, prediction: da.Array) -> "NDArray":
    """
    Compute the confusion matrix

    This is a simplified dask-friendly implementation of :func:`scikit-learn:sklearn.metrics.confusion_matrix` for the
    binary classification problem.

    Args:
        truth: dask array of shape (n_samples,)
            Ground truth (correct) target values.
        prediction: dask array of shape (n_samples,)
            Estimated targets as returned by a classifier.

    Returns:
        Confusion matrix in the order of (tn, fp, fn, tp)

    Examples
    --------

    This example is modified from the docstring of :func:`scikit-learn:sklearn.metrics.confusion_matrix`

    >>> y_true = da.asarray([0, 1, 0, 1])
    >>> y_pred = da.asarray([1, 1, 1, 0])
    >>> tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    >>> (tn, fp, fn, tp)
    (0, 2, 1, 1)

    This example is modified from the user guide `documentation on confusion matrix`_:

    >>> x_true = da.asarray([0, 0, 0, 1, 1, 1, 1, 1])
    >>> x_pred = da.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    >>> tn, fp, fn, tp = confusion_matrix(x_true, x_pred).ravel().tolist()
    >>> tn, fp, fn, tp
    (2, 1, 2, 3)

    .. _documentation on confusion matrix: https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix

    """
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("truth and prediction must be 1D")

    _check_array_is_boolean(truth)
    _check_array_is_boolean(prediction)

    sample_weights: da.Array = da.ones(truth.shape[0], dtype=np.int_)
    matrix: COO = COO((truth, prediction), sample_weights, shape=(2, 2))

    return matrix.todense()


def matthew_corrcoef(
    truth: da.Array | None = None, prediction: da.Array | None = None, confuse_matrix: "NDArray | None" = None
) -> float:
    """
    Compute the Matthew's Correlation Coefficient

    Args:
        truth: dask array of shape (n_samples,)
            Ground truth (correct) target values.
        prediction: dask array of shape (n_samples,)
            Estimated targets as returned by a classifier.
        confuse_matrix: Numpy array of shape (2, 2)
            Result from computing the confusion matrix using :func:`confusion_matrix`.
            If this is ``None``, then the result is computed using :func:`confusion_matrix` using ``truth`` and
            ``prediction``.

    Returns:

    Examples
    --------

    Example from Wikipedia page on `Matthew's Correlation Coefficient`_

    >>> actual = da.asarray([1,1,1,1,1,1,1,1,0,0,0,0])
    >>> pred = da.asarray([0,0,1,1,1,1,1,1,0,0,0,1])
    >>> float(matthew_corrcoef(truth=actual, prediction=pred))
    0.478

    .. _Matthew's Correlation Coefficient: https://en.wikipedia.org/wiki/Phi_coefficient#Example

    """
    if confuse_matrix is None:
        if truth is None or prediction is None:
            raise ValueError("If confusion matrix is None, must provide truth and prediction")
        confuse_matrix = confusion_matrix(truth, prediction)

    tn, fp, fn, tp = confuse_matrix.ravel().tolist()
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / denominator
