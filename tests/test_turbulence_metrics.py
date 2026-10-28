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
import gc
from collections.abc import Callable
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from rojak.turbulence.metrics import (
    _populate_confusion_matrix,
    area_under_curve,
    binary_classification_curve,
    binary_classification_rate_from_cumsum,
    confusion_matrix,
    contingency_table,
    critical_success_index,
    jaccard_index_multidim,
    matthews_corr_coeff,
    matthews_corr_coeff_multidim,
    received_operating_characteristic,
    stratified_contingency_table,
)

if TYPE_CHECKING:
    import xarray as xr
    from dask.distributed import Client


@pytest.mark.parametrize("as_pandas", [True, False])
def test_binary_classification_equiv_sklearn_example(as_pandas: bool, client: "Client") -> None:
    y = np.asarray([1, 1, 2, 2])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    decrease_idx = np.argsort(scores)[::-1]
    y = y[decrease_idx]
    scores = da.asarray(scores[decrease_idx])

    positive_label: int = 2
    roc = received_operating_characteristic(
        da.asarray(y),
        scores,
        positive_classification_label=positive_label,
        num_intervals=-1,
    )

    y_as_cumsum_of_condition = np.cumsum(y == positive_label)
    if as_pandas:
        y_as_cumsum_of_condition = pd.Series(y_as_cumsum_of_condition)
    cumsum_roc = binary_classification_rate_from_cumsum(y_as_cumsum_of_condition)
    assert cumsum_roc is not None

    np.testing.assert_equal(roc.true_positives.compute(), cumsum_roc.true_positives_rate)
    np.testing.assert_equal(roc.false_positives.compute(), cumsum_roc.false_positives_rate)


@pytest.mark.parametrize("as_pandas", [True, False])
@pytest.mark.parametrize("cumsum", [-np.arange(5), np.arange(0, 10, 2)])
def test_binary_classification_rate_from_cumsum_fails(cumsum: np.typing.NDArray, as_pandas: bool) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        binary_classification_rate_from_cumsum(cumsum if not as_pandas else pd.Series(cumsum))


def test_roc_computable_after_inputs_released(client: "Client") -> None:
    # Regression test for futures of the ROC being cancelled once the (persisted) inputs are no longer referenced.
    # This mirrors how DiagnosticsAmdarVerification.nearest_value_roc invokes received_operating_characteristic
    rng = np.random.default_rng(1)
    num_points: int = 20_000
    values = np.sort(rng.normal(size=num_points))[::-1].copy()
    truth = rng.random(num_points) < (1 / (1 + np.exp(-values)))
    chunks: int = num_points // 7

    results = []
    for _ in range(3):
        values_array = da.asarray(values, chunks=chunks).persist()
        results.append(
            received_operating_characteristic(
                da.asarray(truth, chunks=chunks).persist(), values_array, num_intervals=-1
            )
        )
        del values_array
    gc.collect()

    # As values are continuous and distinct, every point is a threshold
    expected_tpr = np.concatenate(([0], np.cumsum(truth))) / truth.sum()
    expected_fpr = np.concatenate(([0], np.cumsum(~truth))) / (~truth).sum()
    expected_thresholds = np.concatenate(([np.inf], values))
    for roc in results:
        fpr, tpr, thresholds = da.compute(roc.false_positives, roc.true_positives, roc.thresholds)
        np.testing.assert_allclose(fpr, expected_fpr)
        np.testing.assert_allclose(tpr, expected_tpr)
        np.testing.assert_equal(thresholds, expected_thresholds)
        np.testing.assert_allclose(
            area_under_curve(roc.false_positives, roc.true_positives), np.trapezoid(expected_tpr, expected_fpr)
        )


def test_binary_classification_curve_buckets_by_num_intervals() -> None:
    rng = np.random.default_rng(2)
    num_intervals: int = 5
    # Clusters 10 apart with small jitter => only the jumps between clusters exceed the step size of range / 5
    values = np.sort(np.concatenate([rng.uniform(0, 0.01, 400) + offset for offset in range(0, 50, 10)]))[::-1].copy()
    positive_rate: float = 0.3
    truth = rng.random(values.size) < positive_rate

    step_size = (values.max() - values.min()) / num_intervals
    expected_indices = np.append(np.nonzero(np.abs(np.diff(values)) > step_size)[0], values.size - 1)
    assert expected_indices.size == num_intervals  # boundaries between the 5 clusters + final point
    expected_tp = np.cumsum(truth)[expected_indices]

    result = binary_classification_curve(
        da.asarray(truth, chunks=300), da.asarray(values, chunks=300), num_intervals=num_intervals
    )
    np.testing.assert_equal(result.true_positives.compute(), expected_tp)
    np.testing.assert_equal(result.false_positives.compute(), 1 + expected_indices - expected_tp)
    np.testing.assert_equal(result.thresholds.compute(), values[expected_indices])


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64])
def test_confusion_matrix_counts_do_not_overflow(dtype: type) -> None:
    # More points per chunk than an int8 can count
    truth = np.tile([True, True, False, False], 250)
    prediction = np.tile([True, False, True, False], 250)
    tn, fp, fn, tp = (
        confusion_matrix(da.asarray(truth.astype(dtype), chunks=500), da.asarray(prediction.astype(dtype), chunks=500))
        .ravel()
        .tolist()
    )
    assert (tn, fp, fn, tp) == (250, 250, 250, 250)


@pytest.mark.parametrize(
    ("truth_array", "pred_array"),
    [(da.eye(2), da.ones(4)), (da.ones(4), da.eye(2)), (da.eye(2), da.eye(2))],
)
def test_confusion_matrix_single_dim_throw_error(truth_array: da.Array, pred_array: da.Array) -> None:
    with pytest.raises(ValueError, match="truth and prediction must be 1D"):
        confusion_matrix(truth_array, pred_array)


@pytest.mark.parametrize(("truth_array", "pred_array"), [(None, da.ones(2)), (da.ones(2), None)])
def test_populate_confusion_matrix_throw_error(truth_array: da.Array | None, pred_array: da.Array | None) -> None:
    with pytest.raises(ValueError, match="If confusion matrix is None, must provide truth and prediction"):
        _populate_confusion_matrix(truth=truth_array, prediction=pred_array)


@pytest.fixture
def get_two_dummy_bool_arrays(make_dummy_cat_data: Callable) -> Callable:
    dummy_ds = make_dummy_cat_data({})

    def _inner(cast_to_bool: bool = True) -> "tuple[xr.DataArray, xr.DataArray]":
        first_dummy: xr.DataArray = np.rint(dummy_ds.temperature)
        second_dummy: xr.DataArray = np.rint(dummy_ds.vorticity)
        if cast_to_bool:
            first_dummy = first_dummy.astype("bool")
            second_dummy = second_dummy.astype("bool")
        return first_dummy, second_dummy

    return _inner


def test_matthews_corr_coeff_multidim_equiv_single_dim(get_two_dummy_bool_arrays: Callable, client: "Client") -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays(cast_to_bool=False)

    matthews = matthews_corr_coeff_multidim(first_dummy.astype("bool"), second_dummy.astype("bool"), "time")
    dim_1 = first_dummy["longitude"].size
    dim_2 = first_dummy["latitude"].size
    dim_3 = first_dummy["pressure_level"].size
    first_dummy = first_dummy.astype("int")
    second_dummy = second_dummy.astype("int")
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                np.testing.assert_almost_equal(
                    matthews.isel(longitude=i, latitude=j, pressure_level=k),
                    matthews_corr_coeff(
                        truth=da.asarray(
                            first_dummy.isel(longitude=i, latitude=j, pressure_level=k),
                        ),
                        prediction=da.asarray(second_dummy.isel(longitude=i, latitude=j, pressure_level=k)),
                    ),
                )


def test_equiv_representation_of_matthews_corr_coeff(get_two_dummy_bool_arrays: Callable) -> None:
    """
    Equivalent MCC equation is from `wikipedia`_,

    .. math::

       \varphi = \frac{n_{11} n_{00} - n_{10} n_{01}}
       {\\sqrt{n_{1\bullet} n_{0\bullet} n_{\bullet0} n_{\bullet1}}}

    .. _wikipedia: https://en.wikipedia.org/wiki/Phi_coefficient#Definition

    """
    first_dummy, second_dummy = get_two_dummy_bool_arrays()

    table = contingency_table(first_dummy, second_dummy, sum_over="time")
    numerator: xr.DataArray = table.n_11 * table.n_00 - table.n_10 * table.n_01
    denominator: xr.DataArray = np.sqrt(
        (table.n_11 + table.n_10) * (table.n_01 + table.n_00) * (table.n_11 + table.n_01) * (table.n_10 + table.n_00),
    )  # pyright: ignore [reportAssignmentType]
    other_phi = numerator / denominator

    np.testing.assert_array_equal(matthews_corr_coeff_multidim(first_dummy, second_dummy, "time"), other_phi)


def test_check_equivalence_of_sum_in_either(get_two_dummy_bool_arrays: Callable) -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays()

    n = first_dummy["time"].size
    table = contingency_table(first_dummy, second_dummy, sum_over="time")
    compute_from_or = (first_dummy | second_dummy).sum(dim="time")
    np.testing.assert_array_equal(n - table.n_00, compute_from_or)


def test_equivalence_jaccard_index_and_critical_skill_score(get_two_dummy_bool_arrays: Callable) -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays(cast_to_bool=False)

    jaccards = jaccard_index_multidim(first_dummy.astype("bool"), second_dummy.astype("bool"), "time")
    dim_1 = first_dummy["longitude"].size
    dim_2 = first_dummy["latitude"].size
    dim_3 = first_dummy["pressure_level"].size
    first_dummy = first_dummy.astype("int")
    second_dummy = second_dummy.astype("int")
    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                np.testing.assert_almost_equal(
                    jaccards.isel(longitude=i, latitude=j, pressure_level=k),
                    critical_success_index(
                        truth=da.asarray(
                            first_dummy.isel(longitude=i, latitude=j, pressure_level=k),
                        ),
                        prediction=da.asarray(second_dummy.isel(longitude=i, latitude=j, pressure_level=k)),
                    ),
                )


def test_jaccard_index_multidim(get_two_dummy_bool_arrays: Callable) -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays()

    np.testing.assert_array_equal(
        jaccard_index_multidim(first_dummy, second_dummy, "time"),
        (first_dummy & second_dummy).sum(dim="time") / (first_dummy | second_dummy).sum(dim="time"),
    )


def test_conditional_probability(get_two_dummy_bool_arrays: Callable) -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays()
    table = contingency_table(first_dummy, second_dummy, sum_over="time")

    np.testing.assert_array_equal(table.n_11 / (table.n_11 + table.n_10), table.n_11 / first_dummy.sum(dim="time"))
    np.testing.assert_array_equal(table.n_11 / (table.n_11 + table.n_01), table.n_11 / second_dummy.sum(dim="time"))


def test_stratified_contingency_equiv_two_cases_variadic(get_two_dummy_bool_arrays: Callable) -> None:
    first_dummy, second_dummy = get_two_dummy_bool_arrays()
    third_dummy, _ = get_two_dummy_bool_arrays()
    single_version = stratified_contingency_table(first_dummy, second_dummy, third_dummy, sum_over="time")
    variadic_version = stratified_contingency_table(
        first_dummy, second_dummy, third_dummy, ~third_dummy, sum_over="time"
    )

    for single_, variadic_ in zip(single_version, variadic_version, strict=True):
        np.testing.assert_array_equal(single_.n_11, variadic_.n_11)
        np.testing.assert_array_equal(single_.n_01, variadic_.n_01)
        np.testing.assert_array_equal(single_.n_10, variadic_.n_10)
        np.testing.assert_array_equal(single_.n_00, variadic_.n_00)
