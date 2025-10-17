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
from collections.abc import Callable
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from rojak.turbulence.metrics import (
    _populate_confusion_matrix,
    binary_classification_rate_from_cumsum,
    confusion_matrix,
    contingency_table,
    matthews_corr_coeff,
    matthews_corr_coeff_multidim,
    received_operating_characteristic,
)

if TYPE_CHECKING:
    import xarray as xr


@pytest.mark.parametrize("as_pandas", [True, False])
def test_binary_classification_equiv_sklearn_example(as_pandas: bool) -> None:
    y = np.asarray([1, 1, 2, 2])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    decrease_idx = np.argsort(scores)[::-1]
    y = y[decrease_idx]
    scores = da.asarray(scores[decrease_idx])

    positive_label: int = 2
    roc = received_operating_characteristic(
        da.asarray(y), scores, positive_classification_label=positive_label, num_intervals=-1
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


@pytest.mark.parametrize(
    ("truth_array", "pred_array"), [(da.eye(2), da.ones(4)), (da.ones(4), da.eye(2)), (da.eye(2), da.eye(2))]
)
def test_confusion_matrix_single_dim_throw_error(truth_array: da.Array, pred_array: da.Array) -> None:
    with pytest.raises(ValueError, match="truth and prediction must be 1D"):
        confusion_matrix(truth_array, pred_array)


@pytest.mark.parametrize(("truth_array", "pred_array"), [(None, da.ones(2)), (da.ones(2), None)])
def test_populate_confusion_matrix_throw_error(truth_array: da.Array | None, pred_array: da.Array | None) -> None:
    with pytest.raises(ValueError, match="If confusion matrix is None, must provide truth and prediction"):
        _populate_confusion_matrix(truth=truth_array, prediction=pred_array)


def test_matthews_corr_coeff_multidim_equiv_single_dim(make_dummy_cat_data: Callable) -> None:
    dummy_ds = make_dummy_cat_data({})
    first_dummy: xr.DataArray = np.rint(dummy_ds.temperature)
    second_dummy: xr.DataArray = np.rint(dummy_ds.vorticity)

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


def test_equiv_representation_of_matthews_corr_coeff(make_dummy_cat_data: Callable) -> None:
    """
    Equivalent MCC equation is from `wikipedia`_,

    .. math::

       \varphi = \frac{n_{11} n_{00} - n_{10} n_{01}}
       {\\sqrt{n_{1\bullet} n_{0\bullet} n_{\bullet0} n_{\bullet1}}}

    .. _wikipedia: https://en.wikipedia.org/wiki/Phi_coefficient#Definition

    """
    dummy_ds = make_dummy_cat_data({})
    first_dummy: xr.DataArray = np.rint(dummy_ds.temperature).astype("bool")
    second_dummy: xr.DataArray = np.rint(dummy_ds.vorticity).astype("bool")

    table = contingency_table(first_dummy, second_dummy, "time")
    numerator: xr.DataArray = table.n_11 * table.n_00 - table.n_10 * table.n_01
    denominator: xr.DataArray = np.sqrt(
        (table.n_11 + table.n_10) * (table.n_01 + table.n_00) * (table.n_11 + table.n_01) * (table.n_10 + table.n_00)
    )  # pyright: ignore [reportAssignmentType]
    other_phi = numerator / denominator

    np.testing.assert_array_equal(matthews_corr_coeff_multidim(first_dummy, second_dummy, "time"), other_phi)


def test_check_equivalence_of_sum_in_either(make_dummy_cat_data: Callable) -> None:
    dummy_ds = make_dummy_cat_data({})
    first_dummy: xr.DataArray = np.rint(dummy_ds.temperature).astype("bool")
    second_dummy: xr.DataArray = np.rint(dummy_ds.vorticity).astype("bool")

    n = first_dummy["time"].size
    table = contingency_table(first_dummy, second_dummy, "time")
    compute_from_or = (first_dummy | second_dummy).sum(dim="time")
    np.testing.assert_array_equal(n - table.n_00, compute_from_or)
