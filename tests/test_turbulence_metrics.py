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

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from rojak.turbulence.metrics import binary_classification_rate_from_cumsum, received_operating_characteristic


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
