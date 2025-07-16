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

import sys
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import xarray as xr
from dask import array as da

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

NumpyOrDataArray = npt.NDArray | xr.DataArray


class GoHomeYouAreDrunkError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


# Limits = NamedTuple("Limits", [("lower", float), ("upper", float)])
type DiagnosticName = str
DistributionParameters = NamedTuple("DistributionParameters", [("mean", float), ("variance", float)])


class Limits[T](NamedTuple):
    lower: T
    upper: T


Coordinate = NamedTuple("Coordinate", [("latitude", float), ("longitude", float)])


def is_xr_data_array(obj: object) -> TypeIs[xr.DataArray]:
    return isinstance(obj, xr.DataArray)


def is_np_array(obj: object) -> TypeIs[npt.NDArray]:
    return isinstance(obj, np.ndarray)


def is_dask_array(array: object) -> TypeIs["da.Array"]:
    return isinstance(array, da.Array)
