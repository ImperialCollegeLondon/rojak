from typing import NamedTuple

import numpy.typing as npt
import xarray as xr

ArrayLike = npt.NDArray | xr.DataArray


class GoHomeYouAreDrunkError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


# Limits = NamedTuple("Limits", [("lower", float), ("upper", float)])
type DiagnosticName = str
DistributionParameters = NamedTuple("DistributionParameters", [("mean", float), ("variance", float)])


class Limits[T](NamedTuple):
    lower: T
    upper: T
