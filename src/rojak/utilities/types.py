import numpy.typing as npt
import xarray as xr

ArrayLike = npt.NDArray | xr.DataArray


class GoHomeYouAreDrunkError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
