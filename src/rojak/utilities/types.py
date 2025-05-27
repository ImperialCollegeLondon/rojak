import xarray as xr
import numpy.typing as npt

ArrayLike = npt.NDArray | xr.DataArray


class GoHomeYouAreDrunk(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
