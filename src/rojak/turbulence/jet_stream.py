from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from numpy.typing import NDArray

from rojak.core.derivatives import CartesianDimension, GradientMode, spatial_gradient
from rojak.turbulence.calculations import wind_speed


class JetStreamAlgorithm(ABC):
    @abstractmethod
    def identify_jet_stream(self) -> "xr.DataArray": ...


class AlphaVelField(JetStreamAlgorithm):
    """
    Identifies jet stream using :math:`\\alpha vel` scalar variable from [Koch2006]_

    """

    _ALPHA_VEL_THRESHOLD: ClassVar[float] = 30  # Value from [Koch2006]_ based on systemic studies
    _pressure_coord_name: ClassVar[str] = "pressure_level"
    _wind_speed: xr.DataArray

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray) -> None:
        assert u_wind[self._pressure_coord_name].units == v_wind[self._pressure_coord_name].units, (
            "Wind speed must have pressure coordinate in the same units"
        )
        assert u_wind[self._pressure_coord_name].units == "hPa", "Pressure coordinate must be in hPa"
        self._wind_speed = wind_speed(u_wind, v_wind)

    def identify_jet_stream(self) -> "xr.DataArray":
        diff_on_pressure: xr.DataArray = self._wind_speed[self._pressure_coord_name].diff(self._pressure_coord_name)
        is_decreasing: float
        if (diff_on_pressure < 0).all():
            is_decreasing = 1
        elif (diff_on_pressure > 0).all():
            is_decreasing = -1  # increasing => 100 to 400 => from sky to ground so flip integral sign
        else:
            raise ValueError("Pressure levels must be strictly decreasing or increasing")

        max_pressure_diff: NDArray = np.abs(
            (self._wind_speed[self._pressure_coord_name][0] - self._wind_speed[self._pressure_coord_name][-1]).values
        )
        alpha_vel: xr.DataArray = (
            (1 / max_pressure_diff) * is_decreasing * self._wind_speed.integrate(self._pressure_coord_name)
        )  # pyright: ignore [reportAssignmentType]

        return alpha_vel > self._ALPHA_VEL_THRESHOLD


# Modified from: https://github.com/scikit-image/scikit-image/blob/e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/feature/peak.py#L37
def get_peak_mask(two_dimensional_slice: NDArray[np.floating], threshold: float) -> NDArray[np.bool_]:
    assert two_dimensional_slice.ndim == 2  # noqa: PLR2004

    # Footprint becomes the 8 adjacent values
    footprint: NDArray = np.ones((3,) * 2)
    max_regions: NDArray[np.floating] = ndi.maximum_filter(two_dimensional_slice, footprint=footprint, mode="grid-wrap")
    max_mask: NDArray[np.bool_] = two_dimensional_slice == max_regions

    if np.all(max_mask):  # no peaks identified as everything is a peak
        max_mask[:] = False

    max_mask &= two_dimensional_slice > threshold

    return max_mask


class WindSpeedCondSchiemann(JetStreamAlgorithm):
    """
    Identifies jet stream using conditions placed on the wind speed from [Schiemann2009]_
    """

    _u_wind: "xr.DataArray"
    _wind_speed: "xr.DataArray"
    _MINIMUM_WIND_SPEED_THRESHOLD: ClassVar[float] = 30

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray) -> None:
        self._u_wind = u_wind
        self._wind_speed = wind_speed(u_wind, v_wind, is_abs=True)

    def _local_maxima_skimage(self) -> xr.DataArray:
        assert {"latitude", "pressure_level"}.issubset(self._wind_speed.coords)
        # vectorize=True => loop over non core_dims
        # https://tutorial.xarray.dev/advanced/apply_ufunc/automatic-vectorizing-numpy.html#conclusion
        return xr.apply_ufunc(
            get_peak_mask,
            self._wind_speed,
            input_core_dims=[["latitude", "pressure_level"]],
            output_core_dims=[["latitude", "pressure_level"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.bool],
            kwargs={"threshold": self._MINIMUM_WIND_SPEED_THRESHOLD},
        )

    def _local_maxima(self) -> "xr.DataArray":
        f_y: xr.DataArray = spatial_gradient(
            self._wind_speed, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.Y
        )["dfdy"]
        f_yy: xr.DataArray = spatial_gradient(f_y, "deg", GradientMode.GEOSPATIAL, dimension=CartesianDimension.Y)[
            "dfdy"
        ]

        f_p: xr.DataArray = self._wind_speed.differentiate("pressure_level")
        f_pp: xr.DataArray = f_p.differentiate("pressure_level")

        f_yp: xr.DataArray = f_y.differentiate("pressure_level")
        determinant_hessian = f_yy * f_pp - f_yp * f_yp

        # local_maxima: "xr.DataArray" = xr.ones_like(self._wind_speed, dtype="int")
        return xr.where(((f_y == 0) & (f_p == 0) & (determinant_hessian > 0) & (f_yy < 0)), 1, 0)

    def identify_jet_stream(self) -> "xr.DataArray":
        # return self._local_maxima() & (self._wind_speed >= self._MINIMUM_WIND_SPEED_THRESHOLD) & (self._u_wind >= 0)
        return self._local_maxima_skimage() & (self._u_wind >= 0)
