from abc import ABC, abstractmethod
from typing import ClassVar

import xarray as xr

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

    def identify_jet_stream(self) -> "xr.DataArray": ...


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
        return self._local_maxima() & (self._wind_speed >= self._MINIMUM_WIND_SPEED_THRESHOLD) & (self._u_wind >= 0)
