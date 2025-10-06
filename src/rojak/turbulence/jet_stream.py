from abc import ABC, abstractmethod
from typing import ClassVar

import xarray as xr

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

    def __init__(self, u_wind: xr.DataArray, v_wind: xr.DataArray) -> None:
        self._u_wind = u_wind
        self._wind_speed = wind_speed(u_wind, v_wind)

    def identify_jet_stream(self) -> "xr.DataArray": ...
