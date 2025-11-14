import numba
import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from rojak.core import constants


@numba.vectorize([numba.float64(numba.float64), numba.float32(numba.float32)])
def _e_sat_ice_ufunc(temperature: float) -> float:
    # Equation stolen wholesale from: https://github.com/contrailcirrus/pycontrails/blob/2f14ef8714ce0286aeabb2611db83bd26c7386ae/pycontrails/physics/thermo.py#L131
    return 100.0 * np.exp(
        (-6024.5282 / temperature)
        + 24.7219
        + (0.010613868 * temperature)
        - (1.3198825e-5 * (temperature * temperature))
        - 0.49382577 * np.log(temperature)
    )


def e_sat_ice(temperature: "xr.DataArray") -> "xr.DataArray":
    return xr.apply_ufunc(
        _e_sat_ice_ufunc, temperature, dask="parallelized" if is_dask_collection(temperature) else "forbidden"
    )


# Stolen wholesale from: https://github.com/contrailcirrus/pycontrails/blob/2f14ef8714ce0286aeabb2611db83bd26c7386ae/pycontrails/physics/thermo.py#L405
def rhi(specific_humidity: "xr.DataArray", temperature: "xr.DataArray", pressure: "xr.DataArray") -> "xr.DataArray":
    return (
        specific_humidity * pressure * (constants.GAS_CONSTANT_VAPOUR / constants.GAS_CONSTANT_DRY_AIR)
    ) / e_sat_ice(temperature)


# Slightly modified from: https://github.com/contrailcirrus/pycontrails/blob/2f14ef8714ce0286aeabb2611db83bd26c7386ae/pycontrails/models/issr.py#L155
def issr(
    air_temperature: "xr.DataArray",
    specific_humidity: "xr.DataArray | None" = None,
    air_pressure: "xr.DataArray | None" = None,
    relative_humidity_ice: "xr.DataArray | None" = None,
    rhi_threshold: float = 1.0,
    as_mask: bool = True,
) -> "xr.DataArray":
    if relative_humidity_ice is None:
        if specific_humidity is None or air_pressure is None:
            raise TypeError(
                "If relative_humidity_ice is None, both specific_humidity and air_pressure must be provided"
            )
        relative_humidity_ice = rhi(specific_humidity, air_temperature, air_pressure)

    # -constant.ABSOLUTE_ZERO = 273K = 0C
    sufficiently_cold = air_temperature < -constants.ABSOLUTE_ZERO
    sufficiently_humid = relative_humidity_ice > rhi_threshold

    issr_ = sufficiently_cold & sufficiently_humid

    return issr_ if as_mask else issr_.astype(relative_humidity_ice.dtype)
