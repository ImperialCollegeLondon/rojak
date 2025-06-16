import xarray as xr
from xarray import testing as xrt

from rojak.core.calculations import pressure_to_altitude_std_atm


def test_pressure_to_altitude_standard_atmosphere(create_met_data_mock) -> None:
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, pressure_to_altitude_std_atm(pressures), rtol=1e-3)
