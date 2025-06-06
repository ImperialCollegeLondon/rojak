from typing import TYPE_CHECKING

import xarray as xr
import xarray.testing as xrt

from rojak.core.data import MetData

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_pressure_to_altitude_standard_atmosphere(mocker: "MockerFixture") -> None:
    mocker.patch.multiple(MetData, __abstractmethods__=set())
    met_data = MetData()  # pyright: ignore[reportAbstractUsage]
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, met_data.pressure_to_altitude_std_atm(pressures), rtol=1e-3)
