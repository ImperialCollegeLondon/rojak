import numpy as np
import xarray as xr

from rojak.turbulence.contrails import e_sat_ice


# Stolen wholesale from: https://github.com/contrailcirrus/pycontrails/blob/2f14ef8714ce0286aeabb2611db83bd26c7386ae/tests/unit/test_thermo_sac.py#L38
def test_e_sat_increasing():
    """Check that thermo.e_sat_ice, thermo.e_sat_liquid are increasing and positive."""
    temperature = np.linspace(150, 350, 10000)
    e_sat_ = e_sat_ice(xr.DataArray(temperature))
    assert np.all(e_sat_ > 0)
    assert np.all(np.diff(e_sat_) > 0)


# Other tests from pycontrails for ISSR are harder to steal
