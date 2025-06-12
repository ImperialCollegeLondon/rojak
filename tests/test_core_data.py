from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import pytest
import xarray as xr
import xarray.testing as xrt
import xarray.testing.strategies as xrst
from hypothesis import given

from rojak.core.data import MetData
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import SpatialDomain

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def create_met_data_mock(mocker: "MockerFixture"):
    mocker.patch.multiple(MetData, __abstractmethods__=set())
    return MetData()  # pyright: ignore[reportAbstractUsage]


def test_pressure_to_altitude_standard_atmosphere(create_met_data_mock) -> None:
    met_data = create_met_data_mock
    # Values from https://github.com/Unidata/MetPy/blob/60c94ebd5f314b85d770118cb7bfbe369a668c8c/tests/calc/test_basic.py#L327
    pressures = xr.DataArray([975.2, 987.5, 956.0, 943.0])
    alts = xr.DataArray([321.5, 216.5, 487.6, 601.7])
    xrt.assert_allclose(alts, met_data.pressure_to_altitude_std_atm(pressures), rtol=1e-3)


@given(st.data())
def test_select_domain_fails_on_dims(data) -> None:
    dummy_data = xr.Dataset(
        data_vars={"a": (data.draw(xrst.variables(dims=xrst.dimension_names(min_dims=1, max_dims=4))))}
    )
    with pytest.raises(AssertionError) as excinfo:
        Era5Data(dummy_data).select_domain(
            SpatialDomain(minimum_longitude=0, maximum_longitude=10, minimum_latitude=0, maximum_latitude=10),
            dummy_data,
        )

    assert excinfo.type is AssertionError


@given(st.data())
def test_select_domain_not_within_longitude(data) -> None:
    def array_strategy(shape, dtype):
        return npst.arrays(dtype, shape, elements={"max_value": 0})

    dummy_data = xr.Dataset(
        data_vars={
            "a": data.draw(
                xrst.variables(
                    array_strategy_fn=array_strategy,
                    dims=st.just(["longitude", "latitude", "time", "level"]),
                )
            )
        }
    )
    with pytest.raises(ValueError, match="Longitudinal coordinate") as excinfo:
        Era5Data(dummy_data).select_domain(
            SpatialDomain(minimum_longitude=0, maximum_longitude=10, minimum_latitude=0, maximum_latitude=10),
            dummy_data,
        )
    assert excinfo.type is ValueError


@given(st.data())
def test_select_domain_not_within_latitude(data) -> None:
    array_data = data.draw(
        npst.arrays(
            npst.floating_dtypes(),
            (2, 10, 3, 3),
        )
    )
    lat_vals = data.draw(npst.arrays(npst.floating_dtypes(), st.just(10), elements={"max_value": 0, "min_value": -90}))
    other_coords = data.draw(npst.arrays(npst.floating_dtypes(), st.just(3)))
    dummy_data = xr.Dataset(
        data_vars={
            "a": xr.DataArray(
                data=array_data,
                dims=["longitude", "latitude", "time", "level"],
                coords={"longitude": [-180, 10], "latitude": lat_vals, "time": other_coords, "level": other_coords},
            ),
        }
    )
    with pytest.raises(ValueError, match="Latitudinal coordinate") as excinfo:
        Era5Data(dummy_data).select_domain(
            SpatialDomain(minimum_longitude=-90, maximum_longitude=0, minimum_latitude=1, maximum_latitude=10),
            dummy_data,
        )
    assert excinfo.type is ValueError
