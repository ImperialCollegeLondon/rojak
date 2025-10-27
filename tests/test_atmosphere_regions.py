import dask.array as da
import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

from rojak.atmosphere.regions import _region_labeller, label_regions


def test_region_labeller_equiv_scipy_default_3d() -> None:
    array = np.random.default_rng().choice(2, 125).reshape((5, 5, 5))
    from_scipy, _ = ndi.label(array)  # pyright: ignore [reportGeneralTypeIssues]
    np.testing.assert_array_equal(_region_labeller(array, connectivity=1), from_scipy)


def test_label_regions_2d() -> None:
    array = xr.DataArray(
        da.from_array(np.random.default_rng().choice(2, 120).reshape((4, 5, 6))),
        dims=("longitude", "latitude", "pressure_level"),
    )
    # print(array.compute())
    labelled = label_regions(array, num_dims=2, connectivity=1)
    for level_index in range(6):
        from_scipy, _ = ndi.label(array[:, :, level_index])  # pyright: ignore [reportGeneralTypeIssues]
        np.testing.assert_array_equal(
            labelled.isel(pressure_level=level_index).transpose("longitude", "latitude"), from_scipy
        )


def test_label_regions_3d() -> None:
    array = xr.DataArray(
        da.from_array(np.random.default_rng().choice(2, 840).reshape((4, 5, 6, 7))),
        dims=("longitude", "latitude", "pressure_level", "time"),
    )
    labelled = label_regions(array)
    for time_index in range(7):
        from_scipy, _ = ndi.label(array.isel(time=time_index), structure=ndi.generate_binary_structure(3, 3))  # pyright: ignore [reportGeneralTypeIssues]
        np.testing.assert_array_equal(
            labelled.isel(time=time_index).transpose("longitude", "latitude", "pressure_level"), from_scipy
        )


def test_label_region_fail_value_error() -> None:
    array = xr.DataArray(
        da.from_array(np.random.default_rng().choice(2, 840).reshape((4, 5, 6, 7))),
        dims=("longitude", "latitude", "pressure_level", "time"),
    )

    with pytest.raises(ValueError, match="num_dims must be 2 or 3 as they are the spatial dimensions") as excinfo:
        label_regions(array, num_dims=4)

    assert excinfo.type is ValueError

    with pytest.raises(ValueError, match="num_dims must be 2 or 3 as they are the spatial dimensions") as excinfo:
        label_regions(array, num_dims=1)

    assert excinfo.type is ValueError


@pytest.mark.parametrize(
    "core_dims",
    [
        pytest.param(["x", "y", "z"], id="not subset of dims"),
        pytest.param(["latitude", "longitude"], id="wrong num dims"),
    ],
)
def test_label_regions_assertion_error(core_dims: list[str] | None) -> None:
    array = xr.DataArray(
        da.from_array(np.random.default_rng().choice(2, 840).reshape((4, 5, 6, 7))),
        dims=("longitude", "latitude", "pressure_level", "time"),
    )
    with pytest.raises(AssertionError):
        label_regions(array, core_dims=core_dims)
