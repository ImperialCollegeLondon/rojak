import dask.array as da
import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

from rojak.atmosphere.jet_stream import JetStreamAlgorithmFactory
from rojak.atmosphere.regions import (
    DistanceMeasure,
    _parent_region_mask,
    _region_labeller,
    chebyshev_distance_from_a_to_b,
    distance_from_a_to_b,
    euclidean_distance_from_a_to_b,
    find_parent_region_of_intersection,
    label_regions,
    nearest_haversine_distance,
    shortest_haversine_distance_from_a_to_b,
)
from rojak.orchestrator.configuration import JetStreamAlgorithms, TurbulenceDiagnostics
from rojak.turbulence.diagnostic import DiagnosticFactory


def test_region_labeller_equiv_scipy_default_3d() -> None:
    array = np.random.default_rng().choice(2, 125).reshape((5, 5, 5))
    from_scipy, _ = ndi.label(array)  # pyright: ignore [reportGeneralTypeIssues]
    np.testing.assert_array_equal(_region_labeller(array, connectivity=1), from_scipy)


@pytest.mark.parametrize("num_dims", [2, 3])
def test_region_labeller_equiv_scipy_custom_structure(num_dims: int) -> None:
    custom_structure = np.zeros((3,) * num_dims, dtype=bool)
    custom_structure[:, 1] = True
    array = np.random.default_rng().choice(2, 5**num_dims).reshape((5,) * num_dims)
    from_scipy, _ = ndi.label(array, structure=custom_structure)  # pyright: ignore [reportGeneralTypeIssues]
    np.testing.assert_array_equal(_region_labeller(array, structure=custom_structure), from_scipy)


def test_region_labeller_raise_type_error() -> None:
    array = np.random.default_rng().choice(2, 125).reshape((5, 5, 5))
    with pytest.raises(TypeError, match="If structure is specified, connectivity must be None"):
        _region_labeller(array, connectivity=1, structure=np.zeros((3, 3, 3)))


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
            labelled.isel(pressure_level=level_index).transpose("longitude", "latitude"),
            from_scipy,
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
            labelled.isel(time=time_index).transpose("longitude", "latitude", "pressure_level"),
            from_scipy,
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


TI1_THRESHOLD: float = 1.3947336218633176e-10


@pytest.fixture
def get_is_ti1_turb(load_cat_data) -> xr.DataArray:
    return (
        DiagnosticFactory(load_cat_data(None, with_chunks=True)).create(TurbulenceDiagnostics.TI1).computed_value
        > TI1_THRESHOLD
    ).compute()


@pytest.fixture
def get_js_regions(load_cat_data) -> xr.DataArray:
    return (
        JetStreamAlgorithmFactory(load_cat_data(None, with_chunks=True))
        .create(JetStreamAlgorithms.ALPHA_VEL_KOCH)
        .identify_jet_stream()
    ).compute()


@pytest.mark.parametrize("num_dim", [2, 3])
def test_parent_region_mask_jit_equiv_guvectorize(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray, num_dim: int
) -> None:
    is_ti1_turb: xr.DataArray = get_is_ti1_turb
    js_regions: xr.DataArray = get_js_regions
    labeled_ti1: xr.DataArray = label_regions(is_ti1_turb, num_dims=num_dim)
    labeled_js: xr.DataArray = label_regions(js_regions, num_dims=num_dim)

    js_intersect_turb = is_ti1_turb & js_regions

    from_jit_turb: xr.DataArray = find_parent_region_of_intersection(
        labeled_ti1,
        js_intersect_turb,
        num_dims=num_dim,
        numba_vectorize=False,
    )
    from_guv_turb: xr.DataArray = find_parent_region_of_intersection(
        labeled_ti1,
        js_intersect_turb,
        num_dims=num_dim,
        numba_vectorize=True,
    )
    xr.testing.assert_equal(from_jit_turb, from_guv_turb)

    from_jit_js: xr.DataArray = find_parent_region_of_intersection(
        labeled_js,
        js_intersect_turb,
        num_dims=num_dim,
        numba_vectorize=False,
    )
    from_guv_js: xr.DataArray = find_parent_region_of_intersection(
        labeled_js,
        js_intersect_turb,
        num_dims=num_dim,
        numba_vectorize=True,
    )
    xr.testing.assert_equal(from_jit_js, from_guv_js)


@pytest.mark.parametrize("num_dim", [2, 3])
def test_label_then_mask_equiv_to_single_step(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray, num_dim: int
) -> None:
    is_ti1_turb: xr.DataArray = get_is_ti1_turb
    js_regions: xr.DataArray = get_js_regions
    labeled_js: xr.DataArray = label_regions(js_regions, num_dims=num_dim)

    js_intersect_turb = is_ti1_turb & js_regions

    from_guv_js: xr.DataArray = find_parent_region_of_intersection(
        labeled_js,
        js_intersect_turb,
        num_dims=num_dim,
        numba_vectorize=True,
    )

    if num_dim == 2:  # noqa: PLR2004
        for time_index in range(js_regions["time"].size):
            for level_index in range(js_regions["pressure_level"].size):
                from_scipy, _ = ndi.label(
                    js_regions.isel(time=time_index, pressure_level=level_index),
                    structure=ndi.generate_binary_structure(num_dim, num_dim),
                )  # pyright: ignore [reportGeneralTypeIssues]
                mask = _parent_region_mask(
                    from_scipy,
                    js_intersect_turb.isel(time=time_index, pressure_level=level_index).values,
                )
                np.testing.assert_array_equal(
                    from_guv_js.isel(time=time_index, pressure_level=level_index).transpose("latitude", "longitude"),
                    mask,
                )
    else:
        for time_index in range(js_regions["time"].size):
            from_scipy, _ = ndi.label(
                js_regions.isel(time=time_index),
                structure=ndi.generate_binary_structure(num_dim, num_dim),
            )  # pyright: ignore [reportGeneralTypeIssues]
            mask = _parent_region_mask(from_scipy, js_intersect_turb.isel(time=time_index).values)
            np.testing.assert_array_equal(
                from_guv_js.isel(time=time_index).transpose("latitude", "longitude", "pressure_level"),
                mask,
            )


@pytest.mark.parametrize("distance_measure", [e.value for e in DistanceMeasure])
@pytest.mark.parametrize("num_dim", [2, 3])
def test_distance_from_a_to_b_equiv_in_multi_dim(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray, num_dim: int, distance_measure: DistanceMeasure
) -> None:
    js_regions = get_js_regions
    turb_regions = get_is_ti1_turb
    computed_distance: xr.DataArray = distance_from_a_to_b(
        js_regions, turb_regions, distance_measure=distance_measure, num_dim=num_dim
    )
    distance_func_to_test = (
        euclidean_distance_from_a_to_b
        if distance_measure == DistanceMeasure.EUCLIDEAN
        else chebyshev_distance_from_a_to_b
    )
    if num_dim == 2:  # noqa: PLR2004
        for time_index in range(js_regions["time"].size):
            for level_index in range(js_regions["pressure_level"].size):
                np.testing.assert_array_equal(
                    computed_distance.isel(time=time_index, pressure_level=level_index).transpose(
                        "latitude", "longitude"
                    ),
                    distance_func_to_test(
                        js_regions.isel(time=time_index, pressure_level=level_index).to_numpy(),
                        turb_regions.isel(time=time_index, pressure_level=level_index).to_numpy(),
                    ),
                )
    else:
        for time_index in range(js_regions["time"].size):
            np.testing.assert_array_equal(
                computed_distance.isel(time=time_index).transpose("latitude", "longitude", "pressure_level"),
                distance_func_to_test(
                    js_regions.isel(time=time_index).to_numpy(),
                    turb_regions.isel(time=time_index).to_numpy(),
                ),
            )


@pytest.mark.parametrize("distance_measure", [e.value for e in DistanceMeasure])
def test_distance_a_to_b_2d_not_equiv_3d(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray, distance_measure: DistanceMeasure
) -> None:
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(
            distance_from_a_to_b(get_js_regions, get_is_ti1_turb, distance_measure=distance_measure, num_dim=2),
            distance_from_a_to_b(get_js_regions, get_is_ti1_turb, distance_measure=distance_measure, num_dim=3),
        )


@pytest.mark.parametrize("distance_measure", [e.value for e in DistanceMeasure])
@pytest.mark.parametrize("num_dim", [2, 3])
def test_distance_a_to_b_inverse_not_equiv(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray, distance_measure: DistanceMeasure, num_dim: int
) -> None:
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(
            distance_from_a_to_b(get_js_regions, get_is_ti1_turb, distance_measure=distance_measure, num_dim=num_dim),
            distance_from_a_to_b(get_is_ti1_turb, get_js_regions, distance_measure=distance_measure, num_dim=num_dim),
        )


def test_great_circle_distance_from_a_to_b_equiv_in_multi_dim(
    get_is_ti1_turb: xr.DataArray, get_js_regions: xr.DataArray
):
    js_regions = get_js_regions
    turb_regions = get_is_ti1_turb
    computed_distance: xr.DataArray = shortest_haversine_distance_from_a_to_b(
        js_regions,
        turb_regions,
    )

    latitude_coord = js_regions["latitude"].to_numpy()
    longitude_coord = js_regions["longitude"].to_numpy()

    for time_index in range(js_regions["time"].size):
        for level_index in range(js_regions["pressure_level"].size):
            np.testing.assert_array_equal(
                computed_distance.isel(time=time_index, pressure_level=level_index).transpose("latitude", "longitude"),
                nearest_haversine_distance(
                    js_regions.isel(time=time_index, pressure_level=level_index).to_numpy(),
                    turb_regions.isel(time=time_index, pressure_level=level_index).to_numpy(),
                    latitude_coord,
                    longitude_coord,
                ),
            )
