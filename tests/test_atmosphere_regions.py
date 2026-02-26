import dask.array as da
import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

from rojak.atmosphere.jet_stream import JetStreamAlgorithmFactory
from rojak.atmosphere.regions import (
    DistanceMeasure,
    DistanceMode,
    _parent_region_mask,
    _region_labeller,
    chebyshev_distance_from_a_to_b,
    distance_from_a_to_b,
    euclidean_distance_from_a_to_b,
    find_parent_region_of_intersection,
    label_regions,
    nearest_haversine_distance,
    shortest_haversine_distance_from_a_to_b,
    shortest_vertical_distance_from_a_to_b,
    vertical_distance_to_positive,
)
from rojak.orchestrator.configuration import JetStreamAlgorithms, TurbulenceDiagnostics
from rojak.turbulence.diagnostic import DiagnosticFactory


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


@pytest.mark.parametrize("distance_mode", [item.value for item in DistanceMode])
@pytest.mark.parametrize("mask_by", [True, False])
@pytest.mark.parametrize("all_present", [True, False])
def test_shortest_and_vertical_distance_to_positive_trivial(
    all_present: bool, mask_by: bool, distance_mode: DistanceMode, make_dummy_cat_data
) -> None:
    dummy_data = make_dummy_cat_data({})
    dummy_array: xr.DataArray = (
        xr.ones_like(dummy_data["temperature"], dtype=bool)
        if all_present
        else xr.zeros_like(dummy_data["temperature"], dtype=bool)
    )
    mask_array: xr.DataArray = (
        xr.ones_like(dummy_data["temperature"], dtype=bool)
        if mask_by
        else xr.zeros_like(dummy_data["temperature"], dtype=bool)
    )

    # If all present, the
    vert_dist_desired = (
        xr.zeros_like(dummy_array, dtype=int) if all_present else xr.full_like(dummy_array, np.inf, dtype=float)
    )

    computed_vert_dist = vertical_distance_to_positive(dummy_array, distance_mode=distance_mode)
    np.testing.assert_array_equal(computed_vert_dist, vert_dist_desired)

    computed_shortest_vert = shortest_vertical_distance_from_a_to_b(
        mask_array, dummy_array, distance_mode=distance_mode
    )
    shortest_vert_desired = computed_vert_dist if mask_by else xr.full_like(dummy_array, np.inf, dtype=float)
    np.testing.assert_array_equal(computed_shortest_vert, shortest_vert_desired)


class TestVerticalDistanceToPositive:
    """
    Test suite for vertical_distance_to_positive function.

    Generated by Claude Sonnet 4.5 using GitHub copilot. Modified to improve code quality and fix issues in generated
    code
    """

    @pytest.fixture
    def mock_pressure_levels(self) -> np.ndarray:
        """Standard pressure levels for testing."""
        return np.array([1000, 850, 700, 500, 300])

    @pytest.fixture
    def boolean_data_array(self, mock_pressure_levels: np.ndarray) -> xr.DataArray:
        """Create a boolean DataArray with multiple dimensions."""
        data = np.array(
            [
                [True, False, True, False, True],
                [False, True, False, True, False],
            ]
        )
        return xr.DataArray(
            data,
            dims=["time", "pressure_level"],
            coords={
                "time": [0, 1],
                "pressure_level": mock_pressure_levels,
            },
        )

    def test_absolute_distance_computation(self, boolean_data_array: xr.DataArray) -> None:
        """Test that absolute mode computes actual coordinate differences."""
        result = vertical_distance_to_positive(
            boolean_data_array,
            vertical_coord_name="pressure_level",
            distance_mode=DistanceMode.ABSOLUTE,
        )

        # At time=0, True at indices [0, 2, 4] (pressure 1000, 700, 300)
        # For pressure_level=850 (index 1), closest True is at 1000 (distance=150) or 700 (distance=150)
        assert result.sel(time=0, pressure_level=850).item() == (1000.0 - 850.0)

    def test_relative_distance_computation(self, boolean_data_array: xr.DataArray) -> None:
        """Test that relative mode computes index-based distances."""
        result = vertical_distance_to_positive(
            boolean_data_array,
            vertical_coord_name="pressure_level",
            distance_mode=DistanceMode.RELATIVE,
        )

        # Same logic as test_absolute_distance_computation, except that it is the relative distance as an index
        assert result.sel(time=0, pressure_level=850).item() == 1

    def test_returns_inf_when_no_true_values(self, mock_pressure_levels: np.ndarray) -> None:
        """Test that inf is returned when all values are False."""
        all_false = xr.DataArray(
            np.zeros((3, 5), dtype=bool),
            dims=["time", "pressure_level"],
            coords={"time": [0, 1, 2], "pressure_level": mock_pressure_levels},
        )

        result = vertical_distance_to_positive(all_false)
        assert np.all(np.isinf(result.to_numpy()))

    def test_returns_zero_for_true_positions(self, boolean_data_array: xr.DataArray):
        """Test that positions with True return distance of 0.

        Generated test did not have correct assertion logic for xarray.DataArray.
        """
        result = vertical_distance_to_positive(boolean_data_array, distance_mode=DistanceMode.RELATIVE)

        # Where input is True, distance should be 0
        assert np.all(result.to_numpy()[boolean_data_array.to_numpy()] == 0.0)

    def test_works_with_dataset(self, mock_pressure_levels: np.ndarray) -> None:
        """Test function works with xr.Dataset input."""
        dataset = xr.Dataset(
            {
                "var1": xr.DataArray(
                    np.array([[True, False, True]]),
                    dims=["time", "pressure_level"],
                    coords={"pressure_level": [1000, 850, 700]},
                ),
                "var2": xr.DataArray(
                    np.array([[False, True, False]]),
                    dims=["time", "pressure_level"],
                    coords={"pressure_level": [1000, 850, 700]},
                ),
            }
        )

        result = vertical_distance_to_positive(dataset)

        assert isinstance(result, xr.Dataset)
        assert "var1" in result
        assert "var2" in result

    def test_raises_assertion_for_non_boolean_dataarray(self, mock_pressure_levels: np.ndarray) -> None:
        """Test that non-boolean DataArray raises assertion error."""
        non_bool_array = xr.DataArray(
            np.array([[1, 2, 3], [4, 5, 6]]),
            dims=["time", "pressure_level"],
            coords={"pressure_level": mock_pressure_levels[:3]},
        )

        with pytest.raises(AssertionError):
            vertical_distance_to_positive(non_bool_array)

    def test_coordinate_renaming_and_restoration(self, boolean_data_array: xr.DataArray) -> None:
        """Test that vertical coordinate is properly renamed and restored."""
        result = vertical_distance_to_positive(boolean_data_array)

        # Check that the result has the original coordinate name
        assert "pressure_level" in result.dims
        assert "vertical_source" not in result.dims
        assert "vertical_target" not in result.dims

    @pytest.mark.parametrize(
        ("pressure_vals", "expected_shape"),
        [
            ([1000, 850, 700], (2, 3)),
            ([1000, 850, 700, 500, 300, 200, 100], (2, 7)),
        ],
    )
    def test_output_shape_matches_input(self, pressure_vals: np.ndarray, expected_shape: tuple[int, int]) -> None:
        """Test that output shape matches input shape across various sizes."""
        data = xr.DataArray(
            np.random.default_rng().choice([True, False], size=expected_shape),
            dims=["time", "pressure_level"],
            coords={"pressure_level": pressure_vals},
        )

        result = vertical_distance_to_positive(data)
        assert result.shape == expected_shape
