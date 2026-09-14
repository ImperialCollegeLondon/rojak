from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

from rojak.atmosphere.jet_stream import JetStreamAlgorithmFactory
from rojak.atmosphere.regions import (
    DistanceMeasure,
    DistanceMode,
    ExtremaKind,
    _apply_and_combine_on_n_extrema_groups,
    _parent_region_mask,
    _project_data_about_extrema,
    _region_labeller,
    _stack_extrema,
    apply_extrema_filter,
    chebyshev_distance_from_a_to_b,
    circular_footprint,
    composite_about_extrema,
    distance_from_a_to_b,
    euclidean_distance_from_a_to_b,
    find_parent_region_of_intersection,
    identify_and_stack_circular_extrema,
    identify_circular_extrema,
    label_regions,
    nearest_haversine_distance,
    shortest_haversine_distance_from_a_to_b,
    shortest_vertical_distance_from_a_to_b,
    vertical_distance_to_positive,
)
from rojak.orchestrator.configuration import JetStreamAlgorithms, TurbulenceDiagnostics
from rojak.turbulence.diagnostic import DiagnosticFactory

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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


class TestExtremaKind:
    def test_minima_value(self) -> None:
        assert ExtremaKind.MINIMA == "minima"

    def test_maxima_value(self) -> None:
        assert ExtremaKind.MAXIMA == "maxima"

    def test_is_str(self) -> None:
        assert isinstance(ExtremaKind.MINIMA, str)
        assert isinstance(ExtremaKind.MAXIMA, str)

    def test_members(self) -> None:
        assert set(ExtremaKind) == {ExtremaKind.MINIMA, ExtremaKind.MAXIMA}


class TestCircularFootprint:
    def test_shape_dtype_and_centre(self) -> None:
        radius: int = 3
        footprint = circular_footprint(radius)
        assert footprint.shape == (2 * radius + 1, 2 * radius + 1)
        assert footprint.dtype == bool

        centre = footprint.shape[0] // 2
        assert footprint[centre, centre]

    @pytest.mark.parametrize("radius", [1, 2, 5, 10, 24])
    def test_corners_are_false(self, radius: int) -> None:
        """Corners of the footprint should always be False for radius > 1."""
        footprint = circular_footprint(radius)
        assert not footprint[0, 0]
        assert not footprint[0, -1]
        assert not footprint[-1, 0]
        assert not footprint[-1, -1]

    def test_radius_zero(self) -> None:
        with pytest.raises(ValueError, match="Radius must be positive"):
            _ = circular_footprint(0)

    def test_radius_one(self) -> None:
        footprint = circular_footprint(1)
        assert footprint.shape == (3, 3)

    @pytest.mark.parametrize("radius", [1, 2, 5, 10, 24])
    def test_symmetry(self, radius) -> None:
        """Footprint should be symmetric along both axes."""
        footprint = circular_footprint(radius)
        np.testing.assert_array_equal(footprint, footprint[::-1])
        np.testing.assert_array_equal(footprint, footprint[:, ::-1])


class TestApplyExtremaFilter:
    @pytest.fixture
    def get_dummy_array(self, make_dummy_cat_data) -> xr.DataArray:
        dummy_data: xr.DataArray = make_dummy_cat_data(None, use_numpy=False, rng_seed=42)["temperature"]
        return dummy_data.isel(pressure_level=0)

    def test_fails_without_specifying_size_or_footprint(self, get_dummy_array) -> None:
        dummy_data = get_dummy_array
        with pytest.raises(RuntimeError, match="no footprint provided"):
            _ = apply_extrema_filter(dummy_data, ExtremaKind.MAXIMA).compute()

    def test_output_shape_dtype_dims_coords_preserved(self, get_dummy_array) -> None:
        dummy_data = get_dummy_array
        result = apply_extrema_filter(dummy_data, ExtremaKind.MAXIMA, size=10)

        for coord_name, coord_value in dummy_data.coords.items():
            assert coord_name in result.coords
            assert result.coords[coord_name].dtype == coord_value.dtype
            np.testing.assert_array_equal(result.coords[coord_name], coord_value)

        assert result.dtype == dummy_data.dtype
        assert set(result.dims) == set(dummy_data.dims)

    @pytest.mark.parametrize(
        "filter_kwargs", [{"size": 3}, {"footprint": np.ones((2, 2)), "axes": (0, 1)}, {"size": 3, "mode": "wrap"}]
    )
    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_returns_dataarray(
        self, get_dummy_array: xr.DataArray, extrema_kind: ExtremaKind, filter_kwargs: dict
    ) -> None:
        result = apply_extrema_filter(get_dummy_array, extrema_kind, **filter_kwargs).compute()
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize(
        "filter_kwargs", [{"size": 2}, {"footprint": np.ones((1, 1)), "axes": (0, 1)}, {"size": 2, "mode": "wrap"}]
    )
    def test_maxima_filter_values_gte_input(self, get_dummy_array: xr.DataArray, filter_kwargs: dict) -> None:
        """Maximum filter output should always be >= input values."""
        result = apply_extrema_filter(get_dummy_array, ExtremaKind.MAXIMA, **filter_kwargs)
        assert (result >= get_dummy_array).all()

    @pytest.mark.parametrize(
        "filter_kwargs", [{"size": 2}, {"footprint": np.ones((1, 1)), "axes": (0, 1)}, {"size": 2, "mode": "wrap"}]
    )
    def test_minima_filter_values_lte_input(self, get_dummy_array, filter_kwargs: dict) -> None:
        """Minimum filter output should always be <= input values."""
        result = apply_extrema_filter(get_dummy_array, ExtremaKind.MINIMA, **filter_kwargs)
        assert (result <= get_dummy_array).all()

    @pytest.mark.parametrize(
        "filter_kwargs", [{"size": 2}, {"footprint": np.ones((1, 1)), "axes": (0, 1)}, {"size": 2, "mode": "wrap"}]
    )
    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_dask_and_numpy_results_are_equal(
        self, get_dummy_array: xr.DataArray, filter_kwargs: dict, extrema_kind: ExtremaKind
    ) -> None:
        """apply_ufunc should produce identical results as doing it on a 2D plane."""

        dask_result = apply_extrema_filter(get_dummy_array, extrema_kind, **filter_kwargs)
        ndi_function = ndi.maximum_filter if extrema_kind == ExtremaKind.MAXIMA else ndi.minimum_filter

        for time_index in range(get_dummy_array["time"].size):
            numpy_result = ndi_function(get_dummy_array.isel(time=time_index), **filter_kwargs)
            np.testing.assert_array_almost_equal(
                numpy_result, dask_result.isel(time=time_index).transpose("longitude", "latitude").compute()
            )

    def test_raises_on_non_3d_input(self) -> None:
        data_2d = xr.DataArray(
            np.ones((10, 10)),
            dims=["latitude", "longitude"],
        )
        with pytest.raises(ValueError, match="3D"):
            apply_extrema_filter(data_2d, ExtremaKind.MAXIMA, size=3).compute()

    def test_raises_on_missing_lat_dim(self) -> None:
        data = xr.DataArray(
            np.ones((5, 10, 10)),
            dims=["time", "y", "longitude"],
        )
        with pytest.raises(ValueError, match="latitude"):
            apply_extrema_filter(data, ExtremaKind.MAXIMA, size=3).compute()

    def test_raises_on_missing_lon_dim(self) -> None:
        data = xr.DataArray(
            np.ones((5, 10, 10)),
            dims=["time", "latitude", "x"],
        )
        with pytest.raises(ValueError, match="longitude"):
            apply_extrema_filter(data, ExtremaKind.MAXIMA, size=3).compute()

    def test_custom_dim_names(self, get_dummy_array) -> None:
        renamed_array = get_dummy_array.rename({"latitude": "lat", "longitude": "lon"})
        result = apply_extrema_filter(
            renamed_array,
            ExtremaKind.MAXIMA,
            latitude_dim_name="lat",
            longitude_dim_name="lon",
            size=3,
        ).compute()
        assert result.dtype == renamed_array.dtype
        assert set(result.dims) == set(renamed_array.dims)
        assert set(result.coords.keys()) == set(renamed_array.coords.keys())

    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_uniform_array_unchanged(self, extrema_kind: ExtremaKind, get_dummy_array: xr.DataArray) -> None:
        uniform_array = xr.ones_like(get_dummy_array)
        result = apply_extrema_filter(uniform_array, extrema_kind, size=3).compute()
        xr.testing.assert_equal(result, uniform_array, check_dim_order=False)


class TestIdentifyCircularExtrema:
    SMALL_FOOTPRINT_RADIUS: int = 1

    def get_array_with_extrema(self, extrema_type: ExtremaKind) -> xr.DataArray:
        data = da.zeros((3, 10, 10)) if extrema_type == ExtremaKind.MAXIMA else da.ones((3, 10, 10))
        data[:, 5, 5] = 1.0 if extrema_type == ExtremaKind.MAXIMA else 0.0
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": np.arange(data.shape[0]),
                "latitude": np.arange(data.shape[1]),
                "longitude": np.arange(data.shape[2]),
            },
        )

    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_returns_dataset_and_basic_properties(self, extrema_kind: ExtremaKind) -> None:
        dummy_data = self.get_array_with_extrema(extrema_kind)
        result = identify_circular_extrema(
            dummy_data,
            extrema_kind,
            extrema_threshold_value=0.5,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
        ).compute()
        assert isinstance(result, xr.Dataset)

        assert "local_extrema" in result
        assert "extrema_regions" in result

        assert result["local_extrema"].dtype == bool
        assert result["extrema_regions"].dtype == bool

        for data_var_value in result.data_vars.values():
            assert data_var_value.dtype == bool
            assert set(data_var_value.dims) == set(dummy_data.dims)
            assert set(data_var_value.coords.keys()) == set(dummy_data.coords.keys())

    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_known_extrema_detected(self, extrema_kind: ExtremaKind) -> None:
        """The known extrema at (lat=5, lon=5) should be detected."""
        result = identify_circular_extrema(
            self.get_array_with_extrema(extrema_kind),
            extrema_kind,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
        )
        assert result["local_extrema"].isel(latitude=5, longitude=5).all().compute()
        assert result["extrema_regions"].isel(latitude=slice(4, 7), longitude=slice(4, 7)).all().compute()
        assert not result["local_extrema"].all().compute()

    @pytest.mark.parametrize(("extrema_kind", "threshold"), [(ExtremaKind.MAXIMA, 5), (ExtremaKind.MINIMA, -0.1)])
    def test_no_extrema_when_threshold_not_met(self, extrema_kind: ExtremaKind, threshold: float) -> None:
        result = identify_circular_extrema(
            self.get_array_with_extrema(extrema_kind),
            extrema_kind,
            extrema_threshold_value=threshold,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
        ).compute()

        assert not result["local_extrema"].all()
        assert not result["extrema_regions"].all()
        assert not result["local_extrema"].any()
        assert not result["extrema_regions"].any()

    @pytest.mark.parametrize("threshold_value", [0, 0.5, 0.9])
    def test_known_maxima_above_threshold(self, threshold_value: float) -> None:
        result = identify_circular_extrema(
            self.get_array_with_extrema(ExtremaKind.MAXIMA),
            ExtremaKind.MAXIMA,
            extrema_threshold_value=threshold_value,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
        ).compute()
        assert result["local_extrema"].any()
        assert result["extrema_regions"].any()
        assert result["local_extrema"].isel(latitude=5, longitude=5).all()

        # All the corner values are False
        assert not result["extrema_regions"].isel(latitude=4, longitude=4).all()
        assert not result["extrema_regions"].isel(latitude=7, longitude=7).all()
        assert not result["extrema_regions"].isel(latitude=4, longitude=7).all()
        assert not result["extrema_regions"].isel(latitude=7, longitude=4).all()

        # Cross diagonal values
        assert result["extrema_regions"].isel(latitude=5, longitude=slice(4, 7)).all()
        assert result["extrema_regions"].isel(longitude=5, latitude=slice(4, 7)).all()

    @pytest.mark.parametrize("threshold_value", [0.1, 0.5, 0.9])
    def test_known_minima_above_threshold(self, threshold_value: float) -> None:
        result = identify_circular_extrema(
            self.get_array_with_extrema(ExtremaKind.MINIMA),
            ExtremaKind.MINIMA,
            extrema_threshold_value=threshold_value,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
        ).compute()
        assert result["local_extrema"].any()
        assert result["extrema_regions"].any()
        assert result["local_extrema"].isel(latitude=5, longitude=5).all()

        # All the corner values are False
        assert not result["extrema_regions"].isel(latitude=4, longitude=4).all()
        assert not result["extrema_regions"].isel(latitude=7, longitude=7).all()
        assert not result["extrema_regions"].isel(latitude=4, longitude=7).all()
        assert not result["extrema_regions"].isel(latitude=7, longitude=4).all()

        # Cross diagonal values
        assert result["extrema_regions"].isel(latitude=5, longitude=slice(4, 7)).all()
        assert result["extrema_regions"].isel(longitude=5, latitude=slice(4, 7)).all()

    def test_raises_on_non_3d_input(self) -> None:
        data_2d = xr.DataArray(
            np.ones((10, 10)),
            dims=["latitude", "longitude"],
        )
        with pytest.raises(ValueError, match="3D"):
            identify_circular_extrema(data_2d, ExtremaKind.MINIMA, extrema_threshold_value=0.5)

    def test_raises_on_missing_lat_dim(self) -> None:
        data = xr.DataArray(
            np.ones((5, 10, 10)),
            dims=["time", "y", "longitude"],
        )
        with pytest.raises(ValueError, match="latitude"):
            identify_circular_extrema(data, ExtremaKind.MINIMA, extrema_threshold_value=0.5)

    def test_raises_on_missing_lon_dim(self) -> None:
        data = xr.DataArray(
            np.ones((5, 10, 10)),
            dims=["time", "latitude", "x"],
        )
        with pytest.raises(ValueError, match="longitude"):
            identify_circular_extrema(data, ExtremaKind.MINIMA, extrema_threshold_value=0.5)

    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_sel_region_indexer(self, extrema_kind: ExtremaKind) -> None:
        """sel_region_indexer should subset the data before processing."""
        indexer = {"latitude": slice(0, 5), "longitude": slice(0, 5)}
        len_after_slice: int = 6
        result = identify_circular_extrema(
            self.get_array_with_extrema(extrema_kind),
            extrema_kind,
            extrema_threshold_value=0.5,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
            sel_region_indexer=indexer,
        )

        assert result["local_extrema"].sizes["latitude"] == len_after_slice
        assert result["local_extrema"].sizes["longitude"] == len_after_slice

    @pytest.mark.parametrize("extrema_kind", [ExtremaKind.MINIMA, ExtremaKind.MAXIMA])
    def test_custom_dim_names(self, extrema_kind: ExtremaKind) -> None:
        result = identify_circular_extrema(
            self.get_array_with_extrema(extrema_kind).rename({"latitude": "lat", "longitude": "lon"}),
            extrema_kind,
            extrema_threshold_value=0.5,
            footprint_radius=self.SMALL_FOOTPRINT_RADIUS,
            lat_dim_name="lat",
            lon_dim_name="lon",
        )
        assert isinstance(result, xr.Dataset)
        assert set(result.coords.keys()).issuperset({"lat", "lon"})


class TestStackExtrema:
    target_shape: tuple[int, int] = (161, 441)
    peak_location_value: int = 2

    @pytest.fixture
    def simple_extrema_mask(self) -> np.ndarray:
        """
        Simple (lat, lon) boolean mask with a single True value at (10, 20).
        Represents a single extremum location.
        """
        mask = np.zeros(self.target_shape, dtype=bool)
        mask[10, 20] = True
        return mask

    @pytest.fixture
    def simple_filtered(self) -> np.ndarray:
        """
        Simple (lat, lon) float array with labelled regions.
        Region label 5.0 is at position (10, 20) matching simple_extrema_mask.
        """
        filtered = np.zeros(self.target_shape, dtype=np.float32)
        filtered[8:13, 18:23] = 5.0  # region around the extremum
        return filtered

    def test_output_shape_single_extremum_and_dtype_int8(self, simple_extrema_mask, simple_filtered):
        """
        Output shape should be (n_extrema, lat, lon) where n_extrema equals the number of True values in
        extrema_locations.
        """
        result = _stack_extrema(simple_extrema_mask, simple_filtered)
        n_extrema = simple_extrema_mask.sum()
        assert result.shape == (
            n_extrema,
            simple_extrema_mask.shape[0],
            simple_extrema_mask.shape[1],
        )
        assert result.dtype == np.int8

    def test_centre_value_is_2(self, simple_extrema_mask, simple_filtered):
        """
        The exact location of each extremum (where extrema_locations is True) must have a value of 2 in the output.
        """
        result = _stack_extrema(simple_extrema_mask, simple_filtered)
        indices = np.argwhere(simple_extrema_mask)
        for i, (lat_idx, lon_idx) in enumerate(indices):
            assert result[i, lat_idx, lon_idx] == self.peak_location_value

    def test_region_values_are_1(self, simple_extrema_mask, simple_filtered):
        """
        Points that share the same label value as the extremum but are not the centre should have a value of 1.
        """
        result = _stack_extrema(simple_extrema_mask, simple_filtered)
        # Region is 8:13, 18:23 — centre is at 10, 20
        # All region points except centre should be 1
        region_mask = simple_filtered == simple_filtered[10, 20]
        centre_mask = np.zeros_like(region_mask)
        centre_mask[10, 20] = True
        region_not_centre = region_mask & ~centre_mask
        assert np.all(result[0][region_not_centre])

    def test_non_region_values_are_0(self, simple_extrema_mask, simple_filtered):
        """Points outside the extremum region should have a value of 0."""
        result = _stack_extrema(simple_extrema_mask, simple_filtered)
        outside_region: np.ndarray = simple_filtered != simple_filtered[10, 20]
        assert not np.any(result[0][outside_region])

    def test_multiple_extrema_produces_correct_slices(self):
        """
        With multiple extrema, each slice should correspond to exactly
        one extremum with its own region and centre.
        """
        mask = np.zeros(self.target_shape, dtype=bool)
        filtered = np.zeros(self.target_shape, dtype=np.float32)

        # Two distinct extrema with different label values
        mask[10, 20] = True
        filtered[8:13, 18:23] = 5.0

        mask[50, 100] = True
        filtered[48:53, 98:103] = 10.0

        result = _stack_extrema(mask, filtered)

        num_extrema: int = 2
        assert result.shape[0] == num_extrema
        assert result[0, 10, 20] == num_extrema
        assert result[1, 50, 100] == num_extrema
        # Each slice should only contain its own region
        assert result[0, 50, 100] == 0
        assert result[1, 10, 20] == 0

    def test_no_extrema_returns_empty(self):
        """
        When extrema_locations has no True values, output should have zero slices along the first dimension.
        """
        mask = np.zeros(self.target_shape, dtype=bool)
        filtered = np.zeros(self.target_shape, dtype=np.float32)
        result = _stack_extrema(mask, filtered)
        assert result.size == 0


class TestApplyAndCombineOnNExtremaGroups:
    @staticmethod
    def _make_simple_group_dataset() -> xr.Dataset:
        lat = np.linspace(65.0, 25.0, 161)
        lon = np.linspace(-100.0, 10.0, 441)
        time = np.array(
            ["2025-12-29T00:00", "2025-12-29T06:00", "2025-12-29T12:00", "2025-12-29T18:00"], dtype="datetime64[ns]"
        )

        n_lat, n_lon, n_time = len(lat), len(lon), len(time)
        extrema_locations = np.zeros((n_time, n_lat, n_lon), dtype=bool)
        extrema_locations[:, 10, 20] = True
        filtered = np.zeros_like(extrema_locations, dtype=np.float32)
        filtered[:, 8:13, 18:23] = 5.0
        return xr.Dataset(
            {
                "extrema_locations": (["time", "latitude", "longitude"], extrema_locations),
                "filtered": (["time", "latitude", "longitude"], filtered),
            },
            coords={
                "time": time,
                "latitude": lat,
                "longitude": lon,
                "n_extrema": ("time", np.arange(n_time, dtype=int)),
            },
        )

    def test_stack_method_invoked_and_return_is_as_expected(self, mocker: "MockerFixture") -> None:
        mock_fn = mocker.patch(
            "rojak.atmosphere.regions._stack_extrema",
        )
        ds = self._make_simple_group_dataset()
        mock_fn.return_value = xr.ones_like(ds["extrema_locations"])
        result = _apply_and_combine_on_n_extrema_groups(
            ds,
            lat_dim_name="latitude",
            lon_dim_name="longitude",
            new_dim_name="extrema_count",
        )
        mock_fn.assert_called()

        assert isinstance(result, xr.DataArray)
        assert "n_extrema" in result.dims
        assert "time" not in result.dims
        assert "time" in result.coords


@pytest.fixture
def load_mslp_data() -> xr.DataArray:
    return xr.open_dataset("tests/_static/test_mslp.nc", engine="h5netcdf")["msl"]


class TestIdentifyAndStackCircularExtrema:
    def test_raises_if_data_not_3d(self, load_mslp_data: xr.DataArray) -> None:
        """Should raise ValueError if input data is not 3D."""
        with pytest.raises(ValueError, match="Target data must be 3D"):
            identify_and_stack_circular_extrema(
                load_mslp_data.isel(time=0),
                ExtremaKind.MINIMA,
            )

    def test_raises_if_wrong_dims(self, load_mslp_data: xr.DataArray) -> None:
        """Should raise ValueError if required dimension names are missing."""
        renamed = load_mslp_data.rename({"latitude": "lat"})
        with pytest.raises(
            ValueError, match="Target data must contain the specified time, latitude and longitude dimensions"
        ):
            identify_and_stack_circular_extrema(renamed, ExtremaKind.MINIMA)

    def test_identify_and_stack_on_mslp_data(self, load_mslp_data: xr.DataArray) -> None:
        result = identify_and_stack_circular_extrema(load_mslp_data, ExtremaKind.MINIMA, extrema_threshold_value=100000)

        # Obtained from manually inspecting the data
        num_minima_in_mslp: int = 22

        # Check that the dimensions and coords are correct
        assert isinstance(result, xr.DataArray)
        assert "n_extrema" in result.dims
        assert "time" in result.coords

        # Check that it produces the correct result
        assert result["n_extrema"].size == num_minima_in_mslp

        # Check time axis is sorted, i.e. increasing
        assert (result["time"].diff(dim=...).astype(int) > 0).all()


class TestProjectDataAboutExtrema:
    target_shape: tuple[int, int] = (161, 441)
    lats: np.ndarray = np.linspace(65.0, 25.0, 161)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 441)

    def test_output_shape_is_n_km_by_n_km(self) -> None:
        mask = np.zeros(self.target_shape, dtype=np.int8)
        mask[10, 20] = 2
        mask[8:13, 18:23] = 1
        field = np.random.default_rng(42).random(self.target_shape)

        result = _project_data_about_extrema(
            mask,
            field,
            lats=self.lats,
            lons=self.lons,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        n_km = len(np.arange(-500, 501, 50))
        assert result.shape == (n_km, n_km)

    def test_returns_nan_when_no_centre(self) -> None:
        result = _project_data_about_extrema(
            np.zeros(self.target_shape, dtype=np.int8),
            np.ones(self.target_shape),
            lats=self.lats,
            lons=self.lons,
            max_km_extent=500,
            grid_km_spacing=50,
        )
        assert np.all(np.isnan(result))

    def test_raises_if_multiple_centres(self) -> None:
        mask = np.zeros(self.target_shape, dtype=np.int8)
        mask[10, 20] = 2
        mask[50, 100] = 2  # second centre — invalid
        field = np.ones(self.target_shape)

        with pytest.raises(ValueError, match="There should only be one extrema"):
            _project_data_about_extrema(
                mask,
                field,
                lats=self.lats,
                lons=self.lons,
                max_km_extent=500,
                grid_km_spacing=50,
            )

    def test_centre_of_output_is_not_nan(self, load_mslp_data) -> None:
        msl_shape = load_mslp_data.isel(time=0).shape
        mask = np.zeros(msl_shape, dtype=np.int8)
        mask[80, 200] = 2  # centre of the domain
        mask[78:83, 218:223] = 1

        result = _project_data_about_extrema(
            mask,
            load_mslp_data.isel(time=0).values,
            lats=load_mslp_data["latitude"].values,
            lons=load_mslp_data["longitude"].values,
            max_km_extent=1000,
            grid_km_spacing=25,
        )
        centre_idx = result.shape[0] // 2
        assert not np.isnan(result[centre_idx, centre_idx])

    def test_custom_int_for_center(self, load_mslp_data) -> None:
        msl_shape = load_mslp_data.isel(time=0).shape
        mask = np.zeros(msl_shape, dtype=np.int8)
        mask[80, 200] = 9  # centre of the domain
        mask[78:83, 218:223] = 1

        result = _project_data_about_extrema(
            mask,
            load_mslp_data.isel(time=0).values,
            lats=load_mslp_data["latitude"].values,
            lons=load_mslp_data["longitude"].values,
            max_km_extent=1000,
            grid_km_spacing=25,
            int_for_center=9,
        )
        assert result.shape[0] > 0
        assert not np.all(np.isnan(result))


class TestCompositeAboutExtrema:
    target_shape: tuple[int, int] = (161, 441)
    lats: np.ndarray = np.linspace(65.0, 25.0, 161)
    lons: np.ndarray = np.linspace(-100.0, 10.0, 441)
    time: np.ndarray = np.array(
        ["2025-12-29T00:00", "2025-12-29T06:00", "2025-12-29T12:00", "2025-12-29T18:00"], dtype="datetime64[h]"
    )

    @pytest.fixture
    def dummy_target_data(self) -> xr.DataArray:
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")
        data = np.stack([lat_grid + lon_grid] * len(self.time), axis=0)
        return xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": self.time,
                "latitude": self.lats,
                "longitude": self.lons,
            },
        )

    @pytest.fixture
    def dummy_extrema_data(self) -> xr.DataArray:
        n_extrema = len(self.time)
        data = np.zeros((n_extrema, self.target_shape[0], self.target_shape[1]), dtype=np.int8)

        for i in range(n_extrema):
            data[i, 8:13, 18:23] = 1  # region
            data[i, 10, 20] = 2  # centre

        return xr.DataArray(
            data,
            dims=["n_extrema", "latitude", "longitude"],
            coords={
                "n_extrema": np.arange(n_extrema),
                "time": ("n_extrema", self.time),
                "latitude": self.lats,
                "longitude": self.lons,
            },
        )

    def test_raises_if_lat_lon_missing_from_target(self, dummy_extrema_data) -> None:
        bad_target = xr.DataArray(
            np.ones((4, 10)),
            dims=["time", "x"],
            coords={"time": self.time},
        )
        with pytest.raises(ValueError, match="do not contain latitude and longitude"):
            composite_about_extrema(bad_target, dummy_extrema_data)

    def test_raises_if_vert_dim_not_in_target(self, dummy_target_data, dummy_extrema_data):
        with pytest.raises(ValueError, match="Expected 'pressure_level' to be present in target_data"):
            composite_about_extrema(dummy_target_data, dummy_extrema_data, vert_dim_name="pressure_level")

    def test_raises_if_n_extrema_not_in_extrema_data(self, dummy_target_data, dummy_extrema_data):
        bad_extrema = dummy_extrema_data.rename({"n_extrema": "wrong_dim"})
        with pytest.raises(ValueError, match="Expected 'n_extrema' to be in"):
            composite_about_extrema(dummy_target_data, bad_extrema)

    def test_raises_if_time_not_in_target(self, dummy_target_data, dummy_extrema_data):
        bad_target = dummy_target_data.rename({"time": "wrong_time"})
        with pytest.raises(ValueError, match="Expected 'time' to be present in target_data"):
            composite_about_extrema(bad_target, dummy_extrema_data)

    def test_raises_if_time_not_in_extrema_coords(self, dummy_target_data, dummy_extrema_data):
        bad_extrema = dummy_extrema_data.drop_vars("time")
        with pytest.raises(ValueError, match="Expected 'time' to be present in extrema_data coords"):
            composite_about_extrema(dummy_target_data, bad_extrema)

    def test_on_mslp_data(self, load_mslp_data) -> None:
        extrema_data = identify_and_stack_circular_extrema(
            load_mslp_data, ExtremaKind.MINIMA, extrema_threshold_value=100000
        )
        result = composite_about_extrema(load_mslp_data, extrema_data)

        # Check basic properties of the result
        assert isinstance(result, xr.DataArray)
        assert "y_km" in result.dims
        assert "x_km" in result.dims
        assert "time" in result.coords
        assert "n_extrema" in result["time"].dims

        # Check km_coords match the defaults for the composite function
        expected = np.arange(-2000, 2001, 25)
        np.testing.assert_array_equal(result["y_km"].values, expected)
        np.testing.assert_array_equal(result["x_km"].values, expected)

        # Obtained from manually inspecting the data
        num_minima_in_mslp: int = 22
        assert result["n_extrema"].size == num_minima_in_mslp

        # Taking the mean over `n_extrema` should not result in any NaNs
        msl_composite_mean = result.mean(dim="n_extrema", skipna=True) / 100
        assert not np.isnan(msl_composite_mean).any()

    def test_is_only_extrema_region_applies_where(self, dummy_target_data, dummy_extrema_data, mocker: "MockerFixture"):
        mock_where = mocker.patch.object(xr.DataArray, "where")
        composite_about_extrema(dummy_target_data, dummy_extrema_data, is_only_extrema_region=True)

        mock_where.assert_called_once()

    def test_is_only_extrema_region_false_does_not_apply_where(
        self, dummy_target_data, dummy_extrema_data, mocker: "MockerFixture"
    ):
        mock_where = mocker.patch.object(xr.DataArray, "where", wraps=xr.DataArray.where)
        composite_about_extrema(dummy_target_data, dummy_extrema_data, is_only_extrema_region=False)
        mock_where.assert_not_called()

    def test_with_vertical_dimension(self, dummy_target_data, dummy_extrema_data, mocker: "MockerFixture"):
        pressure = np.array([500.0, 850.0])
        target_4d = dummy_target_data.expand_dims({"pressure_level": pressure})
        result = composite_about_extrema(
            target_4d,
            dummy_extrema_data,
            vert_dim_name="pressure_level",
        )
        assert "pressure_level" in result.coords
        np.testing.assert_array_equal(result["pressure_level"], pressure)
