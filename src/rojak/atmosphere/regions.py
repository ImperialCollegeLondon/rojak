from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import assert_never, cast

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from numba import guvectorize, int8, njit, vectorize

from rojak.core.geometric import haversine_distance


def _region_labeller(
    target_array: np.ndarray, num_dim: int = 3, connectivity: int | None = None, structure: np.ndarray | None = None
) -> np.ndarray:
    """
    Labels connected regions

    Args:
        target_array: data to label
        num_dim: number of dimensions to identify structures in
        connectivity: number of neighbours which are considered to belong to central element. If None, it will be
        ``num_dim``
        structure: Instead of using :func:``scipy.ndimage.generate_binary_structure``, specify a custom structure

    Returns: labelled array

    Examples
    --------

    Modified from the :func:`scipy.ndimage.label` function documentation:

    >>> import numpy as np
    >>> a = np.array([[0,0,1,1,0,0],
    ...              [0,0,0,1,0,0],
    ...              [1,1,0,0,1,0],
    ...              [0,0,0,1,0,0]])
    >>> _region_labeller(a, num_dim=2, connectivity=1)
    array([[0, 0, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [2, 2, 0, 0, 3, 0],
           [0, 0, 0, 4, 0, 0]], dtype=int32)
    >>> _region_labeller(a, num_dim=2, connectivity=2)
    array([[0, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [2, 2, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0]], dtype=int32)
    """
    if structure is not None:
        if connectivity is not None:
            raise TypeError("If structure is specified, connectivity must be None")

        assert structure.dtype == bool, "Structure must be a boolean array"
    else:
        if connectivity is None:
            connectivity = num_dim
        structure = ndi.generate_binary_structure(rank=num_dim, connectivity=connectivity)

    assert num_dim > 1, "Minimum spatial dimension is 2D"

    # I'm not sure why pyright thinks there are type issues here...
    labeled, _ = ndi.label(  # pyright: ignore[reportGeneralTypeIssues]
        target_array, structure=structure
    )

    return labeled


MAX_SPATIAL_DIMS: int = 3


def _check_num_dims_and_set_core_dims(num_dims: int, core_dims: list[str] | None) -> list[str]:
    if num_dims > MAX_SPATIAL_DIMS or num_dims <= 1:
        raise ValueError("num_dims must be 2 or 3 as they are the spatial dimensions")

    if core_dims is None:
        core_dims = (
            ["longitude", "latitude", "pressure_level"] if num_dims == MAX_SPATIAL_DIMS else ["longitude", "latitude"]
        )
    else:
        assert len(core_dims) == num_dims

    return core_dims


def _check_dims_in_array(dims: list[str], array: xr.DataArray) -> None:
    assert set(dims).issubset(array.dims)


def _check_in_dims_and_coordinates(names: list[str], must_be_in: xr.DataArray) -> None:
    _check_dims_in_array(names, must_be_in)
    assert set(names).issubset(must_be_in.coords.keys())


def label_regions(
    array: xr.DataArray,
    num_dims: int = 3,
    core_dims: list[str] | None = None,
    connectivity: int | None = None,
    structure: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Labels connected regions

    Args:
        array: Array to label
        num_dims: Number of spatial dimension to do the labeling on
        core_dims: Name of core dimensions to iterate over
        connectivity: number of neighbours which are considered to belong to central element. If None, it will be
        ``num_dim``
        structure: Instead of using :func:``scipy.ndimage.generate_binary_structure``, specify a custom structure

    Returns:
        Labelled array

    """
    core_dims = _check_num_dims_and_set_core_dims(num_dims, core_dims)
    _check_dims_in_array(core_dims, array)

    return xr.apply_ufunc(
        _region_labeller,
        array,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
        dask="parallelized",
        kwargs={"num_dim": num_dims, "connectivity": connectivity, "structure": structure},
    )


@njit
def _parent_region_mask(labeled_array: np.ndarray, intersection_mask: np.ndarray) -> np.ndarray:
    original_shape = labeled_array.shape
    # numba does not support fancy indexing
    flattened_labeled = labeled_array.flatten()
    # Find the numbered region of the intersecting point. As the intersection can occur at multiple points,
    # np.unique is used to reduce it to the minimum so that the np.isin is slightly more efficient
    target_regions = np.unique(flattened_labeled[intersection_mask.flatten()])
    return np.isin(flattened_labeled, target_regions).reshape(original_shape)


@guvectorize("void(int32[:, :, :], bool_[:, :, :],bool_[:, :, :])", "(m,n,p),(m,n,p)->(m,n,p)")
def _parent_region_mask_3d_guv(labeled_array: np.ndarray, intersection_mask: np.ndarray, result: np.ndarray) -> None:
    original_shape = labeled_array.shape
    out_mask = _parent_region_mask(labeled_array, intersection_mask)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            for k in range(original_shape[2]):
                result[i, j, k] = out_mask[i, j, k]


@guvectorize("void(int32[:, :], bool_[:, :],bool_[:, :])", "(m,n),(m,n)->(m,n)")
def _parent_region_mask_2d_guv(labeled_array: np.ndarray, intersection_mask: np.ndarray, result: np.ndarray) -> None:
    original_shape = labeled_array.shape
    out_mask = _parent_region_mask(labeled_array, intersection_mask)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            result[i, j] = out_mask[i, j]


def find_parent_region_of_intersection(
    labeled_array: xr.DataArray,
    intersection_mask: xr.DataArray,
    num_dims: int = 3,
    core_dims: list[str] | None = None,
    numba_vectorize: bool = True,
) -> xr.DataArray:
    """
    Finds parent regions of intersection between two arrays

    Args:
        labeled_array: Labeled array
        intersection_mask: Boolean mask of where the arrays intersect at
        num_dims: Number of spatial dimensions the labeling was done on
        core_dims: Name of core dimensions to iterate over
        numba_vectorize: Boolean to control if numba vectorisation is used

    Returns:
        Boolean mask with the parent regions of the intersecting point

    """
    core_dims = _check_num_dims_and_set_core_dims(num_dims, core_dims)
    _check_dims_in_array(core_dims, labeled_array)
    _check_dims_in_array(core_dims, intersection_mask)

    if numba_vectorize:
        function_to_apply = _parent_region_mask_3d_guv if num_dims == MAX_SPATIAL_DIMS else _parent_region_mask_2d_guv
    else:
        function_to_apply = _parent_region_mask

    return xr.apply_ufunc(
        function_to_apply,
        labeled_array,
        intersection_mask,
        input_core_dims=[core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=not numba_vectorize,
        dask="parallelized",
        output_dtypes=[np.bool_],
    )


@vectorize([int8(int8, int8)])
def _bitwise_combine(first: int, second: int) -> int:
    return (first << 2) | (second << 1) | (first & second)


@guvectorize("void(int8[:, :, :, :], int8[:, :, :, :], int8[:, :, :, :])", "(m,n,p,q),(m,n,p,q)->(m,n,p,q)")
def _bitwise_combine_guv_4d(first: np.ndarray, second: np.ndarray, result: np.ndarray) -> None:
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            for k in range(first.shape[2]):
                for l_dim in range(first.shape[3]):
                    result[i, j, k, l_dim] = _bitwise_combine(first[i, j, k, l_dim], second[i, j, k, l_dim])


def _check_arrays_same_shape_and_bool(first: xr.DataArray, second: xr.DataArray) -> None:
    assert first.dtype == second.dtype
    assert first.dtype == np.bool_
    assert first.shape == second.shape


def combine_two_features(first: xr.DataArray, second: xr.DataArray, is_guv: bool = True) -> xr.DataArray:
    _check_arrays_same_shape_and_bool(first, second)

    # Cast to int to do bit twiddling
    first = first.astype("int8")
    second = second.astype("int8")

    # Set bits based on whether each feature is present
    # If only first,    0b100
    # If only second,   0b010
    # If both,          0b111
    # return (first << 2) | (second << 1) | (first & second)
    return xr.apply_ufunc(_bitwise_combine_guv_4d if is_guv else _bitwise_combine, first, second, dask="parallelized")


def _distance_metric_from_a_to_b(
    from_feature: np.ndarray, to_feature: np.ndarray, distance_func: Callable, **kwargs: float | Sequence[float] | str
) -> np.ndarray:
    from_feature = from_feature.astype(bool)
    to_feature = to_feature.astype(bool)
    # As return_distances has been passed to function, I can guarantee that a single numpy array is returned
    distance_to_feature: np.ndarray = cast("np.ndarray", distance_func(~to_feature, return_distances=True, **kwargs))
    distance_from_feature: np.ndarray = np.full_like(from_feature, np.nan, dtype=float)
    distance_from_feature[from_feature] = distance_to_feature[from_feature]
    return distance_from_feature


def euclidean_distance_from_a_to_b(
    from_feature: np.ndarray,
    to_feature: np.ndarray,
    sampling: float | Sequence[float] = 1,
) -> np.ndarray:
    """
    Euclidean distance from feature A to feature B

    For example, to find the distance of a contrail forming region to a turbulent region, then ``from_region`` would
    be a boolean array of where the contrail region is and ``to_region`` would be a boolean array of where
    turbulence is present.

    Note: Description of ``sampling`` arg has been copied from :func:`scipy.ndimage.distance_transform_edt`

    Args:
        from_feature: Feature to compute distances from
        to_feature: Feature to compute distances to
        sampling: Spacing of elements along each dimension. If a sequence, must be of length equal to the input rank;
        if a single number, this is used for all axes. If not specified, a grid spacing of unity is implied.

    Returns:
        Array of the closest distance from the point each feature A to feature B. Any points where feature A is not
        present will have the value ``np.nan``

    Examples
    --------

    Modified from the docstring of :func:`scipy.ndimage.distance_transform_edt`

    >>> to_b = np.array(([0,1,1,1,1],
    ...                  [0,0,1,1,1],
    ...                  [0,1,1,1,1],
    ...                  [0,1,1,1,0],
    ...                  [0,1,1,0,0]), dtype=bool)
    >>> to_b = ~to_b
    >>> from_a = np.ones_like(to_b)
    >>> euclidean_distance_from_a_to_b(from_a, to_b)
    array([[0.        , 1.        , 1.41421356, 2.23606798, 3.        ],
           [0.        , 0.        , 1.        , 2.        , 2.        ],
           [0.        , 1.        , 1.41421356, 1.41421356, 1.        ],
           [0.        , 1.        , 1.41421356, 1.        , 0.        ],
           [0.        , 1.        , 1.        , 0.        , 0.        ]])

    If the feature we are computing distances from is never present, then all values in the array will be ``np.nan``

    >>> np.isnan(euclidean_distance_from_a_to_b(np.zeros_like(to_b), to_b)).all()
    np.True_

    With a sampling of 2 units along x, 1 along y:

    >>> euclidean_distance_from_a_to_b(from_a, to_b, sampling=[2, 1])
    array([[0.        , 1.        , 2.        , 2.82842712, 3.60555128],
           [0.        , 0.        , 1.        , 2.        , 3.        ],
           [0.        , 1.        , 2.        , 2.23606798, 2.        ],
           [0.        , 1.        , 2.        , 1.        , 0.        ],
           [0.        , 1.        , 1.        , 0.        , 0.        ]])

    """
    return _distance_metric_from_a_to_b(from_feature, to_feature, ndi.distance_transform_edt, sampling=sampling)


def chebyshev_distance_from_a_to_b(from_feature: np.ndarray, to_feature: np.ndarray) -> np.ndarray:
    """
    Chebyshev distance from feature A to feature B

    For example, to find the distance of a contrail forming region to a turbulent region, then ``from_region`` would
    be a boolean array of where the contrail region is and ``to_region`` would be a boolean array of where
    turbulence is present.

    Args:
        from_feature: Feature to compute distances from
        to_feature: Feature to compute distances to

    Returns:
        Array of the closest distance from the point each feature A to feature B. Any points where feature A is not
        present will have the value ``np.nan``

    """
    return _distance_metric_from_a_to_b(from_feature, to_feature, ndi.distance_transform_cdt, metric="chessboard")


class DistanceMeasure(StrEnum):
    EUCLIDEAN = "euclidean"
    CHEBYSHEV = "chebyshev"


def distance_from_a_to_b(
    from_feature: xr.DataArray,
    to_feature: xr.DataArray,
    distance_measure: DistanceMeasure,
    num_dim: int = 3,
    core_dims: list[str] | None = None,
    sampling: float | Sequence[float] | None = None,
) -> xr.DataArray:
    """
    Distance from feature A to feature B

    For example, to find the distance of a contrail forming region to a turbulent region, then ``from_region`` would
    be a boolean array of where the contrail region is and ``to_region`` would be a boolean array of where
    turbulence is present.

    Args:
        from_feature: Feature to compute distances from
        to_feature: Feature to compute distances to
        distance_measure: How distance is measured
        num_dim: Number of dimensions to compute distances over
        core_dims: Name of core dimensions to iterate over
        sampling: Only applicable for Euclidean distance, see :func:`scipy.ndimage.distance_transform_edt` for details

    Returns:
        Array of the closest distance from each point in feature A to feature B

    """
    _check_arrays_same_shape_and_bool(from_feature, to_feature)

    core_dims = _check_num_dims_and_set_core_dims(num_dim, core_dims)
    _check_dims_in_array(core_dims, from_feature)
    _check_dims_in_array(core_dims, to_feature)

    if distance_measure != DistanceMeasure.EUCLIDEAN and sampling is not None:
        raise ValueError("Sampling is only supported for euclidean distance")

    if distance_measure != DistanceMeasure.EUCLIDEAN:
        func_kwargs = {}
    else:
        func_kwargs = {"sampling": sampling} if sampling is not None else {"sampling": 1}

    match distance_measure:
        case DistanceMeasure.EUCLIDEAN:
            distance_function: Callable = euclidean_distance_from_a_to_b
        case DistanceMeasure.CHEBYSHEV:
            distance_function: Callable = chebyshev_distance_from_a_to_b
        case _ as unreachable:
            assert_never(unreachable)

    return xr.apply_ufunc(
        distance_function,
        from_feature,
        to_feature,
        kwargs=func_kwargs,
        input_core_dims=[core_dims, core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
        output_dtypes=[np.dtype(float)],
        dask="parallelized",
    )


def nearest_haversine_distance(
    from_feature: np.ndarray,
    to_feature: np.ndarray,
    lat_coords_1d: np.ndarray,
    lon_coords_1d: np.ndarray,
    /,
) -> np.ndarray:
    # Short-circuit on the trivial case
    if not np.any(from_feature) or not np.any(to_feature):
        return np.full(from_feature.shape, np.nan, dtype=float)

    # Indices of the nearest to_feature point
    #    indices_to_nearest[0] will be row indices (for latitude)
    #    indices_to_nearest[1] will be column indices (for longitude)
    indices_to_nearest: np.ndarray = cast(
        "np.ndarray", ndi.distance_transform_edt(~to_feature, return_distances=False, return_indices=True)
    )

    # Maps the indices to the lat and lon values from the coordinate
    nearest_lat = lat_coords_1d[indices_to_nearest[0]]
    nearest_lon = lon_coords_1d[indices_to_nearest[1]]

    # Construct base grid to compute distances from
    source_lon_grid, source_lat_grid = np.meshgrid(lon_coords_1d, lat_coords_1d)
    distance = haversine_distance(
        source_lon_grid,
        source_lat_grid,
        nearest_lon,
        nearest_lat,
    )
    # Mask out points that are not in from_feature
    distance[~from_feature] = np.nan

    return distance


def shortest_haversine_distance_from_a_to_b(
    from_feature: xr.DataArray,
    to_feature: xr.DataArray,
    /,
    longitude_dim_name: str = "longitude",
    latitude_dim_name: str = "latitude",
) -> xr.DataArray:
    _check_arrays_same_shape_and_bool(from_feature, to_feature)

    horizontal_dims = [latitude_dim_name, longitude_dim_name]
    _check_in_dims_and_coordinates(horizontal_dims, from_feature)
    _check_in_dims_and_coordinates(horizontal_dims, to_feature)

    from_feature = from_feature.astype(bool)
    to_feature = to_feature.astype(bool)

    return xr.apply_ufunc(
        nearest_haversine_distance,
        from_feature,
        to_feature,
        from_feature[latitude_dim_name],
        from_feature[longitude_dim_name],
        input_core_dims=[
            horizontal_dims,
            horizontal_dims,
            [latitude_dim_name],
            [longitude_dim_name],
        ],
        output_core_dims=[horizontal_dims],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.dtype(float)],  # Use float for distance, not original dtype
    ).rename("shortest_haversine_distance")
