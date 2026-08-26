from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, assert_never, cast

import numpy as np
import pyproj
import scipy.interpolate as si
import scipy.ndimage as ndi
import xarray as xr
from numba import guvectorize, int8, njit, vectorize

from rojak.core.geometric import haversine_distance
from rojak.utilities.types import is_xr_data_array


def _region_labeller(target_array: np.ndarray, num_dim: int = 3, connectivity: int | None = None) -> np.ndarray:
    """
    Labels connected regions

    Args:
        target_array: data to label
        num_dim: number of dimensions to identify structures in
        connectivity: number of neighbours which are considered to belong to central element. If None, it will be
        ``num_dim``

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
    if connectivity is None:
        connectivity = num_dim

    assert num_dim > 1, "Minimum spatial dimension is 2D"

    # I'm not sure why pyright thinks there are type issues here...
    labeled, _ = ndi.label(  # pyright: ignore[reportGeneralTypeIssues]
        target_array,
        structure=ndi.generate_binary_structure(rank=num_dim, connectivity=connectivity),
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


def _check_dims_in_array(dims: list[str], array: xr.DataArray | xr.Dataset) -> None:
    assert set(dims).issubset(array.dims)


def _check_in_dims_and_coordinates(names: list[str], must_be_in: xr.DataArray | xr.Dataset) -> None:
    _check_dims_in_array(names, must_be_in)
    assert set(names).issubset(must_be_in.coords.keys())


def label_regions(
    array: xr.DataArray,
    num_dims: int = 3,
    core_dims: list[str] | None = None,
    connectivity: int | None = None,
) -> xr.DataArray:
    """
    Labels connected regions

    Args:
        array: Array to label
        num_dims: Number of spatial dimension to do the labeling on
        core_dims: Name of core dimensions to iterate over
        connectivity: number of neighbours which are considered to belong to central element. If None, it will be
        ``num_dim``

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
        kwargs={"num_dim": num_dims, "connectivity": connectivity},
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


class DistanceMode(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


def vertical_distance_to_positive(
    target_data: xr.DataArray | xr.Dataset,
    /,
    *,
    vertical_coord_name: str = "pressure_level",
    distance_mode: DistanceMode = DistanceMode.RELATIVE,
) -> xr.DataArray | xr.Dataset:
    _check_in_dims_and_coordinates([vertical_coord_name], target_data)

    if is_xr_data_array(target_data):
        assert target_data.dtype == np.dtype(bool)
    else:
        assert set(target_data.dtypes.values()) == {np.dtype(bool)}

    vert_src_name: str = "vertical_source"
    vert_target_name: str = "vertical_target"

    vertical_coord: np.ndarray = target_data[vertical_coord_name].to_numpy()
    vertical_coord_size: int = vertical_coord.size
    offset: np.ndarray
    match distance_mode:
        case DistanceMode.ABSOLUTE:
            offset = np.abs(vertical_coord[:, np.newaxis] - vertical_coord[np.newaxis, :])
        case DistanceMode.RELATIVE:
            offset = np.abs(np.arange(vertical_coord_size)[:, np.newaxis] - np.arange(vertical_coord_size))
        case _ as unreachable:
            assert_never(unreachable)
    offset_amounts = xr.DataArray(
        offset,
        dims=[vert_target_name, vert_src_name],
        coords={vert_target_name: vertical_coord, vert_src_name: vertical_coord},
    )

    expanded: xr.DataArray | xr.Dataset = target_data.astype(int).rename({vertical_coord_name: vert_src_name})
    return (
        xr.where(expanded, offset_amounts, np.inf)
        .min(dim=vert_src_name)
        .rename({vert_target_name: vertical_coord_name})
    )


def shortest_vertical_distance_from_a_to_b(
    from_feature: xr.DataArray,
    to_feature: xr.DataArray,
    /,
    *,
    vertical_coord_name: str = "pressure_level",
    distance_mode: DistanceMode = DistanceMode.RELATIVE,
) -> xr.DataArray | xr.Dataset:
    _check_arrays_same_shape_and_bool(from_feature, to_feature)
    _check_in_dims_and_coordinates([vertical_coord_name], from_feature)
    _check_in_dims_and_coordinates([vertical_coord_name], to_feature)

    from_feature = from_feature.astype(bool)
    return vertical_distance_to_positive(
        from_feature & to_feature, vertical_coord_name=vertical_coord_name, distance_mode=distance_mode
    )


class ExtremaKind(StrEnum):
    """Enumeration of extrema types for filtering operations.

    Attributes:
        MINIMA: Represents local minima.
        MAXIMA: Represents local maxima.
    """

    MINIMA = "minima"
    MAXIMA = "maxima"

    def use_less_than_on_threshold(self) -> bool:
        match self:
            case ExtremaKind.MINIMA:
                return True
            case ExtremaKind.MAXIMA:
                return False
            case _ as unreachable:
                assert_never(unreachable)


def apply_extrema_filter(
    target_data: xr.DataArray,
    extrema_kind: ExtremaKind,
    latitude_dim_name: str = "latitude",
    longitude_dim_name: str = "longitude",
    **filter_kwargs: Any,  # noqa: ANN401
) -> xr.DataArray:
    """Apply a spatial extrema filter to a 3D DataArray.

    Applies either a maximum or minimum filter from :mod:`scipy.ndimage` over the latitude and longitude dimensions of
    a 3D DataArray, with support for Dask-backed arrays.

    Args:
        target_data (:class:`xarray.DataArray`): A 3D DataArray to apply the filter to.
            Must contain the specified latitude and longitude dimensions.
        extrema_kind (:class:`ExtremaKind`): The type of extrema filter to apply, either
            :attr:`ExtremaKind.MAXIMA` or :attr:`ExtremaKind.MINIMA`.
        latitude_dim_name (str): Name of the latitude dimension in ``target_data``.
            Defaults to ``"latitude"``.
        longitude_dim_name (str): Name of the longitude dimension in ``target_data``.
            Defaults to ``"longitude"``.
        **filter_kwargs: Additional keyword arguments passed to
            :func:`scipy.ndimage.maximum_filter` or :func:`scipy.ndimage.minimum_filter`.

    Returns:
        :class:`xarray.DataArray`: A DataArray of the same shape and dtype as
        ``target_data`` with the extrema filter applied over the latitude and
        longitude dimensions.

    Raises:
        ValueError: If ``target_data`` is not 3D.
        ValueError: If ``target_data`` does not contain the specified latitude and longitude dimensions.
    """
    if target_data.ndim != 3:  # noqa: PLR2004
        raise ValueError("Target data must be 3D")

    if not set(target_data.dims).issuperset({longitude_dim_name, latitude_dim_name}):
        raise ValueError("Target data must contain the specified longitude and latitude dimensions")

    extrema_func = ndi.maximum_filter if extrema_kind == ExtremaKind.MAXIMA else ndi.minimum_filter
    return xr.apply_ufunc(
        extrema_func,
        target_data,
        kwargs=filter_kwargs,
        input_core_dims=[[latitude_dim_name, longitude_dim_name]],
        output_core_dims=[[latitude_dim_name, longitude_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[target_data.dtype],
    )


def circular_footprint(radius: int) -> np.ndarray:
    """Create a circular boolean footprint for use with :mod:`scipy` morphological filters.

    Args:
        radius (int): Radius of the circle in grid points.

    Returns:
        :class:`numpy.ndarray`: A boolean NumPy ndarray of shape ``(2*radius+1, 2*radius+1)``
        where ``True`` values represent points within the circle.

    Raises:
        ValueError: If ``radius`` is :math:`\\leq 0`.

    Example:
    >>> footprint = circular_footprint(radius=2)
    >>> footprint.astype(int)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
    """
    if radius <= 0:
        raise ValueError("Radius must be positive")

    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2


def __extrema_mask_region(
    extrema_locations: np.ndarray,
    extrema_filtered: np.ndarray,
) -> np.ndarray:
    """Computes mask of extrema regions from an extrema filtered array

    Identifies grid points belonging to extrema regions whose values have been filtered by :func:`apply_extrema_filter`

    Note:
        This is a private function intended to be used via
        :func:`xarray.apply_ufunc` within :func:`identify_circular_extrema`.

    Args:
        extrema_locations (:class:`numpy.ndarray`): A boolean array indicating the locations of local extrema,
            where ``True`` marks an extremum.
        extrema_filtered (:class:`numpy.ndarray`): An array of the same shape as ``extrema_locations`` containing
            the extrema-filtered values.

    Returns:
        :class:`numpy.ndarray`: A boolean array of the same shape as ``extrema_filtered``, where ``True`` indicates
        grid points belonging to a qualifying extrema region.
    """
    mask_values = extrema_filtered[extrema_locations]

    return (
        np.isin(extrema_filtered, mask_values) if mask_values.size != 0 else np.zeros_like(extrema_filtered, dtype=bool)
    )


def __identify_extrema_locations(
    target_data: xr.DataArray,
    extrema_kind: ExtremaKind,
    extrema_threshold_value: float | None,
    lat_dim_name: str,
    lon_dim_name: str,
    **filter_kwargs: Any,  # noqa: ANN401
) -> tuple[xr.DataArray, xr.DataArray]:
    filter_applied: xr.DataArray = apply_extrema_filter(
        target_data,
        extrema_kind,
        latitude_dim_name=lat_dim_name,
        longitude_dim_name=lon_dim_name,
        **filter_kwargs,
    ).persist()
    extrema_locations: xr.DataArray = target_data == filter_applied
    if extrema_threshold_value is not None:
        extrema_locations = extrema_locations & (
            filter_applied < extrema_threshold_value
            if extrema_kind.use_less_than_on_threshold()
            else filter_applied > extrema_threshold_value
        )
    return filter_applied, extrema_locations


def identify_circular_extrema(
    target_data: xr.DataArray,
    extrema_kind: ExtremaKind,
    extrema_threshold_value: float | None = None,
    footprint_radius: int = 24,
    sel_region_indexer: dict | None = None,
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    **filter_kwargs: Any,  # noqa: ANN401
) -> xr.Dataset:
    """Identify circular extrema regions in a 3D spatial DataArray.

    Detects local minima or maxima using a circular footprint filter and returns both the point locations of the
    extrema and the broader regions associated with them, filtered by a threshold condition.

    See Also:
        - :class:`ExtremaKind`: Enumeration of extrema types.
        - :func:`apply_extrema_filter`: The underlying extrema filter function.
        - :func:`circular_footprint`: Generates the circular footprint used for filtering.

    Args:
        target_data (:class:`xarray.DataArray`): A 3D DataArray to identify extrema in. Must contain the specified
            latitude and longitude dimensions.
        extrema_kind (:class:`ExtremaKind`): The type of extrema to identify, either :attr:`ExtremaKind.MAXIMA` or
            :attr:`ExtremaKind.MINIMA`.
        extrema_threshold_value (float, optional): The threshold value used to filter extrema regions. Combined with
            ``is_threshold_less_than`` to determine which extrema are retained.
            Defaults to ``None``.
        footprint_radius (int): Radius in grid points of the circular footprint used for the extrema filter.
            Defaults to ``24``.
        sel_region_indexer (dict, optional): Optional dictionary of indexers passed to :meth:`xarray.DataArray.sel`
            to subset ``target_data`` to a specific region before processing. Defaults to ``None``.
        lat_dim_name (str): Name of the latitude dimension in ``target_data``.
            Defaults to ``"latitude"``.
        lon_dim_name (str): Name of the longitude dimension in ``target_data``.
            Defaults to ``"longitude"``.
        **filter_kwargs: Additional keyword arguments passed to the underlying
            scipy filter via :func:`apply_extrema_filter`.

    Returns:
        :class:`xarray.Dataset`: A Dataset with the same coordinates as ``target_data``
        containing:

        - **local_extrema** (:class:`xarray.DataArray`): A boolean DataArray indicating the locations of local extrema,
          where ``True`` marks an extremum.
        - **extrema_regions** (:class:`xarray.DataArray`): A boolean DataArray indicating grid points belonging to
          qualifying extrema regions after threshold filtering.

    Raises:
        ValueError: If ``target_data`` is not 3D.
        ValueError: If ``target_data`` does not contain the specified latitude and longitude dimensions.

    Example:
        .. code-block:: python

            result = identify_circular_extrema(
                target_data=my_data_array,
                extrema_kind=ExtremaKind.MINIMA,
                extrema_threshold_value=500.0,
                footprint_radius=24,
            )
            local_extrema = result["local_extrema"]
            extrema_regions = result["extrema_regions"]
    """
    if target_data.ndim != 3:  # noqa: PLR2004
        raise ValueError("Target data must be 3D")

    if not set(target_data.dims).issuperset({lat_dim_name, lon_dim_name}):
        raise ValueError("Target data must contain the specified latitude and longitude dimensions")

    if sel_region_indexer is not None:
        target_data = target_data.sel(indexers=sel_region_indexer)

    filter_applied, extrema_locations = __identify_extrema_locations(
        target_data,
        extrema_kind,
        extrema_threshold_value,
        lat_dim_name,
        lon_dim_name,
        footprint=circular_footprint(radius=footprint_radius),
        axes=(0, 1),
        **filter_kwargs,
    )

    extrema_regions = xr.apply_ufunc(
        __extrema_mask_region,
        extrema_locations,
        filter_applied,
        input_core_dims=[[lat_dim_name, lon_dim_name], [lat_dim_name, lon_dim_name]],
        output_core_dims=[[lat_dim_name, lon_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    )

    return xr.Dataset(
        data_vars={
            "local_extrema": extrema_locations,
            "extrema_regions": extrema_regions,
        },
        coords=target_data.coords,
    )


def __stack_extrema_vectorised(
    extrema_locations: np.ndarray,  # (lat, lon) bool
    extrema_filtered: np.ndarray,  # (lat, lon) int/float - labelled regions
) -> np.ndarray:  # (n_extrema, lat, lon) int8
    """Expand a 2D labelled extrema field into a 3D binary mask stack.

    For each detected extremum, produces a 2D spatial slice in which pixels
    belonging to that extremum's labelled region are marked ``1`` and the
    extremum's centre pixel is marked ``2``. All other pixels are ``0``.

    Intended to be called via :func:`xarray.apply_ufunc` with ``vectorize=True``
    so that it operates on a single ``(lat, lon)`` snapshot at a time.

    Args:
        extrema_locations (numpy.ndarray): Boolean mask of shape ``(lat, lon)``
            whose ``True`` entries are the centre points of detected extrema.
        extrema_filtered (numpy.ndarray): Labelled region array of shape
            ``(lat, lon)`` - e.g. the output of :func:`scipy.ndimage.label` -
            where each connected region belonging to an extremum carries a
            unique non-zero integer label. Pixels outside any region are ``0``.

    Returns:
        numpy.ndarray: Stacked mask array of shape ``(n_extrema, lat, lon)``
        and dtype ``int8``. For the ``i``-th extremum:

        - ``0`` - pixel does not belong to this extremum's region.
        - ``1`` - pixel belongs to this extremum's labelled region.
        - ``2`` - pixel is the centre of this extremum.
    """
    extrema_values = extrema_filtered[extrema_locations]  # (n_extrema,)
    indices = np.argwhere(extrema_locations)  # (n_extrema, 2)
    num_extrema: int = len(extrema_values)

    # Vectorised comparison by broadcasting (n_extrema, 1, 1) with (1, lat, lon)
    # extrema_values[:, None, None] shape: (n_extrema, 1,   1  )
    # extrema_filtered[None, :, :] shape:  (1,         lat, lon)
    # result shape:                        (n_extrema, lat, lon)
    output_array: np.ndarray = (extrema_filtered[None, :, :] == extrema_values[:, None, None]).astype(np.int8)

    # Set extrema centers to be equal to 2
    output_array[
        np.arange(num_extrema),  # extrema index
        indices[:, 0],  # lat indices
        indices[:, 1],  # lon indices
    ] = 2

    return output_array


def _apply_and_combine_on_n_extrema_groups(
    target_ds: xr.Dataset,
    *,
    lat_dim_name: str,
    lon_dim_name: str,
    new_dim_name: str,
) -> xr.DataArray:
    """Apply :func:`__stack_extrema_vectorised` to a group and return a stacked result.

    This is the *apply* step of a split-apply-combine strategy executed via
    :meth:`xarray.Dataset.groupby` and :meth:`xarray.core.groupby.DatasetGroupBy.map`.
    Each ``target_ds`` group contains only timesteps that share the same number
    of extrema, which keeps the output shape of :func:`xarray.apply_ufunc`
    uniform within the group, a hard requirement for ``dask="parallelized"``.

    After per-timestep expansion, the new extrema index dimension and the
    ``"time"`` dimension are stacked into a :class:`pandas.MultiIndex` so that
    xarray's built-in combine logic can concatenate results across groups
    correctly.

    Args:
        target_ds (xr.Dataset): A group-slice of the full dataset. Must contain:

            - ``"extrema_locations"`` (:class:`xarray.DataArray`, shape
              ``(..., lat, lon)``, dtype ``bool``) - centre points of detected
              extrema.
            - ``"filtered"`` (:class:`xarray.DataArray`, shape
              ``(..., lat, lon)``, dtype ``int`` or ``float``) - labelled
              extrema regions.
            - ``"n_extrema"`` (scalar coordinate) - the number of extrema
              present in every timestep of this group.

        lat_dim_name (str): Name of the latitude dimension in ``target_ds``.
        lon_dim_name (str): Name of the longitude dimension in ``target_ds``.
        new_dim_name (str): Name assigned to the newly created per-extremum
            index dimension (e.g. ``"extrema_count"``). Its size equals the
            number of extrema in this group.

    Returns:
        xr.DataArray: Expanded mask array stacked along a
        :class:`pandas.MultiIndex` of ``(new_dim_name, "time")``, exposed
        under the dimension name ``"n_extrema"``. Pixel values follow the
        ``0 / 1 / 2`` encoding of :func:`__stack_extrema_vectorised`.
    """
    num_extrema_in_group = target_ds["n_extrema"].values[0]
    extrema_regions: xr.DataArray = xr.apply_ufunc(
        __stack_extrema_vectorised,
        target_ds["extrema_locations"],
        target_ds["filtered"],
        input_core_dims=[[lat_dim_name, lon_dim_name], [lat_dim_name, lon_dim_name]],
        output_core_dims=[
            [new_dim_name, lat_dim_name, lon_dim_name]
        ],  # create a new dimension which is [0, num_extrema_for_group)
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8],
        dask_gufunc_kwargs={"output_sizes": {new_dim_name: num_extrema_in_group}},
    )
    # Stacking to multi-index allows for the Groupby.map() to combine the datasets. See xarray docs for stacking rules
    return extrema_regions.stack(n_extrema=(new_dim_name, "time"))


def identify_and_stack_circular_extrema(
    target_data: xr.DataArray,
    extrema_kind: ExtremaKind,
    extrema_threshold_value: float | None = None,
    footprint_radius: int = 24,
    sel_region_indexer: dict | None = None,
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    time_dim_name: str = "time",
    **filter_kwargs: Any,  # noqa: ANN401
) -> xr.DataArray:
    """Identify circular extrema in a 3D field and return a per-extremum mask stack.

    Detects spatial extrema (minima or maxima) in ``target_data`` using a
    circular structuring element, labels the connected regions surrounding each
    extremum, and expands the result so that every detected extremum occupies
    its own index along a new ``"n_extrema"`` dimension. The output therefore
    encodes both which grid cells belong to each extremum and when each
    extremum occurred.

    Processing follows a split-apply-combine pattern to accommodate the
    variable number of extrema per timestep:

    1. Split - group timesteps by their extrema count so that
       :func:`xarray.apply_ufunc` always receives arrays of a known, fixed
       output size.
    2. Apply - call :func:`__stack_extrema_vectorised` on each group via
       :func:`_apply_and_combine_on_n_extrema_groups`.
    3. Combine - xarray concatenates the groups; results are sorted by
       time and re-indexed with a flat ``n_extrema`` integer coordinate.

    Args:
        target_data (xr.DataArray): Three-dimensional input field. Must have
            exactly the dimensions ``{lat_dim_name, lon_dim_name, time_dim_name}``.
        extrema_kind (ExtremaKind): Whether to detect minima or maxima.
            Passed directly to the internal ``__identify_extrema_locations``
            helper.
        extrema_threshold_value (float, optional): Minimum absolute value a
            grid point must exceed to qualify as an extremum. ``None`` disables
            thresholding. Defaults to ``None``.
        footprint_radius (int): Radius in grid cells of the circular
            structuring element used for the morphological extrema filter.
            Defaults to ``24``.
        sel_region_indexer (dict, optional): If provided, passed to
            :meth:`xarray.DataArray.sel` to spatially subset ``target_data``
            before processing. Defaults to ``None``.
        lat_dim_name (str): Name of the latitude dimension.
            Defaults to ``"latitude"``.
        lon_dim_name (str): Name of the longitude dimension.
            Defaults to ``"longitude"``.
        time_dim_name (str): Name of the time dimension.
            Defaults to ``"time"``.
        **filter_kwargs (Any): Additional keyword arguments forwarded to
            ``__identify_extrema_locations`` (e.g. smoothing parameters).

    Returns:
        xr.DataArray: Array with dimensions
        ``(n_extrema, lat_dim_name, lon_dim_name)`` and coordinates:

        - ``n_extrema`` - flat integer index ``[0, N)`` where ``N`` is the
          total number of extrema across all timesteps.
        - ``time`` - source timestamp of each extremum, attached to the
          ``n_extrema`` dimension.

        Pixel values follow the ``0 / 1 / 2`` encoding described in
        :func:`__stack_extrema_vectorised`.

    Raises:
        ValueError: If ``target_data`` is not 3D.
        ValueError: If the dimensions of ``target_data`` do not match the
            specified ``lat_dim_name``, ``lon_dim_name``, and
            ``time_dim_name``.
    """
    if target_data.ndim != 3:  # noqa: PLR2004
        raise ValueError("Target data must be 3D")

    if not set(target_data.dims) == {lat_dim_name, lon_dim_name, time_dim_name}:
        raise ValueError("Target data must contain the specified time, latitude and longitude dimensions")

    if sel_region_indexer is not None:
        target_data = target_data.sel(indexers=sel_region_indexer)

    filter_applied, extrema_locations = __identify_extrema_locations(
        target_data,
        extrema_kind,
        extrema_threshold_value,
        lat_dim_name,
        lon_dim_name,
        footprint=circular_footprint(radius=footprint_radius),
        axes=(0, 1),
        **filter_kwargs,
    )

    # Count how many extrema are in a given point in time
    n_composite_dim = extrema_locations.sum(dim=["longitude", "latitude"]).compute().values
    # Create a dataset so that both arrays can be used at the GroupBy stage
    extrema_ds = xr.Dataset(
        data_vars={"filtered": filter_applied, "extrema_locations": extrema_locations},
        coords=target_data.coords,
    )

    # New dim occurs from expanding out the extrema from a single time step
    extrema_group_idx_dim: str = "extrema_count"
    extrema_global_idx_dim: str = "n_extrema"

    # Add new `n_extrema` coord that is attached to the existing time dimension - ensure that when the data is split
    #   in the GroupBy, the information about the tiem is still there.
    extrema_ds = extrema_ds.assign_coords({extrema_global_idx_dim: ("time", n_composite_dim)})

    # groupby("n_extrema"): This ensures that each item in a group only has a fixed number of extrema. This allows for
    #                       xr.apply_ufunc() to be used as each iteration over the non-lat-lon dim results in the array
    #                       growing by the same amount.
    #                       This is the "split" in the split-apply-combine strategy
    # map():                Operation on the GroupBy object which applies a function to each group and concats them
    #                       together.
    #                       This method invoked in the "apply" part of the strategy and setting things up so that the
    #                       xr built-in combine method can assemble the new dataset
    # sortby():             The groupby results in the data being in a different order. So, this is just to make it
    #                       consistent with the rest of the data to have time increase linearly
    # pyright flags that this is xr.Dataset and from the types in the docs it should be. However, the `map()` returns
    # a xr.DataArray so the output of the `map()` is an xr.DataArray
    n_extrema_indexed = (
        extrema_ds.groupby(extrema_global_idx_dim)
        .map(
            _apply_and_combine_on_n_extrema_groups,  # pyright: ignore[reportArgumentType]
            lat_dim_name=lat_dim_name,
            lon_dim_name=lon_dim_name,
            new_dim_name=extrema_group_idx_dim,
        )
        .sortby(time_dim_name)
    ).persist()
    time_values = n_extrema_indexed[time_dim_name].values
    return n_extrema_indexed.drop_vars([extrema_global_idx_dim, extrema_group_idx_dim, time_dim_name]).assign_coords(
        n_extrema=np.arange(time_values.size), time=(extrema_global_idx_dim, time_values)
    )  # pyright: ignore[reportReturnType]


def __project_data_about_extrema(
    extrema_mask: np.ndarray,
    data_values: np.ndarray,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    max_km_extent: float,
    grid_km_spacing: float,
    int_for_center: int = 2,
    lat_lon_buffer: float = 0.5,
) -> np.ndarray:
    """Re-project a 2D data field onto a km-scale Cartesian grid centred on an extremum.

    Locates the extremum centre in ``extrema_mask``, constructs a WGS-84 orthographic projection
    (:class:`pyproj.Proj`) anchored at that centre, and interpolates ``data_values`` from its native lat/lon grid
    onto a regular ``(y_km, x_km)`` Cartesian grid using bilinear interpolation via
    :func:`scipy.interpolate.griddata`.

    The re-projection pipeline is:

    1. Identify the centre pixel (value == ``int_for_center``) in ``extrema_mask``.
    2. Build a symmetric km output grid spanning ``[-max_km_extent, +max_km_extent]``.
    3. Use the inverse orthographic projection to find which lat/lon points fall within the output grid extent
       (plus ``lat_lon_buffer``).
    4. Use the forward projection to convert those native lat/lon points to km coordinates.
    5. Interpolate the native data onto the regular km grid with :func:`scipy.interpolate.griddata`.

    Intended to be called via :func:`xarray.apply_ufunc` with ``vectorize=True`` so that it operates on a single
    ``(lat, lon)`` slice at a time.

    Args:
        extrema_mask (numpy.ndarray): Mask array of shape ``(lat, lon)`` and dtype ``int8``, using the
            ``0 / 1 / 2`` encoding produced by :func:`__stack_extrema_vectorised`. The pixel with value
            ``int_for_center`` is taken as the extremum centre.
        data_values (numpy.ndarray): Geophysical field to re-project, of shape ``(lat, lon)``
            (e.g. geopotential height, wind speed).
        lats (numpy.ndarray): 1D array of latitude values in degrees, corresponding to axis 0 of
            ``extrema_mask`` and ``data_values``.
        lons (numpy.ndarray): 1D array of longitude values in degrees, corresponding to axis 1 of ``extrema_mask``
            and ``data_values``.
        max_km_extent (float): Half-width of the output Cartesian grid in kilometres. The grid spans
            ``[-max_km_extent, +max_km_extent]`` in both x and y.
        grid_km_spacing (float): Spacing between output grid points in kilometres.
        int_for_center (int): Pixel value in ``extrema_mask`` that identifies the extremum centre.
            Defaults to ``2``.
        lat_lon_buffer (float): Extra margin in degrees added when subsetting the input lat/lon arrays before
            interpolation, to avoid edge artefacts near the projection boundary. Defaults to ``0.5``.

    Returns:
        numpy.ndarray: Data field interpolated onto the symmetric km grid, of shape ``(n_km, n_km)`` where
        ``n_km = len(np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing))``.
        Returns an array filled with ``NaN`` if no extremum centre is found in ``extrema_mask``.

    Raises:
        ValueError: If more than one extremum centre pixel (i.e. pixels equal to ``int_for_center``) is found in
        ``extrema_mask``.
    """
    km_coords = np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing)
    n_km = len(km_coords)

    # Identify the centres of extrema
    lat_indices, lon_indices = np.nonzero(extrema_mask == int_for_center)
    if len(lat_indices) == 0:
        return np.full((n_km, n_km), np.nan)

    if len(lat_indices) != 1:
        raise ValueError("There should only be on extrema per 2D slice")

    # lat is axis 0, lon is axis 1 -> due to order of input_core_dims
    centre_lat = float(lats[lat_indices[0]])
    centre_lon = float(lons[lon_indices[0]])

    # km grid that we want to map the data onto
    x_km_mesh, y_km_mesh = np.meshgrid(km_coords, km_coords)

    # Inverse projection to find what lat/lon indices are within range
    ortho = pyproj.Proj(proj="ortho", lat_0=centre_lat, lon_0=centre_lon, units="km", ellps="WGS84")
    new_lons, new_lats = ortho(x_km_mesh, y_km_mesh, inverse=True)

    lat_mask = (lats >= new_lats.min() - lat_lon_buffer) & (lats <= new_lats.max() + lat_lon_buffer)
    lon_mask = (lons >= new_lons.min() - lat_lon_buffer) & (lons <= new_lons.max() + lat_lon_buffer)
    lats_within = lats[lat_mask]
    lons_within = lons[lon_mask]

    # data_masked = np.where(extrema_mask > 0, data_values, np.nan)
    data_within = data_values[np.ix_(lat_mask, lon_mask)]

    # Forward projection identify what these data points are in km so that scipy.griddata() can use to to interpolate
    # to our desired grid system
    lon_grid, lat_grid = np.meshgrid(lons_within, lats_within)
    x_km_within, y_km_within = ortho(lon_grid, lat_grid)

    return si.griddata(
        points=(x_km_within.ravel(), y_km_within.ravel()),
        values=data_within.ravel(),
        xi=(x_km_mesh.ravel(), y_km_mesh.ravel()),
        method="linear",
    ).reshape(n_km, n_km)


def composite_about_extrema(
    target_data: xr.DataArray,
    extrema_data: xr.DataArray,
    *,
    time_dim_name: str = "time",
    num_extrema_dim: str = "n_extrema",
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    vert_dim_name: str | None = None,
    max_km_extent: float = 2000,
    grid_km_spacing: float = 25,
    int_for_center: int = 2,
    is_only_extrema_region: bool = False,
) -> xr.DataArray:
    """Composite a geophysical field onto a common km-scale grid centred on each extremum.

    For every extremum recorded in ``extrema_data``, the corresponding timestep of ``target_data`` is extracted and
    re-projected onto a regular Cartesian grid (in km) centred on that extremum using a WGS-84 orthographic
    projection. The result is a stack of re-projected fields - one per extremum - that can subsequently be averaged
    to produce a mean composite structure.

    Optionally supports a vertical dimension, in which case the re-projection is applied independently at each
    level via :func:`xarray.apply_ufunc` with ``vectorize=True``.

    Args:
        target_data (xr.DataArray): Geophysical field to composite. Must contain at minimum the dimensions
            ``lat_dim_name``, ``lon_dim_name``, and ``time_dim_name``. If ``vert_dim_name`` is provided, that
            dimension must also be present.
        extrema_data (xr.DataArray): Per-extremum mask array, as returned by
            :func:`identify_and_stack_circular_extrema`. Must contain: 1) Dimensions ``num_extrema_dim``,
            ``lat_dim_name``, ``lon_dim_name``, and 2) a ``time`` coordinate indexed by ``num_extrema_dim`` that
            maps each extremum to its source timestamp in ``target_data``.
        time_dim_name (str): Name of the time dimension in ``target_data`` and the time coordinate in
            ``extrema_data``.
            Defaults to ``"time"``.
        num_extrema_dim (str): Name of the dimension in ``extrema_data`` that indexes individual extrema.
            Defaults to ``"n_extrema"``.
        lat_dim_name (str): Name of the latitude dimension.
            Defaults to ``"latitude"``.
        lon_dim_name (str): Name of the longitude dimension.
            Defaults to ``"longitude"``.
        vert_dim_name (str, optional): Name of the vertical dimension in ``target_data`` (e.g. ``"level"``).
            If ``None``, the input is treated as purely 2D in space. Defaults to ``None``.
        max_km_extent (float): Half-width of the output Cartesian grid in kilometres. The grid spans
            ``[-max_km_extent, +max_km_extent]`` in both x and y. Defaults to ``2000``.
        grid_km_spacing (float): Spacing between output grid points in kilometres. Defaults to ``25``.
        int_for_center (int): Pixel value in ``extrema_data`` that marks an extremum centre.
            Forwarded to :func:`__project_data_about_extrema`. Defaults to ``2``.
        is_only_extrema_region (bool): If ``True``, pixels in ``target_data`` that fall outside the labelled
            extremum region (i.e. where ``extrema_data == 0``) are masked to ``NaN`` before compositing.
            Defaults to ``False``.

    Returns:
        xr.DataArray: Composited field with dimensions
        ``(n_extrema, [vert_dim_name,] y_km, x_km)`` and coordinates:

        - ``n_extrema`` - integer index for each extremum.
        - ``time`` - source timestamp of each extremum, attached to
          ``n_extrema``.
        - ``y_km``, ``x_km`` - symmetric kilometre-coordinate arrays spanning
          ``[-max_km_extent, +max_km_extent]`` with spacing
          ``grid_km_spacing``.
        - ``vert_dim_name`` - vertical coordinate values, if applicable.

    Raises:
        ValueError: If ``target_data`` or ``extrema_data`` do not contain the required latitude and longitude
            dimensions.
        ValueError: If ``vert_dim_name`` is specified but absent from ``target_data``.
        ValueError: If ``num_extrema_dim`` is absent from ``extrema_data``.
        ValueError: If ``time_dim_name`` is absent from ``target_data`` or from the coordinates of ``extrema_data``.
        ValueError: If ``num_extrema_dim`` is not a dimension of ``extrema_data[time_dim_name]``.
    """
    if not set(target_data.dims).issuperset({lat_dim_name, lon_dim_name}) or not set(extrema_data.dims).issuperset(
        {lat_dim_name, lon_dim_name}
    ):
        raise ValueError("Dimensions of target_data and extrema_data do not contain latitude and longitude dims")

    if vert_dim_name is not None and vert_dim_name not in target_data.dims:
        raise ValueError(f"Expected '{vert_dim_name}' to be present in target_data")

    if num_extrema_dim not in extrema_data.dims:
        raise ValueError(f"Expected '{num_extrema_dim}' to be in {extrema_data.dims}")

    if time_dim_name not in target_data.dims:
        raise ValueError(f"Expected '{time_dim_name}' to be present in target_data")

    if time_dim_name not in extrema_data.coords:
        raise ValueError(f"Expected '{time_dim_name}' to be present in extrema_data coords")

    if num_extrema_dim not in extrema_data[time_dim_name].dims:
        raise ValueError(f"Expected '{num_extrema_dim}' to be in extrema_data[time_dim_name].dims")

    target_with_extrema_dim: xr.DataArray = (
        target_data.sel(indexers={time_dim_name: extrema_data[time_dim_name].values})
        .assign_coords(coords={time_dim_name: extrema_data[num_extrema_dim].values})
        .rename({time_dim_name: num_extrema_dim})
    )
    if is_only_extrema_region:
        target_with_extrema_dim = target_with_extrema_dim.where(extrema_data > 0)
    extrema_times = extrema_data[time_dim_name].values

    # Ensure that dimensions are in the same order for the apply_ufunc() method
    if vert_dim_name is not None:
        target_with_extrema_dim = target_with_extrema_dim.transpose(
            num_extrema_dim, vert_dim_name, lat_dim_name, lon_dim_name
        )
        if vert_dim_name not in extrema_data.dims:
            extrema_data = extrema_data.expand_dims({vert_dim_name: target_data[vert_dim_name]})
        extrema_data = extrema_data.transpose(num_extrema_dim, vert_dim_name, lat_dim_name, lon_dim_name)
    else:
        target_with_extrema_dim = target_with_extrema_dim.transpose(num_extrema_dim, lat_dim_name, lon_dim_name)
        extrema_data = extrema_data.transpose(num_extrema_dim, lat_dim_name, lon_dim_name)

    km_coords = np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing)
    n_km = len(km_coords)

    composited_data: xr.DataArray = xr.apply_ufunc(
        __project_data_about_extrema,
        extrema_data,
        target_with_extrema_dim,
        kwargs={
            "lats": target_data[lat_dim_name].values,
            "lons": target_data[lon_dim_name].values,
            "max_km_extent": max_km_extent,
            "grid_km_spacing": grid_km_spacing,
            "int_for_center": int_for_center,
        },
        input_core_dims=[
            [lat_dim_name, lon_dim_name],  # mask
            [lat_dim_name, lon_dim_name],  # field
        ],
        output_core_dims=[
            ["y_km", "x_km"],
        ],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[target_data.dtype],
        dask_gufunc_kwargs={
            "output_sizes": {
                "y_km": n_km,
                "x_km": n_km,
            }
        },
    )

    coords = {
        "n_extrema": extrema_data["n_extrema"],
        "time": ("n_extrema", extrema_times),
        "y_km": km_coords,
        "x_km": km_coords,
    }
    if vert_dim_name is not None:
        coords[vert_dim_name] = target_data[vert_dim_name]

    return composited_data.assign_coords(coords)
