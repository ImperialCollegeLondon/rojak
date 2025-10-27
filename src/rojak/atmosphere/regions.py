import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from numba import guvectorize, njit


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
        target_array, structure=ndi.generate_binary_structure(rank=num_dim, connectivity=connectivity)
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


def label_regions(
    array: xr.DataArray, num_dims: int = 3, core_dims: list[str] | None = None, connectivity: int | None = None
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
