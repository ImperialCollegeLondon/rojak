import numpy as np
import scipy.ndimage as ndi
import xarray as xr


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
