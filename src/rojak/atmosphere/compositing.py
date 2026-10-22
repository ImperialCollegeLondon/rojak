"""Composite a field onto a common km-scale grid centred on a set of centres.

There are three ways to specify the centres:

- :func:`composite_about_extrema`: The centres are the extrema of a field, as identified by
  :func:`rojak.atmosphere.regions.identify_and_stack_circular_extrema`.
- :func:`composite_about_centre`: The centres are arbitrary points stored in a :class:`xarray.DataArray`, see
  :func:`centres_from_dataframe`. Every vertical level is composited.
- :func:`composite_about_points`: The centres are arbitrary points stored in a :class:`dask.dataframe.DataFrame`, with
  the nearest time step and vertical level selected for each centre. This scales to a large number of centres as the
  centres are grouped by the time chunk of the field they require.

The re-projection of a single field about a single centre is implemented in
:mod:`rojak.atmosphere._compositing_kernels`.
"""

from typing import TYPE_CHECKING

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from rojak.atmosphere._compositing_kernels import (
    project_about_centre,
)

if TYPE_CHECKING:
    from rojak.utilities.types import PandasDataFrame


def _project_data_about_extrema(
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
            ``0 / 1 / 2`` encoding produced by :func:`~rojak.atmosphere.regions._stack_extrema`. The pixel with value
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
        raise ValueError("There should only be one extrema per 2D slice")

    # lat is axis 0, lon is axis 1 -> due to order of input_core_dims
    centre_lat = float(lats[lat_indices[0]])
    centre_lon = float(lons[lon_indices[0]])

    return project_about_centre(data_values, km_coords, centre_lat, centre_lon, lats, lons, lat_lon_buffer, n_km)


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
            :func:`~rojak.atmosphere.regions.identify_and_stack_circular_extrema`. Must contain: 1) Dimensions
            ``num_extrema_dim``, ``lat_dim_name``, ``lon_dim_name``, and 2) a ``time`` coordinate indexed by
            ``num_extrema_dim`` that maps each extremum to its source timestamp in ``target_data``.
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
            Forwarded to :func:`_project_data_about_extrema`. Defaults to ``2``.
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
        _project_data_about_extrema,
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


def centres_from_dataframe(
    centres: "PandasDataFrame | dd.DataFrame",
    *,
    time_col_name: str = "time",
    lat_col_name: str = "latitude",
    lon_col_name: str = "longitude",
    alt_col_name: str = "altitude",
    num_centre_dim: str = "n_centre",
) -> xr.DataArray:
    """Convert a table of composite centres into a :class:`xarray.DataArray` indexed by centre.

    Each row of ``centres`` is treated as a single composite centre. The returned array is 1D along
    ``num_centre_dim``, with its values being the index of each centre (i.e. ``[0, N)``). The time, latitude,
    longitude and altitude of each centre are attached as coordinates along ``num_centre_dim``, analogous to the
    ``time`` coordinate on the ``n_extrema`` dimension produced by
    :func:`~rojak.atmosphere.regions.identify_and_stack_circular_extrema`.

    Args:
        centres (pd.DataFrame | dd.DataFrame): Table with one row per composite centre. Must contain the columns
            ``time_col_name``, ``lat_col_name``, ``lon_col_name`` and ``alt_col_name``. Any other columns are
            ignored. If it is a dask dataframe, the values of the columns are not computed and remain lazy. However,
            the number of rows (both in total and in each partition) is computed so that the size of
            ``num_centre_dim`` is known.
        time_col_name (str): Name of the time column. Defaults to ``"time"``.
        lat_col_name (str): Name of the latitude column. Defaults to ``"latitude"``.
        lon_col_name (str): Name of the longitude column. Defaults to ``"longitude"``.
        alt_col_name (str): Name of the altitude column. Defaults to ``"altitude"``.
        num_centre_dim (str): Name of the dimension indexing each centre. Defaults to ``"n_centre"``.

    Returns:
        xr.DataArray: Array named ``num_centre_dim`` with dimension ``(num_centre_dim,)`` whose values are
        ``[0, N)``, where ``N`` is the number of rows in ``centres``. Its coordinates are:

        - ``num_centre_dim`` - index coordinate with the values ``[0, N)``.
        - ``time_col_name``, ``lat_col_name``, ``lon_col_name`` and ``alt_col_name`` - the values of each column,
          attached to ``num_centre_dim``.

        If ``centres`` is a dask dataframe, the data and the column coordinates are dask arrays. The index coordinate
        is always in memory, as xarray loads index coordinates.

    Raises:
        ValueError: If any of the required columns are missing from ``centres``.

    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "time": pd.to_datetime(["2024-01-01T00:00", "2024-01-01T06:00"]),
    ...         "latitude": [50.0, 55.0],
    ...         "longitude": [-10.0, 0.0],
    ...         "altitude": [10000.0, 11000.0],
    ...     }
    ... )
    >>> centres = centres_from_dataframe(df)
    >>> centres.dims
    ('n_centre',)
    >>> centres.values
    array([0, 1])
    >>> centres["latitude"].values
    array([50., 55.])
    """
    required_columns = [time_col_name, lat_col_name, lon_col_name, alt_col_name]
    missing_columns = set(required_columns).difference(centres.columns)
    if missing_columns:
        raise ValueError(f"Expected columns {sorted(missing_columns)} to be present in centres")

    if is_dask_collection(centres):
        # Keep the columns lazy. lengths=True is required so that the size of the n_centre dim is known
        columns = {col_name: centres[col_name].to_dask_array(lengths=True) for col_name in required_columns}
        num_centres = len(centres)
        centre_count = da.arange(num_centres)
    else:
        columns = {col_name: centres[col_name].to_numpy() for col_name in required_columns}
        num_centres = len(centres)
        centre_count = np.arange(num_centres)

    return xr.DataArray(
        centre_count,
        dims=[num_centre_dim],
        coords={
            num_centre_dim: centre_count,
            **{col_name: (num_centre_dim, column) for col_name, column in columns.items()},
        },
        name=num_centre_dim,
    )


def _project_data_about_centre(
    data_values: np.ndarray,
    centre_lat: float,
    centre_lon: float,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    max_km_extent: float,
    grid_km_spacing: float,
    lat_lon_buffer: float = 0.5,
) -> np.ndarray:
    """Re-project a 2D data field onto a km-scale Cartesian grid centred on a given point.

    Thin wrapper around :func:`~rojak.atmosphere._compositing_kernels.project_about_centre` that builds the km grid
    from ``max_km_extent`` and ``grid_km_spacing`` and has the arguments in the order required by
    :func:`xarray.apply_ufunc`. Intended to be called via :func:`xarray.apply_ufunc` with ``vectorize=True`` so that it
    operates on a single ``(lat, lon)`` slice and a single centre at a time.

    Args:
        data_values (numpy.ndarray): Geophysical field to re-project, of shape ``(lat, lon)``.
        centre_lat (float): Latitude of the centre in degrees. When vectorised by :func:`xarray.apply_ufunc`, this is
            a 0D array, which is converted to a ``float``.
        centre_lon (float): Longitude of the centre in degrees. Can be in either ``[-180, 180)`` or ``[0, 360)``.
            When vectorised by :func:`xarray.apply_ufunc`, this is a 0D array, which is converted to a ``float``.
        lats (numpy.ndarray): 1D array of latitude values in degrees, corresponding to axis 0 of ``data_values``.
        lons (numpy.ndarray): 1D array of longitude values in degrees, corresponding to axis 1 of ``data_values``.
            Can be in either ``[-180, 180)`` or ``[0, 360)``.
        max_km_extent (float): Half-width of the output Cartesian grid in kilometres. The grid spans
            ``[-max_km_extent, +max_km_extent]`` in both x and y.
        grid_km_spacing (float): Spacing between output grid points in kilometres.
        lat_lon_buffer (float): Extra margin in degrees added when subsetting the input lat/lon arrays before
            interpolation. Defaults to ``0.5``.

    Returns:
        numpy.ndarray: Data field interpolated onto the km grid, of shape ``(n_km, n_km)`` with the axes ordered
        ``(y_km, x_km)``, where ``n_km = len(np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing))``. See
        :func:`~rojak.atmosphere._compositing_kernels.project_about_centre` for when the output contains ``NaN``.
    """
    km_coords = np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing)
    return project_about_centre(
        data_values, km_coords, float(centre_lat), float(centre_lon), lats, lons, lat_lon_buffer, len(km_coords)
    )


def composite_about_centre(
    to_composite: xr.DataArray,
    centres: xr.DataArray,
    *,
    time_dim_name: str = "time",
    num_centre_dim: str = "n_centre",
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    vert_dim_name: str | None = None,
    max_km_extent: float = 2000,
    grid_km_spacing: float = 25,
    time_sel_method: str | None = "nearest",
) -> xr.DataArray:
    """Composite a geophysical field onto a common km-scale grid centred on each of the given centres.

    For every centre in ``centres``, the corresponding timestep of ``to_composite`` is extracted and re-projected
    onto a regular Cartesian grid (in km) centred on the latitude and longitude of that centre using a WGS-84
    orthographic projection. The result is a stack of re-projected fields - one per centre - that can subsequently be
    averaged to produce a mean composite structure.

    This is analogous to :func:`composite_about_extrema`, except that the centres are arbitrary points (e.g.
    observations) instead of being derived from the extrema of a field. Unlike :func:`composite_about_extrema`, the
    centre does not need to lie on a grid point of ``to_composite``.

    The longitudes of ``to_composite`` and ``centres`` can each be in either ``[-180, 180)`` or ``[0, 360)``, and
    composites which cross the antimeridian or the 0/360 seam are handled (see
    :func:`~rojak.atmosphere._compositing_kernels.project_about_centre`).

    Optionally supports a vertical dimension, in which case the re-projection is applied independently at each
    level. The altitude of each centre is not used to select a level; every level of ``vert_dim_name`` is
    composited.

    See Also:
        - :func:`centres_from_dataframe`: Constructs ``centres`` from a :class:`pandas.DataFrame` or a
          :class:`dask.dataframe.DataFrame`.
        - :func:`composite_about_extrema`: Composites about extrema instead.

    Args:
        to_composite (xr.DataArray): Geophysical field to composite. Must contain at minimum the dimensions
            ``lat_dim_name``, ``lon_dim_name``, and ``time_dim_name``. If ``vert_dim_name`` is provided, that
            dimension must also be present.
        centres (xr.DataArray): Array of composite centres, as returned by :func:`centres_from_dataframe`. Must
            only have the dimension ``num_centre_dim`` and have the coordinates ``time_dim_name``, ``lat_dim_name``
            and ``lon_dim_name`` along it. If the coordinates are dask arrays, the time coordinate is loaded into
            memory as label-based selection on ``to_composite`` requires concrete values, while the latitude and
            longitude remain lazy.
        time_dim_name (str): Name of the time dimension in ``to_composite`` and the time coordinate in ``centres``.
            Defaults to ``"time"``.
        num_centre_dim (str): Name of the dimension in ``centres`` that indexes individual centres.
            Defaults to ``"n_centre"``.
        lat_dim_name (str): Name of the latitude dimension in ``to_composite`` and the latitude coordinate in
            ``centres``. Defaults to ``"latitude"``.
        lon_dim_name (str): Name of the longitude dimension in ``to_composite`` and the longitude coordinate in
            ``centres``. Defaults to ``"longitude"``.
        vert_dim_name (str, optional): Name of the vertical dimension in ``to_composite`` (e.g. ``"pressure_level"``).
            If ``None``, the input is treated as purely 2D in space. Defaults to ``None``.
        max_km_extent (float): Half-width of the output Cartesian grid in kilometres. The grid spans
            ``[-max_km_extent, +max_km_extent]`` in both x and y. Defaults to ``2000``.
        grid_km_spacing (float): Spacing between output grid points in kilometres. Defaults to ``25``.
        time_sel_method (str, optional): Method passed to :meth:`xarray.DataArray.sel` when matching the time of
            each centre to the time of ``to_composite``. Defaults to ``"nearest"`` as the centres are unlikely to
            coincide exactly with the times in ``to_composite``. If ``None``, an exact match is required.

    Returns:
        xr.DataArray: Composited field of dtype ``float64`` with dimensions
        ``(num_centre_dim, [vert_dim_name,] y_km, x_km)`` and coordinates:

        - ``num_centre_dim`` - index of each centre, as in ``centres``.
        - Every coordinate of ``centres`` along ``num_centre_dim`` apart from ``lat_dim_name`` and ``lon_dim_name``
          (e.g. ``time`` and ``altitude``). The latitude and longitude of the centres are dropped, as they conflict
          with the latitude and longitude dimensions of ``to_composite``. Note that ``time`` is the time of each
          centre, not the (nearest) time step of ``to_composite`` which was composited.
        - ``y_km``, ``x_km`` - symmetric kilometre-coordinate arrays spanning ``[-max_km_extent, +max_km_extent]``
          with spacing ``grid_km_spacing``.
        - ``vert_dim_name`` - vertical coordinate values, if applicable.

        The result is lazy if either ``to_composite`` or the latitude and longitude of ``centres`` are dask arrays.
        Points of the km grid outside of the domain of ``to_composite`` are ``NaN``, so a centre outside of the domain
        produces a composite filled with ``NaN``.

    Raises:
        ValueError: If ``to_composite`` does not contain the required latitude, longitude and time dimensions.
        ValueError: If ``vert_dim_name`` is specified but absent from ``to_composite``.
        ValueError: If ``num_centre_dim`` is absent from ``centres`` or if ``centres`` is not 1D.
        ValueError: If ``centres`` does not have the time, latitude and longitude coordinates along
            ``num_centre_dim``.
        KeyError: If ``time_sel_method`` is ``None`` and the time of a centre is not present in ``to_composite``.
    """
    if not set(to_composite.dims).issuperset({lat_dim_name, lon_dim_name, time_dim_name}):
        raise ValueError("Dimensions of to_composite do not contain the time, latitude and longitude dims")

    if vert_dim_name is not None and vert_dim_name not in to_composite.dims:
        raise ValueError(f"Expected '{vert_dim_name}' to be present in to_composite")

    if centres.dims != (num_centre_dim,):
        raise ValueError(f"Expected centres to only have the dimension '{num_centre_dim}'")

    for coord_name in (time_dim_name, lat_dim_name, lon_dim_name):
        if coord_name not in centres.coords or centres[coord_name].dims != (num_centre_dim,):
            raise ValueError(f"Expected '{coord_name}' to be a coordinate of centres along '{num_centre_dim}'")

    centre_index = centres[num_centre_dim].values
    target_with_centre_dim: xr.DataArray = (
        to_composite.sel(indexers={time_dim_name: centres[time_dim_name].values}, method=time_sel_method)
        .assign_coords(coords={time_dim_name: centre_index})
        .rename({time_dim_name: num_centre_dim})
    )

    # Ensure that dimensions are in the same order for the apply_ufunc() method
    if vert_dim_name is not None:
        target_with_centre_dim = target_with_centre_dim.transpose(
            num_centre_dim, vert_dim_name, lat_dim_name, lon_dim_name
        )
    else:
        target_with_centre_dim = target_with_centre_dim.transpose(num_centre_dim, lat_dim_name, lon_dim_name)

    km_coords = np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing)
    n_km = len(km_coords)

    composited_data: xr.DataArray = xr.apply_ufunc(
        _project_data_about_centre,
        target_with_centre_dim,
        centres[lat_dim_name],
        centres[lon_dim_name],
        kwargs={
            "lats": to_composite[lat_dim_name].values,
            "lons": to_composite[lon_dim_name].values,
            "max_km_extent": max_km_extent,
            "grid_km_spacing": grid_km_spacing,
        },
        input_core_dims=[
            [lat_dim_name, lon_dim_name],  # field
            [],  # centre latitude
            [],  # centre longitude
        ],
        output_core_dims=[
            ["y_km", "x_km"],
        ],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {
                "y_km": n_km,
                "x_km": n_km,
            }
        },
    )

    coords = {
        num_centre_dim: centre_index,
        "y_km": km_coords,
        "x_km": km_coords,
    }
    if vert_dim_name is not None:
        coords[vert_dim_name] = to_composite[vert_dim_name].values

    return composited_data.assign_coords(coords)
