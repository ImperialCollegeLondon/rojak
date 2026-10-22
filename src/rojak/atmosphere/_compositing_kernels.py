"""Numpy kernels for re-projecting a field onto a km-scale grid centred on a point.

The functions in this module operate on in-memory numpy arrays for a single centre (i.e. a single ``(lat, lon)`` slice).
They contain no dask, xarray or pandas logic, so that they can be called from within dask tasks and
:func:`xarray.apply_ufunc`. The orchestration of these kernels over many centres lives in
:mod:`rojak.atmosphere.compositing`.
"""

import numpy as np
import pyproj
import scipy.interpolate as si


def wrap_longitude_offset(lons: np.ndarray, relative_to: float) -> np.ndarray:
    """Longitude offsets relative to a reference longitude, wrapped to the range ``[-180, 180)``

    Args:
        lons (numpy.ndarray): Longitudes in degrees. Can be in either ``[-180, 180)`` or ``[0, 360)``.
        relative_to (float): Reference longitude in degrees, e.g. the longitude of a centre. Can be in either
            ``[-180, 180)`` or ``[0, 360)``.

    Returns:
        numpy.ndarray: ``lons - relative_to`` wrapped to ``[-180, 180)``, i.e. the signed angular difference in degrees
        of each longitude from ``relative_to``. Positive values are east of ``relative_to``.

    Examples
    --------

    >>> wrap_longitude_offset(np.array([-190.0, -180.0, 0.0, 170.0, 180.0, 350.0]), 0.0)
    array([ 170., -180.,    0.,  170., -180.,  -10.])

    Longitudes either side of the antimeridian are close to a reference longitude of 175 degrees

    >>> wrap_longitude_offset(np.array([170.0, 179.5, -180.0, -175.0]), 175.0)
    array([-5. ,  4.5,  5. , 10. ])

    Longitudes either side of the 0/360 seam are close to a reference longitude of 2 degrees

    >>> wrap_longitude_offset(np.array([355.0, 0.0, 5.0]), 2.0)
    array([-7., -2.,  3.])
    """
    return (lons - relative_to + 180) % 360 - 180


def project_about_centre(
    data_values: np.ndarray,
    km_coords: np.ndarray,
    centre_lat: float,
    centre_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_lon_buffer: float,
    n_km: int,
) -> np.ndarray:
    """Re-project a 2D data field onto a km-scale Cartesian grid centred on a given latitude and longitude.

    Constructs a WGS-84 orthographic projection (:class:`pyproj.Proj`) anchored at the centre and interpolates
    ``data_values`` from its native lat/lon grid onto a regular ``(y_km, x_km)`` Cartesian grid using linear
    interpolation via :func:`scipy.interpolate.griddata`. This is the shared implementation used by
    :func:`_project_data_about_extrema` and :func:`_project_data_about_centre`.

    The re-projection pipeline is:

    1. Use the inverse orthographic projection to find which latitudes and longitudes fall within the output grid
       extent (plus ``lat_lon_buffer``). Longitudes are compared as offsets from ``centre_lon`` wrapped to
       ``[-180, 180)`` (see :func:`wrap_longitude_offset`), so that grids crossing the antimeridian or the 0/360 seam
       only select the longitudes near the centre.
    2. Use the forward projection to convert those native lat/lon points to km coordinates.
    3. Discard any points that are not visible from the centre (i.e. on the far side of the globe), as the
       orthographic projection maps them to ``inf`` and :func:`scipy.interpolate.griddata` cannot triangulate them.
    4. Interpolate the remaining native data onto the regular km grid.

    Args:
        data_values (numpy.ndarray): Geophysical field to re-project, of shape ``(lat, lon)``.
        km_coords (numpy.ndarray): 1D array of the km coordinates of the output grid, used for both x and y.
        centre_lat (float): Latitude of the centre in degrees.
        centre_lon (float): Longitude of the centre in degrees. Can be in either ``[-180, 180)`` or ``[0, 360)``,
            independent of the convention used by ``lons``.
        lats (numpy.ndarray): 1D array of latitude values in degrees, corresponding to axis 0 of ``data_values``.
        lons (numpy.ndarray): 1D array of longitude values in degrees, corresponding to axis 1 of ``data_values``.
            Can be in either ``[-180, 180)`` or ``[0, 360)``.
        lat_lon_buffer (float): Extra margin in degrees added when subsetting the input lat/lon arrays before
            interpolation, to avoid edge artefacts near the boundary of the output grid.
        n_km (int): Number of points in ``km_coords``.

    Returns:
        numpy.ndarray: Data field interpolated onto the km grid, of shape ``(n_km, n_km)`` with the axes ordered
        ``(y_km, x_km)``. Points of the km grid that lie outside of the input data (e.g. beyond the edge of a regional
        domain) are ``NaN``. If fewer than 3 visible input points fall within the extent of the km grid (e.g. the
        centre is outside of the domain of the input data), an array filled with ``NaN`` is returned.
    """
    # km grid that we want to map the data onto
    x_km_mesh, y_km_mesh = np.meshgrid(km_coords, km_coords)

    # Inverse projection to find what lat/lon indices are within range
    ortho = pyproj.Proj(proj="ortho", lat_0=centre_lat, lon_0=centre_lon, units="km", ellps="WGS84")
    new_lons, new_lats = ortho(x_km_mesh, y_km_mesh, inverse=True)

    lat_mask = (lats >= new_lats.min() - lat_lon_buffer) & (lats <= new_lats.max() + lat_lon_buffer)
    # Longitudes are compared as offsets from the centre, wrapped to [-180, 180). Otherwise, when the grid crosses the
    # antimeridian, the min and max of new_lons are ~-180 and ~180, which selects every longitude. This also makes it
    # independent of whether the longitudes are in [-180, 180) or [0, 360)
    lon_offsets = wrap_longitude_offset(lons, centre_lon)
    new_lon_offsets = wrap_longitude_offset(new_lons, centre_lon)
    lon_mask = (lon_offsets >= new_lon_offsets.min() - lat_lon_buffer) & (
        lon_offsets <= new_lon_offsets.max() + lat_lon_buffer
    )
    lats_within = lats[lat_mask]
    lons_within = lons[lon_mask]

    data_within = data_values[np.ix_(lat_mask, lon_mask)]

    # Forward projection identify what these data points are in km so that scipy.griddata() can use to to interpolate
    # to our desired grid system
    lon_grid, lat_grid = np.meshgrid(lons_within, lats_within)
    x_km_within, y_km_within = ortho(lon_grid, lat_grid)

    # Points on the far side of the globe are not visible in an orthographic projection and are projected to inf.
    # These must be removed as scipy.griddata() (qhull) fails with a QhullError if any of the points are not finite
    is_finite = np.isfinite(x_km_within) & np.isfinite(y_km_within)
    min_points_for_triangulation: int = 3
    if np.count_nonzero(is_finite) < min_points_for_triangulation:
        return np.full((n_km, n_km), np.nan)

    return si.griddata(
        points=(x_km_within[is_finite], y_km_within[is_finite]),
        values=data_within[is_finite],
        xi=(x_km_mesh.ravel(), y_km_mesh.ravel()),
        method="linear",
    ).reshape(n_km, n_km)
