"""Composite a field onto a common km-scale grid centred on a set of centres.

There are two ways to specify the centres:

- :func:`composite_about_extrema`: The centres are the extrema of a field, as identified by
  :func:`rojak.atmosphere.regions.identify_and_stack_circular_extrema`.
- :func:`composite_about_points`: The centres are arbitrary points stored in a :class:`dask.dataframe.DataFrame`, with
  the nearest time step and vertical level selected for each centre. This scales to a large number of centres as the
  centres are grouped by the chunk of the field they require (along either the time or vertical dimension). To
  composite several fields on the same grid about the same centres, use :class:`CompositeAboutPoints` to only
  locate and group the centres once.

The re-projection of a single field about a single centre is implemented in
:mod:`rojak.atmosphere._compositing_kernels`.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, assert_never

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr
from dask.dataframe.dispatch import make_meta
from dask.delayed import delayed

from rojak.atmosphere._compositing_kernels import (
    bilinear_project_about_centre,
    is_periodic_longitude,
    project_about_centre,
)
from rojak.core.indexing import nearest_index

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


class CompositeInterpolation(StrEnum):
    """Interpolation used by :class:`CompositeAboutPoints` to map data onto the km grid.

    Attributes:
        GRIDDATA: Linear interpolation in projected (km) space with :func:`scipy.interpolate.griddata`. This is the
            same scheme as :func:`composite_about_extrema` but requires a Delaunay triangulation per centre.
        BILINEAR: Bilinear interpolation on the native lat/lon grid evaluated at the inverse-projected km grid points.
            Uses the same projection and km grid as ``GRIDDATA`` and only differs in the interpolation stencil. It
            is significantly faster.
    """

    GRIDDATA = "griddata"
    BILINEAR = "bilinear"


class _LayoutColumn(StrEnum):
    """Columns added to the centres by :class:`CompositeAboutPoints` to describe where their data is located.

    The group dimension is the dimension of the target data whose chunks the centres are grouped by. The other
    dimension is the remaining non-horizontal dimension (i.e. vertical if grouped by time, and time if grouped by
    vertical).
    """

    GROUP_INDEX = "_group_index"
    LOCAL_GROUP_INDEX = "_local_group_index"
    OTHER_INDEX = "_other_index"
    CHUNK_INDEX = "_chunk_index"
    TARGET_TIME = "_target_time"
    TARGET_VERT = "_target_vert"
    SOURCE_PARTITION = "_source_partition"
    SOURCE_ROW = "_source_row"


# The functions below are run by dask on each partition or block. They are module-level, rather than methods, so that
# the tasks do not embed the compositor and its dask arrays


def _locate_centres_in_target(
    centres: "PandasDataFrame",
    *,
    time_column: str,
    vert_column: str | None,
    time_coord: np.ndarray,
    vert_coord: np.ndarray | None,
    is_grouped_by_time: bool,
    group_chunk_edges: np.ndarray,
    time_tolerance: np.timedelta64 | None,
    partition_info: dict | None = None,
) -> "PandasDataFrame":
    """Assign each centre the nearest time step (and vertical level) of the target data and its chunk.

    The chunk is the chunk of the group dimension (time if ``is_grouped_by_time``, otherwise vertical) that contains
    the time step (or vertical level) of the centre. Rows without a valid time, or further than ``time_tolerance``
    from the nearest time step, are dropped. Each row is also tagged with its original partition and position so that
    it has a unique, deterministic sort key. Intended to be called on each partition via
    :meth:`dask.dataframe.DataFrame.map_partitions`.
    """
    # Time column has been normalised to naive UTC in _TargetGrid.locate_centres()
    centre_times: np.ndarray = centres[time_column].to_numpy()
    time_index = nearest_index(time_coord, centre_times)

    if vert_column is not None and vert_coord is not None:
        vert_index = nearest_index(vert_coord, centres[vert_column].to_numpy(dtype=float))
        target_vert = vert_coord[vert_index]
    else:
        vert_index = np.zeros(time_index.shape, dtype=np.intp)
        target_vert = np.full(time_index.shape, np.nan)

    group_index, other_index = (time_index, vert_index) if is_grouped_by_time else (vert_index, time_index)
    chunk_index = np.searchsorted(group_chunk_edges, group_index, side="right") - 1

    is_valid = ~np.isnat(centre_times)
    if time_tolerance is not None:
        is_valid &= np.abs(time_coord[time_index] - centre_times) <= time_tolerance

    return centres.assign(
        **{
            _LayoutColumn.GROUP_INDEX: group_index.astype(np.intp),
            _LayoutColumn.LOCAL_GROUP_INDEX: (group_index - group_chunk_edges[chunk_index]).astype(np.intp),
            _LayoutColumn.OTHER_INDEX: other_index.astype(np.intp),
            _LayoutColumn.CHUNK_INDEX: chunk_index.astype(np.intp),
            _LayoutColumn.TARGET_TIME: time_coord[time_index],
            _LayoutColumn.TARGET_VERT: target_vert,
            _LayoutColumn.SOURCE_PARTITION: np.intp(partition_info["number"] if partition_info is not None else 0),
            _LayoutColumn.SOURCE_ROW: np.arange(len(centres), dtype=np.intp),
        }
    )[is_valid]


def _sort_partition_deterministically(partition: "PandasDataFrame") -> "PandasDataFrame":
    """Sort a partition by its index then by the unique source tags.

    Rows sharing the same group index can otherwise come out of a shuffle in a different order each time it is
    evaluated. As dask pushes column projections below the shuffle, each column extracted from the shuffled dataframe
    can come from a separate shuffle. Without a total order, the rows of the composite and its coordinates may not
    align.
    """
    return partition.sort_values([_LayoutColumn.SOURCE_PARTITION, _LayoutColumn.SOURCE_ROW], kind="stable").sort_index(
        kind="stable"
    )


def _split_into_tasks(counts: np.ndarray, max_per_task: int) -> tuple[list[int], list[int]]:
    """Split the number of centres per chunk into blocks of at most ``max_per_task`` centres.

    Returns:
        The size of each block and the chunk each block belongs to. Empty chunks are dropped.

    >>> _split_into_tasks(np.array([5, 0, 2]), 2)
    ([2, 2, 1, 2], [0, 0, 0, 2])
    """
    block_sizes: list[int] = []
    block_chunk: list[int] = []
    for chunk, count in enumerate[int](counts):
        num_full, remainder = divmod(int(count), max_per_task)
        sizes = [max_per_task] * num_full + ([remainder] if remainder else [])
        block_sizes.extend(sizes)
        block_chunk.extend([chunk] * len(sizes))
    return block_sizes, block_chunk


def _composite_block_about_points(
    centres: np.ndarray,
    field_block: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    km_coords: np.ndarray,
    interpolation: CompositeInterpolation,
    lat_lon_buffer: float,
    out_dtype: np.dtype,
) -> np.ndarray:
    """Composite one block of centres against the chunk of the field that contains all of their data.

    Intended to be called via :func:`dask.array.blockwise` within :meth:`CompositeAboutPoints._composite_blocks`.

    Args:
        centres: Array of shape ``(n_centres, 4)`` whose columns are the centre latitude, centre longitude, index
            along the group dimension within ``field_block`` and index along the other dimension.
        field_block: Array of shape ``(group, other, lat, lon)``, i.e. one chunk of the group dimension (time or
            vertical) and all of the other non-horizontal dimension (vertical or time).
        lats: 1D latitude coordinate of ``field_block``.
        lons: 1D longitude coordinate of ``field_block``.
        km_coords: 1D km coordinates of the output grid along x and y.
        interpolation: Interpolation scheme.
        lat_lon_buffer: Only used by :attr:`CompositeInterpolation.GRIDDATA`.
        out_dtype: dtype of the output.

    Returns:
        numpy.ndarray: Array of shape ``(n_centres, n_km, n_km)``.
    """
    n_km = km_coords.size
    is_lon_periodic = is_periodic_longitude(lons)
    group_indices = centres[:, 2].astype(np.intp)
    other_indices = centres[:, 3].astype(np.intp)

    result = np.empty((centres.shape[0], n_km, n_km), dtype=out_dtype)
    for i, (centre_lat, centre_lon, group_index, other_index) in enumerate[tuple[int, int, int, int]](
        zip(centres[:, 0], centres[:, 1], group_indices, other_indices, strict=True)
    ):
        data_values: np.ndarray = field_block[group_index, other_index]
        match interpolation:
            case CompositeInterpolation.BILINEAR:
                result[i] = bilinear_project_about_centre(
                    data_values,
                    km_coords,
                    float(centre_lat),
                    float(centre_lon),
                    lats,
                    lons,
                    is_lon_periodic=is_lon_periodic,
                )
            case CompositeInterpolation.GRIDDATA:
                result[i] = project_about_centre(
                    data_values, km_coords, float(centre_lat), float(centre_lon), lats, lons, lat_lon_buffer, n_km
                )
            case _ as unreachable:
                assert_never(unreachable)
    return result


@dataclass(frozen=True, eq=False)
class _TargetGrid:
    """Dimensions, coordinates and chunks of the fields composited by :class:`CompositeAboutPoints`.

    Only describes the grid, not the values on it, so the same grid is shared by every field composited about the same
    centres.
    """

    time_dim_name: str
    lat_dim_name: str
    lon_dim_name: str
    vert_dim_name: str | None
    group_dim_name: str
    time_coord: np.ndarray
    vert_coord: np.ndarray | None
    lats: np.ndarray
    lons: np.ndarray
    group_chunks: tuple[int, ...]

    @classmethod
    def from_template(
        cls,
        template: xr.DataArray | xr.Dataset,
        *,
        time_dim_name: str,
        lat_dim_name: str,
        lon_dim_name: str,
        vert_dim_name: str | None,
        group_dim_name: str | None,
    ) -> "_TargetGrid":
        """Read the grid of ``template``, see :func:`composite_about_points` for the errors raised."""
        spatial_dims = {lat_dim_name, lon_dim_name}
        if group_dim_name is None:
            group_dim_name = time_dim_name
        if group_dim_name not in template.dims:
            raise ValueError(f"Expected group_dim_name '{group_dim_name}' to be a dimension of target_data")
        if group_dim_name in spatial_dims:
            raise ValueError(f"Cannot group by the horizontal dimension '{group_dim_name}'")
        if group_dim_name not in {time_dim_name, vert_dim_name}:
            raise ValueError(
                f"Expected group_dim_name '{group_dim_name}' to be either time_dim_name '{time_dim_name}' or "
                f"vert_dim_name '{vert_dim_name}'"
            )

        expected_dims: set[str | None] = {time_dim_name, *spatial_dims, vert_dim_name} - {None}
        if set(template.dims) != expected_dims:
            raise ValueError(f"Expected target_data to have dimensions {expected_dims}, got {template.dims}")

        return cls(
            time_dim_name=time_dim_name,
            lat_dim_name=lat_dim_name,
            lon_dim_name=lon_dim_name,
            vert_dim_name=vert_dim_name,
            group_dim_name=group_dim_name,
            time_coord=template[time_dim_name].to_numpy(),
            vert_coord=template[vert_dim_name].to_numpy() if vert_dim_name is not None else None,
            lats=template[lat_dim_name].to_numpy(),
            lons=template[lon_dim_name].to_numpy(),
            group_chunks=template.chunksizes.get(group_dim_name, (template.sizes[group_dim_name],)),
        )

    @property
    def is_grouped_by_time(self) -> bool:
        return self.group_dim_name == self.time_dim_name

    @property
    def other_dim_name(self) -> str:
        """Remaining non-horizontal dimension, which is a placeholder if there is no vertical dimension."""
        return self.time_dim_name if not self.is_grouped_by_time else (self.vert_dim_name or "_vertical")

    @property
    def group_chunk_edges(self) -> np.ndarray:
        return np.cumsum((0, *self.group_chunks))

    def check_matches(self, field: xr.DataArray) -> None:
        """Raise a ``ValueError`` if ``field`` is not on this grid."""
        expected_dims: set[str | None] = {
            self.time_dim_name,
            self.lat_dim_name,
            self.lon_dim_name,
            self.vert_dim_name,
        } - {None}
        if set(field.dims) != expected_dims:
            raise ValueError(f"Expected target_data to have dimensions {expected_dims}, got {field.dims}")

        # The centres were located using these coordinates, so they must be the same for the indices to be valid
        for dim_name, coord in (
            (self.time_dim_name, self.time_coord),
            (self.vert_dim_name, self.vert_coord),
            (self.lat_dim_name, self.lats),
            (self.lon_dim_name, self.lons),
        ):
            if dim_name is not None and coord is not None and not np.array_equal(field[dim_name].to_numpy(), coord):
                raise ValueError(f"Coordinate '{dim_name}' of target_data differs from the one it was set up with")

    def to_group_major(self, field: xr.DataArray) -> da.Array:
        """Lay out ``field`` as ``(group, other, lat, lon)``, chunked only along the group dimension.

        The group dimension is chunked by ``group_chunks``, which is a no-op if ``field`` is already chunked that way.
        The other dimension is added as a placeholder of size 1 if it is not in ``field``.
        """
        spatial_dims = [self.lat_dim_name, self.lon_dim_name]
        if self.other_dim_name not in field.dims:
            field = field.expand_dims(self.other_dim_name)
        field = field.transpose(self.group_dim_name, self.other_dim_name, *spatial_dims)
        return field.chunk(
            {self.group_dim_name: self.group_chunks, **dict.fromkeys([self.other_dim_name, *spatial_dims], -1)}
        ).data

    def locate_centres(
        self,
        centres: dd.DataFrame,
        *,
        time_column: str,
        vert_column: str | None,
        time_tolerance: np.timedelta64 | None,
    ) -> dd.DataFrame:
        """Annotate each centre with the location of its data on this grid, see :func:`_locate_centres_in_target`."""
        # Normalise to naive UTC so that it can be compared with the (naive) time coordinate of the xr.DataArray. Naive
        # times are assumed to already be in UTC
        centres = centres.assign(**{time_column: dd.to_datetime(centres[time_column], utc=True).dt.tz_convert(None)})
        locate_centres_partial = functools.partial(
            _locate_centres_in_target,
            time_column=time_column,
            vert_column=vert_column,
            vert_coord=self.vert_coord,
            is_grouped_by_time=self.is_grouped_by_time,
            time_tolerance=time_tolerance,
        )
        return centres.map_partitions(
            locate_centres_partial,
            time_coord=delayed(self.time_coord, pure=True),
            group_chunk_edges=delayed(self.group_chunk_edges, pure=True),
            meta=locate_centres_partial(
                make_meta(centres, parent_meta=centres),
                time_coord=self.time_coord,
                group_chunk_edges=self.group_chunk_edges,
            ),
        )

    def group_by_chunk(self, located: dd.DataFrame) -> tuple[dd.DataFrame, np.ndarray]:
        """Shuffle the located centres such that partition ``i`` contains exactly the centres whose data is in chunk
        ``i``.

        Returns:
            The shuffled centres and the number of centres in each chunk.

        Raises:
            ValueError: If there are no centres.
        """
        group_chunk_edges = self.group_chunk_edges
        num_group_chunks = len(self.group_chunks)
        # Number of centres in each chunk. Required to know the shape of the output. Counted before the shuffle below
        # so that the shuffle is only evaluated once (by the composite)
        counts_per_chunk: np.ndarray = da.bincount(
            located[_LayoutColumn.CHUNK_INDEX].to_dask_array(), minlength=num_group_chunks
        ).compute()
        if counts_per_chunk.sum() == 0:
            raise ValueError("There are no centres within the time range of target_data")

        by_chunk: dd.DataFrame = located.set_index(
            _LayoutColumn.GROUP_INDEX,
            divisions=[*group_chunk_edges[:-1].tolist(), int(group_chunk_edges[-1]) - 1],
            sorted=False,
        ).map_partitions(_sort_partition_deterministically)
        assert by_chunk.npartitions == num_group_chunks
        return by_chunk, counts_per_chunk


@dataclass(frozen=True, eq=False)
class CompositeAboutPoints:
    """Composite fields on a common grid about the centres stored in a :class:`dask.dataframe.DataFrame`.

    Build it once with :meth:`from_centres`, which locates, counts and shuffles the centres, then call it on each field
    to composite. Only the interpolation is built for each field. As every composite shares the same (lazy) centre
    arrays, computing the composites together evaluates the shuffle once. See :func:`composite_about_points` for how
    the compositing works.

    It is callable, so it can be applied to every data variable of a :class:`xarray.Dataset` with
    :meth:`xarray.Dataset.map`.

    Example:
        >>> composite = CompositeAboutPoints.from_centres(
        ...     ds, centres, vert_dim_name="pressure_level", vert_column="level"
        ... )  # doctest: +SKIP
        >>> composites = ds[["temperature", "u_wind"]].map(composite)  # doctest: +SKIP
        >>> geopotential_composite = composite(ds["geopotential"])  # doctest: +SKIP
    """

    grid: _TargetGrid
    centre_dim_name: str
    block_sizes: list[int]
    block_chunk: list[int]
    centre_array: da.Array
    centre_coords: dict[str, Any]
    km_coords: np.ndarray
    interpolation: CompositeInterpolation
    lat_lon_buffer: float

    @classmethod
    def from_centres(
        cls,
        template: xr.DataArray | xr.Dataset,
        centres: dd.DataFrame,
        *,
        time_dim_name: str = "time",
        lat_dim_name: str = "latitude",
        lon_dim_name: str = "longitude",
        vert_dim_name: str | None = None,
        group_dim_name: str | None = None,
        time_column: str = "time",
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        vert_column: str | None = None,
        carry_columns: Sequence[str] = (),
        time_tolerance: np.timedelta64 | None = None,
        max_km_extent: float = 1000,
        grid_km_spacing: float = 25,
        interpolation: CompositeInterpolation = CompositeInterpolation.BILINEAR,
        max_centres_per_task: int = 500,
        centre_dim_name: str = "n_centre",
        lat_lon_buffer: float = 0.5,
    ) -> "CompositeAboutPoints":
        """Locate and group the centres on the grid of ``template``.

        Takes the same arguments as :func:`composite_about_points`, with ``template`` in place of ``target_data``.
        Only the dimensions, coordinates and chunks along ``group_dim_name`` of ``template`` are used, not its values.
        The number of centres per chunk is computed.

        Args:
            template (xr.DataArray | xr.Dataset): Has the dimensions, coordinates and chunks of the fields to
                composite. If it is a :class:`xarray.Dataset`, it must be chunked consistently along
                ``group_dim_name``.

        Raises:
            ValueError: See :func:`composite_about_points`.
        """
        grid = _TargetGrid.from_template(
            template,
            time_dim_name=time_dim_name,
            lat_dim_name=lat_dim_name,
            lon_dim_name=lon_dim_name,
            vert_dim_name=vert_dim_name,
            group_dim_name=group_dim_name,
        )

        if (vert_dim_name is None) != (vert_column is None):
            raise ValueError("vert_dim_name and vert_column must either both be specified or both be None")
        required_columns = {time_column, lat_column, lon_column, *carry_columns, vert_column} - {None}
        if missing_columns := required_columns.difference(centres.columns):
            raise ValueError(f"Columns {missing_columns} are missing from centres")
        if max_centres_per_task <= 0:
            raise ValueError("max_centres_per_task must be positive")

        by_chunk, counts_per_chunk = grid.group_by_chunk(
            grid.locate_centres(
                centres, time_column=time_column, vert_column=vert_column, time_tolerance=time_tolerance
            )
        )

        # Further split each partition so that each output block is bounded in size. As the new blocks are a refinement
        # of the partitions, the rechunk does not move centres across chunks
        block_sizes, block_chunk = _split_into_tasks(counts_per_chunk, max_centres_per_task)
        column_as_array = functools.partial(
            cls._column_as_array, by_chunk, partition_lengths=counts_per_chunk.tolist(), block_sizes=block_sizes
        )

        centre_coords: dict[str, Any] = {
            time_dim_name: (centre_dim_name, column_as_array(_LayoutColumn.TARGET_TIME)),
            "centre_latitude": (centre_dim_name, column_as_array(lat_column)),
            "centre_longitude": (centre_dim_name, column_as_array(lon_column)),
        }
        if vert_dim_name is not None:
            centre_coords[vert_dim_name] = (centre_dim_name, column_as_array(_LayoutColumn.TARGET_VERT))
        for column in carry_columns:
            centre_coords[column] = (centre_dim_name, column_as_array(column))

        return cls(
            grid=grid,
            centre_dim_name=centre_dim_name,
            block_sizes=block_sizes,
            block_chunk=block_chunk,
            centre_array=column_as_array(
                [lat_column, lon_column, _LayoutColumn.LOCAL_GROUP_INDEX, _LayoutColumn.OTHER_INDEX]
            ).astype(np.float64),
            centre_coords=centre_coords,
            km_coords=np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing),
            interpolation=CompositeInterpolation(interpolation),
            lat_lon_buffer=lat_lon_buffer,
        )

    def __call__(self, field: xr.DataArray) -> xr.DataArray:
        """Composite ``field`` about the centres.

        Args:
            field (xr.DataArray): Field to composite. Must have the same dimensions and the same time, vertical,
                latitude and longitude coordinates as the template. It is rechunked along the group dimension to match
                the template, so chunk it the same way to avoid a rechunk.

        Returns:
            xr.DataArray: See :func:`composite_about_points`.

        Raises:
            ValueError: If the dimensions or coordinates of ``field`` do not match the template.
        """
        self.grid.check_matches(field)
        out_dtype = field.dtype if np.issubdtype(field.dtype, np.floating) else np.dtype(np.float64)
        return xr.DataArray(
            self._composite_blocks(self.grid.to_group_major(field), out_dtype),
            dims=(self.centre_dim_name, "y_km", "x_km"),
            coords={"y_km": self.km_coords, "x_km": self.km_coords, **self.centre_coords},
            name=field.name if field.name is not None else "composite",
            attrs=field.attrs,
        )

    def _composite_blocks(self, field_data: da.Array, out_dtype: np.dtype) -> da.Array:
        """Composite each block of centres against the chunk of ``field_data`` given by ``block_chunk``.

        Returns:
            dask.array.Array: Array of shape ``(n_centres, n_km, n_km)`` chunked by ``block_sizes``.
        """
        n_km = self.km_coords.size
        # Chunk that matches each block of centres. The blocks are references to the existing chunks
        field_per_block: da.Array = da.concatenate(
            [field_data.blocks[chunk] for chunk in self.block_chunk],  # pyright: ignore[reportAttributeAccessIssue]
            axis=0,
        )

        # Block i of centre_array is paired with block i of field_per_block (align_arrays=False). The lat/lon coords
        # are dask arrays so that they are stored in the graph once instead of being embedded in every task
        return da.blockwise(
            _composite_block_about_points,
            "cyx",
            self.centre_array,
            "ck",
            field_per_block,
            "czab",
            da.from_array(self.grid.lats, chunks=-1),  # pyright: ignore[reportArgumentType]
            "a",
            da.from_array(self.grid.lons, chunks=-1),  # pyright: ignore[reportArgumentType]
            "b",
            new_axes={"y": n_km, "x": n_km},
            adjust_chunks={"c": tuple(self.block_sizes)},
            align_arrays=False,
            concatenate=True,
            dtype=out_dtype,
            meta=np.empty((0, 0, 0), dtype=out_dtype),
            km_coords=self.km_coords,
            interpolation=self.interpolation,
            lat_lon_buffer=self.lat_lon_buffer,
            out_dtype=out_dtype,
        )

    @staticmethod
    def _column_as_array(
        by_chunk: dd.DataFrame, columns: str | list[str], *, partition_lengths: list[int], block_sizes: list[int]
    ) -> da.Array:
        """Extract column(s) of ``by_chunk`` as a dask array, rechunked to match ``block_sizes``."""
        array: da.Array = by_chunk[columns].to_dask_array(lengths=partition_lengths)
        return array.rechunk({0: tuple(block_sizes), 1: -1} if array.ndim == 2 else {0: tuple(block_sizes)})  # pyright: ignore[reportArgumentType] # noqa: PLR2004


def composite_about_points(
    target_data: xr.DataArray,
    centres: dd.DataFrame,
    *,
    time_dim_name: str = "time",
    lat_dim_name: str = "latitude",
    lon_dim_name: str = "longitude",
    vert_dim_name: str | None = None,
    group_dim_name: str | None = None,
    time_column: str = "time",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    vert_column: str | None = None,
    carry_columns: Sequence[str] = (),
    time_tolerance: np.timedelta64 | None = None,
    max_km_extent: float = 2000,
    grid_km_spacing: float = 25,
    interpolation: CompositeInterpolation = CompositeInterpolation.BILINEAR,
    max_centres_per_task: int = 500,
    centre_dim_name: str = "n_centre",
    lat_lon_buffer: float = 0.5,
) -> xr.DataArray:
    """Composite a geophysical field onto a common km-scale grid centred on each row of a dataframe.

    This is the counterpart of :func:`composite_about_extrema` for when the centres are arbitrary points (e.g.
    observations) stored in a :class:`dask.dataframe.DataFrame` rather than extrema on the grid of ``target_data``.
    For each centre, the nearest time step (and, if ``vert_dim_name`` is given, nearest vertical level) of
    ``target_data`` is re-projected onto a regular Cartesian grid (in km) centred on that point using a WGS-84
    orthographic projection.

    Rather than expanding ``target_data`` to have one slice per centre, the centres are grouped by the chunks of
    ``target_data`` along ``group_dim_name`` (time by default, or the vertical dimension). The centres are shuffled
    so that partition ``i`` of the dataframe contains exactly the centres whose nearest time step (or vertical level)
    lies in chunk ``i`` of ``target_data``. Each output block is computed by a single task that depends only on its
    block of centres and the one matching chunk, which keeps the graph small: every task refers to existing chunks by
    key and no data is duplicated or embedded in the graph.

    Note:
        The only computation triggered is the number of centres per chunk, an integer per chunk, which is needed for
        the shape of the output. Persist ``centres`` beforehand if it is expensive to compute.

        The centres in the output are sorted by their index along ``group_dim_name``, not in the order of
        ``centres``. Use ``carry_columns`` to keep an identifier.

    Args:
        target_data (xr.DataArray): Geophysical field to composite with dimensions ``time_dim_name``,
            ``lat_dim_name``, ``lon_dim_name`` and optionally ``vert_dim_name``. It should be chunked along
            ``group_dim_name`` only; other dimensions are rechunked to a single chunk.
        centres (dask.dataframe.DataFrame): Centres to composite about. Must contain ``time_column``,
            ``lat_column``, ``lon_column`` (and ``vert_column`` if ``vert_dim_name`` is given) as columns.
        time_dim_name (str): Name of the time dimension in ``target_data``. Defaults to ``"time"``.
        lat_dim_name (str): Name of the latitude dimension in ``target_data``. Defaults to ``"latitude"``.
        lon_dim_name (str): Name of the longitude dimension in ``target_data``. Defaults to ``"longitude"``.
        vert_dim_name (str, optional): Name of the vertical dimension in ``target_data``. The level nearest to the
            centre's ``vert_column`` value is used. Defaults to ``None``.
        group_dim_name (str, optional): Dimension of ``target_data`` whose chunks the centres are grouped by. Must be
            either ``time_dim_name`` or ``vert_dim_name``, as each centre selects a single slice along these. Each
            task loads one chunk along this dimension and all of the other dimensions, so it should be the dimension
            ``target_data`` is chunked along. If ``None``, ``time_dim_name`` is used. Defaults to ``None``.
        time_column (str): Column with the time of each centre. Naive times are assumed to be UTC.
            Defaults to ``"time"``.
        lat_column (str): Column with the latitude of each centre in degrees. Defaults to ``"latitude"``.
        lon_column (str): Column with the longitude of each centre in degrees. Defaults to ``"longitude"``.
        vert_column (str, optional): Column with the vertical position of each centre in the same units as
            ``target_data[vert_dim_name]``. Required if ``vert_dim_name`` is given. Defaults to ``None``.
        carry_columns (Sequence[str]): Additional columns of ``centres`` to attach as coordinates along
            ``centre_dim_name``, e.g. a cluster label. Defaults to ``()``.
        time_tolerance (numpy.timedelta64, optional): Centres further than this from the nearest time step are
            dropped, e.g. ``np.timedelta64(30, "m")``. If ``None``, all centres are kept. Defaults to ``None``.
        max_km_extent (float): Half-width of the output Cartesian grid in kilometres. Defaults to ``2000``.
        grid_km_spacing (float): Spacing between output grid points in kilometres. Defaults to ``25``.
        interpolation (CompositeInterpolation): Interpolation scheme. Use
            :attr:`CompositeInterpolation.GRIDDATA` for the same results as :func:`composite_about_extrema`.
            Defaults to :attr:`CompositeInterpolation.BILINEAR`.
        max_centres_per_task (int): Maximum number of centres in a single output block. Bounds the memory of each
            task to roughly ``max_centres_per_task * n_km**2 * 8`` bytes. Defaults to ``500``.
        centre_dim_name (str): Name of the output dimension indexing the centres. Defaults to ``"n_centre"``.
        lat_lon_buffer (float): Only used by :attr:`CompositeInterpolation.GRIDDATA`. See
            :func:`_project_data_about_extrema`. Defaults to ``0.5``.

    Returns:
        xr.DataArray: Dask-backed composited field with dimensions ``(centre_dim_name, y_km, x_km)`` and
        coordinates:

        - ``y_km``, ``x_km`` - symmetric kilometre-coordinate arrays.
        - ``time_dim_name`` - time step of ``target_data`` used for each centre.
        - ``vert_dim_name`` - vertical level of ``target_data`` used for each centre, if applicable.
        - ``centre_latitude``, ``centre_longitude`` - location of each centre.
        - Each of ``carry_columns``.

    Raises:
        ValueError: If ``group_dim_name`` is not a dimension of ``target_data``, is a horizontal dimension, or is
            neither ``time_dim_name`` nor ``vert_dim_name``.
        ValueError: If ``target_data`` does not have exactly the specified dimensions.
        ValueError: If only one of ``vert_dim_name`` and ``vert_column`` is specified.
        ValueError: If a required column is missing from ``centres``.
        ValueError: If there are no centres to composite about.
    """
    return CompositeAboutPoints.from_centres(
        target_data,
        centres,
        time_dim_name=time_dim_name,
        lat_dim_name=lat_dim_name,
        lon_dim_name=lon_dim_name,
        vert_dim_name=vert_dim_name,
        group_dim_name=group_dim_name,
        time_column=time_column,
        lat_column=lat_column,
        lon_column=lon_column,
        vert_column=vert_column,
        carry_columns=carry_columns,
        time_tolerance=time_tolerance,
        max_km_extent=max_km_extent,
        grid_km_spacing=grid_km_spacing,
        interpolation=interpolation,
        max_centres_per_task=max_centres_per_task,
        centre_dim_name=centre_dim_name,
        lat_lon_buffer=lat_lon_buffer,
    )(target_data)
