"""Point features found at the extrema of a field: lows and highs, one centre per extremum.

:class:`ExtremaPointFeature` is the graph side: it holds the field and builds the lazy inputs with
:func:`~rojak.atmosphere.regions.identify_circular_extrema`. :class:`ExtremaPointTabulator` is the task side: it finds
the centres and labels their regions in the in-memory blocks of one time chunk.
"""

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, assert_never

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._feature import FeatureChunk, FeatureGeometry, FeatureSource, FeatureTabulator
from rojak.atmosphere.regions import ExtremaKind, circular_footprint, identify_circular_extrema

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from rojak.atmosphere.feature_composite._feature import ChunkInfo

# Names of the chunk inputs, shared by the source (which builds them) and the tabulator (which reads their blocks)
_VALUES: str = "values"
_LOCAL_EXTREMA: str = "local_extrema"
_EXTREMA_REGIONS: str = "extrema_regions"


@dataclass(frozen=True, kw_only=True)
class ExtremaPointTabulator(FeatureTabulator):
    """Task side of :class:`ExtremaPointFeature`: tabulates the extrema and labels their regions in one time chunk.

    Attributes:
        kind (ExtremaKind): Whether the extrema are minima (lows) or maxima (highs).
        footprint_radius (int): Radius in grid cells of the circular footprint of the extrema filter.
    """

    kind: ExtremaKind
    footprint_radius: int

    def __call__(self, blocks: "Mapping[str, np.ndarray]", chunk: "ChunkInfo") -> FeatureChunk:
        """Tabulate one centre per connected component of the extrema and label the cells each centre owns.

        Args:
            blocks (Mapping[str, numpy.ndarray]): The chunk's blocks of the field values, ``local_extrema`` and
                ``extrema_regions``, each of shape ``(time, lat, lon)``.
            chunk (ChunkInfo): Coordinates of the chunk.

        Returns:
            FeatureChunk: The centres, ordered by ``centre_id``, and the labels.
        """
        values = blocks[_VALUES]
        local_extrema = blocks[_LOCAL_EXTREMA]
        extrema_regions = blocks[_EXTREMA_REGIONS]
        n_lats, n_lons = local_extrema.shape[1:]

        time_index, lat_index, lon_index, starts, stops = _components_by_time_step(local_extrema)
        centre_ids = (
            ((chunk.first_time_index + time_index[starts].astype(np.int64)) * n_lats + lat_index[starts]) * n_lons
            + lon_index[starts]
            + 1
        )
        # Order the components by id, i.e. row-major by their representative pixel
        by_id = np.argsort(centre_ids, kind="stable")
        centre_ids, starts, stops = centre_ids[by_id], starts[by_id], stops[by_id]

        centre_time_index = time_index[starts]
        centre_lat_index = lat_index[starts].astype(np.int64)
        centre_lon_index = lon_index[starts].astype(np.int64)
        centre_values = values[centre_time_index, centre_lat_index, centre_lon_index]
        centres = pd.DataFrame(
            {
                Column.CENTRE_ID: centre_ids,
                Column.FEATURE_ID: centre_ids.copy(),
                Column.CENTRE_TIME: chunk.times[centre_time_index],
                Column.CENTRE_LAT: chunk.lats[centre_lat_index].astype(float),
                Column.CENTRE_LON: chunk.lons[centre_lon_index].astype(float),
                Column.CENTRE_VALUE: centre_values,
                Column.CENTRE_LAT_INDEX: centre_lat_index,
                Column.CENTRE_LON_INDEX: centre_lon_index,
                Column.TIME_INDEX: chunk.first_time_index + centre_time_index.astype(np.int64),
            }
        )

        match self.kind:
            case ExtremaKind.MINIMA:
                depth = centre_values
            case ExtremaKind.MAXIMA:
                depth = -centre_values
            case _ as unreachable:
                assert_never(unreachable)

        # Stamp the deepest centre first, equal depths in order of id, so that a cell keeps the first centre to reach it
        footprint = circular_footprint(self.footprint_radius)
        labels = np.zeros(local_extrema.shape, dtype=np.int64)
        for component in np.lexsort((centre_ids, depth)).tolist():
            start, stop = int(starts[component]), int(stops[component])
            step = int(time_index[start])
            within, window = _within_footprint(
                lat_index[start:stop], lon_index[start:stop], footprint, grid_shape=(n_lats, n_lons)
            )
            layer = labels[step][window]  # A view, so assigning into it writes into labels
            owned = within & extrema_regions[step][window] & (layer == 0)
            layer[owned] = centre_ids[component]

        return FeatureChunk(centres=centres, labels=labels)


def _components_by_time_step(
    local_extrema: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixels of the 8-connected components of ``local_extrema`` within each time step.

    Args:
        local_extrema (numpy.ndarray): Boolean ``(time, lat, lon)`` block.

    Returns:
        tuple[numpy.ndarray, ...]: The time, lat and lon index of every extremum pixel, grouped by component and in
        row-major order within each component, then the start and stop of each component in those arrays. The start
        is the component's first pixel in row-major order, its representative.
    """
    # 8-connectivity in (lat, lon) and no connection across time steps
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True
    component_labels = np.zeros(local_extrema.shape, dtype=np.int32)
    ndi.label(local_extrema, structure=structure, output=component_labels)

    pixels = np.flatnonzero(component_labels)  # Row-major order
    component_of_pixel = component_labels.ravel()[pixels]
    grouped = np.argsort(component_of_pixel, kind="stable")  # Stable, so row-major order is kept within a component
    _, starts, sizes = np.unique(component_of_pixel[grouped], return_index=True, return_counts=True)
    time_index, lat_index, lon_index = np.unravel_index(pixels[grouped], local_extrema.shape)
    return time_index, lat_index, lon_index, starts, starts + sizes


def _within_footprint(
    pixel_lats: np.ndarray, pixel_lons: np.ndarray, footprint: np.ndarray, *, grid_shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Cells within the footprint of any of the pixels, in the smallest window of the grid that contains them.

    Args:
        pixel_lats (numpy.ndarray): Latitude indices of the pixels.
        pixel_lons (numpy.ndarray): Longitude indices of the pixels.
        footprint (numpy.ndarray): Square boolean footprint of odd side, centred on a pixel.
        grid_shape (tuple[int, int]): Number of latitudes and longitudes of the grid.

    Returns:
        tuple[numpy.ndarray, tuple[slice, slice]]: Boolean mask of the cells within the footprint, and the window of
        the grid that the mask covers.
    """
    side = footprint.shape[0]
    radius = side // 2
    lat_min, lon_min = int(pixel_lats.min()), int(pixel_lons.min())
    # Cell (0, 0) of the padded mask is grid cell (lat_min - radius, lon_min - radius), which may be off the grid
    rows, columns = pixel_lats - lat_min, pixel_lons - lon_min
    padded = np.zeros((int(rows.max()) + side, int(columns.max()) + side), dtype=bool)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        padded[row : row + side, column : column + side] |= footprint

    lat_start, lon_start = max(lat_min - radius, 0), max(lon_min - radius, 0)
    lat_stop = min(lat_min - radius + padded.shape[0], grid_shape[0])
    lon_stop = min(lon_min - radius + padded.shape[1], grid_shape[1])
    within = padded[
        lat_start - (lat_min - radius) : lat_stop - (lat_min - radius),
        lon_start - (lon_min - radius) : lon_stop - (lon_min - radius),
    ]
    return within, (slice(lat_start, lat_stop), slice(lon_start, lon_stop))


@dataclass(frozen=True, eq=False)
class ExtremaPointFeature(FeatureSource):
    """Lows or highs: one point feature per extremum of a field, found by
    :func:`~rojak.atmosphere.regions.identify_circular_extrema`.

    Each connected component (8-connectivity) of ``local_extrema`` in a time step is one centre, so separate extrema
    with equal values stay separate and a flat extremum spanning several grid cells is one centre (D7). Each centre
    owns the cells of ``extrema_regions`` within the footprint of any pixel of its component for which it is the
    deepest centre in range, which is the rule that defines ``extrema_regions``, written as a label array (D11). A
    component is represented by its first pixel in row-major order, and where equally deep centres are in range of a
    cell, the one with the smallest ``centre_id`` owns it (D19).

    ``centre_id`` is ``(time_index * n_lats + lat_index) * n_lons + lon_index + 1`` for the representative pixel, with
    ``time_index`` along the whole time axis. It is unique across chunks without a global count, independent of the
    chunking, increasing in row-major ``(time, lat, lon)`` order, and at least 1, since 0 means "no owner" in the
    labels.

    Distances are in grid cells and do not wrap around in longitude, as in the extrema filter's default boundary mode.
    Deferred: orientation from a steering flow (phase 5).

    Attributes:
        field (xarray.DataArray): Field to find the extrema in, e.g. mean sea level pressure, with exactly the
            dimensions ``time_dim_name``, ``lat_dim_name`` and ``lon_dim_name``. It may be chunked along time; it is
            rechunked to one chunk in latitude and longitude.
        kind (ExtremaKind): Minima for lows, maxima for highs.
        threshold (float | None): Only extrema below (minima) or above (maxima) it are kept. ``None`` keeps all.
        footprint_radius (int): Radius in grid cells of the circular footprint of the extrema filter, and of the
            region each centre may own. At least 1.
        time_dim_name (str): Name of the time dimension of ``field``.
        lat_dim_name (str): Name of the latitude dimension of ``field``.
        lon_dim_name (str): Name of the longitude dimension of ``field``.

    Raises:
        ValueError: If ``field`` does not have exactly the three named dimensions, or ``footprint_radius`` is below 1.
    """

    field: "xr.DataArray"
    kind: ExtremaKind
    _: KW_ONLY
    threshold: float | None = None
    footprint_radius: int = 24
    time_dim_name: str = "time"
    lat_dim_name: str = "latitude"
    lon_dim_name: str = "longitude"

    def __post_init__(self) -> None:
        """Validate the dimensions of the field and the footprint radius."""
        if len(self.field.dims) != len(self._dims) or set(self.field.dims) != set(self._dims):
            raise ValueError(
                f"field must have exactly the dimensions {self._dims}, but has {self.field.dims}. Set time_dim_name, "
                "lat_dim_name and lon_dim_name to the names of its dimensions"
            )
        if self.footprint_radius < 1:
            raise ValueError(f"footprint_radius must be at least 1 grid cell, but is {self.footprint_radius}")

    @property
    def _dims(self) -> tuple[str, str, str]:
        return self.time_dim_name, self.lat_dim_name, self.lon_dim_name

    @property
    def geometry(self) -> FeatureGeometry:
        """Each low or high is a point."""
        return FeatureGeometry.POINT

    @property
    def has_labels(self) -> bool:
        """The tabulator labels the region that each centre owns."""
        return True

    def chunk_inputs(self) -> "Mapping[str, xr.DataArray]":
        """The field values, ``local_extrema`` and ``extrema_regions``, built lazily without computing anything.

        ``local_extrema`` and ``extrema_regions`` share the extrema filter task of each chunk.

        Returns:
            Mapping[str, xarray.DataArray]: The three arrays with dimensions ``(time, lat, lon)``, chunked like the
            field along time and in one chunk along latitude and longitude.
        """
        field = self.field.transpose(*self._dims).chunk({self.lat_dim_name: -1, self.lon_dim_name: -1})
        extrema = identify_circular_extrema(
            field,
            self.kind,
            extrema_threshold_value=self.threshold,
            footprint_radius=self.footprint_radius,
            lat_dim_name=self.lat_dim_name,
            lon_dim_name=self.lon_dim_name,
        )
        return {
            _VALUES: field,
            _LOCAL_EXTREMA: extrema["local_extrema"],
            _EXTREMA_REGIONS: extrema["extrema_regions"],
        }

    def tabulator(self) -> ExtremaPointTabulator:
        """The parameters-only tabulator sent to each chunk task."""
        return ExtremaPointTabulator(kind=self.kind, footprint_radius=self.footprint_radius)
