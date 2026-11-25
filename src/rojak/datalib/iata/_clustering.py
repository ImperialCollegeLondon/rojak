#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Spatio-temporal density-based clustering of point observations."""

import math
from typing import TYPE_CHECKING

import numba
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from rojak.core.constants import EARTH_AVG_RADIUS
from rojak.core.geometric import DistanceUnits, haversine_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def st_dbscan(
    time: "NDArray[np.datetime64]",
    lat_deg: "NDArray[np.floating]",
    lon_deg: "NDArray[np.floating]",
    altitude: "NDArray[np.floating] | None",
    /,
    *,
    n_points: int,
    not_in_cluster_label: int,
    max_separation_km: float,
    max_minutes: float,
    max_altitude_delta: float | None = None,
) -> "NDArray[np.int64]":
    """
    Cluster point observations by density in space, time and, optionally, altitude

    Two points are neighbours when their great-circle separation is at most ``max_separation_km``, their
    timestamps differ by at most ``max_minutes`` and, if altitudes are given, their altitudes differ by at most
    ``max_altitude_delta``. Every point counts itself as a neighbour. A point with at least ``n_points``
    neighbours is a core point; core points that are neighbours belong to the same cluster.

    Cluster ids are numbered from 0 in ascending order of the first core row of each cluster. A non-core point
    with at least one core neighbour is a border point and takes the lowest cluster id among its core neighbours,
    so the result does not depend on the order in which points are visited. Every other point is noise and takes
    the label ``not_in_cluster_label``.

    Args:
        time: Timestamp of each point. Any ``datetime64`` resolution is accepted.
        lat_deg: Latitude of each point in degrees.
        lon_deg: Longitude of each point in degrees.
        altitude: Altitude of each point, in the same units as ``max_altitude_delta``, or ``None`` to ignore
            altitude.
        n_points: Minimum number of neighbours, the point itself included, for a point to be core.
        not_in_cluster_label: Label of the points that are in no cluster. Must be negative, so that it cannot be
            mistaken for a cluster id.
        max_separation_km: Largest great-circle separation in kilometres between neighbours.
        max_minutes: Largest time difference in minutes between neighbours.
        max_altitude_delta: Largest altitude difference between neighbours. Must be given exactly when
            ``altitude`` is given.

    Returns:
        Cluster id of each point, with ``not_in_cluster_label`` for noise.

    Raises:
        ValueError: If the arrays are not one-dimensional with the same length, ``n_points`` is below 1,
            ``not_in_cluster_label`` is not negative, a threshold is negative or NaN, only one of ``altitude`` and
            ``max_altitude_delta`` is given, a coordinate or altitude is not finite, or a timestamp is NaT.

    Examples:
    >>> time = np.array(["2024-06-01T00:00", "2024-06-01T00:05", "2024-06-01T00:05"], dtype="datetime64[m]")
    >>> st_dbscan(time, np.zeros(3), np.array([0.0, 0.01, 1.0]), None,
    ...           n_points=2, not_in_cluster_label=-1, max_separation_km=5.0, max_minutes=10.0).tolist()
    [0, 0, -1]
    """
    _validate(
        time,
        lat_deg,
        lon_deg,
        altitude,
        n_points,
        not_in_cluster_label,
        max_separation_km,
        max_minutes,
        max_altitude_delta,
    )

    n_obs = time.shape[0]
    if n_obs == 0:
        return np.empty(0, dtype=np.int64)

    first, second = _neighbour_pairs(
        np.asarray(time),
        np.asarray(lat_deg, dtype=np.float64),
        np.asarray(lon_deg, dtype=np.float64),
        None if altitude is None else np.asarray(altitude, dtype=np.float64),
        max_separation_km,
        max_minutes,
        max_altitude_delta,
    )
    # Each point is its own neighbour, which the pairs do not include
    degree = np.bincount(first, minlength=n_obs) + np.bincount(second, minlength=n_obs) + 1
    is_core = degree >= n_points

    labels = _label_core_points(first, second, is_core, not_in_cluster_label)
    _label_border_points(labels, first, second, is_core)
    return labels


def _validate(
    time: "NDArray[np.datetime64]",
    lat_deg: "NDArray[np.floating]",
    lon_deg: "NDArray[np.floating]",
    altitude: "NDArray[np.floating] | None",
    n_points: int,
    not_in_cluster_label: int,
    max_separation_km: float,
    max_minutes: float,
    max_altitude_delta: float | None,
) -> None:
    """Raise ``ValueError`` for any input :func:`st_dbscan` cannot cluster."""
    arrays = [time, lat_deg, lon_deg] if altitude is None else [time, lat_deg, lon_deg, altitude]
    shapes = [np.shape(array) for array in arrays]
    if any(len(shape) != 1 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError(
            "time, lat_deg, lon_deg and altitude must be one-dimensional arrays of the same length, "
            f"got shapes {shapes}"
        )
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")
    if not_in_cluster_label >= 0:
        raise ValueError(f"not_in_cluster_label must be negative, got {not_in_cluster_label}")
    thresholds = {
        "max_separation_km": max_separation_km,
        "max_minutes": max_minutes,
        "max_altitude_delta": max_altitude_delta,
    }
    for name, value in thresholds.items():
        if value is not None and not value >= 0:  # written so that NaN is rejected too
            raise ValueError(f"{name} must be a non-negative number, got {value}")
    if (altitude is None) != (max_altitude_delta is None):
        raise ValueError("altitude and max_altitude_delta must be given together, or both left as None")
    for name, array in (("lat_deg", lat_deg), ("lon_deg", lon_deg), ("altitude", altitude)):
        if array is not None and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(np.isnat(time)):
        raise ValueError("time must not contain NaT")


def _neighbour_pairs(
    time: "NDArray[np.datetime64]",
    lat_deg: "NDArray[np.float64]",
    lon_deg: "NDArray[np.float64]",
    altitude: "NDArray[np.float64] | None",
    max_separation_km: float,
    max_minutes: float,
    max_altitude_delta: float | None,
    *,
    chord_tolerance: float = 1e-9,
    max_window_ns: float = 2.0**62,
) -> "tuple[NDArray[np.intp], NDArray[np.intp]]":
    """
    Index pairs ``(i, j)`` with ``i < j`` of the points that are neighbours

    Candidate pairs come from a k-d tree over unit vectors on the sphere, searched with the chord length that
    corresponds to ``max_separation_km``. The candidates are then filtered exactly by haversine distance, time
    difference and altitude difference in a single compiled pass over the pairs.

    Args:
        chord_tolerance: Relative widening of the tree search, so that pairs exactly on the distance threshold
            survive rounding in the chord length; the exact haversine test that follows decides membership.
        max_window_ns: Cap on the time window in nanoseconds, so that an unbounded window (inf minutes) converts to an
            integer. The default of 2**62 ns is about 146 years, far wider than any realistic span of observations.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    unit_vectors = np.column_stack(
        (np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad))
    )
    earth_radius_km: float = EARTH_AVG_RADIUS * 1e-3
    # Beyond pi radians every pair of points on the sphere is within range, so cap the angle there
    max_angle = min(max_separation_km / earth_radius_km, math.pi)
    chord_radius = 2.0 * math.sin(max_angle / 2.0) * (1.0 + chord_tolerance)
    pairs: NDArray[np.intp] = np.ascontiguousarray(
        cKDTree(unit_vectors).query_pairs(chord_radius, output_type="ndarray"), dtype=np.intp
    )

    # Integer nanoseconds keep the time comparison exact whatever the input resolution
    time_ns = np.ascontiguousarray(time.astype("datetime64[ns]").view(np.int64))
    max_ns = round(min(max_minutes * 60e9, max_window_ns))
    keep = neighbour_pair_mask(
        pairs,
        np.ascontiguousarray(lon_deg),
        np.ascontiguousarray(lat_deg),
        time_ns,
        # The kernel never reads altitude when it is not used, so an empty placeholder keeps its argument types fixed
        np.empty(0, dtype=np.float64) if altitude is None else np.ascontiguousarray(altitude),
        max_separation_km,
        max_ns,
        0.0 if max_altitude_delta is None else max_altitude_delta,
        altitude is not None,
    )
    kept = pairs[keep]
    return kept[:, 0], kept[:, 1]


def _label_core_points(
    first: "NDArray[np.intp]",
    second: "NDArray[np.intp]",
    is_core: "NDArray[np.bool_]",
    not_in_cluster_label: int,
) -> "NDArray[np.int64]":
    """
    Cluster ids for core points, numbered by their first core row, and -1 everywhere else

    Core points are joined into clusters by the connected components of the graph of core-core neighbour pairs.
    """
    n_obs = is_core.shape[0]
    core_edge = is_core[first] & is_core[second]
    graph = coo_array(
        (np.ones(np.count_nonzero(core_edge), dtype=np.int8), (first[core_edge], second[core_edge])),
        shape=(n_obs, n_obs),
    )
    _, component = connected_components(graph, directed=False)

    labels = np.full(n_obs, not_in_cluster_label, dtype=np.int64)
    core_rows = np.flatnonzero(is_core)
    # np.unique sorts by component id; rank the components instead by the row at which each first appears
    _, first_row, core_component = np.unique(component[core_rows], return_index=True, return_inverse=True)
    rank_by_first_row = np.argsort(np.argsort(first_row))
    labels[core_rows] = rank_by_first_row[core_component]
    return labels


def _label_border_points(
    labels: "NDArray[np.int64]", first: "NDArray[np.intp]", second: "NDArray[np.intp]", is_core: "NDArray[np.bool_]"
) -> None:
    """Give each non-core point with a core neighbour the lowest cluster id among those neighbours, in place."""
    point = np.concatenate((first, second))
    neighbour = np.concatenate((second, first))
    to_core = ~is_core[point] & is_core[neighbour]
    point = point[to_core]
    neighbour_label = labels[neighbour[to_core]]

    # Sort by point, then by label, so the first entry for each point carries its lowest adjacent label
    order = np.lexsort((neighbour_label, point))
    point = point[order]
    neighbour_label = neighbour_label[order]
    is_first = np.ones(point.shape[0], dtype=np.bool_)
    is_first[1:] = point[1:] != point[:-1]
    labels[point[is_first]] = neighbour_label[is_first]


# parallel=True is safe here because the clustering runs on the client, never inside a dask task, so these threads
# do not compete with dask worker threads for the same cores.
@numba.njit(parallel=True)
def neighbour_pair_mask(
    pairs: "NDArray[np.intp]",
    lon_deg: "NDArray[np.float64]",
    lat_deg: "NDArray[np.float64]",
    time_ns: "NDArray[np.int64]",
    altitude: "NDArray[np.float64]",
    max_separation_km: float,
    max_ns: int,
    max_altitude_delta: float,
    use_altitude: bool,
) -> "NDArray[np.bool_]":
    """
    Which candidate pairs are neighbours in space, time and, optionally, altitude

    Args:
        pairs: Candidate index pairs, shape ``(n_pairs, 2)``.
        lon_deg: Longitude of each point in degrees.
        lat_deg: Latitude of each point in degrees.
        time_ns: Timestamp of each point as integer nanoseconds.
        altitude: Altitude of each point. Never read when ``use_altitude`` is false, so may then be empty.
        max_separation_km: Largest great-circle separation in kilometres between neighbours.
        max_ns: Largest time difference in nanoseconds between neighbours.
        max_altitude_delta: Largest altitude difference between neighbours. Ignored when ``use_altitude`` is false.
        use_altitude: Whether to apply the altitude test.

    Returns:
        Boolean mask over the rows of ``pairs``, true where the pair are neighbours.
    """
    n_pairs = pairs.shape[0]
    keep = np.empty(n_pairs, dtype=np.bool_)
    for k in numba.prange(n_pairs):
        i = pairs[k, 0]
        j = pairs[k, 1]
        separation_km = haversine_distance(lon_deg[i], lat_deg[i], lon_deg[j], lat_deg[j], DistanceUnits.KILOMETERS)
        delta_ns = time_ns[i] - time_ns[j]
        if delta_ns < 0:
            delta_ns = -delta_ns
        keep[k] = (
            separation_km <= max_separation_km
            and delta_ns <= max_ns
            and (not use_altitude or abs(altitude[i] - altitude[j]) <= max_altitude_delta)
        )
    return keep
