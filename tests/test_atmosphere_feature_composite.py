"""Characterisation tests for compositing observations about features (phase 0 of ``storm_composite_plan.md``).

They pin down a behavioural baseline captured from the prototype ``composite_points_about_lows.py`` (deleted at the
end of phase 1) against brute-force calculations written independently of it. Phase 1 retargets them to
:mod:`rojak.atmosphere.feature_composite._pipeline` by rewriting only the adapter section below; everything else
(synthetic data, brute force and tests) is unchanged from phase 0.

Layout:

1. Synthetic data (copied from the prototype so that it outlives it).
2. Brute force, written independently of the prototype.
3. Adapter: the only code that calls the implementation under test.
4. Tests.
"""

import contextlib
import functools
import warnings
from collections.abc import Callable, Iterator
from enum import StrEnum
from pkgutil import resolve_name
from typing import TYPE_CHECKING, Any, NamedTuple, assert_never

import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyproj
import pytest
import scipy.ndimage as ndi
import xarray as xr
from dask.base import compute
from dask.local import get_sync
from numpy.typing import NDArray

from rojak.atmosphere.regions import ExtremaKind, identify_circular_extrema
from rojak.core.geometric import haversine_distance
from rojak.core.indexing import nearest_index

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

THRESHOLD: float = 100_000
FOOTPRINT_RADIUS: int = 16
DISTANCE_KM: float = 500
MAX_KM_EXTENT: int = 1500
KM_SPACING: int = 50
EXCLUSION_DISTANCE_KM: float = 300
# +/- 6 h controls are within 300 km of the moving lows (so the exclusion discards them), the others are not
CONTROL_SHIFTS: tuple[np.timedelta64, ...] = tuple(
    np.timedelta64(sign * hours, "h") for hours in (6, 12, 18) for sign in (-1, 1)
)
N_PARTITIONS: int = 4
N_POINTS_PER_PARTITION: int = 2500
EVENT: str = "is_event"
EVENT_LATITUDE: float = 50


class Pairing(StrEnum):
    """How observations are paired with centres (test-local ids, mapped to the implementation by the adapter).

    Attributes:
        REGION: The nearest grid cell of the observation is in the labelled region of the centre.
        CIRCLE: Great circle distance to the centre of at most ``DISTANCE_KM``.
        SQUARE: ``|x_km|, |y_km| <= DISTANCE_KM`` in the orthographic frame of the centre.
    """

    REGION = "region"
    CIRCLE = "circle"
    SQUARE = "square"


class CompositeResult(NamedTuple):
    """Computed composite, as returned by the adapter.

    Attributes:
        centres: Columns ``centre_time``, ``centre_lat``, ``centre_lon``: one row per centre.
        pairs: Columns ``id``, ``centre_time``, ``centre_lat``, ``centre_lon``, ``x_km``, ``y_km``: one row per pair.
        counts: Number of observations per ``(y_km, x_km)`` cell, see :func:`_histogram_edges`.
        n_centres: Number of centres.
    """

    centres: pd.DataFrame
    pairs: pd.DataFrame
    counts: np.ndarray
    n_centres: int


# ======================================================================================================================
# 1. Synthetic data
# ======================================================================================================================


def _synthetic_mslp() -> xr.DataArray:
    """Two lows moving east on a 0.5 degree grid, chunked along time."""
    times = pd.date_range("2024-01-01", periods=8, freq="6h")
    lats = np.arange(80, 19.75, -0.5)  # Descending, as in ERA5
    lons = np.arange(-70, 40.25, 0.5)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    fields = []
    for step in range(times.size):
        field = 101500 + 300 * np.sin(np.radians(lat_grid))
        for centre_lat, centre_lon, speed in ((55, -45, 3), (40, -5, 2)):
            squared_distance = (lat_grid - centre_lat) ** 2 + (lon_grid - (centre_lon + speed * step)) ** 2
            field = field - 3000 * np.exp(-squared_distance / (2 * 5**2))
        fields.append(field)
    return xr.DataArray(
        np.stack(fields),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lats, "longitude": lons},
        name="mslp",
    ).chunk({"time": 3})


def _random_points(
    seed: int, n_points: int, start: np.datetime64, n_minutes: int, extent: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Uniform observations, some outside of the grid and up to 3 h outside of the time range.

    ``id`` is unique across seeds. ``is_event`` is a 0/1 int which only depends on latitude and id, independently of the
    lows.
    """
    rng = np.random.default_rng(seed)
    min_lat, max_lat, min_lon, max_lon = extent
    latitude = rng.uniform(min_lat, max_lat, n_points)
    longitude = rng.uniform(min_lon, max_lon, n_points)
    ids = seed * n_points + np.arange(n_points)
    return pd.DataFrame(
        {
            "time": start + rng.integers(-3 * 60, n_minutes + 3 * 60, n_points).astype("timedelta64[m]"),
            "latitude": latitude,
            "longitude": longitude,
            "id": ids,
            EVENT: ((latitude > EVENT_LATITUDE) | (ids % 10 == 0)).astype(np.int64),
        }
    )


def _make_partition_factory(mslp: xr.DataArray) -> Callable[[int], pd.DataFrame]:
    return functools.partial(
        _random_points,
        n_points=N_POINTS_PER_PARTITION,
        start=mslp["time"].to_numpy()[0],
        n_minutes=(mslp["time"].size - 1) * 6 * 60,
        extent=(10, 85, -80, 50),
    )


@contextlib.contextmanager
def _count_computes() -> Iterator[list[int]]:
    """Count the number of times something is computed, using the synchronous scheduler."""
    computes: list[int] = []

    def counting_get(*args: Any, **kwargs: Any) -> Any:
        computes.append(1)
        return get_sync(*args, **kwargs)

    with dask.config.set(scheduler=counting_get):
        yield computes


def _counting(function: Callable[..., Any], calls: list[int]) -> Callable[..., Any]:
    """Wrap ``function`` so that each call is appended to ``calls``. A plain closure, so dask can tokenise it."""

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return function(*args, **kwargs)

    return counted


@pytest.fixture(scope="module")
def mslp() -> xr.DataArray:
    return _synthetic_mslp()


@pytest.fixture(scope="module")
def points(mslp: xr.DataArray) -> pd.DataFrame:
    make_partition = _make_partition_factory(mslp)
    return pd.concat([make_partition(seed) for seed in range(N_PARTITIONS)], ignore_index=True)


@pytest.fixture(scope="module")
def points_dd(mslp: xr.DataArray) -> dd.DataFrame:
    """The same observations as ``points``, generated lazily in each partition."""
    make_partition = _make_partition_factory(mslp)
    return dd.from_map(make_partition, range(N_PARTITIONS), meta=make_partition(0).iloc[:0])


@pytest.fixture(scope="module")
def extrema(mslp: xr.DataArray) -> xr.Dataset:
    return (
        identify_circular_extrema(
            mslp, ExtremaKind.MINIMA, extrema_threshold_value=THRESHOLD, footprint_radius=FOOTPRINT_RADIUS
        )
        .transpose("time", "latitude", "longitude")
        .compute()
    )


@pytest.fixture(scope="module")
def expected_centres(mslp: xr.DataArray, extrema: xr.Dataset) -> pd.DataFrame:
    """One centre per connected component of ``local_extrema`` per time step (plan decision D7).

    Also asserts the D7 guard: every component is a single pixel and the values of the lows are unique per time step.
    The prototype merges lows with equal values, which phase 1 deliberately changes, so the synthetic data must never
    exercise it.
    """
    field_times = _field_times(mslp)
    lats = mslp["latitude"].to_numpy()
    lons = mslp["longitude"].to_numpy()
    values = mslp.to_numpy()
    local_extrema = extrema["local_extrema"].to_numpy()
    eight_connected = ndi.generate_binary_structure(2, 2)

    time_indices: list[int] = []
    lat_indices: list[int] = []
    lon_indices: list[int] = []
    for time_index in range(field_times.size):
        labelled, n_components = ndi.label(local_extrema[time_index], structure=eight_connected)
        for component in range(1, n_components + 1):
            lat_index, lon_index = np.nonzero(labelled == component)
            assert lat_index.size == 1, "D7 guard: a low spans several pixels"
            time_indices.append(time_index)
            lat_indices.append(int(lat_index[0]))
            lon_indices.append(int(lon_index[0]))

    time_index_array = np.array(time_indices, dtype=np.int64)
    lat_index_array = np.array(lat_indices, dtype=np.int64)
    lon_index_array = np.array(lon_indices, dtype=np.int64)
    centres = pd.DataFrame(
        {
            "centre_index": np.arange(time_index_array.size),
            "time_index": time_index_array,
            "centre_lat_index": lat_index_array,
            "centre_lon_index": lon_index_array,
            "centre_time": field_times[time_index_array],
            "centre_lat": lats[lat_index_array].astype(float),
            "centre_lon": lons[lon_index_array].astype(float),
            "centre_value": values[time_index_array, lat_index_array, lon_index_array],
        }
    )
    assert not centres.duplicated(["time_index", "centre_value"]).any(), "D7 guard: lows with equal values"
    # The synthetic field has two lows at every time step
    assert len(centres) == 2 * field_times.size
    return centres


# ======================================================================================================================
# 2. Brute force, independent of the implementation under test
# ======================================================================================================================


def _field_times(mslp: xr.DataArray) -> NDArray[np.datetime64]:
    return mslp["time"].to_numpy().astype("datetime64[ns]")


def _half_time_step(field_times: NDArray[np.datetime64]) -> np.timedelta64:
    """Half the median time step of the field: the tolerance used to snap observations to a time step."""
    median_step_ns = float(np.median(np.diff(field_times).astype(np.int64)))
    return np.timedelta64(round(median_step_ns / 2), "ns")


def _observations_at_time_steps(
    points: pd.DataFrame, field_times: NDArray[np.datetime64], shift: np.timedelta64 | None = None
) -> pd.DataFrame:
    """Observations (with times moved back by ``shift``) within half a time step of a field time, and its index."""
    # Times are whole minutes, so some are exactly half-way between two field times: nearest_index sends those to the
    # later time step, and the implementation must agree
    observation_times = points["time"].to_numpy().astype("datetime64[ns]")
    if shift is not None:
        observation_times = observation_times - shift
    time_index = nearest_index(field_times, observation_times)
    is_snapped = np.abs(field_times[time_index] - observation_times) <= _half_time_step(field_times)
    return points[is_snapped].assign(time_index=time_index[is_snapped])


def _is_within_cells(coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Whether ``values`` lie within the cells of the regularly spaced ``coordinate``, i.e. up to half a cell beyond its
    outermost points. Used for pairing: an observation in a cell takes the region label of that (its nearest) cell."""
    half_spacing = np.abs(np.diff(coordinate)).min() / 2
    return (values >= coordinate.min() - half_spacing) & (values <= coordinate.max() + half_spacing)


def _is_within_outermost_points(coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Whether ``values`` lie within ``[min, max]`` of the points of ``coordinate``. Used for the domain (plan D15),
    which ends at the outermost grid points, not half a cell beyond them."""
    return (values >= coordinate.min()) & (values <= coordinate.max())


def _distance_km(frame: pd.DataFrame) -> np.ndarray:
    return (
        haversine_distance(
            frame["longitude"].to_numpy(dtype=float),
            frame["latitude"].to_numpy(dtype=float),
            frame["centre_lon"].to_numpy(dtype=float),
            frame["centre_lat"].to_numpy(dtype=float),
        )
        / 1000
    )


def _project_about_centres(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the position of each observation in the WGS-84 orthographic (km) frame of its centre.

    Observations on the far side of the globe project to non-finite values and are dropped.
    """
    x_km = np.full(len(frame), np.nan)
    y_km = np.full(len(frame), np.nan)
    observation_lats = frame["latitude"].to_numpy(dtype=float)
    observation_lons = frame["longitude"].to_numpy(dtype=float)
    centre_lats = frame["centre_lat"].to_numpy(dtype=float)
    centre_lons = frame["centre_lon"].to_numpy(dtype=float)
    centre_of_row = frame["centre_index"].to_numpy()
    for centre_index in np.unique(centre_of_row):
        rows = centre_of_row == centre_index
        first = np.flatnonzero(rows)[0]
        ortho = pyproj.Proj(
            proj="ortho",
            lat_0=float(centre_lats[first]),
            lon_0=float(centre_lons[first]),
            units="km",
            ellps="WGS84",
        )
        x_km[rows], y_km[rows] = ortho(observation_lons[rows], observation_lats[rows])
    is_finite = np.isfinite(x_km) & np.isfinite(y_km)
    return frame.assign(x_km=x_km, y_km=y_km)[is_finite]


def _region_owners(
    mslp: xr.DataArray, observations: pd.DataFrame, extrema: xr.Dataset, centres: pd.DataFrame
) -> pd.DataFrame:
    """Pair each observation whose nearest grid cell is in ``extrema_regions`` with the deepest centre at the same time
    step within the footprint (in grid indices) of that cell (plan decision D11)."""
    lats = mslp["latitude"].to_numpy()
    lons = mslp["longitude"].to_numpy()
    observation_lats = observations["latitude"].to_numpy(dtype=float)
    # The brute force is only given observations in the same longitude convention as the grid ([-180, 180]), so no
    # conversion is needed
    observation_lons = observations["longitude"].to_numpy(dtype=float)
    is_on_grid = _is_within_cells(lats, observation_lats) & _is_within_cells(lons, observation_lons)
    lat_index = nearest_index(lats, observation_lats[is_on_grid])
    lon_index = nearest_index(lons, observation_lons[is_on_grid])
    time_index = observations["time_index"].to_numpy()[is_on_grid]
    in_region = extrema["extrema_regions"].to_numpy()[time_index, lat_index, lon_index]
    on_grid = observations[is_on_grid]

    centre_time_index = centres["time_index"].to_numpy()
    centre_lat_index = centres["centre_lat_index"].to_numpy()
    centre_lon_index = centres["centre_lon_index"].to_numpy()
    centre_value = centres["centre_value"].to_numpy()
    centre_index = centres["centre_index"].to_numpy()
    owners = np.full(len(on_grid), -1)
    # Every observation in a region against every centre
    for row in np.flatnonzero(in_region):
        is_candidate = (centre_time_index == time_index[row]) & (
            (centre_lat_index - lat_index[row]) ** 2 + (centre_lon_index - lon_index[row]) ** 2 <= FOOTPRINT_RADIUS**2
        )
        candidates = np.flatnonzero(is_candidate)
        if candidates.size > 0:
            owners[row] = centre_index[candidates[np.argmin(centre_value[candidates])]]
    owned = on_grid.assign(centre_index=owners)[owners >= 0]
    return owned.merge(centres, on=["centre_index", "time_index"])


def _brute_force_pairs(
    mslp: xr.DataArray, points: pd.DataFrame, extrema: xr.Dataset, centres: pd.DataFrame, pairing: Pairing
) -> pd.DataFrame:
    """Every (observation, centre) pair with ``x_km`` and ``y_km``, by an exhaustive cross join per time step."""
    observations = _observations_at_time_steps(points, _field_times(mslp))
    match pairing:
        case Pairing.REGION:
            return _project_about_centres(_region_owners(mslp, observations, extrema, centres))
        case Pairing.CIRCLE:
            cross = observations.merge(centres, on="time_index")
            return _project_about_centres(cross[_distance_km(cross) <= DISTANCE_KM])
        case Pairing.SQUARE:
            projected = _project_about_centres(observations.merge(centres, on="time_index"))
            return projected[(projected["x_km"].abs() <= DISTANCE_KM) & (projected["y_km"].abs() <= DISTANCE_KM)]
        case _ as unreachable:
            assert_never(unreachable)


def _brute_force_valid_controls(
    centres: pd.DataFrame, field_times: NDArray[np.datetime64], exclusion_distance_km: float | None
) -> set[tuple[int, int]]:
    """The valid controls as ``(centre_index, position of the shift in CONTROL_SHIFTS)``.

    A control is valid if the centre time plus the shift is within the time range of the field and, with an exclusion
    distance, no centre at that time is within the exclusion distance of the centre.
    """
    centre_index = centres["centre_index"].to_numpy()
    centre_times = centres["centre_time"].to_numpy()
    centre_lats = centres["centre_lat"].to_numpy(dtype=float)
    centre_lons = centres["centre_lon"].to_numpy(dtype=float)
    valid: set[tuple[int, int]] = set()
    for index, time, lat, lon in zip(centre_index, centre_times, centre_lats, centre_lons, strict=True):
        for position, shift in enumerate(CONTROL_SHIFTS):
            control_time = time + shift
            if not field_times.min() <= control_time <= field_times.max():
                continue
            if exclusion_distance_km is not None:
                is_at_control_time = centre_times == control_time
                n_others = int(is_at_control_time.sum())
                distance_km = (
                    haversine_distance(
                        np.full(n_others, lon),
                        np.full(n_others, lat),
                        centre_lons[is_at_control_time],
                        centre_lats[is_at_control_time],
                    )
                    / 1000
                )
                if np.any(distance_km <= exclusion_distance_km):
                    continue
            valid.add((int(index), position))
    return valid


def _brute_force_control_counts(
    points: pd.DataFrame,
    centres: pd.DataFrame,
    field_times: NDArray[np.datetime64],
    valid_controls: set[tuple[int, int]],
) -> tuple[int, int]:
    """Observations and events of the valid controls.

    For a valid control (centre, shift k): the observations whose time minus k snaps to the centre time (within half a
    time step) and which are within ``DISTANCE_KM`` of the centre. All of them are well within the km grid.
    """
    n_observations = 0
    n_events = 0
    for position, shift in enumerate(CONTROL_SHIFTS):
        cross = _observations_at_time_steps(points, field_times, shift=shift).merge(centres, on="time_index")
        is_valid = np.array(
            [(centre, position) in valid_controls for centre in cross["centre_index"].to_numpy().tolist()], dtype=bool
        )
        selected = cross[is_valid & (_distance_km(cross) <= DISTANCE_KM)]
        n_observations += len(selected)
        n_events += int(selected[EVENT].to_numpy().sum())
    return n_observations, n_events


def _brute_force_domain_counts(mslp: xr.DataArray, points: pd.DataFrame) -> tuple[int, int]:
    """Observations, and events, in the domain of the field.

    Plan D15: within the outermost latitude and longitude points of the field, and within half a time step of one of
    its time steps. The observations are in the longitude convention of the field.
    """
    in_time = _observations_at_time_steps(points, _field_times(mslp))
    is_in_space = _is_within_outermost_points(
        mslp["latitude"].to_numpy(), in_time["latitude"].to_numpy(dtype=float)
    ) & _is_within_outermost_points(mslp["longitude"].to_numpy(), in_time["longitude"].to_numpy(dtype=float))
    return int(is_in_space.sum()), int(in_time[EVENT].to_numpy()[is_in_space].sum())


def _pair_keys(frame: pd.DataFrame) -> set[tuple[int, int, float, float]]:
    """``(observation id, centre time in ns, centre latitude, centre longitude)`` of each pair."""
    centre_times = frame["centre_time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    return set(
        zip(
            frame["id"].to_numpy().tolist(),
            centre_times.tolist(),
            frame["centre_lat"].to_numpy(dtype=float).tolist(),
            frame["centre_lon"].to_numpy(dtype=float).tolist(),
            strict=True,
        )
    )


def _centre_keys(frame: pd.DataFrame) -> set[tuple[int, float, float]]:
    centre_times = frame["centre_time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    return set(
        zip(
            centre_times.tolist(),
            frame["centre_lat"].to_numpy(dtype=float).tolist(),
            frame["centre_lon"].to_numpy(dtype=float).tolist(),
            strict=True,
        )
    )


def _histogram_edges() -> np.ndarray:
    """Cells centred on the km coordinates ``-MAX_KM_EXTENT, ..., MAX_KM_EXTENT``, as the gridded composites."""
    km_coords = np.arange(-MAX_KM_EXTENT, MAX_KM_EXTENT + KM_SPACING, KM_SPACING)
    return np.append(km_coords - KM_SPACING / 2, km_coords[-1] + KM_SPACING / 2)


# ======================================================================================================================
# 3. Adapter
#
# The only code that calls the implementation under test. Phase 1 rewrites ONLY this block to target
# rojak.atmosphere.feature_composite; the results stay neutral pandas/numpy keyed by centre position (time, lat, lon).
# Features of later phases are gated by pytest.skip("phase N: ...") inside the adapter functions that need them.
# ======================================================================================================================

# These are adapter-only imports (kept out of the module-level import block above), so each needs its own E402 waiver
from rojak.atmosphere.compositing import _TargetGrid  # noqa: E402
from rojak.atmosphere.feature_composite._binning import BinSpec  # noqa: E402
from rojak.atmosphere.feature_composite._columns import Column  # noqa: E402
from rojak.atmosphere.feature_composite._controls import ControlDesign  # noqa: E402
from rojak.atmosphere.feature_composite._pairing import (  # noqa: E402
    DistancePairing,
    LabelledRegionPairing,
    PairingStrategy,
)
from rojak.atmosphere.feature_composite._pipeline import composite_observations_about_features  # noqa: E402
from rojak.atmosphere.feature_composite._point_features import ExtremaPointFeature  # noqa: E402

# Function run by each chunk task, patched (by name, before building the graph) to count how often it runs
CHUNK_TASK: str = "rojak.atmosphere.feature_composite._pipeline._composite_chunk"
# Extrema filter run once per time step. Looked up when the graph is built, so it must be patched before building
EXTREMA_FILTER: str = "scipy.ndimage.minimum_filter"
# The one place allowed to compute while the graph is built (counting the observations per chunk)
GRAPH_BUILD_COMPUTE: tuple[type, str] = (_TargetGrid, "group_by_chunk")

type LazyComposite = tuple[xr.Dataset, dd.DataFrame, dd.DataFrame]


def _lows(mslp: xr.DataArray) -> ExtremaPointFeature:
    return ExtremaPointFeature(mslp, ExtremaKind.MINIMA, threshold=THRESHOLD, footprint_radius=FOOTPRINT_RADIUS)


def _pairing_strategy(pairing: Pairing) -> PairingStrategy:
    match pairing:
        case Pairing.REGION:
            return LabelledRegionPairing()
        case Pairing.CIRCLE:
            return DistancePairing(DISTANCE_KM, shape="circle")
        case Pairing.SQUARE:
            return DistancePairing(DISTANCE_KM, shape="square")
        case _ as unreachable:
            assert_never(unreachable)


def build(mslp: xr.DataArray, points: dd.DataFrame, pairing: Pairing) -> LazyComposite:
    """Build the lazy composite (counts, centres, pairs) without computing it."""
    return composite_observations_about_features(
        _lows(mslp),
        points,
        pairing=_pairing_strategy(pairing),
        bins=BinSpec.grid(MAX_KM_EXTENT, KM_SPACING),
        return_pairs=True,
    )


def _compute_all(lazy: LazyComposite) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """Compute the counts, centres and pairs in a single ``dask.compute``, so that they share the chunk tasks."""
    counts, centres, pairs = lazy
    with warnings.catch_warnings():
        # Mixing a Dataset and dataframes in one compute is what lets them share (and the counters see) the chunk tasks
        warnings.filterwarnings("ignore", message="Computing mixed collections")
        counts, centres, pairs = compute(counts, centres, pairs, scheduler="sync")
    return counts, centres, pairs


def compute_together(lazy: LazyComposite) -> None:
    """Compute every output in a single ``dask.compute``, so that shared tasks can be counted."""
    _compute_all(lazy)


def _as_centres(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "centre_time": frame[Column.CENTRE_TIME].to_numpy().astype("datetime64[ns]"),
            "centre_lat": frame[Column.CENTRE_LAT].to_numpy(dtype=float),
            "centre_lon": frame[Column.CENTRE_LON].to_numpy(dtype=float),
        }
    )


def composite(mslp: xr.DataArray, points: dd.DataFrame, pairing: Pairing) -> CompositeResult:
    """Build and compute the composite (centres, pairs and counts) in one pass."""
    counts, centres, pairs = _compute_all(build(mslp, points, pairing))
    histogram = counts["feature_n_obs"].sum("chunk").transpose("y_km", "x_km").to_numpy()
    return CompositeResult(
        centres=_as_centres(centres),
        pairs=_as_centres(pairs).assign(
            id=pairs["id"].to_numpy(),
            x_km=pairs[Column.X_KM].to_numpy(dtype=float),
            y_km=pairs[Column.Y_KM].to_numpy(dtype=float),
        ),
        counts=histogram,
        n_centres=int(counts["n_centres"].sum()),
    )


def build_controls(
    mslp: xr.DataArray, points: dd.DataFrame, *, exclusion_distance_km: float | None, event: str | None
) -> xr.Dataset:
    """Build the lazy composite with observation-shift controls (``CONTROL_SHIFTS``, circle pairing) without
    computing it."""
    if exclusion_distance_km is not None:
        pytest.skip("phase 3: exclusion")
    del event  # Events are phase 2; controls() skips before this is ever non-None
    return composite_observations_about_features(
        _lows(mslp),
        points,
        pairing=DistancePairing(DISTANCE_KM, shape="circle"),
        bins=BinSpec.grid(MAX_KM_EXTENT, KM_SPACING),
        controls=[ControlDesign(f"obs{shift}", observation_offset=shift) for shift in CONTROL_SHIFTS],
        return_pairs=False,
    )


def controls(
    mslp: xr.DataArray, points: dd.DataFrame, *, exclusion_distance_km: float | None, event: str | None = None
) -> dict[str, int]:
    """Totals over every chunk, control and bin.

    Returns:
        ``n_centres``, ``n_controls`` and ``control_n_obs``. Events and domain counts are phase 2, gated by one
        ``pytest.skip`` here so every test that asks for them is skipped without gating the phase 1 control counts.
    """
    if event is not None:
        pytest.skip("phase 2: events and domain counts")
    computed = build_controls(mslp, points, exclusion_distance_km=exclusion_distance_km, event=None).compute(
        scheduler="sync"
    )
    return {
        "n_centres": int(computed["n_centres"].sum()),
        "n_controls": int(computed["n_controls"].sum()),
        "control_n_obs": round(float(computed["control_n_obs"].sum())),
    }


# ======================================================================================================================
# 4. Tests
# ======================================================================================================================


def _with_longitudes_in_0_360(points: pd.DataFrame) -> pd.DataFrame:
    """The same observations with their longitudes re-expressed in [0, 360)."""
    return points.assign(longitude=points["longitude"].to_numpy(dtype=float) % 360)


@pytest.fixture(scope="module")
def composites(mslp: xr.DataArray, points_dd: dd.DataFrame) -> dict[Pairing, CompositeResult]:
    """One computed composite per pairing, shared by the centre, pair and histogram tests."""
    return {pairing: composite(mslp, points_dd, pairing) for pairing in Pairing}


@pytest.fixture(scope="module")
def expected_pairs(
    mslp: xr.DataArray, points: pd.DataFrame, extrema: xr.Dataset, expected_centres: pd.DataFrame
) -> dict[Pairing, pd.DataFrame]:
    return {pairing: _brute_force_pairs(mslp, points, extrema, expected_centres, pairing) for pairing in Pairing}


@pytest.fixture(scope="module", params=[None, EXCLUSION_DISTANCE_KM], ids=["no-exclusion", "exclusion"])
def exclusion_distance_km(request) -> float | None:
    """Each case is its own fixture instance, so that a later phase gating one case does not skip the other."""
    return request.param


@pytest.fixture(scope="module")
def control_counts(mslp: xr.DataArray, points_dd: dd.DataFrame, exclusion_distance_km: float | None) -> dict[str, int]:
    """Count-only control run."""
    return controls(mslp, points_dd, exclusion_distance_km=exclusion_distance_km)


@pytest.fixture(scope="module")
def control_counts_with_events(mslp: xr.DataArray, points_dd: dd.DataFrame) -> dict[str, int]:
    # Without the exclusion (a later phase than events)
    return controls(mslp, points_dd, exclusion_distance_km=None, event=EVENT)


@pytest.fixture(scope="module")
def domain_points(mslp: xr.DataArray, points: pd.DataFrame) -> pd.DataFrame:
    """The observations on which the intended domain (plan D15) and the prototype's domain agree.

    D15: within the outermost grid points and within half a time step of a field time. The prototype instead uses times
    in [first, last] field time and accepts half a grid cell beyond the outermost points. Drop the observations where
    they differ: within half a step before the first or after the last time, and in the half-cell margin beyond the
    outermost points. Phase 2 adds fail-first edge tests for the time half-step, the spatial margin and a global
    (periodic) field.
    """
    field_times = _field_times(mslp)
    observation_times = points["time"].to_numpy().astype("datetime64[ns]")
    is_within_period = (observation_times >= field_times.min()) & (observation_times <= field_times.max())
    is_in_margin = np.zeros(len(points), dtype=bool)
    for dim_name in ("latitude", "longitude"):
        coordinate = mslp[dim_name].to_numpy()
        values = points[dim_name].to_numpy(dtype=float)
        is_in_margin |= _is_within_cells(coordinate, values) & ~_is_within_outermost_points(coordinate, values)
    kept = points[is_within_period & ~is_in_margin]
    assert np.any(~is_within_period)
    assert np.any(is_in_margin)
    return kept


@pytest.fixture(scope="module")
def domain_counts(mslp: xr.DataArray, domain_points: pd.DataFrame) -> dict[str, int]:
    return controls(
        mslp,
        dd.from_pandas(domain_points, npartitions=N_PARTITIONS),
        # The domain does not depend on the controls, so leave out the exclusion (a later phase than domain counts)
        exclusion_distance_km=None,
        event=EVENT,
    )


class TestCentres:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_centres_are_connected_components_of_local_extrema(
        self, composites: dict[Pairing, CompositeResult], expected_centres: pd.DataFrame, pairing: Pairing
    ) -> None:
        computed = composites[pairing].centres
        assert len(computed) == len(expected_centres)
        assert _centre_keys(computed) == _centre_keys(expected_centres)

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_n_centres_is_number_of_centres(
        self, composites: dict[Pairing, CompositeResult], expected_centres: pd.DataFrame, pairing: Pairing
    ) -> None:
        assert composites[pairing].n_centres == len(expected_centres)


class TestPairs:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_pairs_match_brute_force(
        self,
        composites: dict[Pairing, CompositeResult],
        expected_pairs: dict[Pairing, pd.DataFrame],
        pairing: Pairing,
    ) -> None:
        expected = _pair_keys(expected_pairs[pairing])
        assert len(expected) > 0
        assert _pair_keys(composites[pairing].pairs) == expected

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_no_duplicate_pairs(self, composites: dict[Pairing, CompositeResult], pairing: Pairing) -> None:
        pair_frame = composites[pairing].pairs
        assert len(_pair_keys(pair_frame)) == len(pair_frame)

    def test_region_pairs_each_observation_at_most_once(self, composites: dict[Pairing, CompositeResult]) -> None:
        assert composites[Pairing.REGION].pairs["id"].is_unique

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_positions_match_brute_force_projection(
        self,
        composites: dict[Pairing, CompositeResult],
        expected_pairs: dict[Pairing, pd.DataFrame],
        pairing: Pairing,
    ) -> None:
        key_columns = ["id", "centre_time", "centre_lat", "centre_lon"]
        computed = composites[pairing].pairs.sort_values(key_columns).reset_index(drop=True)
        expected = expected_pairs[pairing].sort_values(key_columns).reset_index(drop=True)
        np.testing.assert_allclose(computed["x_km"].to_numpy(), expected["x_km"].to_numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(computed["y_km"].to_numpy(), expected["y_km"].to_numpy(), rtol=0, atol=1e-6)


class TestHistogram:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_counts_match_histogram2d_of_brute_force_positions(
        self,
        composites: dict[Pairing, CompositeResult],
        expected_pairs: dict[Pairing, pd.DataFrame],
        pairing: Pairing,
    ) -> None:
        counts = composites[pairing].counts
        edges = _histogram_edges()
        expected, _, _ = np.histogram2d(
            expected_pairs[pairing]["y_km"].to_numpy(), expected_pairs[pairing]["x_km"].to_numpy(), bins=(edges, edges)
        )
        assert counts.shape == expected.shape
        assert expected.sum() > 0
        np.testing.assert_array_equal(counts, expected)


class TestGraph:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_building_graph_only_computes_in_group_by_chunk(
        self, mslp: xr.DataArray, points_dd: dd.DataFrame, pairing: Pairing, mocker: "MockerFixture"
    ) -> None:
        group_by_chunk = mocker.spy(*GRAPH_BUILD_COMPUTE)
        with _count_computes() as computes:
            build(mslp, points_dd, pairing)
        assert group_by_chunk.call_count >= 1
        assert len(computes) == group_by_chunk.call_count

    def test_building_control_graph_only_computes_in_group_by_chunk(
        self,
        mslp: xr.DataArray,
        points_dd: dd.DataFrame,
        exclusion_distance_km: float | None,
        mocker: "MockerFixture",
    ) -> None:
        # Without events (phase 2), and per exclusion case (phase 3), so that phase 1 still checks the control graph
        group_by_chunk = mocker.spy(*GRAPH_BUILD_COMPUTE)
        with _count_computes() as computes:
            build_controls(mslp, points_dd, exclusion_distance_km=exclusion_distance_km, event=None)
        assert group_by_chunk.call_count >= 1
        assert len(computes) == group_by_chunk.call_count

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_chunk_task_and_extrema_filter_run_once_per_chunk_and_time_step(
        self, mslp: xr.DataArray, points_dd: dd.DataFrame, pairing: Pairing, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunk_calls: list[int] = []
        filter_calls: list[int] = []
        monkeypatch.setattr(CHUNK_TASK, _counting(resolve_name(CHUNK_TASK), chunk_calls))
        monkeypatch.setattr(EXTREMA_FILTER, _counting(resolve_name(EXTREMA_FILTER), filter_calls))
        lazy = build(mslp, points_dd, pairing)
        # Building may evaluate the functions on empty inputs (e.g. for the meta), which is not a run of a task
        chunk_calls.clear()
        filter_calls.clear()

        compute_together(lazy)

        assert len(chunk_calls) == len(mslp.chunksizes["time"])
        assert len(filter_calls) == mslp.sizes["time"]


class TestControls:
    def test_valid_controls_match_brute_force(
        self,
        mslp: xr.DataArray,
        expected_centres: pd.DataFrame,
        control_counts: dict[str, int],
        exclusion_distance_km: float | None,
    ) -> None:
        expected = _brute_force_valid_controls(expected_centres, _field_times(mslp), exclusion_distance_km)
        assert control_counts["n_controls"] == len(expected)

    def test_exclusion_discards_controls_near_another_centre(self, mslp: xr.DataArray, points_dd: dd.DataFrame) -> None:
        # Needs both cases at once, so the whole test belongs to the phase of the exclusion
        with_exclusion = controls(mslp, points_dd, exclusion_distance_km=EXCLUSION_DISTANCE_KM)
        without_exclusion = controls(mslp, points_dd, exclusion_distance_km=None)
        assert with_exclusion["n_controls"] < without_exclusion["n_controls"]

    def test_n_centres_is_number_of_centres(
        self, expected_centres: pd.DataFrame, control_counts: dict[str, int]
    ) -> None:
        assert control_counts["n_centres"] == len(expected_centres)

    def test_control_observation_counts_match_brute_force(
        self,
        mslp: xr.DataArray,
        points: pd.DataFrame,
        expected_centres: pd.DataFrame,
        control_counts: dict[str, int],
        exclusion_distance_km: float | None,
    ) -> None:
        field_times = _field_times(mslp)
        valid = _brute_force_valid_controls(expected_centres, field_times, exclusion_distance_km)
        expected_n_obs, _ = _brute_force_control_counts(points, expected_centres, field_times, valid)
        assert expected_n_obs > 0
        assert control_counts["control_n_obs"] == expected_n_obs

    def test_control_observation_and_event_counts_match_brute_force(
        self,
        mslp: xr.DataArray,
        points: pd.DataFrame,
        expected_centres: pd.DataFrame,
        control_counts_with_events: dict[str, int],
    ) -> None:
        field_times = _field_times(mslp)
        valid = _brute_force_valid_controls(expected_centres, field_times, None)
        expected_n_obs, expected_n_event = _brute_force_control_counts(points, expected_centres, field_times, valid)
        assert 0 < expected_n_event < expected_n_obs
        assert control_counts_with_events["control_n_obs"] == expected_n_obs
        assert control_counts_with_events["control_n_event"] == expected_n_event


class TestDomain:
    def test_domain_observation_count_and_event_rate_match_brute_force(
        self, mslp: xr.DataArray, domain_points: pd.DataFrame, domain_counts: dict[str, int]
    ) -> None:
        # domain_points leaves out the edges where the prototype differs from plan D15, see its docstring
        expected_n_obs, expected_n_event = _brute_force_domain_counts(mslp, domain_points)

        assert 0 < expected_n_event < expected_n_obs < len(domain_points)
        assert domain_counts["domain_n_obs"] == expected_n_obs
        assert domain_counts["domain_n_event"] == expected_n_event
        assert domain_counts["domain_n_event"] / domain_counts["domain_n_obs"] == pytest.approx(
            expected_n_event / expected_n_obs
        )


class TestLongitudeConvention:
    """Observation longitudes in [0, 360) give the same results as in [-180, 180] (the field's convention).

    Only the observations are re-expressed: the synthetic field in [0, 360) would cross the prime meridian.
    """

    def test_region_pairs_do_not_depend_on_observation_longitude_convention(
        self, mslp: xr.DataArray, points: pd.DataFrame, composites: dict[Pairing, CompositeResult]
    ) -> None:
        # Circle and square pairing only use distances, so the region (grid lookup) case is the one that converts
        shifted = composite(
            mslp,
            dd.from_pandas(_with_longitudes_in_0_360(points), npartitions=N_PARTITIONS),
            Pairing.REGION,
        )
        reference = composites[Pairing.REGION]
        # Some observations moved beyond 180 are paired, and some are on the grid but not paired
        is_moved_on_grid = (points["longitude"] < 0) & (points["longitude"] >= mslp["longitude"].min().item())
        is_paired = points["id"].isin(reference.pairs["id"])
        assert np.any(is_moved_on_grid & is_paired)
        assert np.any(is_moved_on_grid & ~is_paired)
        assert len(shifted.pairs) == len(reference.pairs)
        assert _pair_keys(shifted.pairs) == _pair_keys(reference.pairs)
        assert shifted.n_centres == reference.n_centres
        np.testing.assert_array_equal(shifted.counts, reference.counts)

    def test_control_counts_do_not_depend_on_observation_longitude_convention(
        self,
        mslp: xr.DataArray,
        points: pd.DataFrame,
        control_counts: dict[str, int],
        exclusion_distance_km: float | None,
    ) -> None:
        shifted = controls(
            mslp,
            dd.from_pandas(_with_longitudes_in_0_360(points), npartitions=N_PARTITIONS),
            exclusion_distance_km=exclusion_distance_km,
        )
        assert shifted == control_counts

    def test_domain_counts_do_not_depend_on_observation_longitude_convention(
        self, mslp: xr.DataArray, domain_points: pd.DataFrame, domain_counts: dict[str, int]
    ) -> None:
        shifted = controls(
            mslp,
            dd.from_pandas(_with_longitudes_in_0_360(domain_points), npartitions=N_PARTITIONS),
            exclusion_distance_km=None,
            event=EVENT,
        )
        assert shifted["domain_n_obs"] == domain_counts["domain_n_obs"]
        assert shifted["domain_n_event"] == domain_counts["domain_n_event"]
