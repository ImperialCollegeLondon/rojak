"""Tests for the pairing strategies of :mod:`rojak.atmosphere.feature_composite` (``_pairing``, phase 1, step D).

Pairs are checked against brute force written independently of the implementation:

- **Region:** for each observation, the label at its nearest grid cell (``argmin`` over the coordinates) at its local
  time step, after converting its longitude to the grid's convention; observations more than half a cell beyond the
  outermost grid points are unpaired.
- **Circle:** every centre at the same time step with a haversine distance of at most ``DISTANCE_KM``.
- **Square:** every centre at the same time step with ``|x|, |y| <= DISTANCE_KM`` in a fresh WGS-84 orthographic
  projection about the centre (:mod:`pyproj`).

Positions are checked against a per-centre :mod:`pyproj` projection. Observations carry the chunk-absolute
``Column.TIME_INDEX`` of the field time step they were matched to, which the pipeline adds later.

Layout:

1. Synthetic data: hand-built centres and labels on two grids (one either side of the prime meridian, one either side
   of the dateline in ``[0, 360)``), and centres and labels from ``ExtremaPointFeature``'s tabulator.
2. Brute force and helpers.
3. Tests.
"""

import dataclasses
import pickle
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr
from dask.base import tokenize

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._feature import (
    ChunkInfo,
    FeatureChunk,
    FeatureGeometry,
    FeatureSource,
    FeatureTabulator,
)
from rojak.atmosphere.feature_composite._pairing import DistancePairing, LabelledRegionPairing, PairingStrategy
from rojak.atmosphere.feature_composite._point_features import ExtremaPointFeature
from rojak.atmosphere.regions import ExtremaKind
from rojak.core.geometric import haversine_distance

type Shape = Literal["circle", "square"]

LAT: str = "latitude"
LON: str = "longitude"
OBS_ID: str = "obs_id"
EVENT: str = "is_event"
DISTANCE_KM: float = 400.0
TOLERANCE_KM: float = 1e-6
N_OBSERVATIONS: int = 400
FIRST_TIME_INDEX: int = 5
N_TIMES: int = 3
LABEL_RADIUS: int = 6
MAX_PICKLED_PAIRING_BYTES: int = 1_000

# The centre's columns that every pair table carries (section 3.2: the same columns for every strategy)
PAIR_CENTRE_COLUMNS: frozenset[str] = frozenset(
    {
        Column.CENTRE_ID,
        Column.FEATURE_ID,
        Column.CENTRE_TIME,
        Column.CENTRE_LAT,
        Column.CENTRE_LON,
        Column.CENTRE_VALUE,
    }
)
POSITION_COLUMNS: frozenset[str] = frozenset({Column.X_KM, Column.Y_KM})

LATS: np.ndarray = np.arange(60.0, 29.5, -1.0)  # Descending, as in ERA5
GRID_LONS: dict[str, np.ndarray] = {
    "prime-meridian": np.arange(-30.0, 30.5, 1.0),  # [-180, 180) convention
    "dateline": np.arange(150.0, 210.5, 1.0),  # [0, 360) convention, crossing 180
}
# Small 3 x 3 grids for hand-checked cases, one per longitude convention
SMALL_LATS: np.ndarray = np.array([50.0, 49.0, 48.0])
SMALL_LONS: dict[str, np.ndarray] = {
    "prime-meridian": np.array([-1.0, 0.0, 1.0]),
    "dateline": np.array([179.0, 180.0, 181.0]),
}


class Pairing(StrEnum):
    """Test-local ids of the strategies, so the brute force can branch on them."""

    REGION = "region"
    CIRCLE = "circle"
    SQUARE = "square"


def _strategy(pairing: Pairing) -> PairingStrategy:
    match pairing:
        case Pairing.REGION:
            return LabelledRegionPairing()
        case Pairing.CIRCLE:
            return DistancePairing(DISTANCE_KM)
        case Pairing.SQUARE:
            return DistancePairing(DISTANCE_KM, shape="square")


class Scenario(NamedTuple):
    chunk: ChunkInfo
    feature_chunk: FeatureChunk
    observations: pd.DataFrame


# ======================================================================================================================
# 1. Synthetic data
# ======================================================================================================================


def _wrap_to_180(lons: np.ndarray) -> np.ndarray:
    return (lons + 180.0) % 360.0 - 180.0


def _wrap_to_360(lons: np.ndarray) -> np.ndarray:
    return lons % 360.0


LONGITUDE_CONVENTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "[-180, 180)": _wrap_to_180,
    "[0, 360)": _wrap_to_360,
}


def _times(n_times: int) -> np.ndarray:
    return pd.date_range("2024-01-02", periods=n_times, freq="6h").to_numpy().astype("datetime64[ns]")


def _chunk(lats: np.ndarray, lons: np.ndarray, *, n_times: int, first_time_index: int) -> ChunkInfo:
    return ChunkInfo(times=_times(n_times), first_time_index=first_time_index, lats=lats, lons=lons)


def _centres_table(chunk: ChunkInfo, rows: list[tuple[int, int, float, float]]) -> pd.DataFrame:
    """Centres from ``(centre_id, time_index, lat, lon)``, filled in as a tabulator would."""
    centre_ids = np.array([row[0] for row in rows], dtype=np.int64)
    time_index = np.array([row[1] for row in rows], dtype=np.int64)
    lats = np.array([row[2] for row in rows], dtype=float)
    lons = np.array([row[3] for row in rows], dtype=float)
    lat_index = np.array([int(np.argmin(np.abs(chunk.lats - lat))) for lat in lats.tolist()], dtype=np.int64)
    lon_index = np.array([int(np.argmin(np.abs(chunk.lons - lon))) for lon in lons.tolist()], dtype=np.int64)
    return pd.DataFrame(
        {
            Column.CENTRE_ID: centre_ids,
            Column.FEATURE_ID: centre_ids.copy(),
            Column.CENTRE_TIME: chunk.times[time_index - chunk.first_time_index],
            Column.CENTRE_LAT: lats,
            Column.CENTRE_LON: lons,
            Column.CENTRE_VALUE: 99_000.0 + 10.0 * np.arange(centre_ids.size, dtype=float),
            Column.CENTRE_LAT_INDEX: lat_index,
            Column.CENTRE_LON_INDEX: lon_index,
            Column.TIME_INDEX: time_index,
        }
    )


def _observations_table(rows: list[tuple[int, int, float, float]]) -> pd.DataFrame:
    """Observations from ``(obs_id, time_index, lat, lon)``."""
    obs_ids = np.array([row[0] for row in rows], dtype=np.int64)
    return pd.DataFrame(
        {
            OBS_ID: obs_ids,
            Column.TIME_INDEX: np.array([row[1] for row in rows], dtype=np.int64),
            LAT: np.array([row[2] for row in rows], dtype=float),
            LON: np.array([row[3] for row in rows], dtype=float),
            EVENT: obs_ids % 3 == 0,
        }
    )


def _random_observations(chunk: ChunkInfo, seed: int) -> pd.DataFrame:
    """Uniform observations over the grid and 3 degrees beyond it, at the chunk's time steps.

    Longitudes are in ``[-180, 180)`` whatever the grid's convention.
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(chunk.lats.min() - 3.0, chunk.lats.max() + 3.0, N_OBSERVATIONS)
    lons = _wrap_to_180(rng.uniform(chunk.lons.min() - 3.0, chunk.lons.max() + 3.0, N_OBSERVATIONS))
    time_index = chunk.first_time_index + rng.integers(0, chunk.times.size, N_OBSERVATIONS)
    rows = [
        (obs_id, int(time), float(lat), float(lon))
        for obs_id, (time, lat, lon) in enumerate(zip(time_index.tolist(), lats.tolist(), lons.tolist(), strict=True))
    ]
    return _observations_table(rows)


def _stamp_labels(chunk: ChunkInfo, centres: pd.DataFrame) -> np.ndarray:
    """Each centre owns the unowned cells within ``LABEL_RADIUS`` cells of it, in the order of the table."""
    labels = np.zeros((chunk.times.size, chunk.lats.size, chunk.lons.size), dtype=np.int64)
    lat_grid, lon_grid = np.meshgrid(np.arange(chunk.lats.size), np.arange(chunk.lons.size), indexing="ij")
    for centre_id, time_index, lat_index, lon_index in zip(
        centres[Column.CENTRE_ID].tolist(),
        centres[Column.TIME_INDEX].tolist(),
        centres[Column.CENTRE_LAT_INDEX].tolist(),
        centres[Column.CENTRE_LON_INDEX].tolist(),
        strict=True,
    ):
        layer = labels[time_index - chunk.first_time_index]
        within = (lat_grid - lat_index) ** 2 + (lon_grid - lon_index) ** 2 <= LABEL_RADIUS**2
        layer[within & (layer == 0)] = centre_id
    return labels


def _hand_built_scenario(lons: np.ndarray) -> Scenario:
    """Three centres per time step on a 1 degree grid; two of them are 2 degrees apart, so their labels touch and an
    observation can be within ``DISTANCE_KM`` of both."""
    chunk = _chunk(LATS, lons, n_times=N_TIMES, first_time_index=FIRST_TIME_INDEX)
    lon_mid = float(lons[lons.size // 2])
    rows: list[tuple[int, int, float, float]] = []
    for local_time in range(N_TIMES):
        time_index = FIRST_TIME_INDEX + local_time
        positions = (
            (52.0 + local_time, lon_mid - 15.0 + 3.0 * local_time),
            (40.0, lon_mid + 5.0 + local_time),
            (41.0, lon_mid + 7.0 + local_time),
        )
        rows.extend(
            (100 * time_index + number + 1, time_index, lat, lon) for number, (lat, lon) in enumerate(positions)
        )
    centres = _centres_table(chunk, rows)
    feature_chunk = FeatureChunk(centres=centres, labels=_stamp_labels(chunk, centres))
    return Scenario(chunk=chunk, feature_chunk=feature_chunk, observations=_random_observations(chunk, seed=42))


def _gaussian_lows() -> xr.DataArray:
    """Three Gaussian lows moving on a coarse grid (as in the ``_point_features`` tests)."""
    times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lats = np.arange(70, 26, -1.5)
    lons = np.arange(-40, 40, 2.0)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    fields = []
    for step in range(times.size):
        field = 101500 + 300 * np.sin(np.radians(lat_grid)) + 0.5 * lon_grid
        for centre_lat, centre_lon, speed, depth in (
            (59.3, -24.6, 2.1, 3000),
            (45.2, 0.7, 3.2, 2500),
            (54.6, 19.3, -0.5, 2000),
        ):
            squared_distance = (lat_grid - centre_lat) ** 2 + (lon_grid - (centre_lon + speed * step)) ** 2
            field = field - depth * np.exp(-squared_distance / (2 * 5**2))
        fields.append(field)
    return xr.DataArray(
        np.stack(fields),
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": lats, "longitude": lons},
        name="mslp",
    ).chunk({"time": 2})


def _tabulated_scenario() -> Scenario:
    """Centres and labels from ``ExtremaPointFeature``'s tabulator on the second time chunk (time indices 2 and 3)."""
    field = _gaussian_lows()
    feature = ExtremaPointFeature(field, ExtremaKind.MINIMA, threshold=100_000, footprint_radius=4)
    first, stop = 2, 4
    blocks = {name: array.isel(time=slice(first, stop)).to_numpy() for name, array in feature.chunk_inputs().items()}
    chunk = ChunkInfo(
        times=field["time"].to_numpy()[first:stop].astype("datetime64[ns]"),
        first_time_index=first,
        lats=field["latitude"].to_numpy(),
        lons=field["longitude"].to_numpy(),
    )
    feature_chunk = feature.tabulator()(blocks, chunk)
    assert feature_chunk.labels is not None
    assert len(feature_chunk.centres) > 0
    return Scenario(chunk=chunk, feature_chunk=feature_chunk, observations=_random_observations(chunk, seed=7))


SCENARIOS: tuple[str, ...] = (*GRID_LONS, "tabulated")


@pytest.fixture(scope="module", params=SCENARIOS)
def scenario(request) -> Scenario:
    name: str = request.param
    if name == "tabulated":
        return _tabulated_scenario()
    return _hand_built_scenario(GRID_LONS[name])


def _pair(strategy: PairingStrategy, scenario: Scenario, observations: pd.DataFrame | None = None) -> pd.DataFrame:
    return strategy.pair(
        scenario.observations if observations is None else observations,
        scenario.feature_chunk,
        scenario.chunk,
        lat_column=LAT,
        lon_column=LON,
    )


# ======================================================================================================================
# 2. Brute force and helpers
# ======================================================================================================================


def _to_grid_convention(lons: np.ndarray, grid_lons: np.ndarray) -> np.ndarray:
    return _wrap_to_360(lons) if grid_lons.max() > 180 else _wrap_to_180(lons)  # noqa: PLR2004


def _ortho(lat: float, lon: float) -> pyproj.Proj:
    return pyproj.Proj(proj="ortho", lat_0=lat, lon_0=lon, units="km", ellps="WGS84")


def _region_keys(scenario: Scenario) -> set[tuple[int, int]]:
    chunk = scenario.chunk
    labels = scenario.feature_chunk.labels
    assert labels is not None
    observations = scenario.observations
    obs_lats = observations[LAT].to_numpy(dtype=float)
    obs_lons = _to_grid_convention(observations[LON].to_numpy(dtype=float), chunk.lons)
    half_lat = np.abs(np.diff(chunk.lats)).min() / 2
    half_lon = np.abs(np.diff(chunk.lons)).min() / 2
    keys: set[tuple[int, int]] = set()
    for obs_id, time_index, lat, lon in zip(
        observations[OBS_ID].tolist(),
        observations[Column.TIME_INDEX].tolist(),
        obs_lats.tolist(),
        obs_lons.tolist(),
        strict=True,
    ):
        is_on_grid = (chunk.lats.min() - half_lat <= lat <= chunk.lats.max() + half_lat) and (
            chunk.lons.min() - half_lon <= lon <= chunk.lons.max() + half_lon
        )
        if not is_on_grid:
            continue
        lat_index = int(np.argmin(np.abs(chunk.lats - lat)))
        lon_index = int(np.argmin(np.abs(chunk.lons - lon)))
        label = int(labels[time_index - chunk.first_time_index, lat_index, lon_index])
        if label != 0:
            keys.add((int(obs_id), label))
    return keys


def _distance_keys(scenario: Scenario, *, square: bool) -> set[tuple[int, int]]:
    observations = scenario.observations
    centres = scenario.feature_chunk.centres
    obs_ids = observations[OBS_ID].to_numpy()
    obs_times = observations[Column.TIME_INDEX].to_numpy()
    obs_lats = observations[LAT].to_numpy(dtype=float)
    obs_lons = observations[LON].to_numpy(dtype=float)
    keys: set[tuple[int, int]] = set()
    for centre_id, time_index, lat, lon in zip(
        centres[Column.CENTRE_ID].tolist(),
        centres[Column.TIME_INDEX].tolist(),
        centres[Column.CENTRE_LAT].tolist(),
        centres[Column.CENTRE_LON].tolist(),
        strict=True,
    ):
        at_time = obs_times == time_index
        if square:
            x_km, y_km = _ortho(lat, lon)(obs_lons[at_time], obs_lats[at_time])
            within = (np.abs(np.asarray(x_km)) <= DISTANCE_KM) & (np.abs(np.asarray(y_km)) <= DISTANCE_KM)
        else:
            n_at_time = int(at_time.sum())
            distance_km = (
                haversine_distance(
                    obs_lons[at_time], obs_lats[at_time], np.full(n_at_time, lon), np.full(n_at_time, lat)
                )
                / 1000
            )
            within = distance_km <= DISTANCE_KM
        keys.update((int(obs_id), int(centre_id)) for obs_id in obs_ids[at_time][within].tolist())
    return keys


def _brute_force_keys(scenario: Scenario, pairing: Pairing) -> set[tuple[int, int]]:
    match pairing:
        case Pairing.REGION:
            return _region_keys(scenario)
        case Pairing.CIRCLE:
            return _distance_keys(scenario, square=False)
        case Pairing.SQUARE:
            return _distance_keys(scenario, square=True)


def _brute_force_positions(scenario: Scenario, keys: set[tuple[int, int]]) -> pd.DataFrame:
    """``(obs_id, centre_id, x_km, y_km)`` of each pair, from a fresh projection per centre."""
    observations = scenario.observations
    centres = scenario.feature_chunk.centres
    obs_lat = dict(zip(observations[OBS_ID].tolist(), observations[LAT].tolist(), strict=True))
    obs_lon = dict(zip(observations[OBS_ID].tolist(), observations[LON].tolist(), strict=True))
    centre_lat = dict(zip(centres[Column.CENTRE_ID].tolist(), centres[Column.CENTRE_LAT].tolist(), strict=True))
    centre_lon = dict(zip(centres[Column.CENTRE_ID].tolist(), centres[Column.CENTRE_LON].tolist(), strict=True))
    projections: dict[int, pyproj.Proj] = {}
    rows: list[tuple[int, int, float, float]] = []
    for obs_id, centre_id in sorted(keys):
        if centre_id not in projections:
            projections[centre_id] = _ortho(centre_lat[centre_id], centre_lon[centre_id])
        x_km, y_km = projections[centre_id](obs_lon[obs_id], obs_lat[obs_id])
        rows.append((obs_id, centre_id, float(x_km), float(y_km)))
    return pd.DataFrame(rows, columns=[OBS_ID, str(Column.CENTRE_ID), str(Column.X_KM), str(Column.Y_KM)])


def _keys(pairs: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(obs_id), int(centre_id))
        for obs_id, centre_id in zip(pairs[OBS_ID].tolist(), pairs[Column.CENTRE_ID].tolist(), strict=True)
    }


def _sorted_by_key(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.sort_values([OBS_ID, str(Column.CENTRE_ID)]).reset_index(drop=True)


def _expected_columns(observations: pd.DataFrame) -> set[str]:
    return set(observations.columns) | PAIR_CENTRE_COLUMNS | POSITION_COLUMNS


# --------------------------------------------------------------------------------------------------------------------
# Stub features for check_compatible (module level so that they pickle). Minimal frozen dataclasses, as an extender
# would write them, with geometry as a property.
# --------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _StubTabulator(FeatureTabulator):
    def __call__(self, blocks: Mapping[str, np.ndarray], chunk: ChunkInfo) -> FeatureChunk:
        return FeatureChunk(centres=pd.DataFrame(), labels=None)


@dataclasses.dataclass(frozen=True)
class StubPointFeature(FeatureSource):
    """A point feature without labels (``has_labels`` left at its default)."""

    @property
    def geometry(self) -> FeatureGeometry:
        return FeatureGeometry.POINT

    def chunk_inputs(self) -> Mapping[str, xr.DataArray]:
        return {}

    def tabulator(self) -> FeatureTabulator:
        return _StubTabulator()


@dataclasses.dataclass(frozen=True)
class StubLabelledPointFeature(StubPointFeature):
    @property
    def has_labels(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class StubLineFeature(StubPointFeature):
    @property
    def geometry(self) -> FeatureGeometry:
        return FeatureGeometry.LINE


@dataclasses.dataclass(frozen=True)
class StubLabelledLineFeature(StubLineFeature):
    @property
    def has_labels(self) -> bool:
        return True


def _extrema_point_feature() -> ExtremaPointFeature:
    field = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("time", "latitude", "longitude"),
        coords={"time": _times(2), "latitude": SMALL_LATS, "longitude": np.arange(4.0)},
    )
    return ExtremaPointFeature(field, ExtremaKind.MINIMA)


def _pair_nothing(self, observations, feature_chunk, chunk, *, lat_column, lon_column):
    return observations.iloc[:0]


def _pairing_subclass(namespace: dict[str, object]) -> type:
    return type("PairingSubclass", (PairingStrategy,), namespace)


# ======================================================================================================================
# 3. Tests
# ======================================================================================================================


class TestPairsMatchBruteForce:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_pairs_match_brute_force(self, scenario: Scenario, pairing: Pairing) -> None:
        expected = _brute_force_keys(scenario, pairing)

        result = _pair(_strategy(pairing), scenario)

        assert len(expected) > 0
        assert _keys(result) == expected
        # One row per pair: no duplicates
        assert len(result) == len(expected)

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_positions_match_pyproj(self, scenario: Scenario, pairing: Pairing) -> None:
        expected = _sorted_by_key(_brute_force_positions(scenario, _brute_force_keys(scenario, pairing)))

        result = _sorted_by_key(_pair(_strategy(pairing), scenario))

        assert result[[OBS_ID, str(Column.CENTRE_ID)]].to_numpy().tolist() == (
            expected[[OBS_ID, str(Column.CENTRE_ID)]].to_numpy().tolist()
        )
        assert np.allclose(
            result[Column.X_KM].to_numpy(dtype=float),
            expected[Column.X_KM].to_numpy(dtype=float),
            rtol=0,
            atol=TOLERANCE_KM,
        )
        assert np.allclose(
            result[Column.Y_KM].to_numpy(dtype=float),
            expected[Column.Y_KM].to_numpy(dtype=float),
            rtol=0,
            atol=TOLERANCE_KM,
        )

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_columns_are_observation_centre_and_position_columns(self, scenario: Scenario, pairing: Pairing) -> None:
        result = _pair(_strategy(pairing), scenario)

        # Exactly these, so every strategy returns the same columns (no merge suffixes, no strategy-specific extras)
        assert set(result.columns) == _expected_columns(scenario.observations)
        assert len(result.columns) == len(set(result.columns))

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_centre_columns_are_those_of_the_paired_centre(self, scenario: Scenario, pairing: Pairing) -> None:
        result = _pair(_strategy(pairing), scenario)
        centres = scenario.feature_chunk.centres.set_index(Column.CENTRE_ID)
        paired_centres = centres.loc[result[Column.CENTRE_ID].to_numpy()]

        assert len(result) > 0
        for column in (Column.FEATURE_ID, Column.CENTRE_LAT, Column.CENTRE_LON, Column.CENTRE_VALUE):
            assert result[column].tolist() == paired_centres[column].tolist()
        assert (
            result[Column.CENTRE_TIME].to_numpy().astype("datetime64[ns]").tolist()
            == paired_centres[Column.CENTRE_TIME].to_numpy().astype("datetime64[ns]").tolist()
        )
        # A pair links an observation to a centre at the same time step
        assert result[Column.TIME_INDEX].tolist() == paired_centres[Column.TIME_INDEX].tolist()

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_observation_columns_are_carried_unchanged(self, scenario: Scenario, pairing: Pairing) -> None:
        result = _pair(_strategy(pairing), scenario)
        observations = scenario.observations.set_index(OBS_ID)
        source = observations.loc[result[OBS_ID].to_numpy()]

        assert len(result) > 0
        for column in observations.columns:
            assert result[column].tolist() == source[column].tolist()

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_does_not_modify_inputs(self, scenario: Scenario, pairing: Pairing) -> None:
        observations = scenario.observations.copy()
        centres = scenario.feature_chunk.centres.copy()
        labels = scenario.feature_chunk.labels
        assert labels is not None
        labels_copy = labels.copy()

        _pair(_strategy(pairing), scenario)

        pd.testing.assert_frame_equal(scenario.observations, observations)
        pd.testing.assert_frame_equal(scenario.feature_chunk.centres, centres)
        assert np.array_equal(labels, labels_copy)


class TestLabelledRegionPairing:
    def test_pairs_each_observation_at_most_once(self, scenario: Scenario) -> None:
        result = _pair(LabelledRegionPairing(), scenario)

        assert len(result) > 0
        assert result[OBS_ID].is_unique

    def test_uses_label_at_local_time_step(self) -> None:
        # Labels only at local time step 1, i.e. chunk-absolute time index 8
        chunk = _chunk(SMALL_LATS, SMALL_LONS["prime-meridian"], n_times=3, first_time_index=7)
        centres = _centres_table(chunk, [(42, 8, 49.0, 0.0)])
        labels = np.zeros((3, 3, 3), dtype=np.int64)
        labels[1] = 42
        observations = _observations_table([(0, 7, 49.2, 0.3), (1, 8, 49.2, 0.3), (2, 9, 49.2, 0.3)])

        result = LabelledRegionPairing().pair(
            observations, FeatureChunk(centres=centres, labels=labels), chunk, lat_column=LAT, lon_column=LON
        )

        assert _keys(result) == {(1, 42)}

    @pytest.mark.parametrize("observation_convention", list(LONGITUDE_CONVENTIONS))
    @pytest.mark.parametrize("grid", list(SMALL_LONS))
    def test_observations_beyond_half_a_cell_off_the_grid_are_not_paired(
        self, grid: str, observation_convention: str
    ) -> None:
        lons = SMALL_LONS[grid]
        lon_mid = float(lons[1])
        chunk = _chunk(SMALL_LATS, lons, n_times=1, first_time_index=3)
        centres = _centres_table(chunk, [(42, 3, 49.0, lon_mid)])
        # Every cell is labelled, so only the grid's extent decides; the grid spacing is 1 degree
        labels = np.full((1, 3, 3), 42, dtype=np.int64)
        offsets = [
            # Paired: on the grid or less than half a cell beyond its outermost points
            (0, 49.0, 0.0),
            (1, 50.4, 0.0),
            (2, 47.6, 0.0),
            (3, 49.0, 1.4),
            (4, 49.0, -1.4),
            # Not paired: more than half a cell beyond
            (5, 50.6, 0.0),
            (6, 47.4, 0.0),
            (7, 49.0, 1.6),
            (8, 49.0, -1.6),
            (9, 50.6, 1.6),
        ]
        to_convention = LONGITUDE_CONVENTIONS[observation_convention]
        observations = _observations_table(
            [(obs_id, 3, lat, float(to_convention(np.array([lon_mid + offset]))[0])) for obs_id, lat, offset in offsets]
        )

        result = LabelledRegionPairing().pair(
            observations, FeatureChunk(centres=centres, labels=labels), chunk, lat_column=LAT, lon_column=LON
        )

        assert _keys(result) == {(obs_id, 42) for obs_id in range(5)}

    def test_unlabelled_cells_are_not_paired(self) -> None:
        chunk = _chunk(SMALL_LATS, SMALL_LONS["prime-meridian"], n_times=1, first_time_index=0)
        centres = _centres_table(chunk, [(42, 0, 49.0, 0.0)])
        labels = np.zeros((1, 3, 3), dtype=np.int64)
        labels[0, 1, 1] = 42
        # Nearest cells (49, 0), (50, 0) and (49, 1); only the first is labelled
        observations = _observations_table([(0, 0, 49.1, 0.2), (1, 0, 49.8, 0.0), (2, 0, 49.0, 0.9)])

        result = LabelledRegionPairing().pair(
            observations, FeatureChunk(centres=centres, labels=labels), chunk, lat_column=LAT, lon_column=LON
        )

        assert _keys(result) == {(0, 42)}


class TestDistancePairing:
    @pytest.mark.parametrize("pairing", [Pairing.CIRCLE, Pairing.SQUARE])
    def test_closely_spaced_centres_both_pair_with_an_observation_between_them(self, pairing: Pairing) -> None:
        chunk = _chunk(np.array([46.0, 45.0, 44.0]), np.array([-1.0, 0.0, 1.0, 2.0]), n_times=1, first_time_index=0)
        centres = _centres_table(chunk, [(1, 0, 45.0, 0.0), (2, 0, 45.0, 1.0)])
        observations = _observations_table([(0, 0, 45.0, 0.5)])

        result = _strategy(pairing).pair(
            observations, FeatureChunk(centres=centres, labels=None), chunk, lat_column=LAT, lon_column=LON
        )

        assert _keys(result) == {(0, 1), (0, 2)}

    @pytest.mark.parametrize("pairing", [Pairing.CIRCLE, Pairing.SQUARE])
    @pytest.mark.parametrize("grid", list(GRID_LONS))
    def test_observations_pair_with_several_centres_in_scenario(self, grid: str, pairing: Pairing) -> None:
        # The hand-built scenario has two centres 2 degrees apart at every time step
        result = _pair(_strategy(pairing), _hand_built_scenario(GRID_LONS[grid]))

        assert result[OBS_ID].duplicated().any()

    @pytest.mark.parametrize(("pairing", "expected_obs_ids"), [(Pairing.CIRCLE, {1}), (Pairing.SQUARE, {0, 1})])
    def test_square_includes_corners_that_circle_excludes(self, pairing: Pairing, expected_obs_ids: set[int]) -> None:
        centre_lat, centre_lon = 45.0, 0.0
        ortho = _ortho(centre_lat, centre_lon)
        # Positions in the centre's km frame: a corner of the square (about 1.27 d away), on the y axis, and beyond d
        positions_km = [(0.9 * DISTANCE_KM, 0.9 * DISTANCE_KM), (0.0, 0.9 * DISTANCE_KM), (0.0, 1.1 * DISTANCE_KM)]
        rows: list[tuple[int, int, float, float]] = []
        for obs_id, (x_km, y_km) in enumerate(positions_km):
            lon, lat = ortho(x_km, y_km, inverse=True)
            rows.append((obs_id, 0, float(lat), float(lon)))
        chunk = _chunk(np.array([50.0, 45.0, 40.0]), np.array([-5.0, 0.0, 5.0]), n_times=1, first_time_index=0)
        centres = _centres_table(chunk, [(1, 0, centre_lat, centre_lon)])

        result = _strategy(pairing).pair(
            _observations_table(rows),
            FeatureChunk(centres=centres, labels=None),
            chunk,
            lat_column=LAT,
            lon_column=LON,
        )

        assert _keys(result) == {(obs_id, 1) for obs_id in expected_obs_ids}

    def test_default_shape_is_circle(self) -> None:
        assert DistancePairing(DISTANCE_KM).shape == "circle"

    def test_shape_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            DistancePairing(DISTANCE_KM, "square")  # pyright: ignore[reportCallIssue]

    @pytest.mark.parametrize("max_distance_km", [0.0, -1.0, -DISTANCE_KM])
    @pytest.mark.parametrize("shape", ["circle", "square"])
    def test_non_positive_distance_raises(self, max_distance_km: float, shape: Shape) -> None:
        with pytest.raises(ValueError, match="max_distance_km"):
            DistancePairing(max_distance_km, shape=shape)

    @pytest.mark.parametrize("shape", ["triangle", "Circle", ""])
    def test_unknown_shape_raises(self, shape: str) -> None:
        with pytest.raises(ValueError, match="shape"):
            DistancePairing(DISTANCE_KM, shape=shape)  # pyright: ignore[reportArgumentType]


class TestPairsOnlyAtTheSameTimeStep:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_observation_at_another_time_step_is_not_paired(self, pairing: Pairing) -> None:
        chunk = _chunk(SMALL_LATS, SMALL_LONS["prime-meridian"], n_times=2, first_time_index=4)
        centres = _centres_table(chunk, [(1, 4, 49.0, 0.0)])
        labels = np.zeros((2, 3, 3), dtype=np.int64)
        labels[0] = 1
        # Both observations sit on the centre; only the one at the centre's time step pairs
        observations = _observations_table([(0, 5, 49.0, 0.0), (1, 4, 49.0, 0.0)])

        result = _strategy(pairing).pair(
            observations, FeatureChunk(centres=centres, labels=labels), chunk, lat_column=LAT, lon_column=LON
        )

        assert _keys(result) == {(1, 1)}


class TestLongitudeConvention:
    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_observation_longitude_convention_does_not_change_pairs(self, scenario: Scenario, pairing: Pairing) -> None:
        observations = scenario.observations
        in_180 = observations.assign(**{LON: _wrap_to_180(observations[LON].to_numpy(dtype=float))})
        in_360 = observations.assign(**{LON: _wrap_to_360(observations[LON].to_numpy(dtype=float))})
        # Some observations really change: those west of the prime meridian
        assert np.any(in_180[LON].to_numpy() != in_360[LON].to_numpy())

        from_180 = _sorted_by_key(_pair(_strategy(pairing), scenario, in_180))
        from_360 = _sorted_by_key(_pair(_strategy(pairing), scenario, in_360))

        assert len(from_180) > 0
        assert _keys(from_180) == _keys(from_360)
        assert len(from_180) == len(from_360)
        assert np.allclose(
            from_180[Column.X_KM].to_numpy(dtype=float),
            from_360[Column.X_KM].to_numpy(dtype=float),
            rtol=0,
            atol=TOLERANCE_KM,
        )
        assert np.allclose(
            from_180[Column.Y_KM].to_numpy(dtype=float),
            from_360[Column.Y_KM].to_numpy(dtype=float),
            rtol=0,
            atol=TOLERANCE_KM,
        )


class TestEmpty:
    @pytest.fixture
    def hand_built(self) -> Scenario:
        return _hand_built_scenario(GRID_LONS["prime-meridian"])

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_empty_observations_give_empty_table_with_all_columns(self, hand_built: Scenario, pairing: Pairing) -> None:
        observations = hand_built.observations.iloc[:0]

        result = _pair(_strategy(pairing), hand_built, observations)

        assert len(result) == 0
        assert set(result.columns) == _expected_columns(observations)

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_no_centres_give_empty_table_with_all_columns(self, hand_built: Scenario, pairing: Pairing) -> None:
        labels = hand_built.feature_chunk.labels
        assert labels is not None
        no_centres = Scenario(
            chunk=hand_built.chunk,
            feature_chunk=FeatureChunk(centres=hand_built.feature_chunk.centres.iloc[:0], labels=np.zeros_like(labels)),
            observations=hand_built.observations,
        )

        result = _pair(_strategy(pairing), no_centres)

        assert len(result) == 0
        assert set(result.columns) == _expected_columns(hand_built.observations)

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_no_observations_at_the_centre_times_give_empty_table_with_all_columns(self, pairing: Pairing) -> None:
        chunk = _chunk(SMALL_LATS, SMALL_LONS["prime-meridian"], n_times=2, first_time_index=0)
        centres = _centres_table(chunk, [(1, 0, 49.0, 0.0)])
        labels = np.zeros((2, 3, 3), dtype=np.int64)
        labels[0] = 1
        observations = _observations_table([(0, 1, 49.0, 0.0), (1, 1, 48.5, 0.5)])

        result = _strategy(pairing).pair(
            observations, FeatureChunk(centres=centres, labels=labels), chunk, lat_column=LAT, lon_column=LON
        )

        assert len(result) == 0
        assert set(result.columns) == _expected_columns(observations)


class TestCheckCompatible:
    @pytest.mark.parametrize("feature_type", [StubPointFeature, StubLineFeature])
    def test_region_pairing_rejects_feature_without_labels(self, feature_type: type[StubPointFeature]) -> None:
        with pytest.raises(ValueError, match="DistancePairing"):
            LabelledRegionPairing().check_compatible(feature_type())

    @pytest.mark.parametrize("feature_type", [StubLabelledPointFeature, StubLabelledLineFeature])
    def test_region_pairing_accepts_feature_with_labels(self, feature_type: type[StubPointFeature]) -> None:
        LabelledRegionPairing().check_compatible(feature_type())

    @pytest.mark.parametrize("shape", ["circle", "square"])
    @pytest.mark.parametrize("feature_type", [StubLineFeature, StubLabelledLineFeature])
    def test_distance_pairing_rejects_line_feature(self, feature_type: type[StubPointFeature], shape: Shape) -> None:
        with pytest.raises(ValueError, match=r"(?i)line"):
            DistancePairing(DISTANCE_KM, shape=shape).check_compatible(feature_type())

    @pytest.mark.parametrize("shape", ["circle", "square"])
    @pytest.mark.parametrize("feature_type", [StubPointFeature, StubLabelledPointFeature])
    def test_distance_pairing_accepts_point_feature(self, feature_type: type[StubPointFeature], shape: Shape) -> None:
        DistancePairing(DISTANCE_KM, shape=shape).check_compatible(feature_type())

    @pytest.mark.parametrize("pairing", list(Pairing))
    def test_extrema_point_feature_is_compatible_with_every_strategy(self, pairing: Pairing) -> None:
        _strategy(pairing).check_compatible(_extrema_point_feature())

    @pytest.mark.parametrize(
        "feature_type", [StubPointFeature, StubLabelledPointFeature, StubLineFeature, StubLabelledLineFeature]
    )
    def test_base_accepts_any_feature(self, feature_type: type[StubPointFeature]) -> None:
        strategy = _pairing_subclass({"pair": _pair_nothing})()

        strategy.check_compatible(feature_type())


class TestPairingStrategyIsAbstract:
    def test_abstract_members_are_pair(self) -> None:
        # check_compatible is concrete (D18)
        assert PairingStrategy.__abstractmethods__ == frozenset({"pair"})

    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            PairingStrategy()  # pyright: ignore[reportAbstractUsage]

    def test_subclass_missing_pair_raises_type_error(self) -> None:
        incomplete = _pairing_subclass({})

        with pytest.raises(TypeError, match="pair"):
            incomplete()

    def test_subclass_with_pair_instantiates(self) -> None:
        complete = _pairing_subclass({"pair": _pair_nothing})()

        assert isinstance(complete, PairingStrategy)


STRATEGIES: tuple[PairingStrategy, ...] = (
    LabelledRegionPairing(),
    DistancePairing(DISTANCE_KM),
    DistancePairing(DISTANCE_KM, shape="square"),
)
STRATEGY_IDS: tuple[str, ...] = ("region", "circle", "square")


class TestStrategiesAreParameterObjects:
    @pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
    def test_is_a_pairing_strategy_and_a_dataclass(self, strategy: PairingStrategy) -> None:
        assert isinstance(strategy, PairingStrategy)
        assert dataclasses.is_dataclass(strategy)

    def test_region_pairing_has_no_fields(self) -> None:
        assert dataclasses.fields(LabelledRegionPairing) == ()

    def test_distance_pairing_fields(self) -> None:
        assert [field.name for field in dataclasses.fields(DistancePairing)] == ["max_distance_km", "shape"]

    @pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
    def test_is_frozen(self, strategy: PairingStrategy) -> None:
        assert dataclasses.is_dataclass(strategy)
        # A frozen dataclass rejects every attribute, so a strategy without fields is checked with a new one
        names = [field.name for field in dataclasses.fields(strategy)] or ["new_attribute"]
        for name in names:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(strategy, name, None)

    @pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
    def test_pickles_and_round_trips(self, strategy: PairingStrategy) -> None:
        restored = pickle.loads(pickle.dumps(strategy))

        assert restored == strategy
        assert type(restored) is type(strategy)

    @pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
    def test_restored_strategy_pairs_the_same(self, strategy: PairingStrategy) -> None:
        hand_built = _hand_built_scenario(GRID_LONS["prime-meridian"])
        restored: PairingStrategy = pickle.loads(pickle.dumps(strategy))

        pd.testing.assert_frame_equal(_pair(restored, hand_built), _pair(strategy, hand_built))

    @pytest.mark.parametrize("strategy", STRATEGIES, ids=STRATEGY_IDS)
    def test_pickled_size_is_small(self, strategy: PairingStrategy) -> None:
        # Rule 5: strategies hold parameters only, so they cost almost nothing to send to a task
        assert len(pickle.dumps(strategy)) < MAX_PICKLED_PAIRING_BYTES

    def test_tokenises_deterministically_from_fields(self) -> None:
        assert tokenize(DistancePairing(DISTANCE_KM)) == tokenize(DistancePairing(DISTANCE_KM))
        assert tokenize(DistancePairing(DISTANCE_KM)) != tokenize(DistancePairing(2 * DISTANCE_KM))
        assert tokenize(DistancePairing(DISTANCE_KM)) != tokenize(DistancePairing(DISTANCE_KM, shape="square"))
        assert tokenize(LabelledRegionPairing()) == tokenize(LabelledRegionPairing())
