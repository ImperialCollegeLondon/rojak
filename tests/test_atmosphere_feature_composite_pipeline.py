"""Tests for the pipeline facade of :mod:`rojak.atmosphere.feature_composite` (``_pipeline``, phase 1, step E1).

``composite_observations_about_features`` is the facade (section 3.9: a Facade so callers never assemble the dask
graph themselves). It is exercised here with a trivial, test-local feature (section 6: the feature-agnostic test)
rather than ``ExtremaPointFeature``, to prove the pipeline does not depend on lows. The phase 0 baseline in
``tests/test_atmosphere_feature_composite.py`` separately retargets the same facade against lows.

Layout:

1. A minimal, complete feature written against the ``FeatureSource``/``FeatureTabulator`` interface, as an extender
   would, with centres fixed ahead of time rather than found in the data.
2. Synthetic data: a small grid/field/observations for correctness and schema tests, and a large grid/field for the
   graph tests that need a block big enough that embedding it would be obvious.
3. Brute force, written independently of the pipeline (section 2: data flow; D15 is out of scope, phase 2).
4. Tests: feature-agnostic correctness, validation, output schema, ``return_pairs``, the graph rules (section 0,
   rules 1-8; section 6), and the default time tolerance (D15 phase 1 default: half the median field time step).
"""

import contextlib
import dataclasses
import warnings
from collections.abc import Iterator, Mapping
from pkgutil import resolve_name
from typing import TYPE_CHECKING, Any

import cloudpickle
import dask.array as da
import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr
from dask._task_spec import DataNode, Task
from dask.base import collections_to_expr, compute, is_dask_collection
from dask.local import get_sync

from rojak.atmosphere.compositing import _TargetGrid
from rojak.atmosphere.feature_composite._binning import BinSpec
from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._controls import ControlDesign
from rojak.atmosphere.feature_composite._feature import (
    ChunkInfo,
    FeatureChunk,
    FeatureGeometry,
    FeatureSource,
    FeatureTabulator,
)
from rojak.atmosphere.feature_composite._pairing import DistancePairing, LabelledRegionPairing
from rojak.atmosphere.feature_composite._pipeline import composite_observations_about_features
from rojak.core.geometric import haversine_distance
from rojak.core.indexing import nearest_index

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Patched (by name, before building the graph) to count how often the chunk task runs
CHUNK_TASK: str = "rojak.atmosphere.feature_composite._pipeline._composite_chunk"
# The one place allowed to compute while the graph is built (counting the observations per chunk)
GRAPH_BUILD_COMPUTE: tuple[type, str] = (_TargetGrid, "group_by_chunk")
MAX_PICKLED_TASK_BYTES: int = 50_000


# ======================================================================================================================
# 1. A minimal, complete feature written against the interface (section 6: the feature-agnostic test)
# ======================================================================================================================


@dataclasses.dataclass(frozen=True, eq=False)
class FixedCentresTabulator(FeatureTabulator):
    """Looks up this chunk's rows from a centre table fixed ahead of time, and stamps them as discs.

    ``centres`` holds every centre of every time step (small, so it is a plain field rather than a lazy one); only the
    rows within this chunk's time range are returned, as a real tabulator would select from its own computation.
    """

    centres: pd.DataFrame
    radius_cells: int = 2

    def __call__(self, blocks: Mapping[str, np.ndarray], chunk: ChunkInfo) -> FeatureChunk:
        del blocks  # The centres are fixed ahead of time, not read from the block
        first, stop = chunk.first_time_index, chunk.first_time_index + chunk.times.size
        in_chunk = self.centres[
            (self.centres[Column.TIME_INDEX] >= first) & (self.centres[Column.TIME_INDEX] < stop)
        ].sort_values(Column.CENTRE_ID)

        labels = np.zeros((chunk.times.size, chunk.lats.size, chunk.lons.size), dtype=np.int64)
        lat_grid, lon_grid = np.meshgrid(np.arange(chunk.lats.size), np.arange(chunk.lons.size), indexing="ij")
        for centre_id, time_index, lat_index, lon_index in zip(
            in_chunk[Column.CENTRE_ID].tolist(),
            in_chunk[Column.TIME_INDEX].tolist(),
            in_chunk[Column.CENTRE_LAT_INDEX].tolist(),
            in_chunk[Column.CENTRE_LON_INDEX].tolist(),
            strict=True,
        ):
            layer = labels[time_index - first]
            within = (lat_grid - lat_index) ** 2 + (lon_grid - lon_index) ** 2 <= self.radius_cells**2
            layer[within & (layer == 0)] = centre_id

        return FeatureChunk(centres=in_chunk.reset_index(drop=True), labels=labels)


@dataclasses.dataclass(frozen=True, eq=False)
class FixedCentresFeature(FeatureSource):
    """A trivial point feature whose centres are fixed ahead of time, unrelated to lows (section 6).

    ``field`` is only used to give the pipeline a time-chunked ``(time, lat, lon)`` template; its values are never
    read by the tabulator.
    """

    field: xr.DataArray
    centres: pd.DataFrame
    radius_cells: int = 2

    @property
    def geometry(self) -> FeatureGeometry:
        return FeatureGeometry.POINT

    @property
    def has_labels(self) -> bool:
        return True

    def chunk_inputs(self) -> Mapping[str, xr.DataArray]:
        return {"values": self.field}

    def tabulator(self) -> FeatureTabulator:
        return FixedCentresTabulator(centres=self.centres, radius_cells=self.radius_cells)


@dataclasses.dataclass(frozen=True, eq=False)
class UnlabelledFeature(FeatureSource):
    """A feature that never fills ``FeatureChunk.labels`` (``has_labels`` left at its default), for validation."""

    field: xr.DataArray

    @property
    def geometry(self) -> FeatureGeometry:
        return FeatureGeometry.POINT

    def chunk_inputs(self) -> Mapping[str, xr.DataArray]:
        return {"values": self.field}

    def tabulator(self) -> FeatureTabulator:
        raise NotImplementedError("check_compatible() must reject this feature before a tabulator is ever built")


# ======================================================================================================================
# 2. Synthetic data
# ======================================================================================================================

LATS: np.ndarray = np.arange(50.0, 29.5, -1.0)  # 21 points, descending as in ERA5: 50, 49, ..., 30
LONS: np.ndarray = np.arange(-20.0, 20.5, 1.0)  # 41 points: -20, -19, ..., 20
N_TIME: int = 6
TIME_CHUNK: int = 3  # Two time chunks
RADIUS_CELLS: int = 2
CENTRE_A_INDEX: tuple[int, int] = (5, 10)  # lat 45, lon -10
CENTRE_B_INDEX: tuple[int, int] = (15, 35)  # lat 35, lon 15: outside the observation domain below, always unpaired
OBS_LAT_RANGE: tuple[float, float] = (30.0, 50.0)
OBS_LON_RANGE: tuple[float, float] = (-20.0, 0.0)  # Excludes centre B's longitude
N_OBSERVATIONS: int = 500
DISTANCE_KM: float = 250.0
MAX_KM_EXTENT: float = 400.0
KM_SPACING: float = 100.0

LARGE_N_LAT: int = 220
LARGE_N_LON: int = 220
LARGE_N_TIME: int = 4
LARGE_TIME_CHUNK: int = 2


def _times() -> pd.DatetimeIndex:
    return pd.date_range("2024-03-01", periods=N_TIME, freq="6h")


def _field_times() -> np.ndarray:
    return _times().to_numpy().astype("datetime64[ns]")


@pytest.fixture(scope="module")
def small_field() -> xr.DataArray:
    data = da.zeros((N_TIME, LATS.size, LONS.size), chunks=(TIME_CHUNK, -1, -1))
    return xr.DataArray(
        data, dims=("time", "latitude", "longitude"), coords={"time": _times(), "latitude": LATS, "longitude": LONS}
    )


@pytest.fixture(scope="module")
def centres() -> pd.DataFrame:
    """Two fixed centres per time step: ``A`` (always paired) and ``B`` (always unpaired, see ``OBS_LON_RANGE``)."""
    rows: list[tuple[int, int, int, int]] = []
    for time_index in range(N_TIME):
        rows.append((2 * time_index, time_index, *CENTRE_A_INDEX))
        rows.append((2 * time_index + 1, time_index, *CENTRE_B_INDEX))
    centre_id, time_index, lat_index, lon_index = (np.array(column) for column in zip(*rows, strict=True))
    return pd.DataFrame(
        {
            Column.CENTRE_ID: centre_id,
            Column.FEATURE_ID: centre_id,
            Column.CENTRE_TIME: _field_times()[time_index],
            Column.CENTRE_LAT: LATS[lat_index].astype(float),
            Column.CENTRE_LON: LONS[lon_index].astype(float),
            Column.CENTRE_LAT_INDEX: lat_index,
            Column.CENTRE_LON_INDEX: lon_index,
            Column.TIME_INDEX: time_index,
        }
    )


def centre_b_ids(centre_table: pd.DataFrame) -> set[int]:
    return set(centre_table.loc[centre_table[Column.CENTRE_ID] % 2 == 1, Column.CENTRE_ID].tolist())


@pytest.fixture(scope="module")
def small_feature(small_field: xr.DataArray, centres: pd.DataFrame) -> FixedCentresFeature:
    return FixedCentresFeature(small_field, centres, radius_cells=RADIUS_CELLS)


@pytest.fixture(scope="module")
def points() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    lat = rng.uniform(*OBS_LAT_RANGE, N_OBSERVATIONS)
    lon = rng.uniform(*OBS_LON_RANGE, N_OBSERVATIONS)
    start = _field_times()[0]
    minutes = rng.integers(0, (N_TIME - 1) * 6 * 60, N_OBSERVATIONS)
    return pd.DataFrame(
        {
            "time": start + minutes.astype("timedelta64[m]"),
            "latitude": lat,
            "longitude": lon,
            "id": np.arange(N_OBSERVATIONS),
        }
    )


@pytest.fixture(scope="module")
def points_dd(points: pd.DataFrame) -> dd.DataFrame:
    return dd.from_pandas(points, npartitions=4)


@pytest.fixture(scope="module")
def bins() -> BinSpec:
    return BinSpec.grid(MAX_KM_EXTENT, KM_SPACING)


@pytest.fixture(scope="module")
def large_field() -> xr.DataArray:
    """A grid large enough that one block, if ever embedded whole in a task, would dwarf ``MAX_PICKLED_TASK_BYTES``."""
    lats = np.linspace(80.0, 20.0, LARGE_N_LAT)
    lons = np.linspace(-100.0, 100.0, LARGE_N_LON)
    times = pd.date_range("2024-06-01", periods=LARGE_N_TIME, freq="6h")
    data = da.zeros((LARGE_N_TIME, LARGE_N_LAT, LARGE_N_LON), chunks=(LARGE_TIME_CHUNK, -1, -1))
    return xr.DataArray(
        data, dims=("time", "latitude", "longitude"), coords={"time": times, "latitude": lats, "longitude": lons}
    )


@pytest.fixture(scope="module")
def large_centres(large_field: xr.DataArray) -> pd.DataFrame:
    lat_index, lon_index = LARGE_N_LAT // 2, LARGE_N_LON // 2
    return pd.DataFrame(
        {
            Column.CENTRE_ID: [0],
            Column.FEATURE_ID: [0],
            Column.CENTRE_TIME: [large_field["time"].to_numpy()[0]],
            Column.CENTRE_LAT: [float(large_field["latitude"].to_numpy()[lat_index])],
            Column.CENTRE_LON: [float(large_field["longitude"].to_numpy()[lon_index])],
            Column.CENTRE_LAT_INDEX: [lat_index],
            Column.CENTRE_LON_INDEX: [lon_index],
            Column.TIME_INDEX: [0],
        }
    )


@pytest.fixture(scope="module")
def large_feature(large_field: xr.DataArray, large_centres: pd.DataFrame) -> FixedCentresFeature:
    return FixedCentresFeature(large_field, large_centres)


@pytest.fixture(scope="module")
def large_points_dd(large_field: xr.DataArray, large_centres: pd.DataFrame) -> dd.DataFrame:
    lat0 = float(large_centres[Column.CENTRE_LAT].iloc[0])
    lon0 = float(large_centres[Column.CENTRE_LON].iloc[0])
    time0 = large_field["time"].to_numpy()[0]
    points = pd.DataFrame(
        {
            "time": np.full(10, time0),
            "latitude": np.full(10, lat0),
            "longitude": np.full(10, lon0),
            "id": np.arange(10),
        }
    )
    return dd.from_pandas(points, npartitions=2)


# ======================================================================================================================
# 3. Brute force, independent of the pipeline
# ======================================================================================================================


def _half_time_step() -> np.timedelta64:
    field_times = _field_times()
    median_step_ns = float(np.median(np.diff(field_times).astype(np.int64)))
    return np.timedelta64(round(median_step_ns / 2), "ns")


def _locate_in_time(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest field time index and whether it is within half a time step, as the pipeline's default tolerance does."""
    field_times = _field_times()
    observation_times = times.astype("datetime64[ns]")
    time_index = nearest_index(field_times, observation_times)
    is_snapped = np.abs(field_times[time_index] - observation_times) <= _half_time_step()
    return time_index, is_snapped


def _brute_force_distance_pairs(
    points: pd.DataFrame, centre_table: pd.DataFrame, max_distance_km: float
) -> pd.DataFrame:
    time_index, is_snapped = _locate_in_time(points["time"].to_numpy())
    located = points[is_snapped].assign(**{Column.TIME_INDEX: time_index[is_snapped]})
    cross = located.merge(centre_table, on=Column.TIME_INDEX)
    distance_km = (
        haversine_distance(
            cross["longitude"].to_numpy(dtype=float),
            cross["latitude"].to_numpy(dtype=float),
            cross[Column.CENTRE_LON].to_numpy(dtype=float),
            cross[Column.CENTRE_LAT].to_numpy(dtype=float),
        )
        / 1000
    )
    return cross[distance_km <= max_distance_km]


def _brute_force_region_pairs(points: pd.DataFrame, centre_table: pd.DataFrame, radius_cells: int) -> pd.DataFrame:
    """Each observation paired with the centre whose disc (in grid-index space) contains its nearest cell.

    Centres ``A`` and ``B`` are always far enough apart that at most one centre's disc ever reaches a cell, so no
    tie-break is needed, unlike the real tabulator's stamping order (D19).
    """
    time_index, is_snapped = _locate_in_time(points["time"].to_numpy())
    located = points[is_snapped].assign(**{Column.TIME_INDEX: time_index[is_snapped]})
    lat_index = nearest_index(LATS, located["latitude"].to_numpy(dtype=float))
    lon_index = nearest_index(LONS, located["longitude"].to_numpy(dtype=float))

    centre_time = centre_table[Column.TIME_INDEX].to_numpy()
    centre_lat_index = centre_table[Column.CENTRE_LAT_INDEX].to_numpy()
    centre_lon_index = centre_table[Column.CENTRE_LON_INDEX].to_numpy()
    centre_id = centre_table[Column.CENTRE_ID].to_numpy()
    obs_time_index = located[Column.TIME_INDEX].to_numpy()

    owners = np.zeros(len(located), dtype=np.int64)
    for row in range(len(located)):
        candidates = np.flatnonzero(
            (centre_time == obs_time_index[row])
            & ((centre_lat_index - lat_index[row]) ** 2 + (centre_lon_index - lon_index[row]) ** 2 <= radius_cells**2)
        )
        if candidates.size > 0:
            assert candidates.size == 1, "centres A and B must never both reach the same cell"
            owners[row] = centre_id[candidates[0]]
    owned = located.assign(**{Column.CENTRE_ID: owners})[owners != 0]
    return owned.merge(centre_table, on=[Column.TIME_INDEX, Column.CENTRE_ID])


def _project(pairs: pd.DataFrame) -> pd.DataFrame:
    """Position of each pair in the orthographic km frame of its centre, by a fresh per-centre ``pyproj`` projection."""
    x_km = np.full(len(pairs), np.nan)
    y_km = np.full(len(pairs), np.nan)
    obs_lat = pairs["latitude"].to_numpy(dtype=float)
    obs_lon = pairs["longitude"].to_numpy(dtype=float)
    centre_lat = pairs[Column.CENTRE_LAT].to_numpy(dtype=float)
    centre_lon = pairs[Column.CENTRE_LON].to_numpy(dtype=float)
    for positions in pairs.groupby(Column.CENTRE_ID, sort=False).indices.values():
        ortho = pyproj.Proj(
            proj="ortho",
            lat_0=float(centre_lat[positions[0]]),
            lon_0=float(centre_lon[positions[0]]),
            units="km",
            ellps="WGS84",
        )
        x_km[positions], y_km[positions] = ortho(obs_lon[positions], obs_lat[positions])
    return pairs.assign(**{Column.X_KM: x_km, Column.Y_KM: y_km})


def _expected_histogram(pairs: pd.DataFrame, bin_spec: BinSpec) -> np.ndarray:
    projected = _project(pairs)
    sample = np.column_stack([projected[dim].to_numpy(dtype=float) for dim in bin_spec.dims])
    edges = bin_spec.edges
    counts, _ = np.histogramdd(sample, bins=[edges[dim] for dim in bin_spec.dims])
    return counts.astype(np.int64)


# ======================================================================================================================
# 4. Graph helpers
# ======================================================================================================================


@contextlib.contextmanager
def _counting_scheduler() -> Iterator[list[int]]:
    """Count the number of times something is computed, using the synchronous scheduler."""
    computes: list[int] = []

    def counting_get(*args: Any, **kwargs: Any) -> Any:
        computes.append(1)
        return get_sync(*args, **kwargs)

    with dask.config.set(scheduler=counting_get):
        yield computes


def _counting(function: Any, calls: list[int]) -> Any:
    """Wrap ``function`` so that each call is appended to ``calls``. A plain closure, so dask can tokenise it."""

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return function(*args, **kwargs)

    return counted


def _iter_task_literals(node: object) -> Iterator[object]:
    """Recursively yield every literal argument embedded in one graph node (task, data node or nested container).

    Used to check rule 5 (objects sent to tasks hold parameters only, never a dask collection). Handles both the
    ``dask._task_spec.Task``/``DataNode`` representation and plain ``(callable, *args)`` tuples, since a merged graph
    of mixed collections (an array-backed ``xarray.Dataset`` and dataframes) may contain either.
    """
    if isinstance(node, DataNode):
        yield from _iter_task_literals(node.value)
    elif isinstance(node, Task):
        for arg in (*node.args, *node.kwargs.values()):
            yield from _iter_task_literals(arg)
    elif isinstance(node, tuple) and node and callable(node[0]):
        for arg in node[1:]:
            yield from _iter_task_literals(arg)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_task_literals(item)
    elif isinstance(node, dict):
        for item in node.values():
            yield from _iter_task_literals(item)
    else:
        yield node


def _merged_graph(*collections: object) -> Mapping[object, object]:
    return collections_to_expr(list(collections), optimize_graph=False).__dask_graph__()


# ======================================================================================================================
# 5. Tests
# ======================================================================================================================


class TestFeatureAgnosticPipeline:
    """A trivial, test-local feature run through the full pipeline (section 6)."""

    def test_distance_pairing_counts_match_brute_force(
        self,
        small_feature: FixedCentresFeature,
        points_dd: dd.DataFrame,
        points: pd.DataFrame,
        centres: pd.DataFrame,
        bins: BinSpec,
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins
        )

        expected_pairs = _brute_force_distance_pairs(points, centres, DISTANCE_KM)
        expected = _expected_histogram(expected_pairs, bins)
        assert expected.sum() > 0
        computed = result["feature_n_obs"].sum("chunk").transpose(*bins.dims).compute().to_numpy()
        np.testing.assert_array_equal(computed, expected)
        assert int(result["n_centres"].sum().compute()) == len(centres)

    def test_region_pairing_counts_match_brute_force(
        self,
        small_feature: FixedCentresFeature,
        points_dd: dd.DataFrame,
        points: pd.DataFrame,
        centres: pd.DataFrame,
        bins: BinSpec,
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=LabelledRegionPairing(), bins=bins
        )

        expected_pairs = _brute_force_region_pairs(points, centres, RADIUS_CELLS)
        expected = _expected_histogram(expected_pairs, bins)
        assert expected.sum() > 0
        computed = result["feature_n_obs"].sum("chunk").transpose(*bins.dims).compute().to_numpy()
        np.testing.assert_array_equal(computed, expected)
        assert int(result["n_centres"].sum().compute()) == len(centres)


class TestValidation:
    def test_feature_that_is_not_a_feature_source_raises_type_error(
        self, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        with pytest.raises(TypeError):
            composite_observations_about_features(
                "not-a-feature",  # pyright: ignore[reportArgumentType]
                points_dd,
                pairing=DistancePairing(DISTANCE_KM),
                bins=bins,
            )

    def test_pairing_that_is_not_a_pairing_strategy_raises_type_error(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        with pytest.raises(TypeError):
            composite_observations_about_features(
                small_feature,
                points_dd,
                pairing="not-a-pairing",  # pyright: ignore[reportArgumentType]
                bins=bins,
            )

    def test_region_pairing_with_unlabelled_feature_raises_value_error(
        self, small_field: xr.DataArray, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        with pytest.raises(ValueError, match="label"):
            composite_observations_about_features(
                UnlabelledFeature(small_field), points_dd, pairing=LabelledRegionPairing(), bins=bins
            )

    def test_duplicate_control_labels_raise_value_error(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        duplicate_controls = [
            ControlDesign("ctl", observation_offset=np.timedelta64(-6, "h")),
            ControlDesign("ctl", observation_offset=np.timedelta64(6, "h")),
        ]
        with pytest.raises(ValueError, match="ctl"):
            composite_observations_about_features(
                small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, controls=duplicate_controls
            )

    def test_bins_that_is_not_a_bin_spec_raises_type_error(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame
    ) -> None:
        with pytest.raises(TypeError):
            composite_observations_about_features(
                small_feature,
                points_dd,
                pairing=DistancePairing(DISTANCE_KM),
                bins="not-a-bin-spec",  # pyright: ignore[reportArgumentType]
            )


class TestOutputSchema:
    def test_variables_dims_and_coords_without_controls(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins
        )

        assert set(result.data_vars) == {"feature_n_obs", "n_centres"}
        assert result["feature_n_obs"].dims == ("chunk", "y_km", "x_km")
        assert result["n_centres"].dims == ("chunk",)
        np.testing.assert_array_equal(result["y_km"].to_numpy(), bins.coords["y_km"])
        np.testing.assert_array_equal(result["x_km"].to_numpy(), bins.coords["x_km"])
        assert "control" not in result.dims

    def test_variables_dims_and_coords_with_controls(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        designs = [
            ControlDesign("obs-6h", observation_offset=np.timedelta64(-6, "h")),
            ControlDesign("obs+12h", observation_offset=np.timedelta64(12, "h")),
        ]
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, controls=designs
        )

        assert set(result.data_vars) == {"feature_n_obs", "n_centres", "control_n_obs", "n_controls"}
        assert result["control_n_obs"].dims == ("chunk", "control", "y_km", "x_km")
        assert result["n_controls"].dims == ("chunk", "control")
        assert result["control"].to_numpy().tolist() == ["obs-6h", "obs+12h"]

    def test_everything_is_dask_backed(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins
        )

        assert is_dask_collection(result["feature_n_obs"].data)
        assert is_dask_collection(result["n_centres"].data)

    def test_attrs_record_feature_geometry_pairing_bins_and_controls(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        pairing = DistancePairing(DISTANCE_KM)
        designs = [ControlDesign("obs-6h", observation_offset=np.timedelta64(-6, "h"))]

        result = composite_observations_about_features(
            small_feature, points_dd, pairing=pairing, bins=bins, controls=designs
        )

        assert result.attrs["feature"] == type(small_feature).__name__
        assert result.attrs["geometry"] == FeatureGeometry.POINT.value
        assert repr(pairing) in result.attrs["pairing"]
        assert repr(bins) in result.attrs["bins"]
        assert tuple(result.attrs["controls"]) == ("obs-6h",)
        assert result.attrs["normalisation_unit"] == "feature instance"

    def test_no_controls_gives_empty_controls_attr(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins
        )

        assert tuple(result.attrs["controls"]) == ()


class TestReturnPairs:
    def test_returns_a_dataset_and_two_dask_dataframes(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        result = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )

        assert isinstance(result, tuple)
        counts, computed_centres, pairs = result
        assert isinstance(counts, xr.Dataset)
        assert isinstance(computed_centres, dd.DataFrame)
        assert isinstance(pairs, dd.DataFrame)

    def test_centres_include_unpaired_centres_but_pairs_do_not(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, centres: pd.DataFrame, bins: BinSpec
    ) -> None:
        counts, computed_centres, pairs = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )
        del counts
        computed_centres = computed_centres.compute()
        computed_pairs = pairs.compute()

        unpaired = centre_b_ids(centres)
        assert unpaired <= set(computed_centres[Column.CENTRE_ID].tolist())
        assert not computed_pairs[Column.CENTRE_ID].isin(unpaired).any()

    def test_summed_pair_counts_equal_feature_n_obs_sum(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        counts, computed_centres, pairs = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )
        del computed_centres
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Computing mixed collections")
            computed_counts, computed_pairs = compute(counts, pairs, scheduler="sync")

        assert int(computed_counts["feature_n_obs"].sum()) == len(computed_pairs)
        assert len(computed_pairs) > 0


WITHIN_TOLERANCE_ID: int = 1
OUTSIDE_TOLERANCE_ID: int = 2
ANCHOR_ID: int = 3


class TestTimeTolerance:
    """D15 phase 1 default: half the median field time step (6 h fields -> 3 h tolerance)."""

    def test_observation_within_half_a_time_step_is_paired(
        self, small_feature: FixedCentresFeature, bins: BinSpec
    ) -> None:
        centre_a_time = _field_times()[0]
        centre_a_lat = float(LATS[CENTRE_A_INDEX[0]])
        centre_a_lon = float(LONS[CENTRE_A_INDEX[1]])
        within = pd.DataFrame(
            {
                "time": [centre_a_time - np.timedelta64(2 * 60 + 59, "m")],
                "latitude": [centre_a_lat],
                "longitude": [centre_a_lon],
                "id": [WITHIN_TOLERANCE_ID],
            }
        )
        outside = pd.DataFrame(
            {
                "time": [centre_a_time - np.timedelta64(3 * 60 + 1, "m")],
                "latitude": [centre_a_lat],
                "longitude": [centre_a_lon],
                "id": [OUTSIDE_TOLERANCE_ID],
            }
        )
        both = dd.from_pandas(pd.concat([within, outside], ignore_index=True), npartitions=1)

        _, _, pairs = composite_observations_about_features(
            small_feature, both, pairing=DistancePairing(10.0), bins=bins, return_pairs=True
        )
        paired_ids = set(pairs.compute()["id"].tolist())

        assert WITHIN_TOLERANCE_ID in paired_ids
        assert OUTSIDE_TOLERANCE_ID not in paired_ids

    def test_explicit_time_tolerance_overrides_the_default(
        self, small_feature: FixedCentresFeature, bins: BinSpec
    ) -> None:
        # 3 h 01 min from the centre time: outside the default tolerance (half of 6 h), but inside an explicit,
        # wider tolerance passed by the caller. An anchor observation exactly on the centre time is included so that
        # some observation always locates to a time step, whatever the tolerance (otherwise group_by_chunk raises).
        centre_a_time = _field_times()[0]
        centre_a_lat = float(LATS[CENTRE_A_INDEX[0]])
        centre_a_lon = float(LONS[CENTRE_A_INDEX[1]])
        anchor = pd.DataFrame(
            {"time": [centre_a_time], "latitude": [centre_a_lat], "longitude": [centre_a_lon], "id": [ANCHOR_ID]}
        )
        boundary = pd.DataFrame(
            {
                "time": [centre_a_time - np.timedelta64(3 * 60 + 1, "m")],
                "latitude": [centre_a_lat],
                "longitude": [centre_a_lon],
                "id": [WITHIN_TOLERANCE_ID],
            }
        )
        observations = dd.from_pandas(pd.concat([anchor, boundary], ignore_index=True), npartitions=1)

        _, _, default_pairs = composite_observations_about_features(
            small_feature, observations, pairing=DistancePairing(10.0), bins=bins, return_pairs=True
        )
        _, _, wide_pairs = composite_observations_about_features(
            small_feature,
            observations,
            pairing=DistancePairing(10.0),
            bins=bins,
            return_pairs=True,
            time_tolerance=np.timedelta64(4, "h"),
        )

        assert WITHIN_TOLERANCE_ID not in set(default_pairs.compute()["id"].tolist())
        assert WITHIN_TOLERANCE_ID in set(wide_pairs.compute()["id"].tolist())


class TestGraph:
    def test_building_graph_only_computes_in_group_by_chunk(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec, mocker: "MockerFixture"
    ) -> None:
        group_by_chunk = mocker.spy(*GRAPH_BUILD_COMPUTE)
        with _counting_scheduler() as computes:
            composite_observations_about_features(
                small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins
            )
        assert group_by_chunk.call_count >= 1
        assert len(computes) == group_by_chunk.call_count

    def test_building_graph_with_controls_only_computes_in_group_by_chunk(
        self, small_feature: FixedCentresFeature, points_dd: dd.DataFrame, bins: BinSpec, mocker: "MockerFixture"
    ) -> None:
        designs = [
            ControlDesign("obs-6h", observation_offset=np.timedelta64(-6, "h")),
            ControlDesign("obs+6h", observation_offset=np.timedelta64(6, "h")),
        ]
        group_by_chunk = mocker.spy(*GRAPH_BUILD_COMPUTE)
        with _counting_scheduler() as computes:
            composite_observations_about_features(
                small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, controls=designs
            )
        assert group_by_chunk.call_count >= 1
        assert len(computes) == group_by_chunk.call_count

    def test_chunk_task_runs_once_per_chunk_when_computed_together(
        self,
        small_feature: FixedCentresFeature,
        small_field: xr.DataArray,
        points_dd: dd.DataFrame,
        bins: BinSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(CHUNK_TASK, _counting(resolve_name(CHUNK_TASK), calls))

        counts, computed_centres, pairs = composite_observations_about_features(
            small_feature, points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )
        # Building may evaluate the task on empty inputs (e.g. for the meta), which is not a run of a chunk task
        calls.clear()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Computing mixed collections")
            compute(counts, computed_centres, pairs, scheduler="sync")

        assert len(calls) == len(small_field.chunksizes["time"])


class TestGraphTaskSize:
    """Section 6: pickled per-task size bounded, and no dask collection embedded in a task (rule 5)."""

    def test_pickled_task_size_is_bounded(
        self, large_feature: FixedCentresFeature, large_points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        counts, computed_centres, pairs = composite_observations_about_features(
            large_feature, large_points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )

        graph = _merged_graph(counts, computed_centres, pairs)
        assert len(graph) > 0
        too_big = {key: len(cloudpickle.dumps(node)) for key, node in graph.items()}
        too_big = {key: size for key, size in too_big.items() if size >= MAX_PICKLED_TASK_BYTES}
        assert not too_big, f"Tasks at least as big as one field block (rule 1): {too_big}"

    def test_no_dask_collection_is_embedded_in_a_task(
        self, large_feature: FixedCentresFeature, large_points_dd: dd.DataFrame, bins: BinSpec
    ) -> None:
        counts, computed_centres, pairs = composite_observations_about_features(
            large_feature, large_points_dd, pairing=DistancePairing(DISTANCE_KM), bins=bins, return_pairs=True
        )

        graph = _merged_graph(counts, computed_centres, pairs)
        embedded = [leaf for node in graph.values() for leaf in _iter_task_literals(node) if is_dask_collection(leaf)]
        assert not embedded
