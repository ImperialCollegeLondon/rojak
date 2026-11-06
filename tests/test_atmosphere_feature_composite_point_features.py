"""Tests for :class:`ExtremaPointFeature` and its tabulator (phase 1, step C of ``storm_composite_plan.md``).

The centres and labels are checked against brute force written independently of the implementation: connected
components (8-connectivity) of ``identify_circular_extrema``'s ``local_extrema`` per time step (D7), represented by
their first pixel in row-major order (D19), and a cell-by-cell loop for the "deepest centre within the footprint of any
pixel of its component, ties to the smallest ``centre_id``" labelling rule (D11, D19).

Layout:

1. Synthetic data: smooth Gaussian lows (a realistic field) and hand-built cones (exact geometry for the edge cases).
2. Brute force and helpers.
3. Tests.

The exact ``centre_id`` offset is left free: the tests pin that ids are >= 1 (0 means "no owner" in the labels), unique
across chunks, independent of how the field is chunked, and increasing in row-major ``(time, lat, lon)`` order, which is
what makes "smallest id" the first pixel in row-major order (D19).
"""

import contextlib
import dataclasses
import itertools
import pickle
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from typing import Any, NamedTuple

import dask.config
import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi
import xarray as xr
from dask.base import compute, is_dask_collection, tokenize
from dask.local import get_sync

from rojak.atmosphere import regions
from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._feature import (
    ChunkInfo,
    FeatureChunk,
    FeatureGeometry,
    FeatureSource,
    FeatureTabulator,
)
from rojak.atmosphere.feature_composite._point_features import ExtremaPointFeature
from rojak.atmosphere.regions import ExtremaKind, circular_footprint, identify_circular_extrema

# Name-mangled private function of ``regions``; aliased at module level because mangling applies inside test classes
_extrema_mask_region: Callable[[np.ndarray, np.ndarray], np.ndarray] = getattr(regions, "_" + "_extrema_mask_region")

# Gaussian lows
GAUSSIAN_THRESHOLD: float = 100_000
GAUSSIAN_RADIUS: int = 4
GAUSSIAN_TIME_CHUNK: int = 2
# Hand-built cones: background 0, lows at -depth, so a threshold of -0.5 discards the flat background
HAND_SHAPE: tuple[int, int] = (21, 31)
HAND_RADIUS: int = 4
HAND_THRESHOLD: float = -0.5
PLATEAU_PIXELS: tuple[tuple[int, int], ...] = ((9, 12), (9, 13), (9, 14), (10, 12), (10, 13), (10, 14))

MAX_PICKLED_TABULATOR_BYTES: int = 1_000
CENTRE_COLUMNS: tuple[Column, ...] = (
    Column.CENTRE_ID,
    Column.FEATURE_ID,
    Column.CENTRE_TIME,
    Column.CENTRE_LAT,
    Column.CENTRE_LON,
    Column.CENTRE_VALUE,
    Column.CENTRE_LAT_INDEX,
    Column.CENTRE_LON_INDEX,
    Column.TIME_INDEX,
)
INTEGER_COLUMNS: tuple[Column, ...] = (
    Column.CENTRE_ID,
    Column.FEATURE_ID,
    Column.CENTRE_LAT_INDEX,
    Column.CENTRE_LON_INDEX,
    Column.TIME_INDEX,
)
FLOAT_COLUMNS: tuple[Column, ...] = (Column.CENTRE_LAT, Column.CENTRE_LON, Column.CENTRE_VALUE)
FEATURE_FIELDS: frozenset[str] = frozenset(
    {"field", "kind", "threshold", "footprint_radius", "time_dim_name", "lat_dim_name", "lon_dim_name"}
)
KEYWORD_ONLY_FIELDS: frozenset[str] = frozenset(
    {"threshold", "footprint_radius", "time_dim_name", "lat_dim_name", "lon_dim_name"}
)


# ======================================================================================================================
# 1. Synthetic data
# ======================================================================================================================


def _times(n_times: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n_times, freq="6h")


def _gaussian_lows() -> xr.DataArray:
    """Three Gaussian lows moving on a coarse grid, chunked along time.

    The centres are off-grid and not half-way between grid points, and the background varies in both directions, so
    the extrema are generic (no accidental ties). Two of the lows come close enough for their footprints to overlap.
    """
    times = _times(4)
    lats = np.arange(70, 26, -1.5)  # Descending, as in ERA5
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
    ).chunk({"time": GAUSSIAN_TIME_CHUNK})


def _cone_layer(bottoms: Sequence[tuple[Sequence[tuple[int, int]], float]], *, slope: float = 1.0) -> np.ndarray:
    """One time step: ``min(0, -depth + slope * distance)`` over the lows, with distance in grid indices to the nearest
    pixel of each low's bottom. Strictly increasing away from each bottom, so the only extrema below the threshold are
    the bottoms themselves."""
    lat_index, lon_index = np.indices(HAND_SHAPE)
    layer = np.zeros(HAND_SHAPE)
    for pixels, depth in bottoms:
        distance = np.min([np.hypot(lat_index - lat, lon_index - lon) for lat, lon in pixels], axis=0)
        layer = np.minimum(layer, -depth + slope * distance)
    return layer


def _hand_field(layers: Sequence[np.ndarray], *, time_chunk: int = 1) -> xr.DataArray:
    return xr.DataArray(
        np.stack(layers),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": _times(len(layers)),
            "latitude": np.linspace(70.0, 40.0, HAND_SHAPE[0]),
            "longitude": np.linspace(-30.0, 30.0, HAND_SHAPE[1]),
        },
        name="mslp",
    ).chunk({"time": time_chunk})


def _sign(kind: ExtremaKind) -> float:
    """Highs are the lows negated."""
    return 1.0 if kind is ExtremaKind.MINIMA else -1.0


def _hand_feature(layers: Sequence[np.ndarray], kind: ExtremaKind, *, time_chunk: int = 1) -> ExtremaPointFeature:
    sign = _sign(kind)
    return ExtremaPointFeature(
        _hand_field([sign * layer for layer in layers], time_chunk=time_chunk),
        kind,
        threshold=sign * HAND_THRESHOLD,
        footprint_radius=HAND_RADIUS,
    )


def _equal_separate_lows() -> list[np.ndarray]:
    """Two lows with identical minima, far apart: the prototype merged them into one."""
    return [_cone_layer([([(6, 8)], 10.0), ([(14, 22)], 10.0)])]


def _plateau_low() -> list[np.ndarray]:
    """One flat-bottomed low: a 2 x 3 block of pixels at the same minimum."""
    return [_cone_layer([(PLATEAU_PIXELS, 10.0)])]


def _diagonal_low() -> list[np.ndarray]:
    """One low whose two bottom pixels touch only diagonally (one component under 8-connectivity)."""
    return [_cone_layer([([(5, 5), (6, 6)], 10.0)])]


def _tied_overlapping_lows() -> list[np.ndarray]:
    """Two equal lows 5 cells apart: each is outside the other's footprint, but their footprints overlap."""
    return [_cone_layer([([(10, 10)], 10.0), ([(10, 15)], 10.0)])]


def _deeper_and_shallower_lows() -> list[np.ndarray]:
    """A deep (-10) and a shallow (-8) low 6 cells apart with overlapping footprints. Steep, so both are extrema."""
    return [_cone_layer([([(10, 10)], 10.0), ([(10, 16)], 8.0)], slope=3.0)]


def _low_at_origin() -> list[np.ndarray]:
    """A low at time, latitude and longitude index 0, where a plain row-major encoding of the id would be 0."""
    return [_cone_layer([([(0, 0)], 10.0)])]


def _stationary_and_moving_lows() -> list[np.ndarray]:
    """Four time steps: one low that doesn't move (same grid indices every step) and one moving east."""
    return [_cone_layer([([(8, 10)], 10.0), ([(14, 5 + 4 * step)], 9.0)]) for step in range(4)]


def _lows_only_in_second_chunk() -> list[np.ndarray]:
    """Four time steps (two chunks of two); the first chunk has no low at all."""
    low = _cone_layer([([(10, 10)], 10.0)])
    return [np.zeros(HAND_SHAPE), np.zeros(HAND_SHAPE), low, low]


HAND_CASES: dict[str, Callable[[], list[np.ndarray]]] = {
    "equal_separate": _equal_separate_lows,
    "plateau": _plateau_low,
    "diagonal": _diagonal_low,
    "tied_overlapping": _tied_overlapping_lows,
    "deeper_and_shallower": _deeper_and_shallower_lows,
    "origin": _low_at_origin,
    "stationary_and_moving": _stationary_and_moving_lows,
}


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
    """Wrap ``function`` so that each call is appended to ``calls``."""

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return function(*args, **kwargs)

    return counted


# ======================================================================================================================
# 2. Brute force and helpers
# ======================================================================================================================


class Component(NamedTuple):
    """A connected component of ``local_extrema`` at one time step.

    Attributes:
        time_index: Index along the whole time axis.
        pixels: ``(n, 2)`` lat and lon indices of its pixels, in row-major order.
        value: Field value at its pixels (equal at every pixel of a component).
    """

    time_index: int
    pixels: np.ndarray
    value: float

    @property
    def representative(self) -> tuple[int, int, int]:
        return self.time_index, int(self.pixels[0, 0]), int(self.pixels[0, 1])


def _reference_extrema(
    field: xr.DataArray, kind: ExtremaKind, threshold: float, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """``local_extrema`` and ``extrema_regions`` from :func:`identify_circular_extrema`, as ``(time, lat, lon)``."""
    extrema = (
        identify_circular_extrema(field, kind, extrema_threshold_value=threshold, footprint_radius=radius)
        .transpose("time", "latitude", "longitude")
        .compute()
    )
    return extrema["local_extrema"].to_numpy(), extrema["extrema_regions"].to_numpy()


def _brute_force_components(local_extrema: np.ndarray, values: np.ndarray) -> list[Component]:
    eight_connected = ndi.generate_binary_structure(2, 2)
    components: list[Component] = []
    for time_index in range(local_extrema.shape[0]):
        labelled, n_components = ndi.label(local_extrema[time_index], structure=eight_connected)
        for label in range(1, n_components + 1):
            # np.nonzero lists the pixels in row-major order
            lat_index, lon_index = np.nonzero(labelled == label)
            components.append(
                Component(
                    time_index,
                    np.column_stack([lat_index, lon_index]),
                    float(values[time_index, lat_index[0], lon_index[0]]),
                )
            )
    return components


def _brute_force_labels(
    regions_mask: np.ndarray,
    components: Sequence[Component],
    ids: dict[tuple[int, int, int], int],
    kind: ExtremaKind,
    radius: int,
) -> np.ndarray:
    """Cell by cell: among the components at the same time step with any pixel within ``radius`` (grid indices) of a
    region cell, the deepest owns it; equal depths go to the smallest id."""
    labels = np.zeros(regions_mask.shape, dtype=np.int64)
    for time_index, lat_index, lon_index in zip(*np.nonzero(regions_mask), strict=True):
        best: tuple[float, int] | None = None
        for component in components:
            if component.time_index != time_index:
                continue
            squared_distance = (component.pixels[:, 0] - lat_index) ** 2 + (component.pixels[:, 1] - lon_index) ** 2
            if squared_distance.min() > radius**2:
                continue
            depth = component.value if kind is ExtremaKind.MINIMA else -component.value
            candidate = (depth, ids[component.representative])
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            labels[time_index, lat_index, lon_index] = best[1]
    return labels


def _within(pixels: Sequence[tuple[int, int]], radius: int) -> np.ndarray:
    """Cells of a hand-built layer within ``radius`` grid indices of any of ``pixels``."""
    lat_index, lon_index = np.indices(HAND_SHAPE)
    squared = np.min([(lat_index - lat) ** 2 + (lon_index - lon) ** 2 for lat, lon in pixels], axis=0)
    return squared <= radius**2


def _tabulate(feature: ExtremaPointFeature, start: int, stop: int) -> FeatureChunk:
    """Run the tabulator on time steps ``[start, stop)`` as one chunk, as a chunk task would."""
    blocks = {
        name: array.isel({array.dims[0]: slice(start, stop)}).to_numpy()
        for name, array in feature.chunk_inputs().items()
    }
    field = feature.field
    chunk = ChunkInfo(
        times=field[feature.time_dim_name].to_numpy()[start:stop].astype("datetime64[ns]"),
        first_time_index=start,
        lats=field[feature.lat_dim_name].to_numpy(),
        lons=field[feature.lon_dim_name].to_numpy(),
    )
    return feature.tabulator()(blocks, chunk)


def _labels_of(result: FeatureChunk) -> np.ndarray:
    labels = result.labels
    assert labels is not None
    return labels


def _tabulate_by_chunks(feature: ExtremaPointFeature) -> tuple[pd.DataFrame, np.ndarray]:
    """Centres and labels of every time chunk of the field, concatenated."""
    bounds = np.cumsum((0, *feature.field.chunksizes[feature.time_dim_name])).tolist()
    results = [_tabulate(feature, start, stop) for start, stop in itertools.pairwise(bounds)]
    centres = pd.concat([result.centres for result in results], ignore_index=True)
    return centres, np.concatenate([_labels_of(result) for result in results], axis=0)


def _tabulate_whole(feature: ExtremaPointFeature) -> tuple[pd.DataFrame, np.ndarray]:
    result = _tabulate(feature, 0, feature.field.sizes[feature.time_dim_name])
    return result.centres, _labels_of(result)


def _ids_by_position(centres: pd.DataFrame) -> dict[tuple[int, int, int], int]:
    """``centre_id`` by ``(time_index, lat_index, lon_index)`` of the centre."""
    return {
        (int(time_index), int(lat_index), int(lon_index)): int(centre_id)
        for time_index, lat_index, lon_index, centre_id in zip(
            centres[Column.TIME_INDEX].to_numpy().tolist(),
            centres[Column.CENTRE_LAT_INDEX].to_numpy().tolist(),
            centres[Column.CENTRE_LON_INDEX].to_numpy().tolist(),
            centres[Column.CENTRE_ID].to_numpy().tolist(),
            strict=True,
        )
    }


def _id_at(centres: pd.DataFrame, lat_index: int, lon_index: int, time_index: int = 0) -> int:
    return _ids_by_position(centres)[(time_index, lat_index, lon_index)]


class Case(NamedTuple):
    """A tabulated field with its independent reference."""

    feature: ExtremaPointFeature
    kind: ExtremaKind
    radius: int
    field_values: np.ndarray
    local_extrema: np.ndarray
    regions: np.ndarray
    components: list[Component]
    centres: pd.DataFrame
    labels: np.ndarray


def _case(feature: ExtremaPointFeature, *, threshold: float, radius: int) -> Case:
    local_extrema, regions_mask = _reference_extrema(feature.field, feature.kind, threshold, radius)
    values = feature.field.to_numpy()
    centres, labels = _tabulate_by_chunks(feature)
    return Case(
        feature=feature,
        kind=feature.kind,
        radius=radius,
        field_values=values,
        local_extrema=local_extrema,
        regions=regions_mask,
        components=_brute_force_components(local_extrema, values),
        centres=centres,
        labels=labels,
    )


def _hand_case(layers: Sequence[np.ndarray], kind: ExtremaKind, *, time_chunk: int = 1) -> Case:
    return _case(
        _hand_feature(layers, kind, time_chunk=time_chunk), threshold=_sign(kind) * HAND_THRESHOLD, radius=HAND_RADIUS
    )


@pytest.fixture(scope="module", params=list(ExtremaKind), ids=[kind.value for kind in ExtremaKind])
def gaussian_case(request) -> Case:
    """The Gaussian lows (or, for MAXIMA, the same field negated: highs), tabulated chunk by chunk."""
    kind: ExtremaKind = request.param
    sign = _sign(kind)
    field = sign * _gaussian_lows()
    feature = ExtremaPointFeature(field, kind, threshold=sign * GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS)
    case = _case(feature, threshold=sign * GAUSSIAN_THRESHOLD, radius=GAUSSIAN_RADIUS)
    # Guard: the synthetic data has lows at every time step
    assert {component.time_index for component in case.components} == set(range(field.sizes["time"]))
    assert len(case.components) >= 2 * field.sizes["time"]
    return case


@pytest.fixture(scope="module")
def minima_feature() -> ExtremaPointFeature:
    return ExtremaPointFeature(
        _gaussian_lows(), ExtremaKind.MINIMA, threshold=GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS
    )


# ======================================================================================================================
# 3. Tests
# ======================================================================================================================


class TestExtremaPointFeatureInterface:
    def test_is_a_feature_source(self, minima_feature: ExtremaPointFeature) -> None:
        assert isinstance(minima_feature, FeatureSource)

    def test_geometry_is_point(self, minima_feature: ExtremaPointFeature) -> None:
        assert minima_feature.geometry is FeatureGeometry.POINT
        assert "geometry" not in {field.name for field in dataclasses.fields(ExtremaPointFeature)}

    def test_has_labels(self, minima_feature: ExtremaPointFeature) -> None:
        assert minima_feature.has_labels is True

    def test_fields_are_the_phase_1_parameters(self) -> None:
        # No steering flow (phase 5), no sel_region_indexer and no filter_kwargs
        assert {field.name for field in dataclasses.fields(ExtremaPointFeature)} == FEATURE_FIELDS

    def test_options_are_keyword_only(self) -> None:
        keyword_only = {field.name for field in dataclasses.fields(ExtremaPointFeature) if field.kw_only}
        assert keyword_only == KEYWORD_ONLY_FIELDS

    def test_positional_option_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MINIMA, 100_000.0)  # pyright: ignore[reportCallIssue]

    def test_defaults(self) -> None:
        feature = ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MAXIMA)

        assert feature.kind is ExtremaKind.MAXIMA
        assert feature.threshold is None
        assert feature.footprint_radius == 24  # noqa: PLR2004
        assert (feature.time_dim_name, feature.lat_dim_name, feature.lon_dim_name) == ("time", "latitude", "longitude")

    @pytest.mark.parametrize("name", sorted(FEATURE_FIELDS))
    def test_is_frozen(self, minima_feature: ExtremaPointFeature, name: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(minima_feature, name, None)

    def test_identity_equality_and_hashable_despite_data_array_field(self) -> None:
        # eq=False: a generated __eq__ would compare the DataArrays element-wise and __hash__ would hash them
        field = _gaussian_lows()
        feature = ExtremaPointFeature(field, ExtremaKind.MINIMA)
        same_parameters = ExtremaPointFeature(field, ExtremaKind.MINIMA)

        assert feature == feature  # noqa: PLR0124
        assert (feature == same_parameters) is False
        assert len({feature, same_parameters}) == 2  # noqa: PLR2004


class TestValidation:
    def test_wrong_dimension_names_raise(self) -> None:
        field = _gaussian_lows().rename({"latitude": "lat", "longitude": "lon"})

        with pytest.raises(ValueError, match="lat_dim_name"):
            ExtremaPointFeature(field, ExtremaKind.MINIMA)

    def test_extra_dimension_raises(self) -> None:
        field = _gaussian_lows().expand_dims(level=[250.0])

        with pytest.raises(ValueError, match="exactly the dimensions"):
            ExtremaPointFeature(field, ExtremaKind.MINIMA)

    @pytest.mark.parametrize("footprint_radius", [0, -1])
    def test_footprint_radius_below_one_raises(self, footprint_radius: int) -> None:
        with pytest.raises(ValueError, match="footprint_radius must be at least 1"):
            ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MINIMA, footprint_radius=footprint_radius)

    def test_field_in_another_dimension_order_gives_the_same_centres_and_labels(
        self, minima_feature: ExtremaPointFeature
    ) -> None:
        transposed = minima_feature.field.transpose("latitude", "longitude", "time")
        feature = ExtremaPointFeature(
            transposed, ExtremaKind.MINIMA, threshold=GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS
        )

        centres, labels = _tabulate_by_chunks(feature)
        expected_centres, expected_labels = _tabulate_by_chunks(minima_feature)

        pd.testing.assert_frame_equal(centres, expected_centres)
        np.testing.assert_array_equal(labels, expected_labels)


class TestChunkInputs:
    def test_building_inputs_and_tabulator_does_not_compute(self) -> None:
        field = _gaussian_lows()
        with _count_computes() as computes:
            feature = ExtremaPointFeature(
                field, ExtremaKind.MINIMA, threshold=GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS
            )
            inputs = feature.chunk_inputs()
            feature.tabulator()

        assert len(inputs) > 0
        assert computes == []

    def test_inputs_are_lazy_data_arrays(self, minima_feature: ExtremaPointFeature) -> None:
        for array in minima_feature.chunk_inputs().values():
            assert isinstance(array, xr.DataArray)
            assert is_dask_collection(array)

    def test_inputs_are_time_lat_lon_on_the_field_grid(self, minima_feature: ExtremaPointFeature) -> None:
        field = minima_feature.field
        for array in minima_feature.chunk_inputs().values():
            assert array.dims == ("time", "latitude", "longitude")
            assert array.shape == field.shape
            for dim_name in array.dims:
                np.testing.assert_array_equal(array[dim_name].to_numpy(), field[dim_name].to_numpy())

    def test_inputs_are_chunked_like_the_field_in_time_and_whole_in_space(
        self, minima_feature: ExtremaPointFeature
    ) -> None:
        time_chunks = minima_feature.field.chunksizes["time"]
        assert len(time_chunks) > 1
        for array in minima_feature.chunk_inputs().values():
            assert array.chunksizes["time"] == time_chunks
            assert len(array.chunksizes["latitude"]) == 1
            assert len(array.chunksizes["longitude"]) == 1

    def test_extrema_filter_runs_once_per_time_step_when_inputs_computed_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The inputs share the extrema filter; with their keys kept (rule 2), it runs once per time step
        field = _gaussian_lows()
        filter_calls: list[int] = []
        monkeypatch.setattr("scipy.ndimage.minimum_filter", _counting(ndi.minimum_filter, filter_calls))
        inputs = ExtremaPointFeature(
            field, ExtremaKind.MINIMA, threshold=GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS
        ).chunk_inputs()
        # Building may evaluate the filter on empty inputs (e.g. for the meta), which is not a run of a task
        filter_calls.clear()

        compute(*inputs.values(), scheduler="sync", optimize_graph=False)

        assert len(filter_calls) == field.sizes["time"]

    def test_custom_dimension_names(self, minima_feature: ExtremaPointFeature) -> None:
        renamed = minima_feature.field.rename({"time": "valid_time", "latitude": "lat", "longitude": "lon"})
        feature = ExtremaPointFeature(
            renamed,
            ExtremaKind.MINIMA,
            threshold=GAUSSIAN_THRESHOLD,
            footprint_radius=GAUSSIAN_RADIUS,
            time_dim_name="valid_time",
            lat_dim_name="lat",
            lon_dim_name="lon",
        )
        for array in feature.chunk_inputs().values():
            assert array.dims == ("valid_time", "lat", "lon")

        centres, labels = _tabulate_by_chunks(feature)
        expected_centres, expected_labels = _tabulate_by_chunks(minima_feature)

        pd.testing.assert_frame_equal(centres, expected_centres)
        np.testing.assert_array_equal(labels, expected_labels)


class TestTabulator:
    def test_is_a_frozen_dataclass_feature_tabulator(self, minima_feature: ExtremaPointFeature) -> None:
        tabulator = minima_feature.tabulator()

        assert isinstance(tabulator, FeatureTabulator)
        assert dataclasses.is_dataclass(tabulator)
        fields = dataclasses.fields(tabulator)
        assert len(fields) > 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(tabulator, fields[0].name, None)

    def test_holds_only_scalars_and_enums(self, minima_feature: ExtremaPointFeature) -> None:
        tabulator = minima_feature.tabulator()
        assert dataclasses.is_dataclass(tabulator)

        for field in dataclasses.fields(tabulator):
            value = getattr(tabulator, field.name)
            assert not is_dask_collection(value), field.name
            assert isinstance(value, bool | int | float | str | Enum | None), field.name

    def test_pickles_small_and_round_trips(self, minima_feature: ExtremaPointFeature) -> None:
        tabulator = minima_feature.tabulator()

        pickled = pickle.dumps(tabulator)
        restored = pickle.loads(pickled)

        assert len(pickled) < MAX_PICKLED_TABULATOR_BYTES
        assert restored == tabulator
        assert isinstance(restored, FeatureTabulator)

    def test_tokenize_is_deterministic_for_equal_instances(self) -> None:
        first = ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MINIMA, footprint_radius=GAUSSIAN_RADIUS)
        second = ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MINIMA, footprint_radius=GAUSSIAN_RADIUS)

        assert first.tabulator() == second.tabulator()
        assert tokenize(first.tabulator()) == tokenize(second.tabulator())

    @pytest.mark.parametrize(
        ("kind", "footprint_radius"),
        [(ExtremaKind.MAXIMA, GAUSSIAN_RADIUS), (ExtremaKind.MINIMA, GAUSSIAN_RADIUS + 1)],
        ids=["kind", "footprint_radius"],
    )
    def test_tokenize_differs_when_parameters_differ(self, kind: ExtremaKind, footprint_radius: int) -> None:
        reference = ExtremaPointFeature(_gaussian_lows(), ExtremaKind.MINIMA, footprint_radius=GAUSSIAN_RADIUS)
        other = ExtremaPointFeature(_gaussian_lows(), kind, footprint_radius=footprint_radius)

        assert tokenize(reference.tabulator()) != tokenize(other.tabulator())

    def test_restored_tabulator_gives_the_same_result(self) -> None:
        feature = _hand_feature(_tied_overlapping_lows(), ExtremaKind.MINIMA)
        blocks = {name: array.to_numpy() for name, array in feature.chunk_inputs().items()}
        chunk = ChunkInfo(
            times=feature.field["time"].to_numpy(),
            first_time_index=0,
            lats=feature.field["latitude"].to_numpy(),
            lons=feature.field["longitude"].to_numpy(),
        )
        restored = pickle.loads(pickle.dumps(feature.tabulator()))

        expected = feature.tabulator()(blocks, chunk)
        result = restored(blocks, chunk)

        pd.testing.assert_frame_equal(result.centres, expected.centres)
        np.testing.assert_array_equal(_labels_of(result), _labels_of(expected))


class TestCentres:
    def test_centres_are_connected_components_of_local_extrema(self, gaussian_case: Case) -> None:
        expected = {component.representative for component in gaussian_case.components}

        assert len(gaussian_case.centres) == len(expected)
        assert set(_ids_by_position(gaussian_case.centres)) == expected

    def test_has_the_centre_columns(self, gaussian_case: Case) -> None:
        assert set(CENTRE_COLUMNS) <= set(gaussian_case.centres.columns)

    def test_column_dtypes(self, gaussian_case: Case) -> None:
        centres = gaussian_case.centres
        for column in INTEGER_COLUMNS:
            assert pd.api.types.is_integer_dtype(centres[column]), column
        for column in FLOAT_COLUMNS:
            assert pd.api.types.is_float_dtype(centres[column]), column
        assert pd.api.types.is_datetime64_any_dtype(centres[Column.CENTRE_TIME])

    def test_feature_id_equals_centre_id(self, gaussian_case: Case) -> None:
        centres = gaussian_case.centres
        np.testing.assert_array_equal(centres[Column.FEATURE_ID].to_numpy(), centres[Column.CENTRE_ID].to_numpy())

    def test_time_coordinates_and_value_are_those_of_the_grid_at_the_centre(self, gaussian_case: Case) -> None:
        centres = gaussian_case.centres
        field = gaussian_case.feature.field
        time_index = centres[Column.TIME_INDEX].to_numpy()
        lat_index = centres[Column.CENTRE_LAT_INDEX].to_numpy()
        lon_index = centres[Column.CENTRE_LON_INDEX].to_numpy()

        np.testing.assert_array_equal(
            centres[Column.CENTRE_TIME].to_numpy().astype("datetime64[ns]"),
            field["time"].to_numpy().astype("datetime64[ns]")[time_index],
        )
        np.testing.assert_array_equal(
            centres[Column.CENTRE_LAT].to_numpy(dtype=float), field["latitude"].to_numpy()[lat_index]
        )
        np.testing.assert_array_equal(
            centres[Column.CENTRE_LON].to_numpy(dtype=float), field["longitude"].to_numpy()[lon_index]
        )
        np.testing.assert_array_equal(
            centres[Column.CENTRE_VALUE].to_numpy(dtype=float),
            gaussian_case.field_values[time_index, lat_index, lon_index],
        )

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_equal_valued_separate_extrema_are_two_centres(self, kind: ExtremaKind) -> None:
        # D7: the prototype grouped extremum pixels by (time, value), merging these two into one
        case = _hand_case(_equal_separate_lows(), kind)

        assert len(case.components) == 2  # noqa: PLR2004
        assert set(_ids_by_position(case.centres)) == {(0, 6, 8), (0, 14, 22)}
        assert case.centres[Column.CENTRE_ID].nunique() == 2  # noqa: PLR2004
        values = case.centres[Column.CENTRE_VALUE].to_numpy()
        assert values[0] == values[1]

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_flat_extremum_is_one_centre_at_its_first_pixel_in_row_major_order(self, kind: ExtremaKind) -> None:
        case = _hand_case(_plateau_low(), kind)
        # Guard: every pixel of the plateau is marked as an extremum
        assert int(case.local_extrema.sum()) == len(PLATEAU_PIXELS)

        assert len(case.centres) == 1
        assert set(_ids_by_position(case.centres)) == {(0, *min(PLATEAU_PIXELS))}
        field = case.feature.field
        assert case.centres[Column.CENTRE_LAT].tolist() == [field["latitude"].to_numpy()[PLATEAU_PIXELS[0][0]]]
        assert case.centres[Column.CENTRE_LON].tolist() == [field["longitude"].to_numpy()[PLATEAU_PIXELS[0][1]]]

    def test_diagonally_touching_extremum_pixels_are_one_centre(self) -> None:
        case = _hand_case(_diagonal_low(), ExtremaKind.MINIMA)
        assert int(case.local_extrema.sum()) == 2  # noqa: PLR2004

        assert set(_ids_by_position(case.centres)) == {(0, 5, 5)}

    @pytest.mark.parametrize("name", sorted(HAND_CASES))
    def test_hand_built_centres_match_brute_force(self, name: str) -> None:
        case = _hand_case(HAND_CASES[name](), ExtremaKind.MINIMA)

        assert set(_ids_by_position(case.centres)) == {component.representative for component in case.components}
        assert len(case.centres) == len(case.components)


class TestCentreIds:
    def test_ids_are_positive_and_unique(self, gaussian_case: Case) -> None:
        ids = gaussian_case.centres[Column.CENTRE_ID].to_numpy()

        assert np.all(ids >= 1)
        assert np.unique(ids).size == ids.size

    def test_ids_increase_in_row_major_time_lat_lon_order(self, gaussian_case: Case) -> None:
        # So that the smallest id is the first pixel in row-major order (D19)
        ordered = gaussian_case.centres.sort_values(
            [Column.TIME_INDEX, Column.CENTRE_LAT_INDEX, Column.CENTRE_LON_INDEX]
        )

        assert np.all(np.diff(ordered[Column.CENTRE_ID].to_numpy()) > 0)

    def test_centre_at_the_origin_has_a_positive_id_and_owns_its_cell(self) -> None:
        # (time, lat, lon) = (0, 0, 0) encodes to 0 without an offset, which would read as "no owner" in the labels
        case = _hand_case(_low_at_origin(), ExtremaKind.MINIMA)
        centre_id = _id_at(case.centres, 0, 0)

        assert centre_id >= 1
        assert case.labels[0, 0, 0] == centre_id

    def test_time_index_is_chunk_absolute(self) -> None:
        feature = _hand_feature(_stationary_and_moving_lows(), ExtremaKind.MINIMA, time_chunk=2)

        second_chunk = _tabulate(feature, 2, 4).centres

        assert sorted(set(second_chunk[Column.TIME_INDEX].tolist())) == [2, 3]
        expected_times = feature.field["time"].to_numpy().astype("datetime64[ns]")[2:4]
        assert set(second_chunk[Column.CENTRE_TIME].to_numpy().astype("datetime64[ns]").tolist()) == set(
            expected_times.tolist()
        )

    def test_ids_are_unique_across_chunks_for_a_low_that_does_not_move(self) -> None:
        feature = _hand_feature(_stationary_and_moving_lows(), ExtremaKind.MINIMA, time_chunk=2)

        centres, _ = _tabulate_by_chunks(feature)

        stationary = centres[(centres[Column.CENTRE_LAT_INDEX] == 8) & (centres[Column.CENTRE_LON_INDEX] == 10)]  # noqa: PLR2004
        assert len(stationary) == 4  # noqa: PLR2004
        assert centres[Column.CENTRE_ID].is_unique

    @pytest.mark.parametrize("which", ["gaussian", "stationary_and_moving"])
    def test_centres_and_labels_do_not_depend_on_chunking(self, which: str) -> None:
        if which == "gaussian":
            feature = ExtremaPointFeature(
                _gaussian_lows(), ExtremaKind.MINIMA, threshold=GAUSSIAN_THRESHOLD, footprint_radius=GAUSSIAN_RADIUS
            )
        else:
            feature = _hand_feature(_stationary_and_moving_lows(), ExtremaKind.MINIMA, time_chunk=2)

        by_chunks, labels_by_chunks = _tabulate_by_chunks(feature)
        whole, whole_labels = _tabulate_whole(feature)

        def ordered(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.sort_values(Column.CENTRE_ID).reset_index(drop=True)

        pd.testing.assert_frame_equal(ordered(by_chunks), ordered(whole))
        np.testing.assert_array_equal(labels_by_chunks, whole_labels)


class TestLabels:
    def test_shape_and_integer_dtype(self, gaussian_case: Case) -> None:
        assert gaussian_case.labels.shape == gaussian_case.field_values.shape
        assert np.issubdtype(gaussian_case.labels.dtype, np.integer)

    def test_labels_are_centre_ids_or_zero(self, gaussian_case: Case) -> None:
        ids = set(gaussian_case.centres[Column.CENTRE_ID].tolist())

        assert set(np.unique(gaussian_case.labels).tolist()) <= ids | {0}
        assert np.any(gaussian_case.labels == 0)
        assert np.any(gaussian_case.labels != 0)

    def test_labels_match_brute_force(self, gaussian_case: Case) -> None:
        expected = _brute_force_labels(
            gaussian_case.regions,
            gaussian_case.components,
            _ids_by_position(gaussian_case.centres),
            gaussian_case.kind,
            gaussian_case.radius,
        )

        assert np.any(expected != 0)
        np.testing.assert_array_equal(gaussian_case.labels, expected)

    def test_labelled_cells_are_a_subset_of_extrema_regions(self, gaussian_case: Case) -> None:
        assert not np.any((gaussian_case.labels != 0) & ~gaussian_case.regions)

    def test_labelled_cells_are_the_extrema_mask_region(self, gaussian_case: Case) -> None:
        # With generic values, every region cell is within the footprint of the extremum that defines it, so every
        # region cell is owned. Checked against regions.__extrema_mask_region on the filter computed here with scipy
        kind = gaussian_case.kind
        extrema_filter = ndi.minimum_filter if kind is ExtremaKind.MINIMA else ndi.maximum_filter
        threshold = _sign(kind) * GAUSSIAN_THRESHOLD
        for time_index, values in enumerate(gaussian_case.field_values):
            filtered = extrema_filter(values, footprint=circular_footprint(GAUSSIAN_RADIUS))
            is_extremum = (values == filtered) & (
                filtered < threshold if kind is ExtremaKind.MINIMA else filtered > threshold
            )
            expected = _extrema_mask_region(is_extremum, filtered)

            np.testing.assert_array_equal(gaussian_case.labels[time_index] != 0, expected)

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    @pytest.mark.parametrize("name", sorted(HAND_CASES))
    def test_hand_built_labels_match_brute_force(self, name: str, kind: ExtremaKind) -> None:
        case = _hand_case(HAND_CASES[name](), kind)

        expected = _brute_force_labels(case.regions, case.components, _ids_by_position(case.centres), kind, HAND_RADIUS)

        assert np.any(expected != 0)
        np.testing.assert_array_equal(case.labels, expected)
        assert not np.any((case.labels != 0) & ~case.regions)

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_equal_valued_separate_extrema_each_own_their_footprint(self, kind: ExtremaKind) -> None:
        case = _hand_case(_equal_separate_lows(), kind)
        first = _id_at(case.centres, 6, 8)
        second = _id_at(case.centres, 14, 22)

        expected = first * _within([(6, 8)], HAND_RADIUS) + second * _within([(14, 22)], HAND_RADIUS)

        assert first != second
        np.testing.assert_array_equal(case.labels[0], expected)

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_flat_extremum_footprint_is_measured_from_every_pixel(self, kind: ExtremaKind) -> None:
        case = _hand_case(_plateau_low(), kind)
        centre_id = _id_at(case.centres, *PLATEAU_PIXELS[0])
        # Guard: these cells are within the footprint of the far pixels of the plateau but not of the representative
        beyond_representative = [(14, 14), (10, 18)]
        for cell in beyond_representative:
            assert not _within([PLATEAU_PIXELS[0]], HAND_RADIUS)[cell]
            assert _within(PLATEAU_PIXELS, HAND_RADIUS)[cell]

        for cell in beyond_representative:
            assert case.labels[0][cell] == centre_id
        np.testing.assert_array_equal(case.labels[0], centre_id * _within(PLATEAU_PIXELS, HAND_RADIUS))

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_tie_between_equal_extrema_goes_to_the_smaller_id(self, kind: ExtremaKind) -> None:
        case = _hand_case(_tied_overlapping_lows(), kind)
        left = _id_at(case.centres, 10, 10)
        right = _id_at(case.centres, 10, 15)
        near_left = _within([(10, 10)], HAND_RADIUS)
        near_right = _within([(10, 15)], HAND_RADIUS)
        shared = near_left & near_right
        assert np.any(shared)

        labels = case.labels[0]

        assert np.all(labels[shared] == min(left, right))
        assert np.all(labels[near_left & ~near_right] == left)
        assert np.all(labels[near_right & ~near_left] == right)
        assert not np.any(labels[~near_left & ~near_right])

    @pytest.mark.parametrize("kind", list(ExtremaKind))
    def test_deeper_extremum_owns_the_cells_shared_with_a_shallower_one(self, kind: ExtremaKind) -> None:
        case = _hand_case(_deeper_and_shallower_lows(), kind)
        deeper = _id_at(case.centres, 10, 10)
        shallower = _id_at(case.centres, 10, 16)
        shared = _within([(10, 10)], HAND_RADIUS) & _within([(10, 16)], HAND_RADIUS) & case.regions[0]
        # Guard: there are shared region cells (e.g. half-way between the two)
        assert shared[10, 13]

        labels = case.labels[0]

        assert np.all(labels[shared] == deeper)
        assert labels[10, 20] == shallower


class TestNoExtrema:
    def test_empty_field_gives_empty_centres_with_the_centre_columns(self) -> None:
        feature = _hand_feature([np.zeros(HAND_SHAPE), np.zeros(HAND_SHAPE)], ExtremaKind.MINIMA, time_chunk=2)

        result = _tabulate(feature, 0, 2)

        assert len(result.centres) == 0
        assert set(CENTRE_COLUMNS) <= set(result.centres.columns)
        labels = _labels_of(result)
        assert labels.shape == (2, *HAND_SHAPE)
        assert np.issubdtype(labels.dtype, np.integer)
        assert not np.any(labels)

    def test_chunk_without_extrema_has_the_dtypes_of_a_chunk_with_extrema(self) -> None:
        # So that per-chunk tables concatenate without changing dtype
        feature = _hand_feature(_lows_only_in_second_chunk(), ExtremaKind.MINIMA, time_chunk=2)

        empty = _tabulate(feature, 0, 2)
        non_empty = _tabulate(feature, 2, 4)

        assert len(empty.centres) == 0
        assert len(non_empty.centres) == 2  # noqa: PLR2004
        for column in CENTRE_COLUMNS:
            assert empty.centres[column].dtype == non_empty.centres[column].dtype, column
        assert not np.any(_labels_of(empty))
        assert _labels_of(empty).dtype == _labels_of(non_empty).dtype
