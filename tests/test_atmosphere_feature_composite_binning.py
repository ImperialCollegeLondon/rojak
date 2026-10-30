"""Tests for the bins of :mod:`rojak.atmosphere.feature_composite` (``_binning``, phase 1, step D): ``BinSpec.grid``
and ``count_pairs``.

The km grid is checked against the one :mod:`rojak.atmosphere.compositing` builds for gridded composites
(``np.arange(-max_km_extent, max_km_extent + 1, grid_km_spacing)``), and the counts against :func:`numpy.histogramdd`
and against hand-computed cells, following numpy's convention: every bin is half-open ``[left, right)`` except the
last, which also includes its right edge.
"""

import dataclasses
import inspect
import pickle
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from rojak.atmosphere.compositing import _project_data_about_extrema
from rojak.atmosphere.feature_composite._binning import BinSpec, count_pairs
from rojak.atmosphere.feature_composite._columns import Column

DIMS: tuple[str, str] = ("y_km", "x_km")
DERIVED_PROPERTIES: tuple[str, ...] = ("dims", "coords", "edges", "shape")
# (max_km_extent, spacing_km): typical composites, a single cell either side of the centre, and spacings that are not
# exact in binary floating point
GRID_CASES: tuple[tuple[float, float], ...] = (
    (1500.0, 50.0),
    (2000.0, 25.0),
    (100.0, 100.0),
    (10.0, 2.5),
    (7.7, 1.1),
)


def _compositing_coords(max_km_extent: float, spacing_km: float) -> np.ndarray:
    """The km coordinates of :mod:`rojak.atmosphere.compositing` (see ``_project_data_about_extrema``)."""
    return np.arange(-max_km_extent, max_km_extent + 1, spacing_km)


def _expected_edges(coords: np.ndarray, spacing_km: float) -> np.ndarray:
    """Midpoints between the coordinates, extended by half a spacing beyond the outermost ones."""
    midpoints = (coords[:-1] + coords[1:]) / 2
    return np.concatenate([[coords[0] - spacing_km / 2], midpoints, [coords[-1] + spacing_km / 2]])


def _pairs(y_km: np.ndarray, x_km: np.ndarray, index: np.ndarray | None = None) -> pd.DataFrame:
    """A pair table with the positions, x before y and another column, so only the names can select them."""
    return pd.DataFrame(
        {
            Column.CENTRE_ID: np.arange(y_km.size, dtype=np.int64),
            Column.X_KM: x_km,
            "is_event": np.arange(y_km.size) % 2 == 0,
            Column.Y_KM: y_km,
        },
        index=index,
    )


class TestBinSpecGrid:
    @pytest.mark.parametrize(("max_km_extent", "spacing_km"), GRID_CASES)
    def test_coords_match_compositing_grid(self, max_km_extent: float, spacing_km: float) -> None:
        expected = _compositing_coords(max_km_extent, spacing_km)

        bins = BinSpec.grid(max_km_extent, spacing_km)

        for dim in DIMS:
            assert bins.coords[dim].shape == expected.shape
            assert np.allclose(bins.coords[dim], expected, rtol=0, atol=1e-9)

    @pytest.mark.parametrize(("max_km_extent", "spacing_km"), GRID_CASES)
    def test_shape_matches_gridded_composite(self, max_km_extent: float, spacing_km: float) -> None:
        # The gridded composite of a slice without a centre is a NaN array of the shape of its km grid
        gridded = _project_data_about_extrema(
            np.zeros((2, 2), dtype=np.int8),
            np.zeros((2, 2)),
            lats=np.array([0.0, 1.0]),
            lons=np.array([0.0, 1.0]),
            max_km_extent=max_km_extent,
            grid_km_spacing=spacing_km,
        )

        assert BinSpec.grid(max_km_extent, spacing_km).shape == gridded.shape

    def test_hand_computed_coords_and_edges(self) -> None:
        bins = BinSpec.grid(100.0, 50.0)

        for dim in DIMS:
            assert bins.coords[dim].tolist() == [-100.0, -50.0, 0.0, 50.0, 100.0]
            assert bins.edges[dim].tolist() == [-125.0, -75.0, -25.0, 25.0, 75.0, 125.0]
        assert bins.shape == (5, 5)

    @pytest.mark.parametrize(("max_km_extent", "spacing_km"), GRID_CASES)
    def test_edges_are_midpoints_extended_by_half_a_spacing(self, max_km_extent: float, spacing_km: float) -> None:
        expected = _expected_edges(_compositing_coords(max_km_extent, spacing_km), spacing_km)

        bins = BinSpec.grid(max_km_extent, spacing_km)

        for dim in DIMS:
            assert bins.edges[dim].shape == expected.shape
            assert np.allclose(bins.edges[dim], expected, rtol=0, atol=1e-9)

    def test_dims_are_y_then_x(self) -> None:
        bins = BinSpec.grid(1500.0, 50.0)

        assert bins.dims == DIMS
        assert bins.dims == (Column.Y_KM, Column.X_KM)

    def test_coords_and_edges_are_mappings_by_dim(self) -> None:
        bins = BinSpec.grid(1500.0, 50.0)

        assert isinstance(bins.coords, Mapping)
        assert isinstance(bins.edges, Mapping)
        assert set(bins.coords) == set(DIMS)
        assert set(bins.edges) == set(DIMS)

    @pytest.mark.parametrize(("max_km_extent", "spacing_km"), GRID_CASES)
    def test_shape_is_number_of_coords_per_dim(self, max_km_extent: float, spacing_km: float) -> None:
        bins = BinSpec.grid(max_km_extent, spacing_km)

        assert bins.shape == tuple(bins.coords[dim].size for dim in bins.dims)
        assert bins.shape == tuple(bins.edges[dim].size - 1 for dim in bins.dims)

    @pytest.mark.parametrize("name", DERIVED_PROPERTIES)
    def test_derived_values_are_properties_not_fields(self, name: str) -> None:
        assert isinstance(inspect.getattr_static(BinSpec, name), property)
        assert name not in {field.name for field in dataclasses.fields(BinSpec)}


class TestBinSpecValidation:
    @pytest.mark.parametrize(
        ("max_km_extent", "spacing_km"),
        [
            (1500.0, 0.0),
            (1500.0, -50.0),
            (0.0, 50.0),
            (-1500.0, 50.0),
            (1500.0, 70.0),
            (100.0, 200.0),
            (100.0, 30.0),
        ],
        ids=[
            "zero-spacing",
            "negative-spacing",
            "zero-extent",
            "negative-extent",
            "extent-not-multiple",
            "spacing-beyond-extent",
            "extent-not-multiple-small",
        ],
    )
    def test_invalid_grid_raises(self, max_km_extent: float, spacing_km: float) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 - the message is free; the type is the contract
            BinSpec.grid(max_km_extent, spacing_km)

    @pytest.mark.parametrize(("max_km_extent", "spacing_km"), [(7.7, 1.1), (6.3, 2.1), (0.3, 0.1)])
    def test_multiple_within_float_tolerance_is_accepted(self, max_km_extent: float, spacing_km: float) -> None:
        # e.g. 7.7 / 1.1 == 7.000000000000001 and 7.7 % 1.1 is about 1.1, so an exact test would wrongly reject these
        BinSpec.grid(max_km_extent, spacing_km)


class TestBinSpecIsAParameterObject:
    def test_is_a_frozen_dataclass(self) -> None:
        bins = BinSpec.grid(1500.0, 50.0)
        names = [field.name for field in dataclasses.fields(bins)]

        assert names
        for name in names:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(bins, name, None)

    def test_pickles_and_round_trips(self) -> None:
        bins = BinSpec.grid(1500.0, 50.0)

        restored: BinSpec = pickle.loads(pickle.dumps(bins))

        assert type(restored) is BinSpec
        assert restored.dims == bins.dims
        assert restored.shape == bins.shape
        for dim in DIMS:
            assert np.array_equal(restored.coords[dim], bins.coords[dim])
            assert np.array_equal(restored.edges[dim], bins.edges[dim])


class TestCountPairs:
    def test_matches_histogramdd_including_points_on_edges(self) -> None:
        max_km_extent, spacing_km = 1500.0, 50.0
        edges = _expected_edges(_compositing_coords(max_km_extent, spacing_km), spacing_km)
        rng = np.random.default_rng(42)
        n_random = 2000
        # Random points, some beyond the outermost edges, then points exactly on every edge in y, in x and in both,
        # including both outermost edges (numpy keeps the right-most one in the last bin)
        y_km = np.concatenate([rng.uniform(-1.3, 1.3, n_random) * max_km_extent, edges, np.zeros(edges.size), edges])
        x_km = np.concatenate([rng.uniform(-1.3, 1.3, n_random) * max_km_extent, np.zeros(edges.size), edges, edges])
        # A shuffled, non-default index, as left by filtering a pair table
        pairs = _pairs(y_km, x_km, index=rng.permutation(y_km.size) + 1000)
        expected, _ = np.histogramdd(np.column_stack([y_km, x_km]), bins=(edges, edges))

        counts = count_pairs(pairs, BinSpec.grid(max_km_extent, spacing_km))

        assert 0 < expected.sum() < y_km.size
        assert counts.shape == expected.shape
        assert np.array_equal(counts, expected)

    def test_hand_computed_cells(self) -> None:
        # Coordinates -100, -50, 0, 50, 100 and edges -125, -75, -25, 25, 75, 125; y is axis 0, x is axis 1
        points = [
            ((0.0, 0.0), (2, 2)),
            ((50.0, 0.0), (3, 2)),  # north of the centre: y bin 3, x bin 2
            ((0.0, -50.0), (2, 1)),  # west of the centre
            ((25.0, 0.0), (3, 2)),  # on an interior edge: the upper bin
            ((-25.0, -25.0), (2, 2)),
            ((-125.0, -125.0), (0, 0)),  # on the lowest edge: included
            ((125.0, 125.0), (4, 4)),  # on the highest edge: included in the last bin
            ((124.9, -124.9), (4, 0)),
        ]
        dropped = [(125.1, 0.0), (0.0, -125.1), (-200.0, 200.0), (130.0, 130.0)]
        y_km = np.array([y for (y, _), _ in points] + [y for y, _ in dropped])
        x_km = np.array([x for (_, x), _ in points] + [x for _, x in dropped])
        expected = np.zeros((5, 5), dtype=np.int64)
        for _, cell in points:
            expected[cell] += 1

        counts = count_pairs(_pairs(y_km, x_km), BinSpec.grid(100.0, 50.0))

        assert np.array_equal(counts, expected)

    def test_counts_are_integers(self) -> None:
        counts = count_pairs(_pairs(np.array([0.0, 10.0]), np.array([0.0, -10.0])), BinSpec.grid(100.0, 50.0))

        assert np.issubdtype(counts.dtype, np.integer)

    def test_empty_pairs_give_zeros(self) -> None:
        bins = BinSpec.grid(1500.0, 50.0)

        counts = count_pairs(_pairs(np.array([], dtype=float), np.array([], dtype=float)), bins)

        assert counts.shape == bins.shape
        assert not counts.any()

    def test_does_not_modify_pairs(self) -> None:
        rng = np.random.default_rng(42)
        pairs = _pairs(rng.uniform(-200.0, 200.0, 50), rng.uniform(-200.0, 200.0, 50))
        original = pairs.copy()

        count_pairs(pairs, BinSpec.grid(100.0, 50.0))

        pd.testing.assert_frame_equal(pairs, original)
