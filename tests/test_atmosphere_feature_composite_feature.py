"""Tests for the column contract and the feature interface of :mod:`rojak.atmosphere.feature_composite`.

Covers ``_columns`` (the ``Column`` contract of the per-chunk tables) and ``_feature`` (the ``FeatureSource`` and
``FeatureTabulator`` extension points, ``FeatureGeometry``, ``FeatureChunk`` and ``ChunkInfo``); phase 1, step B of
``storm_composite_plan.md``.
"""

import dataclasses
import pickle
from collections.abc import Mapping
from enum import StrEnum

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._feature import (
    ChunkInfo,
    FeatureChunk,
    FeatureGeometry,
    FeatureSource,
    FeatureTabulator,
)

# Members the per-chunk tables rely on in phase 1 (section 3.1 of the plan)
PHASE_1_COLUMNS: tuple[str, ...] = (
    "CENTRE_ID",
    "FEATURE_ID",
    "CENTRE_TIME",
    "CENTRE_LAT",
    "CENTRE_LON",
    "CENTRE_VALUE",
    "CENTRE_LAT_INDEX",
    "CENTRE_LON_INDEX",
    "TIME_INDEX",
    "DISTANCE_KM",
    "X_KM",
    "Y_KM",
    "CONTROL",
)
# A parameters-only tabulator pickles to a few hundred bytes; an embedded array or collection would not
MAX_PICKLED_TABULATOR_BYTES: int = 1_000


# --------------------------------------------------------------------------------------------------------------------
# A minimal, complete feature written against the interface, as an extender would (section 6: feature-agnostic test).
# Module level so that it pickles.
# --------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ValuesBelowThreshold(FeatureTabulator):
    """One centre per grid cell of ``blocks["values"]`` whose value is below ``threshold``."""

    threshold: float

    def __call__(self, blocks: Mapping[str, np.ndarray], chunk: ChunkInfo) -> FeatureChunk:
        values = blocks["values"]
        n_lat, n_lon = values.shape[1:]
        local_time, lat_index, lon_index = np.nonzero(values < self.threshold)
        time_index = chunk.first_time_index + local_time
        centre_id = (time_index * n_lat + lat_index) * n_lon + lon_index
        centres = pd.DataFrame(
            {
                Column.CENTRE_ID: centre_id,
                Column.FEATURE_ID: centre_id,
                Column.TIME_INDEX: time_index,
                Column.CENTRE_TIME: chunk.times[local_time],
                Column.CENTRE_LAT: chunk.lats[lat_index],
                Column.CENTRE_LON: chunk.lons[lon_index],
                Column.CENTRE_LAT_INDEX: lat_index,
                Column.CENTRE_LON_INDEX: lon_index,
                Column.CENTRE_VALUE: values[local_time, lat_index, lon_index],
            }
        )
        return FeatureChunk(centres=centres, labels=None)


@dataclasses.dataclass(frozen=True)
class FieldBelowThreshold(FeatureSource):
    """Graph side of :class:`ValuesBelowThreshold`: holds the lazy field."""

    values: xr.DataArray
    threshold: float

    @property
    def geometry(self) -> FeatureGeometry:
        return FeatureGeometry.POINT

    def chunk_inputs(self) -> Mapping[str, xr.DataArray]:
        return {"values": self.values}

    def tabulator(self) -> FeatureTabulator:
        return ValuesBelowThreshold(self.threshold)


@dataclasses.dataclass(frozen=True)
class LabelledFieldBelowThreshold(FieldBelowThreshold):
    """A feature that declares it fills ``FeatureChunk.labels`` (D18)."""

    @property
    def has_labels(self) -> bool:
        return True


def _values_field() -> xr.DataArray:
    data = np.random.default_rng(42).random((4, 3, 5))
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.arange("2005-02-01T00", "2005-02-02T00", 6, dtype="datetime64[h]"),
            "latitude": np.array([60.0, 50.0, 40.0]),
            "longitude": np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),
        },
    )


def _chunk_info() -> ChunkInfo:
    return ChunkInfo(
        times=np.array(["2005-02-01T12", "2005-02-01T18"], dtype="datetime64[ns]"),
        first_time_index=2,
        lats=np.array([60.0, 50.0]),
        lons=np.array([-20.0, -10.0, 0.0]),
    )


def _blocks() -> dict[str, np.ndarray]:
    values = np.ones((2, 2, 3))
    values[0, 1, 2] = -1.0
    values[1, 0, 0] = -2.0
    return {"values": values}


# Members that make a subclass complete; dropping any one must leave it abstract
_SOURCE_MEMBERS: dict[str, object] = {
    "geometry": property(lambda self: FeatureGeometry.POINT),
    "chunk_inputs": lambda self: {},
    "tabulator": lambda self: ValuesBelowThreshold(0.0),
}
_TABULATOR_MEMBERS: dict[str, object] = {
    "__call__": lambda self, blocks, chunk: FeatureChunk(centres=pd.DataFrame(), labels=None),
}


def _subclass(base: type, members: dict[str, object], *, without: str | None = None) -> type:
    namespace = {name: member for name, member in members.items() if name != without}
    return type(f"{base.__name__}Subclass", (base,), namespace)


# --------------------------------------------------------------------------------------------------------------------
# _columns
# --------------------------------------------------------------------------------------------------------------------


class TestColumn:
    def test_is_a_str_enum(self) -> None:
        assert issubclass(Column, StrEnum)
        assert all(isinstance(member, str) for member in Column)

    def test_has_the_phase_1_members(self) -> None:
        assert set(PHASE_1_COLUMNS) <= set(Column.__members__)

    def test_has_no_orientation_before_phase_5(self) -> None:
        assert "ORIENTATION" not in Column.__members__

    def test_values_are_unique(self) -> None:
        # Members sharing a value become aliases and vanish from iteration, so compare against __members__
        values = [member.value for member in Column.__members__.values()]
        assert len(values) == len(set(values))

    def test_values_are_non_empty(self) -> None:
        assert all(member.value for member in Column)

    def test_usable_as_dataframe_column_names(self) -> None:
        frame = pd.DataFrame({Column.CENTRE_ID: [1, 2], Column.X_KM: [0.5, -0.5]})

        assert frame[Column.X_KM].tolist() == [0.5, -0.5]
        # Selecting by the plain string is the same column: users reading returned pair tables need no enum
        assert frame[str(Column.CENTRE_ID)].tolist() == [1, 2]
        assert list(frame.columns) == [str(Column.CENTRE_ID), str(Column.X_KM)]

    def test_column_names_survive_merge_on_member(self) -> None:
        left = pd.DataFrame({Column.CENTRE_ID: [1, 2], Column.X_KM: [0.5, -0.5]})
        right = pd.DataFrame({str(Column.CENTRE_ID): [2, 1], str(Column.Y_KM): [3.0, 4.0]})

        merged = left.merge(right, on=Column.CENTRE_ID).sort_values(Column.CENTRE_ID)

        assert merged[Column.Y_KM].tolist() == [4.0, 3.0]


# --------------------------------------------------------------------------------------------------------------------
# _feature
# --------------------------------------------------------------------------------------------------------------------


class TestFeatureGeometry:
    def test_is_a_str_enum_with_point_and_line(self) -> None:
        assert issubclass(FeatureGeometry, StrEnum)
        assert set(FeatureGeometry.__members__) == {"POINT", "LINE"}

    def test_values_are_distinct(self) -> None:
        assert FeatureGeometry.POINT != FeatureGeometry.LINE


class TestFeatureSourceIsAbstract:
    def test_abstract_members_are_geometry_chunk_inputs_and_tabulator(self) -> None:
        # has_labels is concrete (D18); with_time_offset is deferred to phase 3, so nothing has to implement it now
        assert FeatureSource.__abstractmethods__ == frozenset(_SOURCE_MEMBERS)

    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            FeatureSource()  # pyright: ignore[reportAbstractUsage]

    @pytest.mark.parametrize("missing", sorted(_SOURCE_MEMBERS))
    def test_subclass_missing_a_member_raises_type_error(self, missing: str) -> None:
        incomplete = _subclass(FeatureSource, _SOURCE_MEMBERS, without=missing)

        with pytest.raises(TypeError, match=missing):
            incomplete()

    def test_subclass_with_every_member_instantiates(self) -> None:
        # Guards the parametrised test above: the TypeError there comes from the missing member alone
        complete = _subclass(FeatureSource, _SOURCE_MEMBERS)()

        assert isinstance(complete, FeatureSource)
        assert complete.has_labels is False


class TestFeatureTabulatorIsAbstract:
    def test_abstract_members_are_call(self) -> None:
        assert FeatureTabulator.__abstractmethods__ == frozenset(_TABULATOR_MEMBERS)

    def test_base_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            FeatureTabulator()  # pyright: ignore[reportAbstractUsage]

    @pytest.mark.parametrize("missing", sorted(_TABULATOR_MEMBERS))
    def test_subclass_missing_a_member_raises_type_error(self, missing: str) -> None:
        incomplete = _subclass(FeatureTabulator, _TABULATOR_MEMBERS, without=missing)

        with pytest.raises(TypeError, match=missing):
            incomplete()

    def test_subclass_with_every_member_instantiates(self) -> None:
        complete = _subclass(FeatureTabulator, _TABULATOR_MEMBERS)()

        assert isinstance(complete, FeatureTabulator)


class TestConcreteFeatureSource:
    def test_frozen_dataclass_subclass_instantiates(self) -> None:
        source = FieldBelowThreshold(_values_field(), threshold=0.5)

        assert isinstance(source, FeatureSource)

    def test_concrete_property_satisfies_abstract_geometry(self) -> None:
        source = FieldBelowThreshold(_values_field(), threshold=0.5)

        assert source.geometry is FeatureGeometry.POINT
        assert "geometry" not in FieldBelowThreshold.__abstractmethods__

    def test_geometry_is_not_a_dataclass_field(self) -> None:
        # A property the feature declares, not a constructor argument: the user doesn't choose the geometry
        assert "geometry" not in {field.name for field in dataclasses.fields(FieldBelowThreshold)}

    def test_has_labels_defaults_to_false(self) -> None:
        assert FieldBelowThreshold(_values_field(), threshold=0.5).has_labels is False

    def test_has_labels_can_be_overridden(self) -> None:
        assert LabelledFieldBelowThreshold(_values_field(), threshold=0.5).has_labels is True

    def test_tabulator_is_a_feature_tabulator_carrying_the_parameters(self) -> None:
        tabulator = FieldBelowThreshold(_values_field(), threshold=0.5).tabulator()

        assert isinstance(tabulator, FeatureTabulator)
        assert tabulator == ValuesBelowThreshold(0.5)

    def test_chunk_inputs_are_the_lazy_fields(self) -> None:
        field = _values_field()

        inputs = FieldBelowThreshold(field, threshold=0.5).chunk_inputs()

        assert set(inputs) == {"values"}
        assert inputs["values"] is field


class TestConcreteFeatureTabulator:
    def test_returns_feature_chunk_with_expected_centres(self) -> None:
        chunk = _chunk_info()

        result = ValuesBelowThreshold(0.0)(_blocks(), chunk)

        assert isinstance(result, FeatureChunk)
        assert result.labels is None
        # Hand-computed from _blocks(): (t=0, lat=1, lon=2) and (t=1, lat=0, lon=0), chunk starts at time index 2
        centres = result.centres.sort_values(Column.TIME_INDEX)
        assert centres[Column.TIME_INDEX].tolist() == [2, 3]
        assert centres[Column.CENTRE_LAT].tolist() == [50.0, 60.0]
        assert centres[Column.CENTRE_LON].tolist() == [0.0, -20.0]
        assert centres[Column.CENTRE_VALUE].tolist() == [-1.0, -2.0]
        assert centres[Column.CENTRE_TIME].tolist() == list(chunk.times)

    def test_pickles_and_round_trips(self) -> None:
        tabulator = ValuesBelowThreshold(0.0)

        restored = pickle.loads(pickle.dumps(tabulator))

        assert restored == tabulator
        assert isinstance(restored, FeatureTabulator)
        pd.testing.assert_frame_equal(
            restored(_blocks(), _chunk_info()).centres, tabulator(_blocks(), _chunk_info()).centres
        )

    def test_pickled_size_is_small(self) -> None:
        # Rule 5: tabulators hold parameters only, so they cost almost nothing to send to a task
        assert len(pickle.dumps(ValuesBelowThreshold(0.0))) < MAX_PICKLED_TABULATOR_BYTES


class TestFeatureChunk:
    def test_fields_are_centres_and_labels(self) -> None:
        assert [field.name for field in dataclasses.fields(FeatureChunk)] == ["centres", "labels"]

    def test_labels_may_be_none(self) -> None:
        chunk = FeatureChunk(centres=pd.DataFrame(), labels=None)

        assert chunk.labels is None

    def test_labels_may_be_an_array(self) -> None:
        labels = np.zeros((2, 3, 4), dtype=np.int64)

        chunk = FeatureChunk(centres=pd.DataFrame(), labels=labels)

        assert chunk.labels is labels

    @pytest.mark.parametrize("name", ["centres", "labels"])
    def test_is_frozen(self, name: str) -> None:
        chunk = FeatureChunk(centres=pd.DataFrame(), labels=None)

        with pytest.raises(dataclasses.FrozenInstanceError):
            # setattr, not attribute assignment, so pyright doesn't reject the deliberate write to a frozen field
            setattr(chunk, name, None)


class TestChunkInfo:
    def test_fields(self) -> None:
        assert {field.name for field in dataclasses.fields(ChunkInfo)} == {"times", "first_time_index", "lats", "lons"}

    def test_holds_coordinates_by_reference(self) -> None:
        lats = np.array([60.0, 50.0])
        lons = np.array([-20.0, -10.0, 0.0])

        info = ChunkInfo(times=np.array([], dtype="datetime64[ns]"), first_time_index=0, lats=lats, lons=lons)

        assert info.lats is lats
        assert info.lons is lons

    @pytest.mark.parametrize("name", ["times", "first_time_index", "lats", "lons"])
    def test_is_frozen(self, name: str) -> None:
        info = _chunk_info()

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(info, name, None)
