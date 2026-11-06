"""Tests for the control designs of :mod:`rojak.atmosphere.feature_composite` (``_controls``, phase 1).

Phase 1 covers observation shifts only: no field offsets, exclusion or analysis/observation periods (phase 3).
Valid controls are checked against a brute-force double loop over centres and designs; shifts against hand-written
times.
"""

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._controls import (
    ControlDesign,
    shift_observations,
    valid_controls,
    validate_controls,
)

TIME: str = "obs_time"
LAT: str = "obs_latitude"
LON: str = "obs_longitude"

FIELD_START: np.datetime64 = np.datetime64("2024-01-08T00:00")
FIELD_END: np.datetime64 = np.datetime64("2024-01-22T00:00")

DESIGNS: tuple[ControlDesign, ...] = (
    ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D")),
    ControlDesign("obs+7D", observation_offset=np.timedelta64(7, "D")),
    ControlDesign("obs-6h", observation_offset=np.timedelta64(-6, "h")),
    ControlDesign("obs+6h", observation_offset=np.timedelta64(6, "h")),
    ControlDesign("obs0", observation_offset=np.timedelta64(0, "h")),
)


def _centres(times: list[np.datetime64] | np.ndarray) -> pd.DataFrame:
    """Centre table with the columns a feature tabulator provides (only ID and time matter for validity)."""
    n_centres = len(times)
    return pd.DataFrame(
        {
            Column.CENTRE_ID: np.arange(100, 100 + n_centres, dtype=np.int64),
            Column.FEATURE_ID: np.arange(100, 100 + n_centres, dtype=np.int64),
            Column.CENTRE_TIME: pd.to_datetime(np.asarray(times, dtype="datetime64[ns]")),
            Column.CENTRE_LAT: np.linspace(30.0, 60.0, n_centres),
            Column.CENTRE_LON: np.linspace(-40.0, 20.0, n_centres),
        }
    )


def _brute_force_valid(
    centres: pd.DataFrame,
    designs: tuple[ControlDesign, ...],
    field_time_range: tuple[np.datetime64, np.datetime64],
) -> set[tuple[int, str]]:
    start, end = field_time_range
    expected: set[tuple[int, str]] = set()
    for centre_id, centre_time in zip(centres[Column.CENTRE_ID], centres[Column.CENTRE_TIME], strict=True):
        for design in designs:
            control_time = np.datetime64(centre_time) + design.observation_offset
            if start <= control_time <= end:
                expected.add((int(centre_id), design.label))
    return expected


def _rows(result: pd.DataFrame) -> list[tuple[int, str]]:
    return [
        (int(centre_id), str(label))
        for centre_id, label in zip(result[Column.CENTRE_ID], result[Column.CONTROL], strict=True)
    ]


def _observations(times: list[str]) -> pd.DataFrame:
    n_obs = len(times)
    return pd.DataFrame(
        {
            TIME: pd.to_datetime(times),
            LAT: np.linspace(40.0, 50.0, n_obs),
            LON: np.linspace(-10.0, 10.0, n_obs),
            "is_event": np.arange(n_obs) % 2 == 0,
        }
    )


class TestControlDesign:
    def test_stores_label_and_offset(self) -> None:
        design = ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D"))

        assert design.label == "obs-7D"
        assert design.observation_offset == np.timedelta64(-7, "D")

    def test_zero_offset_is_allowed(self) -> None:
        design = ControlDesign("same-time", observation_offset=np.timedelta64(0, "h"))

        assert design.observation_offset == np.timedelta64(0, "h")

    def test_empty_label_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="label"):
            ControlDesign("", observation_offset=np.timedelta64(7, "D"))

    @pytest.mark.parametrize(
        "offset",
        [
            pytest.param(7, id="int"),
            pytest.param(pd.Timedelta(7, "D"), id="pandas-timedelta"),
            pytest.param("7D", id="string"),
            pytest.param(np.datetime64("2024-01-01"), id="datetime64"),
        ],
    )
    def test_offset_that_is_not_timedelta64_raises_type_error(self, offset: object) -> None:
        with pytest.raises(TypeError):
            ControlDesign("bad", observation_offset=offset)  # pyright: ignore[reportArgumentType]

    def test_offset_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            ControlDesign("positional", np.timedelta64(7, "D"))  # pyright: ignore[reportCallIssue]

    def test_is_frozen(self) -> None:
        design = ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D"))

        with pytest.raises(dataclasses.FrozenInstanceError):
            design.label = "changed"  # pyright: ignore[reportAttributeAccessIssue]

    def test_round_trips_through_pickle(self) -> None:
        design = ControlDesign("obs+6h", observation_offset=np.timedelta64(6, "h"))

        restored = pickle.loads(pickle.dumps(design))

        assert restored == design
        assert hash(restored) == hash(design)


class TestValidateControls:
    def test_distinct_labels_pass(self) -> None:
        validate_controls(DESIGNS)

    def test_empty_sequence_passes(self) -> None:
        validate_controls([])

    def test_duplicate_labels_raise_value_error(self) -> None:
        controls = [
            ControlDesign("obs", observation_offset=np.timedelta64(-7, "D")),
            ControlDesign("other", observation_offset=np.timedelta64(6, "h")),
            ControlDesign("obs", observation_offset=np.timedelta64(7, "D")),
        ]

        with pytest.raises(ValueError, match="obs"):
            validate_controls(controls)


class TestShiftObservations:
    @pytest.mark.parametrize(
        ("offset", "observed", "expected"),
        [
            # An observation made at t + offset is moved back to t
            (np.timedelta64(7, "D"), "2024-01-15T06:00", "2024-01-08T06:00"),
            (np.timedelta64(-7, "D"), "2024-01-15T06:00", "2024-01-22T06:00"),
            (np.timedelta64(6, "h"), "2024-01-15T06:00", "2024-01-15T00:00"),
            (np.timedelta64(-6, "h"), "2024-01-15T06:00", "2024-01-15T12:00"),
            (np.timedelta64(0, "h"), "2024-01-15T06:00", "2024-01-15T06:00"),
            (np.timedelta64(7, "D"), "2024-03-03T23:30", "2024-02-25T23:30"),  # across the leap day
        ],
    )
    def test_moves_times_back_by_observation_offset(self, offset: np.timedelta64, observed: str, expected: str) -> None:
        design = ControlDesign("ctl", observation_offset=offset)

        result = shift_observations(_observations([observed]), design, time_column=TIME)

        assert result[TIME].tolist() == [pd.Timestamp(expected)]

    @pytest.mark.parametrize("design", DESIGNS, ids=[design.label for design in DESIGNS])
    def test_observation_at_centre_time_plus_offset_lines_up_with_centre(self, design: ControlDesign) -> None:
        centre_times = pd.to_datetime(["2024-01-10T00:00", "2024-01-15T06:00", "2024-01-20T18:00"])
        observed_times = [str(time) for time in centre_times + pd.Timedelta(design.observation_offset)]

        result = shift_observations(_observations(observed_times), design, time_column=TIME)

        assert result[TIME].tolist() == centre_times.tolist()

    def test_other_columns_are_unchanged(self) -> None:
        observations = _observations(["2024-01-10T00:00", "2024-01-11T03:00", "2024-01-12T09:00"])
        design = ControlDesign("obs+7D", observation_offset=np.timedelta64(7, "D"))

        result = shift_observations(observations, design, time_column=TIME)

        assert list(result.columns) == list(observations.columns)
        for column in (LAT, LON, "is_event"):
            assert result[column].tolist() == observations[column].tolist()

    def test_does_not_modify_input(self) -> None:
        observations = _observations(["2024-01-10T00:00", "2024-01-11T03:00"])
        original = observations.copy()
        design = ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D"))

        result = shift_observations(observations, design, time_column=TIME)

        pd.testing.assert_frame_equal(observations, original)
        assert result is not observations

    def test_empty_observations_give_empty_result(self) -> None:
        observations = _observations([])
        design = ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D"))

        result = shift_observations(observations, design, time_column=TIME)

        assert len(result) == 0
        assert list(result.columns) == list(observations.columns)


class TestValidControls:
    def test_matches_brute_force_over_centres_and_offsets(self) -> None:
        # Centres every 6 h from a week before to a week after the field, so every design is valid for some centres
        # and invalid for others, and several land exactly on the boundaries
        times = pd.date_range("2024-01-01T00:00", "2024-01-29T00:00", freq="6h").to_numpy()
        centres = _centres(times)
        field_time_range = (FIELD_START, FIELD_END)

        result = valid_controls(centres, DESIGNS, field_time_range=field_time_range)

        rows = _rows(result)
        assert len(rows) == len(set(rows))
        assert set(rows) == _brute_force_valid(centres, DESIGNS, field_time_range)

    @pytest.mark.parametrize("design", DESIGNS, ids=[design.label for design in DESIGNS])
    def test_boundaries_are_inclusive(self, design: ControlDesign) -> None:
        one_hour = np.timedelta64(1, "h")
        # Centre times whose control time is: just before the start, at the start, at the end, just after the end
        times = [
            FIELD_START - design.observation_offset - one_hour,
            FIELD_START - design.observation_offset,
            FIELD_END - design.observation_offset,
            FIELD_END - design.observation_offset + one_hour,
        ]
        centres = _centres(times)
        ids = centres[Column.CENTRE_ID].tolist()

        result = valid_controls(centres, [design], field_time_range=(FIELD_START, FIELD_END))

        assert sorted(_rows(result)) == [(ids[1], design.label), (ids[2], design.label)]

    def test_offset_is_added_to_centre_time(self) -> None:
        # Hand-checked: a centre one day before the end of the field has a valid +6 h control but not a +7 D one,
        # and a centre one day after the start has a valid -6 h control but not a -7 D one
        centres = _centres([np.datetime64("2024-01-21T00:00"), np.datetime64("2024-01-09T00:00")])
        late, early = centres[Column.CENTRE_ID].tolist()

        result = valid_controls(centres, DESIGNS, field_time_range=(FIELD_START, FIELD_END))

        assert set(_rows(result)) == {
            (late, "obs-7D"),
            (late, "obs-6h"),
            (late, "obs+6h"),
            (late, "obs0"),
            (early, "obs+7D"),
            (early, "obs-6h"),
            (early, "obs+6h"),
            (early, "obs0"),
        }

    def test_has_centre_id_and_control_columns(self) -> None:
        centres = _centres([np.datetime64("2024-01-15T00:00")])

        result = valid_controls(centres, DESIGNS, field_time_range=(FIELD_START, FIELD_END))

        assert set(result.columns) == {Column.CENTRE_ID, Column.CONTROL}
        assert len(result) == len(DESIGNS)

    def test_empty_centres_give_empty_table_with_columns(self) -> None:
        centres = _centres([])

        result = valid_controls(centres, DESIGNS, field_time_range=(FIELD_START, FIELD_END))

        assert len(result) == 0
        assert set(result.columns) == {Column.CENTRE_ID, Column.CONTROL}

    def test_no_designs_give_empty_table_with_columns(self) -> None:
        centres = _centres([np.datetime64("2024-01-15T00:00")])

        result = valid_controls(centres, [], field_time_range=(FIELD_START, FIELD_END))

        assert len(result) == 0
        assert set(result.columns) == {Column.CENTRE_ID, Column.CONTROL}

    def test_centres_outside_every_window_give_empty_table(self) -> None:
        centres = _centres([np.datetime64("2023-06-01T00:00"), np.datetime64("2025-06-01T00:00")])

        result = valid_controls(centres, DESIGNS, field_time_range=(FIELD_START, FIELD_END))

        assert len(result) == 0
        assert set(result.columns) == {Column.CENTRE_ID, Column.CONTROL}

    def test_does_not_modify_centres(self) -> None:
        centres = _centres([np.datetime64("2024-01-09T00:00"), np.datetime64("2024-01-21T00:00")])
        original = centres.copy()

        valid_controls(centres, DESIGNS, field_time_range=(FIELD_START, FIELD_END))

        pd.testing.assert_frame_equal(centres, original)
