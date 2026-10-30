"""Tests for the feature-relative km frame of :mod:`rojak.atmosphere.feature_composite` (``_frame``, phase 1).

Every pair is checked against a direct, per-pair WGS-84 orthographic projection with :mod:`pyproj`, independently of
how ``_frame`` groups pairs by centre.
"""

import numpy as np
import pandas as pd
import pyproj
import pytest

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._frame import project_pairs

LAT: str = "obs_latitude"
LON: str = "obs_longitude"
OBS_ID: str = "obs_id"
TOLERANCE_KM: float = 1e-6

# (lat, lon) of centres: mid-latitude, equator and prime meridian, southern ocean, next to the dateline, near the pole
CENTRES: tuple[tuple[float, float], ...] = ((50.0, -30.0), (0.0, 0.0), (-60.0, 150.0), (10.0, 179.5), (80.0, -170.0))


def _wrap_longitude(lons: np.ndarray) -> np.ndarray:
    return (lons + 180.0) % 360.0 - 180.0


def _pairs(centres: tuple[tuple[float, float], ...], n_per_centre: int = 20) -> pd.DataFrame:
    """Observations scattered within 20 degrees of each centre, rows of different centres interleaved."""
    rng = np.random.default_rng(42)
    frames = [
        pd.DataFrame(
            {
                Column.CENTRE_ID: centre_id,
                Column.CENTRE_LAT: centre_lat,
                Column.CENTRE_LON: centre_lon,
                LAT: np.clip(centre_lat + rng.uniform(-20.0, 20.0, n_per_centre), -89.0, 89.0),
                LON: _wrap_longitude(centre_lon + rng.uniform(-20.0, 20.0, n_per_centre)),
            }
        )
        for centre_id, (centre_lat, centre_lon) in enumerate(centres, start=1)
    ]
    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs.iloc[rng.permutation(len(pairs))].reset_index(drop=True)
    return pairs.assign(**{OBS_ID: np.arange(len(pairs))})


def _expected_xy(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Brute force: a fresh projection per pair."""
    xy = [
        pyproj.Proj(proj="ortho", lat_0=centre_lat, lon_0=centre_lon, units="km", ellps="WGS84")(lon, lat)
        for centre_lat, centre_lon, lat, lon in zip(
            pairs[Column.CENTRE_LAT], pairs[Column.CENTRE_LON], pairs[LAT], pairs[LON], strict=True
        )
    ]
    return np.array([x for x, _ in xy], dtype=float), np.array([y for _, y in xy], dtype=float)


def _single_centre_pairs(centre: tuple[float, float], points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Column.CENTRE_ID: 1,
            Column.CENTRE_LAT: centre[0],
            Column.CENTRE_LON: centre[1],
            LAT: [lat for lat, _ in points],
            LON: [lon for _, lon in points],
            OBS_ID: np.arange(len(points)),
        }
    )


class TestProjectPairs:
    def test_matches_pyproj_per_pair_for_several_centres(self) -> None:
        pairs = _pairs(CENTRES)
        expected_x, expected_y = _expected_xy(pairs)

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON).sort_values(OBS_ID)

        # Every observation is within 20 degrees of its centre, so none is on the far side
        assert result[OBS_ID].tolist() == pairs[OBS_ID].tolist()
        assert np.allclose(result[Column.X_KM].to_numpy(), expected_x, rtol=0, atol=TOLERANCE_KM)
        assert np.allclose(result[Column.Y_KM].to_numpy(), expected_y, rtol=0, atol=TOLERANCE_KM)

    @pytest.mark.parametrize(("centre_lat", "centre_lon"), CENTRES)
    def test_point_at_centre_maps_to_origin(self, centre_lat: float, centre_lon: float) -> None:
        pairs = _single_centre_pairs((centre_lat, centre_lon), [(centre_lat, centre_lon)])

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON)

        assert len(result) == 1
        assert abs(result[Column.X_KM].item()) < TOLERANCE_KM
        assert abs(result[Column.Y_KM].item()) < TOLERANCE_KM

    def test_north_and_east_are_positive_y_and_x(self) -> None:
        # Hand-checkable orientation of the frame: +x east, +y north
        pairs = _single_centre_pairs((0.0, 0.0), [(1.0, 0.0), (0.0, 1.0)])

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON).sort_values(OBS_ID)

        (north_x, east_x), (north_y, east_y) = result[Column.X_KM].to_numpy(), result[Column.Y_KM].to_numpy()
        assert north_y > 0
        assert abs(north_x) < TOLERANCE_KM
        assert east_x > 0
        assert abs(east_y) < TOLERANCE_KM

    def test_far_side_points_are_dropped(self) -> None:
        # More than 90 degrees of arc from the centre at (0, 0): behind the horizon of the orthographic projection
        near = [(0.0, 30.0), (45.0, -45.0), (-60.0, 10.0)]
        far = [(0.0, 150.0), (0.0, 180.0), (-20.0, -170.0), (10.0, 100.0)]
        pairs = _single_centre_pairs((0.0, 0.0), near + far)

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON)

        assert sorted(result[OBS_ID].tolist()) == list(range(len(near)))
        assert np.isfinite(result[Column.X_KM].to_numpy()).all()
        assert np.isfinite(result[Column.Y_KM].to_numpy()).all()

    def test_longitude_convention_of_observations_does_not_matter(self) -> None:
        pairs = _pairs(CENTRES)
        shifted = pairs.assign(**{LON: pairs[LON] % 360.0})

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON).sort_values(OBS_ID)
        from_shifted = project_pairs(shifted, lat_column=LAT, lon_column=LON).sort_values(OBS_ID)

        assert np.allclose(
            result[Column.X_KM].to_numpy(), from_shifted[Column.X_KM].to_numpy(), rtol=0, atol=TOLERANCE_KM
        )
        assert np.allclose(
            result[Column.Y_KM].to_numpy(), from_shifted[Column.Y_KM].to_numpy(), rtol=0, atol=TOLERANCE_KM
        )

    def test_keeps_other_columns(self) -> None:
        pairs = _pairs(CENTRES).assign(is_event=lambda frame: frame[OBS_ID] % 2 == 0)

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON).sort_values(OBS_ID)

        assert set(pairs.columns) | {Column.X_KM, Column.Y_KM} == set(result.columns)
        for column in pairs.columns:
            assert result[column].tolist() == pairs[column].tolist()

    def test_does_not_modify_input(self) -> None:
        pairs = _pairs(CENTRES)
        original = pairs.copy()

        project_pairs(pairs, lat_column=LAT, lon_column=LON)

        pd.testing.assert_frame_equal(pairs, original)

    def test_empty_table_returns_empty_table_with_new_columns(self) -> None:
        pairs = _single_centre_pairs((0.0, 0.0), [])

        result = project_pairs(pairs, lat_column=LAT, lon_column=LON)

        assert len(result) == 0
        assert set(pairs.columns) | {Column.X_KM, Column.Y_KM} == set(result.columns)
