"""Tests for the spatio-temporal DBSCAN in rojak.datalib.iata._clustering."""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from rojak.core.constants import EARTH_AVG_RADIUS
from rojak.datalib.iata._clustering import st_dbscan

BASE = np.datetime64("2024-06-01T00:00:00", "s")


def _seconds(values: ArrayLike) -> NDArray[np.datetime64]:
    """Timestamps offset from BASE by the given number of seconds."""
    return BASE + np.round(np.asarray(values, dtype=float)).astype(np.int64).astype("timedelta64[s]")


def _minutes(values: ArrayLike) -> NDArray[np.datetime64]:
    """Timestamps offset from BASE by the given number of minutes."""
    return _seconds(np.asarray(values, dtype=float) * 60.0)


def _pairwise_km(lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.float64]:
    """Great-circle distance in km between every pair of points, by the haversine formula."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_AVG_RADIUS * np.arcsin(np.sqrt(a)) / 1000.0


def _neighbour_matrix(
    time: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    alt: NDArray[np.float64],
    max_km: float,
    max_minutes: float,
    max_alt: float | None,
) -> NDArray[np.bool_]:
    """Symmetric neighbour relation, including each point's relation to itself."""
    gap_minutes = np.abs((time[:, None] - time[None, :]) / np.timedelta64(1, "m"))
    near = (_pairwise_km(lat, lon) <= max_km) & (gap_minutes <= max_minutes)
    if max_alt is not None:
        near &= np.abs(alt[:, None] - alt[None, :]) <= max_alt
    return near


def _cluster(
    time: NDArray[np.datetime64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    alt: NDArray[np.float64],
    *,
    n_points: int,
    max_km: float,
    max_minutes: float,
    max_alt: float | None,
) -> NDArray[np.int64]:
    """Call st_dbscan with altitude passed only when the altitude threshold is set."""
    return st_dbscan(
        time,
        lat,
        lon,
        alt if max_alt is not None else None,
        n_points=n_points,
        max_separation_km=max_km,
        max_minutes=max_minutes,
        max_altitude_delta=max_alt,
        not_in_cluster_label=-1,
    )


class TestHandBuiltClusters:
    def test_isolated_point_is_noise(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.zeros(3),
            np.array([0.0, 0.01, 50.0]),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]

    def test_two_separated_groups_get_separate_ids(self) -> None:
        lon = np.array([-0.06, -0.05, -0.04, 0.04, 0.05, 0.06])
        labels = st_dbscan(
            _minutes(np.zeros(6)),
            np.zeros(6),
            lon,
            None,
            n_points=3,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]

    @pytest.mark.parametrize(
        ("lon", "expected"),
        [
            # Left group first in the rows: left is id 0, right is id 1.
            (
                [-0.07, -0.06, -0.05, -0.04, 0.04, 0.05, 0.06, 0.07, 0.0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0],
            ),
            # Right group first in the rows: right is id 0, left is id 1.
            # The border point is still adjacent to both, so it takes id 0.
            (
                [0.04, 0.05, 0.06, 0.07, -0.07, -0.06, -0.05, -0.04, 0.0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0],
            ),
        ],
    )
    def test_border_point_between_two_clusters_takes_lowest_id(self, lon: list[float], expected: list[int]) -> None:
        # Each group has four members, so a core point needs three neighbours besides itself.
        # The border point at lon 0.0 is about 4.5 km from one member of each group and about 5.6 km
        # from the next nearest, so it has three neighbours including itself and is not core.
        labels = st_dbscan(
            _minutes(np.zeros(9)),
            np.zeros(9),
            np.array(lon),
            None,
            n_points=4,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    def test_pair_meets_n_points_two(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.zeros(2),
            np.array([0.0, 0.01]),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0]

    def test_pair_below_n_points_three_is_noise(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.zeros(2),
            np.array([0.0, 0.01]),
            None,
            n_points=3,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [-1, -1]

    @pytest.mark.parametrize(("n_points", "expected"), [(3, [0, 0, 0]), (4, [-1, -1, -1])])
    def test_identical_coordinates_cluster_together(self, n_points: int, expected: list[int]) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.full(3, 52.0),
            np.full(3, 4.0),
            None,
            n_points=n_points,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    @pytest.mark.parametrize(("max_km", "expected"), [(15.0, [0, 0]), (5.0, [-1, -1])])
    def test_antimeridian_pair_is_neighbours_across_the_dateline(self, max_km: float, expected: list[int]) -> None:
        # 0.1 degrees of longitude at the equator is about 11.1 km, across the 180 degree meridian.
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.zeros(2),
            np.array([179.95, -179.95]),
            None,
            n_points=2,
            max_separation_km=max_km,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    @pytest.mark.parametrize(("max_km", "expected"), [(23.0, [0, 0]), (22.0, [-1, -1])])
    def test_polar_pair_separated_through_the_pole(self, max_km: float, expected: list[int]) -> None:
        # Points 0.1 degrees from the pole on opposite sides of it are about 22.24 km apart.
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.array([89.9, 89.9]),
            np.array([0.0, 180.0]),
            None,
            n_points=2,
            max_separation_km=max_km,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    def test_points_at_the_pole_are_neighbours_whatever_their_longitude(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.array([90.0, 90.0]),
            np.array([0.0, 45.0]),
            None,
            n_points=2,
            max_separation_km=0.001,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0]

    @pytest.mark.parametrize(
        ("alt", "max_alt", "expected"),
        [
            # Altitude steps by 2000 m between rows 1 and 2, which exceeds the 500 m delta.
            (np.array([10000.0, 10000.0, 12000.0, 12000.0, 12000.0]), 500.0, [-1, -1, 0, 0, 0]),
            (np.array([10000.0, 10000.0, 12000.0, 12000.0, 12000.0]), 2500.0, [0, 0, 0, 0, 0]),
            (None, None, [0, 0, 0, 0, 0]),
        ],
    )
    def test_altitude_threshold_splits_an_otherwise_connected_cluster(
        self, alt: NDArray[np.float64] | None, max_alt: float | None, expected: list[int]
    ) -> None:
        # Spacing is about 0.56 km, so every pair is within 5 km and only altitude separates the points.
        labels = st_dbscan(
            _minutes(np.zeros(5)),
            np.zeros(5),
            np.array([0.0, 0.005, 0.010, 0.015, 0.020]),
            alt,
            n_points=3,
            max_separation_km=5.0,
            max_minutes=10.0,
            max_altitude_delta=max_alt,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    @pytest.mark.parametrize(
        ("max_minutes", "expected"),
        [(0.3, [-1, -1, -1]), (0.5, [0, 0, -1]), (0.7, [0, 0, 0])],
    )
    def test_non_integer_max_minutes_is_respected(self, max_minutes: float, expected: list[int]) -> None:
        # Gaps are 24 s (0.4 min), 36 s (0.6 min) and 60 s (1.0 min) between consecutive rows.
        labels = st_dbscan(
            _seconds([0.0, 24.0, 60.0]),
            np.zeros(3),
            np.zeros(3),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=max_minutes,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == expected

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_any_datetime_resolution_is_accepted(self, unit: str) -> None:
        time = np.array(["2024-06-01", "2024-06-01", "2024-06-03"], dtype=f"datetime64[{unit}]")
        labels = st_dbscan(
            time,
            np.zeros(3),
            np.zeros(3),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]

    @pytest.mark.parametrize("max_km", [20100.0, 1e9, np.inf])
    def test_separation_beyond_half_the_circumference_reaches_antipodal_points(self, max_km: float) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(2)),
            np.zeros(2),
            np.array([0.0, 180.0]),
            None,
            n_points=2,
            max_separation_km=max_km,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0]

    def test_unbounded_time_window_links_points_far_apart_in_time(self) -> None:
        labels = st_dbscan(
            _minutes([0.0, 1e6]),
            np.zeros(2),
            np.zeros(2),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=np.inf,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0]

    def test_integer_valued_float_coordinates_are_accepted(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.array([1.0, 1.0, 40.0]),
            np.array([2.0, 2.0, 3.0]),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]

    def test_n_points_one_makes_every_point_core(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.zeros(3),
            np.array([0.0, 0.01, 50.0]),
            None,
            n_points=1,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, 1]

    def test_n_points_one_with_no_neighbours_gives_one_cluster_per_point(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.zeros(3),
            np.array([0.0, 50.0, 100.0]),
            None,
            n_points=1,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 1, 2]

    @pytest.mark.parametrize("alt", [None, np.empty(0)])
    def test_empty_input_returns_empty_int64_array(self, alt: NDArray[np.float64] | None) -> None:
        labels = st_dbscan(
            _minutes(np.empty(0)),
            np.empty(0),
            np.empty(0),
            alt,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            max_altitude_delta=None if alt is None else 100.0,
            not_in_cluster_label=-1,
        )
        assert labels.dtype == np.int64
        assert labels.shape == (0,)


class TestThresholdEdges:
    def test_zero_distance_and_window_keep_only_coincident_points(self) -> None:
        # The third point is about 0.11 m north of the first two, so it is outside a zero-distance threshold.
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.array([0.0, 0.0, 1e-6]),
            np.zeros(3),
            None,
            n_points=2,
            max_separation_km=0.0,
            max_minutes=0.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]

    def test_zero_time_window_keeps_only_identical_timestamps(self) -> None:
        labels = st_dbscan(
            _minutes([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.array([0.0, 0.0, 0.005]),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=0.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]

    def test_zero_altitude_delta_keeps_only_equal_altitudes(self) -> None:
        labels = st_dbscan(
            _minutes(np.zeros(3)),
            np.zeros(3),
            np.zeros(3),
            np.array([100.0, 100.0, 101.0]),
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            max_altitude_delta=0.0,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == [0, 0, -1]


class TestRowOrder:
    def test_cluster_numbering_follows_the_first_core_row(self) -> None:
        left = [0.00, 0.01, 0.02]
        right = [10.00, 10.01, 10.02]
        labels_left_first = st_dbscan(
            _minutes(np.zeros(6)),
            np.zeros(6),
            np.array(left + right),
            None,
            n_points=3,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        labels_right_first = st_dbscan(
            _minutes(np.zeros(6)),
            np.zeros(6),
            np.array(right + left),
            None,
            n_points=3,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=-1,
        )
        assert labels_left_first.tolist() == [0, 0, 0, 1, 1, 1]
        # The right-hand group now occupies the first rows, so it receives id 0.
        assert labels_right_first.tolist() == [0, 0, 0, 1, 1, 1]


class TestInputValidation:
    @pytest.mark.parametrize(
        ("n_time", "n_lat", "n_lon", "n_alt"),
        [(2, 3, 3, 3), (3, 2, 3, 3), (3, 3, 2, 3), (3, 3, 3, 2)],
    )
    def test_raises_when_array_lengths_differ(self, n_time: int, n_lat: int, n_lon: int, n_alt: int) -> None:
        with pytest.raises(ValueError, match="same length"):
            st_dbscan(
                _minutes(np.zeros(n_time)),
                np.zeros(n_lat),
                np.zeros(n_lon),
                np.zeros(n_alt),
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                max_altitude_delta=500.0,
                not_in_cluster_label=-1,
            )

    def test_raises_when_inputs_are_not_one_dimensional(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            st_dbscan(
                _minutes(np.zeros((2, 2))),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                None,
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                not_in_cluster_label=-1,
            )

    @pytest.mark.parametrize("n_points", [0, -1])
    def test_raises_when_n_points_below_one(self, n_points: int) -> None:
        with pytest.raises(ValueError, match="n_points"):
            st_dbscan(
                _minutes(np.zeros(3)),
                np.zeros(3),
                np.zeros(3),
                None,
                n_points=n_points,
                max_separation_km=10.0,
                max_minutes=10.0,
                not_in_cluster_label=-1,
            )

    @pytest.mark.parametrize("label", [0, 1])
    def test_raises_when_not_in_cluster_label_is_not_negative(self, label: int) -> None:
        # A non-negative label could be mistaken for a cluster id
        with pytest.raises(ValueError, match="not_in_cluster_label"):
            st_dbscan(
                _minutes(np.zeros(3)),
                np.zeros(3),
                np.zeros(3),
                None,
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                not_in_cluster_label=label,
            )

    @pytest.mark.parametrize("label", [-1, -7])
    def test_noise_takes_the_given_label(self, label: int) -> None:
        labels = st_dbscan(
            _minutes(np.array([0.0, 1.0, 600.0])),
            np.zeros(3),
            np.array([0.0, 0.01, 0.0]),
            None,
            n_points=2,
            max_separation_km=5.0,
            max_minutes=10.0,
            not_in_cluster_label=label,
        )
        assert labels.tolist() == [0, 0, label]

    @pytest.mark.parametrize(
        ("max_km", "max_minutes", "max_alt"),
        [
            (-1.0, 10.0, None),
            (10.0, -1.0, None),
            (10.0, 10.0, -1.0),
            (np.nan, 10.0, None),
            (10.0, np.nan, None),
            (10.0, 10.0, np.nan),
        ],
    )
    def test_raises_on_negative_threshold(self, max_km: float, max_minutes: float, max_alt: float | None) -> None:
        with pytest.raises(ValueError, match="negative"):
            _cluster(
                _minutes(np.zeros(3)),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                n_points=2,
                max_km=max_km,
                max_minutes=max_minutes,
                max_alt=max_alt,
            )

    def test_raises_when_altitude_given_without_delta(self) -> None:
        with pytest.raises(ValueError, match="altitude"):
            st_dbscan(
                _minutes(np.zeros(3)),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                max_altitude_delta=None,
                not_in_cluster_label=-1,
            )

    def test_raises_when_delta_given_without_altitude(self) -> None:
        with pytest.raises(ValueError, match="altitude"):
            st_dbscan(
                _minutes(np.zeros(3)),
                np.zeros(3),
                np.zeros(3),
                None,
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                max_altitude_delta=500.0,
                not_in_cluster_label=-1,
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("field", ["lat", "lon", "alt"])
    def test_raises_on_non_finite_lat_lon_or_altitude(self, field: str, bad: float) -> None:
        lat = np.zeros(3)
        lon = np.zeros(3)
        alt = np.zeros(3)
        target = {"lat": lat, "lon": lon, "alt": alt}[field]
        target[1] = bad
        with pytest.raises(ValueError, match="finite"):
            st_dbscan(
                _minutes(np.zeros(3)),
                lat,
                lon,
                alt,
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                max_altitude_delta=500.0,
                not_in_cluster_label=-1,
            )

    def test_raises_on_nat_timestamp(self) -> None:
        time = _minutes(np.zeros(3))
        time[1] = np.datetime64("NaT", "s")
        with pytest.raises(ValueError, match="NaT"):
            st_dbscan(
                time,
                np.zeros(3),
                np.zeros(3),
                None,
                n_points=2,
                max_separation_km=10.0,
                max_minutes=10.0,
                not_in_cluster_label=-1,
            )


# Regression data: fixed inputs with their expected labels. Coordinates are rounded to two decimals, altitudes are
# whole metres and times are whole minutes, so the inputs are exact. No pairwise distance, time gap or altitude gap
# lies within 1e-6 of its threshold, so the expected labels do not depend on floating-point rounding.
_A_LAT = np.array(
    [
        51.74,
        50.57,
        51.21,
        51.56,
        51.43,
        51.83,
        51.72,
        51.84,
        50.05,
        50.87,
        50.97,
        50.13,
        50.01,
        51.66,
        51.97,
        51.57,
        50.63,
        51.41,
        50.6,
        51.48,
        50.56,
        51.57,
        51.98,
        51.97,
        51.77,
        51.83,
        51.42,
        51.11,
        51.85,
        50.18,
    ],
    dtype=float,
)
_A_LON = np.array(
    [
        -0.26,
        0.21,
        -0.04,
        0.59,
        -0.61,
        -0.88,
        -0.75,
        0.93,
        -0.73,
        -0.55,
        -0.25,
        0.25,
        0.56,
        -0.84,
        0.86,
        0.53,
        -0.36,
        -0.62,
        0.19,
        0.19,
        -0.47,
        -0.66,
        0.53,
        0.35,
        0.6,
        -0.56,
        0.45,
        -0.85,
        -0.99,
        -0.64,
    ],
    dtype=float,
)
_A_MINUTES = np.array(
    [
        138,
        187,
        63,
        129,
        153,
        148,
        62,
        142,
        55,
        152,
        68,
        77,
        229,
        172,
        33,
        3,
        22,
        90,
        214,
        13,
        27,
        30,
        62,
        217,
        75,
        144,
        3,
        189,
        134,
        145,
    ],
    dtype=float,
)
_A_ALT = np.array(
    [
        11819,
        10287,
        8286,
        11179,
        10197,
        10492,
        8180,
        10041,
        11550,
        10226,
        9655,
        10275,
        8156,
        9368,
        11418,
        10671,
        9114,
        9627,
        8269,
        10618,
        11159,
        9939,
        10528,
        10993,
        11177,
        10772,
        10793,
        8157,
        9308,
        11128,
    ],
    dtype=float,
)
_A_LABELS_N2_NO_ALT: list[int] = [
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    -1,
    -1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]
_A_LABELS_N3_ALT: list[int] = [
    0,
    -1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    -1,
    -1,
    0,
    0,
    0,
    0,
    0,
    -1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    -1,
    0,
    1,
]
_A_LABELS_N10_NO_ALT: list[int] = [
    0,
    -1,
    0,
    0,
    0,
    0,
    0,
    -1,
    -1,
    0,
    0,
    -1,
    -1,
    0,
    -1,
    0,
    -1,
    0,
    -1,
    0,
    -1,
    0,
    -1,
    0,
    0,
    0,
    0,
    0,
    0,
    -1,
]
_A_LABELS_N10_ALT: list[int] = [
    -1,
    -1,
    0,
    -1,
    0,
    0,
    0,
    -1,
    -1,
    -1,
    0,
    -1,
    -1,
    0,
    -1,
    -1,
    -1,
    0,
    -1,
    0,
    -1,
    0,
    -1,
    -1,
    -1,
    0,
    -1,
    -1,
    0,
    -1,
]

_B_LAT = np.array(
    [
        52.83,
        51.53,
        52.93,
        50.24,
        51.82,
        51.13,
        52.41,
        50.52,
        52.61,
        51.63,
        52.71,
        51.43,
        51.29,
        52.37,
        52.95,
        51.11,
        52.91,
        52.79,
        50.53,
        51.83,
        52.11,
        52.83,
        52.0,
        50.4,
    ],
    dtype=float,
)
_B_LON = np.array(
    [
        -0.01,
        -0.03,
        0.0,
        1.83,
        -0.6,
        -1.1,
        0.09,
        0.56,
        1.76,
        0.33,
        -0.93,
        1.72,
        -0.03,
        0.7,
        -0.1,
        -1.13,
        0.77,
        1.08,
        -1.24,
        -0.16,
        -0.55,
        -1.32,
        -1.11,
        1.85,
    ],
    dtype=float,
)
_B_MINUTES = np.array(
    [15, 106, 4, 45, 63, 88, 20, 5, 35, 109, 39, 64, 8, 98, 114, 33, 102, 45, 36, 41, 18, 116, 11, 51], dtype=float
)
_B_LABELS: list[int] = [0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, -1]


class TestFixedInputRegression:
    @pytest.mark.parametrize(
        ("n_points", "use_altitude", "expected"),
        [
            (2, False, _A_LABELS_N2_NO_ALT),
            (3, True, _A_LABELS_N3_ALT),
            (10, False, _A_LABELS_N10_NO_ALT),
            (10, True, _A_LABELS_N10_ALT),
        ],
    )
    def test_labels_match_expected_for_fixed_inputs(
        self, n_points: int, use_altitude: bool, expected: list[int]
    ) -> None:
        labels = st_dbscan(
            _minutes(_A_MINUTES),
            _A_LAT,
            _A_LON,
            _A_ALT if use_altitude else None,
            n_points=n_points,
            max_separation_km=60.0,
            max_minutes=90.5,
            max_altitude_delta=1500.5 if use_altitude else None,
            not_in_cluster_label=-1,
        )
        assert labels.dtype == np.int64
        assert labels.tolist() == expected

    def test_border_point_adjacent_to_two_clusters_takes_lower_id(self) -> None:
        labels = st_dbscan(
            _minutes(_B_MINUTES),
            _B_LAT,
            _B_LON,
            None,
            n_points=4,
            max_separation_km=90.0,
            max_minutes=45.5,
            not_in_cluster_label=-1,
        )
        assert labels.tolist() == _B_LABELS
        # Row 9 is a border point with core neighbours in clusters 0 and 1.
        assert labels[9] == 0


def _property_case(
    n_points: int, max_alt: float | None
) -> tuple[
    NDArray[np.datetime64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Seeded random points with their labels, neighbour matrix and core mask."""
    rng = np.random.default_rng(42)
    n = 40
    lat = rng.uniform(50.0, 52.0, n)
    lon = rng.uniform(-1.0, 1.0, n)
    time = _minutes(rng.uniform(0.0, 120.0, n))
    alt = rng.uniform(8000.0, 12000.0, n)
    max_km = 40.0
    max_minutes = 30.5
    labels = _cluster(time, lat, lon, alt, n_points=n_points, max_km=max_km, max_minutes=max_minutes, max_alt=max_alt)
    nb = _neighbour_matrix(time, lat, lon, alt, max_km, max_minutes, max_alt)
    core = nb.sum(axis=1) >= n_points
    return time, lat, lon, alt, labels, nb, core


_PROPERTY_PARAMS = [(2, None), (3, 1000.5), (5, None), (5, 1500.5)]
_N_POINTS_WITHOUT_BORDER_POINTS = 2


class TestLabelProperties:
    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_output_is_int64_with_input_length(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, _, _ = _property_case(n_points, max_alt)
        assert labels.dtype == np.int64
        assert labels.shape == (40,)

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_cluster_ids_are_contiguous_from_zero_with_noise_as_minus_one(
        self, n_points: int, max_alt: float | None
    ) -> None:
        _, _, _, _, labels, _, _ = _property_case(n_points, max_alt)
        ids = np.unique(labels[labels >= 0])
        assert ids.tolist() == list(range(len(ids)))
        assert set(np.unique(labels).tolist()) <= set(range(len(ids))) | {-1}

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_core_points_are_always_clustered(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, _, core = _property_case(n_points, max_alt)
        assert core.any()
        assert np.all(labels[core] >= 0)

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_adjacent_core_points_share_a_label(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, nb, core = _property_case(n_points, max_alt)
        rows, cols = np.nonzero(nb & core[:, None] & core[None, :])
        assert np.array_equal(labels[rows], labels[cols])

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_noise_is_exactly_the_points_without_a_core_neighbour(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, nb, core = _property_case(n_points, max_alt)
        for i in range(len(labels)):
            has_core_neighbour = bool(np.any(nb[i] & core))
            assert (labels[i] == -1) == (not has_core_neighbour)

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_border_point_takes_lowest_adjacent_cluster_id(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, nb, core = _property_case(n_points, max_alt)
        checked = 0
        for i in np.flatnonzero(~core):
            adjacent = labels[nb[i] & core]
            if adjacent.size:
                assert labels[i] == adjacent.min()
                checked += 1
        # With n_points=2 any point that has a neighbour is core, so there are no border points to check.
        if n_points > _N_POINTS_WITHOUT_BORDER_POINTS:
            assert checked > 0

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_every_cluster_contains_a_core_point(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, _, core = _property_case(n_points, max_alt)
        for k in np.unique(labels[labels >= 0]):
            assert np.any(core & (labels == k))

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_cluster_ids_ascend_with_smallest_core_row_index(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, _, core = _property_case(n_points, max_alt)
        ids = np.unique(labels[labels >= 0])
        first_core_rows = [np.flatnonzero(core & (labels == k))[0] for k in ids]
        assert np.all(np.diff(first_core_rows) > 0)

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_point_with_no_neighbour_in_range_is_noise(self, n_points: int, max_alt: float | None) -> None:
        _, _, _, _, labels, nb, _ = _property_case(n_points, max_alt)
        # Only the point itself is in range, so the point cannot be core and has no core neighbour.
        isolated = nb.sum(axis=1) == 1
        assert isolated.any()
        assert np.all(labels[isolated] == -1)


class TestRowPermutation:
    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_noise_mask_is_unchanged_when_rows_are_permuted(self, n_points: int, max_alt: float | None) -> None:
        time, lat, lon, alt, labels, _, _ = _property_case(n_points, max_alt)
        perm = np.random.default_rng(42).permutation(len(labels))
        permuted = _cluster(
            time[perm],
            lat[perm],
            lon[perm],
            alt[perm],
            n_points=n_points,
            max_km=40.0,
            max_minutes=30.5,
            max_alt=max_alt,
        )
        assert np.array_equal(permuted == -1, labels[perm] == -1)

    @pytest.mark.parametrize(("n_points", "max_alt"), _PROPERTY_PARAMS)
    def test_core_point_grouping_is_unchanged_when_rows_are_permuted(
        self, n_points: int, max_alt: float | None
    ) -> None:
        time, lat, lon, alt, labels, _, core = _property_case(n_points, max_alt)
        perm = np.random.default_rng(42).permutation(len(labels))
        permuted = _cluster(
            time[perm],
            lat[perm],
            lon[perm],
            alt[perm],
            n_points=n_points,
            max_km=40.0,
            max_minutes=30.5,
            max_alt=max_alt,
        )
        core_rows = np.flatnonzero(core)
        positions = np.argsort(perm)[core_rows]
        same_before = labels[core_rows][:, None] == labels[core_rows][None, :]
        same_after = permuted[positions][:, None] == permuted[positions][None, :]
        assert np.array_equal(same_before, same_after)
