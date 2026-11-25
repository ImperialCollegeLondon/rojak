"""Tests for rojak.datalib.iata.turbulence_aware: the preprocessor, its clustering and the clustering config."""

import math
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import dask.config
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike, NDArray
from pydantic import ValidationError

from rojak.core.data import DataPreprocessor
from rojak.datalib.iata import ClusteringConfig, ConditionColumn, IataObsPreprocessor
from rojak.datalib.iata._clustering import st_dbscan

_START = pd.Timestamp("2025-01-01T00:00:00")
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_THRESHOLD = 0.3
_OUTPUT_COLUMNS = frozenset({"time", "altitude", "latitude", "longitude", "peak", "mean"})
_FLOAT_COLUMNS = ("altitude", "latitude", "longitude", "peak", "mean")
# Used to line up output rows with input rows: values are rounded so that parser differences cannot reorder them
_KEY_COLUMNS = ["time", "latitude", "longitude", "altitude", "peak", "mean"]
# Shape of the synthetic input: the number of CSV files, and how many clusters the fixture should yield at least
_RAW_FILES = 4
_MIN_CLUSTERS = 3
_SINGLE_FILE_ROWS = 2


def _config(**overrides: Any) -> ClusteringConfig:
    """Clustering configuration with test defaults, where ``overrides`` replace individual fields."""
    # Any: the keyword overrides are deliberately untyped so that invalid values can be passed to the constructor
    fields: dict[str, Any] = {
        "n_points": 3,
        "max_separation_km": 10.0,
        "max_minutes": 10.0,
        "max_altitude_delta_ft": 1000.0,
        "condition_column": ConditionColumn.PEAK,
        "condition_threshold": _THRESHOLD,
    }
    fields.update(overrides)
    return ClusteringConfig(**fields)


def _rows(
    *,
    latitude: ArrayLike,
    longitude: ArrayLike,
    seconds: ArrayLike,
    altitude: ArrayLike,
    peak: ArrayLike,
    mean: ArrayLike,
) -> pd.DataFrame:
    """
    Raw-format observations from parallel columns

    ``seconds`` are offsets from 2025-01-01T00:00:00Z, and NaN gives a missing time. ``id`` and ``tafi`` are
    placeholders that the preprocessor must drop.
    """
    lat = np.asarray(latitude, dtype=np.float64)
    count = lat.shape[0]
    times = _START + pd.to_timedelta(np.asarray(seconds, dtype=np.float64), unit="s")
    return pd.DataFrame(
        {
            "observationtime": np.asarray(times.strftime(_TIME_FORMAT)),
            "id": [f"AB{index:05d}" for index in range(count)],
            "altitude": np.asarray(altitude, dtype=np.float64),
            "latitude": lat,
            "longitude": np.asarray(longitude, dtype=np.float64),
            "peak": np.asarray(peak, dtype=np.float64),
            "mean": np.asarray(mean, dtype=np.float64),
            "tafi": np.zeros(count, dtype=np.int64),
        }
    )


def _synthetic_observations(rng: np.random.Generator) -> pd.DataFrame:
    """
    Raw-format observations: three dense groups, isolated qualifying reports and scattered background reports

    The groups are six hours apart in time so that they cannot merge, and the rows are shuffled so that every
    group is spread over several files.
    """
    pieces: list[pd.DataFrame] = []
    for group, (centre_lat, centre_lon) in enumerate([(50.0, 8.0), (52.0, 2.5), (45.0, 20.0)]):
        count = 12
        pieces.append(
            _rows(
                latitude=np.round(centre_lat + rng.uniform(-0.03, 0.03, count), 6),
                longitude=np.round(centre_lon + rng.uniform(-0.03, 0.03, count), 6),
                seconds=np.round(group * 6 * 3600 + rng.uniform(0, 240, count)),
                altitude=np.round(33000 + rng.uniform(-200, 200, count), 1),
                peak=np.round(rng.uniform(0.6, 1.0, count), 6),
                mean=np.round(rng.uniform(0.6, 1.0, count), 6),
            )
        )
    # Qualifying on peak, far from every other report, so they are noise rather than clustered
    pieces.append(
        _rows(
            latitude=[30.0, 30.0, 30.0],
            longitude=[-40.0, -40.0, -40.0],
            seconds=[2 * 3600.0, 9 * 3600.0, 20 * 3600.0],
            altitude=[35000.0, 35000.0, 35000.0],
            peak=[0.9, 0.9, 0.9],
            mean=[0.1, 0.1, 0.1],
        )
    )
    count = 60
    pieces.append(
        _rows(
            latitude=np.round(rng.uniform(35, 65, count), 6),
            longitude=np.round(rng.uniform(-15, 25, count), 6),
            seconds=np.round(rng.uniform(0, 24 * 3600, count)),
            altitude=np.round(rng.uniform(10000, 40000, count), 1),
            peak=np.round(rng.uniform(0, 1, count), 6),
            mean=np.round(rng.uniform(0, 1, count), 6),
        )
    )
    frame = pd.concat(pieces, ignore_index=True)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _split(frame: pd.DataFrame, parts: int) -> list[pd.DataFrame]:
    """Split ``frame`` into ``parts`` contiguous pieces of near-equal length."""
    return [frame.iloc[indices] for indices in np.array_split(np.arange(len(frame)), parts)]


def _write_parts(directory: Path, parts: list[pd.DataFrame]) -> Path:
    """Write each piece to its own ``part_NN.csv`` file in ``directory``, so that sorted glob order is file order."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, part in enumerate(parts):
        part.to_csv(directory / f"part_{index:02d}.csv", index=False)
    return directory


def _read_raw(directory: Path) -> pd.DataFrame:
    """Input observations read back from the CSV files in sorted file order, with ``time`` parsed as naive UTC."""
    raw = pd.concat(
        [pd.read_csv(path) for path in sorted(directory.glob("*.csv"))],
        ignore_index=True,
    )
    raw["time"] = pd.to_datetime(raw["observationtime"], utc=True).dt.tz_localize(None)
    return raw


def _expected_labels(raw: pd.DataFrame, config: ClusteringConfig) -> NDArray[np.int64]:
    """
    Cluster label of every input row, from st_dbscan run directly on the qualifying rows

    A row qualifies when its condition value is strictly above the threshold and its latitude, longitude,
    altitude and time are all present. The qualifying rows are sorted by time, then latitude, longitude and
    altitude, and the labels are written back to the rows they came from. Everything else is -1.
    """
    time = raw["time"].to_numpy(dtype="datetime64[ns]")
    lat = raw["latitude"].to_numpy(dtype=np.float64)
    lon = raw["longitude"].to_numpy(dtype=np.float64)
    alt = raw["altitude"].to_numpy(dtype=np.float64)
    condition = raw[config.condition_column.value].to_numpy(dtype=np.float64)

    usable = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt) & ~np.isnat(time)
    qualifying = np.flatnonzero((condition > config.condition_threshold) & usable)
    # np.lexsort takes its primary key last
    order = qualifying[
        np.lexsort((alt[qualifying], lon[qualifying], lat[qualifying], time[qualifying].astype(np.int64)))
    ]

    labels = np.full(len(raw), -1, dtype=np.int64)
    labels[order] = st_dbscan(
        time[order],
        lat[order],
        lon[order],
        alt[order] if config.max_altitude_delta_ft is not None else None,
        n_points=config.n_points,
        max_separation_km=config.max_separation_km,
        max_minutes=config.max_minutes,
        max_altitude_delta=config.max_altitude_delta_ft,
        not_in_cluster_label=-1,
    )
    return labels


def _in_key_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by their key columns, with the values in the key rounded to six decimal places."""
    frame = frame.reset_index(drop=True)
    keys = pd.DataFrame(
        {
            "time": frame["time"],
            **{name: frame[name].round(6) for name in _KEY_COLUMNS[1:]},
        }
    )
    return frame.loc[keys.sort_values(_KEY_COLUMNS, kind="stable").index].reset_index(drop=True)


def _assert_same_observations(out: pd.DataFrame, raw: pd.DataFrame) -> None:
    """The output holds the same observations as the input, with their values, in any row order."""
    out_sorted = _in_key_order(out)
    raw_sorted = _in_key_order(raw)
    assert len(out_sorted) == len(raw_sorted)
    np.testing.assert_array_equal(
        out_sorted["time"].to_numpy(dtype="datetime64[ns]"),
        raw_sorted["time"].to_numpy(dtype="datetime64[ns]"),
    )
    for name in _FLOAT_COLUMNS:
        np.testing.assert_allclose(
            out_sorted[name].to_numpy(dtype=np.float64),
            raw_sorted[name].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )


def _assert_labels_match_direct_clustering(out: pd.DataFrame, raw: pd.DataFrame, config: ClusteringConfig) -> None:
    """Clustered output agrees row by row with st_dbscan run directly on the qualifying input rows."""
    _assert_same_observations(out, raw)
    expected = _in_key_order(raw.assign(cluster=_expected_labels(raw, config), in_cluster=lambda f: f["cluster"] != -1))
    actual = _in_key_order(out)
    np.testing.assert_array_equal(actual["cluster"].to_numpy(), expected["cluster"].to_numpy())
    np.testing.assert_array_equal(actual["in_cluster"].to_numpy(), expected["in_cluster"].to_numpy())


def _preprocess(raw_dir: Path, output: Path, clustering: ClusteringConfig | None) -> pd.DataFrame:
    """Run the preprocessor over every CSV in ``raw_dir`` and read the parquet output back."""
    IataObsPreprocessor(raw_dir, glob_pattern="*.csv", clustering=clustering).apply_preprocessor(output)
    return dd.read_parquet(str(output)).compute()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """
    Four CSV files of raw observations, a stray non-CSV file, and one file with nothing qualifying

    Clusters are spread across the files, so the clustering has to work across partition boundaries.
    """
    frame = _synthetic_observations(np.random.default_rng(42))
    parts = _split(frame, _RAW_FILES)
    parts[2] = parts[2].assign(peak=0.0, mean=0.0)
    directory = _write_parts(tmp_path / "raw", parts)
    (directory / "notes.txt").write_text("this file is not a CSV\n")
    return directory


class TestSyntheticFixture:
    """Guards that the shared synthetic input really exercises what the other tests rely on."""

    def test_input_is_split_across_several_partitions(self, raw_dir: Path) -> None:
        assert dd.read_csv(str(raw_dir / "*.csv")).npartitions >= _RAW_FILES

    def test_one_partition_has_nothing_qualifying(self, raw_dir: Path) -> None:
        assert pd.read_csv(raw_dir / "part_02.csv")["peak"].max() <= _THRESHOLD

    def test_several_clusters_and_qualifying_noise_are_present(self, raw_dir: Path) -> None:
        raw = _read_raw(raw_dir)
        labels = _expected_labels(raw, _config())
        assert len(set(labels.tolist()) - {-1}) >= _MIN_CLUSTERS
        qualifying = raw["peak"].to_numpy() > _THRESHOLD
        assert (qualifying & (labels == -1)).any()


class TestIataObsPreprocessorConstruction:
    def test_is_a_data_preprocessor(self) -> None:
        assert issubclass(IataObsPreprocessor, DataPreprocessor)

    def test_instance_is_a_data_preprocessor(self, raw_dir: Path) -> None:
        assert isinstance(IataObsPreprocessor(raw_dir, glob_pattern="*.csv"), DataPreprocessor)

    def test_directory_without_glob_pattern_raises_value_error(self, raw_dir: Path) -> None:
        with pytest.raises(ValueError, match="glob"):
            IataObsPreprocessor(raw_dir)

    def test_glob_pattern_matching_no_file_raises_value_error(self, raw_dir: Path) -> None:
        with pytest.raises(ValueError, match=r"No file .* matches the glob pattern"):
            IataObsPreprocessor(raw_dir, glob_pattern="*.parquet")

    def test_glob_pattern_matching_only_directories_raises_value_error(self, raw_dir: Path) -> None:
        (raw_dir / "subdir.csv").mkdir()
        with pytest.raises(ValueError, match="No file"):
            IataObsPreprocessor(raw_dir, glob_pattern="subdir*")

    def test_missing_path_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            IataObsPreprocessor(tmp_path / "missing.csv")

    def test_single_file_is_processed_without_glob_pattern(self, tmp_path: Path) -> None:
        frame = _rows(
            latitude=[50.0, 51.0],
            longitude=[8.0, 9.0],
            seconds=[0.0, 3600.0],
            altitude=[30000.0, 31000.0],
            peak=[0.1, 0.2],
            mean=[0.3, 0.4],
        )
        single = tmp_path / "single.csv"
        frame.to_csv(single, index=False)
        output = tmp_path / "out"
        IataObsPreprocessor(single).apply_preprocessor(output)
        out = dd.read_parquet(str(output)).compute()
        assert len(out) == _SINGLE_FILE_ROWS

    def test_glob_pattern_skips_files_that_do_not_match(self, raw_dir: Path, tmp_path: Path) -> None:
        out = _preprocess(raw_dir, tmp_path / "out", clustering=None)
        assert len(out) == len(_read_raw(raw_dir))

    def test_apply_preprocessor_creates_missing_output_directory(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "out"
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv").apply_preprocessor(output)
        assert output.is_dir()

    def test_clustering_is_off_by_default(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv").apply_preprocessor(output)
        out = dd.read_parquet(str(output)).compute()
        assert "cluster" not in out.columns
        assert "in_cluster" not in out.columns


class TestOutputFilename:
    def test_unclustered_output_has_a_fixed_name(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv").apply_preprocessor(output)
        assert [path.name for path in output.iterdir()] == ["unclustered-iata-data.parquet"]

    def test_clustered_output_is_named_after_the_clustering_parameters(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        config = _config()
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv", clustering=config).apply_preprocessor(output)
        assert [path.name for path in output.iterdir()] == [f"clustered-{config.to_descriptor()}.parquet"]

    def test_output_is_readable_at_the_named_path(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv", clustering=_config()).apply_preprocessor(output)
        (dataset,) = output.iterdir()
        assert "cluster" in dd.read_parquet(str(dataset)).columns

    def test_different_clusterings_are_written_side_by_side(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        for n_points in (2, 3):
            config = _config(n_points=n_points)
            IataObsPreprocessor(raw_dir, glob_pattern="*.csv", clustering=config).apply_preprocessor(output)
        IataObsPreprocessor(raw_dir, glob_pattern="*.csv").apply_preprocessor(output)
        assert len({path.name for path in output.iterdir()}) == 3  # noqa: PLR2004

    def test_same_parameters_overwrite_the_previous_output(self, raw_dir: Path, tmp_path: Path) -> None:
        output = tmp_path / "out"
        for _ in range(2):
            IataObsPreprocessor(raw_dir, glob_pattern="*.csv", clustering=_config()).apply_preprocessor(output)
        (dataset,) = output.iterdir()
        assert len(dd.read_parquet(str(dataset)).compute()) == len(_read_raw(raw_dir))


class TestBaselineOutput:
    @pytest.fixture
    def baseline(self, raw_dir: Path, tmp_path: Path) -> pd.DataFrame:
        return _preprocess(raw_dir, tmp_path / "out", clustering=None)

    def test_columns_are_exactly_the_kept_observation_fields(self, baseline: pd.DataFrame) -> None:
        assert set(baseline.columns) == _OUTPUT_COLUMNS
        assert len(baseline.columns) == len(_OUTPUT_COLUMNS)

    def test_id_and_tafi_are_dropped(self, baseline: pd.DataFrame) -> None:
        assert "id" not in baseline.columns
        assert "tafi" not in baseline.columns
        assert "observationtime" not in baseline.columns

    def test_time_is_naive_datetime64(self, baseline: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(baseline["time"])
        assert baseline["time"].dt.tz is None

    @pytest.mark.parametrize("column", _FLOAT_COLUMNS)
    def test_observation_values_are_float64(self, baseline: pd.DataFrame, column: str) -> None:
        assert baseline[column].dtype == np.float64

    def test_row_count_is_preserved_across_partitions(self, raw_dir: Path, baseline: pd.DataFrame) -> None:
        assert len(baseline) == len(_read_raw(raw_dir))

    def test_values_match_the_input_rows(self, raw_dir: Path, baseline: pd.DataFrame) -> None:
        _assert_same_observations(baseline, _read_raw(raw_dir))

    def test_trailing_z_time_is_read_as_naive_utc(self, tmp_path: Path) -> None:
        frame = _rows(
            latitude=[50.0],
            longitude=[8.0],
            seconds=[6.5 * 3600],
            altitude=[30000.0],
            peak=[0.1],
            mean=[0.2],
        )
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        out = _preprocess(raw_dir, tmp_path / "out", clustering=None)
        assert out["time"].to_numpy(dtype="datetime64[ns]")[0] == np.datetime64("2025-01-01T06:30:00")


class TestClusteredOutput:
    @pytest.fixture
    def clustered(self, raw_dir: Path, tmp_path: Path) -> pd.DataFrame:
        return _preprocess(raw_dir, tmp_path / "out", clustering=_config())

    def test_output_has_observation_fields_and_both_cluster_columns(self, clustered: pd.DataFrame) -> None:
        assert set(clustered.columns) == _OUTPUT_COLUMNS | {"cluster", "in_cluster"}

    def test_cluster_is_int64_and_in_cluster_is_bool(self, clustered: pd.DataFrame) -> None:
        assert clustered["cluster"].dtype == np.int64
        assert clustered["in_cluster"].dtype == np.bool_

    def test_in_cluster_is_true_exactly_when_cluster_is_not_noise(self, clustered: pd.DataFrame) -> None:
        np.testing.assert_array_equal(clustered["in_cluster"].to_numpy(), clustered["cluster"].to_numpy() != -1)

    @pytest.mark.parametrize("condition_column", list(ConditionColumn))
    def test_labels_match_direct_clustering_of_qualifying_rows(
        self, raw_dir: Path, tmp_path: Path, condition_column: ConditionColumn
    ) -> None:
        config = _config(condition_column=condition_column)
        out = _preprocess(raw_dir, tmp_path / condition_column.value, clustering=config)
        _assert_labels_match_direct_clustering(out, _read_raw(raw_dir), config)

    def test_non_qualifying_rows_are_noise(self, raw_dir: Path, clustered: pd.DataFrame) -> None:
        raw = _in_key_order(_read_raw(raw_dir))
        out = _in_key_order(clustered)
        _assert_same_observations(out, raw)
        not_qualifying = raw["peak"].to_numpy() <= _THRESHOLD
        assert not_qualifying.any()
        assert (out.loc[not_qualifying, "cluster"] == -1).all()
        assert not out.loc[not_qualifying, "in_cluster"].any()

    def test_row_count_is_preserved(self, raw_dir: Path, clustered: pd.DataFrame) -> None:
        assert len(clustered) == len(_read_raw(raw_dir))


class TestInvalidRowsAreDropped:
    @pytest.mark.parametrize("field", ["latitude", "longitude", "altitude", "observationtime"])
    @pytest.mark.parametrize("clustering", [None, "config"])
    def test_row_with_missing_field_is_dropped(self, tmp_path: Path, field: str, clustering: str | None) -> None:
        frame = _rows(
            latitude=[50.0, 50.0, 50.0, 50.0],
            longitude=[8.0, 8.0, 8.0, 8.0],
            seconds=[0.0, 60.0, 120.0, 90.0],
            altitude=[30000.0, 30000.0, 30000.0, 30000.0],
            peak=[1.0, 1.0, 1.0, 0.95],
            mean=[1.0, 1.0, 1.0, 1.0],
        )
        frame.loc[3, field] = np.nan
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        out = _preprocess(raw_dir, tmp_path / "out", clustering=None if clustering is None else _config())

        assert len(out) == len(frame) - 1
        assert not np.isclose(out["peak"].to_numpy(), 0.95).any()
        if clustering is not None:
            assert out["in_cluster"].all()
            assert set(out["cluster"].tolist()) == {0}

    @pytest.mark.parametrize("field", ["latitude", "longitude", "altitude"])
    @pytest.mark.parametrize("value", [np.inf, -np.inf])
    def test_row_with_infinite_value_is_dropped(self, tmp_path: Path, field: str, value: float) -> None:
        frame = _rows(
            latitude=[50.0, 50.0],
            longitude=[8.0, 8.0],
            seconds=[0.0, 60.0],
            altitude=[30000.0, 30000.0],
            peak=[0.1, 0.1],
            mean=[0.1, 0.1],
        )
        frame.loc[1, field] = value
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        out = _preprocess(raw_dir, tmp_path / "out", clustering=None)

        assert len(out) == 1

    def test_missing_peak_or_mean_does_not_drop_the_row(self, tmp_path: Path) -> None:
        frame = _rows(
            latitude=[50.0, 50.0],
            longitude=[8.0, 8.0],
            seconds=[0.0, 60.0],
            altitude=[30000.0, 30000.0],
            peak=[np.nan, 0.1],
            mean=[0.1, np.nan],
        )
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        out = _preprocess(raw_dir, tmp_path / "out", clustering=None)

        assert len(out) == len(frame)

    def test_partition_that_is_entirely_invalid_leaves_labels_aligned(self, tmp_path: Path) -> None:

        def cluster_of_three(seconds: list[float]) -> pd.DataFrame:
            return _rows(
                latitude=[50.0] * 3,
                longitude=[8.0] * 3,
                seconds=seconds,
                altitude=[30000.0] * 3,
                peak=[1.0] * 3,
                mean=[1.0] * 3,
            )

        good = cluster_of_three([0.0, 60.0, 120.0])
        bad = good.assign(latitude=np.nan)
        # An hour apart, so the two good partitions form separate clusters with an invalid partition between them
        raw_dir = _write_parts(tmp_path / "raw", [good, bad, cluster_of_three([3600.0, 3660.0, 3720.0])])
        out = _preprocess(raw_dir, tmp_path / "out", clustering=_config())

        assert len(out) == 2 * len(good)
        assert out["in_cluster"].all()
        assert sorted(set(out["cluster"].tolist())) == [0, 1]


class TestAltitudeThresholdInFeet:
    @pytest.mark.parametrize(
        ("max_altitude_delta_ft", "clustered"),
        [(700.0, False), (900.0, True), (None, True)],
    )
    def test_altitude_gap_of_800_feet_is_compared_against_threshold_in_feet(
        self, tmp_path: Path, max_altitude_delta_ft: float | None, clustered: bool
    ) -> None:
        frame = _rows(
            latitude=[50.0, 50.0],
            longitude=[8.0, 8.0],
            seconds=[0.0, 60.0],
            altitude=[30000.0, 30800.0],
            peak=[1.0, 1.0],
            mean=[1.0, 1.0],
        )
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        config = _config(n_points=2, max_altitude_delta_ft=max_altitude_delta_ft)
        out = _preprocess(raw_dir, tmp_path / "out", clustering=config)
        assert out["in_cluster"].tolist() == [clustered, clustered]
        assert set(out["cluster"].tolist()) == ({0} if clustered else {-1})


class TestConditionColumnSelection:
    @pytest.mark.parametrize(
        ("condition_column", "clustered_latitude"),
        [(ConditionColumn.PEAK, 50.0), (ConditionColumn.MEAN, 20.0)],
    )
    def test_only_the_selected_column_is_compared_with_threshold(
        self, tmp_path: Path, condition_column: ConditionColumn, clustered_latitude: float
    ) -> None:
        # The pair at latitude 50 is high on peak and low on mean; the pair at latitude 20 is the reverse
        frame = _rows(
            latitude=[50.0, 50.0, 20.0, 20.0],
            longitude=[8.0, 8.0, -30.0, -30.0],
            seconds=[0.0, 60.0, 6 * 3600.0, 6 * 3600.0 + 60.0],
            altitude=[30000.0, 30000.0, 30000.0, 30000.0],
            peak=[1.0, 1.0, 0.0, 0.0],
            mean=[0.0, 0.0, 1.0, 1.0],
        )
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        config = _config(n_points=2, condition_column=condition_column, condition_threshold=0.5)
        out = _preprocess(raw_dir, tmp_path / "out", clustering=config)
        assert set(out.loc[out["in_cluster"].to_numpy(), "latitude"].tolist()) == {clustered_latitude}

    @pytest.mark.parametrize(
        ("condition_column", "value", "clustered"),
        [
            (ConditionColumn.PEAK, 0.3, False),
            (ConditionColumn.PEAK, 0.31, True),
            (ConditionColumn.MEAN, 0.3, False),
            (ConditionColumn.MEAN, 0.31, True),
        ],
    )
    def test_value_equal_to_threshold_does_not_qualify(
        self, tmp_path: Path, condition_column: ConditionColumn, value: float, clustered: bool
    ) -> None:
        selected = [value, value]
        other = [0.0, 0.0]
        peak, mean = (selected, other) if condition_column is ConditionColumn.PEAK else (other, selected)
        frame = _rows(
            latitude=[50.0, 50.0],
            longitude=[8.0, 8.0],
            seconds=[0.0, 60.0],
            altitude=[30000.0, 30000.0],
            peak=peak,
            mean=mean,
        )
        raw_dir = _write_parts(tmp_path / "raw", [frame])
        config = _config(n_points=2, condition_column=condition_column, condition_threshold=_THRESHOLD)
        out = _preprocess(raw_dir, tmp_path / "out", clustering=config)
        assert out["in_cluster"].tolist() == [clustered, clustered]


class TestNoQualifyingRows:
    def test_all_rows_are_noise_when_nothing_qualifies(self, raw_dir: Path, tmp_path: Path) -> None:
        out = _preprocess(raw_dir, tmp_path / "out", clustering=_config(condition_threshold=10.0))
        assert len(out) == len(_read_raw(raw_dir))
        assert (out["cluster"].to_numpy() == -1).all()
        assert not out["in_cluster"].to_numpy().any()


class TestSchedulerIndependence:
    @pytest.mark.parametrize("scheduler", ["threads", "synchronous"])
    def test_labels_match_direct_clustering_under_scheduler(
        self, raw_dir: Path, tmp_path: Path, scheduler: str
    ) -> None:
        with dask.config.set(scheduler=scheduler):
            out = _preprocess(raw_dir, tmp_path / scheduler, clustering=_config())
        _assert_labels_match_direct_clustering(out, _read_raw(raw_dir), _config())

    def test_threaded_and_synchronous_runs_give_identical_labels(self, raw_dir: Path, tmp_path: Path) -> None:
        with dask.config.set(scheduler="threads"):
            threaded = _in_key_order(_preprocess(raw_dir, tmp_path / "threads", clustering=_config()))
        with dask.config.set(scheduler="synchronous"):
            synchronous = _in_key_order(_preprocess(raw_dir, tmp_path / "synchronous", clustering=_config()))
        np.testing.assert_array_equal(threaded["cluster"].to_numpy(), synchronous["cluster"].to_numpy())
        np.testing.assert_array_equal(threaded["in_cluster"].to_numpy(), synchronous["in_cluster"].to_numpy())


class TestConditionColumnEnum:
    def test_is_a_str_enum(self) -> None:
        assert issubclass(ConditionColumn, StrEnum)

    def test_members_are_the_raw_column_names(self) -> None:
        assert {member.value for member in ConditionColumn} == {"peak", "mean"}
        assert ConditionColumn.PEAK == "peak"
        assert ConditionColumn.MEAN == "mean"


class TestClusteringConfigDescriptor:
    def test_descriptor_lists_every_parameter(self) -> None:
        config = ClusteringConfig(
            n_points=3,
            max_separation_km=10.0,
            max_minutes=15.0,
            max_altitude_delta_ft=1000.0,
            condition_column=ConditionColumn.PEAK,
            condition_threshold=0.3,
        )
        assert config.to_descriptor() == "n_3-10_km-15_mins-1000_ft-peak_0_30"

    @pytest.mark.parametrize(
        ("overrides", "expected_part"),
        [
            ({"max_separation_km": 10.4}, "-10_km-"),
            ({"max_separation_km": 10.6}, "-11_km-"),
            ({"max_minutes": 14.4}, "-14_mins-"),
            ({"max_minutes": 14.6}, "-15_mins-"),
            ({"max_altitude_delta_ft": 999.6}, "-1000_ft-"),
            ({"max_altitude_delta_ft": 1000.4}, "-1000_ft-"),
        ],
    )
    def test_separation_time_and_altitude_are_rounded_to_the_nearest_integer(
        self, overrides: dict[str, Any], expected_part: str
    ) -> None:
        descriptor = _config(**overrides).to_descriptor()
        assert expected_part in descriptor
        assert "." not in descriptor.split("-peak_")[0]

    def test_values_that_round_to_the_same_integer_share_a_descriptor(self) -> None:
        assert _config(max_separation_km=10.1).to_descriptor() == _config(max_separation_km=9.9).to_descriptor()

    def test_unset_altitude_delta_is_written_as_no_alt(self) -> None:
        descriptor = _config(max_altitude_delta_ft=None, condition_column=ConditionColumn.MEAN).to_descriptor()
        assert "-no_alt-" in descriptor
        assert "None" not in descriptor

    @pytest.mark.parametrize(
        ("threshold", "expected_suffix"),
        [(0.3, "peak_0_30"), (0.0, "peak_0_00"), (-1.5, "peak_-1_50"), (0.126, "peak_0_13"), (2.0, "peak_2_00")],
    )
    def test_threshold_is_rounded_to_two_decimals_without_a_decimal_point(
        self, threshold: float, expected_suffix: str
    ) -> None:
        assert _config(condition_threshold=threshold).to_descriptor().endswith(expected_suffix)

    @pytest.mark.parametrize(
        "overrides", [{}, {"n_points": 5}, {"max_altitude_delta_ft": None}, {"condition_threshold": -2.0}]
    )
    def test_descriptor_only_has_file_name_safe_characters(self, overrides: dict[str, Any]) -> None:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", _config(**overrides).to_descriptor())

    def test_parameters_that_differ_give_different_descriptors(self) -> None:
        base = _config().to_descriptor()
        variants = [
            _config(n_points=4),
            _config(max_separation_km=11.0),
            _config(max_minutes=11.0),
            _config(max_altitude_delta_ft=None),
            _config(condition_column=ConditionColumn.MEAN),
            _config(condition_threshold=_THRESHOLD + 1.0),
        ]
        assert len({base, *(variant.to_descriptor() for variant in variants)}) == len(variants) + 1


class TestClusteringConfig:
    def test_zero_thresholds_and_single_point_neighbourhoods_are_valid(self) -> None:
        config = ClusteringConfig(
            n_points=1,
            max_separation_km=0.0,
            max_minutes=0.0,
            max_altitude_delta_ft=0.0,
            condition_column=ConditionColumn.MEAN,
            condition_threshold=0.0,
        )
        assert config.n_points == 1
        assert config.max_altitude_delta_ft == 0.0

    def test_altitude_delta_defaults_to_none(self) -> None:
        config = ClusteringConfig(
            n_points=3,
            max_separation_km=10.0,
            max_minutes=10.0,
            condition_column=ConditionColumn.PEAK,
            condition_threshold=0.3,
        )
        assert config.max_altitude_delta_ft is None

    def test_omitting_a_required_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            # n_points is deliberately omitted, so the required-argument check is what is being exercised
            ClusteringConfig(  # pyright: ignore[reportCallIssue]
                max_separation_km=10.0,
                max_minutes=10.0,
                condition_column=ConditionColumn.PEAK,
                condition_threshold=0.3,
            )

    @pytest.mark.parametrize(
        ("overrides", "field_name"),
        [
            ({"n_points": 0}, "n_points"),
            ({"n_points": -3}, "n_points"),
            ({"max_separation_km": -1.0}, "max_separation_km"),
            ({"max_separation_km": math.nan}, "max_separation_km"),
            ({"max_minutes": -1.0}, "max_minutes"),
            ({"max_minutes": math.nan}, "max_minutes"),
            ({"max_altitude_delta_ft": -1.0}, "max_altitude_delta_ft"),
            ({"max_altitude_delta_ft": math.nan}, "max_altitude_delta_ft"),
        ],
    )
    def test_invalid_value_raises_validation_error_naming_the_field(
        self, overrides: dict[str, Any], field_name: str
    ) -> None:
        with pytest.raises(ValidationError, match=field_name):
            _config(**overrides)

    @pytest.mark.parametrize("threshold", [-0.1, -5.0, 0.0, 2.5])
    def test_condition_threshold_is_unbounded(self, threshold: float) -> None:
        # A plain float compared against the column, so negative values are legitimate and accepted
        assert _config(condition_threshold=threshold).condition_threshold == threshold

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [("n_points", 1), ("max_separation_km", 0.0), ("max_minutes", 0.0), ("max_altitude_delta_ft", 0.0)],
    )
    def test_minimal_or_zero_value_is_accepted(self, field_name: str, value: float) -> None:
        assert getattr(_config(**{field_name: value}), field_name) == value

    def test_is_frozen(self) -> None:
        config = _config()
        with pytest.raises(ValidationError):
            config.n_points = 7  # pyright: ignore[reportAttributeAccessIssue]
