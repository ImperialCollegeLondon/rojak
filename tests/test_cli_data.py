from typing import TYPE_CHECKING

import dask.dataframe as dd
import pandas as pd
import pytest

from rojak.cli import data_interface
from rojak.cli.main import app
from rojak.datalib.ecmwf.era5 import InvalidEra5RequestConfigurationError
from rojak.datalib.iata import ClusteringConfig, ConditionColumn
from tests.test_cli import runner, strip_ansi

if TYPE_CHECKING:
    from pathlib import Path

    from typer.testing import Result


def test_retrieve_data_ukmo_amdar(tmp_path) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        runner.invoke(
            app,
            [
                "data",
                "amdar",
                "retrieve",
                "-s",
                "ukmo",
                "-y",
                "2024",
                "-m",
                "-1",
                "-d",
                "-1",
                "-o",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )
    assert excinfo.type is NotImplementedError


def test_preprocess_data_ukmo_amdar(tmp_path) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        runner.invoke(
            app,
            ["data", "amdar", "preprocess", "-s", "ukmo", "-i", str(tmp_path)],
            catch_exceptions=False,
        )
    assert excinfo.type is NotImplementedError


@pytest.fixture
def retrieve_madis_data(tmp_path) -> tuple["Path", "Result"]:
    result = runner.invoke(
        app,
        [
            "data",
            "amdar",
            "retrieve",
            "-s",
            "madis",
            "-y",
            "2024",
            "-m",
            "1",
            "-d",
            "1",
            "-o",
            str(tmp_path),
        ],
    )
    # pyright throws a false positive asking for an import from typer.testing instead of click.testing
    # BUT, typer.testing.Result is not exported...
    return tmp_path, result  # pyright: ignore[reportReturnType]


@pytest.fixture
def retrieve_madis_data_single_file(tmp_path) -> tuple["Path", "Result"]:
    result = runner.invoke(
        app,
        [
            "data",
            "amdar",
            "retrieve",
            "-s",
            "madis",
            "-y",
            "2024",
            "-m",
            "1",
            "-d",
            "1",
            "-o",
            str(tmp_path),
            "-g",
            "20240101_00*.gz",
        ],
    )
    # pyright throws a false positive asking for an import from typer.testing instead of click.testing
    # BUT, typer.testing.Result is not exported...
    return tmp_path, result  # pyright: ignore[reportReturnType]


@pytest.mark.slow
def test_retrieve_data_madis(retrieve_madis_data) -> None:
    output_path, result = retrieve_madis_data
    assert result.exit_code == 0
    for hour in range(24):
        assert (output_path / "2024" / "01" / f"20240101_{hour:02d}00.gz").exists()


def test_retrieve_data_madis_single_file(retrieve_madis_data_single_file) -> None:
    output_path, result = retrieve_madis_data_single_file
    assert result.exit_code == 0
    assert (output_path / "2024" / "01" / "20240101_0000.gz").exists()


@pytest.mark.slow
def test_preprocess_data_madis(retrieve_madis_data) -> None:
    input_path, retrieve_result = retrieve_madis_data
    assert retrieve_result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "data",
            "amdar",
            "preprocess",
            "-s",
            "madis",
            "-i",
            str(input_path),
            "--glob-pattern",
            "**/*.gz",
        ],
    )
    assert result.exit_code == 0
    for hour in range(24):
        created_file = input_path / "2024" / "01" / f"20240101_{hour:02d}00.parquet"
        assert created_file.exists()
        ddf = dd.read_parquet(str(created_file))
        # Call head to check that it can be computed
        ddf.head()
        assert "windSpeed" in ddf.columns
        assert "maxEDR" in ddf.columns


def test_preprocess_data_madis_single_file(retrieve_madis_data_single_file) -> None:
    input_path, retrieve_result = retrieve_madis_data_single_file
    assert retrieve_result.exit_code == 0
    result = runner.invoke(
        app,
        [
            "data",
            "amdar",
            "preprocess",
            "-s",
            "madis",
            "-i",
            str(input_path),
            "--glob-pattern",
            "**/*.gz",
        ],
    )
    assert result.exit_code == 0
    created_file = input_path / "2024" / "01" / "20240101_0000.parquet"
    assert created_file.exists()
    ddf = dd.read_parquet(str(created_file))
    # Call head to check that it can be computed
    ddf.head()
    assert "windSpeed" in ddf.columns


@pytest.mark.parametrize(
    ("data_set_name", "default_name", "matches"),
    [
        pytest.param("nonsense", None, "Invalid value for '-n' / '--data-set-name'", id="invalid_dataset_name"),
        pytest.param(None, "nonsense", "Missing option '-n'", id="missing_dataset_name"),
        pytest.param(
            "nonsense",
            "cat",
            "Invalid value for '-n' / '--data-set-name'",
            id="invalid_dataset_name_default_cat",
        ),
        pytest.param(
            "nonsense",
            "surface",
            "Invalid value for '-n' / '--data-set-name'",
            id="invalid_dataset_name_default_surface",
        ),
        pytest.param(
            "nonsense",
            "contrail",
            "Invalid value for '-n' / '--data-set-name'",
            id="invalid_dataset_name_default_contrail",
        ),
        pytest.param(
            "pressure-level",
            "nonsense",
            "Invalid value for '--default-name'",
            id="invalid_default_name_pl",
        ),
        pytest.param(
            "single-level",
            "nonsense",
            "Invalid value for '--default-name'",
            id="invalid_default_name_sl",
        ),
        pytest.param("pressure-level", None, "Default not specified.", id="default_not_specified"),
    ],
)
def test_retrieve_meteorology_era5_invalid_config(data_set_name, default_name, matches, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "data",
            "meteorology",
            "retrieve",
            "-s",
            "era5",
            "-y",
            "2024",
            "-m",
            "1",
            "-d",
            "1",
            "-n",
            data_set_name,
            "-o",
            str(tmp_path),
            "--default-name",
            default_name,
        ],
    )
    assert result.exit_code != 0
    if isinstance(result.exception, SystemExit):
        assert matches in strip_ansi(result.output)
    else:
        assert isinstance(result.exception, InvalidEra5RequestConfigurationError)
        assert matches in str(result.exception)


_IATA_HEADER = '"observationtime","id","altitude","latitude","longitude","peak","mean","tafi"'
# Exit code of a usage error reported by typer/click
_EXIT_USAGE_ERROR: int = 2
# Flags that set each clustering option to a valid value, keyed by option name
_IATA_CLUSTERING_FLAGS: dict[str, tuple[str, ...]] = {
    "--n-points": ("--n-points", "3"),
    "--max-separation-km": ("--max-separation-km", "50"),
    "--max-minutes": ("--max-minutes", "10"),
    "--condition-column": ("--condition-column", "peak"),
    "--condition-threshold": ("--condition-threshold", "0.2"),
    "--max-altitude-delta-ft": ("--max-altitude-delta-ft", "1000"),
}
_IATA_REQUIRED_WHEN_CLUSTERING: tuple[str, ...] = (
    "--n-points",
    "--max-separation-km",
    "--max-minutes",
    "--condition-column",
    "--condition-threshold",
)


def _invoke(*argv: str) -> "Result":
    # A wide terminal stops rich from wrapping option names and messages across lines
    return runner.invoke(app, list(argv), env={"COLUMNS": "200"})


def _text(result: "Result") -> str:
    """Output with ANSI codes removed and all whitespace, including line wraps, collapsed to single spaces"""
    return " ".join(strip_ansi(result.output).split())


def _preprocess(*options: str) -> list[str]:
    return ["data", "iata", "preprocess", *options]


def _clustering_flags(*names: str) -> list[str]:
    return [flag for name in names for flag in _IATA_CLUSTERING_FLAGS[name]]


def _write_iata_csv(path: "Path", rows: list[str]) -> "Path":
    path.write_text("\n".join([_IATA_HEADER, *rows]) + "\n")
    return path


class _SpyPreprocessor:
    """Records how the command constructs and applies an IATA preprocessor, without reading any data"""

    def __init__(
        self,
        input_path: "Path",
        glob_pattern: str | None = None,
        clustering: ClusteringConfig | None = None,
    ) -> None:
        self.input_path = input_path
        self.glob_pattern = glob_pattern
        self.clustering = clustering
        self.applied_to: list[Path] = []

    def apply_preprocessor(self, output_directory: "Path") -> None:
        self.applied_to.append(output_directory)


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> list[_SpyPreprocessor]:
    """Every preprocessor the command constructs, in order, standing in for ``IataObsPreprocessor``"""
    created: list[_SpyPreprocessor] = []

    def construct(
        input_path: "Path",
        glob_pattern: str | None = None,
        clustering: ClusteringConfig | None = None,
    ) -> _SpyPreprocessor:
        spy = _SpyPreprocessor(input_path, glob_pattern, clustering)
        created.append(spy)
        return spy

    monkeypatch.setattr(data_interface, "IataObsPreprocessor", construct)
    return created


class TestIataPreprocessHelp:
    """The ``rojak data iata preprocess`` command is registered and documents its options"""

    def test_data_help_lists_iata_group(self) -> None:
        result = _invoke("data", "--help")
        assert result.exit_code == 0, _text(result)
        assert "iata" in _text(result)

    def test_preprocess_help_exits_cleanly(self) -> None:
        result = _invoke("data", "iata", "preprocess", "--help")
        assert result.exit_code == 0, _text(result)

    @pytest.mark.parametrize(
        "option",
        [
            "--input_dir",
            "--output_dir",
            "--glob-pattern",
            "--cluster",
            "--no-cluster",
            "--n-points",
            "--max-separation-km",
            "--max-minutes",
            "--max-altitude-delta-ft",
            "--condition-column",
            "--condition-threshold",
        ],
    )
    def test_preprocess_help_lists_option(self, option: str) -> None:
        result = _invoke("data", "iata", "preprocess", "--help")
        assert option in _text(result)

    def test_preprocess_help_has_clustering_panel(self) -> None:
        result = _invoke("data", "iata", "preprocess", "--help")
        assert "Clustering" in _text(result)

    def test_preprocess_help_lists_condition_column_choices(self) -> None:
        result = _invoke("data", "iata", "preprocess", "--help")
        text = _text(result)
        assert "peak" in text
        assert "mean" in text


class TestIataPreprocessWritesParquet:
    """End-to-end runs on tiny synthetic CSV files, writing real parquet output"""

    def test_without_clustering_writes_observation_columns_only(self, tmp_path: "Path") -> None:
        source = _write_iata_csv(
            tmp_path / "obs.csv",
            [
                '"2024-06-01T00:00:00Z",1,35000,1.0,103.0,0.5,0.1,0',
                '"2024-06-01T00:02:00Z",2,10000,2.0,104.0,0.2,0.3,0',
            ],
        )
        output = tmp_path / "out"

        result = _invoke(*_preprocess("-i", str(source), "-o", str(output)))

        assert result.exit_code == 0, _text(result)
        observations = pd.read_parquet(output)
        assert set(observations.columns) == {"time", "altitude", "latitude", "longitude", "peak", "mean"}
        assert sorted(observations["peak"].tolist()) == [0.2, 0.5]
        assert sorted(observations["altitude"].tolist()) == [10000.0, 35000.0]
        # The trailing Z is dropped, leaving naive UTC times
        assert observations["time"].dt.tz is None
        assert sorted(observations["time"].tolist()) == [
            pd.Timestamp("2024-06-01T00:00:00"),
            pd.Timestamp("2024-06-01T00:02:00"),
        ]

    def test_clustering_writes_cluster_and_in_cluster_columns(self, tmp_path: "Path") -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        # Reports 1, 2 and 3 lie within 50 km, 10 minutes and 1000 ft of each other, so with n_points=3 they form a
        # single cluster. Report 4 is far away and alone. Report 5 is near report 1 but its peak (0.1) is not above
        # the 0.2 threshold, so it does not qualify.
        _write_iata_csv(
            raw / "a.csv",
            [
                '"2024-06-01T00:00:00Z",1,35000,1.0,103.0,0.5,0.1,0',
                '"2024-06-01T00:02:00Z",2,35100,1.01,103.0,0.6,0.2,0',
            ],
        )
        _write_iata_csv(
            raw / "b.csv",
            [
                '"2024-06-01T00:04:00Z",3,35200,1.0,103.02,0.4,0.3,0',
                '"2024-06-01T06:00:00Z",4,10000,50.0,10.0,0.9,0.8,0',
                '"2024-06-01T00:03:00Z",5,35000,1.0,103.01,0.1,0.05,0',
            ],
        )
        (raw / "notes.txt").write_text("not a CSV file, so the glob must skip it")
        output = tmp_path / "out"

        result = _invoke(
            *_preprocess(
                "-i",
                str(raw),
                "-g",
                "*.csv",
                "-o",
                str(output),
                "--cluster",
                *_clustering_flags(
                    "--n-points",
                    "--max-separation-km",
                    "--max-minutes",
                    "--condition-column",
                    "--condition-threshold",
                    "--max-altitude-delta-ft",
                ),
            )
        )

        assert result.exit_code == 0, _text(result)
        observations = pd.read_parquet(output).set_index("time")
        assert {"cluster", "in_cluster"} <= set(observations.columns)
        assert observations["cluster"].dtype == "int64"
        assert observations["in_cluster"].dtype == bool
        expected_cluster = {
            pd.Timestamp("2024-06-01T00:00:00"): 0,
            pd.Timestamp("2024-06-01T00:02:00"): 0,
            pd.Timestamp("2024-06-01T00:03:00"): -1,
            pd.Timestamp("2024-06-01T00:04:00"): 0,
            pd.Timestamp("2024-06-01T06:00:00"): -1,
        }
        assert observations["cluster"].to_dict() == expected_cluster
        assert observations["in_cluster"].to_dict() == {
            time: cluster != -1 for time, cluster in expected_cluster.items()
        }


class TestIataPreprocessBuildsConfiguration:
    """Arguments are turned into the preprocessor's arguments, with the preprocessor replaced by a spy"""

    def test_no_clustering_option_passes_no_clustering_config(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])
        output = tmp_path / "out"

        result = _invoke(*_preprocess("-i", str(source), "-o", str(output)))

        assert result.exit_code == 0, _text(result)
        assert len(spies) == 1
        assert spies[0].input_path == source
        assert spies[0].glob_pattern is None
        assert spies[0].clustering is None
        assert spies[0].applied_to == [output]

    def test_no_cluster_flag_passes_no_clustering_config(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(*_preprocess("-i", str(source), "--no-cluster"))

        assert result.exit_code == 0, _text(result)
        assert spies[0].clustering is None

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            pytest.param(
                [
                    "--n-points",
                    "3",
                    "--max-separation-km",
                    "50",
                    "--max-minutes",
                    "10",
                    "--condition-column",
                    "peak",
                    "--condition-threshold",
                    "0.2",
                    "--max-altitude-delta-ft",
                    "1000",
                ],
                ClusteringConfig(
                    n_points=3,
                    max_separation_km=50.0,
                    max_minutes=10.0,
                    condition_column=ConditionColumn.PEAK,
                    condition_threshold=0.2,
                    max_altitude_delta_ft=1000.0,
                ),
                id="all_options",
            ),
            pytest.param(
                [
                    "--n-points",
                    "3",
                    "--max-separation-km",
                    "50",
                    "--max-minutes",
                    "10",
                    "--condition-column",
                    "peak",
                    "--condition-threshold",
                    "0.2",
                ],
                ClusteringConfig(
                    n_points=3,
                    max_separation_km=50.0,
                    max_minutes=10.0,
                    condition_column=ConditionColumn.PEAK,
                    condition_threshold=0.2,
                ),
                id="without_altitude_delta",
            ),
            pytest.param(
                [
                    "--n-points",
                    "2",
                    "--max-separation-km",
                    "5.5",
                    "--max-minutes",
                    "0",
                    "--condition-column",
                    "mean",
                    "--condition-threshold",
                    "0.1",
                ],
                ClusteringConfig(
                    n_points=2,
                    max_separation_km=5.5,
                    max_minutes=0.0,
                    condition_column=ConditionColumn.MEAN,
                    condition_threshold=0.1,
                ),
                id="mean_column_zero_minutes",
            ),
            pytest.param(
                [
                    "--n-points",
                    "3",
                    "--max-separation-km",
                    "50",
                    "--max-minutes",
                    "10",
                    "--condition-column",
                    "PEAK",
                    "--condition-threshold",
                    "0.2",
                ],
                ClusteringConfig(
                    n_points=3,
                    max_separation_km=50.0,
                    max_minutes=10.0,
                    condition_column=ConditionColumn.PEAK,
                    condition_threshold=0.2,
                ),
                id="condition_column_upper_case",
            ),
            pytest.param(
                [
                    "--n-points",
                    "3",
                    "--max-separation-km",
                    "50",
                    "--max-minutes",
                    "10",
                    "--condition-column",
                    "mean",
                    "--condition-threshold=-0.5",
                ],
                ClusteringConfig(
                    n_points=3,
                    max_separation_km=50.0,
                    max_minutes=10.0,
                    condition_column=ConditionColumn.MEAN,
                    condition_threshold=-0.5,
                ),
                id="negative_condition_threshold",
            ),
        ],
    )
    def test_cluster_options_build_clustering_config(
        self, spies: list[_SpyPreprocessor], tmp_path, flags: list[str], expected: ClusteringConfig
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(*_preprocess("-i", str(source), "--cluster", *flags))

        assert result.exit_code == 0, _text(result)
        assert len(spies) == 1
        assert spies[0].clustering == expected

    def test_glob_pattern_and_directory_are_passed_through(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()

        result = _invoke(*_preprocess("-i", str(raw), "-g", "*.csv", "-o", str(tmp_path / "out")))

        assert result.exit_code == 0, _text(result)
        assert spies[0].input_path == raw
        assert spies[0].glob_pattern == "*.csv"

    def test_output_dir_defaults_to_input_directory(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()

        result = _invoke(*_preprocess("-i", str(raw), "-g", "*.csv"))

        assert result.exit_code == 0, _text(result)
        assert spies[0].applied_to == [raw]

    def test_output_dir_defaults_to_parent_directory_of_single_input_file(
        self, spies: list[_SpyPreprocessor], tmp_path
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        source = _write_iata_csv(raw / "obs.csv", [])

        result = _invoke(*_preprocess("-i", str(source)))

        assert result.exit_code == 0, _text(result)
        assert spies[0].applied_to == [raw]
        assert spies[0].applied_to[0] != source

    def test_explicit_output_dir_is_used_instead_of_input_directory(
        self, spies: list[_SpyPreprocessor], tmp_path
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        output = tmp_path / "elsewhere"

        result = _invoke(*_preprocess("-i", str(raw), "-g", "*.csv", "--output_dir", str(output)))

        assert result.exit_code == 0, _text(result)
        assert spies[0].applied_to == [output]


class TestIataPreprocessRejectsInvalidArguments:
    """Invalid arguments exit with a usage error (code 2) and name the problem, before any preprocessing starts"""

    def test_input_dir_is_required(self, spies: list[_SpyPreprocessor]) -> None:
        result = _invoke(*_preprocess("--no-cluster"))
        text = _text(result)
        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "Missing option" in text
        assert "--input_dir" in text
        assert spies == []

    def test_nonexistent_input_dir_is_rejected(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        missing = tmp_path / "does_not_exist"

        result = _invoke(*_preprocess("-i", str(missing)))
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "does not exist" in text
        assert spies == []

    def test_output_dir_that_is_a_file_is_rejected(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])
        output_file = tmp_path / "not_a_directory.txt"
        output_file.write_text("")

        result = _invoke(*_preprocess("-i", str(source), "-o", str(output_file)))
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "is a file" in text
        assert spies == []

    def test_n_points_zero_is_rejected_with_range_message(self, spies: list[_SpyPreprocessor], tmp_path) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(
            *_preprocess(
                "-i",
                str(source),
                "--cluster",
                "--n-points",
                "0",
                *_clustering_flags(
                    "--max-separation-km", "--max-minutes", "--condition-column", "--condition-threshold"
                ),
            )
        )
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "--n-points" in text
        assert "x>=1" in text
        assert spies == []

    @pytest.mark.parametrize(
        "negative_flag",
        ["--max-separation-km=-1", "--max-minutes=-1", "--max-altitude-delta-ft=-1"],
    )
    def test_negative_threshold_is_rejected_with_range_message(
        self, spies: list[_SpyPreprocessor], tmp_path, negative_flag: str
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])
        option = negative_flag.split("=", maxsplit=1)[0]

        result = _invoke(
            *_preprocess(
                "-i",
                str(source),
                "--cluster",
                *_clustering_flags(*_IATA_REQUIRED_WHEN_CLUSTERING, "--max-altitude-delta-ft"),
                negative_flag,
            )
        )
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert option in text
        assert "x>=0" in text
        assert spies == []

    @pytest.mark.parametrize(
        ("nan_flag", "field"),
        [
            ("--max-separation-km=nan", "max_separation_km"),
            ("--max-minutes=nan", "max_minutes"),
            ("--max-altitude-delta-ft=nan", "max_altitude_delta_ft"),
        ],
    )
    def test_nan_threshold_is_rejected_naming_the_field(
        self, spies: list[_SpyPreprocessor], tmp_path, nan_flag: str, field: str
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(
            *_preprocess(
                "-i",
                str(source),
                "--cluster",
                *_clustering_flags(*_IATA_REQUIRED_WHEN_CLUSTERING, "--max-altitude-delta-ft"),
                nan_flag,
            )
        )
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert f"--{field.replace('_', '-')}" in text
        assert spies == []

    def test_unknown_condition_column_is_rejected_and_lists_choices(
        self, spies: list[_SpyPreprocessor], tmp_path
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(
            *_preprocess(
                "-i",
                str(source),
                "--cluster",
                *_clustering_flags("--n-points", "--max-separation-km", "--max-minutes", "--condition-threshold"),
                "--condition-column",
                "foo",
            )
        )
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert "--condition-column" in text
        assert "'peak'" in text
        assert "'mean'" in text
        assert spies == []

    @pytest.mark.parametrize("omitted", _IATA_REQUIRED_WHEN_CLUSTERING)
    def test_cluster_with_a_required_option_missing_is_rejected(
        self, spies: list[_SpyPreprocessor], tmp_path, omitted: str
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])
        present = [name for name in _IATA_REQUIRED_WHEN_CLUSTERING if name != omitted]

        result = _invoke(*_preprocess("-i", str(source), "--cluster", *_clustering_flags(*present)))
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert omitted in text
        assert spies == []

    def test_cluster_with_no_clustering_options_is_rejected_and_names_all_required(
        self, spies: list[_SpyPreprocessor], tmp_path
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(*_preprocess("-i", str(source), "--cluster"))
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        for name in _IATA_REQUIRED_WHEN_CLUSTERING:
            assert name in text
        assert spies == []

    @pytest.mark.parametrize("option", list(_IATA_CLUSTERING_FLAGS))
    @pytest.mark.parametrize("no_cluster_flags", [[], ["--no-cluster"]], ids=["flag_omitted", "no_cluster_flag"])
    def test_clustering_option_without_cluster_is_rejected_and_named(
        self, spies: list[_SpyPreprocessor], tmp_path, option: str, no_cluster_flags: list[str]
    ) -> None:
        source = _write_iata_csv(tmp_path / "obs.csv", [])

        result = _invoke(*_preprocess("-i", str(source), *no_cluster_flags, *_clustering_flags(option)))
        text = _text(result)

        assert result.exit_code == _EXIT_USAGE_ERROR
        assert option in text
        assert spies == []
