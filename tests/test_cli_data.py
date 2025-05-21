from typing import TYPE_CHECKING, Tuple

import pytest

from rojak.cli.main import app
from tests.test_cli import runner

if TYPE_CHECKING:
    from pathlib import Path
    from click.testing import Result


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
            [
                "data",
                "amdar",
                "preprocess",
                "-s",
                "ukmo",
                "-i",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )
    assert excinfo.type is NotImplementedError


@pytest.fixture
def retrieve_madis_data(tmp_path) -> Tuple["Path", "Result"]:
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
    return tmp_path, result


@pytest.fixture
def retrieve_madis_data_single_file(tmp_path) -> Tuple["Path", "Result"]:
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
    return tmp_path, result


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
        assert (input_path / "2024" / "01" / f"20240101_{hour:02d}00.parquet").exists()


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
    assert (input_path / "2024" / "01" / "20240101_0000.parquet").exists()
