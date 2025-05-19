import pytest

from rojak.cli.main import app
from tests.test_cli import runner


def test_retrieve_data_ukmo_amdar(tmp_path) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        runner.invoke(
            app,
            [
                "data",
                "retrieve",
                "-s",
                "ukmo-amdar",
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
                "preprocess",
                "-s",
                "ukmo-amdar",
                "-i",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )
    assert excinfo.type is NotImplementedError
