import pytest
from typer.testing import CliRunner

from rojak.cli.main import app

runner = CliRunner()


def test_turbulence_command() -> None:
    result = runner.invoke(app, ["turbulence"])
    assert result.exit_code == 0
    assert "HELLO" in result.stdout


def test_get_data_command() -> None:
    result = runner.invoke(app, ["get-data"])
    assert result.exit_code == 0
    assert "potatoes" in result.stdout


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
