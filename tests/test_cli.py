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
