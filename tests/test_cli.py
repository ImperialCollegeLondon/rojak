import re

from typer.testing import CliRunner

runner = CliRunner()


def strip_ansi(text: str) -> str:
    pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return pattern.sub("", text)
