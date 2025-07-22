import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from rojak.orchestrator.configuration import (
    Context as ConfigContext,
)
from rojak.orchestrator.configuration import (
    DataConfig,
    SpatialDomain,
    TurbulenceCalibrationConfig,
    TurbulenceCalibrationPhaseOption,
    TurbulenceCalibrationPhases,
    TurbulenceConfig,
    TurbulenceDiagnostics,
    TurbulencePhases,
    TurbulenceThresholds,
)
from rojak.orchestrator.turbulence import DISTRIBUTION_PARAMS_TYPE_ADAPTER, THRESHOLDS_TYPE_ADAPTER, TurbulenceLauncher

if TYPE_CHECKING:
    from rojak.turbulence.analysis import HistogramData


@pytest.fixture
def create_calibration_only_config(tmp_path_factory) -> Callable:
    def _calibration_only_config(num_diagnostics: int) -> ConfigContext:
        plots_dir: Path = tmp_path_factory.mktemp("plots")
        output_dir: Path = tmp_path_factory.mktemp("output")
        diagnostics: list[TurbulenceDiagnostics] = random.sample(list(TurbulenceDiagnostics), k=num_diagnostics)
        return ConfigContext(
            name="calibration_only",
            image_format="png",
            output_dir=output_dir,
            plots_dir=plots_dir,
            turbulence_config=TurbulenceConfig(
                chunks={"pressure_level": 3, "latitude": 721, "longitude": 1440},
                diagnostics=diagnostics,
                phases=TurbulencePhases(
                    calibration_phases=TurbulenceCalibrationPhases(
                        phases=[
                            TurbulenceCalibrationPhaseOption.THRESHOLDS,
                            TurbulenceCalibrationPhaseOption.HISTOGRAM,
                        ],
                        calibration_config=TurbulenceCalibrationConfig(
                            calibration_data_dir=Path("tests/_static/"),
                            percentile_thresholds=TurbulenceThresholds(
                                light=0.97, light_to_moderate=98.0, moderate=99.0
                            ),
                        ),
                    )
                ),
            ),
            data_config=DataConfig(
                spatial_domain=SpatialDomain(
                    minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180
                )
            ),
        )

    return _calibration_only_config


class CalibrationOutputFiles(NamedTuple):
    thresholds: Path | None
    distribution: Path | None


def marry_up_output_files(files: list[Path]) -> CalibrationOutputFiles:
    match files:
        case [file_path] if file_path.stem.startswith("thresholds"):
            return CalibrationOutputFiles(thresholds=files[0], distribution=None)
        case [file_path] if file_path.stem.startswith("distribution_params"):
            return CalibrationOutputFiles(thresholds=None, distribution=files[0])
        case [file_path1, file_path2] if file_path1.stem.startswith("thresholds") and file_path2.stem.startswith(
            "distribution_params"
        ):
            return CalibrationOutputFiles(thresholds=files[0], distribution=files[1])
        case [file_path1, file_path2] if file_path2.stem.startswith("thresholds") and file_path1.stem.startswith(
            "distribution_params"
        ):
            return CalibrationOutputFiles(thresholds=files[1], distribution=files[0])
        case _ as unreachable:
            raise ValueError(f"File path name combination ({unreachable}) should be unreachable")


def test_turbulence_calibration_only(create_calibration_only_config: Callable, client):
    calibration_config: ConfigContext = create_calibration_only_config(6)
    assert calibration_config.turbulence_config is not None

    TurbulenceLauncher(calibration_config).launch()

    output_dir: Path = calibration_config.output_dir / calibration_config.name
    assert output_dir.exists()
    assert output_dir.is_dir()

    output_json_files = list(output_dir.glob("*.json"))
    assert len(output_json_files) == len(calibration_config.turbulence_config.phases.calibration_phases.phases)
    output_files = marry_up_output_files(output_json_files)
    assert output_files.thresholds is not None
    assert output_files.distribution is not None

    distribution_params: dict[str, HistogramData] = DISTRIBUTION_PARAMS_TYPE_ADAPTER.validate_json(
        output_files.distribution.read_text()
    )
    thresholds: dict[str, TurbulenceThresholds] = THRESHOLDS_TYPE_ADAPTER.validate_json(
        output_files.thresholds.read_text()
    )
    calibrated_diagnostics = set(calibration_config.turbulence_config.diagnostics)
    assert calibrated_diagnostics.intersection(thresholds.keys()) == calibrated_diagnostics.union(thresholds.keys())
    assert calibrated_diagnostics.intersection(distribution_params.keys()) == calibrated_diagnostics.union(
        distribution_params.keys()
    )
