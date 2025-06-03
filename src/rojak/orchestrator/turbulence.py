from datetime import datetime
from typing import TYPE_CHECKING, Mapping

from pydantic import TypeAdapter

from rojak.core import data
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import TurbulenceThresholds
from rojak.turbulence.analysis import TurbulenceIntensityThresholds
from rojak.turbulence.diagnostic import DiagnosticFactory, DiagnosticName, DiagnosticSuite

if TYPE_CHECKING:
    from pathlib import Path

    from rojak.core.data import CATData
    from rojak.orchestrator.configuration import Context as ConfigContext
    from rojak.orchestrator.configuration import TurbulenceConfig


class TurbulenceLauncher:
    _config: "TurbulenceConfig"
    _context: "ConfigContext"

    def __init__(self, config: "TurbulenceConfig", context: "ConfigContext") -> None:
        self._config = config
        self._context = context
        self._start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    def launch(self) -> None: ...

    def calibration_stage(self) -> None:
        assert self._config.calibration_data_dir is not None
        calibration_data: "CATData" = Era5Data(
            data.load_from_folder(self._config.calibration_data_dir)
        ).to_clear_air_turbulence_data()
        suite = DiagnosticSuite(DiagnosticFactory(calibration_data), self._config.diagnostics)

        assert self._config.percentile_thresholds is not None
        thresholds = self.compute_thresholds(suite, self._config.percentile_thresholds)
        self.export_thresholds(thresholds)

        # Add in some machinery to do apply the other optional post-processors
        ...

    def export_thresholds(self, diagnostic_thresholds: Mapping[DiagnosticName, "TurbulenceThresholds"]) -> None:
        target_dir: "Path" = (self._context.output_dir / self._context.name).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file: "Path" = target_dir / f"thresholds_{self._start_time}.json"

        type_adapter: TypeAdapter = TypeAdapter(dict[DiagnosticName, TurbulenceThresholds])
        with output_file.open("wb") as output_json:
            output_json.write(type_adapter.dump_json(diagnostic_thresholds, indent=4))

    @staticmethod
    def compute_thresholds(
        suite: DiagnosticSuite, percentile_config: "TurbulenceThresholds"
    ) -> Mapping[DiagnosticName, "TurbulenceThresholds"]:
        return {
            name: TurbulenceIntensityThresholds(percentile_config, diagnostic).execute()
            for name, diagnostic in suite.computed_values()  # DiagnosticName, xr.DataArray
        }
