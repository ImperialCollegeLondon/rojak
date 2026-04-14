import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import xarray as xr
from dask.distributed import Client

from rojak.orchestrator.configuration import Context as ConfigContext
from rojak.orchestrator.configuration import TurbulenceEvaluationPhaseOption
from rojak.orchestrator.turbulence import TurbulenceLauncher
from rojak.plot.turbulence_plotter import (
    chain_diagnostic_names,
    create_configurable_multi_diagnostic_plot,
)
from rojak.plot.utilities import (
    StandardColourMaps,
    get_a_default_cmap,
)

if TYPE_CHECKING:
    from rojak.orchestrator.turbulence import EvaluationStageResult

if __name__ == "__main__":
    # Start dask to run in distributed manner
    client: Client = Client()

    # Load config file passed on as first argument
    config_file = Path(sys.argv[1])
    assert config_file.exists()
    assert config_file.is_file()
    assert config_file.suffix in {".yaml", ".yml"}

    # Deserialize data stored in yaml file
    context = ConfigContext.from_yaml(config_file)

    # Launch the turbulence analysis to get the result from the evaluation stage
    eval_stage_result: "None | EvaluationStageResult" = TurbulenceLauncher(context).launch()
    assert eval_stage_result is not None

    # Verify that EDR was computed, if it wasn't check input config
    assert TurbulenceEvaluationPhaseOption.EDR in eval_stage_result.phase_outcomes

    # Get computed EDR from the evaluation stage
    edr = eval_stage_result.phase_outcomes[TurbulenceEvaluationPhaseOption.EDR].result
    names = [str(item) for item in eval_stage_result.suite.diagnostic_names()]

    # Plot the first time step at 200hPa
    create_configurable_multi_diagnostic_plot(
        xr.Dataset(
            data_vars={name: diagnostic.isel(time=0).sel(pressure_level=200) for name, diagnostic in edr.items()}
        ),
        names,
        str(context.plots_dir / f"multi_edr_{chain_diagnostic_names(names)}.{context.image_format}"),
        column="diagnostics",
        plot_kwargs={
            "subplot_kws": {
                "projection": ccrs.LambertConformal(
                    central_longitude=(-45),
                    central_latitude=35,
                )
            },
            "cbar_kwargs": {
                "label": "EDR",
                "orientation": "horizontal",
                "spacing": "uniform",
                "pad": 0.02,
                "shrink": 0.6,
                "extend": "max",
            },
            "vmin": 0,
            "vmax": 0.8,
            "col_wrap": min(3, len(names)),
            "cmap": get_a_default_cmap(StandardColourMaps.TURBULENCE_PROBABILITY, resample_to=8),
        },
        savefig_kwargs={"bbox_inches": "tight"},
    )

    # Close dask client
    client.close()
