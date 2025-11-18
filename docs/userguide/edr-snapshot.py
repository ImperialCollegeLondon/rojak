import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pypalettes
import xarray as xr
from dask.distributed import Client

from rojak.orchestrator.configuration import Context as ConfigContext
from rojak.orchestrator.configuration import TurbulenceDiagnostics, TurbulenceEvaluationPhaseOption
from rojak.orchestrator.turbulence import TurbulenceLauncher
from rojak.plot.turbulence_plotter import diagnostic_label_mapping

if TYPE_CHECKING:
    from rojak.orchestrator.turbulence import EvaluationStageResult

if __name__ == "__main__":
    client: Client = Client()
    config_file = Path(sys.argv[1])
    assert config_file.exists()
    assert config_file.is_file()
    assert config_file.suffix in {".yaml", ".yml"}
    context = ConfigContext.from_yaml(config_file)

    eval_stage_result: "None | EvaluationStageResult" = TurbulenceLauncher(context).launch()
    assert eval_stage_result is not None

    # Verify that EDR was computed, if it wasn't check input config
    assert TurbulenceEvaluationPhaseOption.EDR in eval_stage_result.phase_outcomes

    edr = eval_stage_result.phase_outcomes[TurbulenceEvaluationPhaseOption.EDR].result
    as_ds = xr.Dataset(
        data_vars={name: diagnostic.isel(time=0).sel(pressure_level=200) for name, diagnostic in edr.items()}
    )

    projection = ccrs.LambertConformal(
        central_longitude=(-45),
        central_latitude=35,
    )

    chained_names: str = "_".join(eval_stage_result.suite.diagnostic_names())
    names = [str(item) for item in eval_stage_result.suite.diagnostic_names()]
    # pyright thinks xr.plot doesn't exists...
    fg: xr.plot.FacetGrid = (  # pyright: ignore [reportAttributeAccessIssue, reportCallIssue]
        as_ds[names]
        .to_dataarray("diagnostics")
        .plot(
            col="diagnostics",
            x="longitude",
            y="latitude",
            col_wrap=min(3, len(names)),
            transform=ccrs.PlateCarree(),
            vmin=0,
            vmax=0.8,
            subplot_kws={"projection": projection},
            cbar_kwargs={
                "label": "EDR",
                "orientation": "horizontal",
                "spacing": "uniform",
                "pad": 0.02,
                "shrink": 0.6,
            },
            # cmap="Blues",
            cmap=pypalettes.load_cmap("cancri", cmap_type="continuous", reverse=True).resampled(8),
            robust=True,
        )
    )
    for ax, diagnostic in zip(
        fg.axs.flat, eval_stage_result.suite.diagnostic_names(), strict=False
    ):  # GeoAxes, TurbulenceDiagnostic
        ax.set_title(diagnostic_label_mapping[TurbulenceDiagnostics(diagnostic)])
    fg.map(lambda: plt.gca().coastlines())  # pyright: ignore[reportAttributeAccessIssue]
    fg.fig.savefig(str(context.plots_dir / f"multi_edr_{chained_names}.{context.image_format}"), bbox_inches="tight")
    plt.close(fg.fig)

    client.close()
