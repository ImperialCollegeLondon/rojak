import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from rojak.orchestrator.configuration import TurbulenceSeverityPercentileConfig
from rojak.turbulence.analysis import TurbulenceIntensityThresholds


def dummy_data_for_percentiles_flattened() -> NDArray:
    return np.arange(101)


def dummy_turbulence_percentile_configs() -> list[TurbulenceSeverityPercentileConfig]:
    return [TurbulenceSeverityPercentileConfig(name="light", lower_bound=97, upper_bound=99)]


def test_turbulence_intensity_threshold_post_processor():
    processor: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(), xr.DataArray(dummy_data_for_percentiles_flattened())
    )
    output = processor.execute()
    assert output["light"] == 97

    processor: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(), xr.DataArray(da.asarray(dummy_data_for_percentiles_flattened()))
    )
    output = processor.execute()
    assert output["light"] == 97
