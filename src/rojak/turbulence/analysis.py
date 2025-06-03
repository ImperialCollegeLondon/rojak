import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray

from rojak.core.analysis import PostProcessor
from rojak.orchestrator.configuration import TurbulenceSeverityPercentileConfig

type IntensityName = str
type IntensityValues = dict[IntensityName, float]


class TurbulenceIntensityThresholds(PostProcessor):
    _threshold_configs: list[TurbulenceSeverityPercentileConfig]
    _computed_diagnostic: xr.DataArray

    def __init__(
        self, threshold_config: list[TurbulenceSeverityPercentileConfig], computed_diagnostic: xr.DataArray
    ) -> None:
        self._threshold_configs = threshold_config
        self._computed_diagnostic = computed_diagnostic

    def _compute_percentiles(self, target_percentiles: list[float]) -> NDArray:
        if is_dask_collection(self._computed_diagnostic):
            # flattened_array = da.asarray(self._computed_diagnostic.data, chunks="auto").flatten()
            flattened_array = da.asarray(self._computed_diagnostic, chunks="auto").flatten()
            return da.percentile(flattened_array, target_percentiles, internal_method="dask").compute()
        return np.percentile(self._computed_diagnostic.stack(all=[...]), target_percentiles)

    def _compute_lower_bounds(self) -> NDArray:
        lower_bounds = [config.lower_bound for config in self._threshold_configs]
        return self._compute_percentiles(lower_bounds)

    def execute(self) -> IntensityValues:
        percentiles = self._compute_lower_bounds()
        return {str(config.name): float(percentile) for config, percentile in zip(self._threshold_configs, percentiles)}
