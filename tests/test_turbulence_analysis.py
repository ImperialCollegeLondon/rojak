import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray

from rojak.orchestrator.configuration import TurbulenceSeverityPercentileConfig
from rojak.turbulence.analysis import DiagnosticHistogramDistribution, HistogramData, TurbulenceIntensityThresholds


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


def test_histogram_distribution_serial_and_parallel_match():
    serial_data_array = xr.DataArray(np.arange(10001))
    parallel_data_array = xr.DataArray(da.asarray(np.arange(10001), chunks=100))
    assert is_dask_collection(parallel_data_array)
    parallel_result: HistogramData = DiagnosticHistogramDistribution(parallel_data_array).execute()
    serial_result: HistogramData = DiagnosticHistogramDistribution(serial_data_array).execute()

    assert parallel_result.hist_values == serial_result.hist_values
    assert parallel_result.bins == serial_result.bins
    assert parallel_result.mean == serial_result.mean
    assert np.abs(parallel_result.variance - serial_result.variance) < 1e-8
