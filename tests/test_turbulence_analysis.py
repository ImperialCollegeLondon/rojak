import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray

from rojak.orchestrator.configuration import TurbulenceThresholds
from rojak.turbulence.analysis import DiagnosticHistogramDistribution, HistogramData, TurbulenceIntensityThresholds


def dummy_data_for_percentiles_flattened() -> NDArray:
    return np.arange(101)


def dummy_turbulence_percentile_configs() -> TurbulenceThresholds:
    return TurbulenceThresholds(light=90, light_to_moderate=93, moderate=95, moderate_to_severe=97, severe=99)


def dummy_turbulence_percentile_configs_with_none() -> TurbulenceThresholds:
    return TurbulenceThresholds(light=90, moderate=95, severe=99)


def test_turbulence_intensity_threshold_post_processor():
    processor: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(), xr.DataArray(dummy_data_for_percentiles_flattened())
    )
    output = processor.execute()
    assert output == dummy_turbulence_percentile_configs()

    processor_dask: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(), xr.DataArray(da.asarray(dummy_data_for_percentiles_flattened()))
    )
    output_dask = processor_dask.execute()
    assert output_dask == dummy_turbulence_percentile_configs()

    processor_with_none = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs_with_none(), xr.DataArray(dummy_data_for_percentiles_flattened())
    )
    assert processor_with_none.execute() == dummy_turbulence_percentile_configs_with_none()


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
