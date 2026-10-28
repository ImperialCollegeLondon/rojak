from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import config as dask_config
from dask.base import is_dask_collection
from numpy.typing import NDArray

from rojak.atmosphere.contrails import issr
from rojak.atmosphere.jet_stream import AlphaVelField, JetStreamAlgorithmFactory
from rojak.orchestrator.configuration import (
    JetStreamAlgorithms,
    RelationshipBetweenTypes,
    TurbulenceDiagnostics,
    TurbulenceThresholds,
)
from rojak.turbulence.analysis import (
    DiagnosticHistogramDistribution,
    HistogramData,
    MatthewsCorrelationOnDataset,
    RelationshipBetweenAlphaVelAndTurbulence,
    RelationshipBetweenXAndTurbulence,
    TransformToEDR,
    TurbulenceIntensityThresholds,
)
from rojak.turbulence.diagnostic import DiagnosticFactory

if TYPE_CHECKING:
    from rojak.core.data import CATData


def dummy_data_for_percentiles_flattened() -> NDArray:
    return np.arange(101)


def dummy_turbulence_percentile_configs() -> TurbulenceThresholds:
    return TurbulenceThresholds(light=90, light_to_moderate=93, moderate=95, moderate_to_severe=97, severe=99)


def dummy_turbulence_percentile_configs_with_none() -> TurbulenceThresholds:
    return TurbulenceThresholds(light=90, moderate=95, severe=99)


def test_turbulence_intensity_threshold_post_processor() -> None:
    processor: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(),
        xr.DataArray(dummy_data_for_percentiles_flattened()),
    )
    output = processor.execute()
    assert output == dummy_turbulence_percentile_configs()

    processor_dask: TurbulenceIntensityThresholds = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs(),
        xr.DataArray(da.asarray(dummy_data_for_percentiles_flattened())),
    )
    output_dask = processor_dask.execute()
    desired = dummy_turbulence_percentile_configs()
    # tdigest method for percentile no longer gives the exact value due to the interpolation
    # assert output_dask == dummy_turbulence_percentile_configs()
    np.testing.assert_allclose(
        np.asarray(
            [
                output_dask.light,
                output_dask.light_to_moderate,
                output_dask.moderate,
                output_dask.moderate_to_severe,
                output_dask.severe,
            ],
        ),
        np.asarray(
            [desired.light, desired.light_to_moderate, desired.moderate, desired.moderate_to_severe, desired.severe],
        ),
        rtol=0.005,
    )

    processor_with_none = TurbulenceIntensityThresholds(
        dummy_turbulence_percentile_configs_with_none(),
        xr.DataArray(dummy_data_for_percentiles_flattened()),
    )
    assert processor_with_none.execute() == dummy_turbulence_percentile_configs_with_none()


def test_histogram_distribution_serial_and_parallel_match() -> None:
    serial_data_array = xr.DataArray(np.arange(10001))
    parallel_data_array = xr.DataArray(da.asarray(np.arange(10001), chunks=100))
    assert is_dask_collection(parallel_data_array)
    parallel_result: HistogramData = DiagnosticHistogramDistribution(parallel_data_array).execute()
    serial_result: HistogramData = DiagnosticHistogramDistribution(serial_data_array).execute()

    tolerance: float = 1e-8
    assert parallel_result.hist_values == serial_result.hist_values
    assert parallel_result.bins == serial_result.bins
    assert np.abs(parallel_result.mean - serial_result.mean) < tolerance
    assert np.abs(parallel_result.variance - serial_result.variance) < tolerance


def _dummy_diagnostic_for_edr() -> xr.DataArray:
    rng = np.random.default_rng(3)
    # Include negative values to exercise the clipping
    return xr.DataArray(rng.lognormal(size=(4, 6, 8)) - 0.1, dims=("time", "latitude", "longitude"))


def test_transform_to_edr_serial_and_parallel_match() -> None:
    serial_data_array = _dummy_diagnostic_for_edr()
    parallel_data_array = serial_data_array.chunk({"time": 1})
    serial_result = TransformToEDR(serial_data_array, mean=-0.5, variance=0.8).execute()
    parallel_result = TransformToEDR(parallel_data_array, mean=-0.5, variance=0.8).execute()

    assert is_dask_collection(parallel_result)
    xr.testing.assert_allclose(parallel_result.compute(), serial_result)


def test_transform_to_edr_is_lazy() -> None:
    # Regression test: execute() must not trigger any computation (e.g. via persist) so that callers which only use
    # the result once (e.g. the ensemble EDR export) don't materialise every diagnostic's EDR in memory
    def fail_if_computed(*args, **kwargs) -> None:
        raise AssertionError("TransformToEDR.execute() triggered a computation")

    parallel_data_array = _dummy_diagnostic_for_edr().chunk({"time": 1})
    with dask_config.set(scheduler=fail_if_computed):
        result = TransformToEDR(parallel_data_array, mean=-0.5, variance=0.8).execute()
    assert is_dask_collection(result)


THRESHOLDS = {
    "richardson": -0.02333501569770461,
    "ngm1": 1.5464450075189453e-07,
    "ngm2": 2.8370187359291123e-11,
    "ti1": 1.9974807296001738e-11,
    "ti2": 1.3336295227041907e-07,
    "ubf": 0.0017884688715023437,
    "f2d": 3.593564906824321e-10,
    "f3d": 1.6440732248102948e-07,
    "temperature_gradient": 7.818259539721061e-06,
    "ncsu1": 3.0043645392271477e-08,
    "deformation": 2.1893555939536445e-17,
    "directional_shear": 2.7019866593036568e-06,
    "endlich": 7.752662349957973e-05,
    "wind_speed": 27.6774845123291,
    "colson_panofsky": 14.131871996453508,
    "vertical_wind_shear": 0.005864380393177271,
    "magnitude_pv": 1.112234986067051e-05,
    "gradient_pv": 5.890546917560539e-12,
    "nva": 6.927956677982809e-09,
    "dutton": 10.50140212904902,
    "edr_lunnon": 5.125841076049772e-07,
    "brown1": 9.900932255838597e-05,
    "brown2": 1.0176133094965872e-10,
    "vorticity_squared": 5.136512104542135e-09,
}


# NOTE: Test checks that it runs and NOT the correctness
@pytest.mark.parametrize("relationship_type", [rr.value for rr in RelationshipBetweenTypes])
def test_relationship_between_issr_and_turbulence(load_cat_data, client, relationship_type: RelationshipBetweenTypes):
    cat_data: CATData = load_cat_data(None, with_chunks=True)
    issr_ = issr(
        cat_data.temperature(),
        specific_humidity=cat_data.specific_humidity(),
        air_pressure=cat_data.temperature()["pressure_level"] * 100,
    )
    diagnostic_factory = DiagnosticFactory(cat_data)
    computed_rel = RelationshipBetweenXAndTurbulence(
        issr_,
        xr.Dataset(
            data_vars={
                name: diagnostic_factory.create(TurbulenceDiagnostics(name)).computed_value for name in THRESHOLDS
            },
            coords=cat_data.temperature().coords,
        ),
        relationship_type,
        diagnostic_thresholds=THRESHOLDS,
        feature_name="issr",
    ).execute()
    assert set(computed_rel.data_vars.keys()).issubset(THRESHOLDS.keys())

    for data_var in computed_rel.data_vars.values():
        assert data_var.any(dim=None), "Test that there is at least one true value in the whole array"


# NOTE: Test checks that it runs and NOT the correctness
@pytest.mark.parametrize("relationship_type", [rr.value for rr in RelationshipBetweenTypes])
def test_relationship_between_alpha_vel_jet_stream_and_turbulence(
    load_cat_data,
    client,
    relationship_type: RelationshipBetweenTypes,
):
    cat_data: CATData = load_cat_data(None, with_chunks=True)
    diagnostic_factory = DiagnosticFactory(cat_data)
    computed_rel = RelationshipBetweenAlphaVelAndTurbulence(
        cast("AlphaVelField", JetStreamAlgorithmFactory(cat_data).create(JetStreamAlgorithms.ALPHA_VEL_KOCH)),
        xr.Dataset(
            data_vars={
                name: diagnostic_factory.create(TurbulenceDiagnostics(name)).computed_value for name in THRESHOLDS
            },
            coords=cat_data.temperature().coords,
        ),
        relationship_type,
        diagnostic_thresholds=THRESHOLDS,
    ).execute()

    assert set(computed_rel.data_vars.keys()).issubset(THRESHOLDS.keys())

    for data_var in computed_rel.data_vars.values():
        assert data_var.any(dim=None), "Test that there is at least one true value in the whole array"


def _numpy_matthews_corr_coeff(truth: NDArray, prediction: NDArray) -> float:
    tp = np.sum(truth & prediction)
    tn = np.sum(~truth & ~prediction)
    fp = np.sum(~truth & prediction)
    fn = np.sum(truth & ~prediction)
    return float((tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))


def test_matthews_correlation_on_dataset_matches_numpy() -> None:
    rng = np.random.default_rng(4)
    shape = (6, 20, 30)
    first = rng.random(shape) < 0.4  # noqa: PLR2004
    # Correlated with first so that the coefficient is non-trivial
    second = first ^ (rng.random(shape) < 0.2)  # noqa: PLR2004
    third = rng.random(shape) < 0.5  # noqa: PLR2004
    dims = ("time", "latitude", "longitude")
    dataset = xr.Dataset({name: (dims, values) for name, values in (("a", first), ("b", second), ("c", third))}).chunk(
        {"time": 1}
    )

    result = MatthewsCorrelationOnDataset(dataset, "diagnostic").execute()

    for name1, values1 in (("a", first), ("b", second), ("c", third)):
        for name2, values2 in (("a", first), ("b", second), ("c", third)):
            expected = 1.0 if name1 == name2 else _numpy_matthews_corr_coeff(values1.ravel(), values2.ravel())
            np.testing.assert_allclose(result.sel(diagnostic1=name1, diagnostic2=name2).item(), expected)
