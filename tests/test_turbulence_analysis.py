from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
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
    RelationshipBetweenAlphaVelAndTurbulence,
    RelationshipBetweenXAndTurbulence,
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
