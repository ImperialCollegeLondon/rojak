import pytest
import xarray as xr
from distributed import Future, futures_of

from rojak.orchestrator.configuration import TurbulenceDiagnostics
from rojak.turbulence.diagnostic import (
    UBF,
    BrownIndex1,
    BrownIndex2,
    BruntVaisalaFrequency,
    ColsonPanofsky,
    DeformationSquared,
    Diagnostic,
    DiagnosticFactory,
    DirectionalShear,
    DuttonIndex,
    EDRLunnon,
    Endlich,
    Frontogenesis2D,
    Frontogenesis3D,
    GradientPotentialVorticity,
    GradientRichardson,
    HorizontalDivergence,
    HorizontalTemperatureGradient,
    MagnitudePotentialVorticity,
    Ncsu1,
    NegativeVorticityAdvection,
    NestedGridModel1,
    NestedGridModel2,
    TurbulenceIndex1,
    TurbulenceIndex2,
    VerticalVorticitySquared,
    VerticalWindShear,
    WindDirection,
    WindSpeed,
)


# IMPORTANT: This verifies that the diagnostics can run in a distributed manner. It does NOT currently verify the
#            correctness of the calculations
@pytest.mark.parametrize(
    ("diagnostic", "target_class"),
    [
        (TurbulenceDiagnostics.F2D, Frontogenesis2D),
        (TurbulenceDiagnostics.F3D, Frontogenesis3D),
        (TurbulenceDiagnostics.TEMPERATURE_GRADIENT, HorizontalTemperatureGradient),
        (TurbulenceDiagnostics.ENDLICH, Endlich),
        (TurbulenceDiagnostics.TI1, TurbulenceIndex1),
        (TurbulenceDiagnostics.TI2, TurbulenceIndex2),
        (TurbulenceDiagnostics.NCSU1, Ncsu1),
        (TurbulenceDiagnostics.COLSON_PANOFSKY, ColsonPanofsky),
        (TurbulenceDiagnostics.UBF, UBF),
        (TurbulenceDiagnostics.BRUNT_VAISALA, BruntVaisalaFrequency),
        (TurbulenceDiagnostics.VWS, VerticalWindShear),
        (TurbulenceDiagnostics.RICHARDSON, GradientRichardson),
        (TurbulenceDiagnostics.WIND_SPEED, WindSpeed),
        (TurbulenceDiagnostics.DEF, DeformationSquared),
        (TurbulenceDiagnostics.WIND_DIRECTION, WindDirection),
        (TurbulenceDiagnostics.HORIZONTAL_DIVERGENCE, HorizontalDivergence),
        (TurbulenceDiagnostics.MAGNITUDE_PV, MagnitudePotentialVorticity),
        (TurbulenceDiagnostics.PV_GRADIENT, GradientPotentialVorticity),
        (TurbulenceDiagnostics.VORTICITY_SQUARED, VerticalVorticitySquared),
        (TurbulenceDiagnostics.DIRECTIONAL_SHEAR, DirectionalShear),
        (TurbulenceDiagnostics.NGM1, NestedGridModel1),
        (TurbulenceDiagnostics.NGM2, NestedGridModel2),
        (TurbulenceDiagnostics.BROWN1, BrownIndex1),
        (TurbulenceDiagnostics.BROWN2, BrownIndex2),
        (TurbulenceDiagnostics.NVA, NegativeVorticityAdvection),
        (TurbulenceDiagnostics.DUTTON, DuttonIndex),
        (TurbulenceDiagnostics.EDR_LUNNON, EDRLunnon),
    ],
)
def test_turbulence_diagnostics_compute_on_distributed(
    client, load_cat_data, diagnostic: TurbulenceDiagnostics, target_class
) -> None:
    factory = DiagnosticFactory(load_cat_data(None, with_chunks=True))
    instantiated_diagnostic = factory.create(diagnostic)

    assert isinstance(instantiated_diagnostic, Diagnostic)
    assert isinstance(instantiated_diagnostic, target_class)

    futures = futures_of(instantiated_diagnostic._compute().persist())
    for item in futures:
        assert isinstance(item, Future)

    assert isinstance(instantiated_diagnostic.computed_value, xr.DataArray)
