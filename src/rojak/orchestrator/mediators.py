from typing import TYPE_CHECKING, NamedTuple, Sequence

from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.turbulence.diagnostic import DiagnosticSuite


Coordinate = NamedTuple("Coordinate", [("latitude", float), ("longitude", float)])


def bilinear_interpolation(
    longitude: Sequence[float], latitude: Sequence[float], function_value: "NDArray", target_coordinate: Coordinate
) -> float:
    assert len(longitude) == len(latitude)
    assert len(longitude) > 1
    assert function_value.ndim == 2  # noqa: PLR2004

    return RegularGridInterpolator((longitude, latitude), function_value.T, method="linear")(
        (target_coordinate.longitude, target_coordinate.latitude)
    )[0]


class DiagnosticsAmdarComparator:
    """
    Class handles the comparison of diagnostics computed from meteorological data and AMDAR observational data
    """

    _amdar_data: "AmdarTurbulenceData"
    _diagnostics_suite: "DiagnosticSuite"

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "DiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite
