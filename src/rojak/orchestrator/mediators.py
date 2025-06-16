from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rojak.core.data import AmdarTurbulenceData
    from rojak.turbulence.diagnostic import DiagnosticSuite


class DiagnosticsAmdarComparator:
    """
    Class handles the comparison of diagnostics computed from meteorological data and AMDAR observational data
    """

    _amdar_data: "AmdarTurbulenceData"
    _diagnostics_suite: "DiagnosticSuite"

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "DiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite
