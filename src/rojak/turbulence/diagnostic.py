from abc import ABC, abstractmethod

import xarray as xr

type DiagnosticName = str


class Diagnostic(ABC):
    _name: DiagnosticName
    _computed_value: None | xr.DataArray = None

    def __init__(self, name: DiagnosticName) -> None:
        self._name = name

    @abstractmethod
    def _compute(self) -> xr.DataArray:
        pass

    @property
    def name(self) -> DiagnosticName:
        return self._name
