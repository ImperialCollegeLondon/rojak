import calendar
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, NamedTuple

import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path


class Date(NamedTuple):
    year: int
    month: int
    day: int


class DataRetriever(ABC):
    @abstractmethod
    def download_files(
        self,
        years: List[int],
        months: List[int],
        days: List[int],
        base_output_dir: "Path",
    ) -> None: ...

    @abstractmethod
    def _download_file(self, date: Date, base_output_dir: "Path") -> None: ...

    @staticmethod
    def compute_date_combinations(years: list[int], months: list[int], days: list[int]) -> list[Date]:
        if len(months) == 1 and months[0] == -1:
            months = list(range(1, 13))
        if len(days) == 1 and days[0] == -1:
            return [
                Date(y, m, d)
                for y, m in itertools.product(years, months)
                for d in range(1, calendar.monthrange(y, m)[1] + 1)
            ]
        return [Date(*combination) for combination in itertools.product(years, months, days)]


class DataPreprocessor(ABC):
    @abstractmethod
    def apply_preprocessor(self, output_directory: "Path") -> None: ...


# NOTE: cf_name is the key that'll be used
@dataclass(frozen=True)
class DataVarSchema:
    database_name: str  # Name of variable in the dataset
    cf_name: str  # CF standard name for the variable


class CATPrognosticData:
    _dataset: xr.Dataset

    required_variables: ClassVar[frozenset[str]] = frozenset(
        [
            "temperature",
            "divergence_of_wind",
            "geopotential",
            "specific_humidity",
            "eastward_wind",
            "northward_wind",
            "potential_vorticity",
            "vorticity",
        ]
    )
    required_coords: ClassVar[frozenset[str]] = frozenset(["pressure_level", "latitude", "longitude", "time"])

    def __init__(self, dataset: xr.Dataset) -> None:
        if dataset.data_vars.keys() < self.required_variables:
            missing_variables = self.required_variables - dataset.data_vars.keys()
            raise ValueError(
                f"Attempting to instantiate CATPrognosticData with missing data variables: {missing_variables}"
            )
        if dataset.coords.keys() < self.required_coords:
            missing_coords = self.required_coords - dataset.coords.keys()
            raise ValueError(f"Attempting to instantiate CATPrognosticData with missing coords: {missing_coords}")
        self._dataset = dataset

    def temperature(self) -> xr.DataArray:
        return self._dataset["temperature"]

    def divergence(self) -> xr.DataArray:
        return self._dataset["divergence_of_wind"]

    def geopotential(self) -> xr.DataArray:
        return self._dataset["geopotential"]

    def specific_humidity(self) -> xr.DataArray:
        return self._dataset["specific_humidity"]

    def u_wind(self) -> xr.DataArray:
        return self._dataset["eastward_wind"]

    def v_wind(self) -> xr.DataArray:
        return self._dataset["northward_wind"]

    def potential_vorticity(self) -> xr.DataArray:
        return self._dataset["potential_vorticity"]

    def vorticity(self) -> xr.DataArray:
        return self._dataset["vorticity"]


class MetData(ABC):
    @abstractmethod
    def to_clear_air_turbulence_data(self) -> CATPrognosticData: ...

    # TODO: TEST
    @staticmethod
    def pressure_to_altitude_std_atm(pressure: xr.DataArray) -> xr.DataArray:
        """
        Equation 3.106 on page 106 in Wallace, J. M., and Hobbs, P. V., “Atmospheric Science: An Introductory Survey,”
        Elsevier Science & Technology, San Diego, UNITED STATES, 2006.
        z = \frac{T_0}{\\Gamma} \\left[ 1 - \\left( \frac{p}{p_0} \right)^{\frac{R\\Gamma}{g}} \right]
        """
        reference_temperature: float = 288.0  # kelvin
        gamma: float = 6.5  # K/km
        reference_pressure: float = 1013.25  # hPa
        gas_constant_dry_air: float = 297  # J / (K kg)
        gravitational_acceleration: float = 9.80665  # m / s^2
        return (reference_temperature / gamma) * (
            1
            - (pressure / reference_pressure)  # fmt: skip
            ** ((gas_constant_dry_air * gamma) / gravitational_acceleration)
        )

    # To be added later
    # @abstractmethod
    # def to_contrails_data(self) -> xr.Dataset: ...
