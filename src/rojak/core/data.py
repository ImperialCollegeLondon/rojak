import calendar
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path
    import xarray as xr


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
    def compute_date_combinations(
        years: list[int], months: list[int], days: list[int]
    ) -> list[Date]:
        if len(months) == 1 and months[0] == -1:
            months = list(range(1, 13))
        if len(days) == 1 and days[0] == -1:
            return [
                Date(y, m, d)
                for y, m in itertools.product(years, months)
                for d in range(1, calendar.monthrange(y, m)[1] + 1)
            ]
        return [
            Date(*combination) for combination in itertools.product(years, months, days)
        ]


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

    def __init__(self, dataset: xr.Dataset) -> None:
        self._dataset = dataset


class MetData(ABC):
    @abstractmethod
    def to_clear_air_turbulence_data(self) -> CATPrognosticData: ...

    # To be added later
    # @abstractmethod
    # def to_contrails_data(self) -> xr.Dataset: ...
