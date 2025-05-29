from typing import TYPE_CHECKING, List, Literal

import cdsapi
from rich.progress import track

from rojak.core.data import DataRetriever
from rojak.datalib.ecmwf.constants import (
    blank_default,
    data_defaults,
    reanalysis_dataset_names,
    six_hourly,
)

if TYPE_CHECKING:
    from pathlib import Path

    from rojak.core.data import Date


class InvalidEra5RequestConfigurationError(Exception):
    def __init__(self, message):
        super().__init__(message)


type Era5DefaultsName = Literal["cat", "surface", "contrail"] | None
type Era5DatasetName = Literal["pressure-level", "single-level"]


class Era5Retriever(DataRetriever):
    request_body: dict
    request_dataset_name: str
    cds_client: cdsapi.Client
    folder_name: str

    def __init__(
        self,
        dataset_name: Era5DatasetName,
        folder_name: str,
        default_name: Era5DefaultsName = None,
        pressure_levels: list[int] | None = None,
        variables: list[str] | None = None,
        times: list[str] | None = None,
    ) -> None:
        print(default_name)
        if default_name is None:
            if pressure_levels is None or variables is None:
                raise InvalidEra5RequestConfigurationError(
                    "Default not specified. As such, which variables and pressure levels must be specified."
                )
            self.request_body = blank_default
        else:
            self.request_body = data_defaults[default_name]

        if pressure_levels is not None:
            self.request_body["pressure_level"] = pressure_levels

        if variables is not None:
            self.request_body["variable"] = variables

        if times is not None:
            self.request_body["time"] = times
        else:
            self.request_body["time"] = six_hourly

        self.folder_name = folder_name
        self.request_dataset_name = reanalysis_dataset_names[dataset_name]
        self.cds_client: cdsapi.Client = cdsapi.Client()

    def download_files(
        self,
        years: List[int],
        months: List[int],
        days: List[int],
        base_output_dir: "Path",
    ) -> None:
        dates: list["Date"] = self.compute_date_combinations(years, months, days)
        (base_output_dir / self.folder_name).resolve().mkdir(parents=True, exist_ok=True)
        for date in track(dates):
            self._download_file(date, base_output_dir)

    def _download_file(self, date: "Date", base_output_dir: "Path") -> None:
        this_request = self.request_body
        this_request["year"] = date.year
        this_request["month"] = date.month
        this_request["day"] = date.day
        self.cds_client.retrieve(
            self.request_dataset_name,
            this_request,
            target=(base_output_dir / self.folder_name / f"{date.year}-{date.month}-{date.day}.nc"),
        )
