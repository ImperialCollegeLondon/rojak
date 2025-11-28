#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
from typing import TYPE_CHECKING, ClassVar, Literal

import cdsapi
from rich.progress import track

from rojak.core.calculations import pressure_to_altitude_icao
from rojak.core.data import CATData, DataRetriever, DataVarSchema, MetData
from rojak.datalib.ecmwf.constants import (
    blank_default,
    data_defaults,
    reanalysis_dataset_names,
    six_hourly,
)

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from rojak.core.data import Date
    from rojak.orchestrator.configuration import SpatialDomain

logger = logging.getLogger(__name__)


class InvalidEra5RequestConfigurationError(Exception):
    def __init__(self, message: str) -> None:
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
                    "Default not specified. As such, which variables and pressure levels must be specified.",
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
        years: list[int],
        months: list[int],
        days: list[int],
        base_output_dir: "Path",
    ) -> None:
        dates: list[Date] = self.compute_date_combinations(years, months, days)
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


class Era5Data(MetData):
    # Instance variables
    _on_pressure_level: "xr.Dataset"

    # Class variables which are not set on an instance
    temperature: ClassVar[DataVarSchema] = DataVarSchema("t", "temperature")
    divergence: ClassVar[DataVarSchema] = DataVarSchema("d", "divergence_of_wind")
    geopotential: ClassVar[DataVarSchema] = DataVarSchema("z", "geopotential")
    specific_humidity: ClassVar[DataVarSchema] = DataVarSchema("q", "specific_humidity")
    eastward_wind: ClassVar[DataVarSchema] = DataVarSchema("u", "eastward_wind")
    northward_wind: ClassVar[DataVarSchema] = DataVarSchema("v", "northward_wind")
    potential_vorticity: ClassVar[DataVarSchema] = DataVarSchema("pv", "potential_vorticity")
    vorticity: ClassVar[DataVarSchema] = DataVarSchema("vo", "vorticity")
    vertical_velocity: ClassVar[DataVarSchema] = DataVarSchema("w", "vertical_velocity")

    def __init__(self, on_pressure_level: "xr.Dataset") -> None:
        super().__init__()
        self._on_pressure_level = on_pressure_level

    def to_clear_air_turbulence_data(self, domain: "SpatialDomain") -> CATData:
        logger.debug("Converting data to CATData")
        target_variables: list[DataVarSchema] = [
            Era5Data.temperature,
            Era5Data.divergence,
            Era5Data.geopotential,
            Era5Data.specific_humidity,
            Era5Data.eastward_wind,
            Era5Data.northward_wind,
            Era5Data.potential_vorticity,
            Era5Data.vorticity,
        ]
        target_var_names: list[str] = [var.database_name for var in target_variables]
        target_data: xr.Dataset = self._on_pressure_level[target_var_names]
        # On ERA5 data 0 < longitude < 360 => shift to make it -180 < longitude < 180
        target_data = self.shift_longitude(target_data)
        target_data = target_data.rename({"valid_time": "time"})
        target_data = self.select_domain(domain, target_data, level_coordinate_name="pressure_level")
        target_data = target_data.rename_vars({var.database_name: var.cf_name for var in target_variables})
        target_data = target_data.assign_coords(
            altitude=(
                "pressure_level",
                pressure_to_altitude_icao(target_data["pressure_level"].to_numpy()),
            ),
        )
        target_data = target_data.transpose("latitude", "longitude", "time", "pressure_level")
        return CATData(target_data)
