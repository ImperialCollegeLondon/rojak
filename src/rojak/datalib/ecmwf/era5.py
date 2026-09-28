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
"""
ECMWF ERA5 reanalysis data retrieval and adaptation into :class:`~rojak.core.data.CATData`

This module implements the :mod:`rojak.core.data` interfaces for the ERA5 reanalysis product: downloading data
from the Copernicus Climate Data Store (:class:`Era5Retriever`), and adapting a loaded ERA5 dataset into the
:class:`~rojak.core.data.CATData` required by :mod:`rojak.turbulence` (:class:`Era5Data`).
"""

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, override

import cdsapi
from rich.progress import track

from rojak.core.calculations import pressure_to_altitude_icao
from rojak.core.data import CATData, DataRetriever, DataVarSchema, MetData
from rojak.datalib.ecmwf._constants import (
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
    """Raised when a CDS API request cannot be built from the arguments given to :class:`Era5Retriever`"""

    def __init__(self, message: str) -> None:
        """
        Args:
            message: Description of why the request configuration is invalid
        """
        super().__init__(message)


# type Era5DefaultsName = Literal["cat", "surface", "contrail", "minimal-cat-contrail"] | None
# type Era5DatasetName = Literal["pressure-level", "single-level"]


class Era5DatasetName(StrEnum):
    PRESSURE_LEVEL = "pressure-level"
    SINGLE_LEVEL = "single-level"


class Era5DefaultsName(StrEnum):
    CAT = "cat"
    SURFACE = "surface"
    CONTRAIL = "contrail"
    MINIMAL_CAT_CONTRAIL = "minimal-cat-contrail"


class Era5Retriever(DataRetriever):
    """Downloads ERA5 reanalysis data from the Copernicus Climate Data Store (CDS) via :mod:`cdsapi`"""

    request_body: dict
    request_dataset_name: str
    cds_client: cdsapi.Client
    folder_name: str

    def __init__(
        self,
        dataset_name: Era5DatasetName,
        folder_name: str,
        default_name: Era5DefaultsName | None = None,
        pressure_levels: list[int] | None = None,
        variables: list[str] | None = None,
        times: list[str] | None = None,
    ) -> None:
        """
        Args:
            dataset_name: ERA5 dataset to request from, either pressure-level or single-level data
            folder_name: Name of the subdirectory (within ``base_output_dir`` passed to :meth:`download_files`) to
                download the files into
            default_name: Name of a default request body to use as the base of the request.
                If ``None``, an empty request body is used and ``pressure_levels``/``variables`` must be
                provided instead.
            pressure_levels: Pressure levels (in hPa) to request. If provided, overrides the levels in
                ``default_name``'s request body. Required if ``default_name`` is ``None`` and ``dataset_name`` is
                ``"pressure-level"``.
            variables: Variables to request. If provided, overrides the variables in ``default_name``'s request
                body. Required if ``default_name`` is ``None``.
            times: Times of day to request. If provided, overrides the times in ``default_name``'s request body.
                Defaults to six hourly if not given.

        Raises:
            InvalidEra5RequestConfigurationError: If ``default_name`` is ``None`` and ``pressure_levels`` (for a
                pressure-level dataset) or ``variables`` is not provided
        """
        if default_name is None:
            if pressure_levels is None and dataset_name == Era5DatasetName.PRESSURE_LEVEL:
                raise InvalidEra5RequestConfigurationError(
                    "Default not specified. As such, which pressure levels must be specified.",
                )
            if variables is None:
                raise InvalidEra5RequestConfigurationError(
                    "Default not specified. As such, which must be specified.",
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

    @override
    def download_files(
        self,
        years: list[int],
        months: list[int],
        days: list[int],
        base_output_dir: "Path",
    ) -> None:
        """
        Download the ERA5 data file for every combination of ``years``, ``months``, and ``days``

        Args:
            years: Years to download data for
            months: Months to download data for. ``[-1]`` means every month.
            days: Days to download data for. ``[-1]`` means every day of the month.
            base_output_dir: Directory containing :attr:`folder_name`, which the files are downloaded into
        """
        dates: list[Date] = self.compute_date_combinations(years, months, days)
        (base_output_dir / self.folder_name).resolve().mkdir(parents=True, exist_ok=True)
        for date in track(dates):
            self._download_file(date, base_output_dir)

    @override
    def _download_file(self, date: "Date", base_output_dir: "Path") -> None:
        """
        Download the ERA5 data file for a single date via the CDS API

        Args:
            date: Date to download the data file for
            base_output_dir: Directory containing :attr:`folder_name`, which the file is downloaded into
        """
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
    """
    Adapts a loaded ERA5 dataset on pressure levels into the :class:`~rojak.core.data.CATData` required by
    :mod:`rojak.turbulence`

    The class-level :class:`~rojak.core.data.DataVarSchema` attributes map each ERA5 variable's short name (as
    stored in the raw dataset) to its CF standard name (as required by :class:`~rojak.core.data.CATPrognosticData`).
    """

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
        """
        Args:
            on_pressure_level: Raw ERA5 dataset on pressure levels, as downloaded by :class:`Era5Retriever`
        """
        super().__init__()
        self._on_pressure_level = on_pressure_level

    @override
    def to_clear_air_turbulence_data(self, domain: "SpatialDomain") -> CATData:
        """
        Adapt the raw ERA5 dataset into a :class:`~rojak.core.data.CATData`, restricted to ``domain``

        Selects the variables required for CAT diagnostics, shifts longitude from ``[0, 360)`` to ``[-180, 180)``,
        renames the time and variable names to the conventions expected by
        :class:`~rojak.core.data.CATPrognosticData`, restricts the data to ``domain`` (see
        :meth:`~rojak.core.data.MetData.select_domain`), and computes the altitude coordinate from pressure level.

        Args:
            domain: Spatial (and optionally vertical) domain to build the data for

        Returns:
            :class:`~rojak.core.data.CATData` containing the fields required to compute CAT diagnostics over
            ``domain``
        """
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
        target_data = self.shift_ds_longitude(target_data)
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
        return CATData(target_data, pressure_level_prefix=100)  # pressure_level in hPa
