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

import calendar
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, List, Literal, Mapping, NamedTuple, Sequence

import dask_geopandas as dgpd
import numpy as np
import xarray as xr

from rojak.core import derivatives
from rojak.core.calculations import pressure_to_altitude_std_atm
from rojak.core.constants import MAX_LONGITUDE
from rojak.core.derivatives import VelocityDerivative
from rojak.core.geometric import create_grid_data_frame
from rojak.core.indexing import make_value_based_slice
from rojak.turbulence import calculations as turb_calc
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd
    from numpy.typing import NDArray
    from shapely.geometry import Polygon

    from rojak.orchestrator.configuration import SpatialDomain

logger = logging.getLogger(__name__)


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
    required_coords: ClassVar[frozenset[str]] = frozenset(
        ["pressure_level", "latitude", "longitude", "time", "altitude"]
    )

    def __init__(self, dataset: xr.Dataset) -> None:
        if not set(dataset.data_vars.keys()).issuperset(self.required_variables):
            missing_variables = self.required_variables - dataset.data_vars.keys()
            raise ValueError(
                f"Attempting to instantiate CATPrognosticData with missing data variables: {missing_variables}"
            )
        if not set(dataset.coords.keys()).issuperset(self.required_coords):
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

    def altitude(self) -> xr.DataArray:
        return self._dataset["altitude"]

    def time_window(self) -> Limits[np.datetime64]:
        return Limits(self._dataset["time"].min().to_numpy().item(), self._dataset["time"].max().to_numpy().item())


class CATData(CATPrognosticData):
    _potential_temperature: xr.DataArray | None = None
    _velocity_derivatives: dict[VelocityDerivative, xr.DataArray] | None = None
    _shear_deformation: xr.DataArray | None = None
    _stretching_deformation: xr.DataArray | None = None

    def __init__(self, dataset: xr.Dataset) -> None:
        super().__init__(dataset)

    def potential_temperature(self) -> xr.DataArray:
        if self._potential_temperature is None:
            self._potential_temperature = turb_calc.potential_temperature(
                self.temperature(), self.temperature()["pressure_level"]
            )
        return self._potential_temperature

    def velocity_derivatives(self) -> dict[VelocityDerivative, xr.DataArray]:
        if self._velocity_derivatives is None:
            self._velocity_derivatives = derivatives.vector_derivatives(self.u_wind(), self.v_wind(), "deg")
        return self._velocity_derivatives

    def specific_velocity_derivative(self, target_derivative: VelocityDerivative) -> xr.DataArray:
        if self._velocity_derivatives is None:
            return self.velocity_derivatives()[target_derivative]
        return self._velocity_derivatives[target_derivative]

    def shear_deformation(self) -> xr.DataArray:
        if self._shear_deformation is None:
            self._shear_deformation = turb_calc.shearing_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DV_DX),
                self.specific_velocity_derivative(VelocityDerivative.DU_DY),
            )
        return self._shear_deformation

    def stretching_deformation(self) -> xr.DataArray:
        if self._stretching_deformation is None:
            self._stretching_deformation = turb_calc.stretching_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DU_DX),
                self.specific_velocity_derivative(VelocityDerivative.DV_DY),
            )
        return self._stretching_deformation

    def total_deformation(self) -> xr.DataArray:
        return turb_calc.magnitude_of_vector(self.shear_deformation(), self.stretching_deformation(), is_squared=True)

    def jacobian_horizontal_velocity(self) -> xr.DataArray:
        vec_derivs = self.velocity_derivatives()
        return (
            vec_derivs[VelocityDerivative.DU_DX] * vec_derivs[VelocityDerivative.DV_DY]
            - vec_derivs[VelocityDerivative.DU_DY] * vec_derivs[VelocityDerivative.DV_DX]
        )


def load_from_folder(
    path_to_folder: "Path",
    glob_pattern: str = "*.nc",
    chunks: Mapping | None = None,
    engine: Literal["netcdf4", "scipy", "pydap", "h5netcdf", "zarr"] = "h5netcdf",
    is_decoded: bool = True,
) -> "xr.Dataset":
    if chunks is None:
        raise ValueError("Chunks for ERA5 multi-file load cannot be None")
    logger.debug("Loading CATData from folder")
    return xr.open_mfdataset(
        str(path_to_folder / glob_pattern),
        chunks=chunks,
        parallel=True,
        engine=engine,
        decode_coords=is_decoded,
        decode_cf=is_decoded,
        decode_timedelta=True,
    )


class MetData(ABC):
    _longitude_coord_name: str
    _latitude_coord_name: str

    def __init__(self, longitude_name: str = "longitude", latitude_name: str = "latitude") -> None:
        self._longitude_coord_name = longitude_name
        self._latitude_coord_name = latitude_name

    def select_domain(
        self, domain: "SpatialDomain", data: xr.Dataset, level_coordinate_name: str = "level"
    ) -> xr.Dataset:
        assert {self._longitude_coord_name, self._latitude_coord_name, "time", level_coordinate_name}.issubset(
            data.dims
        ), "Dataset must contain longitude, latitude, time and level dimensions"

        longitude_coord = data[self._longitude_coord_name]
        max_lon = longitude_coord.max()
        min_lon = longitude_coord.min()
        if max_lon > MAX_LONGITUDE or min_lon < -MAX_LONGITUDE:
            data = self.shift_longitude(data)

        level_coordinate = data[level_coordinate_name]
        level_slice: slice = (
            make_value_based_slice(level_coordinate.data, domain.minimum_level, domain.maximum_level)
            if domain.minimum_level is not None or domain.maximum_level is not None
            else slice(None)
        )

        return data.sel(
            {
                level_coordinate_name: level_slice,
                "time": slice(None),
                self._longitude_coord_name: make_value_based_slice(
                    longitude_coord.data, domain.minimum_longitude, domain.maximum_longitude
                ),
                self._latitude_coord_name: make_value_based_slice(
                    data[self._latitude_coord_name].data, domain.minimum_latitude, domain.maximum_latitude
                ),
            }
        )

    @abstractmethod
    def to_clear_air_turbulence_data(self, domain: "SpatialDomain") -> CATData: ...

    # Modified from pycontrails
    # https://github.com/contrailcirrus/pycontrails/blob/8a25266bcf5ead003a6b344395462ab56943e668/pycontrails/core/met.py#L2430
    def shift_longitude(self, data: xr.Dataset, domain_bound: float = -180, sort_data: bool = True) -> xr.Dataset:
        # Utility function to shift data to have longitude in the range of [domain_bound, 360 + domain_bound]
        # This also sorts it so that the data is then ascending from domain_bound
        shifted_data: xr.Dataset = data.assign_coords(
            longitude=((data[self._longitude_coord_name] - domain_bound) % 360) + domain_bound
        )
        return shifted_data.sortby(self._longitude_coord_name, ascending=True) if sort_data else shifted_data

    # To be added later
    # @abstractmethod
    # def to_contrails_data(self) -> xr.Dataset: ...


def as_geo_dataframe(data_frame: "dd.DataFrame") -> dgpd.GeoDataFrame:
    """
    Method to convert a data frame into a GeoDataFrame.

    Args:
        data_frame (dd.DataFrame): The data frame to convert.

    Returns:
        dgpd.GeoDataFrame: The converted data frame.
    """
    gddf = data_frame.set_geometry(dgpd.points_from_xy(data_frame, x="longitude", y="latitude"))
    return gddf.set_crs("epsg:4326")


class AmdarDataRepository(ABC):
    """
    Abstract AMDAR data repository interface.
    """

    _path_to_files: str | list

    def __init__(self, path_to_files: str | list) -> None:
        self._path_to_files = path_to_files

    @abstractmethod
    def load(self) -> "dd.DataFrame":
        """
        Method to load data from path into a dask dataframe.

        Returns:
            dd.DataFrame: The loaded data frame.
        """
        ...

    @staticmethod
    def _find_closest_pressure_level(
        current_altitude: float, altitudes: "NDArray", pressures: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> float:
        """
        Method to find the closest pressure level for a given altitude.

        Args:
            current_altitude (float): Altitude for a given data point in meters
            pressures (np.ndarray[Any, np.dtype[np.float]]): Pressure levels in hPa
            altitudes: Pressure levels converted to altitude using standard atmosphere in meters

        Returns:
            float: Closest pressure level
        """
        index = np.abs(altitudes - current_altitude).argmin()
        return pressures[index]

    def _compute_closest_pressure_level(
        self,
        data_frame: "dd.DataFrame",
        pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]",
        altitude_column: str,
    ) -> "dd.Series":
        """
        Method to compute closest pressure level for the entire data frame

        Args:
            data_frame (dd.DataFrame): Dataframe to compute the closest pressure level for
            pressure_levels (np.ndarray[Any, np.dtype[np.float]]): Pressure levels in hPa
            altitude_column (str): Name of the altitude column

        Returns:
            dd.Series: New series with the closest pressure level
        """
        altitudes = pressure_to_altitude_std_atm(pressure_levels)
        return data_frame[altitude_column].apply(
            self._find_closest_pressure_level, args=(altitudes, pressure_levels), meta=("level", float)
        )

    @abstractmethod
    def _call_compute_closest_pressure_level(
        self, data_frame: "dd.DataFrame", pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "dd.Series":
        """
        Wrapper method to be implemented by child classes to call _compute_closest_pressure_level with the appropriate
        altitude column name

        Args:
            data_frame (dd.DataFrame): Dataframe to compute the closest pressure level for
            pressure_levels (np.ndarray[Any, np.dtype[np.float]]): Pressure levels in hPa

        Returns:
            dd.Series: New series with the closest pressure level
        """
        ...

    @abstractmethod
    def _instantiate_amdar_turbulence_data_class(
        self, data_frame: "dd.DataFrame", grid: "dgpd.GeoDataFrame"
    ) -> "AmdarTurbulenceData":
        """
        Method to instantiate a concrete instance of the AmdarTurbulenceData class

        Args:
            data_frame (dd.DataFrame): Dataframe containing AMDAR data to instantiate a concrete instance with
            grid (dgpd.GeoDataFrame): Grid used to spatially bucket the data in data_frame

        Returns:
            AmdarTurbulenceData: Instantiated concrete implementation of abstract AmdarTurbulenceData class
        """
        ...

    @abstractmethod
    def _time_column_rename_mapping(self) -> dict[str, str]: ...

    def to_amdar_turbulence_data(
        self, target_region: "SpatialDomain | Polygon", grid_size: float, target_pressure_levels: Sequence[float]
    ) -> "AmdarTurbulenceData":
        """
        Public method which coordinates the loading of data from disk and processing it such that it has been spatially
        bucket in the horizontal domain and has the closest pressure level (vertical domain) stored.

        Args:
            target_region (SpatialDomain | Polygon):    Region of data to keep. This should be selected to match the
                                                        met data it will be compared against
            grid_size (float):  Step size of grid. This controls the discretisation of the target_region and should be
                                selected to match the met data it will be compared against
            target_pressure_levels: Pressure levels (vertical coordinate) that the data will be bucketed into. This
                                    must match the met data it will be compared against

        Returns:
            AmdarTurbulenceData: Instance containing the data loaded from file with the spatial operations applied.

        """
        raw_data_frame: "dd.DataFrame" = self.load()

        raw_data_frame["level"] = self._call_compute_closest_pressure_level(
            raw_data_frame, np.asarray(target_pressure_levels, dtype=np.float64)
        )

        grid: "dgpd.GeoDataFrame" = create_grid_data_frame(target_region, grid_size)
        grid_dataframe: "dd.DataFrame" = grid.persist()
        within_region: "dgpd.GeoDataFrame" = as_geo_dataframe(raw_data_frame).sjoin(grid).optimize()
        within_region["grid_box"] = within_region["index_right"].apply(
            lambda row: grid_dataframe.loc[row, "geometry"], meta=("grid_box", object)
        )
        # within_region = within_region.drop(columns=["index_right"])
        if self._time_column_rename_mapping():
            within_region = within_region.rename(columns=self._time_column_rename_mapping())

        return self._instantiate_amdar_turbulence_data_class(within_region.optimize(), grid)


class AmdarTurbulenceData(ABC):
    _data_frame: "dd.DataFrame"
    _grid: "dgpd.GeoDataFrame"

    MINIMUM_ALTITUDE: ClassVar[float] = 8500  # Approx. 28,000 ft

    def __init__(self, data_frame: "dd.DataFrame", grid: "dgpd.GeoDataFrame") -> None:
        required_columns = {"grid_box", "datetime", "index_right", "level", "geometry"}
        assert required_columns.issubset(data_frame.columns), (
            f"Columns {required_columns - set(data_frame.columns)} from missing the data frame"
        )
        self._data_frame = self.__apply_quality_control(data_frame)
        self._grid = grid

    @abstractmethod
    def _minimum_altitude_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        raise NotImplementedError("Method must be implemented by child class")

    @abstractmethod
    def _drop_manoeuvre_data_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        raise NotImplementedError("Method must be implemented by child class")

    @abstractmethod
    def turbulence_column_names(self) -> list[str]:
        raise NotImplementedError("Method must be implemented by child class")

    def __apply_quality_control(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        data_frame = data_frame.drop_duplicates()
        data_frame = self._minimum_altitude_qc(data_frame)
        return self._drop_manoeuvre_data_qc(data_frame).optimize()

    @property
    def data_frame(self) -> "dd.DataFrame":
        return self._data_frame

    @property
    def grid(self) -> "dgpd.GeoDataFrame":
        return self._grid

    def clip_to_time_window(self, window: Limits[np.datetime64]) -> "dd.DataFrame":
        return self.data_frame.loc[
            (self.data_frame["datetime"] >= window.lower) & (self.data_frame["datetime"] <= window.upper)
        ]
