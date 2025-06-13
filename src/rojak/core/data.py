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
from rojak.core.constants import MAX_LONGITUDE
from rojak.core.derivatives import VelocityDerivative
from rojak.core.geometric import create_grid_data_frame
from rojak.core.indexing import make_value_based_slice
from rojak.turbulence import calculations as turb_calc

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd
    import geopandas as gpd
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

    # TODO: TEST
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

    # TODO: TEST
    def temperature(self) -> xr.DataArray:
        return self._dataset["temperature"]

    # TODO: TEST
    def divergence(self) -> xr.DataArray:
        return self._dataset["divergence_of_wind"]

    # TODO: TEST
    def geopotential(self) -> xr.DataArray:
        return self._dataset["geopotential"]

    # TODO: TEST
    def specific_humidity(self) -> xr.DataArray:
        return self._dataset["specific_humidity"]

    # TODO: TEST
    def u_wind(self) -> xr.DataArray:
        return self._dataset["eastward_wind"]

    # TODO: TEST
    def v_wind(self) -> xr.DataArray:
        return self._dataset["northward_wind"]

    # TODO: TEST
    def potential_vorticity(self) -> xr.DataArray:
        return self._dataset["potential_vorticity"]

    # TODO: TEST
    def vorticity(self) -> xr.DataArray:
        return self._dataset["vorticity"]

    # TODO: TEST
    def altitude(self) -> xr.DataArray:
        return self._dataset["altitude"]


class CATData(CATPrognosticData):
    _potential_temperature: xr.DataArray | None = None
    _velocity_derivatives: dict[VelocityDerivative, xr.DataArray] | None = None
    _shear_deformation: xr.DataArray | None = None
    _stretching_deformation: xr.DataArray | None = None

    def __init__(self, dataset: xr.Dataset) -> None:
        super().__init__(dataset)

    # TODO: TEST
    def potential_temperature(self) -> xr.DataArray:
        if self._potential_temperature is None:
            self._potential_temperature = turb_calc.potential_temperature(
                self.temperature(), self.temperature()["pressure_level"]
            )
        return self._potential_temperature

    # TODO: TEST
    def velocity_derivatives(self) -> dict[VelocityDerivative, xr.DataArray]:
        if self._velocity_derivatives is None:
            self._velocity_derivatives = derivatives.vector_derivatives(self.u_wind(), self.v_wind(), "deg")
        return self._velocity_derivatives

    # TODO: TEST
    def specific_velocity_derivative(self, target_derivative: VelocityDerivative) -> xr.DataArray:
        if self._velocity_derivatives is None:
            return self.velocity_derivatives()[target_derivative]
        return self._velocity_derivatives[target_derivative]

    # TODO: TEST
    def shear_deformation(self) -> xr.DataArray:
        if self._shear_deformation is None:
            self._shear_deformation = turb_calc.shearing_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DV_DX),
                self.specific_velocity_derivative(VelocityDerivative.DU_DY),
            )
        return self._shear_deformation

    # TODO: TEST
    def stretching_deformation(self) -> xr.DataArray:
        if self._stretching_deformation is None:
            self._stretching_deformation = turb_calc.stretching_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DU_DX),
                self.specific_velocity_derivative(VelocityDerivative.DV_DY),
            )
        return self._stretching_deformation

    # TODO: TEST
    def total_deformation(self) -> xr.DataArray:
        return turb_calc.magnitude_of_vector(self.shear_deformation(), self.stretching_deformation(), is_squared=True)

    # TODO: TEST
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


def pressure_to_altitude_std_atm(pressure: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
    """
    Equation 3.106 on page 104 in Wallace, J. M., and Hobbs, P. V., “Atmospheric Science: An Introductory Survey,”
    Elsevier Science & Technology, San Diego, UNITED STATES, 2006.
    ..math:: z = \frac{T_0}{\\Gamma} \\left[ 1 - \\left( \frac{p}{p_0} \right)^{\frac{R\\Gamma}{g}} \right]
    """
    reference_temperature: float = 288.0  # kelvin
    gamma: float = 0.0065  # 6.5 K/km => 0.0065 K/m
    reference_pressure: float = 1013.25  # hPa
    gas_constant_dry_air: float = 287  # J / (K kg)
    gravitational_acceleration: float = 9.80665  # m / s^2
    return (reference_temperature / gamma) * (
        1 - ((pressure / reference_pressure) ** ((gas_constant_dry_air * gamma) / gravitational_acceleration))
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
    gddf = data_frame.set_geometry(dgpd.points_from_xy(data_frame, x="longitude", y="latitude"))
    return gddf.set_crs("epsg:4326")


class AmdarData(ABC):
    _path_to_files: str | list
    _altitude_column_name: str

    def __init__(self, path_to_files: str | list, altitude_column_name: str) -> None:
        self._path_to_files = path_to_files
        self._altitude_column_name = altitude_column_name

    @abstractmethod
    def load(self) -> "dd.DataFrame": ...

    @staticmethod
    def find_closest_pressure_level(
        current_altitude: float, altitudes: "NDArray", pressures: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> float:
        index = np.abs(altitudes - current_altitude).argmin()
        return pressures[index]

    def _compute_closest_pressure_level(
        self,
        data_frame: "dd.DataFrame",
        pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]",
    ) -> "dd.DataFrame":
        altitudes = pressure_to_altitude_std_atm(pressure_levels)
        return data_frame[self._altitude_column_name].apply(
            self.find_closest_pressure_level, args=(altitudes, pressure_levels), meta=("level", float)
        )

    def to_amdar_turbulence_data(
        self, target_region: "SpatialDomain | Polygon", grid_size: float, target_pressure_levels: Sequence[float]
    ) -> "AmdarTurbulenceData":
        raw_data_frame: "dd.DataFrame" = self.load()

        raw_data_frame["level"] = self._compute_closest_pressure_level(
            raw_data_frame, np.asarray(target_pressure_levels, dtype=np.float64)
        )

        grid: "dgpd.GeoDataFrame" = create_grid_data_frame(target_region, grid_size)
        geo_dataframe: "dgpd.GeoDataFrame" = as_geo_dataframe(raw_data_frame)
        grid_dataframe: "gpd.GeoDataFrame" = grid.compute()
        within_region: "dgpd.GeoDataFrame" = geo_dataframe.sjoin(grid)
        geo_dataframe["grid_box"] = within_region["index_right"].apply(
            lambda row: grid_dataframe.loc[row, "geometry"], meta=("grid_box", object)
        )
        geo_dataframe.drop(columns=["index_right"])

        return AmdarTurbulenceData(geo_dataframe)


class AmdarTurbulenceData:
    _data_frame: "dd.DataFrame"

    def __init__(self, data_frame: "dd.DataFrame") -> None:
        self._data_frame = data_frame
