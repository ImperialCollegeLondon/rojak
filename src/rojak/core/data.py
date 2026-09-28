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
Generic infrastructure for retrieving, loading, and accessing meteorological data

This module provides:

- :class:`DataRetriever`/:class:`DataPreprocessor`: abstract interfaces for downloading and preprocessing raw
  meteorological data files.
- :class:`CATPrognosticData`/:class:`CATData`: containers validating and exposing the meteorological fields (e.g., wind,
  temperature, vorticity, ...) required to compute CAT diagnostics, with :class:`CATData`
  additionally deriving further quantities (potential temperature, velocity derivatives, deformation, ISSR) on
  demand. These effectively form the interface that every meteorological data source must be adapted into.
- :class:`MetData`: an abstract base for a source of meteorological data, handling spatial domain selection
  and longitude wraparound (:func:`shift_longitude`). Each concrete subclass is
  responsible for adapting its own underlying data into a :class:`CATData`. This is the extension point of the
  architecture: new data sources can be added by implementing a new ``MetData`` subclass, without having to
  change any of the diagnostic code in :mod:`rojak.turbulence`, which only ever depends on the
  ``CATData``/``CATPrognosticData`` interface.
- :class:`AmdarDataRepository`/:class:`AmdarTurbulenceData`: abstract interfaces for loading AMDAR (Aircraft
  Meteorological Data Relay) turbulence observations, spatially bucketing them onto a grid, and mapping them to
  their closest pressure level.
"""

import calendar
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import dask.array as da
import dask_geopandas as dgpd
import numpy as np
import xarray as xr

from rojak.atmosphere.contrails import issr
from rojak.core import derivatives
from rojak.core.calculations import pressure_to_altitude_icao
from rojak.core.constants import MAX_LONGITUDE
from rojak.core.derivatives import LatLonUnits, VelocityDerivative
from rojak.core.geometric import create_grid_data_frame
from rojak.core.indexing import make_value_based_slice
from rojak.turbulence import calculations as turb_calc
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd
    from shapely.geometry import Polygon

    from rojak.orchestrator.configuration import SpatialDomain

logger = logging.getLogger(__name__)


class Date(NamedTuple):
    """A single calendar date, as used by :class:`DataRetriever`"""

    year: int
    month: int
    day: int


class DataRetriever(ABC):
    """Abstract interface for downloading raw data files for a set of years, months, and days"""

    @abstractmethod
    def download_files(
        self,
        years: list[int],
        months: list[int],
        days: list[int],
        base_output_dir: "Path",
    ) -> None:
        """
        Download the data files for every combination of ``years``, ``months``, and ``days``

        Args:
            years: Years to download data for
            months: Months to download data for. ``[-1]`` means every month.
            days: Days to download data for. ``[-1]`` means every day of the month.
            base_output_dir: Directory to download the files into
        """
        ...

    @abstractmethod
    def _download_file(self, date: Date, base_output_dir: "Path") -> None:
        """
        Download the data file for a single date

        Args:
            date: Date to download the data file for
            base_output_dir: Directory to download the file into
        """
        ...

    @staticmethod
    def compute_date_combinations(years: list[int], months: list[int], days: list[int]) -> list[Date]:
        """
        Expand years/months/days into every :class:`Date` combination

        Args:
            years: Years to combine
            months: Months to combine. ``[-1]`` expands to every month of the year (1-12).
            days: Days to combine. ``[-1]`` expands to every day of each month in ``months`` (accounting for the
                number of days in that month/year).

        Returns:
            List of every :class:`Date` combination of ``years``, ``months``, and ``days``
        """
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
    """Abstract interface for preprocessing raw data files before they are used"""

    @abstractmethod
    def apply_preprocessor(self, output_directory: "Path") -> None:
        """
        Preprocess the raw data files in ``output_directory``

        Args:
            output_directory: Directory containing the raw data files to preprocess
        """
        ...


# NOTE: cf_name is the key that'll be used
@dataclass(frozen=True)
class DataVarSchema:
    """Mapping between a variable's name in the source dataset and its CF standard name"""

    database_name: str  # Name of variable in the dataset
    cf_name: str  # CF standard name for the variable


class CATPrognosticData:
    """
    Validated container for the raw meteorological fields required to compute CAT diagnostics

    This is effectively the interface that every :class:`MetData` source must be able to populate: any concrete
    ``MetData`` subclass (e.g. for a specific reanalysis or forecast product) adapts its own underlying dataset
    into a :class:`CATData` (which extends this class) so that downstream turbulence diagnostics
    (:mod:`rojak.turbulence`) only ever depend on this common interface. This is what makes it possible to add
    support for a new data source without having to change any of the diagnostic code that consumes it.

    On construction, checks that the underlying dataset contains :attr:`required_variables` and
    :attr:`required_coords`, then exposes each required field via an accessor method.
    """

    _dataset: xr.Dataset
    _pressure_level_prefix: float

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
        ],
    )
    required_coords: ClassVar[frozenset[str]] = frozenset(
        ["pressure_level", "latitude", "longitude", "time", "altitude"],
    )

    def __init__(self, dataset: xr.Dataset, pressure_level_prefix: float) -> None:
        """
        Args:
            dataset: Dataset containing at least :attr:`required_variables` and :attr:`required_coords`
            pressure_level_prefix: Multiplier applied to the pressure level coordinate to convert it to Pascals,
                see :meth:`pressure_level`

        Raises:
            ValueError: If ``dataset`` is missing any of :attr:`required_variables` or :attr:`required_coords`
        """
        if not set(dataset.data_vars.keys()).issuperset(self.required_variables):
            missing_variables = self.required_variables - dataset.data_vars.keys()
            raise ValueError(
                f"Attempting to instantiate CATPrognosticData with missing data variables: {missing_variables}",
            )
        if not set(dataset.coords.keys()).issuperset(self.required_coords):
            missing_coords = self.required_coords - dataset.coords.keys()
            raise ValueError(f"Attempting to instantiate CATPrognosticData with missing coords: {missing_coords}")
        self._dataset = dataset.sortby("time").persist()
        self._pressure_level_prefix = pressure_level_prefix

    def temperature(self) -> xr.DataArray:
        """Air temperature in Kelvin"""
        return self._dataset["temperature"]

    def divergence(self) -> xr.DataArray:
        """Horizontal divergence of wind in m/s"""
        return self._dataset["divergence_of_wind"]

    def geopotential(self) -> xr.DataArray:
        """Geopotential in m^2/s^2"""
        return self._dataset["geopotential"]

    def specific_humidity(self) -> xr.DataArray:
        """Specific humidity"""
        return self._dataset["specific_humidity"]

    def u_wind(self) -> xr.DataArray:
        """Zonal (eastward) wind speed in m/s"""
        return self._dataset["eastward_wind"]

    def v_wind(self) -> xr.DataArray:
        """Meridional (northward) wind speed in m/s"""
        return self._dataset["northward_wind"]

    def potential_vorticity(self) -> xr.DataArray:
        """Potential vorticity"""
        return self._dataset["potential_vorticity"]

    def vorticity(self) -> xr.DataArray:
        """Vertical component of vorticity in m/s"""
        return self._dataset["vorticity"]

    def altitude(self) -> xr.DataArray:
        """Altitude"""
        return self._dataset["altitude"]

    def pressure_level(self, convert_to_pascals: bool = False) -> xr.DataArray:
        """
        Pressure level coordinate

        Args:
            convert_to_pascals: If ``True``, scale the coordinate by :attr:`_pressure_level_prefix` to convert it
                to Pascals. Defaults to ``False``.

        Returns:
            Pressure level coordinate, optionally converted to Pascals
        """
        if convert_to_pascals:
            return self._pressure_level_prefix * self._dataset["pressure_level"]
        return self._dataset["pressure_level"]

    def time_window(self) -> Limits[np.datetime64]:
        """Minimum and maximum values of the time coordinate"""
        return Limits(self._dataset["time"].min().to_numpy().item(), self._dataset["time"].max().to_numpy().item())


class CATData(CATPrognosticData):
    """
    :class:`CATPrognosticData` extended with the derived quantities required by CAT diagnostics

    This is the concrete interface that :meth:`MetData.to_clear_air_turbulence_data` must return: every
    :class:`MetData` source is required to be able to produce one of these, which is what lets
    :mod:`rojak.turbulence` diagnostics work against any data source uniformly.

    Each derived quantity is computed lazily on first access and cached (persisted) thereafter.
    """

    _potential_temperature: xr.DataArray | None = None
    _velocity_derivatives: dict[VelocityDerivative, xr.DataArray] | None = None
    _shear_deformation: xr.DataArray | None = None
    _stretching_deformation: xr.DataArray | None = None

    def __init__(self, dataset: xr.Dataset, pressure_level_prefix: float) -> None:
        """See :meth:`CATPrognosticData.__init__`"""
        super().__init__(dataset, pressure_level_prefix)

    def potential_temperature(self) -> xr.DataArray:
        """Potential temperature, see :func:`rojak.turbulence.calculations.potential_temperature`"""
        if self._potential_temperature is None:
            self._potential_temperature = turb_calc.potential_temperature(
                self.temperature(),
                self.temperature()["pressure_level"],
            ).persist()
        return self._potential_temperature

    def velocity_derivatives(self) -> dict[VelocityDerivative, xr.DataArray]:
        """All four horizontal velocity derivatives, see :func:`rojak.core.derivatives.vector_derivatives`"""
        if self._velocity_derivatives is None:
            self._velocity_derivatives = derivatives.vector_derivatives(self.u_wind(), self.v_wind(), LatLonUnits.DEG)
        return self._velocity_derivatives

    def specific_velocity_derivative(self, target_derivative: VelocityDerivative) -> xr.DataArray:
        """
        A single component of :meth:`velocity_derivatives`

        Args:
            target_derivative: Velocity derivative component to retrieve

        Returns:
            The requested velocity derivative component
        """
        if self._velocity_derivatives is None:
            return self.velocity_derivatives()[target_derivative]
        return self._velocity_derivatives[target_derivative]

    def shear_deformation(self) -> xr.DataArray:
        """Shear deformation, see :func:`rojak.turbulence.calculations.shearing_deformation`"""
        if self._shear_deformation is None:
            self._shear_deformation = turb_calc.shearing_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DV_DX),
                self.specific_velocity_derivative(VelocityDerivative.DU_DY),
            ).persist()
        return self._shear_deformation

    def stretching_deformation(self) -> xr.DataArray:
        """Stretch deformation, see :func:`rojak.turbulence.calculations.stretching_deformation`"""
        if self._stretching_deformation is None:
            self._stretching_deformation = turb_calc.stretching_deformation(
                self.specific_velocity_derivative(VelocityDerivative.DU_DX),
                self.specific_velocity_derivative(VelocityDerivative.DV_DY),
            ).persist()
        return self._stretching_deformation

    def total_deformation(self) -> xr.DataArray:
        """Total deformation, see :func:`rojak.turbulence.calculations.total_deformation`"""
        return turb_calc.magnitude_of_vector(self.shear_deformation(), self.stretching_deformation(), is_squared=False)

    def jacobian_horizontal_velocity(self) -> xr.DataArray:
        """Determinant of the Jacobian of the horizontal velocity field, i.e. du/dx * dv/dy - du/dy * dv/dx"""
        vec_derivs = self.velocity_derivatives()
        return (
            vec_derivs[VelocityDerivative.DU_DX] * vec_derivs[VelocityDerivative.DV_DY]
            - vec_derivs[VelocityDerivative.DU_DY] * vec_derivs[VelocityDerivative.DV_DX]
        )

    def ice_supersaturated_regions(self, rhi_threshold: float = 0.9) -> xr.DataArray:
        """
        Boolean mask of ice-supersaturated regions (ISSR), see :func:`rojak.atmosphere.contrails.issr`

        Args:
            rhi_threshold: Relative humidity (with respect to ice) threshold above which a region is considered
                ice-supersaturated. Defaults to ``0.9``.

        Returns:
            Boolean mask of where the relative humidity (with respect to ice) exceeds ``rhi_threshold``
        """
        return issr(
            air_temperature=self.temperature(),
            specific_humidity=self.specific_humidity(),
            air_pressure=self.pressure_level(convert_to_pascals=True),
            rhi_threshold=rhi_threshold,
        )

    def issr_along_path(
        self, lon_points: np.ndarray, lat_points: np.ndarray, points_dim_name: str = "waypoints"
    ) -> xr.DataArray:
        """
        Boolean mask of ice-supersaturated regions (ISSR), interpolated onto a path of longitude/latitude points

        Temperature and specific humidity are interpolated onto ``lon_points``/``lat_points`` (e.g. as computed by
        :func:`rojak.core.geometric.geodesic_waypoints_between`) before checking for ice supersaturation, see
        :func:`rojak.atmosphere.contrails.issr`.

        Args:
            lon_points: 1D array of longitude points to interpolate onto
            lat_points: 1D array of latitude points to interpolate onto, of the same shape as ``lon_points``
            points_dim_name: Name of the new dimension along the interpolated points. Defaults to ``"waypoints"``.

        Returns:
            Boolean mask of ice-supersaturated regions along the path

        Raises:
            ValueError: If ``lon_points``/``lat_points`` are not 1D, or do not have the same shape
        """
        if lon_points.ndim != 1 or lat_points.ndim != 1:
            raise ValueError("lon_points and lat_points must have one dimension")
        if lon_points.shape != lat_points.shape:
            raise ValueError("lon_points and lat_points must have the same shape")

        temp_ds: xr.Dataset = xr.Dataset(
            data_vars={"temperature": self.temperature(), "specific_humidity": self.specific_humidity()}
        )
        interp_ds = temp_ds.interp(
            longitude=xr.DataArray(lon_points, dims=points_dim_name),
            latitude=xr.DataArray(lat_points, dims=points_dim_name),
        )
        return issr(
            air_temperature=interp_ds["temperature"],
            specific_humidity=interp_ds["specific_humidity"],
            air_pressure=self.pressure_level(convert_to_pascals=True),
        )


def load_from_folder(
    path_to_folder: "Path",
    glob_pattern: str = "*.nc",
    chunks: Mapping | None = None,
    engine: Literal["netcdf4", "scipy", "pydap", "h5netcdf", "zarr"] = "netcdf4",
    is_decoded: bool = True,
) -> "xr.Dataset":
    """
    Lazily load and concatenate every file matching ``glob_pattern`` in a folder into a single Dataset

    Args:
        path_to_folder: Directory containing the data files
        glob_pattern: Glob pattern (relative to ``path_to_folder``) matching the files to load. Defaults to
            ``"*.nc"``.
        chunks: Dask chunking to use, passed to :func:`xarray.open_mfdataset`. Must not be ``None``.
        engine: Backend engine used to read the files. Defaults to ``"netcdf4"``.
        is_decoded: Whether to decode coordinates, CF conventions, and timedeltas on load. Defaults to ``True``.

    Returns:
        Dataset formed by concatenating every matching file

    Raises:
        ValueError: If ``chunks`` is ``None``
    """
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
        compat="override",
    )


# Modified from pycontrails
# https://github.com/contrailcirrus/pycontrails/blob/8a25266bcf5ead003a6b344395462ab56943e668/pycontrails/core/met.py#L2430
def shift_longitude[T: (xr.Dataset, xr.DataArray)](
    data: T, *, domain_bound: float = -180, sort_data: bool = True, longitude_coord_name: str = "longitude"
) -> T:
    """
    Shift the longitude coordinate to lie within ``[domain_bound, 360 + domain_bound)``

    Args:
        data: Array/Dataset whose longitude coordinate is to be shifted
        domain_bound: Lower bound of the target longitude range. Defaults to ``-180``.
        sort_data: If ``True`` (default), sort ``data`` by the shifted longitude coordinate in ascending order
        longitude_coord_name: Name of the longitude coordinate. Defaults to ``"longitude"``.

    Returns:
        ``data`` with its longitude coordinate shifted (and, if ``sort_data``, sorted)
    """
    # Utility function to shift data to have longitude in the range of [domain_bound, 360 + domain_bound]
    # This also sorts it so that the data is then ascending from domain_bound
    shifted_data: T = data.assign_coords(
        coords={
            longitude_coord_name: ((data[longitude_coord_name] - domain_bound) % 360) + domain_bound,
        }
    )
    return shifted_data.sortby(longitude_coord_name, ascending=True) if sort_data else shifted_data


class MetData(ABC):
    """
    Abstract base for a generic source of meteorological data

    Subclasses are responsible for loading their own underlying data; this base class provides shared spatial
    domain selection (:meth:`select_domain`) and longitude wraparound handling (:meth:`shift_ds_longitude`). Every
    subclass must implement :meth:`to_clear_air_turbulence_data`, adapting its own underlying data into the common
    :class:`CATData`/:class:`CATPrognosticData` interface. This is the extension point of the architecture: adding
    support for a new meteorological data source only requires a new ``MetData`` subclass that can produce a
    ``CATData``, and every consumer downstream in :mod:`rojak.turbulence` keeps working unchanged.
    """

    _longitude_coord_name: str
    _latitude_coord_name: str

    def __init__(self, longitude_name: str = "longitude", latitude_name: str = "latitude") -> None:
        """
        Args:
            longitude_name: Name of the longitude coordinate in the underlying data. Defaults to ``"longitude"``.
            latitude_name: Name of the latitude coordinate in the underlying data. Defaults to ``"latitude"``.
        """
        self._longitude_coord_name = longitude_name
        self._latitude_coord_name = latitude_name

    def select_domain(
        self,
        domain: "SpatialDomain",
        data: xr.Dataset,
        level_coordinate_name: str = "level",
    ) -> xr.Dataset:
        """
        Subset a Dataset to the given spatial (and, if specified, vertical) domain

        Longitude is shifted (see :func:`shift_longitude`) first if it lies outside +/- :data:`MAX_LONGITUDE`, so
        that the domain selection works regardless of whether the data's longitude convention is ``[0, 360)`` or
        ``[-180, 180)``.

        Args:
            domain: Spatial (and optionally vertical) domain to select
            data: Dataset to select the domain from. Must have the configured longitude/latitude coordinates plus
                ``"time"`` and ``level_coordinate_name`` dimensions.
            level_coordinate_name: Name of the vertical level coordinate. Defaults to ``"level"``.

        Returns:
            ``data`` subset to ``domain``

        Raises:
            AssertionError: If ``data`` is missing the longitude, latitude, time, or level dimension
        """
        assert {self._longitude_coord_name, self._latitude_coord_name, "time", level_coordinate_name}.issubset(
            data.dims,
        ), "Dataset must contain longitude, latitude, time and level dimensions"

        longitude_coord = data[self._longitude_coord_name]
        max_lon = longitude_coord.max()
        min_lon = longitude_coord.min()
        if max_lon > MAX_LONGITUDE or min_lon < -MAX_LONGITUDE:
            data = shift_longitude(data, longitude_coord_name=self._longitude_coord_name)

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
                    longitude_coord.data,
                    domain.minimum_longitude,
                    domain.maximum_longitude,
                ),
                self._latitude_coord_name: make_value_based_slice(
                    data[self._latitude_coord_name].data,
                    domain.minimum_latitude,
                    domain.maximum_latitude,
                ),
            },
        )

    @abstractmethod
    def to_clear_air_turbulence_data(self, domain: "SpatialDomain") -> CATData:
        """
        Build the :class:`CATData` required by :mod:`rojak.turbulence` from this data source, over ``domain``

        Args:
            domain: Spatial (and optionally vertical) domain to build the data for

        Returns:
            :class:`CATData` containing the fields required to compute CAT diagnostics over ``domain``
        """
        ...

    def shift_ds_longitude(self, data: xr.Dataset, domain_bound: float = -180, sort_data: bool = True) -> xr.Dataset:
        """See :func:`shift_longitude`, using this instance's configured longitude coordinate name"""
        return shift_longitude(
            data, domain_bound=domain_bound, sort_data=sort_data, longitude_coord_name=self._longitude_coord_name
        )

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

    This class handles the I/O of the data
    """

    _path_to_files: str | list
    _use_min_turbulence_vars: bool

    def __init__(self, path_to_files: str | list, is_minimal_turb_vars: bool) -> None:
        """
        Args:
            path_to_files: Path(s) to the AMDAR data file(s) to load
            is_minimal_turb_vars: Whether to load only the minimal set of turbulence-related variables
        """
        self._path_to_files = path_to_files
        self._use_min_turbulence_vars = is_minimal_turb_vars

    @abstractmethod
    def load(self) -> "dd.DataFrame":
        """
        Method to load data from path into a dask dataframe.

        Returns:
            dd.DataFrame: The loaded data frame.
        """
        ...

    @staticmethod
    def _compute_closest_pressure_level(
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
        altitudes = pressure_to_altitude_icao(pressure_levels)
        # Optimize is necessary to handle an edge case where nchunks != npartitions when converting to a dask array
        obs_altitudes = data_frame[altitude_column].optimize()
        # Computing the chunks through lengths=True will introduce divisions into the resulting dd.dataframe
        closest_index: da.Array = (
            np.abs(obs_altitudes.to_dask_array(lengths=True)[:, None] - altitudes).argmin(axis=1).persist()
        )
        closest_pressure: da.Array = da.from_array(pressure_levels)[closest_index]

        # pyright as it is throwing up a false positive that da.Array.to_dask_dataframe() doesn't exist
        into_dask_series: dd.Series = closest_pressure.to_dask_dataframe(columns="level")  # pyright: ignore[reportAttributeAccessIssue]
        if data_frame.divisions[0] is None:
            # Divisions need to be cleared as the dataframe it is added to doesn't have divisions
            # So, when this series is added to the dataframe and the dataframe is optimised, it panics as the dataframe
            # does not have any divisions while this series does.
            into_dask_series = into_dask_series.clear_divisions()

        return into_dask_series.persist()

    @abstractmethod
    def _call_compute_closest_pressure_level(
        self,
        data_frame: "dd.DataFrame",
        pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]",
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
        self,
        data_frame: "dd.DataFrame",
        grid: "dgpd.GeoDataFrame",
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
    def _time_column_rename_mapping(self) -> dict[str, str]:
        """Mapping from this repository's raw time-column name(s) to the standardised name(s) expected downstream"""
        ...

    def to_amdar_turbulence_data(
        self,
        target_region: "SpatialDomain | Polygon",
        grid_size: float,
        target_pressure_levels: Sequence[float],
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
        raw_data_frame: dd.DataFrame = self.load()
        raw_data_frame = raw_data_frame.assign(
            level=self._call_compute_closest_pressure_level(
                raw_data_frame,
                np.asarray(target_pressure_levels, dtype=np.float64),
            ),
        ).persist()

        grid: dgpd.GeoDataFrame = create_grid_data_frame(target_region, grid_size)
        within_region: dgpd.GeoDataFrame = as_geo_dataframe(raw_data_frame).sjoin(grid).optimize()
        if self._time_column_rename_mapping():
            within_region = within_region.rename(columns=self._time_column_rename_mapping())

        return self._instantiate_amdar_turbulence_data_class(within_region.persist(), grid)


class AmdarTurbulenceData(ABC):
    """
    Abstract container for AMDAR turbulence observations, spatially bucketed onto a grid

    Quality control (deduplication, minimum altitude, and manoeuvre filtering) is applied to the data on
    construction, see :meth:`__apply_quality_control`.
    """

    _data_frame: "dd.DataFrame"
    _grid: "dgpd.GeoDataFrame"

    MINIMUM_ALTITUDE: ClassVar[float] = 8500  # Approx. 28,000 ft

    def __init__(self, data_frame: "dd.DataFrame", grid: "dgpd.GeoDataFrame") -> None:
        """
        Args:
            data_frame: AMDAR observations, spatially joined onto ``grid`` (see
                :meth:`AmdarDataRepository.to_amdar_turbulence_data`). Must contain ``"datetime"``,
                ``"index_right"``, ``"level"``, and ``"geometry"`` columns.
            grid: Grid ``data_frame`` was spatially bucketed onto

        Raises:
            AssertionError: If ``data_frame`` is missing any required column
        """
        required_columns = {"datetime", "index_right", "level", "geometry"}
        assert required_columns.issubset(data_frame.columns), (
            f"Columns {required_columns - set(data_frame.columns)} from missing the data frame"
        )
        self._data_frame = self.__apply_quality_control(data_frame).persist()
        self._grid = grid

    @abstractmethod
    def _minimum_altitude_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        """
        Drop observations below :attr:`MINIMUM_ALTITUDE`

        Args:
            data_frame: Observations to filter

        Returns:
            ``data_frame`` with observations below :attr:`MINIMUM_ALTITUDE` removed
        """
        raise NotImplementedError("Method must be implemented by child class")

    @abstractmethod
    def _drop_manoeuvre_data_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        """
        Drop observations recorded during aircraft manoeuvres (e.g. climbing, turning), which are not reliable
        turbulence observations

        Args:
            data_frame: Observations to filter

        Returns:
            ``data_frame`` with manoeuvre observations removed
        """
        raise NotImplementedError("Method must be implemented by child class")

    @staticmethod
    @abstractmethod
    def turbulence_column_names() -> list[str]:
        """Names of the columns holding turbulence-related quantities"""
        raise NotImplementedError("Method must be implemented by child class")

    def __apply_quality_control(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        """Drop duplicate observations, then apply :meth:`_minimum_altitude_qc` and :meth:`_drop_manoeuvre_data_qc`"""
        data_frame = data_frame.drop_duplicates()
        data_frame = self._minimum_altitude_qc(data_frame)
        return self._drop_manoeuvre_data_qc(data_frame).optimize()

    @property
    def data_frame(self) -> "dd.DataFrame":
        """Quality-controlled AMDAR observations"""
        return self._data_frame

    @property
    def grid(self) -> "dgpd.GeoDataFrame":
        """Grid :attr:`data_frame` is spatially bucketed onto"""
        return self._grid

    def turbulence_frequency_statistics(self, column_name: str, greater_than: float) -> dict:
        """
        Compute the frequency of observed turbulence exceeding a threshold

        Args:
            column_name: Name of the column to compute statistics for
            greater_than: Threshold value that ``column_name`` must exceed to be counted as a turbulent observation

        Returns:
            Dict with ``num_turb_obs`` (count of observations exceeding ``greater_than``), ``num_observations``
            (total count of non-null observations), and ``frequency`` (their ratio)
        """
        num_observations: int = self.data_frame[column_name].count().compute()
        num_turb_obs: int = (self.data_frame[column_name] > greater_than).sum().compute()
        return {
            "num_turb_obs": num_turb_obs,
            "num_observations": num_observations,
            "frequency": num_turb_obs / num_observations,
        }

    def clip_to_time_window(self, window: Limits[np.datetime64]) -> "dd.DataFrame":
        """
        Restrict observations to a time window

        Args:
            window: Time window to restrict observations to (inclusive of both bounds)

        Returns:
            :attr:`data_frame` restricted to observations within ``window``
        """
        return self.data_frame.loc[
            (self.data_frame["datetime"] >= window.lower) & (self.data_frame["datetime"] <= window.upper)
        ]
