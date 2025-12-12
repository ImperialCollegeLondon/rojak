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
import dataclasses
import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Generator
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, assert_never, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from dask.base import is_dask_collection
from scipy import integrate

from rojak.core.indexing import (
    map_index_to_coordinate_value,
    map_order,
    map_values_to_nearest_coordinate_index,
    map_values_to_nearest_index_irregular_grid,
)
from rojak.orchestrator.configuration import (
    AggregationMetricOption,
    SpatialGroupByStrategy,
)
from rojak.turbulence.metrics import (
    BinaryClassificationResult,
    area_under_curve,
    binary_classification_rate_from_cumsum,
    received_operating_characteristic,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rojak.core.data import AmdarTurbulenceData
    from rojak.orchestrator.configuration import DiagnosticValidationCondition
    from rojak.turbulence.diagnostic import DiagnosticSuite
    from rojak.turbulence.metrics import BinaryClassificationRateFromLabels
    from rojak.utilities.types import DiagnosticName, Limits


logger = logging.getLogger(__name__)


class NotWithinTimeFrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_dataframe_dtypes(data_frame: "dd.DataFrame") -> dict[str, pd.Series]:
    return {col_name: pd.Series(dtype=col_dtype) for col_name, col_dtype in data_frame.dtypes.to_dict().items()}


def to_coo_array(
    coordinates: da.Array,
    data_at_coords: da.Array,
    output_dtype: np.dtype,
    resulting_shape: tuple[int, ...],
    resulting_chunks: tuple[tuple[int, ...], ...] | int | str,
    fill_value: float | int | None,
    axis_to_drop: list[int] | None = None,
) -> da.Array:
    instantiate_coo = functools.partial(sparse.COO, shape=resulting_shape, fill_value=fill_value)

    def to_coo(coords_: da.Array, data_: da.Array) -> sparse.COO:
        return instantiate_coo(coords_, data_)

    if axis_to_drop is None:
        axis_to_drop = list(range(coordinates.ndim))

    as_coo: sparse.COO = da.map_blocks(
        to_coo, coordinates, data_at_coords, dtype=output_dtype, drop_axis=axis_to_drop
    ).compute()

    return da.from_array(as_coo, chunks=resulting_chunks)  # pyright: ignore[reportArgumentType]


class ObservationCoordinates(NamedTuple):
    vertical_index: da.Array
    latitude_index: dd.Series
    longitude_index: dd.Series
    time_index: dd.Series

    def as_arrays(
        self,
        lat_index_key: str = "lat_index",
        lon_index_key: str = "lon_index",
        vertical_index_key: str = "level_index",
        time_index_key: str = "time_index",
    ) -> dict[str, da.Array]:
        return {
            lon_index_key: self.longitude_index.to_dask_array(lengths=True),
            lat_index_key: self.latitude_index.to_dask_array(lengths=True),
            time_index_key: self.time_index.to_dask_array(lengths=True),
            vertical_index_key: self.vertical_index,
        }


class IndexingFormat(Enum):
    COORDINATES = auto()
    FLAT = auto()


class AmdarDataHarmoniser:
    _observational_data: dd.DataFrame
    _grid_prototype: xr.DataArray

    _time_coord: str
    _level_coord: str
    _latitude_coord: str
    _longitude_coord: str
    _coordinate_axes: list[str]
    _time_window_delta: np.timedelta64

    latitude_column: ClassVar[str] = "latitude"
    longitude_column: ClassVar[str] = "longitude"
    time_column: ClassVar[str] = "datetime"
    level_column: ClassVar[str] = "level"

    lat_index_column: ClassVar[str] = "lat_index"
    lon_index_column: ClassVar[str] = "lon_index"
    time_index_column: ClassVar[str] = "time_index"
    level_index_column: ClassVar[str] = "level_index"

    def __init__(  # noqa: PLR0913
        self,
        amdar_data: "AmdarTurbulenceData",
        grid_prototype: xr.DataArray,
        time_window: "Limits[np.datetime64]",
        time_coord: str = "time",
        level_coord: str = "pressure_level",
        latitude_coord: str = "latitude",
        longitude_coord: str = "longitude",
        time_window_delta: np.timedelta64 | None = None,
    ) -> None:
        if time_window.lower < grid_prototype[time_coord].min() or time_window.upper > grid_prototype[time_coord].max():
            raise NotWithinTimeFrameError("Time window is not within time coordinate of met data")

        required_coords: set[str] = {longitude_coord, latitude_coord, time_coord, level_coord}
        assert required_coords.issubset(grid_prototype.coords)
        assert required_coords.issubset(grid_prototype.dims)
        assert grid_prototype.ndim == len(required_coords)

        self._observational_data = amdar_data.clip_to_time_window(time_window).persist()
        self._grid_prototype = grid_prototype

        self._time_coord = time_coord
        self._level_coord = level_coord
        self._latitude_coord = latitude_coord
        self._longitude_coord = longitude_coord
        self._coordinate_axes = list(required_coords)
        if time_window_delta is None:
            self._time_window_delta = np.timedelta64(3, "h")
        else:
            self._time_window_delta = time_window_delta

    @property
    def observational_data(self) -> dd.DataFrame:
        return self._observational_data

    @property
    def grid_prototype(self) -> xr.DataArray:
        return self._grid_prototype

    def harmonised_coordinates(self) -> list[str]:
        return self._coordinate_axes

    def coordinates_of_observations(self) -> ObservationCoordinates:
        level_index: da.Array = cast(
            "da.Array",
            map_values_to_nearest_index_irregular_grid(
                self.observational_data[self.level_column], self._grid_prototype[self._level_coord].values
            ),
        )
        latitude_index: dd.Series = self.observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df[self.latitude_column],
                self._grid_prototype[self._latitude_coord].values,
            ),
            # Specifying meta this way resolves this error,
            #   AttributeError: 'DataFrame' object has no attribute 'name'
            meta=(self.lat_index_column, int),
        )
        longitude_index: dd.Series = self.observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df[self.longitude_column],
                self._grid_prototype[self._longitude_coord].values,
            ),
            meta=(self.lon_index_column, int),
        )
        time_index: dd.Series = self.observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df[self.time_column],
                self._grid_prototype[self._time_coord].values,
                valid_window=self._time_window_delta,
            ),
            meta=(self.time_index_column, int),
        )
        return ObservationCoordinates(
            level_index,
            latitude_index,
            longitude_index,
            time_index,
        )

    def observations_index_to_grid(self, indexing_format: IndexingFormat, stack_on_axis: int | None = None) -> da.Array:
        coords_of_observation: dict[str, da.Array] = self.coordinates_of_observations().as_arrays()

        map_coord_axes_to_columns: dict[str, str] = {
            self._longitude_coord: self.lon_index_column,
            self._latitude_coord: self.lat_index_column,
            self._time_coord: self.time_index_column,
            self._level_coord: self.level_index_column,
        }
        # Retrieves the index for each row of data and stores them as dask arrays
        indexing_columns: list[da.Array] = [
            coords_of_observation[map_coord_axes_to_columns[coord_name]] for coord_name in self._coordinate_axes
        ]

        axis_order: list[int] = list(self._grid_prototype.get_axis_num(self._coordinate_axes))
        # Order of coordinate axes are not known beforehand. Therefore, use the axis order so that the index
        # values matches the dimension of the array
        in_order_of_array_coords: list[da.Array] = map_order(indexing_columns, axis_order)

        match indexing_format:
            case IndexingFormat.COORDINATES:
                if stack_on_axis is None:
                    raise TypeError("stack_on_axis cannot be None, if coordinates are to be returned")
                # Combine the coordinates on the specified axis
                return da.stack(in_order_of_array_coords, axis=stack_on_axis)
            case IndexingFormat.FLAT:
                if stack_on_axis is not None:
                    raise TypeError("stack_on_axis must be None, if flattened indices are to be returned")
                # Combine them such that coordinates contains [(x1, y1, z1, t1), ..., (xn, yn, zn, tn)] which are the
                # values to slices from the computed diagnostic
                coordinates: da.Array = da.stack(in_order_of_array_coords, axis=1)
                return da.ravel_multi_index(coordinates.T, self._grid_prototype.shape)
            case _ as unreachable:
                assert_never(unreachable)

    def has_observation(self) -> xr.DataArray:
        raveled_index: da.Array = self.observations_index_to_grid(IndexingFormat.FLAT)
        store_into: da.Array = da.zeros(self._grid_prototype.size, dtype=bool).persist()

        if raveled_index.size > 0:
            store_into[raveled_index] = True  # pyright:ignore[reportIndexIssue]

        store_into = store_into.reshape(self._grid_prototype.shape).rechunk(self._grid_prototype.chunks)  # pyright: ignore[reportAttributeAccessIssue]

        return xr.DataArray(
            data=store_into,
            coords=self._grid_prototype.coords,
            dims=self._grid_prototype.dims,
            name="number_of_observations",
        )

    def grid_has_positive_turbulence_observation(
        self, positive_obs_condition: "list[DiagnosticValidationCondition]"
    ) -> xr.Dataset:
        assert {condition.observed_turbulence_column_name for condition in positive_obs_condition}.issubset(
            self.observational_data.columns
        )
        raveled_index: da.Array = self.observations_index_to_grid(IndexingFormat.FLAT)

        data_vars: dict[str, xr.DataArray] = {}
        for condition in positive_obs_condition:
            was_observed: da.Array = (
                self.observational_data[condition.observed_turbulence_column_name] > condition.value_greater_than
            ).to_dask_array(lengths=True)

            store_into_dataarray: da.Array = da.zeros(self._grid_prototype.size, dtype=bool)
            store_into_dataarray[raveled_index] = was_observed  # pyright: ignore[reportIndexIssue]
            store_into_dataarray = store_into_dataarray.reshape(self._grid_prototype.shape).rechunk(  # pyright:ignore[reportAttributeAccessIssue]
                self._grid_prototype.chunks
            )

            data_vars[condition.observed_turbulence_column_name] = xr.DataArray(
                data=store_into_dataarray.persist(),
                coords=self._grid_prototype.coords,
                dims=self._grid_prototype.dims,
                name=condition.observed_turbulence_column_name,
            )

        return xr.Dataset(data_vars=data_vars, coords=self._grid_prototype.coords)

    def observation_data_with_indices(self) -> dd.DataFrame:
        coords_of_observation: ObservationCoordinates = self.coordinates_of_observations()
        return self.observational_data.assign(
            **{
                self.level_index_column: coords_of_observation.vertical_index,
                self.lat_index_column: coords_of_observation.latitude_index,
                self.lon_index_column: coords_of_observation.latitude_index,
                self.time_index_column: coords_of_observation.time_index,
            }
        )


class DiagnosticsAmdarDataHarmoniser:
    """
    Class brings together the turbulence diagnostics computed from meteorological data and AMDAR observational data
    to form a unified data set
    """

    _amdar_data: "AmdarTurbulenceData"
    _diagnostics_suite: "DiagnosticSuite"

    TIME_WINDOW_DELTA: ClassVar[np.timedelta64] = np.timedelta64(3, "h")

    def __init__(self, amdar_data: "AmdarTurbulenceData", diagnostics_suite: "DiagnosticSuite") -> None:
        self._amdar_data = amdar_data
        self._diagnostics_suite = diagnostics_suite

    @staticmethod
    def _check_time_window_within_met_data(
        time_window: "Limits[np.datetime64]",
        first_diagnostic: "xr.DataArray",
    ) -> None:
        time_coordinate: xr.DataArray = first_diagnostic["time"]
        if time_window.lower < time_coordinate.min() or time_window.upper > time_coordinate.max():
            raise NotWithinTimeFrameError("Time window is not within time coordinate of met data")

    def _get_grid_prototype(self, time_window: "Limits[np.datetime64]") -> "xr.DataArray":
        grid_prototype: xr.DataArray = self._diagnostics_suite.get_prototype_computed_diagnostic()
        self._check_time_window_within_met_data(time_window, grid_prototype)
        return grid_prototype

    def _get_observational_data(self, time_window: "Limits[np.datetime64]") -> dd.DataFrame:
        return self._amdar_data.clip_to_time_window(time_window).persist()

    @property
    def common_time_column_name(self) -> str:
        return "time_index"

    @property
    def grid_box_column_name(self) -> str:
        return "index_right"

    @property
    def latitude_index_column(self) -> str:
        return "lat_index"

    @property
    def longitude_index_column(self) -> str:
        return "lon_index"

    @property
    def vertical_coordinate_index_column(self) -> str:
        return "level_index"

    @property
    def coordinate_axes(self) -> list[str]:
        return ["longitude", "latitude", "pressure_level", "time"]

    @property
    def index_column_names(self) -> list[str]:
        return [
            self.longitude_index_column,
            self.latitude_index_column,
            self.vertical_coordinate_index_column,
            self.common_time_column_name,
        ]

    def coordinate_axes_to_index_col_name(self) -> dict[str, str]:
        return dict(zip(self.coordinate_axes, self.index_column_names, strict=True))

    @property
    def harmonised_diagnostics(self) -> list[str]:
        return self._diagnostics_suite.diagnostic_names()

    def _get_gridded_coordinates(
        self, observational_data: dd.DataFrame, grid_prototype: xr.DataArray, stack_on_axis: int
    ) -> da.Array:
        assert grid_prototype.ndim == len(self.coordinate_axes), (
            "Grid prototype must have same number of dimensions as defined coordinates"
        )

        coords_of_observation: dict[str, da.Array] = self._coordinates_of_observations(
            observational_data, grid_prototype
        )
        # Retrieves the index for each row of data and stores them as dask arrays
        indexing_columns: list[da.Array] = [
            coords_of_observation[self.coordinate_axes_to_index_col_name()[coord_name]]
            for coord_name in self.coordinate_axes
        ]

        # Get order of the coordinate axes of the prototype diagnostic
        assert set(self.coordinate_axes).issubset(grid_prototype.coords)
        axis_order: list[int] = list(grid_prototype.get_axis_num(self.coordinate_axes))

        # Order of coordinate axes are not known beforehand. Therefore, use the axis order so that the index
        # values matches the dimension of the array
        in_order_of_array_coords: list[da.Array] = map_order(indexing_columns, axis_order)

        # Combine the coordinates on the specified axis
        return da.stack(in_order_of_array_coords, axis=stack_on_axis)

    def _create_diagnostic_value_series(
        self,
        grid_prototype: "xr.DataArray",
        observational_data: dd.DataFrame,
        dataframe_meta: dict[str, pd.Series],
    ) -> dict["DiagnosticName", dd.Series]:
        # Combine them such that coordinates contains [(x1, y1, z1, t1), ..., (xn, yn, zn, tn)] which are the
        # values to slices from the computed diagnostic
        coordinates: da.Array = self._get_gridded_coordinates(observational_data, grid_prototype, stack_on_axis=1)
        assert coordinates.shape[1] == grid_prototype.ndim, "Shape of coordinates should be (nnz, ndim)"

        # da.Array doesn't support np advanced indexing such that coordinates can be used directly
        #   => index into it as a contiguous array
        flattened_index: da.Array = da.ravel_multi_index(coordinates.T, grid_prototype.shape).persist()
        for name in self.harmonised_diagnostics:
            dataframe_meta[name] = pd.Series(dtype=float)

        return {
            diagnostic_name: da.ravel(computed_diagnostic.data)[flattened_index]
            # .compute_chunk_sizes()
            .to_dask_dataframe(
                meta=pd.DataFrame({diagnostic_name: pd.Series(dtype=float)}),
                index=observational_data.index,  # ensures it matches observational_data.index
                columns=diagnostic_name,
            )
            .persist()
            for diagnostic_name, computed_diagnostic in self._diagnostics_suite.computed_values(
                "Vectorised indexing to get diagnostic value for observation",
            )
        }

    def nearest_diagnostic_value(self, time_window: "Limits[np.datetime64]") -> "dd.DataFrame":
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        dataframe_meta: dict[str, pd.Series] = get_dataframe_dtypes(observational_data)
        dataframe_meta["level_index"] = pd.Series(dtype=int)
        dataframe_meta[self.latitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.longitude_index_column] = pd.Series(dtype=int)
        dataframe_meta[self.common_time_column_name] = pd.Series(dtype=int)

        observational_data = observational_data.assign(
            level_index=map_values_to_nearest_index_irregular_grid(
                observational_data["level"], grid_prototype["pressure_level"].values
            )
        ).persist()
        observational_data = observational_data.map_partitions(
            lambda df: df.assign(
                **{
                    self.latitude_index_column: map_values_to_nearest_coordinate_index(
                        df.latitude,
                        grid_prototype["latitude"].values,
                    ),
                    self.longitude_index_column: map_values_to_nearest_coordinate_index(
                        df.longitude,
                        grid_prototype["longitude"].values,
                    ),
                    self.common_time_column_name: map_values_to_nearest_coordinate_index(
                        df.datetime,
                        grid_prototype["time"].values,
                        valid_window=self.TIME_WINDOW_DELTA,
                    ),
                },
            ),
            meta=dd.from_pandas(pd.DataFrame(dataframe_meta), npartitions=observational_data.npartitions),
        ).persist()

        diagnostics_columns = self._create_diagnostic_value_series(grid_prototype, observational_data, dataframe_meta)
        for diagnostic_col_name, as_column in diagnostics_columns.items():
            observational_data = dd.map_partitions(
                lambda base_df, new_col, col_name=diagnostic_col_name: base_df.assign(**{col_name: new_col}),
                observational_data,
                as_column,
            ).persist()
        return observational_data

    def _coordinates_of_observations(
        self, observational_data: dd.DataFrame, grid_prototype: "xr.DataArray"
    ) -> dict[str, da.Array]:
        level_index: da.Array = cast(
            "da.Array",
            map_values_to_nearest_index_irregular_grid(
                observational_data["level"], grid_prototype["pressure_level"].values
            ),
        )
        latitude_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.latitude,
                grid_prototype["latitude"].values,
            ),
            # Specifying meta this way resolves this error,
            #   AttributeError: 'DataFrame' object has no attribute 'name'
            meta=(self.latitude_index_column, int),
        ).to_dask_array(lengths=True)
        longitude_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.longitude,
                grid_prototype["longitude"].values,
            ),
            meta=(self.longitude_index_column, int),
        ).to_dask_array(lengths=True)
        time_index: da.Array = observational_data.map_partitions(
            lambda df: map_values_to_nearest_coordinate_index(
                df.datetime,
                grid_prototype["time"].values,
                valid_window=self.TIME_WINDOW_DELTA,
            ),
            meta=(self.common_time_column_name, int),
        ).to_dask_array(lengths=True)
        return {
            self.vertical_coordinate_index_column: level_index,
            self.latitude_index_column: latitude_index,
            self.longitude_index_column: longitude_index,
            self.common_time_column_name: time_index,
        }

    def grid_turbulence_observation_frequency(
        self, time_window: "Limits[np.datetime64]", positive_obs_condition: "list[DiagnosticValidationCondition]"
    ) -> "xr.Dataset":
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        # Coords for sparse COO must be in (ndim, nnz)
        gridded_coordinates: da.Array = self._get_gridded_coordinates(
            observational_data, grid_prototype, stack_on_axis=0
        )  # shape = (4, n)
        assert gridded_coordinates.shape[0] == grid_prototype.ndim, (
            "Shape of gridded coordinates must be in (ndim, nnz) for sparse COO matrix"
        )
        target_chunks: tuple[tuple[int, ...], ...] | None = grid_prototype.chunks
        assert target_chunks is not None

        as_coo: dict[str, da.Array] = {
            condition.observed_turbulence_column_name: to_coo_array(
                gridded_coordinates,
                (
                    observational_data[condition.observed_turbulence_column_name] > condition.value_greater_than
                ).to_dask_array(lengths=True),
                # Returns a count of the number of times turbulence was observed at each grid point
                np.dtype(np.int_),
                grid_prototype.shape,
                target_chunks,
                0,
                # Arrays being mapped over have shape (4, n), both axes need to be dropped as the shape of the
                # resulting array needs to match (n_lon, n_lat, n_level, n_time)
                axis_to_drop=[0, 1],
            )
            for condition in positive_obs_condition
        }

        return xr.Dataset(
            data_vars={
                condition_name: xr.DataArray(
                    data=regridded,
                    coords=grid_prototype.coords,
                    dims=grid_prototype.dims,
                    name=f"{condition_name}_count",
                )
                for condition_name, regridded in as_coo.items()
            },
            coords=grid_prototype.coords,
        )

    def grid_positive_turb_observations(
        self, time_window: "Limits[np.datetime64]", positive_obs_condition: "list[DiagnosticValidationCondition]"
    ) -> xr.Dataset:
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        # target_columns = [cond.observed_turbulence_column_name for cond in positive_obs_condition] + [
        #     "longitude",
        #     "latitude",
        #     "datetime",
        #     "level",
        # ]
        # observational_data: dd.DataFrame = self._get_observational_data(time_window)[target_columns]
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        coordinates: da.Array = self._get_gridded_coordinates(observational_data, grid_prototype, stack_on_axis=1).T
        # assert coordinates.shape[1] == grid_prototype.ndim, "Shape of coordinates should be (nnz, ndim)"

        data_variables: dict[str, xr.DataArray] = {}
        for condition in positive_obs_condition:
            was_observed: da.Array = (
                observational_data[condition.observed_turbulence_column_name] > condition.value_greater_than
            ).to_dask_array(lengths=True)

            positive_obs_idx: da.Array = da.ravel_multi_index(
                coordinates[:, da.flatnonzero(was_observed)], grid_prototype.shape
            )
            positive_obs_idx.compute_chunk_sizes()

            data_for_da: da.Array
            if positive_obs_idx.size > 0:
                positive_obs_idx = positive_obs_idx.persist()

                data_for_da: da.Array = da.zeros(grid_prototype.shape, dtype=np.int_).ravel().persist()
                data_for_da[positive_obs_idx] = 1  # pyright: ignore[reportIndexIssue]

                # Logic for adding the number of observations up
                # pyright: ignore[reportGeneralTypeIssues]
                # unique_values, unique_counts = da.unique(positive_obs_idx, return_counts=True)
                # # Arrays have unknown sizes so evaluate them
                # unique_counts = unique_counts.compute_chunk_sizes()
                # unique_values = unique_values.compute_chunk_sizes()
                # multiple_obs_mask = (unique_counts > 1).compute_chunk_sizes()
                # multiple_obs_idx = unique_values[multiple_obs_mask].compute_chunk_sizes()
                # if multiple_obs_idx.size > 0:
                #     data_for_da[multiple_obs_idx] = unique_counts[multiple_obs_idx]
                # test_grid_turbulence_observation_frequency_multiple_chunks

                # pyright - false positive on there not being the mreshape method on dask.Array
                data_for_da = data_for_da.reshape(grid_prototype.shape)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                data_for_da = da.zeros(grid_prototype.shape)

            data_variables[condition.observed_turbulence_column_name] = xr.DataArray(
                data=data_for_da.persist(),
                coords=grid_prototype.coords,
                dims=grid_prototype.dims,
            )

        return xr.Dataset(data_vars=data_variables, coords=grid_prototype.coords)

    def grid_total_observations_count(self, time_window: "Limits[np.datetime64]") -> xr.DataArray:
        grid_prototype: xr.DataArray = self._get_grid_prototype(time_window)
        observational_data: dd.DataFrame = self._get_observational_data(time_window)

        # Coords for sparse COO must be in (ndim, nnz)
        gridded_coordinates: da.Array = self._get_gridded_coordinates(
            observational_data, grid_prototype, stack_on_axis=0
        )  # shape = (4, n)
        assert gridded_coordinates.shape[0] == grid_prototype.ndim, (
            "Shape of gridded coordinates must be in (ndim, nnz) for sparse COO matrix"
        )
        target_chunks: tuple[tuple[int, ...], ...] | None = grid_prototype.chunks
        assert target_chunks is not None

        return xr.DataArray(
            data=to_coo_array(
                gridded_coordinates,
                # Chunks of gridded coords is in the shape of ((chunks_for_ndim), (chunks_for_length))
                # So, for the ones array, use chunks of length dimension
                da.ones(gridded_coordinates.shape[1], dtype=int, chunks=gridded_coordinates.chunks[1]),
                np.dtype(np.int_),
                grid_prototype.shape,
                target_chunks,
                0,
                axis_to_drop=[0, 1],
            ),
            coords=grid_prototype.coords,
            dims=grid_prototype.dims,
            name="number_of_observations",
        )


def _any_aggregation() -> dd.Aggregation:
    return dd.Aggregation(
        name="aggregate_by_any",
        chunk=lambda within_partition: within_partition.any(),
        agg=lambda across_partitions: across_partitions.any(),
    )


def metric_based_aggregation(
    condition: "DiagnosticValidationCondition",
    minimum_group_size: int,
    aggregation_function: AggregationMetricOption,
    integration_scheme: Literal["trapz", "simpson"] = "simpson",
) -> dd.Aggregation:
    assert minimum_group_size > 0, "Minimum group size must be positive"

    def apply_condition(on_chunk: "pd.api.typing.SeriesGroupBy") -> "pd.Series":
        return on_chunk.apply(lambda row: row > condition.value_greater_than)

    def aggregate_with_cumsum(on_partition: "pd.api.typing.SeriesGroupBy") -> "pd.Series":
        return on_partition.cumsum()

    def prevalence_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        return float(on_group.to_numpy()[-1] / group_size)

    def tss_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        binary_classification: BinaryClassificationRateFromLabels | None = binary_classification_rate_from_cumsum(
            on_group,
        )
        if binary_classification is None:
            return -1

        tss: NDArray = binary_classification.true_positives_rate + binary_classification.false_positives_rate - 1
        return tss.max()

    def auc_from_cumsum(on_group: "pd.Series") -> float:
        group_size: int = on_group.size
        if group_size < minimum_group_size:
            return np.nan

        binary_classification: BinaryClassificationRateFromLabels | None = binary_classification_rate_from_cumsum(
            on_group,
        )
        if binary_classification is None:
            return -1

        if integration_scheme == "trapz":
            return float(
                np.trapezoid(binary_classification.true_positives_rate, x=binary_classification.false_positives_rate),
            )
        return float(
            integrate.simpson(binary_classification.true_positives_rate, x=binary_classification.false_positives_rate),
        )

    finalisers: dict[AggregationMetricOption, Callable[[pd.Series], float]] = {
        AggregationMetricOption.AUC: auc_from_cumsum,
        AggregationMetricOption.TSS: tss_from_cumsum,
        AggregationMetricOption.PREVALENCE: prevalence_from_cumsum,
    }

    return dd.Aggregation(
        name=f"{aggregation_function}_on_{condition.observed_turbulence_column_name}_{condition.value_greater_than:0.2f}",
        chunk=apply_condition,
        agg=aggregate_with_cumsum,
        # See https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
        # The example for nunique has this groupby
        # My understanding is that it forces the partition into the original groups, on which, I am applying
        # the cumsum => result is still separated based on the groups
        finalize=lambda on_aggregation_result: on_aggregation_result.groupby(
            level=list(range(on_aggregation_result.index.nlevels)),
        ).apply(finalisers[aggregation_function]),
    )


@dataclasses.dataclass
class RocVerificationResult:
    # first key: amdar column name
    # second key: diagnostic column name
    by_amdar_col_then_diagnostic: dict[str, dict[str, BinaryClassificationResult]]
    _auc_by_amdar_then_diagnostic: dict[str, dict[str, float]] | None = None

    # define iterators and getters on this class
    def iterate_by_amdar_column(self) -> Generator[tuple[str, dict[str, BinaryClassificationResult]], None, None]:
        yield from self.by_amdar_col_then_diagnostic.items()

    def _area_under_curve(self) -> dict[str, dict[str, float]]:
        if self._auc_by_amdar_then_diagnostic is None:
            auc = defaultdict(dict)
            for column, by_diagnostic in self.by_amdar_col_then_diagnostic.items():
                for diagnostic_name, roc_results in by_diagnostic.items():
                    auc[column][diagnostic_name] = area_under_curve(
                        roc_results.false_positives,
                        roc_results.true_positives,
                    )
            self._auc_by_amdar_then_diagnostic = dict(auc)
        return self._auc_by_amdar_then_diagnostic

    def auc_for_amdar_column(self, amdar_column: str) -> dict[str, float]:
        return self._area_under_curve()[amdar_column]


# Keep this extendable for verification against other forms of data??
class DiagnosticsAmdarVerification:
    _data_harmoniser: "AmdarDataHarmoniser"
    _harmonised_data: "dd.DataFrame | None"
    _turbulence_diagnostics: xr.Dataset

    _grid_box_column: str

    def __init__(
        self,
        data_harmoniser: AmdarDataHarmoniser,
        turbulence_diagnostics: xr.Dataset,
        grid_box_column: str = "index_right",
    ) -> None:
        assert grid_box_column in data_harmoniser.observational_data.columns

        if not set(data_harmoniser.harmonised_coordinates()).issubset(turbulence_diagnostics.coords):
            raise ValueError(
                "Coordinates of grid prototype amdar harmonised with does not match turbulence diagnostics"
            )
        for harmonised_coord in data_harmoniser.harmonised_coordinates():
            if not np.issubdtype(turbulence_diagnostics[harmonised_coord].dtype, np.datetime64) and not np.allclose(
                turbulence_diagnostics[harmonised_coord], data_harmoniser.grid_prototype[harmonised_coord]
            ):
                raise ValueError(
                    "Values of coordinates of grid prototype amdar harmonised are not equal to diagnostics"
                )

        if not is_dask_collection(turbulence_diagnostics):
            raise TypeError("Turbulence diagnostics must have dask-backed arrays")

        self._data_harmoniser = data_harmoniser
        self._harmonised_data = None

        self._turbulence_diagnostics = turbulence_diagnostics
        self._grid_box_column = grid_box_column

    def _add_nearest_diagnostic_value(self) -> dd.DataFrame:
        raveled_index: da.Array = self._data_harmoniser.observations_index_to_grid(IndexingFormat.FLAT).persist()
        target_dataframe: dd.DataFrame = self._data_harmoniser.observation_data_with_indices().persist()
        return target_dataframe.assign(
            **{
                diagnostic_name: da.ravel(diagnostic_dataarray.data)[raveled_index]
                .to_dask_dataframe(
                    meta=pd.DataFrame({diagnostic_name: pd.Series(dtype=float)}),
                    index=target_dataframe.index,
                    columns=diagnostic_name,
                )
                .persist()
                for diagnostic_name, diagnostic_dataarray in self._turbulence_diagnostics.data_vars.items()
            }
        )

    @property
    def harmonised_diagnostics(self) -> list[str]:
        return list(map(str, self._turbulence_diagnostics.keys()))

    @property
    def grid_box_column(self) -> str:
        return self._grid_box_column

    @property
    def data(self) -> "dd.DataFrame":
        if self._harmonised_data is None:
            self._harmonised_data = self._add_nearest_diagnostic_value().persist()
        assert isinstance(self._harmonised_data, dd.DataFrame)
        return self._harmonised_data

    def _grid_spatial_columns(self, group_by_strategy: SpatialGroupByStrategy) -> list[str]:
        match group_by_strategy:
            case SpatialGroupByStrategy.GRID_BOX:
                return [
                    self.grid_box_column,
                    AmdarDataHarmoniser.level_index_column,
                ]
            case SpatialGroupByStrategy.GRID_POINT:
                return [
                    AmdarDataHarmoniser.lat_index_column,
                    AmdarDataHarmoniser.lon_index_column,
                    AmdarDataHarmoniser.level_index_column,
                ]
            case SpatialGroupByStrategy.HORIZONTAL_BOX:
                return [self.grid_box_column]
            case SpatialGroupByStrategy.HORIZONTAL_POINT:
                return [AmdarDataHarmoniser.lat_index_column, AmdarDataHarmoniser.lon_index_column]
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def _get_partition_level_values(partition: pd.Index, level_name: str) -> pd.Index:
        return partition.get_level_values(level_name)

    @staticmethod
    def _concat_columns_as_str(data_frame: dd.DataFrame, column_names: list[str], separator: str = "_") -> dd.Series:
        multi_index_columns = [data_frame[col_name].astype(str) for col_name in column_names]
        return functools.reduce(lambda left, right: left + separator + right, multi_index_columns)

    def _retrieve_grouped_columns(
        self,
        data_frame: dd.DataFrame,
        grouped_columns: list[str],
        trigger_reset_index: bool,
    ) -> dd.DataFrame:
        data_frame = data_frame.map_partitions(
            lambda df: df.assign(
                **{
                    grouped_column: self._get_partition_level_values(df.index, grouped_column)
                    for grouped_column in grouped_columns
                },
            ),
        )

        if self.grid_box_column in set(grouped_columns) and trigger_reset_index:
            # set_index is an expensive operation due to the shuffles it triggers
            #   However, this cost has been undertaken as it means that the data can be joined INTO a GeoPandas
            #   grid turning the entire thing into a GeoDataFrame without having to invoke dgpd.from_dask_dataframe.
            #   It also means that the crs will be inherited and not require manual intervention
            data_frame = data_frame.set_index(data_frame[self.grid_box_column], drop=True)
        elif trigger_reset_index:
            data_frame = data_frame.reset_index(drop=True)

        return data_frame.persist()

    def _retrieve_index_column_values(
        self,
        data_frame: dd.DataFrame,
        grouped_columns: list[str],
        prototype_array: "xr.DataArray",
        drop_index_cols: bool,
    ) -> dd.DataFrame:
        assert {"pressure_level", "longitude", "latitude"}.issubset(prototype_array.coords)
        index_columns: list[str] = list(
            {
                AmdarDataHarmoniser.lon_index_column,
                AmdarDataHarmoniser.lat_index_column,
                AmdarDataHarmoniser.level_index_column,
            }.intersection(grouped_columns),
        )
        if not index_columns:  # set is empty
            return data_frame

        col_name_mapping: dict[str, str] = {
            AmdarDataHarmoniser.lon_index_column: "longitude",
            AmdarDataHarmoniser.lat_index_column: "latitude",
            AmdarDataHarmoniser.level_index_column: "pressure_level",
        }

        data_frame_dtypes = get_dataframe_dtypes(data_frame)
        for index_col_name in index_columns:
            value_col_name = col_name_mapping[index_col_name]
            data_frame_dtypes[value_col_name] = pd.Series(dtype=float)

        data_frame = data_frame.map_partitions(
            lambda df: df.assign(
                **{
                    col_name_mapping[col_name]: map_index_to_coordinate_value(
                        df[col_name],
                        prototype_array[col_name_mapping[col_name]].to_numpy(),
                    )
                    for col_name in index_columns
                },
            ),
            meta=pd.DataFrame(data_frame_dtypes),
        )

        if drop_index_cols:
            return data_frame.drop(columns=index_columns)

        return data_frame

    def num_obs_per(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        group_by_strategy: SpatialGroupByStrategy,
    ) -> dd.DataFrame:
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)
        turbulence_col: str = validation_conditions[0].observed_turbulence_column_name
        groupby_columns: list[str] = self._grid_spatial_columns(group_by_strategy)
        minimum_columns: list[str] = [*groupby_columns, turbulence_col]
        num_obs = target_data[minimum_columns].groupby(groupby_columns).count()
        num_obs = self._retrieve_grouped_columns(num_obs, groupby_columns, True)
        return num_obs.rename(columns={turbulence_col: "num_obs"})

    def _spatial_data_grouping(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        group_by_strategy: SpatialGroupByStrategy,
    ) -> dd.DataFrame:
        target_data: dd.DataFrame = self.data
        space_columns = self._grid_spatial_columns(group_by_strategy)
        validation_columns = self._get_validation_column_names(validation_conditions)
        target_cols = space_columns + validation_columns + self.harmonised_diagnostics
        target_data = target_data[target_cols]

        # 1) Use the columns that will be used to spatially group the data as the index of the dataframe
        #    Without a column as the index, the aggregation to compute AUC fails with,
        #       ValueError: If using all scalar values, you must pass an index
        target_data["multi_index"] = self._concat_columns_as_str(target_data, space_columns)

        # Dask doesn't support multi-index so we've got to use a "poor man's" multi-index
        return target_data.set_index("multi_index").persist()

    def aggregate_by_auc(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
        prototype_diagnostic: "xr.DataArray",
        group_by_strategy: SpatialGroupByStrategy,
        minimum_group_size: int,
        aggregation_metric: AggregationMetricOption,
    ) -> dict[str, dd.DataFrame]:
        space_columns = self._grid_spatial_columns(group_by_strategy)
        validation_columns = self._get_validation_column_names(validation_conditions)
        target_data: dd.DataFrame = self._spatial_data_grouping(validation_conditions, group_by_strategy)

        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: metric_based_aggregation(
                condition,
                minimum_group_size,
                aggregation_metric,
            )
            for condition in validation_conditions
        }
        auc_by_diagnostic: dict[str, dd.DataFrame] = {}
        for diagnostic_name in self.harmonised_diagnostics:
            columns_for_diagnostic: list[str] = [*space_columns, diagnostic_name, *validation_columns]
            data_for_diagnostic: dd.DataFrame = target_data[columns_for_diagnostic]
            # 2) Sort values so that diagnostic is in descending order as required by the ROC calculation
            #    The set_index() operation performs a sort => need to do sort after and for each diagnostic
            data_for_diagnostic = data_for_diagnostic.sort_values(by=diagnostic_name, ascending=False)
            data_for_diagnostic = data_for_diagnostic.drop(columns=[diagnostic_name])

            # Specify sort=False to get better performance
            # See https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.groupby.html#dask-dataframe-dataframe-groupby
            grid_point_auc: dd.DataFrame = data_for_diagnostic.groupby(space_columns, sort=False).aggregate(
                aggregation_spec,
                meta={col: pd.Series(dtype=float) for col in validation_columns},
            )
            # 3) Drop values which do not meet the minimum number of observations
            grid_point_auc = grid_point_auc.dropna()
            # 4) Make values < 0 (another form of invalid values, e.g. no positive observations) into NaNs
            grid_point_auc = grid_point_auc.where(grid_point_auc >= 0, other=np.nan).persist()  # pyright: ignore [reportOperatorIssue]

            grid_point_auc = self._retrieve_grouped_columns(grid_point_auc, space_columns, True)
            grid_point_auc = self._retrieve_index_column_values(
                grid_point_auc,
                space_columns,
                prototype_diagnostic,
                True,
            )
            auc_by_diagnostic[diagnostic_name] = grid_point_auc.persist()

        return auc_by_diagnostic

    def nearest_value_roc(
        self,
        validation_conditions: "list[DiagnosticValidationCondition]",
    ) -> RocVerificationResult:
        turbulence_diagnostics = self.harmonised_diagnostics
        validation_columns = self._get_validation_column_names(validation_conditions)
        space_time_columns: list[str] = [
            AmdarDataHarmoniser.lat_index_column,
            AmdarDataHarmoniser.lon_index_column,
            AmdarDataHarmoniser.level_index_column,
            AmdarDataHarmoniser.time_index_column,
        ]
        target_columns: list[str] = space_time_columns + turbulence_diagnostics + validation_columns
        target_data = self.data[target_columns]

        # Apply condition so that any can be used at the groupby stage
        for condition in validation_conditions:
            target_data[condition.observed_turbulence_column_name] = (
                target_data[condition.observed_turbulence_column_name] > condition.value_greater_than
            )
        grouped_by_space_time = target_data.groupby(space_time_columns)
        aggregation_spec: dict = {
            condition.observed_turbulence_column_name: _any_aggregation() for condition in validation_conditions
        }
        for diagnostic_column in turbulence_diagnostics:
            aggregation_spec[diagnostic_column] = "mean"
        aggregated_data = grouped_by_space_time.aggregate(aggregation_spec).persist()

        result: defaultdict[str, dict[str, BinaryClassificationResult]] = defaultdict(dict)
        for diagnostic_val_col in turbulence_diagnostics:
            # descending values
            subset_df = (
                aggregated_data[[*validation_columns, diagnostic_val_col]]
                .sort_values(diagnostic_val_col, ascending=False)
                .persist()
            )
            diagnostic_col_as_array: da.Array = subset_df[diagnostic_val_col].to_dask_array(lengths=True).persist()
            for amdar_turbulence_col in validation_columns:
                result[amdar_turbulence_col][diagnostic_val_col] = received_operating_characteristic(
                    subset_df[amdar_turbulence_col].to_dask_array(lengths=diagnostic_col_as_array.chunks[0]).persist(),
                    diagnostic_col_as_array,
                    num_intervals=-1,
                )
        return RocVerificationResult(dict(result))

    @staticmethod
    def _get_validation_column_names(validation_conditions: "list[DiagnosticValidationCondition]") -> list[str]:
        return [condition.observed_turbulence_column_name for condition in validation_conditions]
