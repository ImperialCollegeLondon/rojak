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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import dask.dataframe as dd

from rojak.core.data import AmdarDataRepository, AmdarTurbulenceData

if TYPE_CHECKING:
    import dask_geopandas as dgpd
    import numpy as np


class UkmoAmdarRepository(AmdarDataRepository):
    TIME_COLUMNS: ClassVar[frozenset[str]] = frozenset({"year", "month", "day", "hour", "minute", "second"})
    TURBULENCE_COL_INDICES: ClassVar[list[int]] = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 20, 21, 22, 24, 25, 26, 27, 28,
                                                  30, 32]  # fmt: skip
    COLUMN_NAMES: ClassVar[list[str]] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'registration_number',
                                           'call_sign', 'latitude', 'longitude', 'altitude', 'roll_angle', 'pressure',
                                           'flight_phase', 'wind_direction', 'wind_speed', 'vert_gust_velocity',
                                           'vert_gust_acceleration', 'turbulence_degree', 'air_temperature',
                                           'relative_humidity']  # fmt: skip

    def __init__(self, path: str | list) -> None:
        super().__init__(path)

    def load(
        self, target_columns: Iterable[str | int] | None = None, column_names: list[str] | None = None
    ) -> dd.DataFrame:
        if target_columns is None:
            target_columns = UkmoAmdarRepository.TURBULENCE_COL_INDICES
        if column_names is None:
            column_names = UkmoAmdarRepository.COLUMN_NAMES

        col_names = set(column_names)
        assert col_names.issuperset(UkmoAmdarRepository.TIME_COLUMNS), "Columns must contain all the time column names"
        assert "turbulence_degree" in col_names, "Turbulence degree must be in column names"

        # 1. skiprows - skips the 183 rows which contain the properties
        # 2. encoding - from utf-8 to cp1252 (legacy windows byte encoding) as UnicodeDecodeError: 'utf-8' codec can't
        #    decode byte: invalid start byte is occasionally thrown when reading in the data
        data: dd.DataFrame = dd.read_csv(
            self._path_to_files,
            skiprows=1,
            na_values=[-9999999],
            usecols=target_columns,
            header=0,
            names=column_names,
            encoding="cp1252",
        )
        data = data.fillna(value={"second": 0})  # Prevents NaNs from making valid datetime a NaT
        data["datetime"] = dd.to_datetime(data[["year", "month", "day", "hour", "minute", "second"]])
        data = data.drop(["year", "month", "day", "hour", "minute", "second"], axis=1)

        # Dictionary key 29 is turbulence degree
        data = data.dropna(subset=["turbulence_degree"])
        # Optimise the graph before manipulating the categories as it throws a key error without it
        # Also, do it before the `categorize` as it "eagerly computes the categories of the chosen columns"
        # data = data.optimize()
        # data = data.categorize(columns=["turbulence_degree"])
        # data["turbulence_degree"] = data["turbulence_degree"].cat.as_ordered()
        # data["turbulence_degree"] = data["turbulence_degree"].cat.rename_categories(
        #     {0: "none", 1: "light", 2: "moderate", 3: "severe"}
        # )

        return data.optimize()

    def _call_compute_closest_pressure_level(
        self, data_frame: "dd.DataFrame", pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "dd.Series":
        return self._compute_closest_pressure_level(data_frame, pressure_levels, "altitude")

    def _instantiate_amdar_turbulence_data_class(
        self, data_frame: "dd.DataFrame", grid: "dgpd.GeoDataFrame"
    ) -> "AmdarTurbulenceData":
        return UkmoAmdarTurbulenceData(data_frame, grid)

    def _time_column_rename_mapping(self) -> dict[str, str]:
        return {}


class UkmoAmdarTurbulenceData(AmdarTurbulenceData):
    def __init__(self, data_frame: "dd.DataFrame", grid: "dgpd.GeoDataFrame") -> None:
        super().__init__(data_frame, grid)

    def _minimum_altitude_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        return data_frame[data_frame["altitude"] >= self.MINIMUM_ALTITUDE]

    def _drop_manoeuvre_data_qc(self, data_frame: "dd.DataFrame") -> "dd.DataFrame":
        # roll_angle is NA in entire month of Jan and May in 2024
        return data_frame

    def turbulence_column_names(self) -> list[str]:
        return ["turbulence_degree"]
