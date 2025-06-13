from typing import TYPE_CHECKING, Any, ClassVar, FrozenSet, Iterable, List

import dask.dataframe as dd

from rojak.core.data import AmdarData

if TYPE_CHECKING:
    import numpy as np


class UkmoAmdarData(AmdarData):
    TIME_COLUMNS: ClassVar[FrozenSet[str]] = frozenset({"year", "month", "day", "hour", "minute", "second"})
    TURBULENCE_COL_INDICES: ClassVar[List[int]] = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 20, 21, 22, 24, 25, 26, 27, 28,
                                                  30, 32]  # fmt: skip
    COLUMN_NAMES: ClassVar[List[str]] = ['year', 'month', 'day', 'hour', 'minute', 'second', 'registration_number',
                                           'call_sign', 'latitude', 'longitude', 'altitude', 'roll_angle', 'pressure',
                                           'flight_phase', 'wind_direction', 'wind_speed', 'vert_gust_velocity',
                                           'vert_gust_acceleration', 'turbulence_degree', 'air_temperature',
                                           'relative_humidity']  # fmt: skip

    def __init__(self, path: str | list) -> None:
        super().__init__(path)

    def load(
        self, target_columns: Iterable[str | int] | None = None, column_names: List[str] | None = None
    ) -> dd.DataFrame:
        if target_columns is None:
            target_columns = UkmoAmdarData.TURBULENCE_COL_INDICES
        if column_names is None:
            column_names = UkmoAmdarData.COLUMN_NAMES

        col_names = set(column_names)
        assert col_names.issuperset(UkmoAmdarData.TIME_COLUMNS), "Columns must contain all the time column names"
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

    def call_compute_closest_pressure_level(
        self, data_frame: "dd.DataFrame", pressure_levels: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "dd.DataFrame":
        return self._compute_closest_pressure_level(data_frame, pressure_levels, "altitude")
