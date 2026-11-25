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
IATA Turbulence Aware observation preprocessing

The data comes from IATA's Turbulence Aware platform, whose observations are automated, aircraft-based turbulence
reports. This module implements the :class:`rojak.core.data.DataPreprocessor` interface for the raw CSV files of
that platform (:class:`IataObsPreprocessor`): it keeps the observation fields, normalises the observation time to
naive UTC and writes the result to parquet. Optionally, reports whose turbulence value exceeds a threshold are
grouped into spatio-temporal clusters (:class:`ClusteringConfig`).
"""

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, override

import dask.dataframe as dd
import numpy as np
from dask.base import compute as dask_compute
from pydantic import BaseModel, Field

from rojak.core.data import DataPreprocessor
from rojak.datalib.iata._clustering import st_dbscan

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd
    from numpy.typing import NDArray


class ConditionColumn(StrEnum):
    """Turbulence value of a report that is compared with the clustering threshold"""

    PEAK = "peak"
    MEAN = "mean"


class ClusteringConfig(BaseModel, frozen=True, extra="forbid"):
    """
    Parameters for clustering the reports whose turbulence value exceeds a threshold

    A report qualifies for clustering when its :attr:`condition_column` value is strictly greater than
    :attr:`condition_threshold`. Qualifying reports are clustered by density in space, time and, optionally,
    altitude (see :func:`rojak.datalib.iata._clustering.st_dbscan`).

    Attributes:
        n_points: Minimum number of neighbouring reports, the report itself included, for a report to be a core
            point of a cluster. Must be at least 1.
        max_separation_km: Largest great-circle separation in kilometres between neighbours. Must not be negative.
        max_minutes: Largest time difference in minutes between neighbours. Must not be negative.
        condition_column: Turbulence value compared with :attr:`condition_threshold`.
        condition_threshold: Value the condition column must strictly exceed for a report to qualify. Any float is
            accepted.
        max_altitude_delta_ft: Largest altitude difference in feet between neighbours, or ``None`` to ignore
            altitude. Must not be negative.

    Raises:
        pydantic.ValidationError: If a value breaks the limits above; NaN is rejected as well. It is a
            :class:`ValueError`.
    """

    n_points: int = Field(ge=1)
    max_separation_km: float = Field(ge=0)
    max_minutes: float = Field(ge=0)
    condition_column: ConditionColumn
    condition_threshold: float
    max_altitude_delta_ft: float | None = Field(default=None, ge=0)

    def to_descriptor(self) -> str:
        """
        Short description of the parameters that is safe to use in a file name

        The separation, time window and altitude difference are rounded to the nearest whole number, so parameters
        that differ by less than that give the same descriptor. The threshold is rounded to two decimals, with the
        decimal point written as an underscore. When altitude is ignored, the altitude part reads ``no_alt``.
        """
        fs_safe_threshold = f"{self.condition_threshold:.2f}".replace(".", "_")
        altitude_setting = (
            f"{self.max_altitude_delta_ft:.0f}_ft" if self.max_altitude_delta_ft is not None else "no_alt"
        )
        return (
            f"n_{self.n_points}-{self.max_separation_km:.0f}_km-{self.max_minutes:.0f}_mins-{altitude_setting}-"
            f"{self.condition_column!s}_{fs_safe_threshold}"
        )


class IataObsPreprocessor(DataPreprocessor):
    """
    Preprocesses raw IATA Turbulence Aware CSV files into parquet files of turbulence observations

    The data comes from IATA's Turbulence Aware platform, whose observations are automated, aircraft-based
    turbulence reports. Only the observation fields are kept: ``time`` (the observation time as naive UTC),
    ``altitude`` (in feet), ``latitude``, ``longitude``, ``peak`` and ``mean``.

    Reports whose latitude, longitude, altitude or time is missing or not finite are dropped, so the output has
    fewer rows than the input whenever there are any.

    If a :class:`ClusteringConfig` is given, two further columns are written: ``cluster``, the cluster id of each
    qualifying report (-1 for noise and for every report that does not qualify), and ``in_cluster``, whether
    ``cluster`` is not -1. Altitudes are compared with :attr:`ClusteringConfig.max_altitude_delta_ft` directly, as
    both are in feet.

    Clustering needs every qualifying report at once, so it works as follows. The source is read twice: once to
    gather the qualifying reports and the number of them in each partition, which are clustered in memory, and once
    more to write the output, with each partition picking up its slice of the labels. This relies on the row
    selection being deterministic, so that both reads select the same rows in the same order. The full label array
    is embedded in each output task; that is fine up to roughly 8 MB of labels (about a million qualifying
    reports), beyond which the labels should be scattered to the workers instead. ``meta`` must be given to
    ``map_partitions`` for the output, as inferring it would run the partition function on an empty frame without
    any labels.
    """

    _RAW_TIME_COLUMN: ClassVar[str] = "observationtime"
    _TIME_COLUMN: ClassVar[str] = "time"
    _LAT_COLUMN: ClassVar[str] = "latitude"
    _LON_COLUMN: ClassVar[str] = "longitude"
    _ALTITUDE_COLUMN: ClassVar[str] = "altitude"
    _VALUE_COLUMNS: ClassVar[tuple[str, ...]] = (_ALTITUDE_COLUMN, _LAT_COLUMN, _LON_COLUMN, "peak", "mean")

    _CLUSTER_COLUMN: ClassVar[str] = "cluster"
    _IN_CLUSTER_COLUMN: ClassVar[str] = "in_cluster"
    _NOT_IN_CLUSTER: ClassVar[int] = -1  # Cluster id of a report that is in no cluster

    filepaths: "list[Path]"
    clustering: ClusteringConfig | None

    def __init__(
        self, input_path: "Path", glob_pattern: str | None = None, clustering: ClusteringConfig | None = None
    ) -> None:
        """
        Args:
            input_path: CSV file to preprocess or, if ``glob_pattern`` is given, the directory to search for CSV
                files matching ``glob_pattern``.
            glob_pattern: If provided, glob pattern used to find files within ``input_path``. If ``None``,
                ``input_path`` must be an existing file.
            clustering: If provided, the parameters for clustering the reports. If ``None``, no clustering is done.

        Raises:
            ValueError: If ``glob_pattern`` is ``None`` and ``input_path`` is not an existing file, or if no file in
                ``input_path`` matches ``glob_pattern``
        """
        if glob_pattern is None:
            if not input_path.is_file():
                raise ValueError(
                    f"File {input_path} is not a file. As glob pattern is not defined this must be a file",
                )
            self.filepaths = [input_path]
        else:
            # Sorted, so that the partition order of the output does not depend on the file system
            self.filepaths = sorted(path for path in input_path.glob(glob_pattern) if path.is_file())
            if not self.filepaths:
                raise ValueError(f"No file in {input_path} matches the glob pattern {glob_pattern}")
        self.clustering = clustering

    @classmethod
    def _drop_invalid_data_points(cls, raw_data: dd.DataFrame) -> dd.DataFrame:
        """Drop the reports whose latitude, longitude, altitude or time is missing or not finite."""
        is_valid_mask = (
            np.isfinite(raw_data[cls._LAT_COLUMN])
            & np.isfinite(raw_data[cls._LON_COLUMN])
            & np.isfinite(raw_data[cls._ALTITUDE_COLUMN])
            & ~raw_data[cls._TIME_COLUMN].isna()
        )
        return raw_data[is_valid_mask]

    def _read_observations(self) -> dd.DataFrame:
        """Lazily read the observation fields of every file, with ``time`` as naive UTC and invalid rows dropped."""
        frame = dd.read_csv(
            [str(path) for path in self.filepaths],
            usecols=[self._RAW_TIME_COLUMN, *self._VALUE_COLUMNS],
            parse_dates=[self._RAW_TIME_COLUMN],
            dtype=dict.fromkeys(self._VALUE_COLUMNS, np.float64),
        )
        # Raw data contains a trailing Z which makes the times parse as timezone-aware and numpy cannot represent
        frame = frame.assign(**{self._TIME_COLUMN: frame[self._RAW_TIME_COLUMN].dt.tz_convert(None)})
        return self._drop_invalid_data_points(frame[[self._TIME_COLUMN, *self._VALUE_COLUMNS]])

    def _construct_output_filename(self) -> str:
        """Name of the parquet output, which includes the clustering parameters when clustering is requested."""
        if self.clustering is None:
            return "unclustered-iata-data.parquet"
        return f"clustered-{self.clustering.to_descriptor()}.parquet"

    @override
    def apply_preprocessor(self, output_directory: "Path") -> None:
        """
        Preprocess every file in :attr:`filepaths` into one parquet dataset inside ``output_directory``

        The dataset is named ``unclustered-iata-data.parquet`` without clustering, or
        ``clustered-<descriptor>.parquet`` with it, where the descriptor is
        :meth:`ClusteringConfig.to_descriptor`. An existing dataset of the same name is overwritten.

        Args:
            output_directory: Directory to write the dataset into. It is created if it does not exist.
        """
        output_directory.mkdir(parents=True, exist_ok=True)

        observations = self._read_observations()

        if self.clustering is not None:
            observations = self._with_clusters(observations, self.clustering)
        observations.to_parquet(str(output_directory / self._construct_output_filename()))

    @classmethod
    def _with_clusters(cls, observations: dd.DataFrame, config: ClusteringConfig) -> dd.DataFrame:
        """
        ``observations`` with the ``cluster`` and ``in_cluster`` columns added

        Computes the qualifying reports and the per-partition counts (one read of the source), clusters them in
        memory and returns a lazy frame that reads the source again to attach the labels.
        """
        meets_condition = observations[_is_meet_condition(observations, config)]
        coordinates, reports_per_partition = dask_compute(
            meets_condition[[cls._TIME_COLUMN, cls._LAT_COLUMN, cls._LON_COLUMN, cls._ALTITUDE_COLUMN]],
            meets_condition.map_partitions(len, meta=(None, np.int64)),
        )

        labels = _cluster_labels(
            coordinates,
            config,
            cls._NOT_IN_CLUSTER,
            time_column=cls._TIME_COLUMN,
            lat_column=cls._LAT_COLUMN,
            lon_column=cls._LON_COLUMN,
            alt_column=cls._ALTITUDE_COLUMN,
        )
        offsets = np.concatenate(([0], np.cumsum(reports_per_partition.to_numpy(dtype=np.int64))))
        return observations.map_partitions(
            _assign_clusters,
            config,
            labels,
            offsets,
            cls._NOT_IN_CLUSTER,
            cls._CLUSTER_COLUMN,
            cls._IN_CLUSTER_COLUMN,
            meta={**observations.dtypes.to_dict(), cls._CLUSTER_COLUMN: np.int64, cls._IN_CLUSTER_COLUMN: np.bool_},
        )


def _is_meet_condition(frame: "dd.DataFrame | pd.DataFrame", config: ClusteringConfig) -> "dd.Series | pd.Series":
    """
    Whether each report's condition value is strictly above the threshold; NaN never is

    This is the single definition of the row selection, used on the whole table and on each partition: the reports
    gathered for clustering and the reports given labels when writing the output must be exactly the same, in the
    same order.
    """
    return frame[config.condition_column.value] > config.condition_threshold


def _cluster_labels(
    coordinates: "pd.DataFrame",
    config: ClusteringConfig,
    not_in_cluster: int,
    *,
    time_column: str,
    lat_column: str,
    lon_column: str,
    alt_column: str,
) -> "NDArray[np.int64]":
    """
    Cluster ids of the reports in ``coordinates``, in the same order

    The reports are clustered in a canonical order (by time, then latitude, longitude and altitude), as the ids
    st_dbscan assigns depend on the row order. This keeps the ids independent of how the input is split into files.
    ``coordinates`` holds the time, latitude, longitude and altitude of each report, in the columns named by the
    ``*_column`` arguments.
    """
    time = coordinates[time_column].to_numpy()
    lat = coordinates[lat_column].to_numpy()
    lon = coordinates[lon_column].to_numpy()
    alt = coordinates[alt_column].to_numpy()
    # np.lexsort takes its primary key last
    order = np.lexsort((alt, lon, lat, time.view(np.int64)))

    labels = np.empty(len(coordinates), dtype=np.int64)
    labels[order] = st_dbscan(
        time[order],
        lat[order],
        lon[order],
        None if config.max_altitude_delta_ft is None else alt[order],
        n_points=config.n_points,
        not_in_cluster_label=not_in_cluster,
        max_separation_km=config.max_separation_km,
        max_minutes=config.max_minutes,
        max_altitude_delta=config.max_altitude_delta_ft,
    )
    return labels


def _assign_clusters(
    partition: "pd.DataFrame",
    config: ClusteringConfig,
    labels: "NDArray[np.int64]",
    offsets: "NDArray[np.int64]",
    not_in_cluster: int,
    cluster_column: str,
    in_cluster_column: str,
    partition_info: "Mapping[str, object] | None" = None,
) -> "pd.DataFrame":
    """
    Add the ``cluster`` and ``in_cluster`` columns to one partition

    The partition's qualifying reports take ``labels[offsets[number]:offsets[number + 1]]`` in order, where
    ``number`` is the partition number; every other report gets the id ``not_in_cluster``. The result has the columns
    ``cluster_column`` and ``in_cluster_column`` added.
    """
    assert partition_info is not None, "dask passes partition_info to every partition"
    partition_number = partition_info["number"]
    assert isinstance(partition_number, int)

    meets_condition = _is_meet_condition(partition, config).to_numpy()
    partition_labels = labels[offsets[partition_number] : offsets[partition_number + 1]]
    assert np.count_nonzero(meets_condition) == len(partition_labels), "the row selection must be deterministic"

    cluster = np.full(len(partition), not_in_cluster, dtype=np.int64)
    cluster[meets_condition] = partition_labels
    return partition.assign(**{cluster_column: cluster, in_cluster_column: cluster != not_in_cluster})
