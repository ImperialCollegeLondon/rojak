"""Pairing strategies: which observations are matched with which centres of a feature.

Every strategy returns the same pair table, so the rest of the pipeline does not depend on which one was used: one row
per pair of an observation and a centre at the same time step, with every column of the observation, the centre's
``CENTRE_ID``, ``FEATURE_ID``, ``CENTRE_TIME``, ``CENTRE_LAT``, ``CENTRE_LON`` and ``CENTRE_VALUE``, and the
observation's position ``X_KM`` and ``Y_KM`` in the centre's km frame (:func:`~._frame.project_pairs`, D16).

A strategy is a frozen dataclass holding parameters only, so it pickles cheaply into each chunk task. It checks that it
can handle a feature with :meth:`PairingStrategy.check_compatible`, which the entry point calls before building the
graph (D18).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np

from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._feature import FeatureGeometry
from rojak.atmosphere.feature_composite._frame import project_pairs
from rojak.core.geometric import haversine_distance
from rojak.core.indexing import nearest_index

if TYPE_CHECKING:
    import pandas as pd

    from rojak.atmosphere.feature_composite._feature import ChunkInfo, FeatureChunk, FeatureSource

# The centre's columns in every pair table, in order. CENTRE_VALUE is optional in FeatureChunk.centres and added apart
_REQUIRED_CENTRE_COLUMNS: list[str] = [
    Column.CENTRE_ID,
    Column.FEATURE_ID,
    Column.CENTRE_TIME,
    Column.CENTRE_LAT,
    Column.CENTRE_LON,
]
_SHAPES: tuple[str, ...] = ("circle", "square")
# Haversine distance is on a sphere and the projection on the ellipsoid, so the square's pre-filter has a margin
_SQUARE_PREFILTER_FACTOR: float = math.sqrt(2) * 1.01


class PairingStrategy(ABC):
    """How the observations of a time chunk are matched with the centres of a feature in that chunk.

    Implement it as a frozen dataclass holding parameters only: it is sent to every chunk task. Override
    :meth:`check_compatible` to reject features that the strategy cannot handle.
    """

    @abstractmethod
    def pair(
        self,
        observations: "pd.DataFrame",
        feature_chunk: "FeatureChunk",
        chunk: "ChunkInfo",
        *,
        lat_column: str,
        lon_column: str,
    ) -> "pd.DataFrame":
        """Pair the observations of a time chunk with the centres of the feature in that chunk.

        An observation is only ever paired with a centre at the same time step, i.e. with the same
        ``Column.TIME_INDEX``. Implementations end by projecting the pairs with :func:`~._frame.project_pairs` (D16).

        Args:
            observations (pandas.DataFrame): The chunk's observations, with ``Column.TIME_INDEX`` (along the whole time
                axis of the field) of the field time step each was matched to, and their latitude and longitude in
                degrees. Longitudes may be in either ``[-180, 180)`` or ``[0, 360)``, independently of the field's.
            feature_chunk (FeatureChunk): The chunk's centres and, if the feature labels regions, its labels.
            chunk (ChunkInfo): Coordinates of the chunk.
            lat_column (str): Name of the column of ``observations`` holding the latitude.
            lon_column (str): Name of the column of ``observations`` holding the longitude.

        Returns:
            pandas.DataFrame: One row per pair, with every column of ``observations``, ``Column.CENTRE_ID``,
            ``Column.FEATURE_ID``, ``Column.CENTRE_TIME``, ``Column.CENTRE_LAT``, ``Column.CENTRE_LON``,
            ``Column.CENTRE_VALUE``, ``Column.X_KM`` and ``Column.Y_KM``, and no other column, even when empty. The
            inputs are not modified.
        """

    def check_compatible(self, feature: "FeatureSource") -> None:
        """Raise if this strategy cannot pair observations with ``feature``. The base accepts any feature.

        Args:
            feature (FeatureSource): The feature to be composited about.

        Raises:
            ValueError: In subclasses, if the strategy cannot handle the feature.
        """
        del feature  # Accepts any feature


@dataclass(frozen=True)
class LabelledRegionPairing(PairingStrategy):
    """Pair each observation with the centre that owns the grid cell nearest to it, if any.

    The owner is read from the feature's labels at the observation's time step, so each observation is paired with at
    most one centre. Observations more than half a grid cell beyond the outermost grid points are not paired.
    Observation longitudes are converted to the grid's convention for the lookup only; the pair table keeps the
    original values.

    Compatible with any feature that labels regions (:attr:`~._feature.FeatureSource.has_labels`), whatever its
    geometry.
    """

    def pair(
        self,
        observations: "pd.DataFrame",
        feature_chunk: "FeatureChunk",
        chunk: "ChunkInfo",
        *,
        lat_column: str,
        lon_column: str,
    ) -> "pd.DataFrame":
        """Pair each observation with the centre owning its nearest grid cell; see :meth:`PairingStrategy.pair`."""

        labels = feature_chunk.labels
        assert labels is not None, "check_compatible() rejects features without labels"

        obs_lats: np.ndarray = observations[lat_column].to_numpy(dtype=float)
        obs_lons = _to_grid_longitude(observations[lon_column].to_numpy(dtype=float), chunk.lons)
        is_on_grid = _is_within_cells(chunk.lats, obs_lats) & _is_within_cells(chunk.lons, obs_lons)

        on_grid: pd.DataFrame = observations[is_on_grid]
        owners = labels[
            on_grid[Column.TIME_INDEX].to_numpy(dtype=np.int64) - chunk.first_time_index,
            nearest_index(chunk.lats, obs_lats[is_on_grid]),
            nearest_index(chunk.lons, obs_lons[is_on_grid]),
        ]
        owned: pd.DataFrame = on_grid.assign(**{Column.CENTRE_ID: owners})[owners != 0]

        pairs: pd.DataFrame = owned.merge(
            _pair_centre_columns(feature_chunk.centres), on=[Column.TIME_INDEX, Column.CENTRE_ID], how="inner"
        )
        return project_pairs(pairs, lat_column=lat_column, lon_column=lon_column)

    def check_compatible(self, feature: "FeatureSource") -> None:
        """Raise if ``feature`` does not label regions.

        Args:
            feature (FeatureSource): The feature to be composited about.

        Raises:
            ValueError: If ``feature.has_labels`` is ``False``.
        """
        if not feature.has_labels:
            raise ValueError(
                f"LabelledRegionPairing needs a feature that labels regions, but {type(feature).__name__} does not "
                "(has_labels is False). Use DistancePairing instead"
            )


@dataclass(frozen=True)
class DistancePairing(PairingStrategy):
    """Pair each observation with every centre at the same time step within ``max_distance_km`` of it.

    With ``shape="circle"``, the great-circle (haversine) distance to the centre is at most ``max_distance_km``. With
    ``shape="square"``, ``|x_km|`` and ``|y_km|`` in the centre's km frame are both at most ``max_distance_km``, the
    support of the gridded composites. An observation can be paired with several centres.

    Compatible with ``POINT`` features only: neighbouring centres along a ``LINE`` feature would count the same
    observation many times.

    Attributes:
        max_distance_km (float): Largest distance in km, or half the side of the square. Positive.
        shape (Literal["circle", "square"]): Whether the distance is great-circle or per axis of the km frame.

    Raises:
        ValueError: If ``max_distance_km`` is not positive, or ``shape`` is neither ``"circle"`` nor ``"square"``.
    """

    max_distance_km: float
    _: KW_ONLY
    shape: Literal["circle", "square"] = "circle"

    def __post_init__(self) -> None:
        """Validate the distance and the shape."""
        if not self.max_distance_km > 0:  # Written this way to reject NaN too
            raise ValueError(f"max_distance_km must be positive, but is {self.max_distance_km}")
        if self.shape not in _SHAPES:
            raise ValueError(f"shape must be one of {_SHAPES}, but is {self.shape!r}")

    def pair(
        self,
        observations: "pd.DataFrame",
        feature_chunk: "FeatureChunk",
        chunk: "ChunkInfo",  # noqa: ARG002 - part of the interface; distance pairing needs no grid
        *,
        lat_column: str,
        lon_column: str,
    ) -> "pd.DataFrame":
        """Pair each observation with every centre within the distance; see :meth:`PairingStrategy.pair`."""
        candidates = observations.merge(_pair_centre_columns(feature_chunk.centres), on=Column.TIME_INDEX, how="inner")
        distance_km = (
            haversine_distance(
                candidates[lon_column].to_numpy(dtype=float),
                candidates[lat_column].to_numpy(dtype=float),
                candidates[Column.CENTRE_LON].to_numpy(dtype=float),
                candidates[Column.CENTRE_LAT].to_numpy(dtype=float),
            )
            / 1000
        )
        match self.shape:
            case "circle":
                return project_pairs(
                    candidates[distance_km <= self.max_distance_km], lat_column=lat_column, lon_column=lon_column
                )
            case "square":
                # Only project the candidates that can be within the corners of the square
                within_corners = distance_km <= self.max_distance_km * _SQUARE_PREFILTER_FACTOR
                pairs = project_pairs(candidates[within_corners], lat_column=lat_column, lon_column=lon_column)
                return pairs[
                    (pairs[Column.X_KM].abs() <= self.max_distance_km)
                    & (pairs[Column.Y_KM].abs() <= self.max_distance_km)
                ]
            case _ as unreachable:
                assert_never(unreachable)

    def check_compatible(self, feature: "FeatureSource") -> None:
        """Raise if ``feature`` is a ``LINE`` feature.

        Args:
            feature (FeatureSource): The feature to be composited about.

        Raises:
            ValueError: If ``feature.geometry`` is ``FeatureGeometry.LINE``.
        """
        if feature.geometry == FeatureGeometry.LINE:
            raise ValueError(
                f"DistancePairing cannot pair with the LINE feature {type(feature).__name__}: its neighbouring centres "
                "would count the same observation many times. Use NearestCentrePairing (planned for phase 5) instead"
            )


def _pair_centre_columns(centres: "pd.DataFrame") -> "pd.DataFrame":
    """The columns of the centres that every pair table carries, plus ``Column.TIME_INDEX`` to match on.

    ``Column.CENTRE_VALUE`` is filled with NaN for a feature whose centres have no value.
    """
    return centres[[Column.TIME_INDEX, *_REQUIRED_CENTRE_COLUMNS]].assign(
        **{Column.CENTRE_VALUE: centres.get(Column.CENTRE_VALUE, np.nan)}
    )


def _to_grid_longitude(lons: np.ndarray, grid_lons: np.ndarray) -> np.ndarray:
    """Convert ``lons`` to the convention of ``grid_lons``, i.e. ``[0, 360)`` or ``[-180, 180)``."""
    return lons % 360 if grid_lons.max() > 180 else (lons + 180) % 360 - 180  # noqa: PLR2004


def _is_within_cells(coordinate: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Whether ``values`` lie within the cells of the regularly spaced ``coordinate``.

    That is, no more than half a cell beyond its outermost points.
    """
    half_spacing = np.abs(np.diff(coordinate)).min() / 2 if coordinate.size > 1 else 0
    return (values >= coordinate.min() - half_spacing) & (values <= coordinate.max() + half_spacing)
