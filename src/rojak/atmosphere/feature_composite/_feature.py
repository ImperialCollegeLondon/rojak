"""The feature interface: what a feature must provide for observations to be composited about it.

A feature has two halves, so that no dask collection is ever sent to a task:

- :class:`FeatureSource`, the **graph side**, holds the lazy, time-chunked fields that the feature is found in. It is
  used only while the dask graph is built.
- :class:`FeatureTabulator`, the **task side**, is a small picklable object that holds parameters only. It runs inside
  each chunk task on the in-memory blocks of those fields and tabulates the feature's centres as a
  :class:`FeatureChunk`.

Whether a feature is a point (a low) or a line (a jet core) is data, :class:`FeatureGeometry`, not a class hierarchy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import pandas as pd
    import xarray as xr


class FeatureGeometry(StrEnum):
    """Shape of one instance of a feature."""

    POINT = "point"
    """One point defines each instance, e.g. lows and highs. Each instance has exactly one centre."""
    LINE = "line"
    """Each instance extends along an axis, e.g. a jet core, and is represented by centres sampled along the axis."""


@dataclass(frozen=True)
class ChunkInfo:
    """Coordinates of the time chunk that a :class:`FeatureTabulator` is called on.

    The coordinate arrays are passed by reference and are not copied.

    Attributes:
        times (numpy.ndarray): Time coordinate of the chunk, i.e. axis 0 of each block.
        first_time_index (int): Index along the whole time axis of the field of the first time step of the chunk. Add
            it to a local time index to get the chunk-absolute ``Column.TIME_INDEX``.
        lats (numpy.ndarray): Latitude coordinate of the field in degrees, i.e. axis 1 of each block.
        lons (numpy.ndarray): Longitude coordinate of the field in degrees, i.e. axis 2 of each block.
    """

    times: "np.ndarray"
    first_time_index: int
    lats: "np.ndarray"
    lons: "np.ndarray"


@dataclass(frozen=True)
class FeatureChunk:
    """The centres of a feature found in one time chunk, as returned by a :class:`FeatureTabulator`.

    Attributes:
        centres (pandas.DataFrame): One row per centre, with the columns named by
            :class:`~rojak.atmosphere.feature_composite._columns.Column`: ``CENTRE_ID`` (unique across all chunks, e.g.
            encoded from the time and grid indices), ``FEATURE_ID`` (equal to ``CENTRE_ID`` for a ``POINT`` feature;
            shared by the centres along one instance of a ``LINE`` feature), ``TIME_INDEX`` (along the whole time axis,
            see :attr:`ChunkInfo.first_time_index`), ``CENTRE_TIME``, ``CENTRE_LAT`` and ``CENTRE_LON``. A feature found
            on the grid also fills ``CENTRE_LAT_INDEX`` and ``CENTRE_LON_INDEX``, and ``CENTRE_VALUE`` where the
            feature has a value at its centre.
        labels (numpy.ndarray | None): Integer array with the shape ``(time, lat, lon)`` of the chunk's blocks, holding
            the ``CENTRE_ID`` of the centre that owns each grid cell, or 0 where no centre does. ``None`` if the
            feature does not label regions, see :attr:`FeatureSource.has_labels`.
    """

    centres: "pd.DataFrame"
    labels: "np.ndarray | None"


class FeatureTabulator(ABC):
    """Task side of a feature: tabulates its centres from the in-memory blocks of one time chunk.

    It is sent to every chunk task, so it must hold **parameters only**, never a dask collection or a data-sized array.
    Implement it as a frozen dataclass: it then pickles cheaply and dask tokenises it deterministically from its
    fields.

    It must also handle a chunk with no time steps, returning empty tables with the same columns and dtypes: the
    pipeline calls it once on such a chunk while building the graph, to find the columns and dtypes of the centre and
    pair tables without computing anything.
    """

    @abstractmethod
    def __call__(self, blocks: "Mapping[str, np.ndarray]", chunk: ChunkInfo) -> FeatureChunk:
        """Tabulate the centres of the feature in one time chunk.

        Args:
            blocks (Mapping[str, numpy.ndarray]): The chunk's in-memory block of each array named by
                :meth:`FeatureSource.chunk_inputs`, under the same name, each of shape ``(time, lat, lon)``.
            chunk (ChunkInfo): Coordinates of the chunk.

        Returns:
            FeatureChunk: The centres found in the chunk and, if the feature labels regions, the labels.
        """


class FeatureSource(ABC):
    """Graph side of a feature: holds the lazy fields it is found in and creates its :class:`FeatureTabulator`.

    Implement it as a frozen dataclass whose fields are the lazy fields and the parameters. Declare the geometry with a
    one-line property, which is not a dataclass field::

        @property
        def geometry(self) -> FeatureGeometry:
            return FeatureGeometry.POINT

    Use a property rather than a class attribute such as ``geometry = FeatureGeometry.POINT``: pyright rejects a class
    attribute that overrides a property.
    """

    @property
    @abstractmethod
    def geometry(self) -> FeatureGeometry:
        """Whether one instance of the feature is a point or a line."""

    @abstractmethod
    def chunk_inputs(self) -> "Mapping[str, xr.DataArray]":
        """The lazy fields the tabulator needs, by name.

        Returns:
            Mapping[str, xarray.DataArray]: Time-chunked arrays with dimensions ``(time, lat, lon)`` on the same grid
            and chunks. The tabulator receives their blocks under the same names.
        """

    @abstractmethod
    def tabulator(self) -> FeatureTabulator:
        """The picklable, parameters-only object that tabulates the centres inside each chunk task."""

    @property
    def has_labels(self) -> bool:
        """Whether the tabulator fills :attr:`FeatureChunk.labels`.

        Declared statically so that a pairing that needs labels is rejected when the graph is built, not inside a
        task. Override it to return ``True`` in a feature whose tabulator labels regions.
        """
        return False
