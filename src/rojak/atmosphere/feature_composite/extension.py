"""Interfaces for adding a new kind of feature or pairing to :mod:`rojak.atmosphere.feature_composite`.

Users running composites with the existing features do not need this module.

**Adding a feature.** A feature has two halves, so that no dask collection is ever sent to a task:

1. A :class:`FeatureSource`, the graph side, holds the lazy, time-chunked fields the feature is found in. Its
   :meth:`~FeatureSource.chunk_inputs` returns them by name, each with dimensions ``(time, lat, lon)`` on the same grid
   and chunks, and its :meth:`~FeatureSource.tabulator` returns the task side.
2. A :class:`FeatureTabulator`, the task side, runs in each chunk task on the in-memory blocks of those fields and
   returns the chunk's centres as a :class:`FeatureChunk`, with the coordinates of the chunk given as a
   :class:`ChunkInfo`. It is sent to every task, so it must hold parameters only, never a dask collection or a
   data-sized array.

Implement both as frozen dataclasses. Declare the :class:`FeatureGeometry` with a one-line ``geometry`` property, and
override ``has_labels`` to return ``True`` if the tabulator labels the region each centre owns::

    @dataclass(frozen=True)
    class TableTabulator(FeatureTabulator):
        centres: pd.DataFrame  # Small: every centre, with a Column.TIME_INDEX

        def __call__(self, blocks: Mapping[str, np.ndarray], chunk: ChunkInfo) -> FeatureChunk:
            stop = chunk.first_time_index + chunk.times.size
            in_chunk = self.centres[self.centres[Column.TIME_INDEX].between(chunk.first_time_index, stop - 1)]
            return FeatureChunk(centres=in_chunk.reset_index(drop=True), labels=None)


    @dataclass(frozen=True, eq=False)
    class TableFeature(FeatureSource):
        field: xr.DataArray  # Only gives the time-chunked grid
        centres: pd.DataFrame

        @property
        def geometry(self) -> FeatureGeometry:
            return FeatureGeometry.POINT

        def chunk_inputs(self) -> Mapping[str, xr.DataArray]:
            return {"values": self.field}

        def tabulator(self) -> FeatureTabulator:
            return TableTabulator(self.centres)

**Adding a pairing.** Subclass :class:`PairingStrategy`, implement ``pair`` so that it returns the same pair table as
the existing strategies, and override ``check_compatible`` to reject features it cannot handle.
"""

from rojak.atmosphere.feature_composite._feature import (
    ChunkInfo,
    FeatureChunk,
    FeatureGeometry,
    FeatureSource,
    FeatureTabulator,
)
from rojak.atmosphere.feature_composite._pairing import PairingStrategy

__all__ = [
    "ChunkInfo",
    "FeatureChunk",
    "FeatureGeometry",
    "FeatureSource",
    "FeatureTabulator",
    "PairingStrategy",
]
