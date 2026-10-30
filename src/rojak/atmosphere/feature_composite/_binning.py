"""Bins of the feature composite: the km coordinates that pairs are counted on, and the counting itself.

Deferred to later phases: vertical bins and event layers (phase 2), and the cross-track profile for line features
(phase 5).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from rojak.atmosphere.feature_composite._columns import Column

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


@dataclass(frozen=True, kw_only=True)
class BinSpec:
    """The bins that pairs are counted on, in the km frame of their centre.

    Create it with a named constructor, :meth:`grid`.

    Attributes:
        max_km_extent (float): Distance in km from the centre to the outermost bin centres along each axis.
        spacing_km (float): Distance in km between neighbouring bin centres.

    Raises:
        ValueError: If ``max_km_extent`` or ``spacing_km`` is not positive and finite, or ``max_km_extent`` is not a
            whole multiple of ``spacing_km``.
    """

    max_km_extent: float
    spacing_km: float

    def __post_init__(self) -> None:
        """Validate the extent and the spacing."""
        for name, value in (("max_km_extent", self.max_km_extent), ("spacing_km", self.spacing_km)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, but is {value}")

        n_spacings = self.max_km_extent / self.spacing_km
        # Within float tolerance, e.g. 7.7 / 1.1 == 7.000000000000001
        if not math.isclose(n_spacings, round(n_spacings), rel_tol=1e-9):
            raise ValueError(
                f"max_km_extent ({self.max_km_extent}) must be a whole multiple of spacing_km ({self.spacing_km}). "
                "Change either so that the centre is a bin centre"
            )

    @classmethod
    def grid(cls, max_km_extent: float, spacing_km: float) -> Self:
        """A 2D ``(y_km, x_km)`` grid of square bins centred on the centre, as the gridded composites use.

        Args:
            max_km_extent (float): Distance in km from the centre to the outermost bin centres along each axis.
            spacing_km (float): Side in km of each bin, i.e. the distance between neighbouring bin centres.

        Returns:
            BinSpec: Bins centred on ``-max_km_extent, ..., 0, ..., max_km_extent`` along ``y_km`` and ``x_km``.

        Raises:
            ValueError: If ``max_km_extent`` or ``spacing_km`` is not positive and finite, or ``max_km_extent`` is
                not a whole multiple of ``spacing_km``.

        Examples:
            >>> bins = BinSpec.grid(100, 50)
            >>> bins.dims
            ('y_km', 'x_km')
            >>> bins.coords["x_km"]
            array([-100.,  -50.,    0.,   50.,  100.])
            >>> bins.edges["x_km"]
            array([-125.,  -75.,  -25.,   25.,   75.,  125.])
        """
        return cls(max_km_extent=max_km_extent, spacing_km=spacing_km)

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of the binned dimensions, which are also the pair-table columns that are binned; ``y_km`` first."""
        return (Column.Y_KM.value, Column.X_KM.value)

    @property
    def coords(self) -> "Mapping[str, np.ndarray]":
        """Bin centres in km, by dimension."""
        return {dim: self._axis_coords() for dim in self.dims}

    @property
    def edges(self) -> "Mapping[str, np.ndarray]":
        """Bin edges in km, by dimension: midpoints between the bin centres, extended by half a spacing at each end."""
        return {dim: self._axis_edges() for dim in self.dims}

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of bins along each dimension, in the order of :attr:`dims`."""
        return tuple(self._axis_coords().size for _ in self.dims)

    def _axis_coords(self) -> np.ndarray:
        # Matches the km grid of rojak.atmosphere.compositing, without arange's overshoot for spacings below 1 km
        n_coords = round(2 * self.max_km_extent / self.spacing_km) + 1
        return np.linspace(-self.max_km_extent, self.max_km_extent, n_coords)

    def _axis_edges(self) -> np.ndarray:
        coords = self._axis_coords()
        half_spacing = self.spacing_km / 2
        return np.concatenate([[coords[0] - half_spacing], (coords[:-1] + coords[1:]) / 2, [coords[-1] + half_spacing]])


def count_pairs(pairs: "pd.DataFrame", bins: BinSpec) -> np.ndarray:
    """Number of pairs in each bin.

    Bins are half-open, ``[left, right)``, except the last along each dimension, which also includes its right edge,
    as in :func:`numpy.histogramdd`. Pairs outside every bin are not counted.

    Args:
        pairs (pandas.DataFrame): Pair table with a column for each of ``bins.dims``, e.g. ``Column.Y_KM`` and
            ``Column.X_KM``. It is not modified.
        bins (BinSpec): The bins.

    Returns:
        numpy.ndarray: Integer counts of shape ``bins.shape``, with the axes in the order of ``bins.dims``.
    """
    sample = np.column_stack([pairs[dim].to_numpy(dtype=float) for dim in bins.dims])
    edges = bins.edges
    counts, _ = np.histogramdd(sample, bins=[edges[dim] for dim in bins.dims])
    return counts.astype(np.int64)
