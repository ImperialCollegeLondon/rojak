"""The facade of the feature composite: builds the dask graph and returns lazy per-chunk counts.

:func:`composite_observations_about_features` is the one entry point, so callers never assemble the graph themselves.
There is one task per time chunk of the feature's fields, :func:`_composite_chunk`, which tabulates the feature in its
blocks, pairs the chunk's observations with the centres, and counts the pairs in the bins. Observation-shift controls
run in the same task, on the same centres.

The graph follows the dask rules of the plan (section 0):

- Nothing data-sized is embedded. The blocks of the feature's fields and the partitions of the observations are passed
  by reference, and the latitudes and longitudes once, as ``delayed``.
- The blocks are split with ``to_delayed(optimize_graph=False)``, so tasks shared between the feature's fields (e.g.
  the extrema filter) keep their keys and run once.
- The task receives parameters only: the feature's tabulator, never the feature, which holds the lazy fields.
- The counts, centres and pairs are all built from the chunk tasks' own ``Delayed`` outputs, so computing them together
  runs each chunk task once.
- The only computation while building the graph is :meth:`~rojak.atmosphere.compositing._TargetGrid.group_by_chunk`'s
  count of observations per chunk: once for the observations, and once for every control together. Nothing is
  persisted.

Deferred to later phases: events, EDR summaries, vertical bins and domain counts (phase 2); field-offset controls,
periods and exclusion (phase 3); alignment (phase 5).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, assert_never, overload

import dask.array as da
import dask.dataframe as dd
import numpy as np
import xarray as xr
from dask.dataframe.dispatch import make_meta
from dask.delayed import delayed

from rojak.atmosphere.compositing import _LayoutColumn, _TargetGrid
from rojak.atmosphere.feature_composite._binning import BinSpec, count_pairs
from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._controls import shift_observations, valid_controls, validate_controls
from rojak.atmosphere.feature_composite._feature import ChunkInfo, FeatureGeometry, FeatureSource
from rojak.atmosphere.feature_composite._pairing import PairingStrategy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from dask.delayed import Delayed

    from rojak.atmosphere.feature_composite._controls import ControlDesign
    from rojak.atmosphere.feature_composite._feature import FeatureChunk, FeatureTabulator

# Dimensions of the counts besides those of the bins
_CHUNK: str = "chunk"
_CONTROL: str = Column.CONTROL.value


class _ChunkResult(NamedTuple):
    """Outputs of one chunk task, in the order the ``Delayed`` outputs are unpacked."""

    centres: "pd.DataFrame"
    pairs: "pd.DataFrame"
    feature_n_obs: np.ndarray
    n_centres: np.ndarray
    control_n_obs: np.ndarray
    n_controls: np.ndarray


@dataclass(frozen=True, kw_only=True)
class _ChunkParameters:
    """Everything a chunk task needs besides its data. Parameters only, never a dask collection (rule 5)."""

    tabulator: "FeatureTabulator"
    pairing: PairingStrategy
    bins: BinSpec
    controls: "tuple[ControlDesign, ...]"
    field_time_range: tuple[np.datetime64, np.datetime64]
    lat_column: str
    lon_column: str


# ----------------------------------------------------------------------------------------------------------------------
# The chunk task. Module-level, and looked up when the graph is built, so that each task only embeds its parameters
# ----------------------------------------------------------------------------------------------------------------------


def _composite_chunk(
    blocks: "Mapping[str, np.ndarray]",
    observations: "pd.DataFrame",
    control_observations: "pd.DataFrame | None",
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    first_time_index: int,
    parameters: _ChunkParameters,
) -> _ChunkResult:
    """Tabulate the feature in one time chunk, pair the chunk's observations with its centres and count the pairs.

    Args:
        blocks (Mapping[str, numpy.ndarray]): The chunk's block of each of the feature's chunk inputs, by name.
        observations (pandas.DataFrame): Partition of :meth:`~rojak.atmosphere.compositing._TargetGrid.group_by_chunk`
            holding the observations whose time step is in this chunk, indexed by that time step.
        control_observations (pandas.DataFrame | None): The same for the shifted observations of every control,
            labelled by ``Column.CONTROL``. ``None`` if and only if there are no controls.
        times (numpy.ndarray): Time coordinate of the chunk.
        lats (numpy.ndarray): Latitude coordinate of the field.
        lons (numpy.ndarray): Longitude coordinate of the field.
        first_time_index (int): Index along the whole time axis of the first time step of the chunk.
        parameters (_ChunkParameters): The tabulator, pairing, bins, controls and column names.

    Returns:
        _ChunkResult: The centres, the pairs, the pairs per bin, the number of centres, and per control, the valid
        control pairs per bin and the number of valid controls.
    """
    chunk = ChunkInfo(times=times, first_time_index=first_time_index, lats=lats, lons=lons)
    feature_chunk = parameters.tabulator(blocks, chunk)
    pairs = parameters.pairing.pair(
        _with_time_index(observations),
        feature_chunk,
        chunk,
        lat_column=parameters.lat_column,
        lon_column=parameters.lon_column,
    )
    control_n_obs, n_controls = _count_controls(control_observations, feature_chunk, chunk, parameters)
    return _ChunkResult(
        centres=feature_chunk.centres,
        pairs=pairs,
        feature_n_obs=count_pairs(pairs, parameters.bins),
        n_centres=np.array(len(feature_chunk.centres), dtype=np.int64),
        control_n_obs=control_n_obs,
        n_controls=n_controls,
    )


def _with_time_index(partition: "pd.DataFrame") -> "pd.DataFrame":
    """The observations of a partition of ``group_by_chunk``, with ``Column.TIME_INDEX`` and no layout columns."""
    # group_by_chunk indexes the partitions by the time index of each observation
    located = partition.reset_index()
    with_time_index = located.assign(**{Column.TIME_INDEX: located[_LayoutColumn.GROUP_INDEX].to_numpy(dtype=np.int64)})
    return with_time_index.drop(columns=list(_LayoutColumn))


def _count_controls(
    control_observations: "pd.DataFrame | None",
    feature_chunk: "FeatureChunk",
    chunk: ChunkInfo,
    parameters: _ChunkParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid control pairs per bin and number of valid controls, per control in the order of ``parameters.controls``.

    A control pair counts only if its ``(centre, control)`` is valid, see
    :func:`~rojak.atmosphere.feature_composite._controls.valid_controls`.
    """
    bins = parameters.bins
    if control_observations is None:
        return np.zeros((0, *bins.shape), dtype=np.int64), np.zeros(0, dtype=np.int64)

    valid = valid_controls(feature_chunk.centres, parameters.controls, field_time_range=parameters.field_time_range)
    pairs = parameters.pairing.pair(
        _with_time_index(control_observations),
        feature_chunk,
        chunk,
        lat_column=parameters.lat_column,
        lon_column=parameters.lon_column,
    )

    n_obs: list[np.ndarray] = []
    n_controls: list[int] = []
    for design in parameters.controls:
        valid_centre_ids = valid.loc[valid[Column.CONTROL] == design.label, Column.CENTRE_ID]
        is_kept = (pairs[Column.CONTROL] == design.label) & pairs[Column.CENTRE_ID].isin(valid_centre_ids)
        n_obs.append(count_pairs(pairs[is_kept], bins))

        n_controls.append(len(valid_centre_ids))
    return np.stack(n_obs), np.array(n_controls, dtype=np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------------------------------------------------------


@overload
def composite_observations_about_features(
    feature: FeatureSource,
    observations: dd.DataFrame,
    *,
    pairing: PairingStrategy,
    bins: BinSpec,
    controls: "Sequence[ControlDesign]" = (),
    return_pairs: Literal[False] = False,
    time_column: str = "time",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    time_tolerance: np.timedelta64 | None = None,
) -> xr.Dataset: ...


@overload
def composite_observations_about_features(
    feature: FeatureSource,
    observations: dd.DataFrame,
    *,
    pairing: PairingStrategy,
    bins: BinSpec,
    controls: "Sequence[ControlDesign]" = (),
    return_pairs: Literal[True],
    time_column: str = "time",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    time_tolerance: np.timedelta64 | None = None,
) -> tuple[xr.Dataset, dd.DataFrame, dd.DataFrame]: ...


def composite_observations_about_features(
    feature: FeatureSource,
    observations: dd.DataFrame,
    *,
    pairing: PairingStrategy,
    bins: BinSpec,
    controls: "Sequence[ControlDesign]" = (),
    return_pairs: bool = False,
    time_column: str = "time",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
    time_tolerance: np.timedelta64 | None = None,
) -> xr.Dataset | tuple[xr.Dataset, dd.DataFrame, dd.DataFrame]:
    """Lazily count the observations about each centre of a feature, and about its controls, per time chunk.

    Each observation is matched to the nearest time step of the feature's fields, within ``time_tolerance``, and paired
    with the centres at that time step by ``pairing``. The pairs are counted in ``bins``, in each centre's km frame.
    For each control, the observations made ``observation_offset`` away from each centre's time are paired with the
    same centre in the same way. A control is only counted for a centre if its observation time is within the time
    range of the fields.

    Building the graph computes the number of observations per time chunk, once for the observations and once for all
    the controls together, so persist ``observations`` beforehand if it is expensive to compute. Nothing else is
    computed: with ``return_pairs=True``, the tabulator and pairing also run once, eagerly, on an empty chunk to learn
    the columns and dtypes of the returned tables (see
    :class:`~rojak.atmosphere.feature_composite._feature.FeatureTabulator`), which touches no real data. Compute the
    result, then derive statistics from it with
    :func:`~rojak.atmosphere.feature_composite._statistics.impact_statistics`.

    Args:
        feature (FeatureSource): The feature to composite about, e.g. lows.
        observations (dask.dataframe.DataFrame): Observations with a datetime column ``time_column``, and latitudes and
            longitudes in degrees. Naive times are assumed to be in UTC. Longitudes may be in either ``[-180, 180)`` or
            ``[0, 360)``, independently of the feature's fields.
        pairing (PairingStrategy): How observations are matched with centres.
        bins (BinSpec): The bins the pairs are counted in.
        controls (Sequence[ControlDesign]): The controls that estimate the counterfactual, each with a distinct label.
            Empty for none.
        return_pairs (bool): If ``True``, also return the centres and the pairs as lazy dataframes.
        time_column (str): Name of the column of ``observations`` holding the time.
        lat_column (str): Name of the column of ``observations`` holding the latitude.
        lon_column (str): Name of the column of ``observations`` holding the longitude.
        time_tolerance (numpy.timedelta64 | None): Observations further than this from the nearest time step of the
            fields are dropped. ``None`` is half the median time step of the fields.

    Returns:
        xarray.Dataset | tuple[xarray.Dataset, dask.dataframe.DataFrame, dask.dataframe.DataFrame]: The lazy counts,
        with the variables

        - ``feature_n_obs`` ``(chunk, *bins.dims)``: pairs per bin;
        - ``n_centres`` ``(chunk)``: number of centres. ``n_features``, the number of distinct features (section 2),
          is deferred to a line feature (phase 6 or later): it always equals ``n_centres`` while every feature is
          ``POINT``, so it is omitted rather than duplicated;
        - with controls, ``control_n_obs`` ``(chunk, control, *bins.dims)``: valid control pairs per bin, and
          ``n_controls`` ``(chunk, control)``: number of valid controls, with the ``control`` coordinate holding the
          labels in the order given;

        and the attributes ``feature``, ``geometry``, ``pairing``, ``bins``, ``controls`` (the labels) and
        ``normalisation_unit`` (what one centre is). With ``return_pairs=True``, a tuple of the counts, every centre
        (including those without pairs), and every pair of an observation with a centre (not those of the controls).
        All three share the chunk tasks, so compute them together, e.g. with :func:`dask.compute`, to run each once.

    Raises:
        TypeError: If ``feature`` is not a ``FeatureSource``, ``pairing`` is not a ``PairingStrategy`` or ``bins`` is
            not a ``BinSpec``.
        ValueError: If ``pairing`` cannot handle ``feature``, two controls share a label, or no observation (or no
            shifted observation of the controls) is within the time range of the feature's fields.
    """
    _validate(feature, pairing, bins, controls)
    inputs = feature.chunk_inputs()
    grid = _grid_of(inputs)
    if time_tolerance is None:
        time_tolerance = _half_median_time_step(grid.time_coord)

    by_chunk, _ = grid.group_by_chunk(
        grid.locate_centres(observations, time_column=time_column, vert_column=None, time_tolerance=time_tolerance)
    )
    control_by_chunk: dd.DataFrame | None = None
    if controls:
        # Every control in one frame, so that they are grouped by chunk with a single count (rule 7)
        shifted = _shifted_for_controls(observations, controls, time_column=time_column)
        control_by_chunk, _ = grid.group_by_chunk(
            grid.locate_centres(shifted, time_column=time_column, vert_column=None, time_tolerance=time_tolerance)
        )

    parameters = _ChunkParameters(
        tabulator=feature.tabulator(),
        pairing=pairing,
        bins=bins,
        controls=tuple(controls),
        field_time_range=(grid.time_coord.min(), grid.time_coord.max()),
        lat_column=lat_column,
        lon_column=lon_column,
    )
    tasks = _chunk_tasks(inputs, by_chunk, control_by_chunk, grid, parameters)
    # In the order of the fields of _ChunkResult
    centres, pairs, feature_n_obs, n_centres, control_n_obs, n_controls = zip(*tasks, strict=True)

    data_vars: dict[str, tuple[tuple[str, ...], da.Array]] = {
        "feature_n_obs": ((_CHUNK, *bins.dims), _stack(feature_n_obs, bins.shape)),
        "n_centres": ((_CHUNK,), _stack(n_centres, ())),
    }
    coords: dict[str, np.ndarray] = dict(bins.coords)
    if controls:
        n_designs = len(controls)
        data_vars["control_n_obs"] = (
            (_CHUNK, _CONTROL, *bins.dims),
            _stack(control_n_obs, (n_designs, *bins.shape)),
        )
        data_vars["n_controls"] = ((_CHUNK, _CONTROL), _stack(n_controls, (n_designs,)))
        coords[_CONTROL] = np.array([design.label for design in controls])
    counts = xr.Dataset(data_vars, coords=coords, attrs=_attrs(feature, pairing, bins, controls))
    if not return_pairs:
        return counts

    # Evaluate the chunk task on empty inputs for the columns and dtypes of the tables, rather than computing a chunk
    meta = _composite_chunk(
        {name: np.empty((0, *array.shape[1:]), dtype=array.dtype) for name, array in inputs.items()},
        make_meta(by_chunk),
        make_meta(control_by_chunk) if control_by_chunk is not None else None,
        grid.time_coord[:0],
        grid.lats,
        grid.lons,
        first_time_index=0,
        parameters=parameters,
    )
    return (
        counts,
        dd.from_delayed(list(centres), meta=meta.centres, verify_meta=False),
        dd.from_delayed(list(pairs), meta=meta.pairs, verify_meta=False),
    )


def _validate(
    feature: FeatureSource, pairing: PairingStrategy, bins: BinSpec, controls: "Sequence[ControlDesign]"
) -> None:
    """Check the arguments of :func:`composite_observations_about_features` before anything is built."""
    # The annotations promise these types, but callers outside pyright can pass anything
    if not isinstance(feature, FeatureSource):
        raise TypeError(
            f"feature must be a FeatureSource, e.g. ExtremaPointFeature, but is {type(feature).__name__}. To composite "
            "about a new kind of feature, subclass FeatureSource from rojak.atmosphere.feature_composite.extension"
        )
    if not isinstance(pairing, PairingStrategy):
        raise TypeError(
            "pairing must be a PairingStrategy, e.g. LabelledRegionPairing() or DistancePairing(500), but is "
            f"{type(pairing).__name__}"
        )
    if not isinstance(bins, BinSpec):
        raise TypeError(f"bins must be a BinSpec, e.g. BinSpec.grid(2000, 50), but is {type(bins).__name__}")
    pairing.check_compatible(feature)
    validate_controls(controls)


def _grid_of(inputs: "Mapping[str, xr.DataArray]") -> _TargetGrid:
    """The grid of the feature's chunk inputs, which all have dimensions ``(time, lat, lon)`` and the same chunks."""
    template = next(iter(inputs.values()))
    time_dim, lat_dim, lon_dim = (str(dim) for dim in template.dims)
    return _TargetGrid.from_template(
        template,
        time_dim_name=time_dim,
        lat_dim_name=lat_dim,
        lon_dim_name=lon_dim,
        vert_dim_name=None,
        group_dim_name=time_dim,
    )


def _half_median_time_step(time_coord: np.ndarray) -> np.timedelta64 | None:
    """Half the median time step, so each observation is matched to at most one time step. ``None`` for one step."""
    if time_coord.size <= 1:
        return None
    steps_ns = np.diff(time_coord.astype("datetime64[ns]")).astype(np.int64)
    return np.timedelta64(round(float(np.median(steps_ns)) / 2), "ns")


def _shifted_for_controls(
    observations: dd.DataFrame, controls: "Sequence[ControlDesign]", *, time_column: str
) -> dd.DataFrame:
    """The observations of every control, moved back to the centre times and labelled by ``Column.CONTROL``."""
    return dd.concat(
        [
            observations.map_partitions(shift_observations, design, time_column=time_column).assign(
                **{Column.CONTROL: design.label}
            )
            for design in controls
        ]
    )


def _chunk_tasks(
    inputs: "Mapping[str, xr.DataArray]",
    by_chunk: dd.DataFrame,
    control_by_chunk: dd.DataFrame | None,
    grid: _TargetGrid,
    parameters: _ChunkParameters,
) -> "list[Delayed]":
    """One chunk task per time chunk, each with the outputs of :class:`_ChunkResult`.

    The blocks and partitions are passed by reference and the latitudes and longitudes once. Only the time coordinate
    of each chunk is embedded, which is the time coordinate once in total.
    """
    chunk_task = delayed(_composite_chunk, nout=len(_ChunkResult._fields), pure=True)
    lats = delayed(grid.lats, pure=True)
    lons = delayed(grid.lons, pure=True)

    edges = grid.group_chunk_edges.tolist()
    blocks = _blocks_by_chunk(inputs)
    partitions = by_chunk.to_delayed()
    control_partitions = control_by_chunk.to_delayed() if control_by_chunk is not None else [None] * len(blocks)
    return [
        chunk_task(
            chunk_blocks,
            partition,
            control_partition,
            grid.time_coord[start:stop],
            lats,
            lons,
            first_time_index=start,
            parameters=parameters,
        )
        for chunk_blocks, partition, control_partition, start, stop in zip(
            blocks, partitions, control_partitions, edges[:-1], edges[1:], strict=True
        )
    ]


def _blocks_by_chunk(inputs: "Mapping[str, xr.DataArray]") -> "list[dict[str, Delayed]]":
    """The block of each chunk input, by name, for each time chunk."""
    # Without optimising, the blocks keep the keys of the existing chunks. Otherwise each input would be optimised
    # separately, which renames the tasks they share (e.g. the extrema filter) and runs them once per input
    names = inputs.keys()
    delayed_lists = [array.data.to_delayed(optimize_graph=False).ravel().tolist() for array in inputs.values()]
    return [dict(zip(names, chunks, strict=True)) for chunks in zip(*delayed_lists, strict=True)]


def _stack(parts: "Sequence[Delayed]", shape: tuple[int, ...]) -> da.Array:
    """Stack one integer output of shape ``shape`` of every chunk task along a new leading ``chunk`` axis."""
    return da.stack([da.from_delayed(part, shape=shape, dtype=np.int64) for part in parts])


def _attrs(
    feature: FeatureSource, pairing: PairingStrategy, bins: BinSpec, controls: "Sequence[ControlDesign]"
) -> dict[str, str | tuple[str, ...]]:
    """What the counts were built from, including what one centre is, so the normalisation is explicit."""
    return {
        "feature": type(feature).__name__,
        "geometry": feature.geometry.value,
        "pairing": repr(pairing),
        "bins": repr(bins),
        "controls": tuple(design.label for design in controls),
        "normalisation_unit": _normalisation_unit(feature.geometry),
    }


def _normalisation_unit(geometry: FeatureGeometry) -> str:
    """What one centre is: one instance of a point feature, or one point sampled along the axis of a line feature."""
    match geometry:
        case FeatureGeometry.POINT:
            return "feature instance"
        case FeatureGeometry.LINE:
            return "axis point"
        case _ as unreachable:
            assert_never(unreachable)
