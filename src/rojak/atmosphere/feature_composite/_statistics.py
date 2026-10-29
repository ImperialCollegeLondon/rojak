"""Impact statistics from the per-chunk counts of the feature composite, and a generic chunk bootstrap.

The pipeline returns lazy per-chunk counts. Statistics are derived from them eagerly, after compute: summed over chunks
they are small arrays of bins, and the dask optimiser fails on the fully lazy version (decision D8).

Deferred to phase 4: Cohen's h, the attributable fractions, excess events per centre and the standardised anomaly.
"""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from rojak.turbulence.metrics import ContingencyTable, odds_ratio_from_table, relative_risk_from_table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Variables of the per-chunk counts (section 2 of the plan) read here
_FEATURE_PREFIX: str = "feature_n_"
_CONTROL_PREFIX: str = "control_n_"
_OBSERVATIONS: str = "obs"
_N_CENTRES: str = "n_centres"
_N_CONTROLS: str = "n_controls"
_CHUNK: str = "chunk"
_HALDANE_CORRECTION: float = 0.5


def impact_statistics(counts: xr.Dataset, *, min_count: float = 10, haldane: bool = False) -> xr.Dataset:
    """How the feature changes traffic and the probability of each event, relative to the controls.

    The counts are summed over ``chunk`` first; every other dimension, including ``control``, is kept. Each event
    ``e`` with a pair of variables ``feature_n_<e>`` and ``control_n_<e>`` gets its own measures, from the 2x2 table

    - ``n_11 = E_feat``: events about the feature,
    - ``n_10 = N_feat - E_feat``: observations about the feature that are not events,
    - ``n_01 = E_ctl``: events in the control,
    - ``n_00 = N_ctl - E_ctl``: observations in the control that are not events,

    where ``N`` is ``feature_n_obs`` or ``control_n_obs`` and ``E`` is ``feature_n_<e>`` or ``control_n_<e>``.

    A zero in a denominator gives NaN, never infinity, while a zero in a numerator is a real 0.

    Args:
        counts (xarray.Dataset): Per-chunk counts as returned by the pipeline, with ``feature_n_obs``,
            ``control_n_obs``, ``n_centres``, ``n_controls`` and any event pairs. Computed here if lazy.
        min_count (float): Bins where the observations summed over chunks are fewer than this, about the feature or in
            the control, are NaN: in ``p_feature_<e>`` or ``p_control_<e>`` for that side, and in the ratios for
            either side. ``traffic_per_centre`` and ``traffic_per_control`` are not masked.
        haldane (bool): If ``True``, add 0.5 to every cell of the table before the odds ratio (the Haldane
            correction), so that it is finite where a cell is zero. The other measures are not corrected.

    Returns:
        xarray.Dataset: In memory, with:

        - ``traffic_per_centre``: ``N_feat / n_centres``, observations per bin per centre.
        - ``traffic_per_control``: ``N_ctl / n_controls``, observations per bin per control.
        - ``traffic_ratio``: their ratio. Above 1 means more traffic about the feature than without it.
        - ``p_feature_<e>`` / ``p_control_<e>``: ``E / N``, probability of the event per observation.
        - ``relative_risk_<e>``: ``p_feature_<e> / p_control_<e>``, from
          :func:`~rojak.turbulence.metrics.relative_risk_from_table`.
        - ``odds_ratio_<e>``: ``n_11 n_00 / (n_10 n_01)``, from
          :func:`~rojak.turbulence.metrics.odds_ratio_from_table`.
    """
    sums = counts.sum(_CHUNK).compute()
    feature_n_obs = sums[f"{_FEATURE_PREFIX}{_OBSERVATIONS}"]
    control_n_obs = sums[f"{_CONTROL_PREFIX}{_OBSERVATIONS}"]
    is_feature_enough = feature_n_obs >= min_count
    is_control_enough = control_n_obs >= min_count
    is_both_enough = is_feature_enough & is_control_enough

    traffic_per_centre = _divide_or_nan(feature_n_obs, sums[_N_CENTRES])
    traffic_per_control = _divide_or_nan(control_n_obs, sums[_N_CONTROLS])
    statistics: dict[str, xr.DataArray] = {
        "traffic_per_centre": traffic_per_centre,
        "traffic_per_control": traffic_per_control,
        "traffic_ratio": _divide_or_nan(traffic_per_centre, traffic_per_control).where(is_both_enough),
    }
    for event in _event_names(counts):
        feature_n_event = sums[f"{_FEATURE_PREFIX}{event}"]
        control_n_event = sums[f"{_CONTROL_PREFIX}{event}"]
        table = ContingencyTable(
            n_11=feature_n_event,
            n_10=feature_n_obs - feature_n_event,
            n_01=control_n_event,
            n_00=control_n_obs - control_n_event,
        )
        odds_table = (
            ContingencyTable(
                n_11=table.n_11 + _HALDANE_CORRECTION,
                n_10=table.n_10 + _HALDANE_CORRECTION,
                n_01=table.n_01 + _HALDANE_CORRECTION,
                n_00=table.n_00 + _HALDANE_CORRECTION,
            )
            if haldane
            else table
        )
        # A zero cell divides by zero inside the helpers, which mask the resulting infinities with NaN themselves
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_risk = relative_risk_from_table(table, use_log=False)
            odds_ratio = odds_ratio_from_table(odds_table, use_log=False)
        statistics[f"p_feature_{event}"] = _divide_or_nan(feature_n_event, feature_n_obs).where(is_feature_enough)
        statistics[f"p_control_{event}"] = _divide_or_nan(control_n_event, control_n_obs).where(is_control_enough)
        statistics[f"relative_risk_{event}"] = relative_risk.where(is_both_enough)
        statistics[f"odds_ratio_{event}"] = odds_ratio.where(is_both_enough)
    return xr.Dataset(statistics)


def _event_names(counts: xr.Dataset) -> list[str]:
    """Events ``e`` with both ``feature_n_<e>`` and ``control_n_<e>`` in ``counts``, in the order of the variables."""
    names = [str(name) for name in counts.data_vars]
    candidates = [name.removeprefix(_FEATURE_PREFIX) for name in names if name.startswith(_FEATURE_PREFIX)]
    return [event for event in candidates if event != _OBSERVATIONS and f"{_CONTROL_PREFIX}{event}" in names]


def _divide_or_nan(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    """``numerator / denominator``, NaN where the denominator is zero rather than infinite."""
    return numerator / denominator.where(denominator != 0)


def bootstrap(
    counts: xr.Dataset,
    statistic: "Callable[[xr.Dataset], xr.Dataset]",
    *,
    n_samples: int = 1000,
    quantiles: "Sequence[float]" = (0.025, 0.5, 0.975),
    seed: int | None = None,
) -> xr.Dataset:
    """Quantiles of any statistic of the counts, from a block bootstrap over the chunks.

    Observations in the same chunk are not independent: they come from the same flights and the same features. The
    chunks are therefore the independent unit, and each sample draws ``n_chunks`` whole chunks with replacement. The
    same draw is used for every variable, so the counts of a chunk stay together, and every other dimension
    (``control``, the bins) is left untouched. Features that last longer than a chunk are still split between chunks,
    so chunks that are long compared to the life of a feature give more conservative intervals.

    ``statistic`` is called once per sample, on a resampled Dataset with the same dimensions and sizes as ``counts``,
    so it may be any function of the counts, e.g. :func:`impact_statistics` or a ``functools.partial`` of it.

    Args:
        counts (xarray.Dataset): Per-chunk counts with a ``chunk`` dimension, as returned by the pipeline. Computed once
            here if lazy, rather than once per sample.
        statistic (Callable[[xarray.Dataset], xarray.Dataset]): Function from counts to the statistics to bootstrap.
        n_samples (int): Number of bootstrap samples. At least 1. Memory scales as ``n_samples`` times the size of one
            result of ``statistic``.
        quantiles (Sequence[float]): Quantiles to return, each within [0, 1].
        seed (int | None): Seed of :func:`numpy.random.default_rng`. ``None`` draws a fresh seed.

    Returns:
        xarray.Dataset: The quantiles of each variable of ``statistic`` over the samples, with a ``quantile``
        dimension in addition to the dimensions of the variable.

    Raises:
        ValueError: If ``n_samples`` is below 1 or a quantile is outside [0, 1].
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, but is {n_samples}")
    outside = [quantile for quantile in quantiles if not 0 <= quantile <= 1]
    if outside:
        raise ValueError(f"Every quantile must be within [0, 1], but these are not: {outside}")

    counts = counts.compute()
    rng = np.random.default_rng(seed)
    n_chunks = counts.sizes[_CHUNK]
    samples = [statistic(counts.isel({_CHUNK: rng.integers(0, n_chunks, size=n_chunks)})) for _ in range(n_samples)]
    return xr.concat(samples, dim="sample").quantile(list(quantiles), dim="sample")
