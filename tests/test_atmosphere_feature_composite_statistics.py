"""Tests for the impact statistics and bootstrap of the feature composite (``_statistics``, phase 1).

Phase 1 covers the traffic ratio, and the probability, relative risk and odds ratio of each event. The expected values
are computed by hand from the 2x2 table

    ========  ==================  ==================
              event               no event
    ========  ==================  ==================
    feature   n_11 = E_feat       n_10 = N_feat - E_feat
    control   n_01 = E_ctl        n_00 = N_ctl - E_ctl
    ========  ==================  ==================

and one bin reuses the aspirin/placebo table of Agresti (2022) that the doctests of ``rojak.turbulence.metrics`` use.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest
import xarray as xr

from rojak.atmosphere.feature_composite._statistics import bootstrap, impact_statistics

TRAFFIC_VARIABLES: frozenset[str] = frozenset({"traffic_per_centre", "traffic_per_control", "traffic_ratio"})


def _event_variables(event: str) -> set[str]:
    return {f"p_feature_{event}", f"p_control_{event}", f"relative_risk_{event}", f"odds_ratio_{event}"}


def _counts(
    *,
    feature_n_obs: Sequence | np.ndarray,
    n_centres: Sequence | np.ndarray,
    control_n_obs: Sequence | np.ndarray,
    n_controls: Sequence | np.ndarray,
    events: Mapping[str, tuple[Sequence | np.ndarray, Sequence | np.ndarray]] | None = None,
    control_labels: Sequence[str] | None = None,
) -> xr.Dataset:
    """Per-chunk counts as the pipeline returns them (section 2 of the plan), for a grid of bins.

    Args:
        feature_n_obs: ``(chunk, y_km, x_km)``
        n_centres: ``(chunk,)``
        control_n_obs: ``(chunk, control, y_km, x_km)``
        n_controls: ``(chunk, control)``
        events: ``{event: (feature_n_<event>, control_n_<event>)}`` with the shapes of the observation counts
        control_labels: Labels of the controls, defaults to ``ctl0``, ``ctl1``, ...
    """
    feature = np.asarray(feature_n_obs, dtype=float)
    control = np.asarray(control_n_obs, dtype=float)
    n_chunks, n_y, n_x = feature.shape
    n_control_designs = control.shape[1]
    labels = list(control_labels) if control_labels is not None else [f"ctl{i}" for i in range(n_control_designs)]
    feature_dims = ("chunk", "y_km", "x_km")
    control_dims = ("chunk", "control", "y_km", "x_km")
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "feature_n_obs": (feature_dims, feature),
        "n_centres": (("chunk",), np.asarray(n_centres, dtype=float)),
        "control_n_obs": (control_dims, control),
        "n_controls": (("chunk", "control"), np.asarray(n_controls, dtype=float)),
    }
    for event, (feature_events, control_events) in (events or {}).items():
        data_vars[f"feature_n_{event}"] = (feature_dims, np.asarray(feature_events, dtype=float))
        data_vars[f"control_n_{event}"] = (control_dims, np.asarray(control_events, dtype=float))
    return xr.Dataset(
        data_vars,
        coords={
            "chunk": np.arange(n_chunks),
            "control": labels,
            "y_km": 50.0 * np.arange(n_y),
            "x_km": 50.0 * np.arange(n_x),
        },
    )


def _value(array: xr.DataArray, **positions: int) -> float:
    """Scalar at the given positions, ignoring dimensions the array does not have.

    ``float`` fails if any dimension is left over (e.g. ``chunk``), so this also checks the dims of the result.
    """
    return float(array.isel({dim: index for dim, index in positions.items() if dim in array.dims}))


# Two chunks, one control, one row of two bins and one event "mog".
#
# Bin 0 is the aspirin/placebo table of Agresti (2022), split over the chunks:
#   feature (placebo): E = 18 + 171 = 189 heart attacks out of N = 189 + 10845 = 11034
#   control (aspirin): E = 5 + 99 = 104 heart attacks out of N = 104 + 10933 = 11037
# Bin 1 has round numbers: feature E = 10 of N = 40, control E = 8 of N = 80.
HAND_FEATURE_N_OBS = [[[6000, 25]], [[5034, 15]]]
HAND_FEATURE_N_MOG = [[[100, 6]], [[89, 4]]]
HAND_CONTROL_N_OBS = [[[[5000, 50]]], [[[6037, 30]]]]
HAND_CONTROL_N_MOG = [[[[50, 5]]], [[[54, 3]]]]
HAND_N_CENTRES = [3, 2]
HAND_N_CONTROLS = [[8], [4]]


def _hand_counts() -> xr.Dataset:
    return _counts(
        feature_n_obs=HAND_FEATURE_N_OBS,
        n_centres=HAND_N_CENTRES,
        control_n_obs=HAND_CONTROL_N_OBS,
        n_controls=HAND_N_CONTROLS,
        events={"mog": (HAND_FEATURE_N_MOG, HAND_CONTROL_N_MOG)},
        control_labels=["obs-7D"],
    )


def _single_bin_counts(
    *,
    feature_n_obs: float,
    feature_n_event: float,
    control_n_obs: float,
    control_n_event: float,
    n_centres: float = 4,
    n_controls: float = 8,
) -> xr.Dataset:
    """One chunk, one control, one bin and one event "mog"."""
    return _counts(
        feature_n_obs=[[[feature_n_obs]]],
        n_centres=[n_centres],
        control_n_obs=[[[[control_n_obs]]]],
        n_controls=[[n_controls]],
        events={"mog": ([[[feature_n_event]]], [[[[control_n_event]]]])},
    )


def _random_counts(n_chunks: int = 40, seed: int = 42) -> xr.Dataset:
    """Two controls, a 2 x 3 grid of bins and one event, with every bin well above the default ``min_count``."""
    rng = np.random.default_rng(seed)
    feature_n_obs = rng.poisson(100, size=(n_chunks, 2, 3))
    control_n_obs = rng.poisson(200, size=(n_chunks, 2, 2, 3))
    return _counts(
        feature_n_obs=feature_n_obs,
        n_centres=rng.poisson(5, size=n_chunks) + 1,
        control_n_obs=control_n_obs,
        n_controls=rng.poisson(20, size=(n_chunks, 2)) + 1,
        events={"mog": (rng.binomial(feature_n_obs, 0.2), rng.binomial(control_n_obs, 0.1))},
        control_labels=["obs-7D", "obs+7D"],
    )


def _total_feature_n_obs(counts: xr.Dataset) -> xr.Dataset:
    return xr.Dataset({"total": counts["feature_n_obs"].sum("chunk")})


class TestImpactStatisticsHandComputed:
    def test_traffic_per_centre_and_per_control(self) -> None:
        result = impact_statistics(_hand_counts())

        # 3 + 2 = 5 centres and 8 + 4 = 12 controls over the two chunks
        assert _value(result["traffic_per_centre"], control=0, y_km=0, x_km=0) == pytest.approx(11034 / 5)
        assert _value(result["traffic_per_centre"], control=0, y_km=0, x_km=1) == pytest.approx(40 / 5)
        assert _value(result["traffic_per_control"], control=0, y_km=0, x_km=0) == pytest.approx(11037 / 12)
        assert _value(result["traffic_per_control"], control=0, y_km=0, x_km=1) == pytest.approx(80 / 12)

    def test_traffic_ratio(self) -> None:
        result = impact_statistics(_hand_counts())

        assert _value(result["traffic_ratio"], control=0, y_km=0, x_km=0) == pytest.approx((11034 / 5) / (11037 / 12))
        assert _value(result["traffic_ratio"], control=0, y_km=0, x_km=1) == pytest.approx(1.2)

    def test_probabilities(self) -> None:
        result = impact_statistics(_hand_counts())

        assert _value(result["p_feature_mog"], control=0, y_km=0, x_km=0) == pytest.approx(189 / 11034)
        assert _value(result["p_feature_mog"], control=0, y_km=0, x_km=1) == pytest.approx(0.25)
        assert _value(result["p_control_mog"], control=0, y_km=0, x_km=0) == pytest.approx(104 / 11037)
        assert _value(result["p_control_mog"], control=0, y_km=0, x_km=1) == pytest.approx(0.1)

    def test_relative_risk(self) -> None:
        result = impact_statistics(_hand_counts())

        # n_11 / (n_11 + n_10) divided by n_01 / (n_01 + n_00)
        agresti = _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0)
        assert agresti == pytest.approx((189 / 11034) / (104 / 11037))
        assert agresti == pytest.approx(1.82, abs=0.005)  # as in the relative_risk_from_table doctest
        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=1) == pytest.approx(2.5)

    def test_odds_ratio(self) -> None:
        result = impact_statistics(_hand_counts())

        # n_11 * n_00 / (n_10 * n_01)
        agresti = _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0)
        assert agresti == pytest.approx((189 * 10933) / (10845 * 104))
        assert agresti == pytest.approx(1.83, abs=0.005)  # as in the odds_ratio_from_table doctest
        # (10 * 72) / (30 * 8)
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=1) == pytest.approx(3.0)

    def test_ratios_are_not_logs(self) -> None:
        # No association: same probability on both sides, so both ratios are 1 (a log would give 0)
        counts = _single_bin_counts(feature_n_obs=50, feature_n_event=10, control_n_obs=100, control_n_event=20)

        result = impact_statistics(counts)

        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == pytest.approx(1.0)
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0) == pytest.approx(1.0)

    def test_sums_over_chunk(self) -> None:
        result = impact_statistics(_hand_counts())

        for name, variable in result.data_vars.items():
            assert "chunk" not in variable.dims, name

    def test_has_traffic_and_event_variables(self) -> None:
        result = impact_statistics(_hand_counts())

        assert set(result.data_vars) == TRAFFIC_VARIABLES | _event_variables("mog")

    def test_keeps_bin_coordinates(self) -> None:
        counts = _hand_counts()

        result = impact_statistics(counts)

        np.testing.assert_array_equal(result["y_km"].to_numpy(), counts["y_km"].to_numpy())
        np.testing.assert_array_equal(result["x_km"].to_numpy(), counts["x_km"].to_numpy())


class TestImpactStatisticsControlDimension:
    @staticmethod
    def _two_controls() -> xr.Dataset:
        # One chunk, one bin. Feature: 10 of 40. Control "a": 8 of 80 over 10 controls; "b": 20 of 80 over 20 controls
        return _counts(
            feature_n_obs=[[[40]]],
            n_centres=[5],
            control_n_obs=[[[[80]], [[80]]]],
            n_controls=[[10, 20]],
            events={"mog": ([[[10]]], [[[[8]], [[20]]]])},
            control_labels=["a", "b"],
        )

    def test_ratios_keep_control_dimension(self) -> None:
        result = impact_statistics(self._two_controls())

        for name in ("traffic_ratio", "relative_risk_mog", "odds_ratio_mog"):
            assert "control" in result[name].dims, name
        assert result["control"].to_numpy().tolist() == ["a", "b"]

    def test_each_control_gives_its_own_values(self) -> None:
        result = impact_statistics(self._two_controls())

        # Traffic: 40 / 5 = 8 per centre; 80 / 10 = 8 and 80 / 20 = 4 per control
        assert _value(result["traffic_ratio"], control=0, y_km=0, x_km=0) == pytest.approx(1.0)
        assert _value(result["traffic_ratio"], control=1, y_km=0, x_km=0) == pytest.approx(2.0)
        # p_feature = 0.25; p_control = 0.1 and 0.25
        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == pytest.approx(2.5)
        assert _value(result["relative_risk_mog"], control=1, y_km=0, x_km=0) == pytest.approx(1.0)
        # (10 * 72) / (30 * 8) and (10 * 60) / (30 * 20)
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0) == pytest.approx(3.0)
        assert _value(result["odds_ratio_mog"], control=1, y_km=0, x_km=0) == pytest.approx(1.0)


class TestImpactStatisticsMasking:
    @staticmethod
    def _counts_for_masking() -> xr.Dataset:
        # Four bins in a row: feature side small, control side small, both large, both exactly at min_count = 10
        return _counts(
            feature_n_obs=[[[100, 5, 100, 10]]],
            n_centres=[4],
            control_n_obs=[[[[5, 100, 100, 10]]]],
            n_controls=[[8]],
            events={"mog": ([[[20, 2, 20, 4]]], [[[[1, 10, 10, 2]]]])},
        )

    @pytest.mark.parametrize("name", ["traffic_ratio", "relative_risk_mog", "odds_ratio_mog"])
    def test_ratios_are_nan_where_either_side_is_below_min_count(self, name: str) -> None:
        result = impact_statistics(self._counts_for_masking(), min_count=10)

        assert np.isnan(_value(result[name], control=0, y_km=0, x_km=0))  # control side too small
        assert np.isnan(_value(result[name], control=0, y_km=0, x_km=1))  # feature side too small
        assert np.isfinite(_value(result[name], control=0, y_km=0, x_km=2))
        assert np.isfinite(_value(result[name], control=0, y_km=0, x_km=3))  # equal to min_count is kept

    def test_probabilities_are_nan_where_their_side_is_below_min_count(self) -> None:
        result = impact_statistics(self._counts_for_masking(), min_count=10)

        assert np.isnan(_value(result["p_feature_mog"], control=0, y_km=0, x_km=1))
        assert np.isnan(_value(result["p_control_mog"], control=0, y_km=0, x_km=0))
        assert _value(result["p_feature_mog"], control=0, y_km=0, x_km=2) == pytest.approx(0.2)
        assert _value(result["p_control_mog"], control=0, y_km=0, x_km=2) == pytest.approx(0.1)

    def test_min_count_applies_to_counts_summed_over_chunks(self) -> None:
        # 6 + 6 observations on each side: each chunk is below min_count = 10, but the sum is not
        counts = _counts(
            feature_n_obs=[[[6]], [[6]]],
            n_centres=[1, 1],
            control_n_obs=[[[[6]]], [[[6]]]],
            n_controls=[[2], [2]],
            events={"mog": ([[[3]], [[3]]], [[[[1]]], [[[1]]]])},
        )

        result = impact_statistics(counts, min_count=10)

        assert _value(result["traffic_ratio"], control=0, y_km=0, x_km=0) == pytest.approx((12 / 2) / (12 / 4))
        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == pytest.approx(3.0)

    def test_default_min_count_is_ten(self) -> None:
        counts = _single_bin_counts(feature_n_obs=9, feature_n_event=3, control_n_obs=100, control_n_event=10)

        result = impact_statistics(counts)

        assert np.isnan(_value(result["traffic_ratio"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["relative_risk_mog"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0))


class TestImpactStatisticsHaldane:
    def test_adds_half_to_every_cell_of_the_odds_ratio_table(self) -> None:
        result = impact_statistics(_hand_counts(), haldane=True)

        # Bin 1: n_11 = 10, n_10 = 30, n_01 = 8, n_00 = 72
        expected = (10.5 * 72.5) / (30.5 * 8.5)
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=1) == pytest.approx(expected)

    def test_does_not_change_relative_risk_or_probabilities(self) -> None:
        plain = impact_statistics(_hand_counts())
        corrected = impact_statistics(_hand_counts(), haldane=True)

        for name in ("relative_risk_mog", "p_feature_mog", "p_control_mog", "traffic_ratio"):
            xr.testing.assert_allclose(corrected[name], plain[name])

    def test_makes_odds_ratio_finite_when_control_has_no_events(self) -> None:
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=10, control_n_obs=80, control_n_event=0)

        result = impact_statistics(counts, haldane=True)

        # n_11 = 10.5, n_10 = 30.5, n_01 = 0.5, n_00 = 80.5
        expected = (10.5 * 80.5) / (30.5 * 0.5)
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0) == pytest.approx(expected)

    def test_is_off_by_default(self) -> None:
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=10, control_n_obs=80, control_n_event=8)

        result = impact_statistics(counts)

        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0) == pytest.approx(3.0)


class TestImpactStatisticsZeroCounts:
    def test_no_control_events_give_nan_not_inf(self) -> None:
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=10, control_n_obs=80, control_n_event=0)

        result = impact_statistics(counts)

        assert np.isnan(_value(result["relative_risk_mog"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0))
        assert _value(result["p_control_mog"], control=0, y_km=0, x_km=0) == 0.0

    def test_all_feature_observations_are_events_gives_nan_odds_ratio(self) -> None:
        # n_10 = 0 is a zero in the denominator of the odds ratio only
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=40, control_n_obs=80, control_n_event=8)

        result = impact_statistics(counts)

        assert np.isnan(_value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0))
        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == pytest.approx(10.0)

    def test_no_feature_events_give_zero_ratios(self) -> None:
        # Zero in the numerator is a real result, not a missing one
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=0, control_n_obs=80, control_n_event=8)

        result = impact_statistics(counts)

        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == 0.0
        assert _value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0) == 0.0

    def test_no_events_on_either_side_give_nan(self) -> None:
        counts = _single_bin_counts(feature_n_obs=40, feature_n_event=0, control_n_obs=80, control_n_event=0)

        result = impact_statistics(counts)

        assert np.isnan(_value(result["relative_risk_mog"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["odds_ratio_mog"], control=0, y_km=0, x_km=0))

    def test_no_centres_give_nan_traffic_not_inf(self) -> None:
        counts = _single_bin_counts(
            feature_n_obs=40, feature_n_event=10, control_n_obs=80, control_n_event=8, n_centres=0
        )

        result = impact_statistics(counts)

        assert np.isnan(_value(result["traffic_per_centre"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["traffic_ratio"], control=0, y_km=0, x_km=0))

    def test_no_controls_give_nan_traffic_not_inf(self) -> None:
        counts = _single_bin_counts(
            feature_n_obs=40, feature_n_event=10, control_n_obs=80, control_n_event=8, n_controls=0
        )

        result = impact_statistics(counts)

        assert np.isnan(_value(result["traffic_per_control"], control=0, y_km=0, x_km=0))
        assert np.isnan(_value(result["traffic_ratio"], control=0, y_km=0, x_km=0))

    def test_empty_bins_give_no_infinities(self) -> None:
        counts = _counts(
            feature_n_obs=[[[0, 40]]],
            n_centres=[0],
            control_n_obs=[[[[0, 80]]]],
            n_controls=[[0]],
            events={"mog": ([[[0, 40]]], [[[[0, 0]]]])},
        )

        result = impact_statistics(counts, min_count=0)

        for name, variable in result.data_vars.items():
            assert not np.isinf(variable.to_numpy()).any(), name


class TestImpactStatisticsEvents:
    def test_no_event_variables_give_only_traffic(self) -> None:
        counts = _counts(
            feature_n_obs=HAND_FEATURE_N_OBS,
            n_centres=HAND_N_CENTRES,
            control_n_obs=HAND_CONTROL_N_OBS,
            n_controls=HAND_N_CONTROLS,
        )

        result = impact_statistics(counts)

        assert set(result.data_vars) == TRAFFIC_VARIABLES
        assert _value(result["traffic_ratio"], control=0, y_km=0, x_km=1) == pytest.approx(1.2)

    def test_every_event_pair_gets_its_own_measures(self) -> None:
        counts = _counts(
            feature_n_obs=[[[40]]],
            n_centres=[5],
            control_n_obs=[[[[80]]]],
            n_controls=[[10]],
            events={"mog": ([[[10]]], [[[[8]]]]), "severe": ([[[4]]], [[[[4]]]])},
        )

        result = impact_statistics(counts)

        assert set(result.data_vars) == TRAFFIC_VARIABLES | _event_variables("mog") | _event_variables("severe")
        assert _value(result["relative_risk_mog"], control=0, y_km=0, x_km=0) == pytest.approx(2.5)
        assert _value(result["relative_risk_severe"], control=0, y_km=0, x_km=0) == pytest.approx(2.0)


class TestImpactStatisticsLazy:
    def test_dask_input_gives_same_result_as_eager(self) -> None:
        eager = _random_counts(n_chunks=6)
        lazy = eager.chunk({"chunk": 2})

        from_eager = impact_statistics(eager)
        from_lazy = impact_statistics(lazy)

        xr.testing.assert_allclose(from_lazy, from_eager)

    def test_result_of_dask_input_is_computed(self) -> None:
        lazy = _random_counts(n_chunks=6).chunk({"chunk": 2})

        result = impact_statistics(lazy)

        for name, variable in result.data_vars.items():
            assert variable.chunks is None, name


class TestBootstrap:
    def test_same_seed_gives_same_result(self) -> None:
        counts = _random_counts()

        first = bootstrap(counts, impact_statistics, n_samples=50, seed=7)
        second = bootstrap(counts, impact_statistics, n_samples=50, seed=7)

        xr.testing.assert_identical(first, second)

    def test_different_seeds_give_different_results(self) -> None:
        counts = _random_counts()

        first = bootstrap(counts, _total_feature_n_obs, n_samples=50, seed=7)
        second = bootstrap(counts, _total_feature_n_obs, n_samples=50, seed=8)

        assert not np.allclose(first["total"].to_numpy(), second["total"].to_numpy())

    def test_identical_chunks_give_zero_width_intervals(self) -> None:
        one_chunk = _random_counts(n_chunks=1)
        counts = xr.concat([one_chunk] * 10, dim="chunk").assign_coords(chunk=np.arange(10))

        result = bootstrap(counts, impact_statistics, n_samples=50, seed=0)

        for name, variable in result.data_vars.items():
            values = variable.to_numpy()
            assert np.isfinite(values).all(), name
            width = variable.max("quantile") - variable.min("quantile")
            assert np.allclose(width.to_numpy(), 0.0, rtol=0, atol=1e-9), name

    def test_median_of_sum_is_close_to_full_sample_sum(self) -> None:
        counts = _random_counts()
        full_sum = counts["feature_n_obs"].sum("chunk")

        result = bootstrap(counts, _total_feature_n_obs, n_samples=500, quantiles=(0.025, 0.5, 0.975), seed=42)

        total = result["total"]
        np.testing.assert_allclose(total.sel(quantile=0.5).to_numpy(), full_sum.to_numpy(), rtol=0.02)
        assert (total.sel(quantile=0.025) <= full_sum).all()
        assert (total.sel(quantile=0.975) >= full_sum).all()

    def test_output_dims_are_quantile_and_statistic_dims(self) -> None:
        counts = _random_counts()
        quantiles = (0.05, 0.5, 0.95)

        result = bootstrap(counts, impact_statistics, n_samples=20, quantiles=quantiles, seed=0)
        statistic = impact_statistics(counts)

        assert set(result.data_vars) == set(statistic.data_vars)
        for name, variable in result.data_vars.items():
            assert set(variable.dims) == {"quantile", *statistic[name].dims}, name
        np.testing.assert_allclose(result["quantile"].to_numpy(), quantiles)
        assert result["control"].to_numpy().tolist() == ["obs-7D", "obs+7D"]

    def test_default_quantiles(self) -> None:
        result = bootstrap(_random_counts(), _total_feature_n_obs, n_samples=20, seed=0)

        np.testing.assert_allclose(result["quantile"].to_numpy(), [0.025, 0.5, 0.975])

    def test_quantiles_zero_and_one_are_allowed(self) -> None:
        result = bootstrap(_random_counts(), _total_feature_n_obs, n_samples=20, quantiles=(0.0, 1.0), seed=0)

        np.testing.assert_allclose(result["quantile"].to_numpy(), [0.0, 1.0])
        assert (result["total"].sel(quantile=0.0) <= result["total"].sel(quantile=1.0)).all()

    def test_statistic_is_applied_to_each_sample(self) -> None:
        calls: list[xr.Dataset] = []

        def statistic(counts: xr.Dataset) -> xr.Dataset:
            calls.append(counts)
            return _total_feature_n_obs(counts)

        n_samples = 25

        bootstrap(_random_counts(), statistic, n_samples=n_samples, seed=0)

        assert len(calls) == n_samples

    def test_resamples_whole_chunks_only(self) -> None:
        # Every value encodes its chunk, control and bin, so a resampled chunk can be traced back to the original:
        # the same chunk must be drawn for every variable, and control and bins must be untouched
        n_chunks, n_control, n_y, n_x = 10, 2, 2, 3
        chunk, control, y, x = np.ogrid[:n_chunks, :n_control, :n_y, :n_x]
        original_control_n_obs = 1000 * chunk + 100 * control + 10 * y + x
        original_feature_n_obs = original_control_n_obs[:, 0] + 0.5
        original_n_controls = 10 * chunk[:, :, 0, 0] + control[:, :, 0, 0]
        counts = _counts(
            feature_n_obs=original_feature_n_obs,
            n_centres=np.arange(n_chunks),
            control_n_obs=original_control_n_obs,
            n_controls=original_n_controls,
            control_labels=["obs-7D", "obs+7D"],
        )
        samples: list[xr.Dataset] = []

        def statistic(sample: xr.Dataset) -> xr.Dataset:
            samples.append(sample)
            return _total_feature_n_obs(sample)

        bootstrap(counts, statistic, n_samples=25, seed=3)

        assert samples
        drew_a_chunk_twice = False
        for sample in samples:
            assert dict(sample.sizes) == dict(counts.sizes)
            assert sample["control"].to_numpy().tolist() == ["obs-7D", "obs+7D"]
            np.testing.assert_array_equal(sample["y_km"].to_numpy(), counts["y_km"].to_numpy())
            np.testing.assert_array_equal(sample["x_km"].to_numpy(), counts["x_km"].to_numpy())
            drawn = sample["n_centres"].to_numpy().astype(int)
            np.testing.assert_array_equal(sample["feature_n_obs"].to_numpy(), original_feature_n_obs[drawn])
            np.testing.assert_array_equal(sample["control_n_obs"].to_numpy(), original_control_n_obs[drawn])
            np.testing.assert_array_equal(sample["n_controls"].to_numpy(), original_n_controls[drawn])
            drew_a_chunk_twice = drew_a_chunk_twice or len(set(drawn.tolist())) < n_chunks
        # With replacement: across 25 samples of 10 chunks, some chunk is drawn twice
        assert drew_a_chunk_twice

    def test_ratio_invariant_to_chunk_weights_has_zero_width(self) -> None:
        # Per chunk, control_n_obs = n_controls * pattern, so sum(control_n_obs) / sum(n_controls) = pattern for any
        # draw of chunks, provided the same chunks are drawn for both variables and control and bins are not resampled
        rng = np.random.default_rng(42)
        n_chunks = 12
        pattern = rng.uniform(1.0, 5.0, size=(2, 2, 3))
        n_controls = rng.integers(1, 20, size=(n_chunks, 2)).astype(float)
        counts = _counts(
            feature_n_obs=rng.poisson(50, size=(n_chunks, 2, 3)),
            n_centres=np.ones(n_chunks),
            control_n_obs=n_controls[:, :, None, None] * pattern,
            n_controls=n_controls,
        )

        def per_control(sample: xr.Dataset) -> xr.Dataset:
            return xr.Dataset({"ratio": sample["control_n_obs"].sum("chunk") / sample["n_controls"].sum("chunk")})

        result = bootstrap(counts, per_control, n_samples=30, seed=1)

        expected = xr.DataArray(pattern, dims=("control", "y_km", "x_km"))
        for quantile in result["quantile"].to_numpy():
            np.testing.assert_allclose(
                result["ratio"].sel(quantile=quantile).transpose("control", "y_km", "x_km").to_numpy(),
                expected.to_numpy(),
                rtol=1e-12,
            )

    @pytest.mark.parametrize("n_samples", [0, -1])
    def test_n_samples_below_one_raises_value_error(self, n_samples: int) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            bootstrap(_random_counts(), _total_feature_n_obs, n_samples=n_samples, seed=0)

    @pytest.mark.parametrize("quantiles", [(-0.1,), (1.1,), (0.025, 1.5), (-0.5, 0.5)])
    def test_quantiles_outside_unit_interval_raise_value_error(self, quantiles: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="quantile"):
            bootstrap(_random_counts(), _total_feature_n_obs, n_samples=10, quantiles=quantiles, seed=0)

    def test_statistic_type_is_generic(self) -> None:
        # Anything with the signature counts -> Dataset works, not just impact_statistics
        statistic: Callable[[xr.Dataset], xr.Dataset] = _total_feature_n_obs

        result = bootstrap(_random_counts(), statistic, n_samples=10, seed=0)

        assert set(result.data_vars) == {"total"}
