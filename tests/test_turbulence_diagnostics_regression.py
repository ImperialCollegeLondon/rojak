import json
from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import distributed
import numpy as np
import pytest
import xarray as xr
from dask.base import compute as dask_compute
from dask.base import is_dask_collection
from dask.highlevelgraph import HighLevelGraph

import rojak.core.derivatives as _derivatives_mod
import rojak.turbulence.diagnostic as _diagnostic_mod
from rojak.core.data import CATData
from rojak.core.derivatives import VelocityDerivative, get_projection_correction_factors
from rojak.orchestrator.configuration import (
    TurbulenceDiagnostics,
    TurbulenceSeverity,
    TurbulenceThresholdMode,
    TurbulenceThresholds,
)
from rojak.turbulence.analysis import (
    ComputeDistributionParametersForEDR,
    DiagnosticHistogramDistribution,
    TurbulenceIntensityThresholds,
    TurbulenceProbabilityBySeverity,
    TurbulentRegionsBySeverity,
)
from rojak.turbulence.calculations import altitude_derivative_on_pressure_level, wind_direction
from rojak.turbulence.diagnostic import (
    CalibrationDiagnosticSuite,
    DiagnosticFactory,
    DiagnosticSuite,
    EvaluationDiagnosticSuite,
    Frontogenesis2D,
    Frontogenesis3D,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from rojak.turbulence.analysis import HistogramData

GOLDEN_DIR = Path(__file__).resolve().parent / "_static" / "goldens"
TASK_COUNT_BASELINE_PATH = GOLDEN_DIR / "task_counts.json"

ALL_DIAGNOSTICS = list(TurbulenceDiagnostics)
# DiagnosticHistogramDistribution.build_lazy_collection: minimum, maximum, mean, variance
NUM_HISTOGRAM_STATISTICS = 4
# compute_distribution_parameters: the statistics, then the counts over the range they give
HISTOGRAM_ROUND_TRIPS = 2
TIME_CHUNKS = {"time": 1}
TIME_CHUNKED = pytest.param(TIME_CHUNKS, id="time_chunked")
LAT_LON_CHUNKED = pytest.param(
    {"latitude": -1, "longitude": -1, "time": 1, "pressure_level": -1},
    id="latlon_chunked",
)
CHUNKINGS = [TIME_CHUNKED, LAT_LON_CHUNKED]

# The synthetic fixture has 4 pressure levels (full set).
FIXTURE_PRESSURE_LEVELS = [175.0, 200.0, 225.0, 250.0]
# PRESSURE_LEVELS is the subset used for TurbulentRegionsBySeverity.
PRESSURE_LEVELS = FIXTURE_PRESSURE_LEVELS
SEVERITIES = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE, TurbulenceSeverity.SEVERE]
# Percentiles (0-100) of the calibration distribution at which each severity starts
PERCENTILE_CONFIG = TurbulenceThresholds(light=90.0, moderate=95.0, severe=99.0)


def make_synthetic_cat_data(*, persist: bool = False, chunks: dict[str, int] | None = None) -> CATData:
    rng = np.random.default_rng(42)
    shape = (8, 8, 2, 4)  # lat, lon, time, lev
    dims = ["latitude", "longitude", "time", "pressure_level"]

    # Realistic pressure levels in hPa
    pressure_levels = np.array(FIXTURE_PRESSURE_LEVELS)
    lats = np.linspace(-30.0, 30.0, 8)
    lons = np.linspace(-30.0, 30.0, 8)
    times = np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]")
    # altitude as coord on pressure_level dim
    altitudes = np.array([12000.0, 11000.0, 10000.0, 9000.0])  # metres

    def var(scale: float = 1.0, offset: float = 0.0) -> xr.DataArray:
        return xr.DataArray(rng.random(shape) * scale + offset, dims=dims)

    data_vars = {
        "temperature": var(50.0, 200.0),  # K, ~200-250 K
        "divergence_of_wind": var(1e-5),  # 1/s
        "geopotential": var(1e5, 8e4),  # m^2/s^2, ~80000-180000
        "specific_humidity": var(1e-4),  # kg/kg
        "eastward_wind": var(40.0, -20.0),  # m/s
        "northward_wind": var(20.0, -10.0),  # m/s
        "potential_vorticity": var(1e-7),  # K m^2 kg^-1 s^-1
        "vorticity": var(1e-5),  # 1/s
    }
    coords = {
        "latitude": lats,
        "longitude": lons,
        "time": times,
        "pressure_level": pressure_levels,
        "altitude": ("pressure_level", altitudes),
    }
    ds = xr.Dataset(data_vars, coords=coords)
    if chunks:
        ds = ds.chunk(chunks)
    return CATData(ds, pressure_level_prefix=100.0, persist=persist)


def count_tasks(*collections: xr.DataArray | xr.Dataset) -> int:
    """Number of keys in the merged dask graph of ``collections``, without computing anything"""
    graphs = []
    for collection in collections:
        graph = collection.__dask_graph__()
        assert graph is not None, "count_tasks needs dask-backed collections"
        graphs.append(graph)
    return len(HighLevelGraph.merge(*graphs))


def _flatten_dataarrays(outputs: "Mapping[str, xr.DataArray]") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for name, array in outputs.items():
        computed = array.drop_vars(list(array.coords)).compute()
        flat[f"{name}__values"] = computed.values
        flat[f"{name}__dims"] = np.array(computed.dims)
    return flat


def _flatten_thresholds(thresholds: "Mapping[str, TurbulenceThresholds]") -> dict[str, np.ndarray]:
    return {f"{name}__values": np.array([t.light, t.moderate, t.severe], dtype=float) for name, t in thresholds.items()}


def _flatten_distribution_parameters(parameters: "Mapping[str, HistogramData]") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for name, histogram in parameters.items():
        flat[f"{name}__mean_variance"] = np.array([histogram.mean, histogram.variance], dtype=float)
        flat[f"{name}__hist"] = np.asarray(histogram.hist_values, dtype=float)
        flat[f"{name}__bins"] = np.asarray(histogram.bins, dtype=float)
    return flat


def _load_golden(output_name: str) -> dict[str, np.ndarray]:
    """Read a committed golden; raises ``FileNotFoundError`` if it has not been generated yet"""
    with np.load(GOLDEN_DIR / f"{output_name}.npz") as archive:
        return {key: archive[key] for key in archive.files}


def _assert_flat_matches(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_array in expected.items():
        if expected_array.dtype.kind == "U":
            np.testing.assert_array_equal(actual[key], expected_array)
        else:
            np.testing.assert_allclose(actual[key], expected_array, rtol=1e-4)


def _calibration_suite(chunks: dict[str, int] | None) -> CalibrationDiagnosticSuite:
    factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))
    return CalibrationDiagnosticSuite(factory, ALL_DIAGNOSTICS)


def _evaluation_suite(chunks: dict[str, int] | None) -> EvaluationDiagnosticSuite:
    thresholds = _calibration_suite(chunks).compute_thresholds(PERCENTILE_CONFIG)
    factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))
    return EvaluationDiagnosticSuite(
        factory,
        ALL_DIAGNOSTICS,
        severities=SEVERITIES,
        pressure_levels=PRESSURE_LEVELS,
        probability_thresholds=thresholds,
        threshold_mode=TurbulenceThresholdMode.BOUNDED,
    )


FRONTOGENESIS = [TurbulenceDiagnostics.F2D, TurbulenceDiagnostics.F3D]


@pytest.mark.filterwarnings("error:'where' used without 'out':UserWarning")
class TestFrontogenesisReciprocal:
    @pytest.mark.parametrize("diagnostic", FRONTOGENESIS, ids=lambda d: d.name)
    def test_emits_no_where_without_out_warning(self, diagnostic: TurbulenceDiagnostics) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data())
        factory.create(diagnostic).computed_value.compute()

    @pytest.mark.parametrize("diagnostic", FRONTOGENESIS, ids=lambda d: d.name)
    def test_repeated_calls_are_bit_identical_and_finite(self, diagnostic: TurbulenceDiagnostics) -> None:
        first = DiagnosticFactory(make_synthetic_cat_data()).create(diagnostic).computed_value.compute()
        second = DiagnosticFactory(make_synthetic_cat_data()).create(diagnostic).computed_value.compute()

        assert np.isfinite(first.values).all()
        xr.testing.assert_identical(first, second)

    @pytest.mark.parametrize("diagnostic", FRONTOGENESIS, ids=lambda d: d.name)
    def test_uniform_potential_temperature_gives_zero_result(self, diagnostic: TurbulenceDiagnostics) -> None:
        """Uniform θ → |∇θ| = 0 everywhere → the xr.where reciprocal fix must yield 0.0, not inf/NaN."""
        rng = np.random.default_rng(7)
        shape = (4, 4, 2, 3)
        dims = ["latitude", "longitude", "time", "pressure_level"]
        coords = {
            "latitude": np.linspace(-10.0, 10.0, 4),
            "longitude": np.linspace(-10.0, 10.0, 4),
            "time": np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]"),
            "pressure_level": np.array([200.0, 225.0, 250.0]),
        }

        def rand(scale: float = 1.0, offset: float = 0.0) -> xr.DataArray:
            return xr.DataArray(rng.random(shape) * scale + offset, dims=dims, coords=coords)

        uniform_theta = xr.DataArray(np.full(shape, 300.0), dims=dims, coords=coords)
        derivatives = {
            VelocityDerivative.DU_DX: rand(1e-4),
            VelocityDerivative.DV_DX: rand(1e-4),
            VelocityDerivative.DU_DY: rand(1e-4),
            VelocityDerivative.DV_DY: rand(1e-4),
        }
        u, v, geo = rand(40.0, -20.0), rand(20.0, -10.0), rand(1e5, 8e4)
        if diagnostic == TurbulenceDiagnostics.F2D:
            inst = Frontogenesis2D(u, v, uniform_theta, geo, derivatives)
        else:
            inst = Frontogenesis3D(u, v, uniform_theta, geo, rand(1e-5), derivatives)

        result = inst.computed_value.compute()
        # The gradient of a constant field is mathematically zero, but projection correction factors
        # introduce floating-point noise at ~1e-24.  The critical property is finite values — the
        # old np.reciprocal(where=...) pattern produced uninitialised memory (inf/NaN) here.
        assert np.isfinite(result.values).all(), "zero-gradient cells must not produce inf/NaN"
        np.testing.assert_allclose(result.values, 0.0, atol=1e-20)


class TestDiagnosticGoldens:
    @pytest.mark.parametrize("chunks", CHUNKINGS)
    @pytest.mark.parametrize("diagnostic", ALL_DIAGNOSTICS, ids=lambda d: d.name)
    def test_diagnostic_matches_golden(self, diagnostic: TurbulenceDiagnostics, chunks: dict[str, int]) -> None:
        # Load the golden first so that a missing file fails before any computation
        with np.load(GOLDEN_DIR / f"{diagnostic.value}.npz") as archive:
            expected = xr.DataArray(archive["values"], dims=tuple(archive["dims"].tolist()))

        actual = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks)).create(diagnostic).computed_value
        xr.testing.assert_allclose(actual.drop_vars(list(actual.coords)).compute(), expected, rtol=1e-4)

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_thresholds_matches_golden(self, chunks: dict[str, int]) -> None:
        expected = _load_golden("compute_thresholds")
        thresholds = _calibration_suite(chunks).compute_thresholds(PERCENTILE_CONFIG)
        _assert_flat_matches(expected, _flatten_thresholds(thresholds))

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_distribution_parameters_matches_golden(self, chunks: dict[str, int]) -> None:
        expected = _load_golden("compute_distribution_parameters")
        parameters = _calibration_suite(chunks).compute_distribution_parameters()
        _assert_flat_matches(expected, _flatten_distribution_parameters(parameters))

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_probabilities_matches_golden(self, chunks: dict[str, int]) -> None:
        expected = _load_golden("probabilities")
        probabilities = _evaluation_suite(chunks).probabilities
        _assert_flat_matches(expected, _flatten_dataarrays(probabilities))

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_turbulent_regions_matches_golden(self, chunks: dict[str, int]) -> None:
        expected = _load_golden("compute_turbulent_regions")
        regions = _evaluation_suite(chunks).compute_turbulent_regions()
        _assert_flat_matches(expected, _flatten_dataarrays(regions))


ANGLE_GRADIENT_DIAGNOSTICS = [TurbulenceDiagnostics.ENDLICH, TurbulenceDiagnostics.DIRECTIONAL_SHEAR]
# Splits the 4 pressure levels across 2 chunks, so a blockwise np.gradient would see half a column per block
PRESSURE_LEVEL_CHUNKED = {"time": 1, "pressure_level": 2, "latitude": -1, "longitude": -1}


class TestAngleGradientChunking:
    """The wind-direction gradient must not depend on how ``pressure_level`` is chunked"""

    @pytest.mark.parametrize("diagnostic", ANGLE_GRADIENT_DIAGNOSTICS, ids=lambda d: d.name)
    def test_pressure_level_chunked_matches_golden(self, diagnostic: TurbulenceDiagnostics) -> None:
        with np.load(GOLDEN_DIR / f"{diagnostic.value}.npz") as archive:
            expected = xr.DataArray(archive["values"], dims=tuple(archive["dims"].tolist()))

        data = make_synthetic_cat_data(chunks=PRESSURE_LEVEL_CHUNKED)
        actual = DiagnosticFactory(data).create(diagnostic).computed_value

        assert is_dask_collection(actual)
        xr.testing.assert_allclose(actual.drop_vars(list(actual.coords)).compute(), expected, rtol=1e-4)

    @pytest.mark.parametrize("diagnostic", ANGLE_GRADIENT_DIAGNOSTICS, ids=lambda d: d.name)
    def test_pressure_level_chunked_matches_numpy(self, diagnostic: TurbulenceDiagnostics) -> None:
        expected = DiagnosticFactory(make_synthetic_cat_data()).create(diagnostic).computed_value
        data = make_synthetic_cat_data(chunks=PRESSURE_LEVEL_CHUNKED)
        actual = DiagnosticFactory(data).create(diagnostic).computed_value

        xr.testing.assert_allclose(actual.compute(), expected)

    def test_gradient_keeps_layout_and_dtype_when_pressure_level_is_not_last(self) -> None:
        data = make_synthetic_cat_data()
        direction = wind_direction(data.u_wind(), data.v_wind()).transpose(
            "pressure_level", "latitude", "longitude", "time"
        )
        expected = _diagnostic_mod._wind_direction_pressure_gradient(direction)

        actual = _diagnostic_mod._wind_direction_pressure_gradient(direction.chunk({"pressure_level": 2}))

        assert actual.dims == direction.dims
        # Checked lazily: dask reports the declared dtype, whatever the blocks compute to
        assert actual.dtype == direction.dtype == np.float64
        xr.testing.assert_allclose(actual.compute(), expected)


class TestTaskCountHarness:
    @pytest.fixture
    def scheduler_submissions(self, monkeypatch: pytest.MonkeyPatch) -> "list[str]":
        """Records every ``Client.compute`` and ``Client.get`` call made while the fixture is active"""
        submissions: list[str] = []
        original_compute = distributed.Client.compute
        original_get = distributed.Client.get

        def counting_compute(self: distributed.Client, *args: "Any", **kwargs: "Any") -> "Any":
            submissions.append("compute")
            return original_compute(self, *args, **kwargs)

        def counting_get(self: distributed.Client, *args: "Any", **kwargs: "Any") -> "Any":
            submissions.append("get")
            return original_get(self, *args, **kwargs)

        monkeypatch.setattr(distributed.Client, "compute", counting_compute)
        monkeypatch.setattr(distributed.Client, "get", counting_get)
        return submissions

    @staticmethod
    def _load_baseline() -> dict:
        with TASK_COUNT_BASELINE_PATH.open() as handle:
            loaded: dict = json.load(handle)
        return loaded

    def test_baseline_covers_all_28_diagnostics(self) -> None:
        baseline = self._load_baseline()
        assert sorted(baseline["diagnostics"]) == sorted(d.value for d in TurbulenceDiagnostics)

    def test_full_suite_task_count_matches_baseline(self) -> None:
        baseline = self._load_baseline()
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=TIME_CHUNKS))
        # persist=False: the baseline pins the size of the *recipe*. A persisted dataset has one task per chunk
        # per diagnostic whatever the recipe looks like, so counting that would measure nothing.
        suite = DiagnosticSuite(factory, ALL_DIAGNOSTICS, persist=False)

        assert count_tasks(*suite.computed_values_as_dict().values()) == baseline["full_suite_tasks"]

    def test_building_suite_submits_baseline_number_of_scheduler_calls(
        self,
        client: distributed.Client,
        scheduler_submissions: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        baseline = self._load_baseline()
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=TIME_CHUNKS))

        DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        assert len(scheduler_submissions) == baseline["build_submissions"]

    def test_round_trip_submissions_match_baseline(
        self,
        client: distributed.Client,
        scheduler_submissions: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        baseline = self._load_baseline()
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=TIME_CHUNKS))
        suite = DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        results = dask_compute(*suite.computed_values_as_dict().values())

        assert all(isinstance(result, xr.DataArray) for result in results)
        assert len(scheduler_submissions) == baseline["round_trip_submissions"]


class TestMemoisation:
    """Memoisation of projection factors and d(geopotential)/dp, which are otherwise rebuilt per diagnostic"""

    def test_projection_correction_factors_returns_same_object_for_same_inputs(self) -> None:
        latitude = xr.DataArray(np.linspace(-30.0, 30.0, 8), dims="latitude")
        longitude = xr.DataArray(np.linspace(-30.0, 30.0, 8), dims="longitude")

        first = get_projection_correction_factors(latitude, longitude, use_dask=True)
        second = get_projection_correction_factors(latitude, longitude, use_dask=True)

        assert first.parallel_scale is second.parallel_scale
        assert first.meridional_scale is second.meridional_scale

    def test_projection_correction_factors_returns_different_object_for_different_inputs(self) -> None:
        longitude = xr.DataArray(np.linspace(-30.0, 30.0, 8), dims="longitude")
        first_latitude = xr.DataArray(np.linspace(-30.0, 30.0, 8), dims="latitude")
        second_latitude = xr.DataArray(np.linspace(-20.0, 40.0, 8), dims="latitude")

        first = get_projection_correction_factors(first_latitude, longitude, use_dask=True)
        second = get_projection_correction_factors(second_latitude, longitude, use_dask=True)

        assert first.parallel_scale is not second.parallel_scale
        assert first.meridional_scale is not second.meridional_scale
        assert not np.allclose(first.meridional_scale.values, second.meridional_scale.values)

    def test_cat_data_d_geopotential_d_pressure_returns_same_object(self) -> None:
        cat_data = make_synthetic_cat_data()

        assert cat_data.d_geopotential_d_pressure() is cat_data.d_geopotential_d_pressure()

    def test_cat_data_d_geopotential_d_pressure_matches_manual_computation(self) -> None:
        cat_data = make_synthetic_cat_data()

        expected = cat_data.geopotential().differentiate("pressure_level").compute()
        actual = cat_data.d_geopotential_d_pressure().compute()

        xr.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_altitude_derivative_precomputed_d_geopotential_d_p_matches_computed(self) -> None:
        """Passing d_geopotential_d_p must give the same result as letting the function compute it."""
        cat_data = make_synthetic_cat_data()
        u = cat_data.u_wind()
        geo = cat_data.geopotential()
        precomputed = cat_data.d_geopotential_d_pressure()

        without = altitude_derivative_on_pressure_level(u, geo).compute()
        with_pre = altitude_derivative_on_pressure_level(u, geo, d_geopotential_d_p=precomputed).compute()

        xr.testing.assert_allclose(without, with_pre, rtol=1e-6)

    def test_projection_cache_has_one_entry_after_full_suite(self) -> None:
        # All 28 diagnostics share the same lat/lon grid, so the cache should contain exactly
        # one entry after suite construction regardless of how many call sites exist.
        _derivatives_mod._PROJECTION_CACHE.clear()
        DiagnosticSuite(DiagnosticFactory(make_synthetic_cat_data(chunks=TIME_CHUNKS)), ALL_DIAGNOSTICS)

        assert len(_derivatives_mod._PROJECTION_CACHE) == 1


class TestLazyRecipes:
    """Lazy diagnostic recipes, factory memoisation, and explicit CP NaN at the first pressure level"""

    @pytest.fixture
    def scheduler_calls(self, monkeypatch: pytest.MonkeyPatch) -> "list[str]":
        """
        Records every ``Client.compute``, ``Client.get`` and ``Client.persist`` call while the fixture is active

        ``persist`` is counted as well as ``compute`` and ``get``: :attr:`Diagnostic.computed_value` persists, and the
        existing ``scheduler_submissions`` fixture does not see it, so it would report zero submissions for a suite
        that persists every dependency.
        """
        calls: list[str] = []

        def record(name: str, original: "Callable[..., Any]") -> "Callable[..., Any]":
            def counting(self: distributed.Client, *args: "Any", **kwargs: "Any") -> "Any":
                calls.append(name)
                return original(self, *args, **kwargs)

            return counting

        for method_name in ("compute", "get", "persist"):
            original: Callable[..., Any] = getattr(distributed.Client, method_name)
            monkeypatch.setattr(distributed.Client, method_name, record(method_name, original))
        return calls

    def test_lazy_value_is_dask_backed(self) -> None:
        cat_data = make_synthetic_cat_data(chunks={"time": 1})
        diagnostic = DiagnosticFactory(cat_data).create(TurbulenceDiagnostics.RICHARDSON)

        assert is_dask_collection(diagnostic.lazy_value)

    def test_lazy_value_returns_same_object_on_repeated_call(self) -> None:
        diagnostic = DiagnosticFactory(make_synthetic_cat_data()).create(TurbulenceDiagnostics.RICHARDSON)

        assert diagnostic.lazy_value is diagnostic.lazy_value

    def test_computed_value_still_works_after_lazy_value_added(self) -> None:
        diagnostic = DiagnosticFactory(make_synthetic_cat_data()).create(TurbulenceDiagnostics.RICHARDSON)

        assert isinstance(diagnostic.computed_value, xr.DataArray)

    def test_suite_construction_does_not_trigger_compute(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks={"time": 1}))
        suite = DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        for name, diagnostic in suite._diagnostics.items():
            assert is_dask_collection(diagnostic.lazy_value), f"{name} was computed during suite construction"

    def test_factory_create_returns_same_instance_for_same_diagnostic(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data())

        assert factory.create(TurbulenceDiagnostics.RICHARDSON) is factory.create(TurbulenceDiagnostics.RICHARDSON)

    def test_factory_create_returns_different_instance_for_different_diagnostics(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data())

        assert factory.create(TurbulenceDiagnostics.RICHARDSON) is not factory.create(
            TurbulenceDiagnostics.NEGATIVE_RICHARDSON
        )

    def test_suite_construction_triggers_zero_scheduler_submissions(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
    ) -> None:
        # Use persist=False so that CATData._variable() does not call Client.persist on first
        # field access — we want to count submissions from the diagnostic recipe building only.
        assert isinstance(client, distributed.Client)  # activates distributed routing
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))

        DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        assert len(scheduler_calls) == 0

    def test_colson_panofsky_output_has_nan_at_first_pressure_level(self) -> None:
        cp = DiagnosticFactory(make_synthetic_cat_data()).create(TurbulenceDiagnostics.COLSON_PANOFSKY)
        value = cp.computed_value.compute()
        first_level = FIXTURE_PRESSURE_LEVELS[0]

        assert first_level in value["pressure_level"].values, "CP must reindex to the full pressure grid"
        assert value.sel(pressure_level=first_level).isnull().all()

    def test_turbulent_regions_with_nan_diagnostic_values_are_nan_not_false(self) -> None:
        rng = np.random.default_rng(42)
        shape = (3, 3, 2, len(PRESSURE_LEVELS))
        dims = ("latitude", "longitude", "time", "pressure_level")
        coords = {
            "latitude": np.linspace(-10.0, 10.0, 3),
            "longitude": np.linspace(-10.0, 10.0, 3),
            "time": np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]"),
            "pressure_level": np.array(PRESSURE_LEVELS),
        }
        values = rng.random(shape) * 2.0
        values[0, 0, 0, 0] = np.nan
        values[2, 1, 1, 2] = np.nan
        diagnostic = xr.DataArray(values, dims=dims, coords=coords)
        severities = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE]
        thresholds = TurbulenceThresholds(light=0.0, moderate=1.0, severe=2.0)

        by_severity = TurbulentRegionsBySeverity(
            diagnostic, PRESSURE_LEVELS, severities, thresholds, TurbulenceThresholdMode.BOUNDED
        ).execute()
        regions = xr.concat(by_severity, xr.Variable("severity", severities)).compute()

        nan_points = np.isnan(values)
        for severity in severities:
            region = regions.sel(severity=severity)
            np.testing.assert_array_equal(region.isnull().transpose(*dims).values, nan_points)

    def test_turbulent_regions_match_brute_force_on_finite_values(self) -> None:
        rng = np.random.default_rng(42)
        shape = (3, 3, 2, len(PRESSURE_LEVELS))
        dims = ("latitude", "longitude", "time", "pressure_level")
        coords = {
            "latitude": np.linspace(-10.0, 10.0, 3),
            "longitude": np.linspace(-10.0, 10.0, 3),
            "time": np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]"),
            "pressure_level": np.array(PRESSURE_LEVELS),
        }
        values = rng.random(shape) * 2.0
        diagnostic = xr.DataArray(values, dims=dims, coords=coords)
        severities = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE]
        thresholds = TurbulenceThresholds(light=0.0, moderate=1.0, severe=2.0)

        by_severity = TurbulentRegionsBySeverity(
            diagnostic, PRESSURE_LEVELS, severities, thresholds, TurbulenceThresholdMode.BOUNDED
        ).execute()
        regions = xr.concat(by_severity, xr.Variable("severity", severities)).compute()

        bounds = {TurbulenceSeverity.LIGHT: (0.0, 1.0), TurbulenceSeverity.MODERATE: (1.0, 2.0)}
        for severity in severities:
            lower, upper = bounds[severity]
            expected = (values >= lower) & (values < upper)
            actual = regions.sel(severity=severity).transpose(*dims).values
            np.testing.assert_array_equal(actual.astype(bool), expected)

    def test_set_computed_overrides_computed_value_without_touching_lazy_value(self) -> None:
        # set_computed lets a loader inject a pre-loaded array as computed_value without
        # touching the lazy recipe — lazy_value stays intact for any downstream that still needs it.
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        diagnostic = factory.create(TurbulenceDiagnostics.WIND_SPEED)

        lazy_before = diagnostic.lazy_value
        assert is_dask_collection(lazy_before)

        injected = xr.DataArray([1.0, 2.0], dims=["pressure_level"])
        diagnostic.set_computed(injected)

        assert diagnostic.computed_value is injected
        assert diagnostic.lazy_value is lazy_before

    def test_shared_dependency_task_keys_are_identical_across_consumers(self) -> None:
        # Richardson is shared between GradientRichardson (RICHARDSON) and Ncsu1 (NCSU1) via
        # lazy_value.  Both consumers must reference the exact same dask task keys so that a
        # single dask.compute() over all diagnostics executes Richardson only once.
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        richardson = factory.create(TurbulenceDiagnostics.RICHARDSON)
        ncsu1 = factory.create(TurbulenceDiagnostics.NCSU1)

        richardson_graph = richardson.lazy_value.__dask_graph__()
        ncsu1_graph = ncsu1.lazy_value.__dask_graph__()
        assert richardson_graph is not None
        assert ncsu1_graph is not None

        richardson_keys = set(richardson_graph)
        ncsu1_keys = set(ncsu1_graph)

        # Every task that builds Richardson must also appear unchanged in NCSU1's graph.
        assert richardson_keys.issubset(ncsu1_keys), (
            "Richardson task keys missing from NCSU1 graph — shared dependency is not deduplicated"
        )


DATASET_TARGETS = [
    TurbulenceDiagnostics.WIND_SPEED,
    TurbulenceDiagnostics.DEF,
    TurbulenceDiagnostics.RICHARDSON,
    TurbulenceDiagnostics.NCSU1,
]


def _precomputed_dataset() -> xr.Dataset:
    """A numpy-backed Dataset with one variable per entry of ``DATASET_TARGETS``, standing in for a zarr load"""
    rng = np.random.default_rng(42)
    dims = ("latitude", "longitude", "time", "pressure_level")
    coords = {
        "latitude": np.linspace(-10.0, 10.0, 3),
        "longitude": np.linspace(-10.0, 10.0, 3),
        "time": np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]"),
        "pressure_level": np.array([200.0, 225.0, 250.0]),
    }
    return xr.Dataset(
        {
            diagnostic.value: xr.DataArray(rng.random((3, 3, 2, 3)), dims=dims, coords=coords)
            for diagnostic in DATASET_TARGETS
        }
    )


class TestDatasetBackedSuite:
    """
    :class:`DiagnosticSuite` materialises its diagnostics as one :class:`xarray.Dataset`.

    :meth:`DiagnosticSuite.from_factory` and :meth:`DiagnosticSuite.from_dataset` replace the
    ``singledispatchmethod`` ``__init__`` and the ``LoadedFromZarr`` wrapper.
    """

    @pytest.fixture
    def scheduler_calls(self, monkeypatch: pytest.MonkeyPatch) -> "list[str]":
        """
        Records every ``Client.compute``, ``Client.get`` and ``Client.persist`` call while the fixture is active

        Duplicates the fixture of the same name in :class:`TestLazyRecipes`, which is scoped to that class.
        """
        calls: list[str] = []

        def record(name: str, original: "Callable[..., Any]") -> "Callable[..., Any]":
            def counting(self: distributed.Client, *args: "Any", **kwargs: "Any") -> "Any":
                calls.append(name)
                return original(self, *args, **kwargs)

            return counting

        for method_name in ("compute", "get", "persist"):
            original: Callable[..., Any] = getattr(distributed.Client, method_name)
            monkeypatch.setattr(distributed.Client, method_name, record(method_name, original))
        return calls

    def test_from_factory_returns_dataset_backed_suite(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))

        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        assert isinstance(suite.dataset, xr.Dataset)

    def test_dataset_has_one_data_variable_per_requested_diagnostic(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))

        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        assert set(suite.dataset.data_vars) == {diagnostic.value for diagnostic in DATASET_TARGETS}

    @pytest.mark.parametrize("diagnostic", DATASET_TARGETS, ids=lambda d: d.name)
    def test_dataset_variable_matches_independently_computed_diagnostic(
        self,
        diagnostic: TurbulenceDiagnostics,
    ) -> None:
        suite = DiagnosticSuite.from_factory(
            DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1})),
            DATASET_TARGETS,
            persist=False,
        )
        expected = (
            DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
            .create(diagnostic)
            .computed_value.compute()
        )

        actual = suite.dataset[diagnostic.value].transpose(*expected.dims).compute()

        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-6)

    def test_from_dataset_exposes_supplied_dataset(self) -> None:
        supplied = _precomputed_dataset()

        suite = DiagnosticSuite.from_dataset(supplied, DATASET_TARGETS)

        xr.testing.assert_identical(suite.dataset, supplied)

    def test_from_dataset_suite_has_no_diagnostic_recipes(self) -> None:
        suite = DiagnosticSuite.from_dataset(_precomputed_dataset(), DATASET_TARGETS)

        assert suite._diagnostics == {}

    def test_loaded_from_zarr_removed(self) -> None:
        assert not hasattr(_diagnostic_mod, "LoadedFromZarr"), "LoadedFromZarr should have been deleted"

    def test_diagnostic_names_returns_list_of_str(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        names: list[str] = suite.diagnostic_names()

        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
        assert sorted(names) == sorted(diagnostic.value for diagnostic in DATASET_TARGETS)

    def test_diagnostic_names_does_not_materialise_the_dataset(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
    ) -> None:
        """
        Asking what is in the suite must stay free

        Reading the names off ``dataset`` persisted all 28 diagnostics, so a caller that only wanted a file name
        paid for the whole suite.
        """
        assert isinstance(client, distributed.Client)  # activates distributed routing
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks=TIME_CHUNKS))
        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS)

        names = suite.diagnostic_names()

        assert names == [diagnostic.value for diagnostic in DATASET_TARGETS]
        assert scheduler_calls == []

    def test_computed_values_as_dict_matches_dataset_data_vars(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        values: dict[str, xr.DataArray] = suite.computed_values_as_dict()

        assert set(values) == {diagnostic.value for diagnostic in DATASET_TARGETS}
        for name, value in values.items():
            assert isinstance(value, xr.DataArray)
            xr.testing.assert_equal(value, suite.dataset[name])

    def test_as_dataset_returns_same_object_as_dataset(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        assert suite.as_dataset() is suite.dataset

    def test_load_from_zarr_round_trip_matches_original(self, tmp_path: Path) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        original = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)
        zarr_store = original.export_as_zarr(tmp_path)
        zarr_store.close()

        loaded = DiagnosticSuite.load_from_zarr(tmp_path, DATASET_TARGETS)

        assert isinstance(loaded.dataset, xr.Dataset)
        assert set(loaded.dataset.data_vars) == set(original.dataset.data_vars)
        for name in original.dataset.data_vars:
            xr.testing.assert_equal(loaded.dataset[name], original.dataset[name])

    def test_persist_false_dataset_is_dask_backed(self) -> None:
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))

        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS, persist=False)

        assert all(is_dask_collection(variable.data) for variable in suite.dataset.data_vars.values())

    def test_building_from_factory_submits_nothing_to_scheduler(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))

        DiagnosticSuite.from_factory(factory, DATASET_TARGETS)

        assert scheduler_calls == []

    def test_reading_dataset_persists_full_suite_exactly_once(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        factory = DiagnosticFactory(make_synthetic_cat_data(persist=False, chunks={"time": 1}))
        suite = DiagnosticSuite.from_factory(factory, DATASET_TARGETS)

        _ = suite.dataset
        _ = suite.dataset

        assert scheduler_calls.count("persist") == 1

    def test_mismatched_grids_raise_rather_than_silently_padding(self) -> None:
        # dataset property uses join="exact" so any mismatch raises loudly.
        # Test this directly: two arrays with different latitude coords must raise.
        rng = np.random.default_rng(0)
        coords_a = {"latitude": np.array([0.0, 1.0]), "pressure_level": np.array([200.0])}
        coords_b = {"latitude": np.array([0.0, 2.0]), "pressure_level": np.array([200.0])}
        a = xr.DataArray(rng.random((2, 1)), dims=["latitude", "pressure_level"], coords=coords_a, name="a")
        b = xr.DataArray(rng.random((2, 1)), dims=["latitude", "pressure_level"], coords=coords_b, name="b")

        with pytest.raises((ValueError, xr.MergeError)):
            xr.merge([a, b], join="exact", compat="no_conflicts")

    def test_from_dataset_raises_when_requested_diagnostic_is_absent(self) -> None:
        supplied = _precomputed_dataset()  # only has DATASET_TARGETS variables
        absent = [TurbulenceDiagnostics.BROWN1]  # not in _precomputed_dataset()

        with pytest.raises(ValueError, match="brown1"):
            DiagnosticSuite.from_dataset(supplied, absent)


_FIELD_DIMS = ("latitude", "longitude", "time", "pressure_level")
TWO_SEVERITIES = [TurbulenceSeverity.LIGHT, TurbulenceSeverity.MODERATE]
TWO_SEVERITY_BOUNDS = {TurbulenceSeverity.LIGHT: (0.0, 1.0), TurbulenceSeverity.MODERATE: (1.0, 2.0)}
REGION_THRESHOLDS = TurbulenceThresholds(light=0.0, moderate=1.0, severe=2.0)


def _field_coords() -> dict[str, np.ndarray]:
    return {
        "latitude": np.linspace(-10.0, 10.0, 3),
        "longitude": np.linspace(-10.0, 10.0, 3),
        "time": np.array(["2018-08-01T00", "2018-08-01T06"], dtype="datetime64[h]"),
        "pressure_level": np.array(PRESSURE_LEVELS),
    }


def _dask_field(values: np.ndarray) -> xr.DataArray:
    """Dask-backed (latitude, longitude, time, pressure_level) DataArray on the small post-processor test grid"""
    return xr.DataArray(da.from_array(values, chunks=(2, 2, 1, 2)), dims=_FIELD_DIMS, coords=_field_coords())  # pyright: ignore[reportArgumentType]


def _calibration_values() -> np.ndarray:
    """Seeded calibration values including a negative and a zero, which the log-based statistics must exclude"""
    values = np.random.default_rng(42).random((3, 3, 2, len(PRESSURE_LEVELS))) * 2.0
    values[0, 0, 0, 0] = -0.5
    values[1, 2, 1, 3] = 0.0
    return values


def _region_values(*, with_nan: bool) -> np.ndarray:
    values = np.random.default_rng(42).random((3, 3, 2, len(PRESSURE_LEVELS))) * 2.0
    if with_nan:
        values[0, 0, 0, 0] = np.nan
        values[2, 1, 1, 2] = np.nan
    return values


def _brute_force_region_masks(values: np.ndarray) -> list[np.ndarray]:
    """
    Per-severity masks in ``(latitude, longitude, time, pressure_level)`` order: 1.0 inside the severity's bounds,
    0.0 outside them, and NaN wherever the diagnostic itself is NaN
    """
    masks: list[np.ndarray] = []
    for severity in TWO_SEVERITIES:
        lower, upper = TWO_SEVERITY_BOUNDS[severity]
        mask = ((values >= lower) & (values < upper)).astype(float)
        mask[np.isnan(values)] = np.nan
        masks.append(mask)
    return masks


def _as_tuple(thresholds: TurbulenceThresholds) -> tuple[float, float, float]:
    assert thresholds.light is not None
    assert thresholds.moderate is not None
    assert thresholds.severe is not None
    return (thresholds.light, thresholds.moderate, thresholds.severe)


def _assert_histograms_match(actual: "HistogramData", expected: "HistogramData") -> None:
    np.testing.assert_allclose(actual.hist_values, expected.hist_values, rtol=1e-10)
    np.testing.assert_allclose(actual.bins, expected.bins, rtol=1e-10)
    assert actual.mean == pytest.approx(expected.mean, rel=1e-10)
    assert actual.variance == pytest.approx(expected.variance, rel=1e-10)


@pytest.fixture
def forbid_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes any ``compute`` on an xarray or dask array raise, so that building a graph is shown to be lazy"""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("graph construction must not compute")

    monkeypatch.setattr(xr.DataArray, "compute", forbidden)
    monkeypatch.setattr(da.Array, "compute", forbidden)


class TestIntensityThresholdsPrepareFinalise:
    """``TurbulenceIntensityThresholds`` exposes the raw dask percentiles from ``build_lazy_collection()``"""

    def test_build_lazy_collection_returns_dask_collection(self) -> None:
        processor = TurbulenceIntensityThresholds(PERCENTILE_CONFIG, _dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_lazy_collection_triggers_no_compute(self, forbid_compute: None) -> None:
        processor = TurbulenceIntensityThresholds(PERCENTILE_CONFIG, _dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_result_of_computed_collection_matches_execute(self) -> None:
        processor = TurbulenceIntensityThresholds(PERCENTILE_CONFIG, _dask_field(_calibration_values()))

        from_split = processor.build_result(dask_compute(processor.build_lazy_collection())[0])
        from_execute = processor.execute()

        assert _as_tuple(from_split) == pytest.approx(_as_tuple(from_execute))

    def test_execute_still_matches_numpy_percentiles(self) -> None:
        values = _calibration_values()
        processor = TurbulenceIntensityThresholds(PERCENTILE_CONFIG, _dask_field(values))

        actual = processor.execute()

        expected = np.percentile(values, [90.0, 95.0, 99.0])
        np.testing.assert_allclose(_as_tuple(actual), expected, rtol=5e-2)  # tdigest is approximate


class TestHistogramPrepareFinalise:
    """``DiagnosticHistogramDistribution`` exposes its dask collections from ``build_lazy_collection()``"""

    def test_build_lazy_collection_returns_dask_collection(self) -> None:
        processor = DiagnosticHistogramDistribution(_dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_lazy_collection_triggers_no_compute(self, forbid_compute: None) -> None:
        processor = DiagnosticHistogramDistribution(_dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_lazy_collection_is_four_scalars_not_the_field(self) -> None:
        """
        The batched caller materialises one of these per diagnostic, so it must not carry the values themselves

        Returning the log of every positive value put a full copy of every diagnostic in the client's memory.
        """
        field = _dask_field(_calibration_values())
        processor = DiagnosticHistogramDistribution(field)

        collection = processor.build_lazy_collection()

        assert collection.shape == (NUM_HISTOGRAM_STATISTICS,)
        assert collection.compute().size == NUM_HISTOGRAM_STATISTICS

    def test_build_result_of_computed_collection_matches_execute(self) -> None:
        processor = DiagnosticHistogramDistribution(_dask_field(_calibration_values()))

        from_split = processor.build_result(dask_compute(processor.build_lazy_collection())[0])
        from_execute = processor.execute()

        _assert_histograms_match(from_split, from_execute)

    def test_execute_still_matches_numpy_histogram_of_log_positive_values(self) -> None:
        values = _calibration_values()
        processor = DiagnosticHistogramDistribution(_dask_field(values))

        actual = processor.execute()

        log_positive = np.log(values[values > 0])
        expected_hist, expected_bins = np.histogram(
            log_positive,
            bins=50,
            range=(float(log_positive.min()), float(log_positive.max())),
            density=True,
        )
        np.testing.assert_allclose(actual.hist_values, expected_hist, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(actual.bins, expected_bins, rtol=1e-6)
        assert actual.mean == pytest.approx(float(log_positive.mean()), rel=1e-6)
        assert actual.variance == pytest.approx(float(log_positive.var()), rel=1e-6)


class TestDistributionParametersPrepareFinalise:
    """``ComputeDistributionParametersForEDR`` exposes its lazy dask collections from ``build_lazy_collection()``"""

    def test_build_lazy_collection_returns_dask_collection(self) -> None:
        processor = ComputeDistributionParametersForEDR(_dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_lazy_collection_triggers_no_compute(self, forbid_compute: None) -> None:
        processor = ComputeDistributionParametersForEDR(_dask_field(_calibration_values()))

        assert is_dask_collection(processor.build_lazy_collection())

    def test_build_result_of_computed_collection_matches_execute(self) -> None:
        processor = ComputeDistributionParametersForEDR(_dask_field(_calibration_values()))

        from_split = processor.build_result(dask_compute(processor.build_lazy_collection())[0])
        from_execute = processor.execute()

        assert from_split.mean == pytest.approx(from_execute.mean, rel=1e-12)
        assert from_split.variance == pytest.approx(from_execute.variance, rel=1e-12)

    def test_execute_matches_brute_force_mean_and_variance_of_log_positive_values(self) -> None:
        values = _calibration_values()
        processor = ComputeDistributionParametersForEDR(_dask_field(values))

        actual = processor.execute()

        log_positive = np.log(values[values > 0])
        assert actual.mean == pytest.approx(float(log_positive.mean()), rel=1e-10)
        assert actual.variance == pytest.approx(float(log_positive.var()), rel=1e-10)


class TestTurbulentRegionsBySeverityLazy:
    """
    ``TurbulentRegionsBySeverity.execute()`` always returns the lazy per-severity list; callers concatenate and compute
    """

    def test_has_parent_class_attribute_removed(self) -> None:
        assert not hasattr(TurbulentRegionsBySeverity, "_has_parent")

    def test_has_parent_not_set_on_instances(self) -> None:
        regions = TurbulentRegionsBySeverity(
            _dask_field(_region_values(with_nan=False)),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        assert not hasattr(regions, "_has_parent")

    def test_has_parent_kwarg_is_rejected(self) -> None:
        extra: dict[str, Any] = {"has_parent": True}

        with pytest.raises(TypeError):
            TurbulentRegionsBySeverity(
                _dask_field(_region_values(with_nan=False)),
                PRESSURE_LEVELS,
                TWO_SEVERITIES,
                REGION_THRESHOLDS,
                TurbulenceThresholdMode.BOUNDED,
                **extra,
            )

    def test_execute_returns_list(self) -> None:
        regions = TurbulentRegionsBySeverity(
            _dask_field(_region_values(with_nan=False)),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        result = regions.execute()

        assert isinstance(result, list)
        assert len(result) == len(TWO_SEVERITIES)

    def test_execute_list_elements_are_float32(self) -> None:
        """
        The caller holds one mask per severity per diagnostic, so the narrowest dtype that carries NaN is used

        ``xr.where(..., np.nan)`` on a boolean promotes to float64, doubling every mask.
        """
        regions = TurbulentRegionsBySeverity(
            _dask_field(_region_values(with_nan=True)),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        result = regions.execute()

        assert all(item.dtype == np.float32 for item in result)

    def test_execute_list_elements_are_lazy_dask_collections(self) -> None:
        regions = TurbulentRegionsBySeverity(
            _dask_field(_region_values(with_nan=False)),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        result = regions.execute()

        assert isinstance(result, list)
        assert all(isinstance(item, xr.DataArray) and is_dask_collection(item) for item in result)

    def test_execute_triggers_no_compute(self, forbid_compute: None) -> None:
        regions = TurbulentRegionsBySeverity(
            _dask_field(_region_values(with_nan=True)),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        result = regions.execute()

        assert isinstance(result, list)

    @pytest.mark.parametrize("with_nan", [False, True], ids=["finite", "with_nan"])
    def test_concatenated_lazy_result_matches_brute_force(self, with_nan: bool) -> None:
        values = _region_values(with_nan=with_nan)
        regions = TurbulentRegionsBySeverity(
            _dask_field(values),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        )

        result = regions.execute()
        assert isinstance(result, list)
        concatenated = xr.concat(result, xr.Variable("severity", TWO_SEVERITIES)).compute()

        expected = _brute_force_region_masks(values)
        for index, severity in enumerate(TWO_SEVERITIES):
            actual = concatenated.sel(severity=severity).transpose(*_FIELD_DIMS).values
            np.testing.assert_array_equal(actual, expected[index])

    def test_probability_by_severity_still_matches_brute_force(self) -> None:
        values = _region_values(with_nan=False)
        probabilities = TurbulenceProbabilityBySeverity(
            _dask_field(values),
            PRESSURE_LEVELS,
            TWO_SEVERITIES,
            REGION_THRESHOLDS,
            TurbulenceThresholdMode.BOUNDED,
        ).execute()

        for _index, severity in enumerate(TWO_SEVERITIES):
            lower, upper = TWO_SEVERITY_BOUNDS[severity]
            expected = ((values >= lower) & (values < upper)).mean(axis=2) * 100
            actual = probabilities.sel(severity=severity).transpose("latitude", "longitude", "pressure_level").values
            np.testing.assert_allclose(actual, expected, rtol=1e-10)


def _round_trips(calls: list[str]) -> int:
    """
    Scheduler round-trips for ``compute``-style reductions

    ``dask.compute`` routes through ``Client.get`` when a distributed client is active, so ``DataArray.compute()`` is
    recorded as ``get``, not ``compute``. Both are counted here; ``persist`` is counted separately.
    """
    return calls.count("compute") + calls.count("get")


def _assert_values_match(actual: xr.DataArray, expected: xr.DataArray, *, exact: bool) -> None:
    """Compare two DataArrays by value, ignoring the dimension order of ``expected``"""
    expected = expected.transpose(*actual.dims)
    if exact:
        np.testing.assert_array_equal(actual.values, expected.values)
    else:
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-10)


class TestBatchedCalibration:
    """
    Calibration and evaluation reductions batch every diagnostic into one scheduler round-trip per method

    Values are checked against the per-diagnostic ``execute()`` path, which is unchanged by the batching.
    """

    @pytest.fixture
    def scheduler_calls(self, monkeypatch: pytest.MonkeyPatch) -> "list[str]":
        """Records every ``Client.compute``, ``Client.get`` and ``Client.persist`` call while the fixture is active"""
        calls: list[str] = []

        def record(name: str, original: "Callable[..., Any]") -> "Callable[..., Any]":
            def counting(self: distributed.Client, *args: "Any", **kwargs: "Any") -> "Any":
                calls.append(name)
                return original(self, *args, **kwargs)

            return counting

        for method_name in ("compute", "get", "persist"):
            original: Callable[..., Any] = getattr(distributed.Client, method_name)
            monkeypatch.setattr(distributed.Client, method_name, record(method_name, original))
        return calls

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_thresholds_issues_one_round_trip(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
        chunks: dict[str, int],
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        suite = _calibration_suite(chunks)

        suite.compute_thresholds(PERCENTILE_CONFIG)

        assert _round_trips(scheduler_calls) == 1

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_distribution_parameters_issues_two_round_trips(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
        chunks: dict[str, int],
    ) -> None:
        """
        Two passes for the whole suite, not two per diagnostic

        A histogram's range is the minimum and maximum of its own values, so the counts cannot be built until the
        statistics have come back. The alternative — one pass that returns the values themselves — puts every
        diagnostic's full field in the client's memory at once.
        """
        assert isinstance(client, distributed.Client)  # activates distributed routing
        suite = _calibration_suite(chunks)

        suite.compute_distribution_parameters()

        assert _round_trips(scheduler_calls) == HISTOGRAM_ROUND_TRIPS

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_probabilities_issues_one_persist(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
        chunks: dict[str, int],
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        suite = _evaluation_suite(chunks)
        scheduler_calls.clear()  # _evaluation_suite computes calibration thresholds; not part of the count

        _ = suite.probabilities

        assert scheduler_calls.count("persist") == 1

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_turbulent_regions_issues_one_round_trip(
        self,
        client: distributed.Client,
        scheduler_calls: "list[str]",
        chunks: dict[str, int],
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        suite = _evaluation_suite(chunks)
        scheduler_calls.clear()  # _evaluation_suite computes calibration thresholds; not part of the count

        suite.compute_turbulent_regions()

        assert _round_trips(scheduler_calls) == 1

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_thresholds_values_match_per_diagnostic_execute(self, chunks: dict[str, int]) -> None:
        batched = _calibration_suite(chunks).compute_thresholds(PERCENTILE_CONFIG)
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))

        assert len(batched) == len(ALL_DIAGNOSTICS)
        for diagnostic_name in ALL_DIAGNOSTICS:
            diagnostic = factory.create(diagnostic_name).computed_value
            reference = TurbulenceIntensityThresholds(PERCENTILE_CONFIG, diagnostic).execute()
            actual = batched[diagnostic_name]
            assert actual.light == pytest.approx(reference.light, rel=1e-12)
            assert actual.moderate == pytest.approx(reference.moderate, rel=1e-12)
            assert actual.severe == pytest.approx(reference.severe, rel=1e-12)

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_distribution_parameters_values_match_per_diagnostic_execute(self, chunks: dict[str, int]) -> None:
        batched = _calibration_suite(chunks).compute_distribution_parameters()
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))

        assert len(batched) == len(ALL_DIAGNOSTICS)
        for diagnostic_name in ALL_DIAGNOSTICS:
            diagnostic = factory.create(diagnostic_name).computed_value
            reference = DiagnosticHistogramDistribution(diagnostic).execute()
            actual = batched[diagnostic_name]
            assert actual.mean == pytest.approx(reference.mean, rel=1e-12)
            assert actual.variance == pytest.approx(reference.variance, rel=1e-12)
            np.testing.assert_allclose(actual.hist_values, reference.hist_values, rtol=1e-10)
            np.testing.assert_allclose(actual.bins, reference.bins, rtol=1e-10)

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_probabilities_values_match_per_diagnostic_execute(self, chunks: dict[str, int]) -> None:
        thresholds = _calibration_suite(chunks).compute_thresholds(PERCENTILE_CONFIG)
        batched = _evaluation_suite(chunks).probabilities
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))

        assert len(batched) == len(ALL_DIAGNOSTICS)
        for diagnostic_name in ALL_DIAGNOSTICS:
            diagnostic = factory.create(diagnostic_name).computed_value
            reference = (
                TurbulenceProbabilityBySeverity(
                    diagnostic,
                    PRESSURE_LEVELS,
                    SEVERITIES,
                    thresholds[diagnostic_name],
                    TurbulenceThresholdMode.BOUNDED,
                )
                .execute()
                .compute()
            )
            _assert_values_match(batched[diagnostic_name], reference, exact=False)

    @pytest.mark.parametrize("chunks", CHUNKINGS)
    def test_compute_turbulent_regions_values_match_per_diagnostic_execute(self, chunks: dict[str, int]) -> None:
        thresholds = _calibration_suite(chunks).compute_thresholds(PERCENTILE_CONFIG)
        batched = _evaluation_suite(chunks).compute_turbulent_regions()
        factory = DiagnosticFactory(make_synthetic_cat_data(chunks=chunks))

        assert len(batched) == len(ALL_DIAGNOSTICS)
        for diagnostic_name in ALL_DIAGNOSTICS:
            diagnostic = factory.create(diagnostic_name).computed_value
            per_severity = TurbulentRegionsBySeverity(
                diagnostic,
                PRESSURE_LEVELS,
                SEVERITIES,
                thresholds[diagnostic_name],
                TurbulenceThresholdMode.BOUNDED,
            ).execute()
            reference = xr.concat(per_severity, xr.Variable("severity", SEVERITIES)).compute()
            _assert_values_match(batched[diagnostic_name], reference, exact=True)
