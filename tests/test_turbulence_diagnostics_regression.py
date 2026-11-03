import json
from pathlib import Path
from typing import TYPE_CHECKING

import distributed
import numpy as np
import pytest
import xarray as xr
from dask.base import compute as dask_compute
from dask.highlevelgraph import HighLevelGraph

from rojak.core.data import CATData
from rojak.core.derivatives import VelocityDerivative
from rojak.orchestrator.configuration import (
    TurbulenceDiagnostics,
    TurbulenceSeverity,
    TurbulenceThresholdMode,
    TurbulenceThresholds,
)
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
TIME_CHUNKED = pytest.param({"time": 1}, id="time_chunked")
LAT_LON_CHUNKED = pytest.param(
    {"latitude": -1, "longitude": -1, "time": 1, "pressure_level": -1},
    id="latlon_chunked",
)
CHUNKINGS = [TIME_CHUNKED, LAT_LON_CHUNKED]

# The synthetic fixture has 4 pressure levels (full set).
FIXTURE_PRESSURE_LEVELS = [175.0, 200.0, 225.0, 250.0]
# PRESSURE_LEVELS is the subset used for TurbulentRegionsBySeverity.  ColsonPanofsky lives on
# pressure_level[1:] so it is missing 175.0 pre-Phase 02.  Phase 02 adds an explicit reindex;
# at that point this constant can be extended back to FIXTURE_PRESSURE_LEVELS.
PRESSURE_LEVELS = [200.0, 225.0, 250.0]
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

    def test_full_suite_task_count_matches_baseline(self, load_cat_data: "Callable") -> None:
        baseline = self._load_baseline()
        factory = DiagnosticFactory(load_cat_data(None, with_chunks=True))
        suite = DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        assert count_tasks(*suite.computed_values_as_dict().values()) == baseline["full_suite_tasks"]

    def test_building_suite_submits_baseline_number_of_scheduler_calls(
        self,
        client: distributed.Client,
        load_cat_data: "Callable",
        scheduler_submissions: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        baseline = self._load_baseline()
        factory = DiagnosticFactory(load_cat_data(None, with_chunks=True))

        DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        assert len(scheduler_submissions) == baseline["build_submissions"]

    def test_round_trip_submissions_match_baseline(
        self,
        client: distributed.Client,
        load_cat_data: "Callable",
        scheduler_submissions: "list[str]",
    ) -> None:
        assert isinstance(client, distributed.Client)  # activates distributed routing
        baseline = self._load_baseline()
        factory = DiagnosticFactory(load_cat_data(None, with_chunks=True))
        suite = DiagnosticSuite(factory, ALL_DIAGNOSTICS)

        results = dask_compute(*suite.computed_values_as_dict().values())

        assert all(isinstance(result, xr.DataArray) for result in results)
        assert len(scheduler_submissions) == baseline["round_trip_submissions"]
