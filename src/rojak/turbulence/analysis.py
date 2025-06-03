import itertools
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Hashable, Mapping

import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray

from rojak.core.analysis import PostProcessor
from rojak.orchestrator.configuration import TurbulenceSeverityPercentileConfig
from rojak.turbulence.diagnostic import DiagnosticName
from rojak.utilities.types import Limits

type IntensityName = str
type IntensityValues = dict[IntensityName, float]


class TurbulenceIntensityThresholds(PostProcessor):
    _threshold_configs: list[TurbulenceSeverityPercentileConfig]
    _computed_diagnostic: xr.DataArray

    def __init__(
        self, threshold_config: list[TurbulenceSeverityPercentileConfig], computed_diagnostic: xr.DataArray
    ) -> None:
        self._threshold_configs = threshold_config
        self._computed_diagnostic = computed_diagnostic

    def _compute_percentiles(self, target_percentiles: list[float]) -> NDArray:
        if is_dask_collection(self._computed_diagnostic):
            # flattened_array = da.asarray(self._computed_diagnostic.data, chunks="auto").flatten()
            flattened_array = da.asarray(self._computed_diagnostic, chunks="auto").flatten()
            return da.percentile(flattened_array, target_percentiles, internal_method="dask").compute()
        return np.percentile(self._computed_diagnostic.stack(all=[...]), target_percentiles)

    def _compute_lower_bounds(self) -> NDArray:
        lower_bounds = [config.lower_bound for config in self._threshold_configs]
        return self._compute_percentiles(lower_bounds)

    def execute(self) -> IntensityValues:
        percentiles = self._compute_lower_bounds()
        return {str(config.name): float(percentile) for config, percentile in zip(self._threshold_configs, percentiles)}


@dataclass
class HistogramData:
    hist_values: list[float]
    bins: list[float]
    mean: float
    variance: float

    def __init__(self, hist_values: np.ndarray, bins: np.ndarray, mean: float, variance: float) -> None:
        self.hist_values = hist_values.tolist()
        self.bins = bins.tolist()
        self.mean = float(mean)
        self.variance = float(variance)

    def as_json_dict(self) -> dict:
        return {
            "density": self.hist_values,
            "bin_edges": self.bins,
            "mean": self.mean,
            "variance": self.variance,
        }

    # def export_to_json(self, plots_directory: str, diagnostic_name: str) -> None:
    #     from json import dump
    #     histogram_data = self.as_json_dict()
    #     with open(f"{plots_directory}/{diagnostic_name.replace(' ', '_')}.json", "w") as outfile:
    #         dump(histogram_data, outfile, indent=4)

    # def create_single_plot(self, plots_directory: str, diagnostic_name: str):
    #     fig = plt.figure()
    #     ax: Axes = fig.add_subplot(1, 1, 1)
    #     self.plot_on_axis(ax)
    #     ax.set_ylabel("Probability (%)")
    #     ax.set_xlabel(diagnostic_name)
    #     fig.tight_layout()
    #     plt.savefig(f"{plots_directory}/{diagnostic_name}.{IMAGE_FORMAT}")
    #
    # def plot_on_axis(self, ax):
    #     ax.hist(self.bins[:-1], self.bins, weights=self.hist_values)
    #     x_coord_values: np.ndarray = np.linspace(self.bins[0], self.bins[-1], 100)
    #     ax.plot(x_coord_values, stats.norm(loc=self.mean, scale=np.sqrt(self.variance)).pdf(x_coord_values))

    def filter_insignificant_bins(self, minimum_value: float = 1e-6) -> "HistogramData":
        values: np.ndarray = np.asarray(self.hist_values)
        mask: np.ndarray = values >= minimum_value
        return HistogramData(values[mask], np.asarray(self.bins)[np.append(mask, [True])], self.mean, self.variance)

    def __str__(self) -> str:
        return (
            f"HistogramData(hist_values={self.hist_values}, bins={self.bins}, "
            f"mean={self.mean}, variance={self.variance})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HistogramData):
            return (
                self.hist_values == list(other.hist_values)
                and self.bins == other.bins
                and self.mean == other.mean
                and self.variance == other.variance
            )
        return False


class DiagnosticHistogramDistribution(PostProcessor):
    _computed_diagnostic: xr.DataArray
    _NUM_HIST_BINS: int = 50

    def __init__(self, computed_diagnostic: xr.DataArray, num_hist_bins: int | None = None) -> None:
        self._computed_diagnostic = computed_diagnostic
        if num_hist_bins is not None:
            self._NUM_HIST_BINS = num_hist_bins

    def _serial_execution(self) -> HistogramData:
        flattened_array = self._computed_diagnostic.stack(all=[...])
        flattened_array = flattened_array[flattened_array > 0]
        log_of_diagnostic = np.log(flattened_array)
        min_and_max = np.percentile(flattened_array, [0, 100])
        h, bins = np.histogram(
            log_of_diagnostic,
            bins=self._NUM_HIST_BINS,
            range=(float(min_and_max[0]), float(min_and_max[1])),
            density=True,
        )
        return HistogramData(h, bins, float(np.mean(log_of_diagnostic)), float(np.var(log_of_diagnostic)))

    def _parallel_execution(self) -> HistogramData:
        flattened_array = da.asarray(self._computed_diagnostic).flatten()
        flattened_array = flattened_array[flattened_array > 0]
        log_of_diagnostic = da.log(flattened_array)
        min_and_max = da.percentile(flattened_array, [0, 100]).compute()
        h, bins = da.histogram(  # pyright: ignore [reportGeneralTypeIssues]
            log_of_diagnostic, bins=self._NUM_HIST_BINS, range=(min_and_max[0], min_and_max[1]), density=True
        )
        return HistogramData(
            hist_values=h.compute(),
            bins=bins,
            mean=da.mean(log_of_diagnostic).compute(),
            variance=da.var(log_of_diagnostic).compute(),
        )

    def execute(self) -> HistogramData:
        if is_dask_collection(self._computed_diagnostic):
            return self._parallel_execution()
        return self._serial_execution()


class CorrelationBetweenDiagnostics(PostProcessor):
    _diagnostic_names: list[DiagnosticName]
    _computed_indices: dict[DiagnosticName, xr.DataArray]
    _sel_condition: Mapping[str, Any]

    def __init__(self, computed_indices: dict[DiagnosticName, xr.DataArray], sel_condition: Mapping[str, Any]) -> None:
        self._computed_indices = computed_indices
        self._diagnostic_names = list(self._computed_indices.keys())
        self._sel_condition = sel_condition

    def execute(self) -> xr.DataArray:
        num_diagnostics: int = len(self._diagnostic_names)
        corr_btw_diagnostics: xr.DataArray = xr.DataArray(
            data=np.ones((num_diagnostics, num_diagnostics)),
            dims=("diagnostic1", "diagnostic2"),
            coords={"diagnostic1": self._diagnostic_names, "diagnostic2": self._diagnostic_names},
        )
        for first_diagnostic, second_diagnostic in itertools.combinations(self._diagnostic_names, 2):
            this_corr: xr.DataArray = (
                xr.corr(
                    self._computed_indices[first_diagnostic].sel(self._sel_condition),
                    self._computed_indices[second_diagnostic].sel(self._sel_condition),
                )
                .stack(flat=[...])
                .compute()
            )
            corr_btw_diagnostics.loc[{"diagnostic1": first_diagnostic, "diagnostic2": second_diagnostic}] = this_corr
            corr_btw_diagnostics.loc[{"diagnostic1": second_diagnostic, "diagnostic2": first_diagnostic}] = this_corr

        return corr_btw_diagnostics


class Hemisphere(StrEnum):
    GLOBAL = "global"
    NORTH = "north"
    SOUTH = "south"


class LatitudinalRegion(StrEnum):
    FULL = "full"
    EXTRATROPICS = "extratropics"
    TROPICS = "tropics"


class LatitudinalCorrelationBetweenDiagnostics(PostProcessor):
    _computed_indices: Mapping[DiagnosticName, xr.DataArray]
    _hemispheres: list[Hemisphere]
    _latitudinal_regions: list[LatitudinalRegion]
    _diagnostic_names: list[DiagnosticName]
    _sel_condition: Mapping[str, Any]

    def __init__(
        self,
        computed_indices: Mapping[DiagnosticName, xr.DataArray],
        sel_condition: Mapping[str, Any],
        hemispheres: list[Hemisphere] | None = None,
        regions: list[LatitudinalRegion] | None = None,
    ) -> None:
        self._computed_indices = computed_indices
        self._diagnostic_names = list(computed_indices.keys())
        if hemispheres is None:
            self._hemispheres = [Hemisphere.GLOBAL, Hemisphere.NORTH, Hemisphere.SOUTH]
        else:
            assert 1 <= len(hemispheres) <= 3
            self._hemispheres = hemispheres
        if regions is None:
            self._regions = [LatitudinalRegion.FULL, LatitudinalRegion.EXTRATROPICS, LatitudinalRegion.TROPICS]
        else:
            assert 1 <= len(regions) <= 3
            self._regions = regions
        self._sel_condition = sel_condition

    @staticmethod
    def _apply_region_filter(array: xr.DataArray, hemisphere: Hemisphere, region: LatitudinalRegion) -> xr.DataArray:
        extratropic_latitudes: Limits = Limits(lower=25, upper=65)
        entire_tropics: Limits = Limits(lower=-25, upper=25)
        half_tropics: Limits = Limits(lower=0, upper=25)
        assert "latitude" in array.coords
        if hemisphere == "global":
            if region == "full":
                return array
            if region == "tropics":
                return array.where(
                    ((array["latitude"] > entire_tropics.lower) & (array["latitude"] < entire_tropics.upper)), drop=True
                )
            # extratropics
            return array.where(
                (
                    (
                        (array["latitude"] > extratropic_latitudes.lower)
                        & (array["latitude"] < extratropic_latitudes.upper)
                    )
                    | (
                        (array["latitude"] > -extratropic_latitudes.upper)
                        & (array["latitude"] < -extratropic_latitudes.lower)
                    )
                ),
                drop=True,
            )
        if hemisphere == "north":
            if region == "full":
                return array.where(array["latitude"] > 0, drop=True)

            target_latitudes: Limits = half_tropics if region == "tropics" else extratropic_latitudes
            return array.where(
                ((array["latitude"] > target_latitudes.lower) & (array["latitude"] < target_latitudes.upper)), drop=True
            )
        # south
        if region == "full":
            return array.where(array["latitude"] < 0, drop=True)

        target_latitudes: Limits = half_tropics if region == "tropics" else extratropic_latitudes
        return array.where(
            ((array["latitude"] > -target_latitudes.upper) & (array["latitude"] < -target_latitudes.lower)),
            drop=True,
        )

    def execute(self) -> xr.DataArray:
        possible_coordinates: set[str] = {"latitude", "longitude", "valid_time"}
        remaining_coordinates: list[Hashable] = [
            coord for coord in list(self._computed_indices.values())[0].dims if coord in possible_coordinates
        ]
        num_diagnostics: int = len(self._diagnostic_names)
        correlations: xr.DataArray = xr.DataArray(
            data=np.ones((num_diagnostics, num_diagnostics, len(self._hemispheres), len(self._regions))),
            dims=("diagnostic1", "diagnostic2", "hemisphere", "region"),
            coords={
                "diagnostic1": self._diagnostic_names,
                "diagnostic2": self._diagnostic_names,
                "hemisphere": self._hemispheres,
                "region": self._regions,
            },
        )
        for first_diagnostic, second_diagnostic in itertools.combinations(self._diagnostic_names, 2):
            for hemisphere, region in itertools.product(
                self._hemispheres, self._regions
            ):  # Hemisphere, LatitudinalRegion
                this_correlation: xr.DataArray = xr.corr(
                    self._apply_region_filter(self._computed_indices[first_diagnostic], hemisphere, region)
                    .sel(self._sel_condition)
                    .stack(flat=remaining_coordinates)
                    .reset_coords(drop=True),
                    self._apply_region_filter(self._computed_indices[second_diagnostic], hemisphere, region)
                    .sel(self._sel_condition)
                    .stack(flat=remaining_coordinates)
                    .reset_coords(drop=True),
                ).compute()
                correlations.loc[
                    {
                        "diagnostic1": first_diagnostic,
                        "diagnostic2": second_diagnostic,
                        "hemisphere": hemisphere,
                        "region": region,
                    }
                ] = this_correlation
                correlations.loc[
                    {
                        "diagnostic1": second_diagnostic,
                        "diagnostic2": first_diagnostic,
                        "hemisphere": hemisphere,
                        "region": region,
                    }
                ] = this_correlation
        return correlations
