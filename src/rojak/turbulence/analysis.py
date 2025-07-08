#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Hashable, Mapping, NamedTuple, assert_never

import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass

from rojak.core.analysis import PostProcessor
from rojak.core.distributed_tools import blocking_wait_futures
from rojak.orchestrator.configuration import TurbulenceSeverity, TurbulenceThresholdMode, TurbulenceThresholds
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from rojak.turbulence.diagnostic import DiagnosticName

type IntensityName = str
type IntensityValues = dict[IntensityName, float]

ClimatologicalEDRConstants = NamedTuple("ClimatologicalEDRConstants", [("c1", float), ("c2", float)])
# From Sharman 2017
OVERALL_CLIMATOLOGICAL_PARAMETER = ClimatologicalEDRConstants(-2.572, 0.5067)


class TurbulenceIntensityThresholds(PostProcessor):
    """
    Computes threshold diagnostic value for each turbulence intensity using percentiles

    To determine if turbulence of a given intensity is encountered, the threshold value for said intensity must first
    be calculated for each diagnostics. Using the specified percentile values, these thresholds are computed to be
    on the calibration dataset in accordance to the methodology detailed in [Williams2017]_
    """

    _percentile_config: TurbulenceThresholds
    _computed_diagnostic: xr.DataArray

    def __init__(self, threshold_config: TurbulenceThresholds, computed_diagnostic: xr.DataArray) -> None:
        self._percentile_config = threshold_config
        self._computed_diagnostic = computed_diagnostic

    def _compute_percentiles(self, target_percentiles: NDArray) -> NDArray:
        if is_dask_collection(self._computed_diagnostic):
            # flattened_array = da.asarray(self._computed_diagnostic.data, chunks="auto").flatten()
            flattened_array = da.asarray(self._computed_diagnostic, chunks="auto").flatten()
            # Must use tdigest method as internal dask version gives incorrect results
            return da.percentile(flattened_array, target_percentiles, internal_method="tdigest").compute()
        return np.percentile(self._computed_diagnostic.stack(all=[...]), target_percentiles)

    def _find_index_without_nones(self) -> list[int | None]:
        new_index: int = 0
        new_list: list[int | None] = []
        for item in self._percentile_config.all_severities:
            if item is None:
                new_list.append(None)
            else:
                new_list.append(new_index)
                new_index += 1
        return new_list

    def execute(self) -> TurbulenceThresholds:
        not_none_mask = [index for index, item in enumerate(self._percentile_config.all_severities) if item is not None]
        not_none_percentiles = self._compute_percentiles(
            np.asarray(self._percentile_config.all_severities, dtype=np.float64)[not_none_mask]
        )
        mapping_to_severity = self._find_index_without_nones()
        return TurbulenceThresholds(
            **{
                str(severity): index_map if index_map is None else not_none_percentiles[index_map]
                for index_map, severity in zip(
                    mapping_to_severity, TurbulenceSeverity.get_in_ascending_order(), strict=False
                )
            },
            _all_severities=[],
        )


@pydantic_dataclass
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
        return HistogramData(
            hist_values=values[mask],
            bins=np.asarray(self.bins)[np.append(mask, [True])],
            mean=self.mean,
            variance=self.variance,
        )

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

    def __hash__(self) -> int:
        return hash((self.hist_values, self.bins, self.mean, self.variance))


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


# ABSTRACTION MIGHT NOT BE NECESSARY. REMOVE IN FUTURE IF IT ISN'T
class EvaluationPostProcessor(PostProcessor, ABC):
    _components: Mapping[str, PostProcessor]

    def __init__(self, components: Mapping[str, PostProcessor] | None = None) -> None:
        self._components = components if components is not None else {}


class TurbulentRegionsBySeverity(EvaluationPostProcessor):
    _computed_diagnostic: xr.DataArray
    _severities: list["TurbulenceSeverity"]
    _thresholds: "TurbulenceThresholds"
    _threshold_mode: "TurbulenceThresholdMode"
    _has_parent: bool = False

    def __init__(
        self,
        computed_diagnostic: xr.DataArray,
        pressure_levels: list[float],
        severities: list["TurbulenceSeverity"],
        thresholds: "TurbulenceThresholds",
        threshold_mode: "TurbulenceThresholdMode",
        has_parent: bool = False,
    ) -> None:
        super().__init__()
        self._computed_diagnostic = computed_diagnostic.sel(pressure_level=pressure_levels)
        self._severities = severities
        self._thresholds = thresholds
        self._threshold_mode = threshold_mode
        self._has_parent = has_parent

    def execute(self) -> xr.DataArray | list[xr.DataArray]:
        by_severity = []
        for severity in self._severities:
            bounds: "Limits" = self._thresholds.get_bounds(severity, self._threshold_mode)
            this_severity = xr.where(
                (self._computed_diagnostic >= bounds.lower) & (self._computed_diagnostic < bounds.upper), True, False
            )
            by_severity.append(this_severity if self._has_parent else this_severity.compute())
        return by_severity if self._has_parent else xr.concat(by_severity, xr.Variable("severity", self._severities))


class TurbulenceProbabilityBySeverity(EvaluationPostProcessor):
    _severities: list["TurbulenceSeverity"]
    _num_time_steps: int

    def __init__(
        self,
        computed_diagnostic: xr.DataArray,
        pressure_levels: list[float],
        severities: list[TurbulenceSeverity],
        thresholds: "TurbulenceThresholds",
        threshold_mode: "TurbulenceThresholdMode",
    ) -> None:
        super().__init__(
            components={
                "turbulent_regions": TurbulentRegionsBySeverity(
                    computed_diagnostic, pressure_levels, severities, thresholds, threshold_mode, has_parent=True
                )
            }
        )
        self._num_time_steps: int = computed_diagnostic["time"].size
        self._severities = severities

    def execute(self) -> xr.DataArray:
        by_severity: list[xr.DataArray] | xr.DataArray = self._components["turbulent_regions"].execute()
        assert isinstance(by_severity, list)
        probabilities = [
            (this_severity.sum(dim="time").compute() / self._num_time_steps) * 100 for this_severity in by_severity
        ]
        # hmmm.... I'm not sure if this will behave the way I expect with the new dimension
        return xr.concat(probabilities, xr.Variable("severity", self._severities))


class TransformToEDR(EvaluationPostProcessor):
    _computed_diagnostic: xr.DataArray
    _mean: float
    _variance: float
    _has_parent: bool

    def __init__(
        self, computed_diagnostic: xr.DataArray, mean: float, variance: float, has_parent: bool = False
    ) -> None:
        super().__init__()
        self._computed_diagnostic = computed_diagnostic
        self._mean = mean
        self._variance = variance
        self._has_parent = has_parent

    def execute(self) -> xr.DataArray:
        # See ECMWF document for details
        # b = c_2 / standard_deviation
        scaling: float = OVERALL_CLIMATOLOGICAL_PARAMETER.c2 / np.sqrt(self._variance)
        # a = c_2 - b * mean
        offset: float = OVERALL_CLIMATOLOGICAL_PARAMETER.c1 - scaling * self._mean
        unmapped_index = xr.where(self._computed_diagnostic > 0, self._computed_diagnostic, 0)
        # Numpy doesn't support fractional powers of negative numbers so pull the negative out
        # https://stackoverflow.com/a/45384691
        # return exponent_term * (np.sign(self.computed_value()) * (np.abs(self.computed_value()) ** scaling))
        # e^a x^b
        mapped_index: xr.DataArray = (np.exp(offset) * (unmapped_index**scaling)).persist()
        blocking_wait_futures(mapped_index)
        return mapped_index


class CorrelationBetweenDiagnostics(PostProcessor):
    _diagnostic_names: list["DiagnosticName"]
    _computed_indices: dict["DiagnosticName", xr.DataArray]
    _sel_condition: Mapping[str, Any]

    def __init__(
        self, computed_indices: dict["DiagnosticName", xr.DataArray], sel_condition: Mapping[str, Any]
    ) -> None:
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
                # .stack(flat=[...])
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
    _computed_indices: Mapping["DiagnosticName", xr.DataArray]
    _hemispheres: list[Hemisphere]
    _latitudinal_regions: list[LatitudinalRegion]
    _diagnostic_names: list["DiagnosticName"]
    _sel_condition: Mapping[str, Any]

    def __init__(
        self,
        computed_indices: Mapping["DiagnosticName", xr.DataArray],
        sel_condition: Mapping[str, Any],
        hemispheres: list[Hemisphere] | None = None,
        regions: list[LatitudinalRegion] | None = None,
    ) -> None:
        self._computed_indices = computed_indices
        self._diagnostic_names = list(computed_indices.keys())
        if hemispheres is None:
            self._hemispheres = [Hemisphere.GLOBAL, Hemisphere.NORTH, Hemisphere.SOUTH]
        else:
            assert 1 <= len(hemispheres) <= len(Hemisphere)
            self._hemispheres = hemispheres
        if regions is None:
            self._regions = [LatitudinalRegion.FULL, LatitudinalRegion.EXTRATROPICS, LatitudinalRegion.TROPICS]
        else:
            assert 1 <= len(regions) <= len(LatitudinalRegion)
            self._regions = regions
        self._sel_condition = sel_condition

    @staticmethod
    def _apply_region_filter(array: xr.DataArray, hemisphere: Hemisphere, region: LatitudinalRegion) -> xr.DataArray:  # noqa: PLR0911
        extratropic_latitudes: Limits = Limits(lower=25, upper=65)
        entire_tropics: Limits = Limits(lower=-25, upper=25)
        half_tropics: Limits = Limits(lower=0, upper=25)
        assert "latitude" in array.coords
        assert min(array["latitude"]) <= entire_tropics.lower
        assert max(array["latitude"]) >= extratropic_latitudes.upper
        # TODO: Make this pattern matching less clunky
        match hemisphere:
            case Hemisphere.GLOBAL:
                match region:
                    case LatitudinalRegion.FULL:
                        return array
                    case LatitudinalRegion.TROPICS:
                        return array.where(
                            ((array["latitude"] > entire_tropics.lower) & (array["latitude"] < entire_tropics.upper)),
                            drop=True,
                        )
                    case LatitudinalRegion.EXTRATROPICS:
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
                    case _ as unreachable:
                        assert_never(unreachable)
            case Hemisphere.NORTH | Hemisphere.SOUTH:
                match region:
                    case LatitudinalRegion.FULL:
                        return array.where(
                            array["latitude"] > 0 if hemisphere == Hemisphere.NORTH else array["latitude"] < 0,
                            drop=True,
                        )
                    case LatitudinalRegion.TROPICS | LatitudinalRegion.EXTRATROPICS:
                        target_latitudes: Limits = (
                            half_tropics if region == LatitudinalRegion.TROPICS else extratropic_latitudes
                        )
                        condition = (
                            (
                                (array["latitude"] > target_latitudes.lower)
                                & (array["latitude"] < target_latitudes.upper)
                            )
                            if hemisphere == Hemisphere.NORTH
                            else (
                                (array["latitude"] > -target_latitudes.upper)
                                & (array["latitude"] < -target_latitudes.lower)
                            )
                        )
                        return array.where(condition, drop=True)
                    case _ as unreachable:
                        assert_never(unreachable)
            # case Hemisphere.NORTH:
            #     match region:
            #         case LatitudinalRegion.FULL:
            #             return array.where(array["latitude"] > 0, drop=True)
            #         case LatitudinalRegion.TROPICS | LatitudinalRegion.EXTRATROPICS:
            #             target_latitudes: Limits = (
            #                 half_tropics if region == LatitudinalRegion.TROPICS else extratropic_latitudes
            #             )
            #             return array.where(
            #                 (
            #                     (array["latitude"] > target_latitudes.lower)
            #                     & (array["latitude"] < target_latitudes.upper)
            #                 ),
            #                 drop=True,
            #             )
            #         case _ as unreachable:
            #             assert_never(unreachable)
            # case Hemisphere.SOUTH:
            #     match region:
            #         case LatitudinalRegion.FULL:
            #             return array.where(array["latitude"] < 0, drop=True)
            #         case LatitudinalRegion.TROPICS | LatitudinalRegion.EXTRATROPICS:
            #             target_latitudes: Limits = (
            #                 half_tropics if region == LatitudinalRegion.TROPICS else extratropic_latitudes
            #             )
            #             return array.where(
            #                 (
            #                     (array["latitude"] > -target_latitudes.upper)
            #                     & (array["latitude"] < -target_latitudes.lower)
            #                 ),
            #                 drop=True,
            #             )
            #         case _ as unreachable:
            #             assert_never(unreachable)
            case _ as unreachable:
                assert_never(unreachable)

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
