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
"""
Post-processing analyses for turbulence diagnostics

This module implements :class:`~rojak.core.analysis.PostProcessor` subclasses that turn raw turbulence diagnostic
values (as computed by :mod:`rojak.turbulence.diagnostic`) into forecasts and skill statistics, namely:

- Threshold-based turbulence detection: computing severity thresholds from percentiles of a calibration dataset
  (:class:`TurbulenceIntensityThresholds`), applying those thresholds to obtain boolean turbulent regions
  (:class:`TurbulentRegionFromThreshold`, :class:`TurbulentRegionsBySeverity`), and the resulting probability of
  encountering each severity (:class:`TurbulenceProbabilityBySeverity`).
- Mapping raw diagnostic values onto the Eddy Dissipation Rate (EDR) scale, using the log-normal distribution of
  the diagnostic on a calibration dataset (:class:`DiagnosticHistogramDistribution`,
  :class:`ComputeDistributionParametersForEDR`, :class:`TransformToEDR`).
- Correlation between turbulence diagnostics, either globally (:class:`CorrelationBetweenDiagnostics`,
  :class:`MatthewsCorrelationOnDataset`, :class:`MatthewsCorrelationOnThresholdedDiagnostics`) or stratified by
  hemisphere and latitudinal region (:class:`LatitudinalCorrelationBetweenDiagnostics`).
- Association measures (:class:`RelationshipBetween` and its subclasses) between an arbitrary binary feature and
  turbulence diagnostics, dispatched via :class:`RelationshipBetweenFactory` and applied across every diagnostic in
  a dataset via :class:`RelationshipBetweenXAndTurbulence`.
"""

import itertools
from abc import ABC
from collections.abc import Hashable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, assert_never, override

import dask.array as da
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from rich.progress import track

from rojak.core.analysis import PostProcessor
from rojak.core.constants import SHARMAN_17_CLIMATOLOGICAL_PARAMETER
from rojak.orchestrator.configuration import (
    RelationshipBetweenTypes,
    TurbulenceSeverity,
    TurbulenceThresholdMode,
    TurbulenceThresholds,
)
from rojak.turbulence.metrics import (
    contingency_table,
    jaccard_index_multidim,
    matthews_corr_coeff,
    matthews_corr_coeff_multidim,
    relative_risk,
    sample_odds_ratio,
)
from rojak.utilities._compat import TypeIs
from rojak.utilities.types import (
    DistributionParameters,
    Limits,
    all_dtypes_match,
    all_dtypes_same,
    is_xr_data_array,
    is_xr_dataset,
)

if TYPE_CHECKING:
    from rojak.atmosphere.jet_stream import AlphaVelField
    from rojak.utilities.types import DiagnosticName

type IntensityName = str
type IntensityValues = dict[IntensityName, float]


class TurbulenceIntensityThresholds(PostProcessor[TurbulenceThresholds]):
    """
    Computes threshold diagnostic value for each turbulence intensity using percentiles

    To determine if turbulence of a given intensity is encountered, the threshold value for said intensity must first
    be calculated for each diagnostics. Using the specified percentile values, these thresholds are computed to be
    on the calibration dataset in accordance to the methodology detailed in [Williams2017]_
    """

    _percentile_config: TurbulenceThresholds
    _computed_diagnostic: xr.DataArray

    def __init__(self, threshold_config: TurbulenceThresholds, computed_diagnostic: xr.DataArray) -> None:
        """
        Args:
            threshold_config: Percentile to use for each turbulence severity
            computed_diagnostic: Computed diagnostic values on the calibration dataset
        """
        self._percentile_config = threshold_config
        self._computed_diagnostic = computed_diagnostic

    def _compute_percentiles(self, target_percentiles: NDArray) -> NDArray:
        """
        Compute the value of ``self._computed_diagnostic`` at the given percentiles

        Args:
            target_percentiles: Percentiles (in the range [0, 100]) to compute

        Returns:
            Diagnostic value at each of the ``target_percentiles``
        """
        if is_dask_collection(self._computed_diagnostic):
            # flattened_array = da.asarray(self._computed_diagnostic.data, chunks="auto").flatten()
            flattened_array = da.asarray(self._computed_diagnostic, chunks="auto").flatten()
            # Must use tdigest method as internal dask version gives incorrect results
            return da.percentile(flattened_array, target_percentiles, internal_method="tdigest").compute()
        return np.percentile(self._computed_diagnostic.stack(all=[...]), target_percentiles)

    def _find_index_without_nones(self) -> list[int | None]:
        """
        Build a mapping from each severity's position in ``TurbulenceThresholds`` to its index in the array of
        computed (non-``None``) percentiles

        Severities configured with ``None`` (i.e. no percentile to compute) are mapped to ``None`` so that they can
        be distinguished from severities that do have a computed percentile value.

        Returns:
            For each severity (in the same order as ``self._percentile_config.all_severities``), either its index
            into the array of computed percentiles, or ``None`` if that severity has no percentile configured
        """
        new_index: int = 0
        new_list: list[int | None] = []
        for item in self._percentile_config.all_severities:
            if item is None:
                new_list.append(None)
            else:
                new_list.append(new_index)
                new_index += 1
        return new_list

    @override
    def execute(self) -> TurbulenceThresholds:
        """
        Compute the percentile-based threshold value for each configured turbulence severity

        Returns:
            :class:`TurbulenceThresholds` with the computed threshold for each severity
        """
        not_none_mask = [index for index, item in enumerate(self._percentile_config.all_severities) if item is not None]
        not_none_percentiles = self._compute_percentiles(
            np.asarray(self._percentile_config.all_severities, dtype=np.float64)[not_none_mask],
        )
        mapping_to_severity = self._find_index_without_nones()
        return TurbulenceThresholds(
            **{
                str(severity): index_map if index_map is None else not_none_percentiles[index_map]
                for index_map, severity in zip(
                    mapping_to_severity,
                    TurbulenceSeverity.get_in_ascending_order(),
                    strict=False,
                )
            },
            _all_severities=[],
        )


@pydantic_dataclass
class HistogramData:
    """
    Histogram of the log-normal distribution of a turbulence diagnostic's values

    See :class:`DiagnosticHistogramDistribution` for how this is computed. numpy arrays passed into the constructor
    are converted to lists so that the instance is JSON-serialisable.
    """

    hist_values: list[float]
    bins: list[float]
    mean: float
    variance: float

    def __init__(self, hist_values: np.ndarray, bins: np.ndarray, mean: float, variance: float) -> None:
        """
        Args:
            hist_values: Density of each histogram bin
            bins: Bin edges, of length ``len(hist_values) + 1``
            mean: Mean of the (logged) diagnostic values the histogram was computed from
            variance: Variance of the (logged) diagnostic values the histogram was computed from
        """
        self.hist_values = hist_values.tolist()
        self.bins = bins.tolist()
        self.mean = float(mean)
        self.variance = float(variance)

    def as_json_dict(self) -> dict[str, float | list[float]]:
        """
        Convert to a plain, JSON-serialisable dictionary

        Returns:
            Dictionary with ``density``, ``bin_edges``, ``mean``, and ``variance`` keys
        """
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
        """
        Drop histogram bins whose density is below ``minimum_value``

        Args:
            minimum_value: Minimum bin density to keep. Defaults to ``1e-6``.

        Returns:
            New :class:`HistogramData` with insignificant bins removed. ``mean`` and ``variance`` are unchanged.
        """
        values: np.ndarray = np.asarray(self.hist_values)
        mask: np.ndarray = values >= minimum_value
        return HistogramData(
            hist_values=values[mask],
            bins=np.asarray(self.bins)[np.append(mask, [True])],
            mean=self.mean,
            variance=self.variance,
        )

    @override
    def __str__(self) -> str:
        """String representation of the histogram's data"""
        return (
            f"HistogramData(hist_values={self.hist_values}, bins={self.bins}, "
            f"mean={self.mean}, variance={self.variance})"
        )

    @override
    def __repr__(self) -> str:
        """See :func:`__str__`"""
        return self.__str__()

    @override
    def __eq__(self, other: object) -> bool:
        """Two :class:`HistogramData` are equal if their ``hist_values``, ``bins``, ``mean``, and ``variance`` match"""
        if isinstance(other, HistogramData):
            return (
                self.hist_values == list(other.hist_values)
                and self.bins == other.bins
                and self.mean == other.mean
                and self.variance == other.variance
            )
        return False

    @override
    def __hash__(self) -> int:
        """Hash based on ``hist_values``, ``bins``, ``mean``, and ``variance``"""
        return hash((self.hist_values, self.bins, self.mean, self.variance))


class DiagnosticHistogramDistribution(PostProcessor[HistogramData]):
    """
    Computes histogram bins for log-normal distribution of turbulence diagnostics.

    Implementation of the methodology in [Sharman2017]_ to map the raw diagnostic value into EDR
    """

    _computed_diagnostic: xr.DataArray
    _NUM_HIST_BINS: int = 50

    def __init__(self, computed_diagnostic: xr.DataArray, num_hist_bins: int | None = None) -> None:
        """
        Args:
            computed_diagnostic: Computed diagnostic values on the calibration dataset
            num_hist_bins: Number of histogram bins to use. Defaults to ``50`` if not provided.
        """
        self._computed_diagnostic = computed_diagnostic
        if num_hist_bins is not None:
            self._NUM_HIST_BINS = num_hist_bins

    def _serial_execution(self) -> HistogramData:
        """Numpy implementation of :func:`execute`"""
        flattened_array = self._computed_diagnostic.stack(all=[...])
        flattened_array = flattened_array[flattened_array > 0]
        log_of_diagnostic = np.log(flattened_array)
        min_and_max = np.percentile(log_of_diagnostic, [0, 100])
        h, bins = np.histogram(
            log_of_diagnostic,
            bins=self._NUM_HIST_BINS,
            range=(float(min_and_max[0]), float(min_and_max[1])),
            density=True,
        )
        return HistogramData(h, bins, float(np.mean(log_of_diagnostic)), float(np.var(log_of_diagnostic)))

    def _parallel_execution(self) -> HistogramData:
        """Dask implementation of :func:`execute`"""
        flattened_array = da.asarray(self._computed_diagnostic).flatten()
        flattened_array = flattened_array[flattened_array > 0]
        log_of_diagnostic = da.log(flattened_array)
        min_and_max = da.percentile(log_of_diagnostic, [0, 100], internal_method="tdigest").compute()
        h, bins = da.histogram(  # pyright: ignore [reportGeneralTypeIssues]
            log_of_diagnostic,
            bins=self._NUM_HIST_BINS,
            range=(min_and_max[0], min_and_max[1]),
            density=True,
        )
        return HistogramData(
            hist_values=h.compute(),
            bins=bins,
            mean=da.mean(log_of_diagnostic).compute(),
            variance=da.var(log_of_diagnostic).compute(),
        )

    @override
    def execute(self) -> HistogramData:
        """
        Compute the histogram of the (positive) log of the diagnostic values

        Values that are not strictly positive are excluded prior to taking the log.

        Returns:
            :class:`HistogramData` of the log-transformed diagnostic values
        """
        if is_dask_collection(self._computed_diagnostic):
            return self._parallel_execution()
        return self._serial_execution()


# ABSTRACTION MIGHT NOT BE NECESSARY. REMOVE IN FUTURE IF IT ISN'T
class _EvaluationPostProcessor(PostProcessor, ABC):
    """
    Base class for post processors that are themselves composed of one or more other :class:`PostProcessor`
    """

    _components: Mapping[str, PostProcessor]

    def __init__(self, components: Mapping[str, PostProcessor] | None = None) -> None:
        """
        Args:
            components: Mapping of name to the :class:`PostProcessor` it identifies. Defaults to an empty mapping.
        """
        self._components = components if components is not None else {}


class TurbulentRegionFromThreshold(PostProcessor[xr.DataArray | xr.Dataset]):
    """
    Computes a boolean mask of where a turbulence diagnostic falls within a given severity's threshold bounds

    Accepts either a single :class:`xarray.DataArray` with a single :class:`TurbulenceThresholds`, or an
    :class:`xarray.Dataset` of diagnostics with a mapping from diagnostic name to that diagnostic's
    :class:`TurbulenceThresholds`.
    """

    _computed_diagnostic: xr.DataArray | xr.Dataset
    _severity: "TurbulenceSeverity"
    _thresholds: "TurbulenceThresholds| Mapping[DiagnosticName, TurbulenceThresholds]"
    _threshold_mode: "TurbulenceThresholdMode"

    def __init__(
        self,
        computed_diagnostic: xr.DataArray | xr.Dataset,
        severity: "TurbulenceSeverity",
        thresholds: "TurbulenceThresholds | Mapping[DiagnosticName, TurbulenceThresholds]",
        threshold_mode: "TurbulenceThresholdMode",
        /,
        **sel_condition,  # noqa: ANN003
    ) -> None:
        """
        Args:
            computed_diagnostic: Computed diagnostic value(s), either a single DataArray or a Dataset of diagnostics
            severity: Turbulence severity to compute the boolean mask for
            thresholds: Threshold(s) to determine ``severity``. Must be a mapping from diagnostic name to
                :class:`TurbulenceThresholds` if ``computed_diagnostic`` is a Dataset, or a single
                :class:`TurbulenceThresholds` if it is a DataArray
            threshold_mode: Whether the thresholds are bounded intervals or lower bounds
            **sel_condition: Passed to :meth:`xarray.DataArray.sel`/:meth:`xarray.Dataset.sel` to subset
                ``computed_diagnostic`` before computing the mask

        Raises:
            TypeError: If ``thresholds`` is not the shape (mapping or single value) expected for the type of
                ``computed_diagnostic``
            ValueError: If ``computed_diagnostic`` is a Dataset whose variables are not a subset of ``thresholds``
        """
        super().__init__()

        if is_xr_dataset(computed_diagnostic):
            if not self._is_mapping(thresholds):
                raise TypeError("If diagnostics are passed in as xr.Dataset, thresholds must be a mapping")

            if not set(computed_diagnostic.data_vars.keys()).issubset(thresholds.keys()):
                raise ValueError("Diagnostics must be a subset of thresholds")
        elif is_xr_data_array(computed_diagnostic) and self._is_mapping(thresholds):
            raise TypeError(
                "If diagnostics are passed in as xr.DataArray, threshold for it (not mapping) must be passed"
            )

        self._computed_diagnostic = computed_diagnostic.sel(**sel_condition)
        self._severity = severity
        self._thresholds = thresholds
        self._threshold_mode = threshold_mode

    @staticmethod
    def _is_mapping(
        threshold: "TurbulenceThresholds | Mapping[DiagnosticName, TurbulenceThresholds]",
    ) -> TypeIs[Mapping["DiagnosticName", "TurbulenceThresholds"]]:
        """Type guard for whether ``threshold`` is a mapping of diagnostic name to :class:`TurbulenceThresholds`"""
        return isinstance(threshold, Mapping)

    def _execute_on_dataarray(self, this_da: xr.DataArray, this_threshold: "TurbulenceThresholds") -> xr.DataArray:
        """
        Compute the boolean turbulence mask for a single diagnostic DataArray

        Args:
            this_da: Computed diagnostic values
            this_threshold: Thresholds to use for ``this_da``

        Returns:
            Boolean mask of where ``this_da`` falls within ``self._severity``'s bounds
        """
        bounds: Limits[float] = this_threshold.get_bounds(self._severity, self._threshold_mode)
        return (this_da >= bounds.lower) & (this_da < bounds.upper)

    @override
    def execute(self) -> xr.DataArray | xr.Dataset:
        """
        Compute the boolean turbulence mask(s) for ``self._computed_diagnostic``

        Returns:
            Boolean DataArray if ``self._computed_diagnostic`` is a DataArray, or a Dataset of boolean DataArrays
            (one per diagnostic) if it is a Dataset

        Raises:
            AssertionError: If ``self._computed_diagnostic`` and ``self._thresholds`` are an unexpected combination
                of types (unreachable given the checks performed in :meth:`__init__`)
        """
        if is_xr_data_array(self._computed_diagnostic) and not self._is_mapping(self._thresholds):
            return self._execute_on_dataarray(self._computed_diagnostic, self._thresholds)
        if is_xr_dataset(self._computed_diagnostic) and self._is_mapping(self._thresholds):
            return xr.Dataset(
                data_vars={
                    diagnostic_name: self._execute_on_dataarray(this_diagnostic, self._thresholds[str(diagnostic_name)])
                    for diagnostic_name, this_diagnostic in self._computed_diagnostic.items()
                },
                coords=self._computed_diagnostic.coords,
            )
        # https://typing.python.org/en/latest/guides/unreachable.html#marking-code-as-unreachable
        raise AssertionError("Unreachable")


class TurbulentRegionsBySeverity(PostProcessor[xr.DataArray | list[xr.DataArray] | xr.DataTree]):
    """
    Computes turbulent regions by severity for a given turbulence diagnostic

    Based on the thresholds for a given turbulence diagnostic, performs a binary classification of whether
    turbulence is present.
    """

    # TODO: Make this architecture more flexible. Currently, it locks the user in a lot
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
        """
        Args:
            computed_diagnostic: Computed diagnostic values
            pressure_levels: Pressure levels to compute turbulent regions for
            severities: Turbulence severities to compute the boolean mask for
            thresholds: Thresholds used to determine each severity
            threshold_mode: Whether the thresholds are bounded intervals or lower bounds
            has_parent: If ``True``, results for each severity are returned lazily (uncomputed) as a list rather
                than concatenated and computed. Set when this post processor is used as a component of another
                post processor (e.g. :class:`TurbulenceProbabilityBySeverity`). Defaults to ``False``.
        """
        super().__init__()
        self._computed_diagnostic = computed_diagnostic.sel(pressure_level=pressure_levels)
        self._severities = severities
        self._thresholds = thresholds
        self._threshold_mode = threshold_mode
        self._has_parent = has_parent

    @override
    def execute(self) -> xr.DataArray | list[xr.DataArray] | xr.DataTree:
        """
        Compute the boolean turbulent region mask for each severity in ``self._severities``

        Returns:
            If ``self._has_parent`` is ``False``, a single DataArray with the masks concatenated along a new
            ``severity`` dimension. Otherwise, an uncomputed list of DataArrays, one per severity.
        """
        by_severity = []
        for severity in self._severities:
            bounds: Limits[float] = self._thresholds.get_bounds(severity, self._threshold_mode)
            this_severity: xr.DataArray = (self._computed_diagnostic >= bounds.lower) & (
                self._computed_diagnostic < bounds.upper
            )
            by_severity.append(this_severity if self._has_parent else this_severity.compute())
        return by_severity if self._has_parent else xr.concat(by_severity, xr.Variable("severity", self._severities))


class TurbulenceProbabilityBySeverity(_EvaluationPostProcessor):
    """
    Computes probability of encountering turbulence of each severity for a given turbulence diagnostic
    """

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
        """
        Args:
            computed_diagnostic: Computed diagnostic values
            pressure_levels: Pressure levels to compute the probability for
            severities: Turbulence severities to compute the probability of encountering
            thresholds: Thresholds used to determine each severity
            threshold_mode: Whether the thresholds are bounded intervals or lower bounds
        """
        super().__init__(
            components={
                "turbulent_regions": TurbulentRegionsBySeverity(
                    computed_diagnostic,
                    pressure_levels,
                    severities,
                    thresholds,
                    threshold_mode,
                    has_parent=True,
                ),
            },
        )
        self._num_time_steps = computed_diagnostic["time"].size
        self._severities = severities

    @override
    def execute(self) -> xr.DataArray:
        """
        Compute the percentage probability of encountering turbulence of each severity, averaged over time

        Returns:
            DataArray of probabilities (in percent) with a ``severity`` dimension
        """
        by_severity: list[xr.DataArray] | xr.DataArray = self._components["turbulent_regions"].execute()
        assert isinstance(by_severity, list)
        probabilities = [this_severity.mean(dim="time") * 100 for this_severity in by_severity]
        # hmmm.... I'm not sure if this will behave the way I expect with the new dimension
        return xr.concat(probabilities, xr.Variable("severity", self._severities))


class ComputeDistributionParametersForEDR(PostProcessor[DistributionParameters]):
    """
    Computes the mean and variance of the log-normal distribution of a turbulence diagnostic's values

    These parameters are used to map raw diagnostic values onto the EDR scale, see :class:`TransformToEDR`.
    """

    _computed_diagnostic: xr.DataArray

    def __init__(self, computed_diagnostic: xr.DataArray) -> None:
        """
        Args:
            computed_diagnostic: Computed diagnostic values on the calibration dataset
        """
        super().__init__()
        self._computed_diagnostic = computed_diagnostic

    @override
    def execute(self) -> DistributionParameters:
        """
        Compute the mean and variance of the log of the (positive) diagnostic values

        Values that are not strictly positive are excluded prior to taking the log.

        Returns:
            Mean and variance of the log-transformed diagnostic values
        """
        only_positive: xr.DataArray = self._computed_diagnostic.where(self._computed_diagnostic > 0, other=np.nan)
        # False positive by pyright as it doesn't recognise np.log as an xr ufunc
        log_of_diagnostic: xr.DataArray = np.log(only_positive)  # pyright: ignore[reportAssignmentType]
        # dim=None => reduce over all dimensions
        # See https://docs.xarray.dev/en/v2026.02.0/generated/xarray.DataArray.mean.html
        return DistributionParameters(
            mean=float(log_of_diagnostic.mean(dim=None, skipna=True).compute()),
            variance=float(log_of_diagnostic.var(dim=None, skipna=True).compute()),
        )


class TransformToEDR(PostProcessor[xr.DataArray]):
    """
    Transforms turbulence diagnostic values into EDR

    Using the mean and variance of the log-normal distribution of the turbulence diagnostic from the calibration
    dataset, converts turbulence diagnostic values into EDR. An implementation of the methodology described in
    [Sharman2017]_.
    """

    _computed_diagnostic: xr.DataArray
    _mean: float | None
    _variance: float | None
    _c1: float
    _c2: float

    def __init__(
        self,
        computed_diagnostic: xr.DataArray,
        mean: float | None = None,
        variance: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
    ) -> None:
        """
        Args:
            computed_diagnostic: Computed diagnostic values to map onto the EDR scale
            mean: Mean of the log-normal distribution of the diagnostic, as computed on the calibration dataset by
                :class:`ComputeDistributionParametersForEDR`. If ``None``, it is computed from
                ``computed_diagnostic`` when :meth:`execute` is called. Must be provided together with ``variance``.
            variance: Variance of the log-normal distribution of the diagnostic. See ``mean``.
            c1: Climatological scaling parameter. Defaults to the value in [Sharman2017]_ if omitted together with
                ``c2``.
            c2: Climatological scaling parameter. Defaults to the value in [Sharman2017]_ if omitted together with
                ``c1``.

        Raises:
            TypeError: If only one of ``c1``/``c2`` is provided
        """
        super().__init__()
        self._computed_diagnostic = computed_diagnostic
        assert (mean is not None and variance is not None) or (mean is None and variance is None)
        self._mean = mean
        self._variance = variance
        if c1 is not None and c2 is not None:
            self._c1 = c1
            self._c2 = c2
        elif c1 is None and c2 is None:
            self._c1 = SHARMAN_17_CLIMATOLOGICAL_PARAMETER.c1
            self._c2 = SHARMAN_17_CLIMATOLOGICAL_PARAMETER.c2
        else:
            raise TypeError("Both c1 and c2 must be both be provided or omitted")

    @override
    def execute(self) -> xr.DataArray:
        """
        Map the diagnostic values onto the EDR scale

        Returns:
            Diagnostic values transformed into EDR
        """
        if self._mean is None or self._variance is None:
            distribution_parameters = ComputeDistributionParametersForEDR(self._computed_diagnostic).execute()
            self._mean = distribution_parameters.mean
            self._variance = distribution_parameters.variance

        # See ECMWF document for details
        # b = c_2 / standard_deviation
        scaling: float = self._c2 / np.sqrt(self._variance)
        # a = c_2 - b * mean
        offset: float = self._c1 - scaling * self._mean
        unmapped_index = self._computed_diagnostic.clip(min=0)
        # Numpy doesn't support fractional powers of negative numbers so pull the negative out
        # https://stackoverflow.com/a/45384691
        # return exponent_term * (np.sign(self.computed_value()) * (np.abs(self.computed_value()) ** scaling))
        # e^a x^b
        mapped_index: xr.DataArray = (np.exp(offset) * (unmapped_index**scaling)).persist()
        return mapped_index


class CorrelationBetweenDiagnostics(PostProcessor[xr.DataArray]):
    """
    Computes the correlation between turbulence diagnostics


    """

    _diagnostic_names: list["DiagnosticName"]
    _computed_indices: dict["DiagnosticName", xr.DataArray]
    _sel_condition: Mapping[str, Any]

    def __init__(
        self,
        computed_indices: dict["DiagnosticName", xr.DataArray],
        sel_condition: Mapping[str, Any],
    ) -> None:
        """
        Args:
            computed_indices: Mapping from diagnostic name to its computed values
            sel_condition: Passed to :meth:`xarray.DataArray.sel` to subset each diagnostic before computing
                correlations
        """
        self._computed_indices = computed_indices
        self._diagnostic_names = list(self._computed_indices.keys())
        self._sel_condition = sel_condition

    @override
    def execute(self) -> xr.DataArray:
        """
        Compute the Pearson correlation coefficient between every pair of diagnostics

        Returns:
            Symmetric DataArray of correlation coefficients with ``diagnostic1`` and ``diagnostic2`` dimensions
        """
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


class MatthewsCorrelationOnDataset(PostProcessor[xr.DataArray]):
    """
    Computes the Matthew's Correlation Coefficient between every pair of boolean DataArrays within a Dataset

    See :func:`rojak.turbulence.metrics.matthews_corr_coeff`.
    """

    _is_dataset: xr.Dataset
    _with_vars: str

    def __init__(self, is_dataset: xr.Dataset, with_vars: str, /) -> None:
        """
        Class computes the Matthew's Correlation Coefficient between DataArrays within a Dataset. Thus, it requires
        the data to be booleans

        Args:
            is_dataset: Dataset containing boolean data
            with_vars: Name of variables in the dataset, e.g. diagnostic
        """
        super().__init__()

        if not is_dask_collection(is_dataset):
            raise TypeError("Dataset containing turbulence diagnostic forecast must be dask collection")

        if not all_dtypes_same(is_dataset):
            raise TypeError("Dataset must contain DataArrays with the same dtype")

        if not all_dtypes_match(is_dataset, np.bool_):
            raise TypeError("Dataset must contain DataArrays boolean dtypes")

        self._is_dataset = is_dataset.astype(int)
        self._with_vars = with_vars

    @override
    def execute(self) -> xr.DataArray:
        """
        Compute the Matthew's Correlation Coefficient between every pair of DataArrays in ``self._is_dataset``

        Returns:
            Symmetric DataArray of correlation coefficients, with dimensions named ``f"{self._with_vars}1"`` and
            ``f"{self._with_vars}2"``
        """
        data_array_names = list(self._is_dataset.keys())
        num_data_arrays: int = len(data_array_names)

        corr_btw_data_arrays: xr.DataArray = xr.DataArray(
            data=np.ones((num_data_arrays, num_data_arrays)),
            dims=(f"{self._with_vars}1", f"{self._with_vars}2"),
            coords={f"{self._with_vars}1": data_array_names, f"{self._with_vars}2": data_array_names},
        )

        for first_data_array, second_data_array in itertools.combinations(data_array_names, 2):
            correlation_between: float = matthews_corr_coeff(
                truth=da.ravel(self._is_dataset[first_data_array].data),
                prediction=da.ravel(self._is_dataset[second_data_array].data),
            )
            corr_btw_data_arrays.loc[
                {f"{self._with_vars}1": first_data_array, f"{self._with_vars}2": second_data_array}
            ] = correlation_between
            corr_btw_data_arrays.loc[
                {f"{self._with_vars}1": second_data_array, f"{self._with_vars}2": first_data_array}
            ] = correlation_between

        return corr_btw_data_arrays


class MatthewsCorrelationOnThresholdedDiagnostics(PostProcessor[xr.DataArray]):
    """
    Computes the Matthew's Correlation Coefficient between diagnostics, thresholded at each turbulence severity

    For each severity, applies :class:`TurbulentRegionFromThreshold` to obtain boolean turbulent regions for every
    diagnostic in ``diagnostic_indices``, then computes the pairwise Matthew's Correlation Coefficient between
    those boolean regions using :class:`MatthewsCorrelationOnDataset`.
    """

    _diagnostic_indices: xr.Dataset
    _severities: list["TurbulenceSeverity"]
    _thresholds: "Mapping[DiagnosticName, TurbulenceThresholds]"
    _threshold_mode: "TurbulenceThresholdMode"

    def __init__(
        self,
        diagnostic_indices: xr.Dataset,
        severities: list["TurbulenceSeverity"],
        thresholds: "Mapping[DiagnosticName, TurbulenceThresholds]",
        threshold_mode: "TurbulenceThresholdMode",
        /,
        **sel_condition,  # noqa: ANN003
    ) -> None:
        """
        Args:
            diagnostic_indices: Dataset of computed diagnostic values
            severities: Turbulence severities to compute the correlation for
            thresholds: Mapping from diagnostic name to its :class:`TurbulenceThresholds`
            threshold_mode: Whether the thresholds are bounded intervals or lower bounds
            **sel_condition: Passed to :meth:`xarray.Dataset.sel` to subset ``diagnostic_indices``
        """
        super().__init__()
        self._diagnostic_indices = diagnostic_indices.sel(**sel_condition)
        self._severities = severities
        self._thresholds = thresholds
        self._threshold_mode = threshold_mode

    @override
    def execute(self) -> xr.DataArray:
        """
        Compute the Matthew's Correlation Coefficient between diagnostics for each severity

        Returns:
            DataArray of correlation coefficients with ``diagnostic1``, ``diagnostic2``, and ``severity`` dimensions
        """
        correlation_across_severities: list[xr.DataArray] = []
        for severity in track(self._severities, "Computing correlation for each severity"):
            threshold_applied = TurbulentRegionFromThreshold(
                self._diagnostic_indices, severity, self._thresholds, self._threshold_mode
            ).execute()
            assert is_xr_dataset(threshold_applied)
            correlation_across_severities.append(
                MatthewsCorrelationOnDataset(threshold_applied, "diagnostic").execute()
            )

        return xr.concat(correlation_across_severities, "severity").assign_coords(coords={"severity": self._severities})


class Hemisphere(StrEnum):
    """Hemisphere to restrict a latitudinal region to, used by :class:`LatitudinalCorrelationBetweenDiagnostics`"""

    GLOBAL = "global"
    NORTH = "north"
    SOUTH = "south"


class LatitudinalRegion(StrEnum):
    """Latitudinal band to restrict data to, used by :class:`LatitudinalCorrelationBetweenDiagnostics`"""

    FULL = "full"
    EXTRATROPICS = "extratropics"
    TROPICS = "tropics"


class LatitudinalCorrelationBetweenDiagnostics(PostProcessor[xr.DataArray]):
    """
    Computes the correlation between turbulence diagnostics by latitudinal region
    """

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
        """
        Args:
            computed_indices: Mapping from diagnostic name to its computed values
            sel_condition: Passed to :meth:`xarray.DataArray.sel` to subset each diagnostic before computing
                correlations
            hemispheres: Hemispheres to compute the correlation for. Defaults to all of :class:`Hemisphere`.
            regions: Latitudinal regions to compute the correlation for. Defaults to all of
                :class:`LatitudinalRegion`.
        """
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
    def _apply_region_filter(array: xr.DataArray, hemisphere: Hemisphere, region: LatitudinalRegion) -> xr.DataArray:
        """
        Restrict ``array`` to the given hemisphere and latitudinal region

        Args:
            array: DataArray with a ``latitude`` coordinate spanning at least the extratropics of both hemispheres
            hemisphere: Hemisphere to restrict to
            region: Latitudinal region (full, tropics, or extratropics) to restrict to, within ``hemisphere``

        Returns:
            ``array`` filtered to latitudes within ``hemisphere`` and ``region``
        """
        extratropic_latitudes: Limits[float] = Limits(lower=25, upper=65)
        entire_tropics: Limits[float] = Limits(lower=-25, upper=25)
        half_tropics: Limits[float] = Limits(lower=0, upper=25)
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
                        target_latitudes: Limits[float] = (
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

    @override
    def execute(self) -> xr.DataArray:
        """
        Compute the Pearson correlation coefficient between every pair of diagnostics, for each hemisphere/region

        Returns:
            Symmetric DataArray of correlation coefficients with ``diagnostic1``, ``diagnostic2``, ``hemisphere``,
            and ``region`` dimensions
        """
        possible_coordinates: set[str] = {"latitude", "longitude", "valid_time"}
        remaining_coordinates: list[Hashable] = [
            coord for coord in next(iter(self._computed_indices.values())).dims if coord in possible_coordinates
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
                self._hemispheres,
                self._regions,
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


class RelationshipBetween(PostProcessor[xr.DataArray]):
    """
    Abstract base class for computing an association measure between two boolean features

    Subclasses implement :meth:`execute` to compute a specific association measure (e.g. Jaccard index, conditional
    probability, Matthew's Correlation Coefficient, odds ratio, or relative risk) between ``this_feature`` and
    ``other_feature``. See :class:`RelationshipBetweenFactory` for constructing the appropriate subclass from a
    :class:`RelationshipBetweenTypes`.
    """

    _this_feature: xr.DataArray
    _other_feature: xr.DataArray
    _sum_over_dim: str | list[str] | None

    def __init__(
        self, this_feature: xr.DataArray, other_feature: xr.DataArray, sum_over_dim: str | list[str] | None = "time"
    ) -> None:
        """
        Args:
            this_feature: First binary variable
            other_feature: Second binary variable. Its coordinates must be a subset of ``this_feature``'s.
            sum_over_dim: Dimension(s) to sum over when computing the contingency table. Defaults to ``"time"``.

        Raises:
            AssertionError: If ``this_feature`` and ``other_feature`` are not both boolean, or if
                ``other_feature``'s coordinates are not a subset of ``this_feature``'s
        """
        assert this_feature.dtype == other_feature.dtype
        assert this_feature.dtype == np.bool_  # For now, require the two to have a boolean dtype
        assert set(this_feature.coords).issuperset(other_feature.coords)

        self._this_feature = this_feature
        self._other_feature = other_feature
        self._sum_over_dim = sum_over_dim

    @override
    def execute(self) -> xr.DataArray:
        """Compute the association measure between ``self._this_feature`` and ``self._other_feature``, see subclasses"""
        # Return a dataarray to appease the pyright gods
        return xr.DataArray()


class JaccardIndex(RelationshipBetween):
    """Jaccard index (see :func:`rojak.turbulence.metrics.jaccard_index_multidim`) between two boolean features"""

    def __init__(
        self, this_feature: xr.DataArray, other_feature: xr.DataArray, sum_over_dims: str | list[str] | None = "time"
    ) -> None:
        """See :meth:`RelationshipBetween.__init__`"""
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)

    @override
    def execute(self) -> xr.DataArray:
        """Compute the Jaccard index between ``self._this_feature`` and ``self._other_feature``"""
        return jaccard_index_multidim(self._this_feature, self._other_feature, self._sum_over_dim)


class ProbabilityThisGivenOther(RelationshipBetween):
    """Conditional probability :math:`P(\\text{this\\_feature} | \\text{other\\_feature})`"""

    def __init__(
        self, this_feature: xr.DataArray, other_feature: xr.DataArray, sum_over_dims: str | list[str] | None = "time"
    ) -> None:
        """See :meth:`RelationshipBetween.__init__`"""
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)

    @override
    def execute(self) -> xr.DataArray:
        """Compute :math:`P(\\text{this\\_feature} | \\text{other\\_feature})` from the contingency table"""
        table = contingency_table(self._this_feature, self._other_feature, sum_over=self._sum_over_dim)
        return table.n_11 / (table.n_11 + table.n_10)


class ProbabilityThisGivenNotOther(RelationshipBetween):
    """Conditional probability :math:`P(\\text{this\\_feature} | \\lnot \\text{other\\_feature})`"""

    def __init__(
        self, this_feature: xr.DataArray, other_feature: xr.DataArray, sum_over_dims: str | list[str] | None = "time"
    ) -> None:
        """See :meth:`RelationshipBetween.__init__`"""
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)

    @override
    def execute(self) -> xr.DataArray:
        """Compute :math:`P(\\text{this\\_feature} | \\lnot \\text{other\\_feature})` from the contingency table"""
        table = contingency_table(self._this_feature, self._other_feature, sum_over=self._sum_over_dim)
        return table.n_01 / (table.n_00 + table.n_01)


class MatthewsCorrelation(RelationshipBetween):
    """Matthew's Correlation Coefficient (see :func:`rojak.turbulence.metrics.matthews_corr_coeff_multidim`)"""

    def __init__(
        self, this_feature: xr.DataArray, other_feature: xr.DataArray, sum_over_dims: str | list[str] | None = "time"
    ) -> None:
        """See :meth:`RelationshipBetween.__init__`"""
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)

    @override
    def execute(self) -> xr.DataArray:
        """Compute the Matthew's Correlation Coefficient between ``self._this_feature`` and ``self._other_feature``"""
        return matthews_corr_coeff_multidim(self._this_feature, self._other_feature, self._sum_over_dim)


class SampleOddsRatio(RelationshipBetween):
    """Sample odds ratio (see :func:`rojak.turbulence.metrics.sample_odds_ratio`) between two boolean features"""

    def __init__(
        self,
        this_feature: xr.DataArray,
        other_feature: xr.DataArray,
        sum_over_dims: str | list[str] | None = "time",
        use_log: bool = True,
    ) -> None:
        """
        Args:
            this_feature: First binary variable
            other_feature: Second binary variable
            sum_over_dims: Dimension(s) to sum over when computing the contingency table. Defaults to ``"time"``.
            use_log: If ``True`` (default), computes the natural logarithm of the odds ratio.
        """
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)
        self._use_log: bool = use_log

    @override
    def execute(self) -> xr.DataArray:
        """Compute the (log) sample odds ratio between ``self._this_feature`` and ``self._other_feature``"""
        return sample_odds_ratio(self._this_feature, self._other_feature, self._sum_over_dim, use_log=self._use_log)


class RelativeRisk(RelationshipBetween):
    """Relative risk (see :func:`rojak.turbulence.metrics.relative_risk`) between two boolean features"""

    def __init__(
        self,
        this_feature: xr.DataArray,
        other_feature: xr.DataArray,
        sum_over_dims: str | list[str] | None = "time",
        use_log: bool = True,
    ) -> None:
        """
        Args:
            this_feature: First binary variable (the exposure)
            other_feature: Second binary variable (the outcome)
            sum_over_dims: Dimension(s) to sum over when computing the contingency table. Defaults to ``"time"``.
            use_log: If ``True`` (default), computes the natural logarithm of the relative risk.
        """
        super().__init__(this_feature, other_feature, sum_over_dim=sum_over_dims)
        self._use_log: bool = use_log

    @override
    def execute(self) -> xr.DataArray:
        """Compute the (log) relative risk between ``self._this_feature`` and ``self._other_feature``"""
        return relative_risk(self._this_feature, self._other_feature, self._sum_over_dim, use_log=self._use_log)


class RelationshipBetweenFactory:
    """
    Factory for constructing the :class:`RelationshipBetween` subclass corresponding to a
    :class:`RelationshipBetweenTypes`
    """

    _this_feature: xr.DataArray
    _other_feature: xr.DataArray
    _sum_over_dim: str | list[str] | None

    def __init__(
        self,
        this_feature: xr.DataArray,
        other_feature: xr.DataArray,
        /,
        *,
        sum_over_dim: str | list[str] | None = "time",
    ) -> None:
        """
        Args:
            this_feature: First binary variable
            other_feature: Second binary variable
            sum_over_dim: Dimension(s) to sum over when computing the contingency table. Defaults to ``"time"``.
        """
        self._this_feature = this_feature
        self._other_feature = other_feature
        self._sum_over_dim = sum_over_dim

    def create(self, type_of_relationship: RelationshipBetweenTypes, *, use_log: bool = True) -> RelationshipBetween:  # noqa: PLR0911
        """
        Construct the :class:`RelationshipBetween` instance for the requested relationship type

        Args:
            type_of_relationship: Type of association measure to compute
            use_log: For the odds ratio and relative risk relationship types, whether to compute the natural
                logarithm. Ignored for other relationship types. Defaults to ``True``.

        Returns:
            Instance of the :class:`RelationshipBetween` subclass corresponding to ``type_of_relationship``, with
            ``this_feature``/``other_feature`` in the order implied by ``type_of_relationship`` (e.g.
            ``PROBABILITY_OTHER_GIVEN_THIS`` swaps the order relative to ``PROBABILITY_THIS_GIVEN_OTHER``)
        """
        match type_of_relationship:
            case RelationshipBetweenTypes.JACCARD_INDEX:
                return JaccardIndex(self._this_feature, self._other_feature, sum_over_dims=self._sum_over_dim)
            case RelationshipBetweenTypes.PROBABILITY_THIS_GIVEN_OTHER:
                return ProbabilityThisGivenOther(
                    self._this_feature,
                    self._other_feature,
                    sum_over_dims=self._sum_over_dim,
                )
            case RelationshipBetweenTypes.PROBABILITY_OTHER_GIVEN_THIS:
                return ProbabilityThisGivenOther(
                    self._other_feature,
                    self._this_feature,
                    sum_over_dims=self._sum_over_dim,
                )
            case RelationshipBetweenTypes.PROBABILITY_THIS_GIVEN_NOT_OTHER:
                return ProbabilityThisGivenNotOther(
                    self._this_feature,
                    self._other_feature,
                    sum_over_dims=self._sum_over_dim,
                )
            case RelationshipBetweenTypes.PROBABILITY_OTHER_GIVEN_NOT_THIS:
                return ProbabilityThisGivenNotOther(
                    self._other_feature,
                    self._this_feature,
                    sum_over_dims=self._sum_over_dim,
                )
            case RelationshipBetweenTypes.MATTHEWS_CORRELATION:
                return MatthewsCorrelation(self._this_feature, self._other_feature, sum_over_dims=self._sum_over_dim)
            case RelationshipBetweenTypes.SAMPLE_ODDS_RATIO:
                return SampleOddsRatio(
                    self._this_feature, self._other_feature, sum_over_dims=self._sum_over_dim, use_log=use_log
                )
            case RelationshipBetweenTypes.RELATIVE_RISK:
                return RelativeRisk(
                    self._this_feature, self._other_feature, sum_over_dims=self._sum_over_dim, use_log=use_log
                )
            case RelationshipBetweenTypes.INVS_RELATIVE_RISK:
                return RelativeRisk(
                    self._other_feature, self._this_feature, sum_over_dims=self._sum_over_dim, use_log=use_log
                )
            case _ as unreachable:
                assert_never(unreachable)


class RelationshipBetweenXAndTurbulence(PostProcessor[xr.Dataset]):
    """
    Computes the association between a feature and every turbulence diagnostic in a Dataset

    For each diagnostic in ``turbulence_diagnostics``, constructs the appropriate :class:`RelationshipBetween` (via
    :class:`RelationshipBetweenFactory`) between ``other_feature`` and that diagnostic (optionally thresholded first,
    see ``diagnostic_thresholds``), and collects the results into a Dataset keyed by diagnostic name.
    """

    _other_feature: xr.DataArray
    _turbulence_diagnostics: xr.Dataset
    _relationship_type: RelationshipBetweenTypes
    _diagnostic_thresholds: Mapping[str, float] | None
    _feature_name: str
    _use_log: bool
    _sum_over_dim: str | list[str] | None

    def __init__(
        self,
        other_feature: xr.DataArray,
        turbulence_diagnostics: xr.Dataset,
        relationship_between: RelationshipBetweenTypes,
        diagnostic_thresholds: Mapping[str, float] | None = None,
        feature_name: str | None = None,
        use_log: bool = True,
        sum_over_dim: str | list[str] | None = "time",
    ) -> None:
        """
        Args:
            other_feature: Binary feature to compute the relationship against each turbulence diagnostic
            turbulence_diagnostics: Dataset of turbulence diagnostics. If ``diagnostic_thresholds`` is not
                provided, these must already be boolean.
            relationship_between: Type of association measure to compute for each diagnostic
            diagnostic_thresholds: Optional mapping from diagnostic name to a threshold value; if provided, each
                diagnostic is first thresholded (``>=``) to obtain a boolean DataArray. Must be a superset of
                ``turbulence_diagnostics``'s variable names.
            feature_name: Name of ``other_feature``, used in the progress bar description. Defaults to
                ``"feature"``.
            use_log: Passed to :meth:`RelationshipBetweenFactory.create`. Defaults to ``True``.
            sum_over_dim: Dimension(s) to sum over when computing the contingency table. Defaults to ``"time"``.

        Raises:
            AssertionError: If ``diagnostic_thresholds`` is provided but is not a superset of
                ``turbulence_diagnostics``'s variable names
        """
        if diagnostic_thresholds is not None:
            assert set(diagnostic_thresholds.keys()).issuperset(turbulence_diagnostics.keys())

        self._other_feature = other_feature
        self._turbulence_diagnostics = turbulence_diagnostics
        self._relationship_type = relationship_between
        self._diagnostic_thresholds = diagnostic_thresholds
        self._feature_name = feature_name if feature_name is not None else "feature"
        self._use_log = use_log
        self._sum_over_dim = sum_over_dim

    @override
    def execute(self) -> xr.Dataset:
        """
        Compute the association measure between ``self._other_feature`` and each turbulence diagnostic

        Returns:
            Dataset with the same variable names as ``self._turbulence_diagnostics``, each containing the computed
            association measure
        """
        return xr.Dataset(
            data_vars={
                diagnostic_name: RelationshipBetweenFactory(
                    self._other_feature,
                    turbulence_diagnostics
                    if self._diagnostic_thresholds is None
                    else turbulence_diagnostics >= self._diagnostic_thresholds[str(diagnostic_name)],
                    sum_over_dim=self._sum_over_dim,
                )
                .create(self._relationship_type, use_log=self._use_log)
                .execute()
                for diagnostic_name, turbulence_diagnostics in track(
                    self._turbulence_diagnostics.items(),
                    f"Relationship between {self._feature_name} and turbulence diagnostics",
                )
            },
        )


class RelationshipBetweenAlphaVelAndTurbulence(RelationshipBetweenXAndTurbulence):
    """
    :class:`RelationshipBetweenXAndTurbulence` specialised to use jet stream regions identified from an
    :class:`~rojak.atmosphere.jet_stream.AlphaVelField` as the ``other_feature``
    """

    _alpha_vel: "AlphaVelField"
    _turbulence_diagnostics: xr.Dataset
    _relationship_type: RelationshipBetweenTypes
    _diagnostic_thresholds: Mapping[str, float] | None

    def __init__(
        self,
        alpha_vel: "AlphaVelField",
        turbulence_diagnostics: xr.Dataset,
        relationship_between: RelationshipBetweenTypes,
        diagnostic_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        """
        Args:
            alpha_vel: Field used to identify jet stream regions via :meth:`~AlphaVelField.identify_jet_stream`
            turbulence_diagnostics: Dataset of turbulence diagnostics
            relationship_between: Type of association measure to compute for each diagnostic
            diagnostic_thresholds: Optional mapping from diagnostic name to a threshold value; if provided, each
                diagnostic is first thresholded (``>=``) to obtain a boolean DataArray
        """
        super().__init__(
            alpha_vel.identify_jet_stream(),
            turbulence_diagnostics,
            relationship_between,
            diagnostic_thresholds=diagnostic_thresholds,
            feature_name="alpha vel jet stream",
        )
