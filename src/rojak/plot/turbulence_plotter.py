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

import functools
import operator
from typing import TYPE_CHECKING, cast

import cartopy.crs as ccrs
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geoviews as gv
import holoviews as hv
import hvplot.dask  # noqa
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pypalettes
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import xarray as xr
from dask.base import is_dask_collection
from shapely import Polygon

from rojak.orchestrator.configuration import SpatialDomain, TurbulenceDiagnostics
from rojak.turbulence.analysis import Hemisphere, LatitudinalRegion
from rojak.utilities.types import is_dask_array, is_np_array

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cartopy.mpl.geoaxes import GeoAxes
    from holoviews.core.overlay import Overlay
    from holoviews.element.chart import Curve
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from numpy.typing import NDArray

    from rojak.orchestrator.configuration import (
        DiagnosticValidationCondition,
    )
    from rojak.turbulence.verification import RocVerificationResult
    from rojak.utilities.types import DiagnosticName

_PLATE_CARREE: "ccrs.Projection" = ccrs.PlateCarree()

hv.extension("matplotlib")  # pyright: ignore[reportCallIssue]
hvplot.extension("matplotlib", compatibility="bokeh")  # pyright: ignore[reportCallIssue]


def xarray_plot_wrapper(
    data_array: "xr.DataArray",
    plot_name: "Path",
    title: str | None = None,
    plot_kwargs: dict | None = None,
    projection: "ccrs.Projection" = _PLATE_CARREE,
) -> None:
    fig: Figure = plt.figure(figsize=(8, 6))
    # Passing it the kwarg projection => it is a cartopy GeoAxis
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=projection)  # pyright: ignore[reportAssignmentType]
    data_array.plot(
        transform=projection,
        robust=True,
        **plot_kwargs if plot_kwargs is not None else {},
    )
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    ax.coastlines()
    plt.savefig(plot_name)
    plt.close(fig)


diagnostic_label_mapping: dict[TurbulenceDiagnostics, str] = {
    TurbulenceDiagnostics.RICHARDSON: r"$\text{Ri}_g$",
    TurbulenceDiagnostics.F2D: "F2D",
    TurbulenceDiagnostics.F3D: "F3D",
    TurbulenceDiagnostics.UBF: "UBF",
    TurbulenceDiagnostics.TI1: "TI1",
    TurbulenceDiagnostics.TI2: "TI2",
    TurbulenceDiagnostics.NCSU1: "NCSU1",
    TurbulenceDiagnostics.ENDLICH: "Endlich",
    TurbulenceDiagnostics.COLSON_PANOFSKY: "CP",
    TurbulenceDiagnostics.WIND_SPEED: "Speed",
    TurbulenceDiagnostics.WIND_DIRECTION: r"$\psi$",
    TurbulenceDiagnostics.BRUNT_VAISALA: r"$N^2$",
    TurbulenceDiagnostics.VWS: "VWS",
    TurbulenceDiagnostics.DEF: "$DEF^2$",
    # "gradient-richardson": r"$\text{Ri}$",
    TurbulenceDiagnostics.DIRECTIONAL_SHEAR: "Directional Shear",
    TurbulenceDiagnostics.TEMPERATURE_GRADIENT: r"$|\nabla_{H} T|$",
    TurbulenceDiagnostics.HORIZONTAL_DIVERGENCE: r"$|\delta|$",
    TurbulenceDiagnostics.NGM1: "NGM1",
    TurbulenceDiagnostics.NGM2: "NGM2",
    TurbulenceDiagnostics.BROWN1: "Brown1",
    TurbulenceDiagnostics.BROWN2: "Brown2",
    TurbulenceDiagnostics.MAGNITUDE_PV: r"$|\text{PV}|$",
    TurbulenceDiagnostics.PV_GRADIENT: r"$|\nabla\text{PV}|$",
    TurbulenceDiagnostics.NVA: "NVA",
    TurbulenceDiagnostics.DUTTON: "Dutton",
    TurbulenceDiagnostics.EDR_LUNNON: "EDRLun",
    TurbulenceDiagnostics.VORTICITY_SQUARED: r"$|\zeta^2|$",
}


def _calculate_extent(spatial_domain: "SpatialDomain | Polygon") -> tuple[float, float, float, float]:
    if isinstance(spatial_domain, SpatialDomain):
        return (
            spatial_domain.minimum_longitude,
            spatial_domain.maximum_longitude,
            spatial_domain.minimum_latitude,
            spatial_domain.maximum_latitude,
        )
    if isinstance(spatial_domain, Polygon):
        min_x, min_y, max_x, max_y = spatial_domain.bounds
        return min_x, max_x, min_y, max_y
    raise TypeError("SpatialDomain must be of type SpatialDomain or Polygon")


def create_turbulence_probability_plot(
    probabilities: "xr.DataArray",
    plot_name: str,
    spatial_domain: "SpatialDomain | Polygon",
    diagnostic: TurbulenceDiagnostics,
    projection: "ccrs.Projection" = _PLATE_CARREE,
    cmap_bounds: "NDArray | None" = None,
    cmap_name: str | None = None,
) -> None:
    fig: Figure = plt.figure(figsize=(8, 6))
    # Passing it the kwarg projection => it is a cartopy GeoAxis
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=projection)  # pyright: ignore[reportAssignmentType]
    extent = _calculate_extent(spatial_domain)
    ax.set_extent(extent)
    # im: "AxesImage" = ax.imshow(probability, extent=extent, cmap="jet", vmin=0, vmax=1)
    # 4, 8, 12, 16 20

    if cmap_bounds is None:
        cmap_bounds = np.arange(21)
    if cmap_name is None:
        cmap_name = "WhiteBlueGreenYellowRed"
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(boundaries=cmap_bounds, ncolors=cmap.N, extend="max")

    im: AxesImage = ax.imshow(probabilities, extent=extent, cmap=cmap, norm=norm, transform=projection)
    ax.set_title(diagnostic_label_mapping[diagnostic])
    fig.colorbar(im, ax=ax, label="Turbulence Percentage", spacing="uniform")
    ax.coastlines()
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)


def create_multi_turbulence_diagnotics_probability_plot(
    probabilities: "xr.Dataset",
    diagnostics: list["DiagnosticName"],
    plot_name: str,
    projection: "ccrs.Projection" = _PLATE_CARREE,
) -> None:
    names = [str(item) for item in diagnostics]
    # pyright thinks xr.plot doesn't exists...
    fg: xr.plot.FacetGrid = (  # pyright: ignore [reportAttributeAccessIssue, reportCallIssue]
        probabilities[names]
        .to_dataarray("diagnostics")
        .plot(
            col="diagnostics",
            x="longitude",
            y="latitude",
            col_wrap=min(3, len(names)),
            transform=projection,
            vmin=0,
            vmax=20,
            # size=4,
            # aspect=1.5,
            subplot_kws={"projection": projection},
            cbar_kwargs={
                "label": "Turbulence Percentage",
                "orientation": "horizontal",
                "spacing": "uniform",
                "pad": 0.02,
                "shrink": 0.6,
            },
            # cmap="Blues",
            cmap=pypalettes.load_cmap("cancri", cmap_type="continuous", reverse=True).resampled(20),
            robust=True,
        )
    )
    # fg.set_titles("{value}")
    for ax, diagnostic in zip(fg.fig.axes, diagnostics, strict=False):  # GeoAxes, TurbulenceDiagnostic
        ax.set_title(diagnostic_label_mapping[TurbulenceDiagnostics(diagnostic)])
        ax.coastlines()
    # fg.map(lambda: plt.gca().coastlines())
    fg.fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fg.fig)


def _get_clustered_indexing(correlation_array: np.ndarray) -> np.ndarray:
    pairwise_distances: np.ndarray = ssd.pdist(correlation_array)
    linkage: np.ndarray = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold: float = pairwise_distances.max() / 2
    idx_to_cluster_array: np.ndarray = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx: np.ndarray = np.argsort(idx_to_cluster_array)
    return idx


# Modified from https://wil.yegelwel.com/cluster-correlation-matrix/
def _cluster_2d_correlations(corr_array: "xr.DataArray") -> "xr.DataArray":
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to each other

    Parameters
    ----------
    corr_array : xr.DataArray
        a NxN correlation matrix

    Returns
    -------
    xr.DataArray
        a NxN correlation matrix with the columns and rows rearranged
    """

    corr_values: np.ndarray = corr_array.values
    idx: np.ndarray = _get_clustered_indexing(corr_values)  # 1D array

    clustered_corr: np.ndarray = corr_values[idx, :][:, idx]
    new_coords = {
        corr_array.dims[0]: corr_array[corr_array.dims[0]][idx],
        corr_array.dims[1]: corr_array[corr_array.dims[1]][idx],
    }

    return xr.DataArray(clustered_corr, dims=corr_array.dims, coords=new_coords)


def cluster_multi_dim_correlations(
    corr_array: "xr.DataArray",
    order_by_hemisphere: "Hemisphere",
    order_by_region: "LatitudinalRegion",
    in_place: bool = False,
) -> "xr.DataArray":
    assert corr_array.dims[0] not in {"hemisphere", "region"} or corr_array.dims[0] not in {"hemisphere", "region"}
    correlation_to_order_by: np.ndarray = corr_array.sel(hemisphere=order_by_hemisphere, region=order_by_region).values
    idx: np.ndarray = _get_clustered_indexing(correlation_to_order_by)

    if in_place:
        return corr_array[idx, :][:, idx]
    clustered: np.ndarray = corr_array.values[idx, :][:, idx]
    new_coords = {
        corr_array.dims[0]: corr_array[corr_array.dims[0]][idx],
        corr_array.dims[1]: corr_array[corr_array.dims[1]][idx],
        "hemisphere": corr_array["hemisphere"],
        "region": corr_array["region"],
    }
    return xr.DataArray(clustered, dims=corr_array.dims, coords=new_coords)


def create_diagnostic_correlation_plot(correlations: xr.DataArray, plot_name: str, x_coord: str, y_coord: str) -> None:
    assert correlations.ndim == 2, f"Correlations matrix must be two-dimensional not {correlations.ndim}"  # noqa: PLR2004

    # Diagnostic array of correlations will always be square
    num_diagnostics: int = correlations.shape[0]
    assert num_diagnostics == correlations.shape[1], "Correlations matrix must be square"

    clustered_correlations: xr.DataArray = _cluster_2d_correlations(correlations)
    # fig: "Figure" = plt.figure(figsize=(15, 11))
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    clustered_correlations.plot.imshow(ax=ax, center=0.0, cmap="bwr", cbar_kwargs={"label": "Correlation"})
    ax.set_xticks(
        np.arange(num_diagnostics),
        labels=[diagnostic_label_mapping[name] for name in clustered_correlations.coords[x_coord].values],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(
        np.arange(num_diagnostics),
        labels=[diagnostic_label_mapping[name] for name in clustered_correlations.coords[y_coord].values],
    )

    for y_index in range(num_diagnostics):
        for x_index in range(num_diagnostics):
            ax.text(
                x_index,
                y_index,
                f"{clustered_correlations[y_index, x_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    # ax.set_title(f"Clustered Correlation Between CAT Diagnostics")
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.close()


def create_multi_correlation_axis_title(hemisphere: "Hemisphere", region: "LatitudinalRegion") -> str:
    if region == "full":
        if hemisphere == "global":
            return "Global"
        return f"{hemisphere.capitalize()}ern Hemisphere"
    if hemisphere == "global":
        return region.capitalize()
    return f"{hemisphere.capitalize()}ern {region.capitalize()}"


def create_multi_region_correlation_plot(
    correlations: "xr.DataArray", plot_name: str, x_coord: str, y_coord: str
) -> None:
    assert correlations.ndim == 4, (  # noqa: PLR2004
        f"Multi-dimensional correlations matrix must be four-dimensional not {correlations.ndim}"
    )
    num_diagnostics: int = correlations.shape[0]
    assert num_diagnostics == correlations.shape[1], "Correlations matrix must be square"

    clustered_correlations: xr.DataArray = cluster_multi_dim_correlations(
        correlations, Hemisphere.GLOBAL, LatitudinalRegion.FULL, in_place=True
    )
    fg: xr.plot.FacetGrid = clustered_correlations.plot.imshow(  # pyright: ignore[reportAttributeAccessIssue]
        x="diagnostic1", y="diagnostic2", col="hemisphere", row="region", center=0.0, cmap="bwr", size=num_diagnostics
    )

    fg.set_xlabels("")

    x_labels: list[str] = [diagnostic_label_mapping[name] for name in clustered_correlations.coords[x_coord].values]
    y_labels: list[str] = [diagnostic_label_mapping[name] for name in clustered_correlations.coords[y_coord].values]

    for row_idx, region in enumerate(clustered_correlations.coords["region"].values):  # int, str
        fg.axs[row_idx, 0].set_ylabel(region.capitalize(), size="large")
        for col_idx, hemisphere in enumerate(clustered_correlations.coords["hemisphere"].values):
            current_axis = fg.axs[row_idx, col_idx]
            current_axis.set_xticks(
                np.arange(num_diagnostics), labels=x_labels, rotation=45, ha="right", rotation_mode="anchor"
            )
            current_axis.set_yticks(np.arange(num_diagnostics), labels=y_labels)
            # current_axis.set_title(create_multi_correlation_axis_title(hemisphere, region))
            if row_idx == 0:
                current_axis.set_title(create_multi_correlation_axis_title(hemisphere, region))

            this_correlation: xr.DataArray = clustered_correlations.sel(region=region, hemisphere=hemisphere)
            for y_index in range(num_diagnostics):
                for x_index in range(num_diagnostics):
                    current_axis.text(
                        x_index,
                        y_index,
                        f"{this_correlation[y_index, x_index]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

    fg.fig.savefig(plot_name, bbox_inches="tight")
    plt.close(fg.fig)


def _evaluate_dask_collection(array: "da.Array | NDArray") -> "NDArray":
    if is_dask_collection(array):
        assert is_dask_array(array)
        return array.compute()
    assert is_np_array(array)
    return array


def create_interactive_roc_curve_plot(roc: "RocVerificationResult") -> dict[str, "Overlay"]:
    plots: dict[str, Overlay] = {}
    for amdar_verification_col, by_diagnostic_roc in roc.iterate_by_amdar_column():
        auc_for_col = roc.auc_for_amdar_column(amdar_verification_col)
        plots_for_col: list[Curve] = [
            dd.from_dask_array(
                da.stack([roc_for_diagnostic.false_positives, roc_for_diagnostic.true_positives], axis=1),
                columns=["POFD", "POD"],
            ).hvplot.line(  # pyright: ignore[reportAttributeAccessIssue]
                x="POFD",
                y="POD",
                label=f"{diagnostic_name} - AUC: {auc_for_col[diagnostic_name]:.2f}",
            )
            for diagnostic_name, roc_for_diagnostic in by_diagnostic_roc.items()
        ]
        plots_for_col.append(
            dd.from_dask_array(
                da.stack([da.linspace(0, 1, 500), da.linspace(0, 1, 500)], axis=1), columns=["POFD", "POD"]
            ).hvplot.line(x="POFD", y="POD", color="black", line_width=1, line_dash="dashed")  # pyright: ignore[reportAttributeAccessIssue]
        )
        plots[amdar_verification_col] = functools.reduce(operator.mul, plots_for_col)

    return plots


def save_hv_plot(
    holoviews_obj: object,
    figure_name: str,
    figure_format: str,
    render_kwargs: dict | None = None,
    savefig_kwargs: dict | None = None,
) -> None:
    fig = hv.render(holoviews_obj, backend="matplotlib", **(render_kwargs if render_kwargs is not None else {}))
    fig.savefig(
        f"{figure_name}.{figure_format}", bbox_inches="tight", **(savefig_kwargs if savefig_kwargs is not None else {})
    )
    plt.close(fig)


def plot_roc_curve(
    false_positive_rates: "Mapping[DiagnosticName, da.Array | NDArray]",
    true_positive_rates: "Mapping[DiagnosticName, da.Array | NDArray]",
    plot_name: str,
    area_under_curve: "Mapping[str, float] | None" = None,
) -> None:
    assert set(false_positive_rates.keys()) == set(true_positive_rates.keys())
    if area_under_curve is not None:
        assert set(false_positive_rates.keys()).issubset(area_under_curve.keys())
    fpr: dict[DiagnosticName, NDArray] = {
        name: _evaluate_dask_collection(rate) for name, rate in false_positive_rates.items()
    }
    tpr: dict[DiagnosticName, NDArray] = {
        name: _evaluate_dask_collection(rate) for name, rate in true_positive_rates.items()
    }

    line_colours: list = cast(
        "list", cast("mcolors.ListedColormap", pypalettes.load_cmap(["tol", "royal", "prism_light"])).colors
    )
    if len(line_colours) < len(false_positive_rates.keys()):
        raise ValueError("More values to plot than colours")

    fig: Figure = plt.figure(figsize=(8, 6))
    for name, colour in zip(fpr.keys(), line_colours, strict=False):
        plt.plot(
            fpr[name],
            tpr[name],
            color=colour,
            label=name if area_under_curve is None else f"{name} - AUC: {area_under_curve[name]:.2f}",
        )
    # Default line width is 1.5 according to docs
    plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500), color="black", linewidth=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()
    plt.legend()
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.close(fig)


def _check_is_col_in_dataframe(col_to_check: str, data_frame: dd.DataFrame | dgpd.GeoDataFrame) -> None:
    if col_to_check not in data_frame.columns:
        raise ValueError("Column to plot not in dataframe")


def make_into_geodataframe(data_frame: dd.DataFrame | dgpd.GeoDataFrame) -> dgpd.GeoDataFrame:
    if isinstance(data_frame, dd.DataFrame):
        if "geometry" not in data_frame.columns:
            raise ValueError("Dataframe must have geometry column")
        return dgpd.from_dask_dataframe(data_frame)
    return data_frame


def create_interactive_heatmap_plot(
    data_frame: dd.DataFrame | dgpd.GeoDataFrame,
    col_to_plot: str,
    opts_kwargs: dict | None = None,
    new_col_name: str | None = None,
    is_matplotlib: bool = True,
) -> "Overlay":
    _check_is_col_in_dataframe(col_to_plot, data_frame)
    is_points_data: bool = {"latitude", "longitude"}.issubset(data_frame.columns)
    if not is_points_data and "geometry" not in data_frame.columns:
        raise ValueError("Dataframe must have geometry column or latitude/longitude columns")

    dimension: hv.Dimension = hv.Dimension(col_to_plot, label=new_col_name if new_col_name is not None else col_to_plot)
    if not is_matplotlib:
        if opts_kwargs is None:
            opts_kwargs = {"tools": ["hover"]}
        elif "tools" not in opts_kwargs:
            opts_kwargs["tools"] = ["hover"]

    gv_element: gv.element.geo.Polygons | gv.element.geo.Points = (
        gv.Points(data_frame, kdims=["longitude", "latitude"], vdims=[dimension], crs=ccrs.PlateCarree())
        if is_points_data
        else gv.Polygons(data_frame, vdims=[dimension])
    ).opts(
        color=dimension,
        colorbar=True,
        cmap=pypalettes.load_cmap("cancri", cmap_type="continuous", reverse=True),
        alpha=0.8,
        **(opts_kwargs if opts_kwargs is not None else {}),
    )  # pyright: ignore[reportAssignmentType]
    coast_ls = {"linewidth": 1, "edgecolor": "gray"} if is_matplotlib else {"line_color": "black", "line_width": 0.5}
    coast: gv.element.geo.Feature = gv.feature.coastline.opts(**coast_ls)  # pyright: ignore[reportAssignmentType]

    return coast * gv_element


def create_interactive_aggregated_auc_plots(
    aggregated_by_auc: "dict[DiagnosticName, dd.DataFrame]",
    validation_conditions: list["DiagnosticValidationCondition"],
    is_point_data: bool,
) -> hv.Layout:
    num_conditions: int = len(validation_conditions)

    all_plots = []
    for diagnostic_name, regional_auc in aggregated_by_auc.items():
        for condition in validation_conditions:
            options_kwargs = {
                "fig_size": 400,
                "title": f"{diagnostic_name} on {condition.observed_turbulence_column_name}",
                "lw": 0,
            }
            if is_point_data:
                options_kwargs["s"] = 5
            all_plots.append(
                create_interactive_heatmap_plot(
                    regional_auc if is_point_data else regional_auc.compute(),
                    condition.observed_turbulence_column_name,
                    opts_kwargs=options_kwargs,
                    new_col_name="AUC",
                )
            )

    return hv.Layout(all_plots).cols(num_conditions)
