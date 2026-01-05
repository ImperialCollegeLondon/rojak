import itertools
from typing import TYPE_CHECKING

import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from rojak.plot.utilities import _PLATE_CARREE, StandardColourMaps, get_a_default_cmap

if TYPE_CHECKING:
    from cartopy import crs as ccrs
    from numpy.typing import NDArray


def combined_regions_cbar_norm(cmap: mcolors.Colormap) -> tuple[mcolors.BoundaryNorm, list[float]]:
    boundaries: list[int] = [0, 0b001, 0b100, 0b111, 8]
    new_cbar_tick_locations = [(first + second) / 2 for first, second in itertools.pairwise(boundaries[1:])]
    return mcolors.BoundaryNorm(boundaries, cmap.N), new_cbar_tick_locations


def create_regions_snapshot_plot(  # noqa: PLR0913
    da_to_plot: xr.DataArray,
    first_feature_name: str,
    second_feature_name: str,
    plot_name: str,
    plot_kwargs: dict | None = None,
    longitude_coord: str = "longitude",
    latitude_coord: str = "latitude",
    col_coord: str = "pressure_level",
    projection: "ccrs.Projection" = _PLATE_CARREE,
    return_facet_grid: bool = False,
) -> xr.plot.FacetGrid | None:  # pyright: ignore[reportAttributeAccessIssue]
    cmap = get_a_default_cmap(StandardColourMaps.FEATURE_REGIONS)
    cbar_norm, new_cbar_tick_locations = combined_regions_cbar_norm(cmap)
    default_plot_kwargs = {
        "x": longitude_coord,
        "y": latitude_coord,
        "col": col_coord,
        "cmap": cmap,
        "norm": cbar_norm,
        "transform": projection,
        # "subplot_kws": {"projection": projection},
        "cbar_kwargs": {
            "orientation": "horizontal",
            "spacing": "uniform",
            "pad": 0.02,
            "shrink": 0.6,
        },
        **(plot_kwargs if plot_kwargs is not None else {}),
    }
    fg: xr.plot.FacetGrid = da_to_plot.plot(**default_plot_kwargs)  # pyright: ignore[reportAttributeAccessIssue]
    fg.map(lambda: plt.gca().coastlines(lw=0.3))  # pyright: ignore[reportAttributeAccessIssue]
    fg.cbar.set_ticks(
        new_cbar_tick_locations,
        labels=[second_feature_name, first_feature_name, f"{first_feature_name} & {second_feature_name}"],
    )

    if return_facet_grid:
        return fg

    fg.fig.savefig(plot_name, dpi=400, bbox_inches="tight")
    plt.close(fg.fig)
    return None


def create_interpolated_combined_regions_plot(
    da_to_plot: xr.DataArray,
    first_feature_name: str,
    second_feature_name: str,
    plot_name: str,
    return_facet_grid: bool = False,
    plot_kwargs: dict | None = None,
) -> xr.plot.FacetGrid:  # pyright: ignore[reportAttributeAccessIssue]
    cmap = get_a_default_cmap(StandardColourMaps.FEATURE_REGIONS)
    cbar_norm, new_cbar_tick_locations = combined_regions_cbar_norm(cmap)
    default_plot_kwargs = {
        "cmap": cmap,
        "norm": cbar_norm,
        # "cbar_kwargs": {
        #     "orientation": "horizontal",
        #     "spacing": "uniform",
        # "pad": 0.02,
        # "shrink": 0.6,
        # "ticks": new_cbar_tick_locations,
        # "ticklabels":  [first_feature_name, second_feature_name, f"{first_feature_name} & {second_feature_name}"]
        # },
        **(plot_kwargs if plot_kwargs is not None else {}),
    }
    fg: xr.plot.FacetGrid = da_to_plot.plot(**default_plot_kwargs)  # pyright: ignore[reportAttributeAccessIssue]
    fg.colorbar.set_ticks(
        new_cbar_tick_locations,
        labels=[second_feature_name, first_feature_name, f"{first_feature_name} & {second_feature_name}"],
    )

    if return_facet_grid:
        return fg

    fg.fig.savefig(plot_name, dpi=400, bbox_inches="tight")
    plt.close(fg.fig)
    return None


def create_distance_from_a_to_b_instantaneous_lat_lon_plot(  # noqa:PLR0913
    distances_from: xr.DataArray,
    to_regions: xr.DataArray,
    pressure_levels: "NDArray",
    plot_name: str,
    time_index: int = 0,
    lat_coord_name: str = "latitude",
    lon_coord_name: str = "longitude",
    column_name: str = "pressure_level",
    plot_kwargs: dict | None = None,
    savefig_kwargs: dict | None = None,
) -> None:
    fg_dist = distances_from.isel(time=time_index).plot(
        x=lon_coord_name,
        y=lat_coord_name,
        col=column_name,
        cmap=get_a_default_cmap(StandardColourMaps.DISTANCE_BETWEEN_REGIONS),
        subplot_kws={"projection": _PLATE_CARREE},
        robust=True,
        **(plot_kwargs if plot_kwargs is not None else {}),
    )
    for index, fg_ax in enumerate(fg_dist.axs.flat):  # int, Axes
        (
            to_regions.isel(time=time_index, pressure_level=index).plot.contour(
                ax=fg_ax, levels=1, colors=["blue"], linewidths=[0.2], linestyles=["dashed"]
            )
        )
        fg_ax.set_title(f"Pressure level = {pressure_levels[index]} hPa")

    fg_dist.map(lambda: plt.gca().coastlines(lw=0.3))  # pyright: ignore[reportAttributeAccessIssue]
    fg_dist.fig.savefig(plot_name, **(savefig_kwargs if savefig_kwargs is not None else {}))
    plt.close(fg_dist.fig)


def create_distance_from_a_to_b_instantaneous_waypoints_plot(  # noqa: PLR0913
    distances_from: xr.DataArray,
    to_regions: xr.DataArray,
    plot_name: str,
    time_index: int = 0,
    waypoint_dim_name: str = "waypoints",
    vertical_dim_name: str = "pressure_level",
    plot_kwargs: dict | None = None,
    savefig_kwargs: dict | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    distances_from.isel(time=time_index).plot(
        x=waypoint_dim_name,
        y=vertical_dim_name,
        yincrease=False,
        ax=ax,
        cmap=get_a_default_cmap(StandardColourMaps.DISTANCE_BETWEEN_REGIONS),
        **(plot_kwargs if plot_kwargs is not None else {}),
    )
    to_regions.isel(time=time_index).plot.contour(
        yincrease=False, ax=ax, levels=1, colors=["red"], linewidths=[0.2], linestyles=["dashed"]
    )
    fig.tight_layout()
    plt.savefig(plot_name, **(savefig_kwargs if savefig_kwargs is not None else {}))
    plt.close(fig)


def create_mean_distances_a_to_b_lat_lon_plot(  # noqa: PLR0913
    distances_from: xr.DataArray,
    plot_name: str,
    mean_over_dim: str = "time",
    lat_coord_name: str = "latitude",
    lon_coord_name: str = "longitude",
    column_name: str = "pressure_level",
    plot_kwargs: dict | None = None,
    savefig_kwargs: dict | None = None,
) -> None:
    fg_mean = distances_from.mean(dim=mean_over_dim, skipna=True).plot(
        x=lon_coord_name,
        y=lat_coord_name,
        col=column_name,
        cmap=get_a_default_cmap(StandardColourMaps.DISTANCE_BETWEEN_REGIONS),
        subplot_kws={"projection": _PLATE_CARREE},
        robust=True,
        vmin=0,
        **(plot_kwargs if plot_kwargs is not None else {}),
    )

    fg_mean.map(lambda: plt.gca().coastlines(lw=0.3))  # pyright: ignore[reportAttributeAccessIssue]
    fg_mean.fig.savefig(plot_name, **(savefig_kwargs if savefig_kwargs is not None else {}))
    plt.close(fg_mean.fig)


def create_mean_distances_a_to_b_waypoints_plot(
    distances_from: xr.DataArray,
    plot_name: str,
    mean_over_dim: str = "time",
    waypoint_dim_name: str = "waypoints",
    vertical_dim_name: str = "pressure_level",
    plot_kwargs: dict | None = None,
    savefig_kwargs: dict | None = None,
) -> None:
    fig = distances_from.mean(dim=mean_over_dim, skipna=True).plot.contourf(
        x=waypoint_dim_name,
        y=vertical_dim_name,
        yincrease=False,
        cmap=get_a_default_cmap(StandardColourMaps.DISTANCE_BETWEEN_REGIONS),
        **(plot_kwargs if plot_kwargs is not None else {}),
    )
    fig.tight_layout()
    plt.savefig(plot_name, **(savefig_kwargs if savefig_kwargs is not None else {}))
    plt.close(fig)
