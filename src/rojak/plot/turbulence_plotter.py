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

from typing import TYPE_CHECKING, Tuple

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon

from rojak.orchestrator.configuration import SpatialDomain, TurbulenceDiagnostics

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from numpy.typing import NDArray


_PLATE_CARREE: "ccrs.Projection" = ccrs.PlateCarree()


def xarray_plot_wrapper(
    data_array: "xr.DataArray",
    plot_name: "Path",
    title: str | None = None,
    plot_kwargs: dict | None = None,
    projection: "ccrs.Projection" = _PLATE_CARREE,
) -> None:
    fig: "Figure" = plt.figure(figsize=(8, 6))
    # Passing it the kwarg projection => it is a cartopy GeoAxis
    ax: "GeoAxes" = fig.add_subplot(1, 1, 1, projection=projection)  # pyright: ignore[reportAssignmentType]
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


def calculate_extent(spatial_domain: "SpatialDomain | Polygon") -> Tuple[float, float, float, float]:
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
    fig: "Figure" = plt.figure(figsize=(8, 6))
    # Passing it the kwarg projection => it is a cartopy GeoAxis
    ax: "GeoAxes" = fig.add_subplot(1, 1, 1, projection=projection)  # pyright: ignore[reportAssignmentType]
    extent = calculate_extent(spatial_domain)
    ax.set_extent(extent)
    # im: "AxesImage" = ax.imshow(probability, extent=extent, cmap="jet", vmin=0, vmax=1)
    # 4, 8, 12, 16 20

    if cmap_bounds is None:
        cmap_bounds = np.arange(21)
    if cmap_name is None:
        cmap_name = "WhiteBlueGreenYellowRed"
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.BoundaryNorm(boundaries=cmap_bounds, ncolors=cmap.N, extend="max")

    im: "AxesImage" = ax.imshow(probabilities, extent=extent, cmap=cmap, norm=norm, transform=projection)
    ax.set_title(diagnostic_label_mapping[diagnostic])
    fig.colorbar(im, ax=ax, label="Turbulence Percentage", spacing="uniform")
    ax.coastlines()
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.close()
