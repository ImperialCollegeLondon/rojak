from enum import StrEnum
from typing import TYPE_CHECKING, assert_never, cast

import pypalettes
from cartopy import crs as ccrs

if TYPE_CHECKING:
    from matplotlib import colors as mcolors

_PLATE_CARREE: "ccrs.Projection" = ccrs.PlateCarree()


class StandardColourMaps(StrEnum):
    TURBULENCE_PROBABILITY = "turbulence_probability"
    CORRELATION_COOL_WARM = "correlation_cool_warm"
    SEQUENTIAL_GREEN = "sequential_green"
    FEATURE_REGIONS = "feature_regions"
    DISTANCE_BETWEEN_REGIONS = "distance_between_regions"


def get_a_default_cmap(
    colour_map: StandardColourMaps,
    resample_to: int | None = None,
    load_kwargs: dict | None = None,
) -> "mcolors.Colormap":
    if load_kwargs is None:
        load_kwargs = {}

    match colour_map:
        case StandardColourMaps.TURBULENCE_PROBABILITY:
            chosen_cmap: mcolors.LinearSegmentedColormap = cast(
                "mcolors.LinearSegmentedColormap", pypalettes.load_cmap("cancri", cmap_type="continuous", reverse=True)
            )
            if resample_to is None:
                resample_to = 20
        case StandardColourMaps.DISTANCE_BETWEEN_REGIONS:
            chosen_cmap: mcolors.LinearSegmentedColormap = cast(
                "mcolors.LinearSegmentedColormap", pypalettes.load_cmap("cancri", cmap_type="continuous", **load_kwargs)
            )
        case StandardColourMaps.CORRELATION_COOL_WARM:
            chosen_cmap: mcolors.LinearSegmentedColormap = cast(
                "mcolors.LinearSegmentedColormap", pypalettes.load_cmap("Blue2DarkRed12Steps", **load_kwargs)
            )
        case StandardColourMaps.SEQUENTIAL_GREEN:
            chosen_cmap: mcolors.LinearSegmentedColormap = cast(
                "mcolors.LinearSegmentedColormap", pypalettes.load_cmap("Ernst", **load_kwargs)
            )
        case StandardColourMaps.FEATURE_REGIONS:
            chosen_cmap: mcolors.LinearSegmentedColormap = cast(
                "mcolors.LinearSegmentedColormap", pypalettes.load_cmap("Casita1", keep_first_n=4)
            )
        case _ as unreachable:
            assert_never(unreachable)

    return chosen_cmap if resample_to is None else chosen_cmap.resampled(resample_to)
