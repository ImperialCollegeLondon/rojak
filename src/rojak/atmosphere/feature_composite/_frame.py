"""The feature-relative km frame: the position of each observation relative to the centre it is paired with.

The frame is the WGS-84 orthographic projection in km anchored at the centre, the same frame that
:mod:`rojak.atmosphere._compositing_kernels` re-projects gridded fields onto: +x is east and +y is north.
"""

from typing import TYPE_CHECKING

import numpy as np

from rojak.atmosphere._compositing_kernels import orthographic_projection
from rojak.atmosphere.feature_composite._columns import Column

if TYPE_CHECKING:
    import pandas as pd


def project_pairs(pairs: "pd.DataFrame", *, lat_column: str, lon_column: str) -> "pd.DataFrame":
    """Position of each observation in the orthographic km frame of the centre it is paired with.

    Observations on the far side of the globe from their centre are not visible in the projection, and are dropped.

    Args:
        pairs (pandas.DataFrame): One row per pair of an observation and a centre, with the centre's
            ``Column.CENTRE_ID``, ``Column.CENTRE_LAT`` and ``Column.CENTRE_LON``, and the observation's latitude and
            longitude in degrees. Longitudes may be in either ``[-180, 180)`` or ``[0, 360)``, independently of the
            centres' convention.
        lat_column (str): Name of the column holding the latitude of the observations.
        lon_column (str): Name of the column holding the longitude of the observations.

    Returns:
        pandas.DataFrame: A new table with the rows of ``pairs`` that are visible from their centre, with every column
        of ``pairs`` plus ``Column.X_KM`` and ``Column.Y_KM``. ``pairs`` is not modified.
    """
    obs_lats = pairs[lat_column].to_numpy(dtype=float)
    obs_lons = pairs[lon_column].to_numpy(dtype=float)
    centre_lats = pairs[Column.CENTRE_LAT].to_numpy(dtype=float)
    centre_lons = pairs[Column.CENTRE_LON].to_numpy(dtype=float)

    x_km = np.full(len(pairs), np.nan)
    y_km = np.full(len(pairs), np.nan)

    # One projection per centre rather than per pair; every row of a centre has the same centre lat/lon. PROJ wraps the
    # longitude offset from the centre to [-180, 180), so either longitude convention projects to the same point
    for positions in pairs.groupby(Column.CENTRE_ID, sort=False).indices.values():
        ortho = orthographic_projection(centre_lats[positions[0]], centre_lons[positions[0]])
        x_km[positions], y_km[positions] = ortho(obs_lons[positions], obs_lats[positions])

    # The projection maps points on the far side of the globe to inf
    is_visible = np.isfinite(x_km) & np.isfinite(y_km)
    return pairs.assign(**{Column.X_KM: x_km, Column.Y_KM: y_km})[is_visible]
