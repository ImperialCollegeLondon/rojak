"""Column contract of the per-chunk centre and pair tables of the feature composite.

The tables are plain :class:`pandas.DataFrame` objects. ``Column`` is a :class:`~enum.StrEnum`, so a member and its
string value select the same column: users reading a returned pair table can use either.
"""

from enum import StrEnum


class Column(StrEnum):
    """Names of the columns that the feature composite reads from and writes to its per-chunk tables."""

    CENTRE_ID = "centre_id"
    """Identifier of a centre, unique across all chunks. Encoded from the time and grid indices of the centre."""
    FEATURE_ID = "feature_id"
    """Identifier of the feature instance that a centre belongs to. Equal to ``centre_id`` for point features."""
    CENTRE_TIME = "centre_time"
    """Time of the centre, i.e. the time step of the field it was found in."""
    CENTRE_LAT = "centre_lat"
    """Latitude of the centre in degrees."""
    CENTRE_LON = "centre_lon"
    """Longitude of the centre in degrees, in the convention of the field it was found in."""
    CENTRE_VALUE = "centre_value"
    """Value of the field at the centre, e.g. the pressure of a low."""
    CENTRE_LAT_INDEX = "centre_lat_index"
    """Index of the centre along the latitude axis of the field."""
    CENTRE_LON_INDEX = "centre_lon_index"
    """Index of the centre along the longitude axis of the field."""
    TIME_INDEX = "time_index"
    """Index along the whole time axis of the field (not the chunk) of the time step of the centre."""
    DISTANCE_KM = "distance_km"
    """Great-circle distance in km from an observation to the centre it is paired with."""
    X_KM = "x_km"
    """Eastward position in km of an observation in the orthographic frame of its centre."""
    Y_KM = "y_km"
    """Northward position in km of an observation in the orthographic frame of its centre."""
    CONTROL = "control"
    """Label of the control design that a row belongs to."""
