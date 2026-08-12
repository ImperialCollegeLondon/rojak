from collections.abc import Callable
from typing import TYPE_CHECKING

import xarray as xr

from rojak.atmosphere.contrails import issr
from rojak.core.geometric import geodesic_waypoints_between, interpolate_to_geodesic_waypoints
from rojak.utilities.types import Coordinate

if TYPE_CHECKING:
    from rojak.core.data import CATData


def test_issr_interpolation(load_cat_data: Callable) -> None:
    instantiated: CATData = load_cat_data(None)

    lhr = Coordinate(51.47138888, -0.45277777)
    jfk = Coordinate(40.641766, -73.780968)

    waypoints = geodesic_waypoints_between(lhr, jfk, 0.25)

    interp_met_data = interpolate_to_geodesic_waypoints(
        lhr,
        jfk,
        0.25,
        xr.Dataset(
            {"air_temperature": instantiated.temperature(), "specific_humidity": instantiated.specific_humidity()}
        ),
    )

    xr.testing.assert_equal(
        instantiated.issr_along_path(waypoints[:, 0], waypoints[:, 1]),
        issr(
            interp_met_data["air_temperature"],
            interp_met_data["specific_humidity"],
            air_pressure=instantiated.pressure_level(convert_to_pascals=True),
        ),
    )
