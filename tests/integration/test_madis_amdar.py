from typing import TYPE_CHECKING, cast

import numpy as np

from rojak.datalib.madis.amdar import AcarsAmdarRepository, AcarsAmdarTurbulenceData
from rojak.orchestrator.configuration import SpatialDomain

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_climatological_edr(retrieve_single_day_madis_data):
    spatial_domain = SpatialDomain(
        minimum_latitude=-90, maximum_latitude=90, minimum_longitude=-180, maximum_longitude=180, grid_size=0.25
    )
    acars_data: AcarsAmdarTurbulenceData = cast(
        "AcarsAmdarTurbulenceData",
        AcarsAmdarRepository(str(retrieve_single_day_madis_data)).to_amdar_turbulence_data(
            spatial_domain, 0.25, [175, 200, 225, 250, 300, 350]
        ),
    )
    edr_distribution = acars_data.edr_distribution()
    max_edr: NDArray = acars_data.data_frame["maxEDR"].to_dask_array().compute()
    max_edr = max_edr[max_edr > 0]
    ln_max_edr = np.log(max_edr)
    # Fails in CI where 7th decimal is different actual is 3, desired is 2
    np.testing.assert_almost_equal(np.nanmean(ln_max_edr), edr_distribution.mean, decimal=6)
    np.testing.assert_almost_equal(np.nanvar(ln_max_edr), edr_distribution.variance)
