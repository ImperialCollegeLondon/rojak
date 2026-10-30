"""Composites of observations about atmospheric features, such as lows, in each feature's own km frame.

Observations, e.g. AMDAR turbulence reports, are composited about every centre of a feature, then compared with
controls to show how the feature changes the traffic and the probability of events.

- The **counterfactual** is what would have been observed if the feature weren't there. It can't be observed
  directly.
- A **control** is the observations or the field resampled with the feature removed, used to estimate the
  counterfactual.

The parts of a composite:

- **Features** find the centres to composite about in time-chunked fields. :class:`ExtremaPointFeature` finds lows or
  highs, one centre per extremum.
- **Pairing** matches observations with the centres at the same time step: :class:`LabelledRegionPairing` by the
  region each centre owns, :class:`DistancePairing` by distance.
- **Bins** count the pairs in each centre's km frame: :meth:`BinSpec.grid` for a 2D ``(y_km, x_km)`` grid.
- **Controls** pair the same centres with observations from another time: :class:`ControlDesign`.
- **Statistics** come from the counts after they are computed: :func:`impact_statistics` for the traffic ratio, the
  relative risk and the odds ratio.

:func:`composite_observations_about_features` builds the dask graph and returns lazy counts per time chunk. With
``return_pairs=True``, it also returns the centre and pair tables, whose columns are named by :class:`Column`::

    counts = composite_observations_about_features(
        ExtremaPointFeature(mslp, ExtremaKind.MINIMA, threshold=100_000),
        observations,
        pairing=LabelledRegionPairing(),
        bins=BinSpec.grid(2000, 50),
        controls=[ControlDesign("obs-7D", observation_offset=np.timedelta64(-7, "D"))],
    ).compute()
    impact = impact_statistics(counts, min_count=20)

To add a new kind of feature or pairing, see :mod:`rojak.atmosphere.feature_composite.extension`.
"""

from rojak.atmosphere.feature_composite._binning import BinSpec
from rojak.atmosphere.feature_composite._columns import Column
from rojak.atmosphere.feature_composite._controls import ControlDesign
from rojak.atmosphere.feature_composite._pairing import DistancePairing, LabelledRegionPairing
from rojak.atmosphere.feature_composite._pipeline import composite_observations_about_features
from rojak.atmosphere.feature_composite._point_features import ExtremaPointFeature
from rojak.atmosphere.feature_composite._statistics import impact_statistics

__all__ = [
    "BinSpec",
    "Column",
    "ControlDesign",
    "DistancePairing",
    "ExtremaPointFeature",
    "LabelledRegionPairing",
    "composite_observations_about_features",
    "impact_statistics",
]
