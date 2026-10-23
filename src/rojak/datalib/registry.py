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
Registry mapping each :class:`~rojak.orchestrator.configuration.MetDataSource` to the
:class:`~rojak.core.data.MetData` adapter that reads it

This is the single place the orchestrators dispatch on the data source. Supporting a new meteorological data
source requires a new ``MetData`` subclass, a new ``MetDataSource`` member, and a call to
:func:`register_met_data_adapter`; no orchestration code needs to change.

:func:`load_cat_data` combines the two halves of loading: opening the data from wherever it lives
(:func:`rojak.core.sources.open_dataset`) and adapting it into :class:`~rojak.core.data.CATData`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from rojak.core import sources
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import MetDataSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from rojak.core.data import CATData, MetData
    from rojak.orchestrator.configuration import MeteorologyConfig, SpatialDomain

type MetDataAdapter = Callable[["xr.Dataset"], "MetData"]
"""Builds a :class:`~rojak.core.data.MetData` from the raw dataset (e.g. a ``MetData`` subclass itself)"""

_MET_DATA_ADAPTERS: dict[MetDataSource, MetDataAdapter] = {MetDataSource.ERA5: Era5Data}


def register_met_data_adapter(source: MetDataSource, adapter: MetDataAdapter, *, overwrite: bool = False) -> None:
    """
    Register ``adapter`` as the way to read data from ``source``

    Args:
        source: Data source the adapter handles
        adapter: Callable building a :class:`~rojak.core.data.MetData` from the raw dataset
        overwrite: Whether to replace an already registered adapter. Defaults to ``False``.

    Raises:
        ValueError: If ``source`` already has an adapter and ``overwrite`` is ``False``
    """
    if source in _MET_DATA_ADAPTERS and not overwrite:
        raise ValueError(f"An adapter is already registered for {source}")
    _MET_DATA_ADAPTERS[source] = adapter


def get_met_data_adapter(source: MetDataSource) -> MetDataAdapter:
    """
    Look up the adapter registered for ``source``

    Raises:
        NotImplementedError: If no adapter is registered for ``source``
    """
    try:
        return _MET_DATA_ADAPTERS[source]
    except KeyError as e:
        raise NotImplementedError(f"No meteorological data adapter registered for {source}") from e


def load_cat_data(config: "MeteorologyConfig", domain: "SpatialDomain", chunks: "Mapping | None") -> "CATData":
    """
    Open the data described by ``config`` and adapt it into :class:`~rojak.core.data.CATData` over ``domain``

    Args:
        config: What the data is and where it lives
        domain: Spatial (and optionally vertical) domain to restrict the data to
        chunks: Dask chunking to load the data with

    Returns:
        :class:`~rojak.core.data.CATData` restricted to ``domain``
    """
    adapter: MetDataAdapter = get_met_data_adapter(config.data_source)
    dataset: xr.Dataset = sources.open_dataset(config.location, chunks=chunks, engine=config.engine)
    return adapter(dataset).to_clear_air_turbulence_data(domain)
