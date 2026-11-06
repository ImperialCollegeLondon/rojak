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
Concrete data source implementations for the meteorological and AMDAR observation providers rojak supports

Each subpackage implements the abstract interfaces defined in :mod:`rojak.core.data`
(:class:`~rojak.core.data.MetData`, :class:`~rojak.core.data.DataRetriever`,
:class:`~rojak.core.data.AmdarDataRepository`, ...) for a specific data provider:

- :mod:`rojak.datalib.ecmwf`: ECMWF ERA5 reanalysis meteorological data, via the Copernicus Climate Data Store.
- :mod:`rojak.datalib.madis`: MADIS ACARS aircraft-reported (AMDAR) turbulence observations.
- :mod:`rojak.datalib.ukmo`: UK Met Office AMDAR turbulence observations.
"""
