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
MADIS ACARS AMDAR observation data source

Implements the :mod:`rojak.core.data` interfaces for downloading
(:class:`~rojak.datalib.madis.amdar.AcarsRetriever`), preprocessing
(:class:`~rojak.datalib.madis.amdar.MadisAmdarPreprocessor`), and loading
(:class:`~rojak.datalib.madis.amdar.AcarsAmdarRepository`) ACARS aircraft-reported turbulence observations from
NOAA's MADIS archive.
"""
