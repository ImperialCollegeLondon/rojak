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
Generic, domain-agnostic infrastructure shared across rojak

This package provides building blocks used throughout the rest of rojak: generic meteorological data loading and
the :class:`~rojak.core.data.CATData`/:class:`~rojak.core.data.CATPrognosticData` interface
(:mod:`rojak.core.data`), coordinate-aware spatial derivatives (:mod:`rojak.core.derivatives`), atmospheric unit
conversions such as pressure/altitude (:mod:`rojak.core.calculations`), geometric and geodesic utilities
(:mod:`rojak.core.geometric`), array/coordinate indexing helpers (:mod:`rojak.core.indexing`), the
:class:`~rojak.core.analysis.PostProcessor` base class used by post-processing pipelines
(:mod:`rojak.core.analysis`), and shared physical/model constants (:mod:`rojak.core.constants`).
"""
