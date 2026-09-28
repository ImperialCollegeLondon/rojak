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
Computation and verification of clear-air turbulence (CAT) diagnostics

This package implements the turbulence diagnostics themselves (:mod:`rojak.turbulence.diagnostic`), the physical
quantities they are built from (:mod:`rojak.turbulence.calculations`), post-processing analyses such as
thresholding, EDR mapping, and association between diagnostics (:mod:`rojak.turbulence.analysis`), the binary
classification and association metrics used to verify diagnostics against observed turbulence
(:mod:`rojak.turbulence.metrics`), and the higher-level verification workflows that tie these together
(:mod:`rojak.turbulence.verification`).
"""
