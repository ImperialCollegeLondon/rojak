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

from typing import NamedTuple

MAX_LATITUDE: float = 90.0
MAX_LONGITUDE: float = 180.0
# Physical constants
GRAVITATIONAL_ACCELERATION: float = 9.80665  # m / s^2
GAS_CONSTANT_DRY_AIR: float = 287  # R_d: J / (K kg)
GAS_CONSTANT_VAPOUR: float = 461.51  # R_v: J / (K kg)
ABSOLUTE_ZERO: float = -273.15  # C
# https://physics.nist.gov/cgi-bin/cuu/Value?gn
EARTH_AVG_RADIUS: float = 6371008.7714  # m


class ClimatologicalEDRConstants(NamedTuple):
    c1: float
    c2: float


# From Sharman 2017
SHARMAN_17_CLIMATOLOGICAL_PARAMETER = ClimatologicalEDRConstants(-2.572, 0.5067)
# Climatological EDR values computed from 2020-2024
TWENTIES_CLIMATOLOGICAL_PARAMETER = ClimatologicalEDRConstants(-3.627092, 0.951802)
