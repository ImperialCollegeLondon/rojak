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
Shared physical and model constants used throughout rojak

This module collects constant values referenced across the codebase:
coordinate bounds, physical constants, and the climatological parameters
used to map turbulence diagnostic values onto the EDR scale.
"""

from typing import Final, NamedTuple

MAX_LATITUDE: Final[float] = 90.0
MAX_LONGITUDE: Final[float] = 180.0
# Physical constants
GRAVITATIONAL_ACCELERATION: Final[float] = 9.80665  # m / s^2
GAS_CONSTANT_DRY_AIR: Final[float] = 287  # R_d: J / (K kg)
GAS_CONSTANT_VAPOUR: Final[float] = 461.51  # R_v: J / (K kg)
ABSOLUTE_ZERO: Final[float] = -273.15  # C
# https://physics.nist.gov/cgi-bin/cuu/Value?gn
EARTH_AVG_RADIUS: float = 6371008.7714  # m


class ClimatologicalEDRConstants(NamedTuple):
    """
    Climatological scaling parameters for mapping a turbulence diagnostic onto the EDR scale

    See :class:`rojak.turbulence.analysis.TransformToEDR`, where ``c1``/``c2`` are used as the offset/scaling.
    coefficients respectively.

    Parameters:
        c1 (float): Offset coefficient which corresponds to the mean of the EDR observations
        c2 (float): Scaling coefficient which corresponds to the standard deviation of the EDR observations
    """

    c1: float
    c2: float


# From Sharman 2017
SHARMAN_17_CLIMATOLOGICAL_PARAMETER: Final[ClimatologicalEDRConstants] = ClimatologicalEDRConstants(-2.572, 0.5067)
# Climatological EDR values computed from 2020-2024
TWENTIES_CLIMATOLOGICAL_PARAMETER: Final[ClimatologicalEDRConstants] = ClimatologicalEDRConstants(-3.627092, 0.951802)
