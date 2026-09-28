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
Generic base class for post-processing pipeline steps

This module provides :class:`PostProcessor`, the common abstract base used throughout rojak (e.g. by the
turbulence post-processors in :mod:`rojak.turbulence.analysis`) for a single step of a post-processing pipeline
that computes and returns some result.
"""

from abc import ABC, abstractmethod


class PostProcessor[T](ABC):
    """Abstract base for a post-processing step that computes and returns a result of type ``T``"""

    @abstractmethod
    def execute(self) -> T:
        """Run this post-processing step and return its result"""
        ...
