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
Generic base classes for post-processing pipeline steps

This module provides :class:`PostProcessor` and :class:`BatchPostProcessor`, the common abstract bases used
throughout rojak (e.g. by the turbulence post-processors in :mod:`rojak.turbulence.analysis`) for a single step
of a post-processing pipeline that computes and returns some result.
"""

from abc import ABC, abstractmethod

import dask.array as da
from dask.base import compute as dask_compute
from numpy.typing import NDArray


class PostProcessor[T](ABC):
    """Abstract base for a post-processing step that computes and returns a result of type ``T``"""

    @abstractmethod
    def execute(self) -> T:
        """Run this post-processing step and return its result"""
        ...


class BatchPostProcessor[T](PostProcessor[T], ABC):
    """
    A :class:`PostProcessor` that exposes a lazy dask collection and a separate result-building step.

    Subclasses implement :meth:`build_lazy_collection` to return a bare :class:`dask.array.Array` and
    :meth:`build_result` to construct the final result from its materialised values.  The concrete
    :meth:`execute` calls both in sequence.

    The split is designed for batch callers that want to collapse N sequential ``dask.compute()`` calls into
    one — they collect :meth:`build_lazy_collection` from each processor, issue a single
    ``dask.compute(*collections)``, and dispatch the results back via :meth:`build_result`.
    """

    @abstractmethod
    def build_lazy_collection(self) -> da.Array:
        """Return a lazy dask array representing the work to be done, without triggering any computation"""
        ...

    @abstractmethod
    def build_result(self, computed: NDArray) -> T:
        """
        Build the final result from a materialised array.

        Args:
            computed: The numpy array produced by materialising :meth:`build_lazy_collection`

        Returns:
            The final result of type ``T``
        """
        ...

    def execute(self) -> T:
        """Materialise :meth:`build_lazy_collection` and pass the result to :meth:`build_result`"""
        return self.build_result(dask_compute(self.build_lazy_collection())[0])
