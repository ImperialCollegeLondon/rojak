from __future__ import annotations

import sys

if sys.version_info >= (3, 13):
    from typing import TypeIs
    from warnings import deprecated
else:
    from typing_extensions import TypeIs, deprecated

__all__ = ["TypeIs", "deprecated"]


def __dir__() -> list[str]:
    return __all__
