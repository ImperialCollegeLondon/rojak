from typing import Sequence


def make_value_based_slice(coordinate: Sequence, min_value: float | None, max_value: float | None) -> slice:
    is_increasing: bool = coordinate[0] < coordinate[-1]
    return slice(min_value, max_value) if is_increasing else slice(max_value, min_value)
