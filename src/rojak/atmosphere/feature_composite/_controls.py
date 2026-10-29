"""Controls: resamplings with the feature removed, used to estimate the counterfactual.

The counterfactual is what would have been observed if the feature weren't there. It can't be observed directly, so
each :class:`ControlDesign` estimates it by pairing the same feature centres with observations from another time: the
traffic and events of a control are those at the centre's place, but ``observation_offset`` away from its time.

Deferred to phase 3: field offsets (pseudo-features from time-relabelled fields), exclusion of controls near a real
feature, the analysis and observation periods, and validation that offsets are whole multiples of the field's time
step.
"""

from collections import Counter
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rojak.atmosphere.feature_composite._columns import Column

if TYPE_CHECKING:
    from collections.abc import Sequence

# Column of the candidate table in valid_controls() holding each design's offset. Never returned.
_OFFSET: str = "_observation_offset"


@dataclass(frozen=True)
class ControlDesign:
    """How to build one control: a resampling with the feature removed, used to estimate the counterfactual.

    A centre at time ``t`` is paired with the observations made at ``t + observation_offset``, at the centre's place.
    The same features therefore see traffic from another time, e.g. a week earlier with
    ``observation_offset=np.timedelta64(-7, "D")``.

    Attributes:
        label (str): Name of the control, e.g. ``"obs-7D"``. It labels the ``control`` dimension of the counts, so it
            must be unique among the controls of one composite.
        observation_offset (numpy.timedelta64): Offset from the centre's time to the time the control's observations
            were made. Keyword only.

    Raises:
        ValueError: If ``label`` is empty.
        TypeError: If ``observation_offset`` is not a :class:`numpy.timedelta64`.
    """

    label: str
    _: KW_ONLY
    observation_offset: np.timedelta64

    def __post_init__(self) -> None:
        """Validate the label and the type of the offset."""
        if self.label == "":
            raise ValueError("label must be a non-empty string naming the control, e.g. 'obs-7D'")
        if not isinstance(self.observation_offset, np.timedelta64):
            raise TypeError(
                "observation_offset must be a numpy.timedelta64, e.g. np.timedelta64(-7, 'D'), but is "
                f"{type(self.observation_offset).__name__}"
            )


def validate_controls(controls: "Sequence[ControlDesign]") -> None:
    """Check that the controls of one composite can be stacked along the ``control`` dimension.

    Args:
        controls (Sequence[ControlDesign]): The control designs. May be empty.

    Raises:
        ValueError: If two or more designs share a label.
    """
    duplicates = sorted(label for label, count in Counter(design.label for design in controls).items() if count > 1)
    if duplicates:
        raise ValueError(f"Each control must have a distinct label, but these are repeated: {duplicates}")


def shift_observations(observations: pd.DataFrame, design: ControlDesign, *, time_column: str) -> pd.DataFrame:
    """Move the observations of a control back to the times of the centres they are paired with.

    An observation made at ``t + observation_offset`` is moved to ``t``, so pairing the shifted observations with the
    unshifted centres pairs each centre with the observations made ``observation_offset`` away from its time.

    Args:
        observations (pandas.DataFrame): Observations with a datetime column ``time_column``. Not modified.
        design (ControlDesign): The control whose offset is undone.
        time_column (str): Name of the column of observation times.

    Returns:
        pandas.DataFrame: A new frame with the same columns in the same order, in which only ``time_column`` has
        changed: it is ``observation_offset`` earlier.
    """
    shifted_times = observations[time_column] - pd.Timedelta(design.observation_offset)
    return observations.assign(**{time_column: shifted_times})


def valid_controls(
    centres: pd.DataFrame,
    controls: "Sequence[ControlDesign]",
    *,
    field_time_range: tuple[np.datetime64, np.datetime64],
) -> pd.DataFrame:
    """The controls that each centre has, i.e. those whose observation time is covered by the field.

    A control is valid for a centre at time ``t`` if ``t + observation_offset`` is within ``field_time_range``, with
    both ends included. Dropping the invalid ones keeps ``n_controls`` correct near the ends of the field.

    Args:
        centres (pandas.DataFrame): Centre table with the columns ``Column.CENTRE_ID`` and ``Column.CENTRE_TIME``. Not
            modified.
        controls (Sequence[ControlDesign]): The control designs. May be empty.
        field_time_range (tuple[numpy.datetime64, numpy.datetime64]): First and last time of the feature's fields.

    Returns:
        pandas.DataFrame: One row per valid ``(centre, control)``, with the columns ``Column.CENTRE_ID`` and
        ``Column.CONTROL`` (the design's label). Empty, with those columns, if there is none.
    """
    designs = pd.DataFrame(
        {
            Column.CONTROL: [design.label for design in controls],
            _OFFSET: np.array([design.observation_offset for design in controls], dtype="timedelta64[ns]"),
        }
    )

    candidates = centres[[Column.CENTRE_ID, Column.CENTRE_TIME]].merge(designs, how="cross")
    control_times = candidates[Column.CENTRE_TIME] + candidates[_OFFSET]

    start, end = field_time_range
    is_valid = control_times.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    return candidates.loc[is_valid, [Column.CENTRE_ID, Column.CONTROL]].reset_index(drop=True)
